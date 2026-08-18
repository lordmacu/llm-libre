import asyncio
import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from llm_libre import tool_emulator as _emu
from llm_libre.quality_suite import SHORT_TOKEN_BUDGET
from llm_libre.client import build_request
from llm_libre.models import Route
from llm_libre.reasoning import CompositeStreamTrimmer, trim

log = logging.getLogger(__name__)

COOLDOWN_BASE_S = 60.0  # backoff de una ruta castigada por una SONDA (propia,
                         # confirmada) o por fallos de pago directos -- ver
                         # _castigar. Round 9: el 429 YA NO usa esto (ver
                         # _castigar_429 y COOLDOWN_429_*, mas abajo) -- una
                         # sonda confirmada es evidencia mas fuerte que un
                         # simple "pariate" del proveedor, y merece un
                         # backoff que SI escala si la ruta sigue rota.
COOLDOWN_TOPE_S = 3600.0
TIMEOUT_S = 90.0

# El PING de salud comparte el tope de la bateria (SHORT_TOKEN_BUDGET) por la MISMA
# razon, y no puede quedarse atras cuando ese numero se mueve.
#
# Tenia `max_tokens: 8`, cuatro veces menos que los 32 que la bateria ya
# demostro insuficientes. Mientras un 200 vacio contaba como exito eso era
# inofensivo; desde que un 200 sin respuesta adentro es (con razon) un intento
# FALLIDO, un ping que no deja pensar al modelo FABRICA el fallo que dice
# medir: la sonda declara muerta a una ruta sana. Medido contra Kilo con
# max_tokens=8, una pasada sobre las 11 rutas gratis daba 5 sanas; entre las
# "muertas" estaban cohere/north-mini-code:free -- la que sirve `auto` en el
# arranque en frio -- y tencent/hy3:free, que saca 5/5 en la bateria.
#
# El dano no es solo un /health pesimista: `_confiabilidad` mira las ultimas 50
# observaciones y las sondas son ~125 por dia, asi que el trafico real no
# alcanza a desmentirlas. El ranking terminaria ordenando por quien contesta
# mas corto, que es exactamente la premisa que este proyecto elimino.
#
# Vive ACA (round 8), no en sondeo.py: es proxy quien ahora TAMBIEN dispara
# sondas por su cuenta (ver UMBRAL_SOSPECHA mas abajo), asi que el payload
# fijo que las hace inequivocas tiene que estar donde se usa. sondeo.py lo
# importa de aca (`from llm_libre.proxy import PING`) para que la sonda
# PERIODICA y la sonda BAJO DEMANDA sigan siendo, literalmente, el mismo
# pedido.
PING = {"messages": [{"role": "user", "content": "ping"}],
        "max_tokens": SHORT_TOKEN_BUDGET, "temperature": 0}

# Round 8. Seis rondas del mismo mecanismo (retryability, una lista de
# codigos, la lista con el default invertido, atribucion a nivel de
# respuesta, atribucion a nivel de cadena) cayeron cada una por un vector
# nuevo -- y los dos ultimos (round 8) eran ESCOTILLAS DEL PROPIO DISENO de
# round 7: una cadena de una sola ruta (el cliente la fuerza con un `model`
# explicito o con `x_min_contexto`; `/v1/ranking` publica `contexto` por
# ruta, asi que no hace falta ningun conocimiento interno) y la rama
# `if emitido:` de completar_stream (sin hermana con quien comparar, y sin
# chequeo de longitud de cadena). Cuando las fugas estan en las EXCEPCIONES
# que uno mismo escribio a proposito, el eje esta mal, no sub-enumerado.
#
# El cambio: TRAFICO DE UN CLIENTE REAL NUNCA EXCLUYE UNA RUTA DIRECTAMENTE.
# Solo puede acumular SOSPECHA -- que no excluye nada y no es visible en
# ningun lado que importe. Cruzar el umbral programa una SONDA PROPIA, con
# el mismo payload fijo (`PING`, arriba) que ya usa la sonda periodica de
# `sondeo.py`. Esa sonda -- nunca el pedido del cliente -- decide: si falla,
# la ruta se castiga (mismo backoff/tope que siempre, `_castigar`); si pasa,
# la sospecha se limpia y la ruta sigue en rotacion.
#
# Por que una sonda SI puede excluir sin ambiguedad y una respuesta real
# NUNCA puede: el payload de la sonda lo escribe EL GATEWAY. No hay nada que
# atribuir -- si la sonda falla, la ruta esta rota, punto. Ninguno de los
# vectores de las seis rondas anteriores (contenido flageado, prompt
# gigante, modelo de razonamiento que se gasta el presupuesto, una cadena
# corta, un early-return de streaming) puede tocar el payload de una sonda,
# porque el cliente nunca lo escribe.
#
# `429` sigue siendo la excepcion, y sigue INTACTO: es el proveedor mismo
# diciendo "pariate" -- evidencia de la ruta tan inequivoca como una sonda,
# sin necesitar que se confirme con una. Su backoff ya esta probado; este
# round no lo toca.
UMBRAL_SOSPECHA = 3  # cuantos fallos de TRAFICO REAL CONSECUTIVOS (round 9:
                      # ya no "dentro de una ventana de tiempo" -- ver el
                      # hallazgo HIGH 3 mas abajo) hacen falta para programar
                      # una sonda propia contra esa ruta. El mismo numero que
                      # el viejo TOPE_FALLOS_SEGUIDOS de round 7 -- no es
                      # sagrado, pero mantiene la intuicion de "tres
                      # seguidos" que el spec/README ya usan en otros lados.

# Round 9, HIGH 3 del gate. La ronda anterior contaba "3 fallos dentro de
# los ultimos 10 minutos" -- pero eso significa que trafico mas lento que
# 1 fallo cada ~200s NUNCA junta tres dentro de la ventana antes de que el
# primero decaiga: medido, 80 fallos consecutivos espaciados 301s aparte no
# disparaban NINGUNA sonda; a 300s (el limite exacto de la ventana) si. Una
# ruta muerta en un despliegue de trafico bajo se queda primera en
# `ordenar` para siempre -- el defecto original de round 1, de vuelta.
#
# La sospecha ahora es un CONTADOR de fallos consecutivos, sin ventana de
# tiempo: se reinicia a 0 en cualquier exito (ver el `pop` en completar()/
# completar_stream()), pase lo que pase entremedio. No evapora "porque el
# servicio esta tranquilo" -- si nunca hay un exito, tres fallos SEPARADOS
# POR HORAS siguen siendo tres fallos consecutivos. Esto no reabre ningun
# vector de las ocho rondas anteriores: esos eran todos sobre EXCLUSION
# DIRECTA por trafico de cliente, y sospecha nunca excluye nada por si
# sola -- sigue haciendo falta que la SONDA (payload del gateway, nunca del
# cliente) confirme. La proteccion real contra "incidentes no relacionados
# que se suman" siempre fue el reinicio-en-cualquier-exito, no el reloj:
# dos incidentes de verdad separados casi siempre tienen un exito real
# entremedio.
LIMITE_PROBE_BAJO_DEMANDA_S = 60.0  # como maximo UNA sonda bajo demanda por
                                     # ruta cada 60s, sin importar cuantos
                                     # pedidos sospechosos lleguen mientras
                                     # tanto. Justificacion del numero: una
                                     # sonda gasta la MISMA cuota gratis que
                                     # el trafico real (el mismo endpoint,
                                     # el mismo limite por minuto del
                                     # proveedor), asi que el costo extra
                                     # que UN cliente hostil puede imponerle
                                     # a UNA ruta queda topeado en, como
                                     # mucho, 60 pedidos extra por hora --
                                     # sin importar cuantos pedidos mande
                                     # el cliente (podria mandar miles y el
                                     # tope sigue siendo el mismo). Y como
                                     # el payload de esa sonda es del
                                     # gateway, el cliente jamas puede hacer
                                     # que ese pedido extra FALLE: una ruta
                                     # sana pasa la sonda siempre, asi que
                                     # el peor resultado posible para el
                                     # atacante es gastar ese cupo fijo sin
                                     # lograr nunca que una ruta sana caiga.

# Round 9, MEDIUM 6 del gate: el limite de arriba es POR RUTA, pero el
# AGREGADO no estaba acotado -- con N rutas en el catalogo (N no lo controla
# el operador, sale del catalogo VIVO), el peor caso es N x 60 sondas/hora.
# Medido: 11 rutas, 15.840 pedidos extra por dia contra la MISMA cuota
# gratis que necesita el trafico real. Tope GLOBAL, independiente de N: como
# maximo `LIMITE_PROBE_GLOBAL_POR_MINUTO` sondas bajo demanda arrancan por
# minuto, sumando TODAS las rutas. Con 5/min (300/hora, 7200/dia) un
# catalogo del tamano de referencia (11 rutas) queda investigado entero en
# un par de minutos aunque las 11 esten sospechosas a la vez, y ningun
# tamano de catalogo puede hacer crecer el costo agregado mas alla de eso.
LIMITE_PROBE_GLOBAL_POR_MINUTO = 5
VENTANA_PROBE_GLOBAL_S = 60.0

# Round 9, MEDIUM 7 del gate ("el 429 es el unico resorte que le queda al
# cliente"). Un 429 SI sigue castigando de inmediato -- es el proveedor
# mismo diciendo "pariate", tan inequivoco como una sonda -- pero hasta
# ahora reutilizaba el backoff exponencial de _castigar (hasta
# COOLDOWN_TOPE_S=3600s). Medido: 12 pedidos de UNA llave alcanzaban para
# enfriar las 3 rutas de un catalogo chico via 429 real, otra llave caia a
# 503, /health se iba a "caido" -- por una ventana de rate-limit que el
# PROPIO proveedor tipicamente resetea en segundos a un minuto, no en una
# hora. El 429 sigue siendo el UNICO resorte que le queda al trafico de un
# cliente para afectar una ruta directo (justificado: es evidencia
# inequivoca de la ruta, no necesita confirmarse con una sonda) -- pero la
# DURACION tiene que ser proporcional a lo que de verdad esta pasando
# (una ventana de rate-limit), no escalar hacia una exclusion de una hora
# cruzando TODAS las llaves. Se respeta el `Retry-After` del proveedor
# cuando lo manda (es la fuente mas precisa posible); si no lo manda, se usa
# un default corto; en cualquier caso se topea (`COOLDOWN_429_MAXIMO_S`)
# para que un `Retry-After` absurdo (o malicioso, via un proveedor
# comprometido) no vuelva a abrir la puerta a una exclusion de horas.
COOLDOWN_429_DEFAULT_S = 30.0
COOLDOWN_429_MAXIMO_S = 300.0

# Round 10, MEDIUM del gate: `_sospechar_pago` (castigo directo para rutas
# de pago, ver mas abajo) reutilizaba el backoff exponencial de `_castigar`
# -- el MISMO defecto que el 429 tenia antes de la constante de arriba, en
# otro lugar. `_castigar` existe para SONDAS CONFIRMADAS; un castigo de
# pago no tiene ninguna sonda detras (por diseno -- ver _sospechar_pago),
# asi que escalar su duracion como si la tuviera no tiene fundamento.
# Medido a traves de la API real: 60->120->240->480->960->1920->3600s en
# apenas 24 pedidos de una llave, con el fallback de pago afuera para
# TODAS las llaves durante una hora -- ~96 pedidos/dia bastan para
# sostener la exclusion. Flat, sin escalar -- igual que _castigar_429.
COOLDOWN_PAGO_DIRECTO_S = 60.0

# Revision de Task 13, hallazgo 2. Solo un 429 castigaba (con backoff
# exponencial, ver _castigar): un 500, un timeout o un error de red no
# dejaban NUNCA un cooldown, asi que una ruta persistentemente rota o
# COLGADA se seguia probando en cada pedido, adelante de rutas sanas segun
# su prioridad, para siempre -- con TIMEOUT_S=90 eso son hasta 5*90s=450s
# por pedido en la cadena mas larga, y /health sigue en "ok" mientras haya
# UNA ruta viva. `blog` es una maquina saturada: colgado-no-rechazado es el
# modo de falla realista, no un 500 limpio. Round 8 sigue resolviendo esto,
# solo que ahora la ruta se excluye cuando SU PROPIA sonda confirma que esta
# rota (arriba), no cuando el conteo de fallos del cliente llega a un tope.


# Cuantos chunks sin nada util (role inicial, finish_reason, razonamiento
# filtrado) se retienen antes de soltarlos. Existe para que un stream que
# TODAVIA no entrego contenido pueda hacer failover limpio -- si esos chunks ya
# hubieran salido, cambiar de ruta mezclaria dos respuestas. Un stream que solo
# escupe razonamiento puede ser larguisimo, asi que la retencion tiene tope.
TOPE_PENDIENTES = 64

# Claves de SOBRE de un chunk SSE: las que se repiten identicas (o triviales) en
# cada chunk del stream y no aportan informacion propia. Existen para poder
# preguntar "aparte del texto, este chunk trae algo?" mirando el chunk ENTERO
# sin que el sobre conteste que si siempre.
#
# Hace falta mirar el chunk entero porque en el protocolo real de OpenAI
# `finish_reason` es HERMANO de `delta`, no una clave adentro, y el chunk de
# `usage` (stream_options.include_usage) llega con `choices: []`. Un guard que
# solo mirara `delta` los descarta a los dos en silencio -- que es perdida de
# datos en un contrato cuya premisa es "cambia solo base_url". No mordia con
# Kilo ni OpenRouter porque ambos mandan `role` en cada delta, pero si muerde
# con un proveedor estricto (el dialecto OpenAI de MiniMax, o los Groq/Cerebras
# que el diseno planea sumar).
#
# Y hace falta EXCLUIR el sobre porque, si contara, cada chunk de razonamiento
# ya recortado pareceria util por traer `id`/`model`/`index`: el stream de puro
# razonamiento dejaria de hacer failover (regresion del fix B1) y la retencion
# se llenaria de basura.
_SOBRE_CHUNK = frozenset({"id", "object", "created", "model",
                          "system_fingerprint", "service_tier"})
_SOBRE_ELECCION = frozenset({"index"})


# El eje de clasificacion siempre fue ATRIBUCION ("de quien es la culpa, el
# pedido o la ruta"), pero hasta la revision round 6 la IMPLEMENTACION lo
# invertia: "todo 4xx es evidencia del pedido SALVO estos siete codigos".
# Ese default oculta cualquier codigo en el que nadie penso todavia -- el
# reviewer probo el mutante agregando 405 al conjunto de "si cuenta" y la
# suite entera seguia en verde, porque nada pinneaba el DEFAULT, solo la
# lista. Medido el costo: las 5 rutas devolviendo 405 (o 409/415/418/431
# /451, cualquier 4xx que nadie anticipo) dejaba al cliente con 503 en el
# 100% de los pedidos, CERO cooldowns, y /health en 200 "ok" -- la misma
# salida silenciosa de la revision 3, con un codigo distinto cada vez.
#
# PRINCIPIO: cuando no se puede saber de quien es la culpa, HAY QUE
# CONTARLO -- para CONFIABILIDAD (una MEDICION, ver Storage.registrar_evento
# y _confiabilidad). Round 8 separa esto de EXCLUSION (cooldown): ver el
# comentario de cabecera de UMBRAL_SOSPECHA para el principio opuesto que le
# corresponde a esa otra pregunta.
#
# Por eso el default esta INVERTIDO: un 4xx es evidencia de LA RUTA salvo
# que este en esta lista CORTA, y cada entrada tiene que poder justificarse
# en una linea como evidencia GENUINA del payload -- nunca de la ruta. Todo
# lo demas, conocido hoy o no, cuenta igual que un 500. `429`/`408`/`425`
# ya NO necesitan estar en ninguna lista (antes SI, del lado de "no
# cuenta"): bajo este default caen solos del lado de la ruta, que es una
# buena senal de que la forma es la correcta.
_CODIGOS_EVIDENCIA_DE_PEDIDO = frozenset({
    400,  # Bad Request: el cuerpo no se pudo ni interpretar -- es sobre la
          # SINTAXIS de este pedido, nunca sobre si la ruta sirve.
    413,  # Payload Too Large: el TAMAÑO de ESTE pedido excede un limite --
          # un pedido mas chico andaria bien contra la MISMA ruta.
    422,  # Unprocessable Entity: sintacticamente valido pero invalido para
          # ESTE pedido puntual (un parametro fuera de rango, un tipo
          # equivocado) -- no dice nada sobre si la ruta esta sana.
})


def _es_error_del_cliente(codigo: int) -> bool:
    """True SOLO si `codigo` esta en `_CODIGOS_EVIDENCIA_DE_PEDIDO` --
    evidencia GENUINA sobre el pedido, nunca sobre la ruta. Cualquier otro
    codigo (401, 403, 404, 405, 409, o el que un proveedor invente manana)
    es evidencia de la ruta por DEFAULT y cuenta igual que un 500 hacia
    confiabilidad (ver Storage.record_event) -- ver el comentario de
    cabecera del conjunto para el principio completo ("cuando no se puede
    saber de quien es la culpa, hay que contarlo"). Desde round 8 esto ya
    NO decide directamente ningun cooldown -- ver UMBRAL_SOSPECHA: esa es
    una pregunta distinta, con un default distinto.

    Solo los tres codigos de la lista corta no cuentan hacia confiabilidad:
    contarlos convertiria el error de UN cliente en una medicion mala para
    TODOS. Verificado contra el registro real de 5 rutas: tres pedidos
    malformados (400) seguidos bastaban para arrastrar la confiabilidad de
    las cinco antes de que existiera esta exclusion."""
    return codigo in _CODIGOS_EVIDENCIA_DE_PEDIDO


def _timeout_de(proveedor) -> float:
    """`Provider.timeout_s` (default None) permite acotar el peor caso de UN
    proveedor puntual -- p.ej. uno que puede colgarse -- sin bajarle el
    timeout a todos. None (el default, y el comportamiento de siempre para
    quien no lo declare) usa el TIMEOUT_S global."""
    return proveedor.timeout_s if proveedor.timeout_s is not None else TIMEOUT_S


def hay_respuesta(datos: dict) -> bool:
    """True si un 200 trae algo que el cliente pueda usar como respuesta.

    La mayoria de los modelos gratis son de razonamiento: se gastan el
    presupuesto de la completion pensando y devuelven `200` con
    `finish_reason: "length"` y `"content": null`. Sin este chequeo eso se
    registra como EXITO, con lo cual la ruta que falla SUBE su confiabilidad,
    `/health` la sigue contando viva y el cliente recibe la respuesta vacia en
    vez de un failover.

    `tool_calls` cuenta como respuesta: una llamada a herramienta legitima
    viaja con `content: null` y toda la carga util ahi. Se exige por VERDAD del
    valor (no por presencia) porque en el mensaje FINAL de una respuesta no
    streaming, `tool_calls: []` significa literalmente "no llame a ninguna
    herramienta" -- al reves que en los deltas de streaming, donde la presencia
    de la clave ya es senal.
    """
    if not isinstance(datos, dict):
        return False
    for eleccion in datos.get("choices") or []:
        if not isinstance(eleccion, dict):
            continue
        msg = eleccion.get("message") or {}
        if not isinstance(msg, dict):
            continue
        contenido = msg.get("content")
        if isinstance(contenido, str) and contenido.strip():
            return True
        if msg.get("tool_calls"):
            return True
    return False


@dataclass
class Respuesta:
    estado: int
    json: dict
    ruta: Route | None
    intentos: int
    razonamiento: str = ""
    # Status HTTP que devolvio el PROVEEDOR en el ultimo intento (0 = ni
    # siquiera hubo respuesta: error de red). No es lo mismo que `estado`, que
    # es lo que este gateway decidio: un 200 que llega vacio queda como
    # `estado=503, codigo_upstream=200`, y esa diferencia es justo lo que hace
    # falta para diagnosticar desde la tabla de sondas si el proveedor esta
    # caido o si respondio bien pero sin nada adentro.
    #
    # En una cadena de varias rutas es el codigo de la ULTIMA intentada; quien
    # necesite atribucion exacta por ruta tiene la tabla `eventos`, que guarda
    # una fila por intento. La sonda de salud pasa siempre una sola ruta, asi
    # que ahi no hay ambiguedad.
    codigo_upstream: int = 0


class Proxy:
    def __init__(self, proveedores: dict, almacen, cliente_http: httpx.AsyncClient):
        self.proveedores = proveedores
        self.almacen = almacen
        self.http = cliente_http
        self.cooldowns: dict[str, float] = {}
        self._castigos: dict[str, int] = {}
        # Round 9: numero de generaciones que un cooldown/castigo tuvo para
        # esta clave (se incrementa en CADA _castigar/_castigar_429/
        # _limpiar_castigo) -- deja que una sonda bajo demanda detecte si
        # algo (tipicamente un 429 real) modifico el estado de la ruta
        # MIENTRAS la sonda estaba en vuelo, y evite pisarlo (MEDIUM 5).
        self._generacion_cooldown: dict[str, int] = {}
        # Round 9 (HIGH 3): contador de fallos CONSECUTIVOS de trafico real
        # por ruta gratis, sin ventana de tiempo -- se reinicia a 0 en
        # cualquier exito. Ver _sospechar().
        self._sospechas: dict[str, int] = {}
        # Round 9 (HIGH 4): igual que arriba pero para rutas de PAGO, que no
        # pasan por sospecha+sonda (nunca se sondean, ver _sospechar_pago) --
        # cuenta hacia un castigo DIRECTO, sin sonda de por medio.
        self._fallos_pago: dict[str, int] = {}
        self._ultimo_probe_bajo_demanda: dict[str, float] = {}
        # Round 9 (MEDIUM 6): marcas de tiempo de sondas bajo demanda
        # arrancadas, sumando TODAS las rutas -- tope agregado independiente
        # del tamano del catalogo. Ver LIMITE_PROBE_GLOBAL_POR_MINUTO.
        self._probes_recientes: list[float] = []
        # Sondas bajo demanda en vuelo (asyncio.Task), por clave -- evita
        # disparar dos sondas superpuestas contra la MISMA ruta si varios
        # pedidos concurrentes cruzan el umbral casi al mismo tiempo, y le
        # da a los tests un lugar donde esperar (`esperar_sondas_pendientes`).
        self._sondas_en_curso: dict[str, asyncio.Task] = {}
        # Round 10, HIGH del gate: rutas que cruzaron el umbral y quieren
        # una sonda pero todavia no consiguieron cupo -- ver
        # `_admitir_sondas_pendientes`. Sin esto, el cupo GLOBAL (round 9,
        # MEDIUM 6) se lo llevaban siempre las mismas rutas: `completar()`
        # recorre la cadena en el MISMO orden en cada pedido (prioridad/
        # confiabilidad), asi que las primeras rutas de la cadena piden
        # sonda primero, SIEMPRE -- y una ruta genuinamente rota, cuya
        # confiabilidad colapsada la manda al FINAL de esa cadena, nunca
        # llegaba a pedir antes de que el cupo del minuto se agotara.
        self._esperando_sonda: dict[str, Route] = {}

    async def completar(self, rutas: list[Route], cuerpo: dict, ahora: float,
                        crudo: bool = False, es_sonda: bool = False,
                        en_intento_facturable: Callable[[Route], None] | None = None
                        ) -> Respuesta:
        """`es_sonda=True` es exclusivamente para uso INTERNO (sondeo
        periodico via `probing.probe_health`, y la sonda bajo demanda de
        `_probar_bajo_demanda` -- nunca lo pasa un pedido de un cliente
        real): cambia como se interpreta un fallo NO-429. Con
        `es_sonda=False` (el default, todo trafico real) un fallo solo
        acumula SOSPECHA (`_sospechar`/`_sospechar_pago`, ver
        UMBRAL_SOSPECHA) -- nunca castiga directo. Con `es_sonda=True` un
        fallo castiga la ruta de inmediato (`_castigar`), sin pasar por
        sospecha ni programar otra sonda: una sonda ya ES la verificacion,
        no hay nada mas que confirmar.

        `en_intento_facturable`, si se pasa, se llama por cada intento
        contra una ruta de PAGO que el proveedor respondio con `200` --
        exitoso o no (HIGH 4, round 9): el proveedor cobra por generar la
        respuesta, no por que el gateway la considere util. `r.ruta`
        (arriba, exito) solo marca el intento que de verdad sirvio al
        cliente; este callback marca TODO intento facturable, para que
        quien factura (api.py) no pierda el caso "200 con contenido vacio"
        -- billable y silencioso hasta esta ronda."""
        intentos = 0
        ultimo_error = None
        ultimo_codigo = 0
        claves_del_pedido = {ruta.key for ruta in rutas}
        for ruta in rutas:
            proveedor = self.proveedores[ruta.provider]
            url, cabeceras, payload = build_request(proveedor, cuerpo, ruta.model_id)
            intentos += 1
            t0 = time.monotonic()
            try:
                resp = await self.http.post(url, headers=cabeceras, json=payload,
                                            timeout=_timeout_de(proveedor))
                codigo = resp.status_code
            except httpx.HTTPError as e:
                codigo, resp, ultimo_error = 0, None, str(e)
            ultimo_codigo = codigo
            # Round-trip completo, NO un time-to-first-token: por este camino la
            # respuesta llega entera de una vez, asi que este numero incluye
            # toda la generacion (7-27 s en un modelo de razonamiento). Va a
            # `latencia_ms`; `ttft_ms` queda en 0 para no contaminar un p50 que
            # significa otra cosa. Ver el comentario de cabecera de almacen.py.
            latencia = int((time.monotonic() - t0) * 1000)
            # HIGH 2 (round 9): el cooldown que un fallo de ESTE intento
            # dispare se estampa con AHORA + cuanto tardo ESTE intento, no
            # con el `ahora` crudo. Si el intento tardo hasta TIMEOUT_S=90s
            # (una ruta colgada -- el caso que este mecanismo entero existe
            # para atrapar) y se estampa con el `ahora` de cuando EMPEZO el
            # intento, el cooldown nace con hasta 90s ya "comidos": con
            # COOLDOWN_BASE_S=60, el primer castigo (60s nominales) nace YA
            # EXPIRADO. Medido: exclusion efectiva max(0, 60*2^(n-1) - 90) =
            # 0s, 30s, 150s, 390s en los primeros cuatro castigos -- la ruta
            # colgada segia de punta en la cadena. Con MockTransport (sin
            # demora real) `latencia` es ~0 y esto es un no-op: no cambia
            # ningun test existente.
            ahora_del_castigo = ahora + latencia / 1000.0

            if codigo == 200 and ruta.tier == "pago" and en_intento_facturable is not None:
                en_intento_facturable(ruta)

            # Un 200 con cuerpo no parseable (p.ej. una pagina de mantenimiento
            # HTML servida con status 200) no es un exito: se trata como intento
            # fallido, sin excepcion escapando y sin castigo (no es rate-limit,
            # esta rota).
            datos = None
            if codigo == 200:
                try:
                    datos = resp.json()
                except ValueError:
                    datos = None
                    ultimo_error = "200 con cuerpo no-JSON"
                else:
                    # JSON valido pero que no es un objeto (p.ej. una lista):
                    # `_limpiar` de mas abajo hace datos.get(...) y reventaria
                    # con AttributeError sin atrapar -- o sea un 500 del
                    # gateway porque el proveedor mando algo raro. El mismo
                    # trato que el cuerpo no-JSON, y ANTES de tocar `datos`.
                    if not isinstance(datos, dict):
                        datos = None
                        ultimo_error = "200 con cuerpo JSON que no es un objeto"

            # Mismo lugar y mismo trato que el guard de arriba: un 200 que no
            # trae respuesta adentro tampoco es un exito. El recorte del
            # razonamiento va ANTES de decidirlo porque lo que cuenta es lo que
            # el cliente va a ver: si tras sacar el <think> no queda nada, esa
            # ruta no respondio (salvo en modo crudo, donde el texto crudo ES
            # la respuesta pedida).
            razon = ""
            if datos is not None:
                razon = "" if crudo else self._limpiar(datos, proveedor.unwraps_canvas)
                if not hay_respuesta(datos):
                    datos = None
                    ultimo_error = "200 sin contenido ni tool_calls"

            exito = codigo == 200 and datos is not None
            self.almacen.record_event(ruta.key, exito, 0, codigo, ahora,
                                          latency_ms=latencia,
                                          is_client_error=_es_error_del_cliente(codigo))

            if exito:
                self._sospechas.pop(ruta.key, None)
                self._fallos_pago.pop(ruta.key, None)
                # MEDIUM 5 (round 9): con es_sonda=True, la limpieza queda
                # diferida a _probar_bajo_demanda (que chequea si un 429
                # castigo la ruta MIENTRAS esta sonda estaba en vuelo antes
                # de limpiar) -- completar() no puede saber eso desde aca.
                if not es_sonda:
                    self._limpiar_castigo(ruta.key)
                if proveedor.emulates_tools:
                    # `cuerpo` es el pedido del CLIENTE, con su `tools` intacto:
                    # build_request lo saca solo de la copia que viaja al
                    # proveedor. Ese array es el allow-list que evita convertir
                    # una respuesta de texto legitima en un tool call inventado.
                    datos = _emu.detect_and_convert(datos, cuerpo.get("tools"),
                                                    cuerpo.get("tool_choice"))
                return Respuesta(200, datos, ruta, intentos, razon, codigo)

            if codigo == 429:
                self._castigar_429(ruta.key, ahora_del_castigo, resp)
            elif not _es_error_del_cliente(codigo):
                self._reaccionar_a_fallo_no_429(ruta, ahora, ahora_del_castigo, es_sonda)
            ultimo_error = ultimo_error or f"HTTP {codigo}"

        # Solo cuentan los cooldowns de las rutas de ESTE pedido: el proxy vive
        # mas alla de una sola llamada y puede tener castigadas rutas ajenas a
        # esta cadena, cuyo vencimiento no le sirve de nada al que esta pidiendo.
        cooldowns_del_pedido = {c: v for c, v in self.cooldowns.items()
                                if c in claves_del_pedido}
        return Respuesta(503, {"error": {
            "message": "sin rutas disponibles",
            "detalle": ultimo_error,
            "proxima_liberacion": (min(cooldowns_del_pedido.values())
                                   if cooldowns_del_pedido else None),
        }}, None, intentos, "", ultimo_codigo)

    async def completar_stream(self, rutas: list[Route], cuerpo: dict, ahora: float,
                               crudo: bool = False,
                               en_ruta_comprometida: Callable[[Route], None] | None = None,
                               en_intento_facturable: Callable[[Route], None] | None = None):
        """Emite lineas SSE ya recortadas, terminando siempre en `data: [DONE]`.

        Hace failover solo ANTES del primer byte util: una vez que al cliente le
        llego contenido de una ruta, cambiar de modelo mezclaria dos respuestas
        distintas en un mismo stream. Por eso una falla de red DESPUES de emitir
        no reintenta la siguiente ruta: cierra el stream ahi mismo.

        `en_ruta_comprometida`, si se pasa, se llama COMO MUCHO una vez por
        llamada a este generador: exactamente cuando (y si) una ruta queda
        confirmada como la que de verdad sirvio la peticion. Se dispara desde
        el mismo lugar que ya decide "esto fue un exito real" para la
        telemetria (`_registrar_exito_una_vez`, mas abajo) -- no desde el
        status 200 crudo, porque un 200 que muere sin emitir nada antes del
        primer byte util TODAVIA hace failover a la siguiente ruta (ver mas
        arriba) y ahi no hubo servicio real.

        `en_intento_facturable`, si se pasa, se llama por cada intento contra
        una ruta de PAGO cuyo status inicial fue `200` -- exitoso o no (HIGH
        4, round 9: el proveedor cobra por generar la respuesta, incluido un
        stream que termina sin contenido util). Distinto de
        `en_ruta_comprometida`: ese marca solo la ruta GANADORA, este marca
        todo intento facturable.

        Nunca se llama con `es_sonda` (round 8: sondeo.py jamas streamea) --
        todo fallo aca es TRAFICO REAL, y CADA rama de fallo (status != 200,
        stream sin contenido util, error de red, y el corte por
        `if emitido:` que no puede seguir la cadena) pasa por
        `_reaccionar_a_fallo_no_429` por igual -- el mismo punto unico de
        decision que usa completar(), para que las dos ramas no puedan
        volver a desincronizarse (ese fue justo el vector de round 8)."""
        for ruta in rutas:
            proveedor = self.proveedores[ruta.provider]
            url, cabeceras, payload = build_request(proveedor, cuerpo, ruta.model_id)
            payload["stream"] = True
            t0 = time.monotonic()
            emitido = False          # ya salio algun chunk hacia el cliente
            util = False             # ...y al menos uno traia contenido o tool_calls
            evento_registrado = False  # ya se conto la telemetria de este intento
            # Solo lo usa el camino de emulacion (abajo): ahi la respuesta se
            # BUFFEREA entera antes de poder decidir si es un tool call, asi que
            # para cuando se llama a _registrar_exito_una_vez ya paso toda la
            # generacion. Sin esto, esas rutas reportaban el tiempo TOTAL como
            # ttft -- 7-27s para un modelo que razona -- y ranking.latency_factor
            # las hundia en el puntaje justo por usarlas. Se estampa cuando llega
            # el PRIMER fragmento, que es lo que ttft significa.
            ttft_medido_ms = None

            def _registrar_exito_una_vez() -> None:
                # Exactamente un evento por intento, nunca cero ni dos: se
                # dispara la primera vez que hay algo UTIL que mandar (asi el
                # ttft mide el primer token real, no el cierre del stream). Si
                # nunca hubo nada util el intento se cierra abajo con un evento
                # FALLIDO, no con este: un 200 que no entrega contenido ni
                # tool_calls no sirvio a nadie, y contarlo como exito le sube
                # la confiabilidad a la ruta que acaba de fallar.
                nonlocal evento_registrado
                if not evento_registrado:
                    ttft = (ttft_medido_ms if ttft_medido_ms is not None
                            else int((time.monotonic() - t0) * 1000))
                    self.almacen.record_event(ruta.key, True, ttft, 200, ahora)
                    evento_registrado = True
                    self._limpiar_castigo(ruta.key)
                    self._sospechas.pop(ruta.key, None)
                    self._fallos_pago.pop(ruta.key, None)
                    if en_ruta_comprometida is not None:
                        en_ruta_comprometida(ruta)

            try:
                async with self.http.stream("POST", url, headers=cabeceras, json=payload,
                                            timeout=_timeout_de(proveedor)) as resp:
                    ahora_del_castigo = ahora + (time.monotonic() - t0)
                    if resp.status_code != 200:
                        if resp.status_code == 429:
                            self._castigar_429(ruta.key, ahora_del_castigo, resp)
                        elif not _es_error_del_cliente(resp.status_code):
                            self._reaccionar_a_fallo_no_429(
                                ruta, ahora, ahora_del_castigo, es_sonda=False)
                        self.almacen.record_event(
                            ruta.key, False, 0, resp.status_code, ahora,
                            is_client_error=_es_error_del_cliente(resp.status_code))
                        continue
                    if ruta.tier == "pago" and en_intento_facturable is not None:
                        en_intento_facturable(ruta)
                    rec = CompositeStreamTrimmer(unwrap_canvas=proveedor.unwraps_canvas)

                    # Emulacion de tools en streaming. Detectar un tool call exige
                    # el TEXTO COMPLETO (el JSON llega partido en deltas), asi que
                    # este camino junta la respuesta entera y recien ahi decide:
                    # o la emite como un chunk de tool_calls, o la suelta como
                    # texto. El precio es perder el streaming incremental para
                    # estas rutas -- inevitable, y solo cuando el cliente pidio
                    # tools contra un proveedor que no las soporta nativamente.
                    #
                    # `nombres_tools` es el allow-list del pedido del CLIENTE: sin
                    # el, un texto legitimo que traiga JSON se convertiria en una
                    # llamada inventada (ver el docstring de tool_emulator).
                    #
                    # OJO con el orden: `emula_tools` se evalua ANTES de mirar
                    # `tools`. Al reves, un `tools` malformado de un cliente
                    # cualquiera reventaba el streaming de TODOS los proveedores,
                    # incluidos los que no emulan nada -- y la excepcion salia
                    # DENTRO del generador, con el 200 y las cabeceras SSE ya
                    # mandadas: stream cortado y sin failover posible.
                    if proveedor.emulates_tools and _emu.tool_names(cuerpo.get("tools")):
                        nombres_tools = _emu.tool_names(cuerpo.get("tools"))
                        acumulado = ""
                        # id/created/model/usage del proveedor: son campos que el
                        # esquema de chunk marca como obligatorios, y el chunk
                        # sintetico era el unico del gateway que los perdia.
                        sobre = {}
                        async for linea_buf in resp.aiter_lines():
                            if not linea_buf.startswith("data:"):
                                continue
                            carga_buf = linea_buf[5:].strip()
                            if carga_buf == "[DONE]":
                                break
                            try:
                                obj_buf = json.loads(carga_buf)
                            except json.JSONDecodeError:
                                continue
                            if not isinstance(obj_buf, dict):
                                continue
                            for campo in ("id", "created", "model", "system_fingerprint"):
                                if campo not in sobre and obj_buf.get(campo) is not None:
                                    sobre[campo] = obj_buf[campo]
                            # `usage` llega en un chunk final propio (con
                            # choices vacio) cuando el cliente pidio
                            # stream_options.include_usage: se pisa siempre para
                            # quedarse con el ultimo, que es el total.
                            if obj_buf.get("usage") is not None:
                                sobre["usage"] = obj_buf["usage"]
                            elec_buf = (obj_buf.get("choices") or [{}])[0]
                            if not isinstance(elec_buf, dict):
                                continue
                            frag = (elec_buf.get("delta") or {}).get("content")
                            if not isinstance(frag, str):
                                continue
                            if ttft_medido_ms is None and frag:
                                ttft_medido_ms = int((time.monotonic() - t0) * 1000)
                            acumulado += rec.feed(frag) if not crudo else frag
                        if not crudo:
                            acumulado += rec.close()

                        llamadas = _emu.parse_tool_calls(acumulado, nombres_tools)
                        if llamadas and cuerpo.get("tool_choice") != "none":
                            _registrar_exito_una_vez()
                            util = True
                            chunk = _emu.build_stream_chunk(llamadas, sobre)
                            yield f"data: {json.dumps(chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                            return
                        if acumulado.strip():
                            # No era un tool call: se entrega como respuesta de
                            # texto normal, que es exactamente lo que el cliente
                            # habria recibido sin emulacion.
                            _registrar_exito_una_vez()
                            util = True
                            chunk = _emu.build_stream_chunk(None, sobre, acumulado)
                            yield f"data: {json.dumps(chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                            return
                        # 200 que no entrego nada: mismo tratamiento que el
                        # camino normal -- intento FALLIDO y failover a la
                        # siguiente ruta. Nada se emitio, asi que sigue limpio.
                        if not evento_registrado:
                            self.almacen.record_event(ruta.key, False, 0, 200, ahora)
                            evento_registrado = True
                            self._reaccionar_a_fallo_no_429(
                                ruta, ahora, ahora + (time.monotonic() - t0), es_sonda=False)
                        continue

                    # Chunks recibidos que todavia no llevan nada util. Se
                    # retienen (no se emiten) hasta que llegue el primero que
                    # SI: mientras nada haya salido, el failover sigue siendo
                    # limpio. Ver TOPE_PENDIENTES.
                    pendientes: list[str] = []
                    async for linea in resp.aiter_lines():
                        if not linea.startswith("data:"):
                            continue
                        carga = linea[5:].strip()
                        if carga == "[DONE]":
                            break
                        try:
                            obj = json.loads(carga)
                        except json.JSONDecodeError:
                            continue
                        eleccion = (obj.get("choices") or [{}])[0]
                        if not isinstance(eleccion, dict):
                            eleccion = {}
                        delta = eleccion.get("delta") or {}
                        # Un chunk de tool_calls (o el de role inicial) suele
                        # viajar con content="": miramos la PRESENCIA de otras
                        # claves, no su valor, porque algo como "tool_calls": []
                        # (valor falsy pero presente) igual es util para el
                        # cliente y no se puede tirar junto con el contenido.
                        #
                        # Se mira el chunk ENTERO, en sus tres niveles, salteando
                        # las claves de sobre (ver _SOBRE_CHUNK/_SOBRE_ELECCION):
                        # `finish_reason` vive al lado de `delta` y `usage` al
                        # nivel superior, y mirando solo `delta` los dos se
                        # perdian.
                        otras = ({k for k in delta if k != "content"}
                                 | {k for k in eleccion
                                    if k != "delta" and k not in _SOBRE_ELECCION}
                                 | {k for k in obj
                                    if k != "choices" and k not in _SOBRE_CHUNK})
                        if not crudo and isinstance(delta.get("content"), str):
                            delta["content"] = rec.feed(delta["content"])
                        contenido = delta.get("content")
                        # Dos preguntas distintas sobre el mismo chunk:
                        #  - hay_texto: trae RESPUESTA (algo que no sea espacio
                        #    en blanco). Es lo que decide si el intento fue un
                        #    exito -- una respuesta de puros espacios no es una
                        #    respuesta, igual que en el camino no-streaming.
                        #  - tiene_contenido: trae texto del cliente, aunque sea
                        #    un " " suelto. Los deltas vienen muy partidos y esos
                        #    espacios son parte de la frase: no se pueden tirar.
                        hay_texto = isinstance(contenido, str) and bool(contenido.strip())
                        tiene_contenido = isinstance(contenido, str) and contenido != ""
                        trozo = f"data: {json.dumps(obj)}\n\n"
                        if not hay_texto and "tool_calls" not in delta:
                            # Nada util TODAVIA. Lo estructural (role,
                            # finish_reason, razonamiento del proveedor) y los
                            # espacios sueltos se guardan para soltarlos EN
                            # ORDEN junto al primer chunk util; un chunk que ya
                            # no tiene nada adentro se descarta.
                            if not (tiene_contenido or otras):
                                continue
                            pendientes.append(trozo)
                            if len(pendientes) > TOPE_PENDIENTES:
                                # Retener sin limite un stream que solo escupe
                                # razonamiento seria una fuga de memoria: se
                                # sueltan (se pierde el failover limpio) pero
                                # el intento sigue contando como fallido si
                                # nunca llega contenido de verdad.
                                log.info(
                                    "stream de %s: mas de %d chunks sin contenido "
                                    "retenidos; se sueltan y este intento ya no puede "
                                    "hacer failover limpio", ruta.key, TOPE_PENDIENTES)
                                for p in pendientes:
                                    yield p
                                pendientes.clear()
                                emitido = True
                            continue
                        _registrar_exito_una_vez()
                        util = True
                        for p in pendientes:
                            yield p
                        pendientes.clear()
                        emitido = True
                        yield trozo
                    resto = rec.close()
                    if resto.strip():
                        _registrar_exito_una_vez()
                        util = True
                        for p in pendientes:
                            yield p
                        pendientes.clear()
                        emitido = True
                        yield ('data: {"choices":[{"delta":{"content":%s}}]}\n\n'
                               % json.dumps(resto))
                    if not util:
                        # 200 que nunca entrego contenido ni tool_calls: el
                        # mismo agujero que arriba, del lado del streaming. La
                        # conexion funciono a nivel HTTP, pero el cliente se
                        # queda sin respuesta -- se registra como intento
                        # FALLIDO y se cae a la siguiente ruta.
                        if not evento_registrado:
                            self.almacen.record_event(ruta.key, False, 0, 200, ahora)
                            evento_registrado = True
                            self._reaccionar_a_fallo_no_429(
                                ruta, ahora, ahora + (time.monotonic() - t0), es_sonda=False)
                        if pendientes:
                            # Lo retenido se va a la basura junto con el intento.
                            # Es lo correcto (nada de eso llego al cliente, asi
                            # que el failover sigue siendo limpio) pero no puede
                            # ser silencioso: son chunks de una ruta que dijo 200.
                            log.info(
                                "stream de %s: se descartan %d chunk(s) retenidos; el "
                                "intento cerro sin contenido ni tool_calls",
                                ruta.key, len(pendientes))
                        if emitido:
                            # Ya se solto lo retenido (ver TOPE_PENDIENTES): no
                            # se puede empalmar otra ruta encima sin mezclar
                            # dos respuestas. Round 8: este ERA el escape hatch
                            # que comprometia la ruta sin comparar con nadie --
                            # ahora es igual que cualquier otra rama de fallo,
                            # `_sospechar` de arriba ya la registro, no hay
                            # nada mas que commitear aca.
                            yield "data: [DONE]\n\n"
                            return
                        continue
                    for p in pendientes:     # p.ej. el chunk final de finish_reason
                        yield p
                    yield "data: [DONE]\n\n"
                    return
            except httpx.HTTPError:
                if not evento_registrado:
                    self.almacen.record_event(ruta.key, False, 0, 0, ahora)
                    self._reaccionar_a_fallo_no_429(
                        ruta, ahora, ahora + (time.monotonic() - t0), es_sonda=False)
                if emitido:
                    yield "data: [DONE]\n\n"
                    return
                continue

        yield 'data: {"error":{"message":"sin rutas disponibles"}}\n\n'
        yield "data: [DONE]\n\n"

    def _castigar(self, clave: str, ahora: float) -> None:
        """Castigo con backoff exponencial -- SONDA confirmada (periodica o
        bajo demanda) o fallos de pago directos (`_sospechar_pago`). Round
        9: el 429 YA NO pasa por aca, ver `_castigar_429`."""
        n = self._castigos.get(clave, 0) + 1
        self._castigos[clave] = n
        self.cooldowns[clave] = ahora + min(COOLDOWN_BASE_S * (2 ** (n - 1)), COOLDOWN_TOPE_S)
        self._generacion_cooldown[clave] = self._generacion_cooldown.get(clave, 0) + 1

    def _castigar_429(self, clave: str, ahora: float, resp: httpx.Response | None) -> None:
        """Round 9, MEDIUM 7: el 429 sigue castigando de inmediato (senal
        inequivoca de la ruta, no necesita sonda) pero YA NO reutiliza el
        backoff exponencial de `_castigar` (que podia escalar hasta
        COOLDOWN_TOPE_S=3600s). Se respeta el `Retry-After` del proveedor
        cuando lo manda; si no, se usa un default corto
        (COOLDOWN_429_DEFAULT_S); en cualquier caso se topea a
        COOLDOWN_429_MAXIMO_S. Ver el comentario de cabecera de las
        constantes para la medicion que motivo el cambio."""
        duracion = self._retry_after_segundos(resp)
        if duracion is None:
            duracion = COOLDOWN_429_DEFAULT_S
        duracion = min(duracion, COOLDOWN_429_MAXIMO_S)
        self.cooldowns[clave] = ahora + duracion
        self._generacion_cooldown[clave] = self._generacion_cooldown.get(clave, 0) + 1

    @staticmethod
    def _retry_after_segundos(resp: httpx.Response | None) -> float | None:
        """Parsea `Retry-After` (segundos, o una fecha HTTP) si el proveedor
        lo manda. `None` si falta o no se puede interpretar -- el llamador
        cae a un default.

        Round 10, fix chico del gate: un `Retry-After` HOSTIL o roto
        (`-5`, `nan`) pasaba el `max(0.0, ...)` de antes y volvia UN
        COOLDOWN DE 0s -- un proveedor diciendo explicitamente "parate" (un
        429 es tan inequivoco como una sonda) terminaba SIN NINGUN
        castigo, martillado de inmediato otra vez. Un valor negativo,
        no-finito (`nan`/`inf`) o directamente ilegible se trata IGUAL que
        ausente -- cae al default corto, nunca a cero."""
        if resp is None:
            return None
        valor = resp.headers.get("Retry-After")
        if not valor:
            return None
        try:
            segundos = float(valor)
        except ValueError:
            segundos = None
        if segundos is not None and math.isfinite(segundos) and segundos >= 0:
            return segundos
        if segundos is not None:
            return None  # negativo o no-finito: como si no hubiera Retry-After
        try:
            from email.utils import parsedate_to_datetime
            fecha = parsedate_to_datetime(valor)
            if fecha is None:
                return None
            import datetime as _dt
            zona = fecha.tzinfo or _dt.timezone.utc
            segundos = (fecha - _dt.datetime.now(zona)).total_seconds()
            return segundos if math.isfinite(segundos) and segundos >= 0 else None
        except Exception:
            return None

    def _limpiar_castigo(self, clave: str) -> None:
        """Un exito borra TODO rastro de castigo previo -- 429 y sondas
        fallidas por igual. Factorizado para que completar() y
        completar_stream() (con su _registrar_exito_una_vez) no puedan
        desincronizarse en cuales diccionarios limpian."""
        self._castigos.pop(clave, None)
        self.cooldowns.pop(clave, None)
        self._generacion_cooldown[clave] = self._generacion_cooldown.get(clave, 0) + 1

    def _reaccionar_a_fallo_no_429(self, ruta: Route, ahora: float, ahora_del_castigo: float,
                                    es_sonda: bool) -> None:
        """Punto UNICO de decision para un fallo NO-429 (ver
        _es_error_del_cliente) -- lo comparten completar() y
        completar_stream() para que las dos ramas no puedan
        desincronizarse en como tratan la misma situacion: ese fue
        exactamente el vector de round 8 (la rama `if emitido:` de
        completar_stream tenia su propio atajo)."""
        if es_sonda:
            self._castigar(ruta.key, ahora_del_castigo)
        elif ruta.tier == "pago":
            self._sospechar_pago(ruta, ahora_del_castigo)
        else:
            self._sospechar(ruta, ahora)

    def _sospechar(self, ruta: Route, ahora: float) -> None:
        """Round 8/9: un fallo de TRAFICO REAL nunca excluye una ruta el
        mismo -- solo acumula sospecha. Al cruzar UMBRAL_SOSPECHA fallos
        CONSECUTIVOS (round 9, HIGH 3: ya no "dentro de una ventana de
        tiempo" -- ver el comentario de cabecera de UMBRAL_SOSPECHA), la
        ruta queda ANOTADA como candidata a una sonda BAJO DEMANDA (mismo
        payload fijo `PING` que la sonda periodica); `_admitir_sondas_pendientes`
        decide, con criterio de EQUIDAD, a quien de los candidatos le toca
        el proximo cupo. Es esa sonda -- nunca este metodo, nunca el pedido
        del cliente -- la que decide si la ruta se castiga
        (`completar(..., es_sonda=True)`, en `_probar_bajo_demanda`).

        Solo para rutas GRATIS -- las de pago pasan por `_sospechar_pago`
        (ver `_reaccionar_a_fallo_no_429`), que castiga DIRECTO sin sonda:
        una sonda bajo demanda gastaria PLATA REAL contra un proveedor de
        pago sin un dueno natural de esa cuenta (el tope diario es POR
        LLAVE, no por ruta)."""
        self._sospechas[ruta.key] = min(self._sospechas.get(ruta.key, 0) + 1, UMBRAL_SOSPECHA)
        if self._sospechas[ruta.key] < UMBRAL_SOSPECHA:
            return
        if ruta.key in self._sondas_en_curso:
            return  # ya hay una sonda en vuelo para esta ruta -- no duplicar
        ultimo = self._ultimo_probe_bajo_demanda.get(ruta.key, float("-inf"))
        if ahora - ultimo < LIMITE_PROBE_BAJO_DEMANDA_S:
            return  # rate-limitada por ruta -- la sospecha queda para la proxima oportunidad
        # No se decide ACA si hay cupo -- se anota como candidata y se le
        # pide a `_admitir_sondas_pendientes` que reparta el cupo disponible
        # por EQUIDAD entre TODAS las candidatas actuales, no solo esta.
        self._esperando_sonda[ruta.key] = ruta
        self._admitir_sondas_pendientes(ahora)

    def _admitir_sondas_pendientes(self, ahora: float) -> None:
        """Round 10, HIGH del gate: reparte el cupo global (round 9,
        MEDIUM 6) por EQUIDAD, no por orden de llegada. `completar()`
        recorre la cadena en el MISMO orden en cada pedido (prioridad,
        luego confiabilidad) -- asi que, con admision "primero que
        llega, primero que entra", las rutas al FRENTE de la cadena piden
        sonda primero SIEMPRE, y una ruta genuinamente rota (cuya
        confiabilidad colapsada la manda al FINAL de la cadena) nunca
        alcanzaba a pedir antes de que el cupo del minuto se agotara.
        Medido: con 6 rutas y trafico que sospecha de todas por igual, la
        ruta en la posicion 5 nunca conseguia una sonda en 60 minutos
        simulados -- exactamente la ruta que MAS necesitaba confirmarse,
        indefinidamente relegada.

        Prioridad: la candidata con MAS TIEMPO sin una sonda bajo demanda
        (nunca sondeada = infinito, gana siempre) se lleva el proximo cupo
        libre -- un round-robin efectivo por staleness, re-evaluado en
        CADA llamada (no una cola FIFO fija): apenas una ruta consigue su
        sonda, dejar de ser la mas vieja hace que la SIGUIENTE mas vieja
        gane la proxima vez, sin importar en que orden pidieron dentro de
        una misma cadena. El TOPE global sigue siendo el mismo (round 9):
        esto cambia A QUIEN se admite, no CUANTAS sondas por minuto."""
        corte_global = ahora - VENTANA_PROBE_GLOBAL_S
        self._probes_recientes[:] = [m for m in self._probes_recientes if m >= corte_global]
        candidatas = sorted(
            self._esperando_sonda.values(),
            key=lambda r: self._ultimo_probe_bajo_demanda.get(r.key, float("-inf")))
        for ruta in candidatas:
            if len(self._probes_recientes) >= LIMITE_PROBE_GLOBAL_POR_MINUTO:
                break  # cupo agotado -- las que quedan esperan a la proxima oportunidad
            if ruta.key in self._sondas_en_curso:
                self._esperando_sonda.pop(ruta.key, None)
                continue
            self._esperando_sonda.pop(ruta.key, None)
            self._probes_recientes.append(ahora)
            self._ultimo_probe_bajo_demanda[ruta.key] = ahora
            tarea = asyncio.create_task(self._probar_bajo_demanda(ruta, ahora))
            self._sondas_en_curso[ruta.key] = tarea
            tarea.add_done_callback(
                lambda _t, clave=ruta.key: self._sondas_en_curso.pop(clave, None))

    def _sospechar_pago(self, ruta: Route, ahora: float) -> None:
        """HIGH 4 (round 9): las rutas de pago quedan afuera de sospecha+
        sonda (ver _sospechar) porque una sonda bajo demanda gastaria plata
        real sin dueno -- pero eso NO puede significar que nada las
        excluya: sin esto, una ruta de pago rota factura cada pedido, para
        siempre, sin que ningun mecanismo la saque de rotacion (medido:
        40/40 llamadas facturables, `/v1/uso` en `pago_hoy: 0` porque solo
        se contaba el EXITO, `TOPE_PAGO_DIARIO` nunca actuaba -- ver
        api.py para el otro lado de este fix, contar lo FACTURADO). Se
        reintroduce un castigo DIRECTO (sin sonda, como round 7), con el
        MISMO umbral que la sospecha de rutas gratis, acotado a rutas de
        pago especificamente: van ULTIMAS en la cadena (§7), asi que solo
        entran en juego cuando YA se agoto todo lo gratis -- el costo de un
        falso positivo aca es mucho mas bajo que para una ruta gratis
        (round 8 completo), y el costo de NO excluir (gasto sin fondo) es
        estrictamente peor.

        Round 10, MEDIUM del gate: la duracion de ese castigo directo NO
        puede reusar `_castigar` (el backoff exponencial existe para
        SONDAS CONFIRMADAS) -- exactamente el mismo defecto que el 429
        tenia antes de round 9, en otro lugar. Medido a traves de la API
        real: 60->120->...->3600s en 24 pedidos de una llave, con el
        fallback de pago afuera para TODAS las llaves por una hora. Flat,
        capped (`COOLDOWN_PAGO_DIRECTO_S`), sin escalar -- igual que
        `_castigar_429`."""
        n = min(self._fallos_pago.get(ruta.key, 0) + 1, UMBRAL_SOSPECHA)
        self._fallos_pago[ruta.key] = n
        if n >= UMBRAL_SOSPECHA:
            self.cooldowns[ruta.key] = ahora + COOLDOWN_PAGO_DIRECTO_S
            self._generacion_cooldown[ruta.key] = self._generacion_cooldown.get(ruta.key, 0) + 1
            self._fallos_pago.pop(ruta.key, None)

    async def _probar_bajo_demanda(self, ruta: Route, ahora: float) -> None:
        """La sonda que `_sospechar` programa. Reutiliza completar() con
        `es_sonda=True` sobre una cadena de UNA sola ruta -- el mismo
        camino que ya castiga de forma inequivoca -- y ademas deja
        constancia en `sondas` (tabla que alimenta tanto confiabilidad como
        `Storage.has_liveness_evidence`), exactamente como
        `probing.probe_health` para la periodica: para el resto del
        sistema, una sonda bajo demanda y una periodica son
        indistinguibles, solo cambia quien la disparo.

        MEDIUM 5 (round 9): un exito de ESTA sonda nunca debe cancelar un
        429 mas nuevo que ella (un cliente cancelando el "pariate" del
        proveedor via una sonda bajo demanda seria un camino a que
        baneen la llave compartida). Se guarda un cooldown YA activo (skip
        total, ni se gasta la sonda) y la GENERACION de castigo/cooldown
        antes de arrancar; solo se limpia si nada la toco mientras la sonda
        estaba en vuelo.

        LOW 8 (round 9): esto corre en un `asyncio.Task` en segundo plano
        -- una excepcion no-HTTP (p.ej. `sqlite3.OperationalError` bajo
        contencion de WAL con el planificador de sondeo escribiendo en
        paralelo) no la atrapa `completar()` (que solo espera
        `httpx.HTTPError`) y quedaria sin recuperar (silenciosa, y con
        `_sospechas` sin limpiar: la ruta quedaria trabada, sin poder volver
        a sospecharse hasta que decayera el rate-limit, con una sospecha
        vieja pegada). Se atrapa, se loguea, y se limpia `_sospechas`
        SIEMPRE para no dejar la ruta en un estado a medio camino."""
        if self.cooldowns.get(ruta.key, 0.0) > ahora:
            self._sospechas.pop(ruta.key, None)
            return
        generacion_antes = self._generacion_cooldown.get(ruta.key, 0)
        try:
            t0 = time.monotonic()
            r = await self.completar([ruta], dict(PING), ahora, es_sonda=True)
            ms = int((time.monotonic() - t0) * 1000)
            # Round 10, fix chico: se estampa con AHORA + cuanto tardo la
            # sonda, no con el `ahora` de cuando se PROGRAMO -- misma clase
            # de bug que HIGH 2 de round 9, un escalon mas arriba: para una
            # ruta colgada (hasta TIMEOUT_S=90s) la fila de `sondas` quedaba
            # fechada hasta 90s en el pasado, pudiendo des-ordenar el
            # `ORDER BY momento DESC` que usa `tiene_evidencia_de_vida`.
            ahora_resuelto = ahora + ms / 1000.0
            if r.codigo_upstream != 429:
                # Fix chico: un 429 contra la SONDA ya tiene su propio
                # castigo proporcional (_castigar_429, adentro de
                # completar()) -- es evidencia de que la ruta esta
                # rate-limitada AHORA, no de que este rota. Grabarlo
                # TAMBIEN como sonda de salud fallida lo confundiria con
                # una ruta genuinamente caida: dos 429 seguidos contra la
                # sonda bastarian para que `tiene_evidencia_de_vida`
                # (round 9) la de por muerta -- una senal de capacidad
                # momentanea no es evidencia de muerte.
                self.almacen.record_probe(ruta.key, "salud", r.estado == 200, ms, 0,
                                             r.codigo_upstream, 0, 0, ahora_resuelto)
            if r.estado == 200 and self._generacion_cooldown.get(ruta.key, 0) == generacion_antes:
                self._limpiar_castigo(ruta.key)
        except Exception:
            log.exception(
                "sonda bajo demanda de %s fallo con una excepcion no-HTTP "
                "(posible contencion de SQLite bajo WAL) -- no se registra "
                "sonda ni se castiga/limpia: la ruta queda SIN VEREDICTO, "
                "no muerta por accidente.", ruta.key)
        finally:
            self._sospechas.pop(ruta.key, None)

    async def esperar_sondas_pendientes(self) -> None:
        """Solo para tests: las sondas bajo demanda corren en segundo plano
        (ver _sospechar) para no sumarle latencia al pedido del cliente que
        las disparo. Awaitear esto deja que las que esten en vuelo AHORA
        terminen antes de afirmar sobre cooldowns/sondas -- sin esto, un
        test que revisa `p.cooldowns` justo despues de disparar el umbral
        puede correr antes de que la sonda en background haya corrido."""
        tareas = list(self._sondas_en_curso.values())
        if tareas:
            await asyncio.gather(*tareas)

    @staticmethod
    def _limpiar(datos: dict, desenvolver_canvas: bool) -> str:
        razon_total = ""
        for eleccion in datos.get("choices", []):
            msg = eleccion.get("message") or {}
            contenido = msg.get("content")
            if isinstance(contenido, str):
                limpio, razon = trim(contenido, desenvolver_canvas)
                msg["content"] = limpio
                razon_total += razon
        return razon_total
