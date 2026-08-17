import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from llm_libre.cliente import armar_peticion
from llm_libre.modelos import Ruta
from llm_libre.razonamiento import RecortadorStreamCompuesto, recortar

log = logging.getLogger(__name__)

COOLDOWN_BASE_S = 60.0
COOLDOWN_TOPE_S = 3600.0
TIMEOUT_S = 90.0

# Revision de Task 13, hallazgo 2. Solo un 429 castigaba (con backoff
# exponencial, ver _castigar): un 500, un timeout o un error de red no
# dejaban NUNCA un cooldown, asi que una ruta persistentemente rota o
# COLGADA se seguia probando en cada pedido, adelante de rutas sanas segun
# su prioridad, para siempre -- con TIMEOUT_S=90 eso son hasta 5*90s=450s
# por pedido en la cadena mas larga, y /health sigue en "ok" mientras haya
# UNA ruta viva. `blog` es una maquina saturada: colgado-no-rechazado es el
# modo de falla realista, no un 500 limpio.
#
# Un hiccup aislado no debe sacar una ruta sana de rotacion -- por eso el
# castigo no es INMEDIATO como el del 429 (que si es una senal inequivoca:
# el proveedor esta pidiendo que se lo deje de llamar). Recien al superar
# TOPE_FALLOS_SEGUIDOS fallos NO-429 SEGUIDOS (sin ningun exito en el medio)
# se aplica el MISMO backoff exponencial que ya usa el 429 (comparten
# _castigos/_castigar): la diferencia es CUANDO se dispara, no cuanto dura.
TOPE_FALLOS_SEGUIDOS = 3

# Round 7. La atribucion de mas arriba (que codigo cuenta como evidencia del
# PEDIDO, Parte 1) se aplicaba solo a nivel de RESPUESTA -- cada ruta se
# juzgaba sola, sin mirar que paso con sus hermanas en la MISMA cadena. Pero
# `completar`/`completar_stream` recorren la cadena entera por pedido, asi
# que saben algo que ninguna respuesta individual puede decir: si el MISMO
# pedido falla en TODAS las rutas de la cadena, el factor comun es el
# pedido, no todas las rutas a la vez -- y eso vale incluso para fallas que
# `_es_error_del_cliente` no puede clasificar como evidencia del pedido
# (un 403 de moderacion, un 451 legal/geografico, un timeout por un prompt
# gigante, un 200 sin contenido de un modelo de razonamiento): las cuatro
# son, correctamente, evidencia de LA RUTA por Parte 1 -- pero tambien las
# cuatro son deterministas, CUALQUIER ruta le devuelve lo mismo a ESTE
# pedido puntual.
#
# Regla: un fallo solo se COMMITEA hacia el contador de fallos duros de SU
# ruta si, en esa MISMA vuelta de la cadena, alguna OTRA ruta tuvo exito --
# evidencia real de que esa ruta en particular fallo, no el pedido. Si la
# cadena se agota entera sin ningun exito Y tiene mas de una ruta, se
# DESCARTAN todos los fallos de esa vuelta: con esta unica senal no hay
# forma de distinguir "el pedido esta envenenado" de "las rutas fallaron
# todas a la vez de verdad".
#
# Por que descartar y no, por ejemplo, contar la mitad: COOLDOWN ES UNA
# EXCLUSION, lo contrario de confiabilidad (una MEDICION). Parte 1 le puso
# el default "cuando no se sabe, contar" a confiabilidad porque ahi una
# falsa alarma se autocorrige sola -- alguien mira el ranking, ve que la
# ruta esta bien, sigue. Una ruta sana EXCLUIDA por error, en cambio, es
# exactamente el apagon con luz verde que el mecanismo de cooldown existe
# para prevenir -- asi que ante la misma duda, el default tiene que ser el
# contrario: no excluir.
#
# `429` no pasa por este mecanismo -- sigue castigando de inmediato
# (`_castigar`, sin pasar por `_fallos_seguidos`), sin importar el resto de
# la cadena: es una senal inequivoca de la capacidad de ESA ruta ahora
# mismo, no algo que un payload pueda inducir identico en varios
# proveedores independientes a la vez.
#
# Una cadena de UNA sola ruta (la sonda de salud SIEMPRE lo es, ver
# sondeo.py -- y tambien un pedido con un unico candidato posible) no tiene
# con que compararse: sigue commiteando de inmediato, sin cambios. Es,
# ademas, la salida de emergencia para una ruta genuinamente rota que ya no
# tiene ninguna hermana sana en su cadena: la sonda (payload fijo, que
# controla el propio gateway -- nunca ambiguo) la sigue enfriando aunque el
# trafico real deje de poder hacerlo.


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
# CONTARLO. Una falsa alarma se recupera sola -- alguien mira el ranking o
# /health, ve que la ruta esta bien, sigue. Una salida silenciosa NO --
# nadie mira nunca. Los costos son asimetricos, y el default tiene que
# inclinarse hacia notar, no hacia callar.
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
    es evidencia de la ruta por DEFAULT y cuenta igual que un 500 hacia el
    cooldown de fallos duros y hacia confiabilidad (ver
    Almacen.registrar_evento) -- ver el comentario de cabecera del
    conjunto para el principio completo ("cuando no se puede saber de
    quien es la culpa, hay que contarlo").

    Solo los tres codigos de la lista corta NI cuentan hacia el cooldown NI
    hacia confiabilidad: contarlos convertiria el error de UN cliente en un
    apagon para TODOS. Verificado contra el registro real de 5 rutas: tres
    pedidos malformados (400) seguidos bastan para dejar las cinco en
    cooldown si se cuentan, con una llave DISTINTA recibiendo 503 mientras
    tanto."""
    return codigo in _CODIGOS_EVIDENCIA_DE_PEDIDO


def _timeout_de(proveedor) -> float:
    """`Proveedor.timeout_s` (default None) permite acotar el peor caso de UN
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
    ruta: Ruta | None
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
        self._fallos_seguidos: dict[str, int] = {}

    async def completar(self, rutas: list[Ruta], cuerpo: dict, ahora: float,
                        crudo: bool = False) -> Respuesta:
        intentos = 0
        ultimo_error = None
        ultimo_codigo = 0
        claves_del_pedido = {ruta.clave for ruta in rutas}
        # Ver el comentario de cabecera de TOPE_FALLOS_SEGUIDOS (round 7):
        # los fallos duros de esta vuelta se acumulan aca SIN commitear hacia
        # `_fallos_seguidos` todavia -- recien se sabe si son evidencia de SU
        # ruta (una hermana tuvo exito) o del pedido (la cadena entera
        # fallo) al terminar de recorrerla.
        fallos_pendientes: list[str] = []
        for ruta in rutas:
            proveedor = self.proveedores[ruta.proveedor]
            url, cabeceras, payload = armar_peticion(proveedor, cuerpo, ruta.modelo_id)
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
                razon = "" if crudo else self._limpiar(datos, proveedor.desenvuelve_canvas)
                if not hay_respuesta(datos):
                    datos = None
                    ultimo_error = "200 sin contenido ni tool_calls"

            exito = codigo == 200 and datos is not None
            self.almacen.registrar_evento(ruta.clave, exito, 0, codigo, ahora,
                                          latencia_ms=latencia,
                                          es_error_cliente=_es_error_del_cliente(codigo))

            if exito:
                self._limpiar_castigo(ruta.clave)
                # Una hermana tuvo exito en ESTA MISMA cadena: los fallos
                # anteriores de esta vuelta SI son evidencia de sus rutas
                # (no del pedido, que a esta ruta le fue bien) -- se
                # commitean recien ahora.
                self._comprometer_fallos_pendientes(fallos_pendientes, ahora)
                return Respuesta(200, datos, ruta, intentos, razon, codigo)

            if codigo == 429:
                self._castigar(ruta.clave, ahora)
            elif not _es_error_del_cliente(codigo):
                fallos_pendientes.append(ruta.clave)
            ultimo_error = ultimo_error or f"HTTP {codigo}"

        # La cadena se agoto SIN NINGUN exito. Con una sola ruta no hay
        # hermana con quien comparar -- se commitea igual que siempre (ver
        # el comentario de cabecera de TOPE_FALLOS_SEGUIDOS). Con mas de
        # una, el fallo es IDENTICO en todas: el factor comun es el pedido,
        # no que las rutas hayan fallado todas a la vez -- se descartan.
        if len(rutas) <= 1:
            self._comprometer_fallos_pendientes(fallos_pendientes, ahora)

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

    async def completar_stream(self, rutas: list[Ruta], cuerpo: dict, ahora: float,
                               crudo: bool = False,
                               en_ruta_comprometida: Callable[[Ruta], None] | None = None):
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
        arriba) y ahi no hubo servicio real. Sirve para que el llamador pueda
        contar uso de pago (u otra cosa) atado a "esta ruta sirvio", sin
        arriesgarse a contar de mas ni de menos.
        """
        # Ver el comentario de cabecera de TOPE_FALLOS_SEGUIDOS (round 7) y
        # el mismo mecanismo en completar(): se acumulan aca sin commitear
        # hasta saber si son evidencia de la ruta (una hermana tuvo exito EN
        # ESTA MISMA cadena) o del pedido (la cadena entera fallo igual).
        fallos_pendientes: list[str] = []
        for ruta in rutas:
            proveedor = self.proveedores[ruta.proveedor]
            url, cabeceras, payload = armar_peticion(proveedor, cuerpo, ruta.modelo_id)
            payload["stream"] = True
            t0 = time.monotonic()
            emitido = False          # ya salio algun chunk hacia el cliente
            util = False             # ...y al menos uno traia contenido o tool_calls
            evento_registrado = False  # ya se conto la telemetria de este intento

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
                    self.almacen.registrar_evento(
                        ruta.clave, True, int((time.monotonic() - t0) * 1000),
                        200, ahora)
                    evento_registrado = True
                    self._limpiar_castigo(ruta.clave)
                    # Esta ruta tuvo exito: los fallos anteriores de ESTA
                    # MISMA cadena si son evidencia de sus rutas (no del
                    # pedido, que aca le fue bien) -- se commitean recien
                    # ahora.
                    self._comprometer_fallos_pendientes(fallos_pendientes, ahora)
                    if en_ruta_comprometida is not None:
                        en_ruta_comprometida(ruta)

            try:
                async with self.http.stream("POST", url, headers=cabeceras, json=payload,
                                            timeout=_timeout_de(proveedor)) as resp:
                    if resp.status_code != 200:
                        if resp.status_code == 429:
                            self._castigar(ruta.clave, ahora)
                        elif not _es_error_del_cliente(resp.status_code):
                            fallos_pendientes.append(ruta.clave)
                        self.almacen.registrar_evento(
                            ruta.clave, False, 0, resp.status_code, ahora,
                            es_error_cliente=_es_error_del_cliente(resp.status_code))
                        continue
                    rec = RecortadorStreamCompuesto(desenvolver_canvas=proveedor.desenvuelve_canvas)
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
                            delta["content"] = rec.alimentar(delta["content"])
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
                                    "hacer failover limpio", ruta.clave, TOPE_PENDIENTES)
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
                    resto = rec.cerrar()
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
                            self.almacen.registrar_evento(ruta.clave, False, 0, 200, ahora)
                            evento_registrado = True
                            fallos_pendientes.append(ruta.clave)
                        if pendientes:
                            # Lo retenido se va a la basura junto con el intento.
                            # Es lo correcto (nada de eso llego al cliente, asi
                            # que el failover sigue siendo limpio) pero no puede
                            # ser silencioso: son chunks de una ruta que dijo 200.
                            log.info(
                                "stream de %s: se descartan %d chunk(s) retenidos; el "
                                "intento cerro sin contenido ni tool_calls",
                                ruta.clave, len(pendientes))
                        if emitido:
                            # Ya se solto lo retenido (ver TOPE_PENDIENTES): no
                            # se puede empalmar otra ruta encima sin mezclar
                            # dos respuestas. Se corta la cadena aca, no se
                            # llega a saber si el resto tambien fallaba --
                            # se commitea de inmediato (no es "la cadena se
                            # agoto sola", es "se decidio no seguir").
                            self._comprometer_fallos_pendientes(fallos_pendientes, ahora)
                            yield "data: [DONE]\n\n"
                            return
                        continue
                    for p in pendientes:     # p.ej. el chunk final de finish_reason
                        yield p
                    yield "data: [DONE]\n\n"
                    return
            except httpx.HTTPError:
                if not evento_registrado:
                    self.almacen.registrar_evento(ruta.clave, False, 0, 0, ahora)
                    fallos_pendientes.append(ruta.clave)
                if emitido:
                    # Idem arriba: se corta la cadena aca, no se agoto sola.
                    self._comprometer_fallos_pendientes(fallos_pendientes, ahora)
                    yield "data: [DONE]\n\n"
                    return
                continue

        # La cadena se agoto SIN NINGUN exito. Mismo criterio que completar():
        # con una sola ruta no hay hermana con quien comparar, se commitea
        # igual que siempre; con mas de una, el fallo es identico en todas
        # -- el factor comun es el pedido, se descartan.
        if len(rutas) <= 1:
            self._comprometer_fallos_pendientes(fallos_pendientes, ahora)

        yield 'data: {"error":{"message":"sin rutas disponibles"}}\n\n'
        yield "data: [DONE]\n\n"

    def _castigar(self, clave: str, ahora: float) -> None:
        n = self._castigos.get(clave, 0) + 1
        self._castigos[clave] = n
        self.cooldowns[clave] = ahora + min(COOLDOWN_BASE_S * (2 ** (n - 1)), COOLDOWN_TOPE_S)

    def _limpiar_castigo(self, clave: str) -> None:
        """Un exito borra TODO rastro de castigo previo -- 429 y fallos duros
        por igual. Factorizado para que completar() y completar_stream()
        (con su _registrar_exito_una_vez) no puedan desincronizarse en
        cuales de los tres diccionarios limpian."""
        self._castigos.pop(clave, None)
        self.cooldowns.pop(clave, None)
        self._fallos_seguidos.pop(clave, None)

    def _registrar_fallo(self, clave: str, ahora: float) -> None:
        """Cuenta un intento fallido que NO fue un 429 (ese ya castiga solo,
        ver _castigar). Un hiccup aislado no saca a una ruta sana de la
        rotacion: recien al llegar a TOPE_FALLOS_SEGUIDOS fallos SEGUIDOS
        (sin exito en el medio) se la castiga -- y el contador se reinicia
        para exigir otra racha completa antes de volver a castigar.

        Privado a proposito (round 7): NO se llama directo desde
        completar()/completar_stream() en el momento en que ocurre el fallo
        -- eso fue el bug (cada ruta se juzgaba sola, sin mirar la cadena).
        Los llamadores acumulan en una lista `fallos_pendientes` y recien la
        pasan por `_comprometer_fallos_pendientes` cuando se sabe si son
        evidencia de la ruta o del pedido. Ver el comentario de cabecera de
        TOPE_FALLOS_SEGUIDOS para la regla completa."""
        n = self._fallos_seguidos.get(clave, 0) + 1
        if n >= TOPE_FALLOS_SEGUIDOS:
            self._castigar(clave, ahora)
            self._fallos_seguidos.pop(clave, None)
        else:
            self._fallos_seguidos[clave] = n

    def _comprometer_fallos_pendientes(self, fallos_pendientes: list[str], ahora: float) -> None:
        """Commitea hacia `_registrar_fallo` cada clave acumulada en
        `fallos_pendientes` (mutandola en el llamador: la vacia al
        terminar) -- el otro extremo del mecanismo de atribucion a nivel de
        cadena, ver TOPE_FALLOS_SEGUIDOS. Un no-op sobre una lista vacia
        (el caso comun: la enorme mayoria de las cadenas o tienen exito
        directo o son de una sola ruta, donde el llamador commitea distinto)."""
        for clave in fallos_pendientes:
            self._registrar_fallo(clave, ahora)
        fallos_pendientes.clear()

    @staticmethod
    def _limpiar(datos: dict, desenvolver_canvas: bool) -> str:
        razon_total = ""
        for eleccion in datos.get("choices", []):
            msg = eleccion.get("message") or {}
            contenido = msg.get("content")
            if isinstance(contenido, str):
                limpio, razon = recortar(contenido, desenvolver_canvas)
                msg["content"] = limpio
                razon_total += razon
        return razon_total
