import difflib
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llm_libre.auth import PerKeyRateLimiter
from llm_libre.modelos import METRICAS_NEUTRAS, Pedido
from llm_libre.openapi import (CHAT_COMPLETIONS_DOCS, DESCRIPCION, HEALTH_DOCS,
                               MODELOS_DOCS, RANKING_DOCS, RESUMEN, TITULO, USO_DOCS,
                               VERSION, personalizar_openapi)
from llm_libre.ranking import score
from llm_libre.router import compatible_routes, order_routes, sort_key

PERFILES = {"rapido", "balanceado", "potente"}
ALIAS = ["auto", "auto:rapido", "auto:potente", "auto:tools", "auto:vision"]


def _leer_campo(campo: str, valor_bruto, mensaje: str, calcular):
    """Corre `calcular` (una funcion de cero argumentos que interpreta
    `valor_bruto` -- un valor que vino TAL CUAL del cuerpo JSON del
    cliente, nunca algo que el gateway arma) y convierte cualquier
    `TypeError`/`ValueError`/`AttributeError` -- la familia de excepciones
    que dispara un campo con el TIPO equivocado (`.strip()` sobre un
    numero, `int()` sobre una lista, `set()` sobre un booleano o sobre una
    lista que contiene una lista) -- en un 400 uniforme, en vez de
    dejarla escapar sin atrapar hasta el manejador generico de FastAPI
    como un 500 opaco.

    Revision post-Task-14 (segundo gate): el mismo bug penetro dos veces
    seguidas -- primero en `x_min_contexto` (arreglado a mano con un
    try/except puntual), despues en `x_requiere` (el mismo patron,
    reinventado) -- y una tercera instancia (`model`, via `.strip()`)
    seguia sin atrapar cuando se encontraron las otras dos. El eje real
    nunca fue "este campo puntual", fue "cualquier campo que este
    endpoint interpreta antes de usarlo llega crudo del cliente, sin
    tipo garantizado" -- ver el docstring de `build_request`, el mismo
    principio de "esto es passthrough, no confiar en la forma". Esta
    funcion es el punto UNICO por el que pasa esa interpretacion de aca
    en mas, para que un cuarto campo (si este endpoint gana uno) no
    tenga que reinventar el try/except ni arriesgarse a olvidarlo."""
    try:
        return calcular()
    except (TypeError, ValueError, AttributeError):
        raise HTTPException(400, {
            "message": mensaje, "campo": campo, "valor_recibido": valor_bruto})


def _hay_imagen(cuerpo: dict) -> bool:
    """True si algun mensaje trae una imagen.

    En el formato OpenAI, `content` es un string (solo texto) o una LISTA de
    partes, y una imagen es una parte con `type: "image_url"` (formato
    clasico) o `"input_image"` (el de la Responses API, que varios clientes
    ya mandan). Se aceptan los dos: el gateway existe para que cambiar
    `base_url` alcance, y rechazar el formato nuevo obligaria al cliente a
    saber contra que proxy esta hablando.

    Sin esto, `requiere_vision` solo se activaba con el alias `auto:vision` o
    con `x_requiere` -- o sea, solo si el cliente AVISABA. Un cliente que
    simplemente manda la imagen, que es lo normal contra una API OpenAI,
    podia terminar en una ruta de solo texto: 200 con una respuesta que
    ignora la imagen, o un 400 del proveedor. Ninguno de los dos dice "esa
    ruta no ve imagenes".

    Ante un cuerpo malformado devuelve False y deja que el proveedor rechace:
    esta funcion elige ruta, no valida el pedido.
    """
    mensajes = cuerpo.get("messages")
    if not isinstance(mensajes, list):
        return False
    for m in mensajes:
        if not isinstance(m, dict):
            continue
        contenido = m.get("content")
        if not isinstance(contenido, list):
            continue
        for parte in contenido:
            if isinstance(parte, dict) and parte.get("type") in ("image_url", "input_image"):
                return True
    return False


def interpretar_pedido(cuerpo: dict) -> Pedido:
    # Un "model" de solo espacios es, a todo efecto practico, ausente: no debe
    # colarse como si fuera un id explicito (quedaria vacio tras el strip y
    # produciria un 404 confuso sobre el modelo '').
    modelo_bruto = cuerpo.get("model")
    modelo_pedido = _leer_campo(
        "model", modelo_bruto, "model debe ser un string",
        lambda: (modelo_bruto or "").strip()) or "auto"
    modelo, perfil = None, "balanceado"
    requiere_tools = bool(cuerpo.get("tools"))
    requiere_vision = _hay_imagen(cuerpo)

    if modelo_pedido == "auto" or modelo_pedido.startswith("auto:"):
        sufijo = modelo_pedido[5:] if ":" in modelo_pedido else ""
        if sufijo in PERFILES:
            perfil = sufijo
        elif sufijo == "tools":
            requiere_tools = True
        elif sufijo == "vision":
            requiere_vision = True
        elif sufijo:
            # Revision post-Task-14 (gate): un sufijo "auto:<algo>" que no es
            # ni un perfil conocido ni "tools"/"vision" (p.ej. "auto:turbo",
            # un typo de "auto:tools") caia por las tres ramas de arriba SIN
            # tocar nada -- silenciosamente identico a pedir "auto" liso, sin
            # ningun aviso de que el sufijo se ignoro. Para un cliente que
            # de verdad queria exigir una capacidad (p.ej. tools, para un
            # agente que espera una tool_call) eso es peligroso en silencio:
            # recibe una respuesta "balanceada" comun, no el 400 que le
            # habria dicho que escribio mal el alias. "auto" sin ":" (sufijo
            # == "") sigue siendo valido y NO entra aca -- ver PERFILES,
            # que ya incluye "balanceado" (asi que "auto:balanceado"
            # tambien resuelve normal, sin pasar por esta rama).
            raise HTTPException(400, {
                "message": f"alias de modelo desconocido: '{modelo_pedido}'",
                "sugerencias": ALIAS,
            })
    else:
        modelo = modelo_pedido

    def _normalizar_exigidas():
        # `x_requiere: "tools"` (un string suelto, en vez de la lista
        # documentada) se acepta como un valor unico -- una API REST
        # comun. Cualquier otra cosa que no sea un string ni una lista (o
        # una lista con un elemento no-hasheable, como `[["tools"]]`)
        # revienta ADENTRO de este `set(...)` con TypeError -- eso es lo
        # que `_leer_campo` atrapa y convierte en 400. `set("tools")`
        # (sin el envoltorio de arriba) iteraria caracter por caracter
        # -- {'t','o','l','s'} -- y la exigencia se ignoraria en
        # silencio; por eso el string se envuelve en una lista ANTES del
        # set(), no despues.
        valor = cuerpo.get("x_requiere") or []
        if isinstance(valor, str):
            valor = [valor]
        return set(valor)
    exigidas = _leer_campo(
        "x_requiere", cuerpo.get("x_requiere"),
        "x_requiere debe ser un string o una lista de strings",
        _normalizar_exigidas)

    x_min_contexto_bruto = cuerpo.get("x_min_contexto")
    min_contexto = _leer_campo(
        "x_min_contexto", x_min_contexto_bruto,
        "x_min_contexto debe ser un numero entero",
        lambda: int(x_min_contexto_bruto) if x_min_contexto_bruto else 0)

    return Pedido(
        modelo=modelo,
        requiere_tools=requiere_tools or "tools" in exigidas,
        requiere_vision=requiere_vision or "vision" in exigidas,
        min_contexto=min_contexto,
        perfil=perfil,
        permitir_pago=bool(cuerpo.get("x_permitir_pago", True)),
    )


@dataclass
class Estado:
    almacen: object
    proxy: object
    llaves: set
    tope_pago_diario: int
    limitador: PerKeyRateLimiter = field(default_factory=lambda: PerKeyRateLimiter(60))
    proveedores: list = field(default_factory=list)   # lo usa el planificador
    http: object = None                                # cliente httpx compartido
    # Generador para el sorteo entre rutas empatadas (ver router.shuffle_ties).
    # None = sin sorteo, orden estrictamente determinista. Se inyecta desde
    # principal.crear_estado() segun ROTAR_EMPATES para que los tests puedan
    # armar un Estado determinista sin pasar por variables de entorno.
    aleatorio: object = None


def _resolver_llave(x_api_key: str | None, authorization: str | None) -> str | None:
    """Acepta la llave por `X-API-Key` (convencion que ya usa `arkiv-api`,
    el gateway hermano) o por `Authorization: Bearer <llave>` (lo que manda
    sin configuracion extra CUALQUIER SDK de OpenAI via su parametro
    `api_key` -- que es, literalmente, la promesa central de este contrato:
    "cambia solo base_url"). Si llegan las dos, `X-API-Key` gana: es la
    convencion mas explicita y la que ya usan los llamadores existentes.

    Un `Authorization` que no trae el prefijo `Bearer ` (u otra forma
    malformada) no resuelve ninguna llave -- ni revienta ni intenta
    adivinar, simplemente cae al mismo 401 que una llave ausente.
    """
    if x_api_key:
        return x_api_key
    if not authorization:
        return None
    partes = authorization.split(None, 1)
    if len(partes) != 2 or partes[0].lower() != "bearer":
        return None
    return partes[1].strip() or None


def crear_app(estado: Estado) -> FastAPI:
    # title/version/summary/description y personalizar_openapi (Task 14) solo
    # enriquecen lo que sirve /docs y /openapi.json -- ver llm_libre.openapi.
    # No tocan ninguna ruta ni su logica: exigir_llave, interpretar_pedido y
    # el passthrough de completions siguen exactamente igual.
    app = FastAPI(title=TITULO, version=VERSION, summary=RESUMEN, description=DESCRIPCION)
    personalizar_openapi(app)

    def exigir_llave(x_api_key: str | None, authorization: str | None = None) -> str:
        llave = _resolver_llave(x_api_key, authorization)
        if not llave or llave not in estado.llaves:
            raise HTTPException(401, "llave invalida")
        if not estado.limitador.allow(llave, time.time()):
            raise HTTPException(429, "demasiadas peticiones para esta llave")
        return llave

    def _rutas_para(cuerpo: dict, llave: str) -> tuple[list, object]:
        pedido = interpretar_pedido(cuerpo)
        activas = estado.almacen.rutas_activas()
        # Un id explicito que ya no existe merece un 404 con pistas, no un 400 generico:
        # es exactamente el fallo que este proyecto existe para evitar.
        if pedido.modelo is not None and not any(r.modelo_id == pedido.modelo for r in activas):
            raise HTTPException(404, {
                "message": f"el modelo '{pedido.modelo}' ya no existe",
                "sugerencias": _parecidos(pedido.modelo, activas),
            })
        tope_alcanzado = False
        if pedido.permitir_pago:
            dia = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if estado.almacen.uso_pago(llave, dia) >= estado.tope_pago_diario:
                pedido = replace(pedido, permitir_pago=False)
                tope_alcanzado = True
        ahora = time.time()
        metricas = _metricas(estado, ahora)
        rutas = order_routes(activas, metricas, pedido, ahora, estado.aleatorio)
        if not rutas:
            _sin_rutas(activas, pedido, metricas, ahora, tope_alcanzado)   # siempre levanta
        return rutas, pedido

    @app.post("/v1/chat/completions", **CHAT_COMPLETIONS_DOCS)
    async def completions(request: Request, x_api_key: str | None = Header(None),
                          authorization: str | None = Header(None)):
        llave = exigir_llave(x_api_key, authorization)
        cuerpo = await request.json()
        rutas, pedido = _rutas_para(cuerpo, llave)
        ahora = time.time()
        crudo = bool(cuerpo.get("x_crudo"))

        def _contar_uso_pago(ruta) -> None:
            # HIGH 4 (round 9): se llama por cada intento FACTURABLE contra
            # una ruta de pago -- el proveedor cobra 200 con contenido util,
            # 200 vacio, y una razonamiento que se gasto el presupuesto por
            # igual (genera tokens en los tres casos); solo un error de RED
            # o un status distinto de 200 no genera cobro. Antes esto solo
            # se llamaba en el EXITO (`r.ruta`/`en_ruta_comprometida`): un
            # 200-vacio de una ruta de pago se facturaba de verdad y no
            # aparecia ni en /v1/uso ni contra TOPE_PAGO_DIARIO -- medido,
            # 40/40 llamadas facturables con `pago_hoy: 0`. Ver proxy.py
            # (`en_intento_facturable`) para donde se decide "facturable".
            estado.almacen.sumar_uso_pago(
                llave, datetime.now(timezone.utc).strftime("%Y-%m-%d"))

        if cuerpo.get("stream"):
            return StreamingResponse(
                estado.proxy.completar_stream(
                    rutas, cuerpo, ahora, crudo,
                    en_intento_facturable=_contar_uso_pago),
                media_type="text/event-stream")
        r = await estado.proxy.completar(rutas, cuerpo, ahora, crudo,
                                         en_intento_facturable=_contar_uso_pago)
        if r.estado == 503 and r.codigo_upstream == 404 and pedido.modelo is not None:
            # ALSO de la revision round 6 -- literalmente la razon de ser
            # del proyecto: `pedido.modelo` SIGUE en nuestro catalogo (paso
            # el check 404 de `_rutas_para`, mas arriba) pero el proveedor
            # real ya no lo tiene: un 404 genuino, en vivo. La ruta ya se
            # llevo el golpe de confiabilidad (404 es evidencia de la ruta
            # por default, ver proxy._es_error_del_cliente), pero sin este
            # chequeo el cliente solo veia un 503 generico
            # ("detalle": "HTTP 404") -- indistinguible de cualquier otra
            # indisponibilidad transitoria, durante toda la ventana de hasta
            # 5h antes del proximo sync de catalogo (nunca para rutas de
            # pago, que no se sondean).
            #
            # Solo con un modelo EXPLICITO: en modo "auto" `pedido.modelo`
            # es None, no hay un id puntual sobre el cual sugerir, y la ruta
            # que fallo no es necesariamente la unica candidata razonable --
            # ese caso se queda con el 503 de siempre.
            #
            # Solo el camino sincronico: en streaming el status 200 y las
            # cabeceras SSE ya salieron antes de que el proxy sepa si la
            # ruta sirvio, asi que no hay margen HTTP para cambiarlo a 404.
            raise HTTPException(404, {
                "message": f"el modelo '{pedido.modelo}' ya no existe",
                "sugerencias": _parecidos(pedido.modelo, estado.almacen.rutas_activas()),
            })
        cabeceras = {"X-Intentos": str(r.intentos)}
        if r.ruta is not None:
            cabeceras["X-Ruta-Usada"] = r.ruta.clave
            cabeceras["X-Tier"] = r.ruta.tier
        cuerpo_resp = r.json
        if r.estado == 200 and r.razonamiento and isinstance(cuerpo_resp, dict):
            # §6.1: el razonamiento recortado se devuelve en un campo aparte,
            # "para quien lo quiera". Antes se recortaba de `content` y se
            # tiraba, asi que con el default `x_crudo: false` no habia forma de
            # recuperarlo. Va al nivel superior (no dentro de `choices`) porque
            # ahi cualquier SDK de OpenAI lo ignora sin romperse, que es la
            # unica condicion que el contrato pone a las extensiones.
            #
            # LIMITACION CONOCIDA, deliberada: en streaming NO se devuelve.
            # Meterlo ahi obliga a emitir un evento SSE no estandar, que es
            # justo lo que el §6 descarta por arriesgar el parseo de los SDK
            # que este contrato existe para complacer. Un cliente que streamea
            # y quiere el razonamiento pide `x_crudo: true` y lo recibe dentro
            # del `content`, tal cual lo mando el proveedor.
            cuerpo_resp = {**cuerpo_resp, "x_razonamiento": r.razonamiento}
        return JSONResponse(cuerpo_resp, status_code=r.estado, headers=cabeceras)

    @app.get("/v1/models", **MODELOS_DOCS)
    def modelos(x_api_key: str | None = Header(None), authorization: str | None = Header(None)):
        exigir_llave(x_api_key, authorization)
        datos = [{"id": r.modelo_id, "object": "model", "owned_by": r.proveedor}
                 for r in estado.almacen.rutas_activas()]
        datos += [{"id": a, "object": "model", "owned_by": "llm-libre"} for a in ALIAS]
        return {"object": "list", "data": datos}

    @app.get("/v1/ranking", **RANKING_DOCS)
    def ranking(x_api_key: str | None = Header(None), authorization: str | None = Header(None)):
        exigir_llave(x_api_key, authorization)
        ahora = time.time()
        metricas = _metricas(estado, ahora)
        # Ordenado con la MISMA clave que usa router.order_routes (perfil
        # "balanceado", el que tambien usa el puntaje de cada fila) -- no un
        # sort propio por puntaje: este endpoint es para auditar POR QUE el
        # router eligio lo que eligio (README), y antes podia mostrar una
        # ruta arriba de todo mientras X-Ruta-Usada decia otra distinta,
        # porque no miraba `prioridad` ni el cooldown -- una ruta castigada
        # (que el router jamas elegiria ahora mismo) podia encabezar la
        # tabla. `en_cooldown_hasta` sigue expuesto por fila para
        # diagnostico; lo que cambia es el ORDEN.
        activas = sorted(estado.almacen.rutas_activas(),
                         key=lambda r: sort_key(r, metricas[r.clave], "balanceado", ahora))
        filas = []
        for r in activas:
            m = metricas[r.clave]
            medida = m.calidad_medida_en is not None
            filas.append({"clave": r.clave, "tier": r.tier, "prioridad": r.prioridad,
                          "puntaje": round(score(m, "balanceado"), 4),
                          # "nunca medida" se dice, no se disfraza: mostrar el
                          # neutro en `calidad` como si alguien lo hubiera
                          # medido es lo que hacia invisible que `auto` estaba
                          # ordenando por un supuesto. El valor que SI entro al
                          # puntaje va aparte, en `calidad_asumida`.
                          "calidad": round(m.calidad, 3) if medida else None,
                          "calidad_medida": medida,
                          "calidad_asumida": None if medida else round(m.calidad, 3),
                          "ultima_sonda_calidad": _iso(m.calidad_medida_en),
                          "ultima_sonda": _iso(m.ultima_sonda_en),
                          "confiabilidad": round(m.confiabilidad, 3),
                          # Dos numeros distintos a proposito: ttft_p50_ms es
                          # tiempo al primer token (solo lo mide el streaming, y
                          # es lo que pesa en el puntaje); latencia_p50_ms es el
                          # round-trip completo (no-streaming y sondas). Antes
                          # compartian columna y el promedio no significaba nada.
                          "ttft_p50_ms": m.ttft_p50_ms,
                          "latencia_p50_ms": m.latencia_p50_ms,
                          "en_cooldown_hasta": m.en_cooldown_hasta,
                          "tools": r.capacidades.tools, "vision": r.capacidades.vision,
                          "contexto": r.capacidades.contexto})
        return {"rutas": filas}

    @app.get("/v1/uso", **USO_DOCS)
    def uso(x_api_key: str | None = Header(None), authorization: str | None = Header(None)):
        llave = exigir_llave(x_api_key, authorization)
        dia = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return {"dia": dia, "pago_hoy": estado.almacen.uso_pago(llave, dia),
                "tope": estado.tope_pago_diario}

    @app.get("/health", **HEALTH_DOCS)
    def health():
        # Honesto: mira si hay una ruta VIVA y servible, no si el proceso esta
        # arriba. "Viva" exige DOS cosas, no una: no estar en cooldown (round
        # 8: SOLO lo dispara un 429 de inmediato, o una SONDA -- periodica o
        # bajo demanda -- que confirma que la ruta esta rota; el trafico de
        # un cliente real nunca excluye una ruta directo, ver el comentario
        # de cabecera de UMBRAL_SOSPECHA en proxy.py) Y evidencia POSITIVA de
        # que sirve (`Almacen.tiene_evidencia_de_vida`).
        #
        # Task 13, revision round 6, Parte 2: ESTO YA NO MIRA `confiabilidad`.
        # `confiabilidad` es un promedio de trafico reciente, y un promedio se
        # arrastra a 0 con cualquier patron repetido de UN cliente -- el caso
        # que lo probo es `403`, genuinamente ambiguo (cuenta suspendida =
        # evidencia de la ruta, vs. contenido moderado = evidencia del
        # PEDIDO) y que el gateway no puede desambiguar sin parsear el cuerpo
        # especifico de cada proveedor. 30 pedidos con contenido moderado de
        # una sola llave alcanzaban para tirar `/health` a "caido" para TODAS
        # las llaves, sobreviviendo un reinicio del proceso contra la misma
        # base -- porque Coolify usa este endpoint como health check y
        # reinicia el contenedor cuando falla.
        #
        # "Evidencia de vida, no ausencia de muerte": un exito reciente
        # prueba que la ruta sirve; mil fallos de un mismo cliente no prueban
        # que no sirve. Los fallos, solos, NUNCA bastan para declarar una
        # ruta muerta aca -- ver el docstring de `tiene_evidencia_de_vida`
        # para el criterio completo (exito real reciente, o sonda de salud
        # reciente exitosa, o ninguna telemetria todavia).
        #
        # `/v1/ranking` (mas abajo) sigue usando `confiabilidad` exactamente
        # como antes -- eso NO cambia. La asimetria es a proposito: una ruta
        # mal puntuada en el ranking solo pierde posicion y se autocorrige
        # sola; una ruta que `/health` declara muerta reinicia el
        # contenedor. El ranking puede darse el lujo de ser sensible: la
        # salud no.
        ahora = time.time()
        activas = estado.almacen.rutas_activas()
        metricas = _metricas(estado, ahora)

        def _viva(r) -> bool:
            m = metricas.get(r.clave, METRICAS_NEUTRAS)
            return (m.en_cooldown_hasta <= ahora
                    and estado.almacen.tiene_evidencia_de_vida(r.clave, ahora))

        libres = [r for r in activas if _viva(r)]
        gratis = [r for r in libres if r.tier == "gratis"]
        if gratis:
            situacion = "ok"
        elif libres:
            situacion = "degradado"
        else:
            situacion = "caido"
        codigo = 200 if situacion == "ok" else 503
        return JSONResponse({"estado": situacion, "rutas_activas": len(activas),
                             "rutas_libres": len(libres),
                             "gratis_libres": len(gratis)}, status_code=codigo)

    return app


def _sin_rutas(activas: list, pedido, metricas: dict, ahora: float,
               tope_alcanzado: bool) -> None:
    """Levanta el error correcto cuando la cadena de intentos sale vacia.

    El §9 del diseno separa dos situaciones que la version anterior mezclaba en
    un solo 400:

    - **Ninguna ruta puede cumplir lo pedido** (capacidades, vision, contexto
      que nadie tiene) -> `400`. Eso si es un error del cliente.
    - **Hay rutas que podrian servir pero estan todas caidas o en cooldown**
      (incluido el caso "la llave supero su tope de pago diario") -> `503`,
      con `proxima_liberacion`.

    Por que importa: `ordenar` filtra los cooldowns, asi que en CUALQUIER
    apagon de los tiers gratis -- el fallo esperado, no uno raro -- la lista
    llegaba vacia y salia un 400. Todo SDK y toda capa de alertas leen 400 como
    "tu peticion esta mal formada": no reintentan y no despiertan a nadie.

    `x_permitir_pago: false` NO se considera aca: es una politica del que
    llama, no una capacidad que falte en el pozo. Un cliente que prohibe el
    pago y se queda sin rutas gratis vivas esta en el caso de
    indisponibilidad (503, reintentable), no en el de peticion invalida.
    """
    compat = compatible_routes(activas, pedido)
    if not compat:
        raise HTTPException(400, {
            "message": "ninguna ruta cumple lo pedido",
            "pedido": pedido.__dict__,
            "rutas_activas": len(activas),
        })
    liberaciones = [metricas[r.clave].en_cooldown_hasta for r in compat
                    if r.clave in metricas and metricas[r.clave].en_cooldown_hasta > ahora]
    raise HTTPException(503, {
        "message": "todas las rutas que podrian servir estan caidas o en cooldown",
        "pedido": pedido.__dict__,
        "rutas_compatibles": len(compat),
        # Cuando se libera la PRIMERA de ellas. None = ninguna esta en castigo
        # (estan descartadas por otra razon, p.ej. el tope de pago).
        "proxima_liberacion": min(liberaciones) if liberaciones else None,
        "tope_pago_alcanzado": tope_alcanzado,
    })


def _parecidos(pedido: str, activas: list) -> list[str]:
    # Round 7, LOW del gate: el llamador del 404-en-vivo (mas arriba) pasa
    # `activas` SIN filtrar -- ese id de `pedido` TODAVIA esta en el
    # catalogo local (esa es la premisa entera de ese caso: sigue en el
    # catalogo, pero el proveedor real ya no lo tiene). Sin excluirlo aca,
    # `get_close_matches` lo encuentra a SI MISMO como el "parecido" mas
    # obvio (distancia cero) y el cliente lee `"el modelo 'a:free' ya no
    # existe"` con `sugerencias: ['a:free', ...]`. Se excluye ACA, en la
    # funcion, y no en cada llamador: una lista de sugerencias nunca debe
    # poder sugerir el mismo id que se acaba de declarar muerto.
    candidatos = [r.modelo_id for r in activas if r.modelo_id != pedido]
    return difflib.get_close_matches(pedido, candidatos, n=3, cutoff=0.3)


def _iso(momento: float | None) -> str | None:
    if momento is None:
        return None
    return datetime.fromtimestamp(momento, timezone.utc).isoformat().replace("+00:00", "Z")


def _metricas(estado: Estado, ahora: float) -> dict:
    base = estado.almacen.metricas()
    for clave, hasta in estado.proxy.cooldowns.items():
        if clave in base:
            # `replace` y no `type(m)(...)` posicional: reconstruir a mano deja
            # afuera cualquier campo nuevo de Metricas (p.ej. calidad_medida_en),
            # y una ruta en cooldown pasaria a parecer "nunca medida".
            base[clave] = replace(base[clave], en_cooldown_hasta=hasta)
    return base
