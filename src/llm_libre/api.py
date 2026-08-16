import difflib
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llm_libre.auth import LimitadorPorLlave
from llm_libre.modelos import METRICAS_NEUTRAS, Pedido
from llm_libre.ranking import puntuar
from llm_libre.router import clave_de_orden, compatibles, ordenar

PERFILES = {"rapido", "balanceado", "potente"}
ALIAS = ["auto", "auto:rapido", "auto:potente", "auto:tools", "auto:vision"]

# Piso de confiabilidad reciente para que /health considere una ruta "viva".
# Una ruta sin ninguna telemetria todavia carga la confiabilidad NEUTRA
# (ver CONFIABILIDAD_NEUTRA en modelos.py), que queda por encima de este piso
# a proposito: una ruta recien vista no debe leerse como rota.
UMBRAL_CONFIABILIDAD_SALUD = 0.5


def interpretar_pedido(cuerpo: dict) -> Pedido:
    # Un "model" de solo espacios es, a todo efecto practico, ausente: no debe
    # colarse como si fuera un id explicito (quedaria vacio tras el strip y
    # produciria un 404 confuso sobre el modelo '').
    modelo_pedido = (cuerpo.get("model") or "").strip() or "auto"
    modelo, perfil = None, "balanceado"
    requiere_tools = bool(cuerpo.get("tools"))
    requiere_vision = False

    if modelo_pedido == "auto" or modelo_pedido.startswith("auto:"):
        sufijo = modelo_pedido[5:] if ":" in modelo_pedido else ""
        if sufijo in PERFILES:
            perfil = sufijo
        elif sufijo == "tools":
            requiere_tools = True
        elif sufijo == "vision":
            requiere_vision = True
    else:
        modelo = modelo_pedido

    exigidas = set(cuerpo.get("x_requiere") or [])
    return Pedido(
        modelo=modelo,
        requiere_tools=requiere_tools or "tools" in exigidas,
        requiere_vision=requiere_vision or "vision" in exigidas,
        min_contexto=int(cuerpo.get("x_min_contexto") or 0),
        perfil=perfil,
        permitir_pago=bool(cuerpo.get("x_permitir_pago", True)),
    )


@dataclass
class Estado:
    almacen: object
    proxy: object
    llaves: set
    tope_pago_diario: int
    limitador: LimitadorPorLlave = field(default_factory=lambda: LimitadorPorLlave(60))
    proveedores: list = field(default_factory=list)   # lo usa el planificador
    http: object = None                                # cliente httpx compartido


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
    app = FastAPI(title="llm-libre")

    def exigir_llave(x_api_key: str | None, authorization: str | None = None) -> str:
        llave = _resolver_llave(x_api_key, authorization)
        if not llave or llave not in estado.llaves:
            raise HTTPException(401, "llave invalida")
        if not estado.limitador.permitir(llave, time.time()):
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
        rutas = ordenar(activas, metricas, pedido, ahora)
        if not rutas:
            _sin_rutas(activas, pedido, metricas, ahora, tope_alcanzado)   # siempre levanta
        return rutas, pedido

    @app.post("/v1/chat/completions")
    async def completions(request: Request, x_api_key: str | None = Header(None),
                          authorization: str | None = Header(None)):
        llave = exigir_llave(x_api_key, authorization)
        cuerpo = await request.json()
        rutas, pedido = _rutas_para(cuerpo, llave)
        ahora = time.time()
        crudo = bool(cuerpo.get("x_crudo"))
        if cuerpo.get("stream"):
            def _contar_si_sirvio_de_pago(ruta) -> None:
                # El proxy llama esto COMO MUCHO una vez, y solo cuando esta
                # ruta de verdad sirvio la respuesta (ver el docstring de
                # `completar_stream`): asi el tope diario de pago tambien ata
                # en la rama de streaming, no solo en la sincronica de abajo.
                # Nunca se llama para una ruta meramente ofrecida, ni para una
                # que gano el failover sin llegar a responder, ni una vez por
                # chunk.
                if ruta.tier == "pago":
                    estado.almacen.sumar_uso_pago(
                        llave, datetime.now(timezone.utc).strftime("%Y-%m-%d"))
            return StreamingResponse(
                estado.proxy.completar_stream(
                    rutas, cuerpo, ahora, crudo,
                    en_ruta_comprometida=_contar_si_sirvio_de_pago),
                media_type="text/event-stream")
        r = await estado.proxy.completar(rutas, cuerpo, ahora, crudo)
        if r.ruta is not None and r.ruta.tier == "pago":
            estado.almacen.sumar_uso_pago(
                llave, datetime.now(timezone.utc).strftime("%Y-%m-%d"))
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

    @app.get("/v1/models")
    def modelos(x_api_key: str | None = Header(None), authorization: str | None = Header(None)):
        exigir_llave(x_api_key, authorization)
        datos = [{"id": r.modelo_id, "object": "model", "owned_by": r.proveedor}
                 for r in estado.almacen.rutas_activas()]
        datos += [{"id": a, "object": "model", "owned_by": "llm-libre"} for a in ALIAS]
        return {"object": "list", "data": datos}

    @app.get("/v1/ranking")
    def ranking(x_api_key: str | None = Header(None), authorization: str | None = Header(None)):
        exigir_llave(x_api_key, authorization)
        ahora = time.time()
        metricas = _metricas(estado, ahora)
        # Ordenado con la MISMA clave que usa router.ordenar (perfil
        # "balanceado", el que tambien usa el puntaje de cada fila) -- no un
        # sort propio por puntaje: este endpoint es para auditar POR QUE el
        # router eligio lo que eligio (README), y antes podia mostrar una
        # ruta arriba de todo mientras X-Ruta-Usada decia otra distinta,
        # porque no miraba `prioridad` ni el cooldown -- una ruta castigada
        # (que el router jamas elegiria ahora mismo) podia encabezar la
        # tabla. `en_cooldown_hasta` sigue expuesto por fila para
        # diagnostico; lo que cambia es el ORDEN.
        activas = sorted(estado.almacen.rutas_activas(),
                         key=lambda r: clave_de_orden(r, metricas[r.clave], "balanceado", ahora))
        filas = []
        for r in activas:
            m = metricas[r.clave]
            medida = m.calidad_medida_en is not None
            filas.append({"clave": r.clave, "tier": r.tier, "prioridad": r.prioridad,
                          "puntaje": round(puntuar(m, "balanceado"), 4),
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

    @app.get("/v1/uso")
    def uso(x_api_key: str | None = Header(None), authorization: str | None = Header(None)):
        llave = exigir_llave(x_api_key, authorization)
        dia = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return {"dia": dia, "pago_hoy": estado.almacen.uso_pago(llave, dia),
                "tope": estado.tope_pago_diario}

    @app.get("/health")
    def health():
        # Honesto: mira si hay una ruta VIVA y servible, no si el proceso esta
        # arriba. "Viva" exige DOS cosas, no una: no estar en cooldown (lo
        # dispara un 429 de inmediato, o TOPE_FALLOS_SEGUIDOS fallos NO-429
        # seguidos -- ver Proxy._castigar/_registrar_fallo; un 4xx del
        # cliente, en cambio, NUNCA cuenta hacia esto, ver
        # _es_error_del_cliente) Y que su confiabilidad reciente -- calculada
        # sobre trafico real de eventos ok/fail, no sobre el cooldown -- no
        # este por el piso. Antes del cooldown de fallos duros (revision de
        # Task 13), una ruta que devolvia 500 en cada intento nunca entraba
        # en cooldown pero tampoco estaba viva; si solo mirara cooldowns este
        # endpoint diria "ok" con esa ruta muerta, que es exactamente el
        # incidente que este endpoint existe para evitar. Hoy esa misma ruta
        # SI termina en cooldown a partir del tercer 500 seguido -- pero la
        # confiabilidad sigue siendo la segunda pata, para cubrir la ventana
        # antes de que el cooldown se dispare.
        ahora = time.time()
        activas = estado.almacen.rutas_activas()
        metricas = _metricas(estado, ahora)

        def _viva(r) -> bool:
            m = metricas.get(r.clave, METRICAS_NEUTRAS)
            return (m.en_cooldown_hasta <= ahora
                    and m.confiabilidad >= UMBRAL_CONFIABILIDAD_SALUD)

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
    compat = compatibles(activas, pedido)
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
    return difflib.get_close_matches(pedido, [r.modelo_id for r in activas], n=3, cutoff=0.3)


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
