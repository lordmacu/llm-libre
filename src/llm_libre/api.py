import difflib
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llm_libre.auth import LimitadorPorLlave
from llm_libre.modelos import Pedido
from llm_libre.ranking import puntuar
from llm_libre.router import ordenar

PERFILES = {"rapido", "balanceado", "potente"}
ALIAS = ["auto", "auto:rapido", "auto:potente", "auto:tools", "auto:vision"]


def interpretar_pedido(cuerpo: dict) -> Pedido:
    modelo_pedido = (cuerpo.get("model") or "auto").strip()
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


def crear_app(estado: Estado) -> FastAPI:
    app = FastAPI(title="llm-libre")

    def exigir_llave(llave: str | None) -> str:
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
        if pedido.permitir_pago:
            dia = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if estado.almacen.uso_pago(llave, dia) >= estado.tope_pago_diario:
                pedido = replace(pedido, permitir_pago=False)
        ahora = time.time()
        rutas = ordenar(activas, _metricas(estado, ahora), pedido, ahora)
        return rutas, pedido

    @app.post("/v1/chat/completions")
    async def completions(request: Request, x_api_key: str | None = Header(None)):
        llave = exigir_llave(x_api_key)
        cuerpo = await request.json()
        rutas, pedido = _rutas_para(cuerpo, llave)
        if not rutas:
            raise HTTPException(400, {
                "message": "ninguna ruta cumple lo pedido",
                "pedido": pedido.__dict__,
                "rutas_activas": len(estado.almacen.rutas_activas()),
            })
        ahora = time.time()
        crudo = bool(cuerpo.get("x_crudo"))
        if cuerpo.get("stream"):
            return StreamingResponse(
                estado.proxy.completar_stream(rutas, cuerpo, ahora, crudo),
                media_type="text/event-stream")
        r = await estado.proxy.completar(rutas, cuerpo, ahora, crudo)
        if r.ruta is not None and r.ruta.tier == "pago":
            estado.almacen.sumar_uso_pago(
                llave, datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        cabeceras = {"X-Intentos": str(r.intentos)}
        if r.ruta is not None:
            cabeceras["X-Ruta-Usada"] = r.ruta.clave
            cabeceras["X-Tier"] = r.ruta.tier
        return JSONResponse(r.json, status_code=r.estado, headers=cabeceras)

    @app.get("/v1/models")
    def modelos(x_api_key: str | None = Header(None)):
        exigir_llave(x_api_key)
        datos = [{"id": r.modelo_id, "object": "model", "owned_by": r.proveedor}
                 for r in estado.almacen.rutas_activas()]
        datos += [{"id": a, "object": "model", "owned_by": "llm-libre"} for a in ALIAS]
        return {"object": "list", "data": datos}

    @app.get("/v1/ranking")
    def ranking(x_api_key: str | None = Header(None)):
        exigir_llave(x_api_key)
        ahora = time.time()
        metricas = _metricas(estado, ahora)
        filas = []
        for r in estado.almacen.rutas_activas():
            m = metricas[r.clave]
            filas.append({"clave": r.clave, "tier": r.tier,
                          "puntaje": round(puntuar(m, "balanceado"), 4),
                          "calidad": round(m.calidad, 3),
                          "confiabilidad": round(m.confiabilidad, 3),
                          "ttft_p50_ms": m.ttft_p50_ms,
                          "en_cooldown_hasta": m.en_cooldown_hasta,
                          "tools": r.capacidades.tools, "vision": r.capacidades.vision,
                          "contexto": r.capacidades.contexto})
        filas.sort(key=lambda f: -f["puntaje"])
        return {"rutas": filas}

    @app.get("/v1/uso")
    def uso(x_api_key: str | None = Header(None)):
        llave = exigir_llave(x_api_key)
        dia = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return {"dia": dia, "pago_hoy": estado.almacen.uso_pago(llave, dia),
                "tope": estado.tope_pago_diario}

    @app.get("/health")
    def health():
        # Honesto: mira si hay una ruta VIVA y servible, no si el proceso esta arriba.
        ahora = time.time()
        activas = estado.almacen.rutas_activas()
        libres = [r for r in activas
                  if estado.proxy.cooldowns.get(r.clave, 0.0) <= ahora]
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


def _parecidos(pedido: str, activas: list) -> list[str]:
    return difflib.get_close_matches(pedido, [r.modelo_id for r in activas], n=3, cutoff=0.3)


def _metricas(estado: Estado, ahora: float) -> dict:
    base = estado.almacen.metricas()
    for clave, hasta in estado.proxy.cooldowns.items():
        if clave in base:
            m = base[clave]
            base[clave] = type(m)(m.calidad, m.confiabilidad, m.ttft_p50_ms, hasta)
    return base
