import difflib
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llm_libre.auth import PerKeyRateLimiter
from llm_libre.models import NEUTRAL_METRICS, RouteRequest
from llm_libre.openapi import (CHAT_COMPLETIONS_DOCS, DESCRIPTION, HEALTH_DOCS,
                               MODELS_DOCS, RANKING_DOCS, SUMMARY, TITLE, USAGE_DOCS,
                               VERSION, customise_openapi)
from llm_libre.ranking import score
from llm_libre.router import compatible_routes, order_routes, sort_key

PROFILES = {"rapido", "balanceado", "potente"}
ALIASES = ["auto", "auto:rapido", "auto:potente", "auto:tools", "auto:vision"]


def _read_field(campo: str, valor_bruto, mensaje: str, calcular):
    """Run `calcular` (a zero-argument function interpreting `valor_bruto` -- a
    value that arrived VERBATIM from the client's JSON body, never something the
    gateway builds) and convert any `TypeError`/`ValueError`/`AttributeError` --
    the family of exceptions a field of the WRONG TYPE raises (`.strip()` on a
    number, `int()` on a list, `set()` on a boolean or on a list containing a
    list) -- into a uniform 400, instead of letting it escape uncaught to
    FastAPI's generic handler as an opaque 500.

    Post-Task-14 review (second gate): the same bug got through twice in a row --
    first in `x_min_contexto` (fixed by hand with a local try/except), then in
    `x_requiere` (the same pattern, reinvented) -- and a third instance (`model`,
    via `.strip()`) was still uncaught when the other two were found. The real
    axis was never "this particular field", it was "every field this endpoint
    interprets before using it arrives raw from the client, with no guaranteed
    type" -- see the docstring of `build_request`, the same "this is passthrough,
    do not trust the shape" principle. This function is the SINGLE point that
    interpretation goes through from now on, so a fourth field (if this endpoint
    gains one) does not have to reinvent the try/except or risk forgetting it."""
    try:
        return calcular()
    except (TypeError, ValueError, AttributeError):
        raise HTTPException(400, {
            "message": mensaje, "campo": campo, "valor_recibido": valor_bruto})


def _has_image(cuerpo: dict) -> bool:
    """True if any message carries an image.

    In the OpenAI format, `content` is either a string (text only) or a LIST of
    parts, and an image is a part with `type: "image_url"` (the classic format)
    or `"input_image"` (the Responses API one, which several clients already
    send). Both are accepted: the gateway exists so that changing `base_url` is
    enough, and rejecting the newer format would force the client to know which
    proxy it is talking to.

    Without this, `needs_vision` was only set by the `auto:vision` alias or by
    `x_requiere` -- that is, only if the client ANNOUNCED it. A client that
    simply sends the image, which is the normal thing against an OpenAI API,
    could end up on a text-only route: a 200 with a response that ignores the
    image, or a 400 from the provider. Neither of those says "that route cannot
    see images".

    On a malformed body it returns False and lets the provider reject: this
    function picks a route, it does not validate the request.
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


def parse_request(cuerpo: dict) -> RouteRequest:
    # A "model" of pure whitespace is, for all practical purposes, absent: it
    # must not sneak through as if it were an explicit id (it would be empty
    # after the strip and would produce a confusing 404 about the model '').
    modelo_bruto = cuerpo.get("model")
    modelo_pedido = _read_field(
        "model", modelo_bruto, "model debe ser un string",
        lambda: (modelo_bruto or "").strip()) or "auto"
    modelo, perfil = None, "balanceado"
    requiere_tools = bool(cuerpo.get("tools"))
    requiere_vision = _has_image(cuerpo)

    if modelo_pedido == "auto" or modelo_pedido.startswith("auto:"):
        sufijo = modelo_pedido[5:] if ":" in modelo_pedido else ""
        if sufijo in PROFILES:
            perfil = sufijo
        elif sufijo == "tools":
            requiere_tools = True
        elif sufijo == "vision":
            requiere_vision = True
        elif sufijo:
            # Post-Task-14 review (gate): an "auto:<something>" suffix that is
            # neither a known profile nor "tools"/"vision" (e.g. "auto:turbo", a
            # typo for "auto:tools") fell through all three branches above WITHOUT
            # touching anything -- silently identical to asking for plain "auto",
            # with no warning that the suffix was ignored. For a client that
            # genuinely wanted to require a capability (e.g. tools, for an agent
            # expecting a tool_call) that is dangerous in silence: it receives an
            # ordinary "balanced" response, not the 400 that would have told it the
            # alias was misspelled. "auto" with no ":" (suffix == "") is still
            # valid and does NOT reach here -- see PROFILES, which already includes
            # "balanceado" (so "auto:balanceado" also resolves normally, without
            # going through this branch).
            raise HTTPException(400, {
                "message": f"alias de modelo desconocido: '{modelo_pedido}'",
                "sugerencias": ALIASES,
            })
    else:
        modelo = modelo_pedido

    def _normalizar_exigidas():
        # `x_requiere: "tools"` (a bare string, instead of the documented
        # list) is accepted as a single value -- an ordinary REST API
        # convenience. Anything else that is neither a string nor a list (or a
        # list with an unhashable element, like `[["tools"]]`) blows up INSIDE
        # this `set(...)` with a TypeError -- which is what `_read_field`
        # catches and turns into a 400. `set("tools")` (without the wrapping
        # above) would iterate character by character -- {'t','o','l','s'} --
        # and the requirement would be silently ignored; that is why the string
        # is wrapped in a list BEFORE the set(), not after.
        valor = cuerpo.get("x_requiere") or []
        if isinstance(valor, str):
            valor = [valor]
        return set(valor)
    exigidas = _read_field(
        "x_requiere", cuerpo.get("x_requiere"),
        "x_requiere debe ser un string o una lista de strings",
        _normalizar_exigidas)

    x_min_contexto_bruto = cuerpo.get("x_min_contexto")
    min_contexto = _read_field(
        "x_min_contexto", x_min_contexto_bruto,
        "x_min_contexto debe ser un numero entero",
        lambda: int(x_min_contexto_bruto) if x_min_contexto_bruto else 0)

    return RouteRequest(
        model=modelo,
        needs_tools=requiere_tools or "tools" in exigidas,
        needs_vision=requiere_vision or "vision" in exigidas,
        min_context=min_contexto,
        profile=perfil,
        allow_paid=bool(cuerpo.get("x_permitir_pago", True)),
    )


@dataclass
class State:
    store: object
    proxy: object
    api_keys: set
    daily_paid_cap: int
    rate_limiter: PerKeyRateLimiter = field(default_factory=lambda: PerKeyRateLimiter(60))
    providers: list = field(default_factory=list)   # used by the scheduler
    http: object = None                             # shared httpx client
    # Generator for the draw between tied routes (see router.shuffle_ties).
    # None = no draw, strictly deterministic order. It is injected from
    # main.build_state() according to ROTAR_EMPATES so tests can build a
    # deterministic State without going through environment variables.
    rng: object = None


def _resolve_api_key(x_api_key: str | None, authorization: str | None) -> str | None:
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


def create_app(estado: State) -> FastAPI:
    # title/version/summary/description and customise_openapi (Task 14) only
    # enrich what /docs and /openapi.json serve -- see llm_libre.openapi. They
    # touch no route and no logic: require_api_key, parse_request and the
    # completions passthrough stay exactly the same.
    app = FastAPI(title=TITLE, version=VERSION, summary=SUMMARY, description=DESCRIPTION)
    customise_openapi(app)

    def require_api_key(x_api_key: str | None, authorization: str | None = None) -> str:
        llave = _resolve_api_key(x_api_key, authorization)
        if not llave or llave not in estado.api_keys:
            raise HTTPException(401, "llave invalida")
        if not estado.rate_limiter.allow(llave, time.time()):
            raise HTTPException(429, "demasiadas peticiones para esta llave")
        return llave

    def _routes_for(cuerpo: dict, llave: str) -> tuple[list, object]:
        pedido = parse_request(cuerpo)
        activas = estado.store.active_routes()
        # An explicit id that no longer exists deserves a 404 with hints, not a
        # generic 400: it is exactly the failure this project exists to prevent.
        if pedido.model is not None and not any(r.model_id == pedido.model for r in activas):
            raise HTTPException(404, {
                "message": f"el modelo '{pedido.model}' ya no existe",
                "sugerencias": _similar_ids(pedido.model, activas),
            })
        tope_alcanzado = False
        if pedido.allow_paid:
            dia = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if estado.store.paid_usage(llave, dia) >= estado.daily_paid_cap:
                pedido = replace(pedido, allow_paid=False)
                tope_alcanzado = True
        ahora = time.time()
        metricas = _metrics(estado, ahora)
        rutas = order_routes(activas, metricas, pedido, ahora, estado.rng)
        if not rutas:
            _no_routes(activas, pedido, metricas, ahora, tope_alcanzado)   # siempre levanta
        return rutas, pedido

    @app.post("/v1/chat/completions", **CHAT_COMPLETIONS_DOCS)
    async def completions(request: Request, x_api_key: str | None = Header(None),
                          authorization: str | None = Header(None)):
        llave = require_api_key(x_api_key, authorization)
        cuerpo = await request.json()
        rutas, pedido = _routes_for(cuerpo, llave)
        ahora = time.time()
        crudo = bool(cuerpo.get("x_crudo"))

        def _contar_uso_pago(ruta) -> None:
            # HIGH 4 (round 9): called for every BILLABLE attempt against a paid
            # route -- the provider charges for a 200 with useful content, an
            # empty 200, and a reasoning model that burned its budget alike (it
            # generates tokens in all three cases); only a NETWORK error or a
            # non-200 status produces no charge. This used to be called only on
            # SUCCESS (`r.route`/`on_route_committed`): an empty 200 from a paid
            # route was genuinely billed and appeared neither in /v1/uso nor
            # against TOPE_PAGO_DIARIO -- measured, 40/40 billable calls with
            # `pago_hoy: 0`. See proxy.py (`on_billable_attempt`) for where
            # "billable" is decided.
            estado.store.add_paid_usage(
                llave, datetime.now(timezone.utc).strftime("%Y-%m-%d"))

        if cuerpo.get("stream"):
            return StreamingResponse(
                estado.proxy.complete_stream(
                    rutas, cuerpo, ahora, crudo,
                    on_billable_attempt=_contar_uso_pago),
                media_type="text/event-stream")
        r = await estado.proxy.complete(rutas, cuerpo, ahora, crudo,
                                         on_billable_attempt=_contar_uso_pago)
        if r.status == 503 and r.upstream_code == 404 and pedido.model is not None:
            # ALSO from the round 6 review -- literally the project's reason to
            # exist: `pedido.model` is STILL in our catalogue (it passed the 404
            # check in `_routes_for`, above) but the real provider no longer has
            # it: a genuine 404, live. The route already took the reliability hit
            # (a 404 is evidence about the route by default, see
            # proxy._is_client_error), but without this check the client only saw
            # a generic 503 ("detalle": "HTTP 404") -- indistinguishable from any
            # other transient unavailability, for the entire window of up to 5h
            # before the next catalogue sync (never, for paid routes, which are
            # not probed).
            #
            # Only with an EXPLICIT model: in "auto" mode `pedido.model` is None,
            # there is no particular id to make suggestions about, and the route
            # that failed is not necessarily the only reasonable candidate -- that
            # case keeps the usual 503.
            #
            # Only the synchronous path: when streaming, the 200 status and the
            # SSE headers have already gone out before the proxy knows whether the
            # route served, so there is no HTTP room left to change it to a 404.
            raise HTTPException(404, {
                "message": f"el modelo '{pedido.model}' ya no existe",
                "sugerencias": _similar_ids(pedido.model, estado.store.active_routes()),
            })
        cabeceras = {"X-Intentos": str(r.attempts)}
        if r.route is not None:
            cabeceras["X-Ruta-Usada"] = r.route.key
            cabeceras["X-Tier"] = r.route.tier
        cuerpo_resp = r.json
        if r.status == 200 and r.reasoning and isinstance(cuerpo_resp, dict):
            # Section 6.1: the trimmed reasoning is returned in a separate
            # field, "for whoever wants it". It used to be trimmed out of
            # `content` and thrown away, so with the default `x_crudo: false`
            # there was no way to recover it. It goes at the top level (not
            # inside `choices`) because there any OpenAI SDK ignores it without
            # breaking, which is the only condition the contract places on
            # extensions.
            #
            # A KNOWN, deliberate LIMITATION: it is NOT returned when streaming.
            # Putting it there would require emitting a non-standard SSE event,
            # which is exactly what section 6 rules out for risking the parsing
            # of the SDKs this contract exists to please. A client that streams
            # and wants the reasoning asks for `x_crudo: true` and receives it
            # inside `content`, exactly as the provider sent it.
            cuerpo_resp = {**cuerpo_resp, "x_razonamiento": r.reasoning}
        return JSONResponse(cuerpo_resp, status_code=r.status, headers=cabeceras)

    @app.get("/v1/models", **MODELS_DOCS)
    def modelos(x_api_key: str | None = Header(None), authorization: str | None = Header(None)):
        require_api_key(x_api_key, authorization)
        datos = [{"id": r.model_id, "object": "model", "owned_by": r.provider}
                 for r in estado.store.active_routes()]
        datos += [{"id": a, "object": "model", "owned_by": "llm-libre"} for a in ALIASES]
        return {"object": "list", "data": datos}

    @app.get("/v1/ranking", **RANKING_DOCS)
    def ranking(x_api_key: str | None = Header(None), authorization: str | None = Header(None)):
        require_api_key(x_api_key, authorization)
        ahora = time.time()
        metricas = _metrics(estado, ahora)
        # Sorted with the SAME key router.order_routes uses (the "balanceado"
        # profile, the same one each row's score uses) -- not a private sort by
        # score: this endpoint exists to audit WHY the router chose what it chose
        # (README), and it used to be able to show one route at the top while
        # X-Ruta-Usada said a different one, because it looked at neither
        # `prioridad` nor the cooldown -- a punished route (one the router would
        # never pick right now) could head the table. `en_cooldown_hasta` is still
        # exposed per row for diagnostics; what changes is the ORDER.
        activas = sorted(estado.store.active_routes(),
                         key=lambda r: sort_key(r, metricas[r.key], "balanceado", ahora))
        filas = []
        for r in activas:
            m = metricas[r.key]
            medida = m.quality_measured_at is not None
            filas.append({"clave": r.key, "tier": r.tier, "prioridad": r.priority,
                          "puntaje": round(score(m, "balanceado"), 4),
                          # "never measured" is stated, not disguised: showing
                          # the neutral value in `calidad` as if someone had
                          # measured it is what made it invisible that `auto` was
                          # ordering by an assumption. The value that DID enter
                          # the score goes separately, in `calidad_asumida`.
                          "calidad": round(m.quality, 3) if medida else None,
                          "calidad_medida": medida,
                          "calidad_asumida": None if medida else round(m.quality, 3),
                          "ultima_sonda_calidad": _iso(m.quality_measured_at),
                          "ultima_sonda": _iso(m.last_probe_at),
                          "confiabilidad": round(m.reliability, 3),
                          # Two different numbers on purpose: ttft_p50_ms is
                          # time to first token (only streaming measures it, and
                          # it is what weighs in the score); latencia_p50_ms is
                          # the complete round-trip (non-streaming and probes).
                          # They used to share a column and the average meant
                          # nothing.
                          "ttft_p50_ms": m.ttft_p50_ms,
                          "latencia_p50_ms": m.latency_p50_ms,
                          "en_cooldown_hasta": m.cooldown_until,
                          "tools": r.capabilities.tools, "vision": r.capabilities.vision,
                          "contexto": r.capabilities.context})
        return {"rutas": filas}

    @app.get("/v1/uso", **USAGE_DOCS)
    def uso(x_api_key: str | None = Header(None), authorization: str | None = Header(None)):
        llave = require_api_key(x_api_key, authorization)
        dia = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return {"dia": dia, "pago_hoy": estado.store.paid_usage(llave, dia),
                "tope": estado.daily_paid_cap}

    @app.get("/health", **HEALTH_DOCS)
    def health():
        # Honest: it looks at whether there is a LIVE, serviceable route, not at
        # whether the process is up. "Live" requires TWO things, not one: not
        # being in cooldown (round 8: ONLY a 429 triggers it immediately, or a
        # PROBE -- periodic or on demand -- confirming the route is broken; a real
        # client's traffic never excludes a route directly, see the header comment
        # of SUSPICION_THRESHOLD in proxy.py) AND POSITIVE evidence that it serves
        # (`Storage.has_liveness_evidence`).
        #
        # Task 13, round 6 review, Part 2: THIS NO LONGER LOOKS AT `reliability`.
        # `reliability` is an average of recent traffic, and an average is dragged
        # to 0 by any repeated pattern from ONE client -- the case that proved it
        # is `403`, genuinely ambiguous (suspended account = evidence about the
        # route, vs moderated content = evidence about the REQUEST) and one the
        # gateway cannot disambiguate without parsing each provider's specific
        # body. 30 requests with moderated content from a single key were enough
        # to drop `/health` to "caido" for ALL keys, surviving a process restart
        # against the same database -- because Coolify uses this endpoint as its
        # health check and restarts the container when it fails.
        #
        # "Evidence of life, not absence of death": a recent success proves the
        # route serves; a thousand failures from one client do not prove it does
        # not. Failures alone are NEVER enough to declare a route dead here -- see
        # the docstring of `has_liveness_evidence` for the full criterion (a recent
        # real success, or a recent successful health probe, or no telemetry yet).
        #
        # `/v1/ranking` (below) still uses `reliability` exactly as before -- that
        # does NOT change. The asymmetry is deliberate: a badly scored route in the
        # ranking merely loses position and self-corrects; a route `/health`
        # declares dead restarts the container. The ranking can afford to be
        # sensitive: health cannot.
        ahora = time.time()
        activas = estado.store.active_routes()
        metricas = _metrics(estado, ahora)

        def _viva(r) -> bool:
            m = metricas.get(r.key, NEUTRAL_METRICS)
            return (m.cooldown_until <= ahora
                    and estado.store.has_liveness_evidence(r.key, ahora))

        libres = [r for r in activas if _viva(r)]
        gratis = [r for r in libres if r.tier == "free"]
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


def _no_routes(activas: list, pedido, metricas: dict, ahora: float,
               tope_alcanzado: bool) -> None:
    """Raise the right error when the chain of attempts comes back empty.

    Design section 9 separates two situations the previous version conflated into
    a single 400:

    - **No route can satisfy the request** (capabilities, vision, a context
      nobody has) -> `400`. That genuinely is a client error.
    - **There are routes that could serve but they are all down or in cooldown**
      (including the case "this key exceeded its daily paid cap") -> `503`, with
      `proxima_liberacion`.

    Why it matters: `order_routes` filters out cooldowns, so in ANY outage of the
    free tiers -- the expected failure, not an exotic one -- the list arrived
    empty and a 400 went out. Every SDK and every alerting layer reads 400 as
    "your request is malformed": they do not retry and they wake nobody.

    `x_permitir_pago: false` is NOT considered here: it is a policy of the
    caller, not a capability missing from the pool. A client that forbids paid
    routes and runs out of live free ones is in the unavailability case (503,
    retryable), not in the invalid-request one.
    """
    compat = compatible_routes(activas, pedido)
    if not compat:
        raise HTTPException(400, {
            "message": "ninguna ruta cumple lo pedido",
            "pedido": pedido.as_wire(),
            "rutas_activas": len(activas),
        })
    liberaciones = [metricas[r.key].cooldown_until for r in compat
                    if r.key in metricas and metricas[r.key].cooldown_until > ahora]
    raise HTTPException(503, {
        "message": "todas las rutas que podrian servir estan caidas o en cooldown",
        "pedido": pedido.as_wire(),
        "rutas_compatibles": len(compat),
        # When the FIRST of them is released. None = none is being punished
        # (they are excluded for another reason, e.g. the paid cap).
        "proxima_liberacion": min(liberaciones) if liberaciones else None,
        "tope_pago_alcanzado": tope_alcanzado,
    })


def _similar_ids(pedido: str, activas: list) -> list[str]:
    # Round 7, LOW from the gate: the caller of the live-404 (above) passes
    # `activas` UNFILTERED -- that `pedido` id is STILL in the local catalogue
    # (that is the entire premise of that case: still in the catalogue, but the
    # real provider no longer has it). Without excluding it here,
    # `get_close_matches` finds IT ITSELF as the most obvious "match" (distance
    # zero) and the client reads `"el modelo 'a:free' ya no existe"` with
    # `sugerencias: ['a:free', ...]`. It is excluded HERE, in the function, and
    # not in each caller: a suggestion list must never be able to suggest the very
    # id just declared dead.
    candidatos = [r.model_id for r in activas if r.model_id != pedido]
    return difflib.get_close_matches(pedido, candidatos, n=3, cutoff=0.3)


def _iso(momento: float | None) -> str | None:
    if momento is None:
        return None
    return datetime.fromtimestamp(momento, timezone.utc).isoformat().replace("+00:00", "Z")


def _metrics(estado: State, ahora: float) -> dict:
    base = estado.store.metrics()
    for clave, hasta in estado.proxy.cooldowns.items():
        if clave in base:
            # `replace` and not a positional `type(m)(...)`: rebuilding by hand
            # leaves out any new Metrics field (e.g. quality_measured_at), and a
            # route in cooldown would start looking "never measured".
            base[clave] = replace(base[clave], cooldown_until=hasta)
    return base
