import difflib
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from llm_libre.assets import content_disposition, localise
from llm_libre.auth import RateLimiter, client_ip

from llm_libre.models import (NEUTRAL_METRICS, UNKNOWN_BUDGET, RequestTrace,
                              RouteRequest)
from llm_libre.openapi import (CHAT_COMPLETIONS_DOCS, DESCRIPTION, HEALTH_DOCS,
                               ASSETS_DOCS, IMAGES_DOCS, MODELS_DOCS, RANKING_DOCS, SUMMARY, TITLE,
                               USAGE_DOCS, VERSION, customise_openapi)
from llm_libre.proxy import CHAT, IMAGES
from llm_libre.ranking import score
from llm_libre.router import compatible_routes, order_routes, sort_key

PROFILES = {"fast", "balanced", "strong"}
ALIASES = ["auto", "auto:fast", "auto:strong", "auto:tools", "auto:vision"]


def _read_field(field_name: str, raw_value, message: str, interpret):
    """Run `interpret` (a zero-argument function interpreting `raw_value` -- a
    value that arrived VERBATIM from the client's JSON body, never something the
    gateway builds) and convert any `TypeError`/`ValueError`/`AttributeError` --
    the family of exceptions a field of the WRONG TYPE raises (`.strip()` on a
    number, `int()` on a list, `set()` on a boolean or on a list containing a
    list) -- into a uniform 400, instead of letting it escape uncaught to
    FastAPI's generic handler as an opaque 500.

    Post-Task-14 review (second gate): the same bug got through twice in a row --
    first in `x_min_context` (fixed by hand with a local try/except), then in
    `x_requires` (the same pattern, reinvented) -- and a third instance (`model`,
    via `.strip()`) was still uncaught when the other two were found. The real
    axis was never "this particular field", it was "every field this endpoint
    interprets before using it arrives raw from the client, with no guaranteed
    type" -- see the docstring of `build_request`, the same "this is passthrough,
    do not trust the shape" principle. This function is the SINGLE point that
    interpretation goes through from now on, so a fourth field (if this endpoint
    gains one) does not have to reinvent the try/except or risk forgetting it."""
    try:
        return interpret()
    except (TypeError, ValueError, AttributeError):
        raise HTTPException(400, {
            "message": message, "field": field_name, "received_value": raw_value})


def _has_image(body: dict) -> bool:
    """True if any message carries an image.

    In the OpenAI format, `content` is either a string (text only) or a LIST of
    parts, and an image is a part with `type: "image_url"` (the classic format)
    or `"input_image"` (the Responses API one, which several clients already
    send). Both are accepted: the gateway exists so that changing `base_url` is
    enough, and rejecting the newer format would force the client to know which
    proxy it is talking to.

    Without this, `needs_vision` was only set by the `auto:vision` alias or by
    `x_requires` -- that is, only if the client ANNOUNCED it. A client that
    simply sends the image, which is the normal thing against an OpenAI API,
    could end up on a text-only route: a 200 with a response that ignores the
    image, or a 400 from the provider. Neither of those says "that route cannot
    see images".

    On a malformed body it returns False and lets the provider reject: this
    function picks a route, it does not validate the request.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return False
    for m in messages:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("image_url", "input_image"):
                return True
    return False


def parse_request(body: dict) -> RouteRequest:
    # A "model" of pure whitespace is, for all practical purposes, absent: it
    # must not sneak through as if it were an explicit id (it would be empty
    # after the strip and would produce a confusing 404 about the model '').
    raw_model = body.get("model")
    requested_model = _read_field(
        "model", raw_model, "model must be a string",
        lambda: (raw_model or "").strip()) or "auto"
    model, profile = None, "balanced"
    needs_tools = bool(body.get("tools"))
    needs_vision = _has_image(body)

    if requested_model == "auto" or requested_model.startswith("auto:"):
        suffix = requested_model[5:] if ":" in requested_model else ""
        if suffix in PROFILES:
            profile = suffix
        elif suffix == "tools":
            needs_tools = True
        elif suffix == "vision":
            needs_vision = True
        elif suffix:
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
            # "balanced" (so "auto:balanced" also resolves normally, without
            # going through this branch).
            raise HTTPException(400, {
                "message": f"unknown model alias: '{requested_model}'",
                "suggestions": ALIASES,
            })
    else:
        model = requested_model

    def _normalise_required():
        # `x_requires: "tools"` (a bare string, instead of the documented
        # list) is accepted as a single value -- an ordinary REST API
        # convenience. Anything else that is neither a string nor a list (or a
        # list with an unhashable element, like `[["tools"]]`) blows up INSIDE
        # this `set(...)` with a TypeError -- which is what `_read_field`
        # catches and turns into a 400. `set("tools")` (without the wrapping
        # above) would iterate character by character -- {'t','o','l','s'} --
        # and the requirement would be silently ignored; that is why the string
        # is wrapped in a list BEFORE the set(), not after.
        value = body.get("x_requires") or []
        if isinstance(value, str):
            value = [value]
        return set(value)
    required = _read_field(
        "x_requires", body.get("x_requires"),
        "x_requires must be a string or a list of strings",
        _normalise_required)

    raw_min_context = body.get("x_min_context")
    min_context = _read_field(
        "x_min_context", raw_min_context,
        "x_min_context must be an integer",
        lambda: int(raw_min_context) if raw_min_context else 0)

    return RouteRequest(
        model=model,
        needs_tools=needs_tools or "tools" in required,
        needs_vision=needs_vision or "vision" in required,
        min_context=min_context,
        profile=profile,
        allow_paid=bool(body.get("x_allow_paid", True)),
    )


@dataclass
class State:
    store: object
    proxy: object
    api_keys: set
    daily_paid_cap: int
    rate_limiter: RateLimiter = field(default_factory=lambda: RateLimiter(60))
    providers: list = field(default_factory=list)   # used by the scheduler
    http: object = None                             # shared httpx client
    # Generator for the draw between tied routes (see router.shuffle_ties).
    # None = no draw, strictly deterministic order. It is injected from
    # main.build_state() according to ROTATE_TIES so tests can build a
    # deterministic State without going through environment variables.
    rng: object = None
    # Bounds what an UNAUTHENTICATED caller can cost, which the per-key limiter
    # cannot: it is keyed by a key, and a request with no valid key never gets
    # one. Applies to failed auth (otherwise key-guessing is free and unlimited)
    # and to the three paths that have no key by design -- /health, which is the
    # most expensive endpoint in the service, and /v1/assets. Generous on
    # purpose: it is a ceiling on abuse, not a quota for ordinary use.
    ip_rate_limiter: RateLimiter = field(default_factory=lambda: RateLimiter(120))
    # Where generated binaries are stored, and the origin their URLs carry.
    # Both None in most tests: with no store the images endpoint simply hands
    # back the provider's own URL, which is what it did before this existed.
    assets: object = None
    public_base_url: str = ""


def _resolve_api_key(x_api_key: str | None, authorization: str | None) -> str | None:
    """Accepts the key via `X-API-Key` (the convention `arkiv-api`, the sibling
    gateway, already uses) or via `Authorization: Bearer <key>` (what ANY OpenAI
    SDK sends with no extra configuration through its `api_key` parameter --
    which is, literally, the central promise of this contract: "change only
    base_url"). If both arrive, `X-API-Key` wins: it is the more explicit
    convention and the one existing callers already use.

    An `Authorization` without the `Bearer ` prefix (or otherwise malformed)
    resolves no key -- it neither blows up nor tries to guess, it simply falls
    through to the same 401 as a missing key.
    """
    if x_api_key:
        return x_api_key
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


def create_app(state: State) -> FastAPI:
    # title/version/summary/description and customise_openapi (Task 14) only
    # enrich what /docs and /openapi.json serve -- see llm_libre.openapi. They
    # touch no route and no logic: require_api_key, parse_request and the
    # completions passthrough stay exactly the same.
    app = FastAPI(title=TITLE, version=VERSION, summary=SUMMARY, description=DESCRIPTION)
    customise_openapi(app)

    def _limit_by_ip(request: Request) -> None:
        """Bound what one source can cost before any key is involved."""
        if state.ip_rate_limiter is None:
            return
        if not state.ip_rate_limiter.allow(client_ip(request), time.time()):
            raise HTTPException(429, "too many requests from this address")

    def require_api_key(x_api_key: str | None, authorization: str | None = None,
                        request: Request | None = None) -> str:
        # The IP limit comes FIRST, before the key is even looked at. The other
        # way round -- which is what this did until the gateway went public --
        # a wrong key raises 401 without ever reaching a limiter, so guessing
        # keys, or simply hammering the endpoint, costs the caller nothing and
        # is bounded by nothing. The per-key limit below can only ever punish
        # someone who already has a valid key.
        if request is not None:
            _limit_by_ip(request)
        key = _resolve_api_key(x_api_key, authorization)
        if not key or key not in state.api_keys:
            raise HTTPException(401, "invalid api key")
        if not state.rate_limiter.allow(key, time.time()):
            raise HTTPException(429, "too many requests for this key")
        return key

    def _routes_for(body: dict, key: str, needs_images: bool = False) -> tuple[list, object]:
        request = parse_request(body)
        if needs_images:
            # Set HERE, not in parse_request: it is decided by which endpoint was
            # called, never by anything the client can put in the body. That is
            # what makes it impossible for a chat request to be routed to an
            # image generator, or the reverse.
            request = replace(request, needs_images=True)
        active = state.store.active_routes()
        # An explicit id that no longer exists deserves a 404 with hints, not a
        # generic 400: it is exactly the failure this project exists to prevent.
        if request.model is not None and not any(r.model_id == request.model for r in active):
            raise HTTPException(404, {
                "message": f"the model '{request.model}' no longer exists",
                "suggestions": _similar_ids(request.model, active),
            })
        cap_reached = False
        if request.allow_paid:
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if state.store.paid_usage(key, day) >= state.daily_paid_cap:
                request = replace(request, allow_paid=False)
                cap_reached = True
        now = time.time()
        # The capability this request is actually asking for -- see _metrics.
        metrics = _metrics(state, now, IMAGES if request.needs_images else CHAT)
        routes = order_routes(active, metrics, request, now, state.rng)
        if not routes:
            _no_routes(active, request, metrics, now, cap_reached)   # always raises
        return routes, request

    @app.post("/v1/chat/completions", **CHAT_COMPLETIONS_DOCS)
    async def completions(request: Request, x_api_key: str | None = Header(None),
                          authorization: str | None = Header(None)):
        key = require_api_key(x_api_key, authorization, request)
        body = await request.json()
        routes, route_request = _routes_for(body, key)
        now = time.time()
        raw = bool(body.get("x_raw"))

        def _count_paid_usage(route) -> None:
            # HIGH 4 (round 9): called for every BILLABLE attempt against a paid
            # route -- the provider charges for a 200 with useful content, an
            # empty 200, and a reasoning model that burned its budget alike (it
            # generates tokens in all three cases); only a NETWORK error or a
            # non-200 status produces no charge. This used to be called only on
            # SUCCESS (`r.route`/`on_route_committed`): an empty 200 from a paid
            # route was genuinely billed and appeared neither in /v1/usage nor
            # against DAILY_PAID_CAP -- measured, 40/40 billable calls with
            # `paid_today: 0`. See proxy.py (`on_billable_attempt`) for where
            # "billable" is decided.
            state.store.add_paid_usage(
                key, datetime.now(timezone.utc).strftime("%Y-%m-%d"))

        # One trace per CLIENT request, shared by every attempt it causes, so the
        # failover chain stays reconstructable afterwards. `body["model"]` verbatim
        # and not the resolved profile: the question this answers is what was asked
        # against what was served, and resolving it here erases the asking side.
        trace = RequestTrace(request_id=uuid4().hex, requested=body.get("model"))
        if body.get("stream"):
            return StreamingResponse(
                state.proxy.complete_stream(
                    routes, body, now, raw,
                    on_billable_attempt=_count_paid_usage, trace=trace),
                media_type="text/event-stream")
        r = await state.proxy.complete(routes, body, now, raw,
                                       on_billable_attempt=_count_paid_usage,
                                       trace=trace)
        if r.status == 503 and r.upstream_code == 404 and route_request.model is not None:
            # ALSO from the round 6 review -- literally the project's reason to
            # exist: `route_request.model` is STILL in our catalogue (it passed the
            # 404 check in `_routes_for`, above) but the real provider no longer has
            # it: a genuine 404, live. The route already took the reliability hit
            # (a 404 is evidence about the route by default, see
            # proxy._is_client_error), but without this check the client only saw
            # a generic 503 ("detail": "HTTP 404") -- indistinguishable from any
            # other transient unavailability, for the entire window of up to 5h
            # before the next catalogue sync (never, for paid routes, which are
            # not probed).
            #
            # Only with an EXPLICIT model: in "auto" mode `route_request.model` is
            # None, there is no particular id to make suggestions about, and the
            # route that failed is not necessarily the only reasonable candidate --
            # that case keeps the usual 503.
            #
            # Only the synchronous path: when streaming, the 200 status and the
            # SSE headers have already gone out before the proxy knows whether the
            # route served, so there is no HTTP room left to change it to a 404.
            raise HTTPException(404, {
                "message": f"the model '{route_request.model}' no longer exists",
                "suggestions": _similar_ids(route_request.model,
                                            state.store.active_routes()),
            })
        headers = {"X-Attempts": str(r.attempts)}
        if r.route is not None:
            headers["X-Route-Used"] = r.route.key
            headers["X-Tier"] = r.route.tier
        response_body = r.json
        if r.status == 200 and r.reasoning and isinstance(response_body, dict):
            # Section 6.1: the trimmed reasoning is returned in a separate
            # field, "for whoever wants it". It used to be trimmed out of
            # `content` and thrown away, so with the default `x_raw: false`
            # there was no way to recover it. It goes at the top level (not
            # inside `choices`) because there any OpenAI SDK ignores it without
            # breaking, which is the only condition the contract places on
            # extensions.
            #
            # A KNOWN, deliberate LIMITATION: it is NOT returned when streaming.
            # Putting it there would require emitting a non-standard SSE event,
            # which is exactly what section 6 rules out for risking the parsing
            # of the SDKs this contract exists to please. A client that streams
            # and wants the reasoning asks for `x_raw: true` and receives it
            # inside `content`, exactly as the provider sent it.
            response_body = {**response_body, "x_reasoning": r.reasoning}
        return JSONResponse(response_body, status_code=r.status, headers=headers)

    @app.post("/v1/images/generations", **IMAGES_DOCS)
    async def images(request: Request, x_api_key: str | None = Header(None),
                     authorization: str | None = Header(None)):
        """OpenAI's image-generation contract, routed by the `images` capability.

        The point of the endpoint is the FILTER: `compatible_routes` drops every
        route that cannot generate before a single request is attempted, so a
        prompt never spends 45s on a chat-only model to be told it cannot draw.
        Today that leaves grok's three imagine-agent-mode routes and mistral's,
        out of 52 -- the other 49 are never touched.

        The same failover, cooldown, telemetry and paid-cap machinery as chat:
        `proxy.generate_images` is a sibling of `proxy.complete`, not a special
        case inside it. See its docstring for what genuinely differs.
        """
        key = require_api_key(x_api_key, authorization, request)
        body = await request.json()
        routes, _ = _routes_for(body, key, needs_images=True)
        now = time.time()

        def _count_paid_usage(route) -> None:
            state.store.add_paid_usage(
                key, datetime.now(timezone.utc).strftime("%Y-%m-%d"))

        r = await state.proxy.generate_images(
            routes, body, now, on_billable_attempt=_count_paid_usage,
            trace=RequestTrace(request_id=uuid4().hex, requested=body.get("model")))
        if r.status == 200 and state.assets is not None:
            # The provider's URL never reaches the client: it expires, it may
            # need headers the client does not have, and it names who served --
            # which is the one thing this gateway promises not to make the
            # client care about. See assets.localise; it degrades to the
            # original URL rather than failing the request.
            r = replace(r, json=await localise(
                r.json, state.assets, state.proxy.http, state.public_base_url,
                body.get("response_format"), now))
        headers = {"X-Attempts": str(r.attempts)}
        if r.route is not None:
            headers["X-Route-Used"] = r.route.key
            headers["X-Tier"] = r.route.tier
        return JSONResponse(r.json, status_code=r.status, headers=headers)

    @app.get("/v1/assets/{asset_id}", **ASSETS_DOCS)
    def asset(asset_id: str, request: Request):
        """Serve a stored binary. DELIBERATELY unauthenticated.

        It has to be: the whole point is that the URL works in an `<img>` tag, a
        markdown preview or a browser address bar, none of which can attach an
        API key. What protects it instead is the id -- a SHA-256 of the content,
        so it cannot be guessed, enumerated, or derived from the prompt.

        `Cache-Control: immutable` is honest here rather than optimistic: the id
        IS the hash of the bytes, so the response for a given id can never
        change. Retention is the only reason one ever stops existing, and then
        it becomes a 404 rather than different content.
        """
        _limit_by_ip(request)
        found = state.assets.get(asset_id) if state.assets is not None else None
        if found is None:
            raise HTTPException(404, "asset not found")
        data, content_type = found
        return Response(data, media_type=content_type, headers={
            "Cache-Control": "public, max-age=31536000, immutable",
            # An SVG is an image everywhere except in a browser, where it can
            # run script -- so it is served as a download. See assets.py.
            "Content-Disposition": content_disposition(content_type),
            # The bytes come from a third party: no sniffing, no framing.
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "default-src 'none'; sandbox",
        })

    @app.get("/v1/models", **MODELS_DOCS)
    def models(request: Request, x_api_key: str | None = Header(None),
               authorization: str | None = Header(None)):
        require_api_key(x_api_key, authorization, request)
        data = [{"id": r.model_id, "object": "model", "owned_by": r.provider}
                for r in state.store.active_routes()]
        data += [{"id": a, "object": "model", "owned_by": "llm-libre"} for a in ALIASES]
        return {"object": "list", "data": data}

    @app.get("/v1/traffic")
    def traffic(request: Request, hours: float = 24.0,
                x_api_key: str | None = Header(None),
                authorization: str | None = Header(None)):
        """Where client requests actually went, over the last `hours`.

        A companion to /v1/ranking, and deliberately a different question.
        /v1/ranking shows what the router BELIEVES about each route right now;
        this shows what it DID. The two disagreeing is the interesting case --
        a route ranked first that keeps appearing in `fell_away_from` is one the
        scores like and reality does not.
        """
        require_api_key(x_api_key, authorization, request)
        # Clamped: `hours` reaches this straight from the query string, and an
        # absurd value would scan the whole table on a machine that is already
        # saturated. 720h is the retention window (probing.RETENTION_DAYS), so
        # anything above it can return nothing more anyway.
        hours = max(0.0, min(float(hours), 720.0))
        return state.store.traffic(since=time.time() - hours * 3600.0)

    @app.get("/v1/ranking", **RANKING_DOCS)
    def ranking(request: Request, x_api_key: str | None = Header(None),
                authorization: str | None = Header(None)):
        require_api_key(x_api_key, authorization, request)
        now = time.time()
        metrics = _metrics(state, now)
        # Sorted with the SAME key router.order_routes uses (the "balanced"
        # profile, the same one each row's score uses) -- not a private sort by
        # score: this endpoint exists to audit WHY the router chose what it chose
        # (README), and it used to be able to show one route at the top while
        # X-Route-Used said a different one, because it looked at neither
        # `priority` nor the cooldown -- a punished route (one the router would
        # never pick right now) could head the table. `cooldown_until` is still
        # exposed per row for diagnostics; what changes is the ORDER.
        active = sorted(state.store.active_routes(),
                        key=lambda r: sort_key(r, metrics[r.key], "balanced", now))
        # What we have inferred about each route's CHAT allowance, for the
        # providers that publish nothing (see Storage.rate_budgets). Chat because
        # that is the capability this table orders by; an image allowance is a
        # different resource and would answer a different question.
        budgets = state.store.rate_budgets(now, capability=CHAT)
        rows = []
        for r in active:
            m = metrics[r.key]
            b = budgets.get(r.key, UNKNOWN_BUDGET)
            measured = m.quality_measured_at is not None
            rows.append({"key": r.key, "tier": r.tier, "priority": r.priority,
                         "score": round(score(m, "balanced"), 4),
                         # "never measured" is stated, not disguised: showing
                         # the neutral value in `quality` as if someone had
                         # measured it is what made it invisible that `auto` was
                         # ordering by an assumption. The value that DID enter
                         # the score goes separately, in `quality_assumed`.
                         "quality": round(m.quality, 3) if measured else None,
                         "quality_measured": measured,
                         "quality_assumed": None if measured else round(m.quality, 3),
                         "last_quality_probe": _iso(m.quality_measured_at),
                         "last_probe": _iso(m.last_probe_at),
                         "reliability": round(m.reliability, 3),
                         # What we believe this route's quota is, over which
                         # window, and how much of it is left -- inferred from our
                         # own history for the providers that do not publish one.
                         # The same measured/assumed split as `quality` above, and
                         # for the same reason: `rate_allowance` is null until the
                         # route has actually been seen refusing, and `rate_floor`
                         # ("we have seen it sustain at least this per hour") is
                         # reported separately so a lower bound is never mistaken
                         # for an allowance.
                         #
                         # `rate_window_s` is part of the measurement: most
                         # refusals we receive are against DAILY quotas, and an
                         # allowance means nothing without the window it belongs
                         # to. `rate_per_hour` normalises it for comparison
                         # against providers that advertise an hourly figure --
                         # read that one as a rate, never as a budget, since
                         # nothing stops a daily quota being spent in one hour.
                         "rate_allowance": b.allowance,
                         "rate_window_s": b.window_s,
                         "rate_per_hour": (round(b.per_hour, 2)
                                           if b.per_hour is not None else None),
                         "rate_measured": b.measured,
                         "rate_floor": b.floor,
                         "rate_used": b.used,
                         "rate_remaining": b.remaining,
                         # Seconds until the allowance runs out at the current
                         # rate, and how long refusals have taken to clear. Both
                         # null while unknown rather than zero, which would read
                         # as "runs out now" and "recovers instantly".
                         "rate_exhausts_in_s": (round(b.exhausts_in_s)
                                                if b.exhausts_in_s is not None else None),
                         "rate_recovery_s": (round(b.recovery_s)
                                             if b.recovery_s is not None else None),
                         "rate_episodes": b.episodes,
                         # Two different numbers on purpose: ttft_p50_ms is
                         # time to first token (only streaming measures it, and
                         # it is what weighs in the score); latency_p50_ms is
                         # the complete round-trip (non-streaming and probes).
                         # They used to share a column and the average meant
                         # nothing.
                         "ttft_p50_ms": m.ttft_p50_ms,
                         "latency_p50_ms": m.latency_p50_ms,
                         # The CHAT cooldown, which is what this field has always
                         # meant and what `auto` routes on.
                         "cooldown_until": m.cooldown_until,
                         # Every live punishment for this route, by the capability
                         # it is evidence ABOUT ("*" = the whole route). Since
                         # punishments became scoped, the single number above can
                         # no longer tell the whole story: a grok agent with an
                         # empty image bucket is unavailable for images and
                         # perfectly healthy for chat, and "why did my image
                         # request 503 while chat works" has to be answerable from
                         # this table.
                         "cooldowns": state.proxy.cooldowns.scopes(r.key),
                         "tools": r.capabilities.tools, "vision": r.capabilities.vision,
                         # Two different axes on purpose: `vision` is image
                         # INPUT, `images` is image OUTPUT. An operator asking
                         # "why did my prompt 503?" needs to see which routes
                         # POST /v1/images/generations could even consider.
                         "images": r.capabilities.images,
                         "context": r.capabilities.context})
        return {"routes": rows}

    @app.get("/v1/usage", **USAGE_DOCS)
    def usage(request: Request, x_api_key: str | None = Header(None),
              authorization: str | None = Header(None)):
        key = require_api_key(x_api_key, authorization, request)
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return {"day": day, "paid_today": state.store.paid_usage(key, day),
                "cap": state.daily_paid_cap}

    @app.get("/health", **HEALTH_DOCS)
    def health(request: Request):
        # Bounded by source address: no key is required here (Coolify uses it as
        # the container health check) and it is the MOST EXPENSIVE endpoint in
        # the service -- it recalculates every route's metrics on every call,
        # ~0.5s across 53 routes. Unlimited and public is the wrong combination
        # for that.
        _limit_by_ip(request)
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
        # to drop `/health` to "down" for ALL keys, surviving a process restart
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
        now = time.time()
        active = state.store.active_routes()
        metrics = _metrics(state, now)

        def _alive(r) -> bool:
            m = metrics.get(r.key, NEUTRAL_METRICS)
            return (m.cooldown_until <= now
                    and state.store.has_liveness_evidence(r.key, now))

        available = [r for r in active if _alive(r)]
        free = [r for r in available if r.tier == "free"]
        if free:
            status = "ok"
        elif available:
            status = "degraded"
        else:
            status = "down"
        code = 200 if status == "ok" else 503
        return JSONResponse({"status": status, "active_routes": len(active),
                             "available_routes": len(available),
                             "free_available": len(free)}, status_code=code)

    return app


def _no_routes(active: list, request, metrics: dict, now: float,
               cap_reached: bool) -> None:
    """Raise the right error when the chain of attempts comes back empty.

    Design section 9 separates two situations the previous version conflated into
    a single 400:

    - **No route can satisfy the request** (capabilities, vision, a context
      nobody has) -> `400`. That genuinely is a client error.
    - **There are routes that could serve but they are all down or in cooldown**
      (including the case "this key exceeded its daily paid cap") -> `503`, with
      `next_release`.

    Why it matters: `order_routes` filters out cooldowns, so in ANY outage of the
    free tiers -- the expected failure, not an exotic one -- the list arrived
    empty and a 400 went out. Every SDK and every alerting layer reads 400 as
    "your request is malformed": they do not retry and they wake nobody.

    `x_allow_paid: false` is NOT considered here: it is a policy of the
    caller, not a capability missing from the pool. A client that forbids paid
    routes and runs out of live free ones is in the unavailability case (503,
    retryable), not in the invalid-request one.
    """
    compatible = compatible_routes(active, request)
    if not compatible:
        raise HTTPException(400, {
            "message": "no route satisfies the request",
            "request": request.as_wire(),
            "active_routes": len(active),
        })
    releases = [metrics[r.key].cooldown_until for r in compatible
                if r.key in metrics and metrics[r.key].cooldown_until > now]
    raise HTTPException(503, {
        "message": "every route that could serve is down or in cooldown",
        "request": request.as_wire(),
        "compatible_routes": len(compatible),
        # When the FIRST of them is released. None = none is being punished
        # (they are excluded for another reason, e.g. the paid cap).
        "next_release": min(releases) if releases else None,
        "paid_cap_reached": cap_reached,
    })


def _similar_ids(requested: str, active: list) -> list[str]:
    # Round 7, LOW from the gate: the caller of the live-404 (above) passes
    # `active` UNFILTERED -- that `requested` id is STILL in the local catalogue
    # (that is the entire premise of that case: still in the catalogue, but the
    # real provider no longer has it). Without excluding it here,
    # `get_close_matches` finds IT ITSELF as the most obvious "match" (distance
    # zero) and the client reads `"the model 'a:free' no longer exists"` with
    # `suggestions: ['a:free', ...]`. It is excluded HERE, in the function, and
    # not in each caller: a suggestion list must never be able to suggest the very
    # id just declared dead.
    candidates = [r.model_id for r in active if r.model_id != requested]
    return difflib.get_close_matches(requested, candidates, n=3, cutoff=0.3)


def _iso(at: float | None) -> str | None:
    if at is None:
        return None
    return datetime.fromtimestamp(at, timezone.utc).isoformat().replace("+00:00", "Z")


def _metrics(state: State, now: float, capability: str = CHAT) -> dict:
    """Route metrics with the proxy's punishments merged in, for ONE capability.

    `capability` matters because a punishment is scoped to what it is evidence
    about (see proxy.ALL_CAPABILITIES): a rate limit on image generation must not
    remove a route from CHAT routing, and a grok agent carries 999 chat requests
    an hour against a handful of images a day, so the two are nowhere near the
    same resource. `cooldowns.until` folds the route-wide punishment together with
    the capability-specific one, so callers never have to remember both exist.
    """
    base = state.store.metrics()
    for key in list(state.proxy.cooldowns):
        until = state.proxy.cooldowns.until(key, capability)
        if key in base:
            # `replace` and not a positional `type(m)(...)`: rebuilding by hand
            # leaves out any new Metrics field (e.g. quality_measured_at), and a
            # route in cooldown would start looking "never measured".
            base[key] = replace(base[key], cooldown_until=until)
    return base
