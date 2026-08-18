import logging
import os
import time

import httpx

from llm_libre.catalog import normalize
from llm_libre.modelos import Ruta
from llm_libre.providers import Provider, fixed_routes, join_path
from llm_libre.proxy import PING
from llm_libre.quality_suite import CASES, evaluate

log = logging.getLogger(__name__)

# `PING` has lived in proxy.py since round 8: the proxy also fires probes of its
# own (on demand, see UMBRAL_SOSPECHA there) and needs the SAME fixed payload as
# this periodic probe so the two are, literally, the same request -- it is
# imported from there instead of duplicated. It is re-exported here (the import
# above) so anyone already doing `from llm_libre.probing import PING` keeps
# working.

# How often the quality battery runs (it burns free quota, hence not every pass)
# and how long old telemetry is retained before being pruned.
QUALITY_EVERY_N_CYCLES = int(os.getenv("SONDEO_CALIDAD_CADA_N_CICLOS", "5"))
RETENTION_DAYS = 30


async def sync_catalogue(http: httpx.AsyncClient, providers: list[Provider],
                         store, now: float) -> int:
    """Refresh the catalogue, provider by provider.

    Removing routes that no longer appear is decided PER PROVIDER, not for the
    whole sync: each provider that answers correctly is persisted (and the routes
    that vanished from ITS catalogue are deactivated) right there, using the
    `provider=` scope of Storage.upsert_routes -- without that scope, an UPDATE of
    "whatever was not seen gets deactivated" not filtered by provider would also
    switch off the routes of providers unrelated to this pass (their
    visto_por_ultima_vez is always older than `now`). Another provider being
    broken or empty in the same pass must not hold that removal back: the case
    this project exists to detect is precisely a model that disappeared from its
    provider, and it cannot depend on every other provider having a perfect day.

    A provider that fails (network, non-200 status, an unparseable or
    unexpectedly shaped body, or a 200 with zero usable models -- more likely a
    broken provider than a genuinely empty catalogue) is not touched at all:
    better to keep what was already known about IT than to erase it.

    Each of those four paths LEAVES A WARNING with the provider and the reason.
    Without that, keeping the old catalogue -- which is the right thing -- becomes
    indistinguishable from everything working: that provider's catalogue freezes
    forever and nobody finds out. This is exactly the layer that exists so a
    catalogue cannot rot without warning (design section 1).

    Before the per-provider loop, a separate sweep
    (`Storage.deactivate_unregistered_providers`) switches off the routes of any
    provider NO LONGER in `providers` -- necessary because the loop below can only
    remove, via its scope, what is STILL in the registry; a provider taken out of
    the YAML (e.g. `openrouter` with no `OPENROUTER_API_KEY`, whose routes all
    ended up in cooldown from 401s) never passes through that loop again, so
    without the sweep its routes would stay `activa=1` forever: visible in
    `GET /v1/models` and `GET /v1/ranking`, and eligible as candidates that would
    always fail.
    """
    known = {p.id for p in providers}
    deactivated = store.deactivate_unregistered_providers(known)
    if deactivated:
        log.warning(
            "catalogue: %d route(s) deactivated because their provider is no "
            "longer in proveedores.yaml (they are not deleted -- the history is "
            "used to detect renames, see Storage.upsert_routes)", deactivated)
    total = 0
    for p in providers:
        if p.fixed_models:
            routes = fixed_routes(p)
            store.upsert_routes(routes, now, deactivate_missing=True, provider=p.id)
            total += len(routes)
            continue
        if not p.models_path:
            continue
        headers = dict(p.extra_headers)
        if p.api_key.strip():
            headers["Authorization"] = "Bearer " + p.api_key
        try:
            # join_path parses and rebuilds, it does not concatenate raw text:
            # see its docstring in providers.py for the bug that avoids.
            r = await http.get(join_path(p.base_url, p.models_path),
                               headers=headers, timeout=30.0)
        except httpx.HTTPError as e:
            log.warning("catalogue of %s: could not query %s (%s: %s). "
                        "Keeping the previous catalogue.",
                        p.id, p.models_path, type(e).__name__, e)
            continue
        if r.status_code != 200:
            log.warning("catalogue of %s: %s answered HTTP %s. "
                        "Keeping the previous catalogue.",
                        p.id, p.models_path, r.status_code)
            continue
        try:
            fresh = normalize(p.id, r.json(), p.prioridad, p.default_capabilities,
                              p.exceptions, emulates_tools=p.emulates_tools)
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            # A non-JSON body (ValueError/JSONDecodeError) or JSON of an
            # unexpected shape -- e.g. an auth error disguised as a 200, which
            # leaves normalize() iterating over things that are not model dicts
            # (AttributeError/KeyError/TypeError). One broken provider must not
            # take down everyone else's sync.
            log.warning("catalogue of %s: could not interpret the response from %s "
                        "(%s: %s). Keeping the previous catalogue.",
                        p.id, p.models_path, type(e).__name__, e)
            continue
        if not fresh:
            # A 200 with zero usable models does not authorise switching off what
            # was already known about this provider: it is treated like any other
            # failure of THIS particular provider, without affecting the others.
            log.warning("catalogue of %s: HTTP 200 with zero usable models. "
                        "More likely a broken provider than a genuinely empty "
                        "catalogue: keeping the previous catalogue.", p.id)
            continue
        # This provider answered correctly: it is persisted NOW, with its own
        # scope, without waiting to learn what happened to the rest of the list.
        store.upsert_routes(fresh, now, deactivate_missing=True, provider=p.id)
        total += len(fresh)
    return total


async def probe_health(proxy, store, routes: list[Ruta], now: float) -> None:
    # Same as probe_quality: PAID routes are NOT probed (design section 8). This
    # function receives active_routes(), which includes minimax/MiniMax-M3, and
    # every pass used to hit it: ~5 billable calls a day that also bypass
    # add_paid_usage, so they show up neither in /v1/uso nor against
    # TOPE_PAGO_DIARIO. Real money, invisible.
    for route in (r for r in routes if r.tier == "gratis"):
        t0 = time.monotonic()
        # HIGH 1 (round 9): without `es_sonda=True` a failure here only
        # accumulated SUSPICION (round 8) -- this periodic probe runs ONCE every
        # 5h per route, far below any suspicion threshold, so it never got to
        # punish anything: 20 periodic probes against a dead route, 20
        # `sondas ok=0` rows, zero cooldowns. A probe (periodic or on demand) is
        # the only source that can punish unambiguously -- see the header comment
        # of UMBRAL_SOSPECHA in proxy.py.
        r = await proxy.completar([route], dict(PING), now, es_sonda=True)
        ms = int((time.monotonic() - t0) * 1000)
        # `ttft_ms=0`, not `ms`: this probe is non-streaming, so what it measured
        # is a complete round-trip and not a time-to-first-token. Writing it into
        # the ttft column mixed two different magnitudes into one p50 (see the
        # header comment of storage.py).
        #
        # `http_code` stores the PROVIDER's status, not the `estado` the gateway
        # synthesises. With the synthetic 503, the `sondas` table could not tell
        # "the provider is down" from "the provider said 200 and came back empty"
        # -- exactly the difference needed to diagnose why the 8-token ping was
        # killing healthy routes.
        #
        # Round 10, small fix: a 429 against the probe is NOT recorded here -- it
        # already has its own proportional punishment (Proxy._castigar_429, inside
        # completar()), and it is evidence that the route is rate-limited NOW, not
        # that it is broken. Recording it as a failed health probe too would
        # confuse it with a downed route: two consecutive 429s would be enough for
        # has_liveness_evidence (round 9) to declare it dead -- a momentary
        # capacity signal is not evidence of death.
        if r.codigo_upstream != 429:
            store.record_probe(route.clave, "salud", r.estado == 200, ms, 0,
                               r.codigo_upstream, 0, 0, now)


async def probe_quality(proxy, store, routes: list[Ruta], now: float) -> None:
    # Paid routes are not probed: there is no sense in spending money scoring the
    # emergency escape hatch.
    for route in (r for r in routes if r.tier == "gratis"):
        results = []
        for case in CASES:
            if case.name == "tools" and not route.capacidades.tools:
                # This route does not declare tool support (see Capacidades, it
                # comes from /models or from the fixed-models YAML): asking for it
                # anyway would only burn free quota for a guaranteed failure, and
                # counting it as a failed case would conflate "does not promise
                # this capability" with "promised it and got it wrong" -- two
                # different things that must not look alike in the quality score.
                # The whole case is skipped: the proxy is not called and it counts
                # neither toward passed nor toward total.
                continue
            body = dict(case.body)
            # Round 10, small fix: the same wiring hole as HIGH 1 (round 9), one
            # function further along. `case.body` is as gateway-authored as `PING`
            # -- they are fixed CASES from quality_suite.py, never something a real
            # client writes -- so a failure here is evidence about the route just
            # as unambiguously. Without `es_sonda=True`, a battery failure fed
            # `_sospechar` (meant for CLIENT traffic) and burned quota from the
            # on-demand probe budget, which is scarce and shared with real traffic.
            r = await proxy.completar([route], body, now, es_sonda=True)
            results.append(r.estado == 200 and case.check(r.json))
        passed, total = evaluate(results)
        store.record_probe(route.clave, "calidad", passed > 0, 0, 0, 200,
                           passed, total, now)


async def cycle(state, counter: int) -> None:
    """One complete probing pass, for the scheduler (Task 12) to invoke in its
    loop: it syncs the catalogue, probes health always, probes quality only every
    QUALITY_EVERY_N_CYCLES passes (the same free quota as real traffic) and prunes
    old telemetry at the end.

    It does not catch exceptions: that is the responsibility of the scheduler
    calling this in an infinite loop, which must not die because one particular
    cycle failed.

    `state` is any object with `.http`, `.proveedores`, `.almacen` and `.proxy`
    (e.g. `llm_libre.api.Estado`); that type is not imported here so as not to
    create a dependency from probing to api.
    """
    now = time.time()
    await sync_catalogue(state.http, state.proveedores, state.almacen, now)
    routes = state.almacen.active_routes()
    await probe_health(state.proxy, state.almacen, routes, now)
    if counter % QUALITY_EVERY_N_CYCLES == 0:
        await probe_quality(state.proxy, state.almacen, routes, now)
    state.almacen.prune(now - RETENTION_DAYS * 86400)
