import logging
import os
import time

import httpx

from llm_libre.assets import RETENTION_S as ASSET_RETENTION_S
from llm_libre.catalog import normalize
from llm_libre.models import Route
from llm_libre.providers import Provider, fixed_routes, join_path
from llm_libre.proxy import PING
from llm_libre.quality_suite import CASES, evaluate

log = logging.getLogger(__name__)

# `PING` has lived in proxy.py since round 8: the proxy also fires probes of its
# own (on demand, see SUSPICION_THRESHOLD there) and needs the SAME fixed payload as
# this periodic probe so the two are, literally, the same request -- it is
# imported from there instead of duplicated. It is re-exported here (the import
# above) so anyone already doing `from llm_libre.probing import PING` keeps
# working.

# How often the quality battery runs (it burns free quota, hence not every pass)
# and how long old telemetry is retained before being pruned.
QUALITY_EVERY_N_CYCLES = int(os.getenv("QUALITY_PROBE_EVERY_N_CYCLES")
                             or os.getenv("SONDEO_CALIDAD_CADA_N_CICLOS")
                             or "5")   # legacy name honoured, see main._env
RETENTION_DAYS = 30

# How long the battery waits between runs, measured from the last one RECORDED --
# not counted in cycles.
#
# `scheduler` starts at `counter = 0` on every process start and `cycle` ran the
# battery when `counter % QUALITY_EVERY_N_CYCLES == 0`. Since 0 % 5 == 0, every
# restart ran a full battery and the counter never survived long enough to apply
# the pacing it was written for. That is invisible while deployments are rare and
# brutal while they are not.
#
# Measured 2026-08-19 against the live deployment: the battery is meant to run
# once every ~25h. On 2026-08-18 it ran 28 times against deepseek and wrote 824
# quality rows across the catalogue -- roughly 4,120 requests to free providers in
# a single day. DeepSeek muted the account at 01:25 the next morning, for 24h.
#
# The elapsed time is IN the database (storage.last_quality_probe_at), so the
# decision is taken from evidence rather than from process bookkeeping: it holds
# across restarts, crashes and redeploys, none of which an in-memory counter
# survives. `counter` still gates the cycle for callers that pass it, so the
# existing semantics are unchanged for anyone who never restarts.
HEALTH_INTERVAL_S = float(
    os.getenv("HEALTH_PROBE_HOURS") or os.getenv("SONDEO_SALUD_HORAS") or "5") * 3600
QUALITY_INTERVAL_S = QUALITY_EVERY_N_CYCLES * HEALTH_INTERVAL_S

# The same lesson as QUALITY_INTERVAL_S above, one level down: the HEALTH sweep
# had no equivalent guard, so it ran unconditionally on every process start.
#
# Measured 2026-08-19 against the live deployment. Four deployments landed
# between 22:34 and 22:41 (two commits, two deployments each), and the telemetry
# shows four full sweeps at 22:36:04 (truncated mid-catalogue, the container was
# killed while probing), 22:37:55, 22:42:48 and 22:46:21. Kilo's free tier took
# 33 requests in eight and a half minutes -- 11 free routes, three complete
# sweeps -- and refused the fourth: `kilo/poolside/laguna-xs-2.1:free` returned
# 429 and left routing, which is what the operator saw on Telegram at 22:46.
#
# `HEALTH_PROBE_HOURS` cannot fix that: it governs the interval BETWEEN sleeps,
# and a burst of redeploys never reaches a sleep. What bounds a burst is a floor
# on how recently the catalogue was swept, read from the database so it survives
# the restart that caused the problem.
#
# Fifteen minutes: long enough to absorb a deploy burst (the four above fit in
# twelve), short enough that a single unlucky deploy does not leave the ranking
# blind for long. Capped at the interval itself so it can never suppress a sweep
# that is legitimately due -- with HEALTH_PROBE_HOURS set to minutes for
# debugging, the floor follows it down instead of overriding it.
HEALTH_FLOOR_S = min(900.0, HEALTH_INTERVAL_S)


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
    without the sweep its routes would stay `active=1` forever: visible in
    `GET /v1/models` and `GET /v1/ranking`, and eligible as candidates that would
    always fail.
    """
    known = {p.id for p in providers}
    # What we have MEASURED about allowances nobody publishes, for catalog's
    # scarcity check. Only the routes whose limit was actually observed are
    # passed: `RateBudget.floor` is bounded by our own demand, so handing it over
    # would demote routes for being idle -- see _is_scarce.
    measured_rates = {k: b.per_hour for k, b in store.rate_budgets(now).items()
                      if b.measured}
    deactivated = store.deactivate_unregistered_providers(known)
    if deactivated:
        log.warning(
            "catalogue: %d route(s) deactivated because their provider is no "
            "longer in providers.yaml (they are not deleted -- the history is "
            "used to detect renames, see Storage.upsert_routes)", deactivated)
    total = 0
    for p in providers:
        if p.fixed_models:
            routes = fixed_routes(p)
            store.upsert_routes(routes, now, deactivate_missing=True, provider=p.id)
            total += len(routes)
            # No `continue` here: a provider may declare fixed routes (e.g. a
            # stable image-generation model) AND also discover its chat models
            # dynamically via models_path. Both passes use the same `now`, so
            # fixed routes (last_seen=now) survive the dynamic pass's
            # `deactivate_missing` sweep (which removes last_seen < now only).
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
            fresh = normalize(p.id, r.json(), p.priority, p.default_capabilities,
                              p.exceptions, emulates_tools=p.emulates_tools,
                              measured_rates=measured_rates)
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


async def probe_health(proxy, store, routes: list[Route], now: float) -> set[str]:
    """...and returns the keys of the routes that ANSWERED.

    That return value is what `probe_quality` gates on: one request has just
    established whether a route is reachable at all, and spending five more on one
    that is not is wasteful in two different ways. See probe_quality.
    """
    # Same as probe_quality: PAID routes are NOT probed (design section 8). This
    # function receives active_routes(), which includes minimax/MiniMax-M3, and
    # every pass used to hit it: ~5 billable calls a day that also bypass
    # add_paid_usage, so they show up neither in /v1/uso nor against
    # DAILY_PAID_CAP. Real money, invisible.
    answered: set[str] = set()
    for route in (r for r in routes if r.tier == "free"):
        t0 = time.monotonic()
        # HIGH 1 (round 9): without `is_probe=True` a failure here only
        # accumulated SUSPICION (round 8) -- this periodic probe runs ONCE every
        # 5h per route, far below any suspicion threshold, so it never got to
        # punish anything: 20 periodic probes against a dead route, 20
        # `sondas ok=0` rows, zero cooldowns. A probe (periodic or on demand) is
        # the only source that can punish unambiguously -- see the header comment
        # of SUSPICION_THRESHOLD in proxy.py.
        r = await proxy.complete([route], dict(PING), now, is_probe=True)
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
        # already has its own proportional punishment (Proxy._punish_429, inside
        # completar()), and it is evidence that the route is rate-limited NOW, not
        # that it is broken. Recording it as a failed health probe too would
        # confuse it with a downed route: two consecutive 429s would be enough for
        # has_liveness_evidence (round 9) to declare it dead -- a momentary
        # capacity signal is not evidence of death.
        if r.upstream_code != 429:
            store.record_probe(route.key, "health", r.status == 200, ms, 0,
                               r.upstream_code, 0, 0, now)
        if r.status == 200:
            answered.add(route.key)
    return answered


async def probe_quality(proxy, store, routes: list[Route], now: float,
                        answered: set[str] | None = None) -> None:
    """Score every free route against the battery.

    `answered` is what `probe_health` just returned: the routes that responded to
    the one-request health probe moments ago, in this same cycle. A route missing
    from it is SKIPPED, and both reasons are worth stating because they are
    different problems.

    It BURNS quota that may be scarce. Measured 2026-08-19: DeepSeek had muted the
    account, and the battery kept spending ten requests a run (five cases, two
    routes) establishing that a muted account is muted.

    And it POISONS the measurement, which is the worse of the two. A run against an
    unreachable route records 0/5, which is indistinguishable from "this model
    answers badly" -- so chatgpt sat at quality 0.333 and deepseek at 0.00 while
    both were perfectly healthy models behind, respectively, a container that never
    started and an account DeepSeek had muted. Recovery then costs as many runs as
    the average is wide (QUALITY_RUNS), long after the outage itself is over.
    Skipping keeps the previous real measurement, and a stale real number beats a
    fresh fabricated zero.

    `None` (the default) probes everything, which is what every caller predating
    this parameter expects.
    """
    # Paid routes are not probed: there is no sense in spending money scoring the
    # emergency escape hatch.
    for route in (r for r in routes if r.tier == "free"):
        if answered is not None and route.key not in answered:
            log.info("quality: %s skipped, it did not answer its health probe in "
                     "this cycle -- the previous measurement is kept rather than "
                     "recording a zero the route did not earn", route.key)
            continue
        results = []
        for case in CASES:
            if case.name == "tools" and not route.capabilities.tools:
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
            # as unambiguously. Without `is_probe=True`, a battery failure fed
            # `_sospechar` (meant for CLIENT traffic) and burned quota from the
            # on-demand probe budget, which is scarce and shared with real traffic.
            r = await proxy.complete([route], body, now, is_probe=True)
            results.append((case, r.status == 200 and case.check(r.json)))
        passed, total = evaluate(results)
        store.record_probe(route.key, "quality", passed > 0, 0, 0, 200,
                           passed, total, now)


async def cycle(state, counter: int) -> None:
    """One complete probing pass, for the scheduler (Task 12) to invoke in its
    loop: it syncs the catalogue, sweeps health unless one was swept less than
    HEALTH_FLOOR_S ago, probes quality only every QUALITY_EVERY_N_CYCLES passes
    AND at most once per QUALITY_INTERVAL_S (the same free quota as real traffic),
    and prunes old telemetry at the end.

    Both spends are gated on the DATABASE, not on `counter`: the scheduler starts
    at zero on every process start, so a redeploy would otherwise re-run the whole
    pass no matter how recently the previous one ran. See HEALTH_FLOOR_S.

    It does not catch exceptions: that is the responsibility of the scheduler
    calling this in an infinite loop, which must not die because one particular
    cycle failed.

    `state` is any object with `.http`, `.providers`, `.store` and `.proxy`
    (e.g. `llm_libre.api.State`); that type is not imported here so as not to
    create a dependency from probing to api.
    """
    now = time.time()
    # The catalogue is synced BEFORE the floor is consulted, and unconditionally:
    # discovery costs no chat quota (one /models per provider) and new routes only
    # ever appear when the process restarts, which is precisely the case the floor
    # is about to short-circuit.
    await sync_catalogue(state.http, state.providers, state.store, now)
    routes = state.store.active_routes()
    # Has the catalogue been swept too recently to sweep it again? See
    # HEALTH_FLOOR_S: a burst of redeploys never reaches the scheduler's sleep, so
    # the interval alone does not bound how often this runs.
    swept_at = state.store.last_health_sweep_at()
    if swept_at is not None and (now - swept_at) < HEALTH_FLOOR_S:
        # The battery goes with it, and not merely to save more quota: it is
        # gated on `answered` -- the routes THIS cycle's health probe reached --
        # and without a sweep there is nothing to gate it on. Running it blind
        # right after a redeploy is the flood this whole guard exists to stop.
        log.info("health: sweep skipped, the catalogue was swept %.0f min ago "
                 "(floor %.0f min)", (now - swept_at) / 60, HEALTH_FLOOR_S / 60)
    else:
        # Marked BEFORE the sweep, not after. What the floor bounds is request
        # VOLUME against the providers, so what it has to record is the attempt.
        # The 2026-08-19 burst opened with a sweep the container was killed
        # inside -- the telemetry stops at grok-*, short of kilo -- and marking on
        # completion would have left that one uncounted and the next restart free
        # to sweep the whole catalogue again, which is what actually happened.
        #
        # The cost of marking early is bounded and small: a sweep that dies at
        # route 3 of 42 leaves the other 39 on the previous cycle's telemetry for
        # up to HEALTH_FLOOR_S. Reliability and liveness are both read over a
        # window, so that is staleness, not blindness.
        state.store.mark_health_sweep(now)
        # The health probe runs FIRST, and what it learned gates the battery: one
        # request has just established which routes are reachable, so the five-request
        # battery is only spent on those. See probe_quality.
        answered = await probe_health(state.proxy, state.store, routes, now)
        # Two gates, and they answer different questions. `counter` is the schedule
        # this loop was written around; `last_quality_probe_at` is what actually
        # happened, and it is the one that survives a restart -- see
        # QUALITY_INTERVAL_S.
        last = state.store.last_quality_probe_at()
        due = last is None or (now - last) >= QUALITY_INTERVAL_S
        if counter % QUALITY_EVERY_N_CYCLES == 0 and due:
            await probe_quality(state.proxy, state.store, routes, now, answered)
        elif not due:
            log.info("quality: battery skipped, it last ran %.1f h ago (interval %.1f h)",
                     (now - last) / 3600, QUALITY_INTERVAL_S / 3600)
    state.store.prune(now - RETENTION_DAYS * 86400)
    # Generated assets age out on their own schedule (assets.RETENTION_S, 30
    # days): they are bytes on a finite volume, and without this the disk grows
    # forever. Same place as the telemetry prune so there is ONE thing that
    # bounds what this service accumulates, not two that can drift apart.
    if getattr(state, "assets", None) is not None:
        state.assets.prune(now - ASSET_RETENTION_S)
