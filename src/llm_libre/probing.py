import calendar
import logging
import os
import time

import httpx

from llm_libre.assets import RETENTION_S as ASSET_RETENTION_S
from llm_libre.catalog import normalize
from llm_libre.contract import parse_health
from llm_libre.models import Route
from llm_libre.providers import Provider, contract_url, fixed_routes, join_path
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

# Returned by `_read_contract` when this provider must be left untouched for the
# whole sweep: neither its /health nor its last stored contract could be used.
_SKIP = object()


async def sync_catalogue(http: httpx.AsyncClient, providers: list[Provider],
                         store, now: float, notifier=None) -> int:
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

    A provider with `reads_capabilities` set (Provider.reads_capabilities) adds a
    FIFTH way to fail, ahead of all the others: its /health is fetched FIRST, before
    anything else is written for that provider. An unusable answer -- a network
    failure, a non-200, a body that is not valid JSON, or a contract whose
    `auth.mode` is "unknown" -- does NOT go straight to skipping the provider:
    the LAST GOOD CONTRACT stored for it is tried first (design section 5.1,
    "keep the previous sweep's capability values"), and when there is one the
    sync proceeds normally with those values and a warning saying they are being
    carried over. Only when there is no usable stored contract either is the
    whole provider `continue`d -- same discipline as a failing /models, for the
    same reason: rewriting this provider's fixed_models or discovered routes with
    fallback capability values would look exactly like a real measurement.

    `auth.mode: "unknown"` is grouped with the outright failures deliberately. It
    means "the proxy could not resolve its account this cycle" (design section
    3.1), and chatgpt-proxy reports it with every plan-gated capability false
    after a restart whose first vendor resolve failed. Acting on that at face
    value would drop dall-e-3, rewrite every chatgpt route to false and fire a
    "lost images" alert over a condition that clears itself -- while a capability
    boolean is only supposed to mean a DURABLE entitlement change (section 3.2).

    A /health that answers but does not speak the contract
    (`contract.parse_health` returns None) is NOT a failure in this sense: it is
    the normal, permanent state for a provider that has not adopted it, so the
    sync proceeds with `contract=None` and every capability falls back to
    `default_capabilities`/`exceptions` exactly as before. A provider without
    `reads_capabilities` makes no /health request at all -- that is the default
    for every provider except the ones that opt in.

    Each of those five paths LEAVES A WARNING with the provider and the reason.
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
        headers = dict(p.extra_headers)
        if p.api_key.strip():
            headers["Authorization"] = "Bearer " + p.api_key
        # The capability contract, BEFORE anything is written for this provider.
        # `_SKIP` means neither this sweep's /health nor the stored one could be
        # used, so the provider is left entirely alone and the previous
        # catalogue survives -- exactly what a failing /models already does.
        contract = None
        if p.reads_capabilities:
            contract = await _read_contract(http, p, headers, store, now, notifier)
            if contract is _SKIP:
                continue
        if p.fixed_models:
            routes = fixed_routes(p, contract=contract)
            store.upsert_routes(routes, now, deactivate_missing=True, provider=p.id)
            total += len(routes)
            # No `continue` here: a provider may declare fixed routes (e.g. a
            # stable image-generation model) AND also discover its chat models
            # dynamically via models_path. Both passes use the same `now`, so
            # fixed routes (last_seen=now) survive the dynamic pass's
            # `deactivate_missing` sweep (which removes last_seen < now only).
        if not p.models_path:
            continue
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
                              measured_rates=measured_rates, contract=contract)
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


async def _read_contract(http, p: Provider, headers: dict, store, now: float,
                         notifier):
    """This sweep's capability contract for `p`, or the last good one, or _SKIP.

    Three outcomes, and each is a different statement:

    - a `ProviderContract` -- either freshly read (and then persisted, with the
      alerts fired) or CARRIED OVER from the last good sweep when this one could
      not be read. Carrying over is design section 5.1: freezing what is known
      beats erasing it, and it is what keeps one bad /health from freezing a
      provider's whole catalogue instead of just its capability values.
    - `None` -- the proxy answered but does not speak the contract. Normal and
      permanent for a proxy that has not adopted it: the caller falls back to
      `default_capabilities`/`exceptions`, exactly as before this existed.
    - `_SKIP` -- unusable now AND nothing usable stored. The caller leaves the
      provider entirely alone.

    A contract whose `auth.mode` is "unknown" AND was actually reported by the
    proxy (`auth.resolved` is True) counts as unusable: the proxy is saying it
    could not resolve its own account this cycle, so its plan-gated booleans
    describe a lookup failure rather than a durable entitlement change (design
    sections 3.1 and 3.2). Neither persisted nor alerted on -- a "lost images"
    alert for a transient vendor blip is exactly the kind of noise that gets an
    alert channel muted.

    An absent or malformed `auth` block also parses to `mode="unknown"`, but
    with `auth.resolved` False -- that is a proxy with nothing to say about
    accounts at all (grok has no plan tiers and no account concept), not one
    that asked its vendor and failed. Treating it the same as the resolved case
    would refuse a perfect eleven-boolean `capabilities` block over silence and,
    with nothing stored yet, freeze that provider's catalogue forever: the exact
    failure this contract exists to prevent, arriving by a different route.
    """
    reason = ""
    fresh = doc = None
    try:
        r = await http.get(contract_url(p.base_url), headers=headers, timeout=15.0)
        r.raise_for_status()
        doc = r.json()
    except (httpx.HTTPError, ValueError) as e:
        reason = f"could not read /health ({type(e).__name__}: {e})"
    else:
        fresh = parse_health(p.id, doc)
        if fresh is None:
            log.warning(
                "capabilities of %s: /health does not implement the "
                "capability contract; falling back to providers.yaml.", p.id)
            return None
        if fresh.auth.mode == "unknown" and fresh.auth.resolved:
            reason = ("/health reports auth.mode='unknown', so the proxy could "
                      "not resolve its account this cycle and its capability "
                      "booleans state nothing durable")
            fresh = None
    if reason:
        stored = parse_health(p.id, store.get_contract(p.id))
        if stored is not None and stored.auth.mode == "unknown":
            stored = None
        if stored is None:
            log.warning(
                "capabilities of %s: %s, and no usable contract from an earlier "
                "sweep is stored. Keeping the previous catalogue for this "
                "provider.", p.id, reason)
            return _SKIP
        log.warning(
            "capabilities of %s: %s. Carrying over the capability values from "
            "the last good sweep; nothing is written to the contract, so "
            "/health keeps showing when they were last confirmed.",
            p.id, reason)
        return stored
    # `_announce_changes` compares against the STORED document, so it must run
    # BEFORE the new one replaces it. What it returns is its own bookkeeping
    # (see Task 12), merged in so a single write persists both the contract and
    # the alert state.
    extra = _announce_changes(store, p.id, fresh, notifier)
    store.put_contract(p.id, {**doc, **extra}, now)
    return fresh


# How close to `expires_at` the operator is told. A week is enough to renew a
# subscription deliberately, and short enough that the warning still means
# something when it arrives.
EXPIRY_WARNING_S = 7 * 86400


def _announce_changes(store, provider: str, contract, notifier) -> dict:
    """Alert on the two transitions worth waking someone for.

    Returns the bookkeeping `sync_catalogue` merges into the document it is
    about to store. Nothing is written here: one writer per document is what
    keeps "the alert fired" and "the contract we saw" from racing each other
    within a single sweep.

    A capability turning OFF is the event this whole design exists for: a plan
    lapsed, a token was revoked, an account was downgraded. Left undetected it
    shows up as a route that fails every request until somebody notices.

    A capability turning ON is logged, not alerted. It is good news, and good
    news does not need a notification.

    The comparison is against the PREVIOUS stored document, so it detects a
    transition rather than a state -- without that, a provider sitting at
    `images: false` would alert on every sweep, twice a day, until it was muted
    and stopped being read at all. The first sweep for a provider alerts on
    nothing: having nothing to compare against is not the same as something
    having turned off.
    """
    if notifier is None:
        return {}
    previous = store.get_contract(provider)
    if previous is not None:
        was = previous.get("capabilities") or {}
        lost = sorted(name for name, value in contract.capabilities.items()
                      if was.get(name) is True and value is False)
        gained = sorted(name for name, value in contract.capabilities.items()
                        if was.get(name) is False and value is True)
        if gained:
            log.info("capabilities of %s: gained %s", provider, ", ".join(gained))
        if lost:
            plan = f" on plan {contract.auth.plan}" if contract.auth.plan else ""
            notifier.notify(
                f"⚠️ {provider}: lost {', '.join(lost)}. The account is "
                f"{contract.auth.mode}{plan}. Routes that needed it have left "
                f"the catalogue.")
    return _expiry_warning(previous, provider, contract, notifier)


def _expiry_warning(previous, provider: str, contract, notifier) -> dict:
    """One warning per provider per day while the subscription is nearly up.

    Per day, not per sweep: the sweep interval is configurable and a burst of
    redeploys runs several of them within minutes, so "once per sweep" is not a
    rate anybody chose. The timestamp rides inside the stored contract document
    rather than in a table of its own -- it is one float per provider whose only
    reader is this function, and `contract.parse_health` ignores unknown
    top-level keys, so it never reaches the contract itself.
    """
    expires_at = contract.auth.expires_at
    if not expires_at:
        return {}
    try:
        # timegm, not mktime: `expires_at` is UTC by the contract, and mktime
        # would read it as local time -- five hours of silent drift here.
        deadline = calendar.timegm(time.strptime(expires_at[:19], "%Y-%m-%dT%H:%M:%S"))
    except ValueError:
        log.warning("capabilities of %s: auth.expires_at=%r is not ISO 8601; "
                    "no expiry warning will be sent.", provider, expires_at)
        return {}
    remaining = deadline - time.time()
    if not 0 < remaining <= EXPIRY_WARNING_S:
        return {}
    now = time.time()
    warned_at = (previous or {}).get("_expiry_warned_at")
    if warned_at is not None and (now - warned_at) < 86400:
        # Carry the old timestamp forward: dropping it would restart the
        # once-a-day clock on the very next sweep.
        return {"_expiry_warned_at": warned_at}
    notifier.notify(
        f"⏳ {provider}: the {contract.auth.plan or 'current'} subscription "
        f"expires in {int(remaining // 86400)} day(s), on {expires_at}. When it "
        f"does, its plan-gated capabilities turn themselves off.")
    return {"_expiry_warned_at": now}


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
    await sync_catalogue(state.http, state.providers, state.store, now,
                         notifier=state.proxy.notifier)
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
