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

COOLDOWN_BASE_S = 60.0   # backoff for a route punished by a PROBE (our own,
                         # confirmed) or by direct paid failures -- see
                         # _punish. Round 9: a 429 NO LONGER uses this (see
                         # _punish_429 and COOLDOWN_429_*, below) -- a
                         # confirmed probe is stronger evidence than a plain
                         # "back off" from the provider, and deserves a
                         # backoff that DOES escalate if the route stays broken.
COOLDOWN_CAP_S = 3600.0
TIMEOUT_S = 90.0

# The health PING shares the battery's cap (SHORT_TOKEN_BUDGET) for the SAME
# reason, and cannot lag behind when that number moves.
#
# It used to be `max_tokens: 8`, four times less than the 32 the battery had
# already proven insufficient. While an empty 200 counted as success that was
# harmless; since a 200 with no answer inside is (rightly) a FAILED attempt, a
# ping that leaves the model no room to think MANUFACTURES the failure it claims
# to measure: the probe declares a healthy route dead. Measured against Kilo with
# max_tokens=8, one pass over the 11 free routes returned 5 healthy; among the
# "dead" ones were cohere/north-mini-code:free -- the one that serves `auto` on a
# cold start -- and tencent/hy3:free, which scores 5/5 in the battery.
#
# The damage is not just a pessimistic /health: `_reliability` looks at the last
# 50 observations and probes are ~125 a day, so real traffic never gets to
# contradict them. The ranking would end up ordering by whoever answers shortest,
# which is exactly the premise this project eliminated.
#
# It lives HERE (round 8), not in probing.py: the proxy is now ALSO firing probes
# of its own (see SUSPICION_THRESHOLD below), so the fixed payload that makes them
# unambiguous has to live where it is used. probing.py imports it from here
# (`from llm_libre.proxy import PING`) so the PERIODIC probe and the ON-DEMAND
# probe stay, literally, the same request.
PING = {"messages": [{"role": "user", "content": "ping"}],
        "max_tokens": SHORT_TOKEN_BUDGET, "temperature": 0}

# Round 8. Six rounds of the same mechanism (retryability, a list of codes, the
# list with the default inverted, response-level attribution, chain-level
# attribution) each fell to a new vector -- and the last two (round 8) were
# ESCAPE HATCHES OF ROUND 7'S OWN DESIGN: a single-route chain (the client forces
# it with an explicit `model` or with `x_min_contexto`; `/v1/ranking` publishes
# `contexto` per route, so no internal knowledge is needed) and the `if emitido:`
# branch of complete_stream (with no sibling to compare against, and no
# chain-length check). When the leaks are in the EXCEPTIONS you wrote on purpose,
# the axis is wrong, not under-enumerated.
#
# The change: A REAL CLIENT'S TRAFFIC NEVER EXCLUDES A ROUTE DIRECTLY. It can
# only accumulate SUSPICION -- which excludes nothing and is not visible anywhere
# that matters. Crossing the threshold schedules OUR OWN PROBE, with the same
# fixed payload (`PING`, above) the periodic probe in `probing.py` already uses.
# That probe -- never the client's request -- decides: if it fails, the route is
# punished (same backoff/cap as always, `_punish`); if it passes, the suspicion is
# cleared and the route stays in rotation.
#
# Why a probe CAN exclude unambiguously and a real response NEVER can: the
# probe's payload is written by THE GATEWAY. There is nothing to attribute -- if
# the probe fails, the route is broken, full stop. None of the vectors from the
# six previous rounds (flagged content, a giant prompt, a reasoning model burning
# its budget, a short chain, a streaming early-return) can touch a probe's
# payload, because the client never writes it.
#
# `429` remains the exception, and remains UNTOUCHED: it is the provider itself
# saying "back off" -- evidence about the route as unambiguous as a probe,
# without needing one to confirm it. Its backoff is already proven; this round
# does not touch it.
SUSPICION_THRESHOLD = 3  # how many CONSECUTIVE REAL-TRAFFIC failures (round 9:
                         # no longer "within a time window" -- see finding HIGH 3
                         # below) are needed to schedule our own probe against
                         # that route. The same number as round 7's old
                         # TOPE_FALLOS_SEGUIDOS -- not sacred, but it keeps the
                         # "three in a row" intuition the spec/README already use
                         # elsewhere.

# Round 9, HIGH 3 from the gate. The previous round counted "3 failures within the
# last 10 minutes" -- but that means traffic slower than 1 failure every ~200s
# NEVER gathers three inside the window before the first one decays: measured, 80
# consecutive failures spaced 301s apart triggered NO probe at all; at 300s (the
# window's exact boundary) they did. A dead route in a low-traffic deployment
# stays first in `order_routes` forever -- round 1's original defect, back again.
#
# Suspicion is now a COUNTER of consecutive failures, with no time window: it
# resets to 0 on any success (see the `pop` in complete()/complete_stream()), no
# matter what happened in between. It does not evaporate "because the service is
# quiet" -- if there is never a success, three failures SEPARATED BY HOURS are
# still three consecutive failures. This reopens none of the eight previous
# rounds' vectors: those were all about DIRECT EXCLUSION by client traffic, and
# suspicion never excludes anything on its own -- the PROBE (gateway payload,
# never the client's) still has to confirm. The real protection against
# "unrelated incidents adding up" was always the reset-on-any-success, not the
# clock: two genuinely separate incidents almost always have a real success
# between them.
ON_DEMAND_PROBE_LIMIT_S = 60.0   # at most ONE on-demand probe per route every
                                 # 60s, no matter how many suspicious requests
                                 # arrive in the meantime. Justification for the
                                 # number: a probe burns the SAME free quota as
                                 # real traffic (same endpoint, same per-minute
                                 # limit from the provider), so the extra cost ONE
                                 # hostile client can impose on ONE route is
                                 # capped at, at most, 60 extra requests per hour
                                 # -- regardless of how many requests the client
                                 # sends (they could send thousands and the cap is
                                 # the same). And because that probe's payload
                                 # belongs to the gateway, the client can never
                                 # make that extra request FAIL: a healthy route
                                 # always passes the probe, so the best possible
                                 # outcome for the attacker is burning that fixed
                                 # quota without ever bringing a healthy route
                                 # down.

# Round 9, MEDIUM 6 from the gate: the limit above is PER ROUTE, but the AGGREGATE
# was unbounded -- with N routes in the catalogue (N is not controlled by the
# operator, it comes from the LIVE catalogue), the worst case is N x 60
# probes/hour. Measured: 11 routes, 15,840 extra requests a day against the SAME
# free quota real traffic needs. A GLOBAL cap, independent of N: at most
# `GLOBAL_PROBE_LIMIT_PER_MINUTE` on-demand probes start per minute, across ALL
# routes. At 5/min (300/hour, 7200/day) a reference-sized catalogue (11 routes)
# is fully investigated within a couple of minutes even if all 11 are suspicious
# at once, and no catalogue size can grow the aggregate cost beyond that.
GLOBAL_PROBE_LIMIT_PER_MINUTE = 5
GLOBAL_PROBE_WINDOW_S = 60.0

# Round 9, MEDIUM 7 from the gate ("the 429 is the only lever the client has
# left"). A 429 DOES still punish immediately -- it is the provider itself saying
# "back off", as unambiguous as a probe -- but until now it reused _punish's
# exponential backoff (up to COOLDOWN_CAP_S=3600s). Measured: 12 requests from ONE
# key were enough to cool all 3 routes of a small catalogue via real 429s, another
# key fell to 503, /health went to "caido" -- over a rate-limit window the
# provider ITSELF typically resets in seconds to a minute, not in an hour. The 429
# remains the ONLY lever a client's traffic has to affect a route directly
# (justified: it is unambiguous evidence about the route, it does not need a probe
# to confirm it) -- but the DURATION has to be proportional to what is actually
# happening (a rate-limit window), not escalate toward an hour-long exclusion
# across ALL keys. The provider's `Retry-After` is respected when it sends one (it
# is the most precise source possible); when it does not, a short default is used;
# either way it is capped (`COOLDOWN_429_MAX_S`) so an absurd `Retry-After` (or a
# malicious one, via a compromised provider) cannot reopen the door to an
# hours-long exclusion.
COOLDOWN_429_DEFAULT_S = 30.0
COOLDOWN_429_MAX_S = 300.0

# Round 10, MEDIUM from the gate: `_suspect_paid` (direct punishment for paid
# routes, see below) reused `_punish`'s exponential backoff -- the SAME defect the
# 429 had before the constant above, in another place. `_punish` exists for
# CONFIRMED PROBES; a paid punishment has no probe behind it (by design -- see
# _suspect_paid), so escalating its duration as if it did has no basis. Measured
# through the real API: 60->120->240->480->960->1920->3600s in barely 24 requests
# from one key, with the paid fallback out for ALL keys for an hour -- ~96
# requests/day are enough to sustain the exclusion. Flat, no escalation -- same as
# _punish_429.
PAID_DIRECT_COOLDOWN_S = 60.0

# Task 13 review, finding 2. Only a 429 used to punish (with exponential backoff,
# see _punish): a 500, a timeout or a network error NEVER left a cooldown, so a
# persistently broken or HUNG route kept being tried on every request, ahead of
# healthy routes according to its priority, forever -- with TIMEOUT_S=90 that is
# up to 5*90s=450s per request on the longest chain, and /health stays "ok" as
# long as ONE route is alive. `blog` is a saturated machine: hung-not-refused is
# the realistic failure mode, not a clean 500. Round 8 still solves this, only now
# the route is excluded when ITS OWN probe confirms it is broken (above), not when
# the client's failure count hits a cap.


# How many chunks with nothing useful in them (the initial role, finish_reason,
# trimmed reasoning) are held back before being released. It exists so a stream
# that has NOT YET delivered content can fail over cleanly -- if those chunks had
# already gone out, switching routes would mix two responses. A stream that emits
# nothing but reasoning can be very long, so the retention is capped.
PENDING_CAP = 64

# ENVELOPE keys of an SSE chunk: the ones that repeat identically (or trivially)
# in every chunk of the stream and carry no information of their own. They exist
# so we can ask "besides the text, does this chunk carry anything?" while looking
# at the WHOLE chunk without the envelope always answering yes.
#
# The whole chunk has to be examined because in OpenAI's real protocol
# `finish_reason` is a SIBLING of `delta`, not a key inside it, and the `usage`
# chunk (stream_options.include_usage) arrives with `choices: []`. A guard looking
# only at `delta` discards both silently -- which is data loss in a contract whose
# premise is "change only base_url". It did not bite with Kilo or OpenRouter
# because both send `role` in every delta, but it does bite with a strict provider
# (MiniMax's OpenAI dialect, or the Groq/Cerebras the design plans to add).
#
# And the envelope has to be EXCLUDED because, if it counted, every
# already-trimmed reasoning chunk would look useful for carrying
# `id`/`model`/`index`: a pure-reasoning stream would stop failing over
# (a regression of fix B1) and the retention buffer would fill with junk.
_CHUNK_ENVELOPE = frozenset({"id", "object", "created", "model",
                             "system_fingerprint", "service_tier"})
_CHOICE_ENVELOPE = frozenset({"index"})


# The classification axis was always ATTRIBUTION ("whose fault is it, the request
# or the route"), but until the round 6 review the IMPLEMENTATION inverted it:
# "every 4xx is evidence about the request EXCEPT these seven codes". That default
# hides any code nobody has thought of yet -- the reviewer tried the mutant of
# adding 405 to the "does count" set and the entire suite stayed green, because
# nothing pinned the DEFAULT, only the list. The cost, measured: all 5 routes
# returning 405 (or 409/415/418/431/451, any 4xx nobody anticipated) left the
# client with a 503 on 100% of requests, ZERO cooldowns, and /health at 200 "ok"
# -- the same silent failure as review 3, with a different code each time.
#
# PRINCIPLE: when it cannot be known whose fault it is, IT MUST BE COUNTED -- for
# RELIABILITY (a MEASUREMENT, see Storage.record_event and _reliability). Round 8
# separates this from EXCLUSION (cooldown): see the header comment of
# SUSPICION_THRESHOLD for the opposite principle that belongs to that other
# question.
#
# That is why the default is INVERTED: a 4xx is evidence about THE ROUTE unless it
# is in this SHORT list, and every entry has to be justifiable in one line as
# GENUINE evidence about the payload -- never about the route. Everything else,
# known today or not, counts the same as a 500. `429`/`408`/`425` NO longer need
# to be in any list (they used to, on the "does not count" side): under this
# default they land on the route's side by themselves, which is a good sign that
# the shape is right.
_REQUEST_EVIDENCE_CODES = frozenset({
    400,  # Bad Request: the body could not even be parsed -- it is about the
          # SYNTAX of this request, never about whether the route works.
    413,  # Payload Too Large: the SIZE of THIS request exceeds a limit -- a
          # smaller request would work fine against the SAME route.
    422,  # Unprocessable Entity: syntactically valid but invalid for THIS
          # particular request (a parameter out of range, a wrong type) -- it says
          # nothing about whether the route is healthy.
})


def _is_client_error(codigo: int) -> bool:
    """True ONLY if `codigo` is in `_REQUEST_EVIDENCE_CODES` -- GENUINE evidence
    about the request, never about the route. Any other code (401, 403, 404, 405,
    409, or whatever a provider invents tomorrow) is evidence about the route by
    DEFAULT and counts the same as a 500 toward reliability (see
    Storage.record_event) -- see the set's header comment for the full principle
    ("when it cannot be known whose fault it is, it must be counted"). Since round
    8 this NO longer decides any cooldown directly -- see SUSPICION_THRESHOLD:
    that is a different question, with a different default.

    Only the three codes in the short list do not count toward reliability:
    counting them would turn ONE client's mistake into a bad measurement for
    EVERYONE. Verified against the real 5-route registry: three consecutive
    malformed requests (400) were enough to drag all five routes' reliability down
    before this exclusion existed."""
    return codigo in _REQUEST_EVIDENCE_CODES


def _timeout_for(proveedor) -> float:
    """`Provider.timeout_s` (default None) bounds the worst case of ONE
    particular provider -- e.g. one that can hang -- without lowering the timeout
    for everyone. None (the default, and the long-standing behaviour for anyone
    who does not declare it) uses the global TIMEOUT_S."""
    return proveedor.timeout_s if proveedor.timeout_s is not None else TIMEOUT_S


def has_answer(datos: dict) -> bool:
    """True if a 200 carries something the client can use as an answer.

    Most free models are reasoning models: they burn the completion budget
    thinking and return `200` with `finish_reason: "length"` and
    `"content": null`. Without this check that is recorded as SUCCESS, which
    means the route that failed RAISES its reliability, `/health` keeps counting
    it alive, and the client receives the empty response instead of a failover.

    `tool_calls` counts as an answer: a legitimate tool call travels with
    `content: null` and all the useful payload there. It is required by the
    TRUTHINESS of the value (not by its presence) because in the FINAL message of
    a non-streaming response, `tool_calls: []` literally means "I called no
    tool" -- the opposite of streaming deltas, where the key's presence is
    already a signal.
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
class ProxyResponse:
    status: int
    json: dict
    route: Route | None
    attempts: int
    reasoning: str = ""
    # HTTP status the PROVIDER returned on the last attempt (0 = there was no
    # response at all: a network error). Not the same as `status`, which is what
    # this gateway decided: a 200 that arrives empty ends up as
    # `status=503, upstream_code=200`, and that difference is exactly what is
    # needed to diagnose, from the probes table, whether the provider is down or
    # answered fine but with nothing inside.
    #
    # In a chain of several routes it is the code of the LAST one tried; anyone
    # needing exact per-route attribution has the `eventos` table, which stores
    # one row per attempt. The health probe always passes a single route, so
    # there is no ambiguity there.
    upstream_code: int = 0


class Proxy:
    def __init__(self, proveedores: dict, almacen, http_client: httpx.AsyncClient):
        self.providers = proveedores
        self.store = almacen
        self.http = http_client
        self.cooldowns: dict[str, float] = {}
        self._punishments: dict[str, int] = {}
        # Round 9: how many generations of cooldown/punishment this key has had
        # (incremented on EVERY _punish/_punish_429/_clear_punishment) -- it lets
        # an on-demand probe detect whether something (typically a real 429)
        # changed the route's state WHILE the probe was in flight, and avoid
        # overwriting it (MEDIUM 5).
        self._cooldown_generation: dict[str, int] = {}
        # Round 9 (HIGH 3): counter of CONSECUTIVE real-traffic failures per free
        # route, with no time window -- it resets to 0 on any success. See
        # _suspect().
        self._suspicions: dict[str, int] = {}
        # Round 9 (HIGH 4): same as above but for PAID routes, which do not go
        # through suspicion+probe (they are never probed, see _suspect_paid) --
        # it counts toward a DIRECT punishment, with no probe in between.
        self._paid_failures: dict[str, int] = {}
        self._last_on_demand_probe: dict[str, float] = {}
        # Round 9 (MEDIUM 6): timestamps of on-demand probes started, across ALL
        # routes -- an aggregate cap independent of catalogue size. See
        # GLOBAL_PROBE_LIMIT_PER_MINUTE.
        self._recent_probes: list[float] = []
        # On-demand probes in flight (asyncio.Task), by key -- prevents firing two
        # overlapping probes against the SAME route if several concurrent requests
        # cross the threshold at nearly the same time, and gives the tests
        # somewhere to wait (`wait_for_pending_probes`).
        self._probes_in_flight: dict[str, asyncio.Task] = {}
        # Round 10, HIGH from the gate: routes that crossed the threshold and want
        # a probe but have not got a slot yet -- see `_admit_pending_probes`.
        # Without this, the GLOBAL quota (round 9, MEDIUM 6) was always taken by
        # the same routes: `complete()` walks the chain in the SAME order on every
        # request (priority/reliability), so the chain's first routes ask for a
        # probe first, ALWAYS -- and a genuinely broken route, whose collapsed
        # reliability sends it to the END of that chain, never got to ask before
        # the minute's quota ran out.
        self._awaiting_probe: dict[str, Route] = {}

    async def complete(self, rutas: list[Route], cuerpo: dict, ahora: float,
                        raw: bool = False, is_probe: bool = False,
                        on_billable_attempt: Callable[[Route], None] | None = None
                        ) -> ProxyResponse:
        """`is_probe=True` is exclusively for INTERNAL use (periodic probing via
        `probing.probe_health`, and the on-demand probe in `_probe_on_demand` --
        a real client's request never passes it): it changes how a NON-429 failure
        is interpreted. With `is_probe=False` (the default, all real traffic) a
        failure only accumulates SUSPICION (`_suspect`/`_suspect_paid`, see
        SUSPICION_THRESHOLD) -- it never punishes directly. With `is_probe=True` a
        failure punishes the route immediately (`_punish`), without going through
        suspicion or scheduling another probe: a probe already IS the
        verification, there is nothing further to confirm.

        `on_billable_attempt`, when passed, is called for every attempt against a
        PAID route the provider answered with `200` -- successful or not (HIGH 4,
        round 9): the provider charges for generating the response, not for the
        gateway considering it useful. `r.route` (above, on success) only marks
        the attempt that genuinely served the client; this callback marks EVERY
        billable attempt, so whoever does the billing (api.py) does not miss the
        "200 with empty content" case -- billable and silent until this round."""
        intentos = 0
        ultimo_error = None
        ultimo_codigo = 0
        claves_del_pedido = {ruta.key for ruta in rutas}
        for ruta in rutas:
            proveedor = self.providers[ruta.provider]
            url, cabeceras, payload = build_request(proveedor, cuerpo, ruta.model_id)
            intentos += 1
            t0 = time.monotonic()
            try:
                resp = await self.http.post(url, headers=cabeceras, json=payload,
                                            timeout=_timeout_for(proveedor))
                codigo = resp.status_code
            except httpx.HTTPError as e:
                codigo, resp, ultimo_error = 0, None, str(e)
            ultimo_codigo = codigo
            # A COMPLETE round-trip, NOT a time-to-first-token: on this path the
            # response arrives all at once, so this number includes the whole
            # generation (7-27 s on a reasoning model). It goes to `latencia_ms`;
            # `ttft_ms` stays 0 so as not to contaminate a p50 that means
            # something else. See the header comment of storage.py.
            latencia = int((time.monotonic() - t0) * 1000)
            # HIGH 2 (round 9): the cooldown a failure of THIS attempt triggers is
            # stamped with NOW + how long THIS attempt took, not with the raw
            # `ahora`. If the attempt took up to TIMEOUT_S=90s (a hung route --
            # the case this whole mechanism exists to catch) and it is stamped
            # with the `ahora` from when the attempt STARTED, the cooldown is born
            # with up to 90s already "eaten": with COOLDOWN_BASE_S=60, the first
            # punishment (60s nominal) is born ALREADY EXPIRED. Measured:
            # effective exclusion max(0, 60*2^(n-1) - 90) = 0s, 30s, 150s, 390s
            # over the first four punishments -- the hung route stayed at the head
            # of the chain. With MockTransport (no real delay) `latencia` is ~0 and
            # this is a no-op: it changes no existing test.
            ahora_del_castigo = ahora + latencia / 1000.0

            if codigo == 200 and ruta.tier == "pago" and on_billable_attempt is not None:
                on_billable_attempt(ruta)

            # A 200 with an unparseable body (e.g. an HTML maintenance page served
            # with status 200) is not a success: it is treated as a failed
            # attempt, with no exception escaping and no punishment (it is not
            # rate-limiting, it is broken).
            datos = None
            if codigo == 200:
                try:
                    datos = resp.json()
                except ValueError:
                    datos = None
                    ultimo_error = "200 con cuerpo no-JSON"
                else:
                    # Valid JSON that is not an object (e.g. a list): `_clean`
                    # below does datos.get(...) and would blow up with an
                    # uncaught AttributeError -- i.e. a 500 from the gateway
                    # because the provider sent something odd. Same treatment as
                    # a non-JSON body, and BEFORE touching `datos`.
                    if not isinstance(datos, dict):
                        datos = None
                        ultimo_error = "200 con cuerpo JSON que no es un objeto"

            # Same place and same treatment as the guard above: a 200 that carries
            # no answer inside is not a success either. Trimming the reasoning
            # happens BEFORE deciding, because what counts is what the client will
            # see: if nothing is left after removing the <think>, that route did
            # not answer (except in raw mode, where the raw text IS the requested
            # answer).
            razon = ""
            if datos is not None:
                razon = "" if raw else self._clean(datos, proveedor.unwraps_canvas)
                if not has_answer(datos):
                    datos = None
                    ultimo_error = "200 sin contenido ni tool_calls"

            exito = codigo == 200 and datos is not None
            self.store.record_event(ruta.key, exito, 0, codigo, ahora,
                                          latency_ms=latencia,
                                          is_client_error=_is_client_error(codigo))

            if exito:
                self._suspicions.pop(ruta.key, None)
                self._paid_failures.pop(ruta.key, None)
                # MEDIUM 5 (round 9): with is_probe=True, the cleanup is
                # deferred to _probe_on_demand (which checks whether a 429
                # punished the route WHILE this probe was in flight before
                # clearing it) -- complete() cannot know that from here.
                if not is_probe:
                    self._clear_punishment(ruta.key)
                if proveedor.emulates_tools:
                    # `cuerpo` is the CLIENT's request, with its `tools` intact:
                    # build_request strips it only from the copy that travels to
                    # the provider. That array is the allow-list that prevents
                    # converting a legitimate text answer into an invented tool call.
                    datos = _emu.detect_and_convert(datos, cuerpo.get("tools"),
                                                    cuerpo.get("tool_choice"))
                return ProxyResponse(200, datos, ruta, intentos, razon, codigo)

            if codigo == 429:
                self._punish_429(ruta.key, ahora_del_castigo, resp)
            elif not _is_client_error(codigo):
                self._react_to_non_429_failure(ruta, ahora, ahora_del_castigo, is_probe)
            ultimo_error = ultimo_error or f"HTTP {codigo}"

        # Only the cooldowns of THIS request's routes count: the proxy outlives
        # a single call and may have punished routes unrelated to this chain,
        # whose expiry is of no use to whoever is asking now.
        cooldowns_del_pedido = {c: v for c, v in self.cooldowns.items()
                                if c in claves_del_pedido}
        return ProxyResponse(503, {"error": {
            "message": "sin rutas disponibles",
            "detalle": ultimo_error,
            "proxima_liberacion": (min(cooldowns_del_pedido.values())
                                   if cooldowns_del_pedido else None),
        }}, None, intentos, "", ultimo_codigo)

    async def complete_stream(self, rutas: list[Route], cuerpo: dict, ahora: float,
                               raw: bool = False,
                               on_route_committed: Callable[[Route], None] | None = None,
                               on_billable_attempt: Callable[[Route], None] | None = None):
        """Emit already-trimmed SSE lines, always ending in `data: [DONE]`.

        It fails over only BEFORE the first useful byte: once content from a route
        has reached the client, switching models would mix two different responses
        into one stream. That is why a network failure AFTER emitting does not
        retry the next route: it closes the stream right there.

        `on_route_committed`, when passed, is called AT MOST once per call to
        this generator: exactly when (and if) a route is confirmed as the one that
        genuinely served the request. It fires from the same place that already
        decides "this was a real success" for telemetry
        (`_registrar_exito_una_vez`, below) -- not from the raw 200 status,
        because a 200 that dies without emitting anything before the first useful
        byte STILL fails over to the next route (see above), and no real service
        happened there.

        `on_billable_attempt`, when passed, is called for every attempt against
        a PAID route whose initial status was `200` -- successful or not (HIGH 4,
        round 9: the provider charges for generating the response, including a
        stream that ends with no useful content). Different from
        `on_route_committed`: that one marks only the WINNING route, this one
        marks every billable attempt.

        It is never called with `is_probe` (round 8: probing.py never streams) --
        every failure here is REAL TRAFFIC, and EVERY failure branch (status !=
        200, a stream with no useful content, a network error, and the cut by
        `if emitido:` that cannot continue the chain) goes through
        `_react_to_non_429_failure` alike -- the same single decision point
        complete() uses, so the two branches cannot drift apart again (that was
        precisely round 8's vector)."""
        for ruta in rutas:
            proveedor = self.providers[ruta.provider]
            url, cabeceras, payload = build_request(proveedor, cuerpo, ruta.model_id)
            payload["stream"] = True
            t0 = time.monotonic()
            emitido = False          # some chunk already went out to the client
            util = False             # ...and at least one carried content or tool_calls
            evento_registrado = False  # this attempt's telemetry was already counted
            # Only the emulation path (below) uses it: there the response is
            # BUFFERED whole before it can be decided whether it is a tool call, so
            # by the time _registrar_exito_una_vez is called the entire generation
            # has already happened. Without this, those routes reported the TOTAL
            # time as ttft -- 7-27s for a reasoning model -- and
            # ranking.latency_factor sank their score precisely for being used. It
            # is stamped when the FIRST fragment arrives, which is what ttft
            # means.
            ttft_medido_ms = None

            def _registrar_exito_una_vez() -> None:
                # Exactly one event per attempt, never zero and never two: it
                # fires the first time there is something USEFUL to send (so ttft
                # measures the first real token, not the stream's close). If there
                # was never anything useful the attempt closes below with a FAILED
                # event, not this one: a 200 delivering neither content nor
                # tool_calls served nobody, and counting it as a success raises the
                # reliability of the route that just failed.
                nonlocal evento_registrado
                if not evento_registrado:
                    ttft = (ttft_medido_ms if ttft_medido_ms is not None
                            else int((time.monotonic() - t0) * 1000))
                    self.store.record_event(ruta.key, True, ttft, 200, ahora)
                    evento_registrado = True
                    self._clear_punishment(ruta.key)
                    self._suspicions.pop(ruta.key, None)
                    self._paid_failures.pop(ruta.key, None)
                    if on_route_committed is not None:
                        on_route_committed(ruta)

            try:
                async with self.http.stream("POST", url, headers=cabeceras, json=payload,
                                            timeout=_timeout_for(proveedor)) as resp:
                    ahora_del_castigo = ahora + (time.monotonic() - t0)
                    if resp.status_code != 200:
                        if resp.status_code == 429:
                            self._punish_429(ruta.key, ahora_del_castigo, resp)
                        elif not _is_client_error(resp.status_code):
                            self._react_to_non_429_failure(
                                ruta, ahora, ahora_del_castigo, is_probe=False)
                        self.store.record_event(
                            ruta.key, False, 0, resp.status_code, ahora,
                            is_client_error=_is_client_error(resp.status_code))
                        continue
                    if ruta.tier == "pago" and on_billable_attempt is not None:
                        on_billable_attempt(ruta)
                    rec = CompositeStreamTrimmer(unwrap_canvas=proveedor.unwraps_canvas)

                    # Tool emulation over streaming. Detecting a tool call needs
                    # the COMPLETE TEXT (the JSON arrives split across deltas), so
                    # this path gathers the whole response and only then decides:
                    # either it emits it as a tool_calls chunk, or it releases it
                    # as text. The price is losing incremental streaming for these
                    # routes -- unavoidable, and only when the client asked for
                    # tools against a provider that does not support them natively.
                    #
                    # `nombres_tools` is the CLIENT request's allow-list: without
                    # it, legitimate text carrying JSON would be turned into an
                    # invented call (see tool_emulator's docstring).
                    #
                    # MIND THE ORDER: `emulates_tools` is evaluated BEFORE looking
                    # at `tools`. The other way round, a malformed `tools` from any
                    # client broke streaming for EVERY provider, including those
                    # that emulate nothing -- and the exception was raised INSIDE
                    # the generator, with the 200 and the SSE headers already sent:
                    # a cut stream with no possible failover.
                    if proveedor.emulates_tools and _emu.tool_names(cuerpo.get("tools")):
                        nombres_tools = _emu.tool_names(cuerpo.get("tools"))
                        acumulado = ""
                        # id/created/model/usage from the provider: fields the
                        # chunk schema marks as required, and the synthetic chunk
                        # was the only one in the gateway that lost them.
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
                            # `usage` arrives in a final chunk of its own (with
                            # empty choices) when the client asked for
                            # stream_options.include_usage: it is always
                            # overwritten so the last one -- the total -- wins.
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
                            acumulado += rec.feed(frag) if not raw else frag
                        if not raw:
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
                            # It was not a tool call: it is delivered as an
                            # ordinary text response, which is exactly what the
                            # client would have received without emulation.
                            _registrar_exito_una_vez()
                            util = True
                            chunk = _emu.build_stream_chunk(None, sobre, acumulado)
                            yield f"data: {json.dumps(chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                            return
                        # A 200 that delivered nothing: same treatment as the
                        # normal path -- a FAILED attempt and failover to the next
                        # route. Nothing was emitted, so it stays clean.
                        if not evento_registrado:
                            self.store.record_event(ruta.key, False, 0, 200, ahora)
                            evento_registrado = True
                            self._react_to_non_429_failure(
                                ruta, ahora, ahora + (time.monotonic() - t0), is_probe=False)
                        continue

                    # Chunks received that do not carry anything useful yet.
                    # They are held back (not emitted) until the first one that
                    # DOES arrive: while nothing has gone out, failover stays
                    # clean. See PENDING_CAP.
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
                        # A tool_calls chunk (or the initial role one) usually
                        # travels with content="": we look at the PRESENCE of
                        # other keys, not their value, because something like
                        # "tool_calls": [] (falsy but present) is still useful to
                        # the client and cannot be thrown away with the content.
                        #
                        # The WHOLE chunk is examined, at its three levels,
                        # skipping the envelope keys (see
                        # _CHUNK_ENVELOPE/_CHOICE_ENVELOPE): `finish_reason` lives
                        # next to `delta` and `usage` at the top level, and
                        # looking only at `delta` lost both.
                        otras = ({k for k in delta if k != "content"}
                                 | {k for k in eleccion
                                    if k != "delta" and k not in _CHOICE_ENVELOPE}
                                 | {k for k in obj
                                    if k != "choices" and k not in _CHUNK_ENVELOPE})
                        if not raw and isinstance(delta.get("content"), str):
                            delta["content"] = rec.feed(delta["content"])
                        contenido = delta.get("content")
                        # Two different questions about the same chunk:
                        #  - hay_texto: does it carry an ANSWER (something that is
                        #    not whitespace)? That is what decides whether the
                        #    attempt was a success -- a response of pure
                        #    whitespace is not a response, same as on the
                        #    non-streaming path.
                        #  - tiene_contenido: does it carry client-facing text,
                        #    even a lone " "? Deltas arrive heavily split and
                        #    those spaces are part of the sentence: they cannot be
                        #    thrown away.
                        hay_texto = isinstance(contenido, str) and bool(contenido.strip())
                        tiene_contenido = isinstance(contenido, str) and contenido != ""
                        trozo = f"data: {json.dumps(obj)}\n\n"
                        if not hay_texto and "tool_calls" not in delta:
                            # Nothing useful YET. The structural bits (role,
                            # finish_reason, the provider's reasoning) and the
                            # lone spaces are kept so they can be released IN
                            # ORDER alongside the first useful chunk; a chunk with
                            # nothing left inside is discarded.
                            if not (tiene_contenido or otras):
                                continue
                            pendientes.append(trozo)
                            if len(pendientes) > PENDING_CAP:
                                # Holding back a stream that emits nothing but
                                # reasoning, without a limit, would be a memory
                                # leak: they are released (clean failover is lost)
                                # but the attempt still counts as failed if real
                                # content never arrives.
                                log.info(
                                    "stream of %s: more than %d contentless chunks "
                                    "held back; releasing them, and this attempt can no "
                                    "hacer failover limpio", ruta.key, PENDING_CAP)
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
                        # A 200 that never delivered content or tool_calls: the
                        # same hole as above, on the streaming side. The
                        # connection worked at the HTTP level, but the client is
                        # left with no answer -- it is recorded as a FAILED
                        # attempt and falls through to the next route.
                        if not evento_registrado:
                            self.store.record_event(ruta.key, False, 0, 200, ahora)
                            evento_registrado = True
                            self._react_to_non_429_failure(
                                ruta, ahora, ahora + (time.monotonic() - t0), is_probe=False)
                        if pendientes:
                            # What was held back is discarded along with the
                            # attempt. That is correct (none of it reached the
                            # client, so failover stays clean) but it cannot be
                            # silent: these are chunks from a route that said 200.
                            log.info(
                                "stream of %s: discarding %d held-back chunk(s); the "
                                "attempt closed with neither content nor tool_calls",
                                ruta.key, len(pendientes))
                        if emitido:
                            # What was held back has already been released (see
                            # PENDING_CAP): another route cannot be spliced on top
                            # without mixing two responses. Round 8: this WAS the
                            # escape hatch that committed the route without
                            # comparing it against anyone -- now it is like any
                            # other failure branch, `_suspect` above already
                            # recorded it, and there is nothing left to commit
                            # here.
                            yield "data: [DONE]\n\n"
                            return
                        continue
                    for p in pendientes:     # p.ej. el chunk final de finish_reason
                        yield p
                    yield "data: [DONE]\n\n"
                    return
            except httpx.HTTPError:
                if not evento_registrado:
                    self.store.record_event(ruta.key, False, 0, 0, ahora)
                    self._react_to_non_429_failure(
                        ruta, ahora, ahora + (time.monotonic() - t0), is_probe=False)
                if emitido:
                    yield "data: [DONE]\n\n"
                    return
                continue

        yield 'data: {"error":{"message":"sin rutas disponibles"}}\n\n'
        yield "data: [DONE]\n\n"

    def _punish(self, clave: str, ahora: float) -> None:
        """Punishment with exponential backoff -- a confirmed PROBE (periodic or
        on demand) or direct paid failures (`_suspect_paid`). Round 9: a 429 NO
        longer goes through here, see `_punish_429`."""
        n = self._punishments.get(clave, 0) + 1
        self._punishments[clave] = n
        self.cooldowns[clave] = ahora + min(COOLDOWN_BASE_S * (2 ** (n - 1)), COOLDOWN_CAP_S)
        self._cooldown_generation[clave] = self._cooldown_generation.get(clave, 0) + 1

    def _punish_429(self, clave: str, ahora: float, resp: httpx.Response | None) -> None:
        """Round 9, MEDIUM 7: a 429 still punishes immediately (an unambiguous
        signal about the route, no probe needed) but NO longer reuses `_punish`'s
        exponential backoff (which could escalate to COOLDOWN_CAP_S=3600s). The
        provider's `Retry-After` is respected when it sends one; otherwise a short
        default is used (COOLDOWN_429_DEFAULT_S); either way it is capped at
        COOLDOWN_429_MAX_S. See the constants' header comment for the measurement
        that motivated the change."""
        duracion = self._retry_after_seconds(resp)
        if duracion is None:
            duracion = COOLDOWN_429_DEFAULT_S
        duracion = min(duracion, COOLDOWN_429_MAX_S)
        self.cooldowns[clave] = ahora + duracion
        self._cooldown_generation[clave] = self._cooldown_generation.get(clave, 0) + 1

    @staticmethod
    def _retry_after_seconds(resp: httpx.Response | None) -> float | None:
        """Parse `Retry-After` (seconds, or an HTTP date) if the provider sends
        one. `None` if it is missing or cannot be interpreted -- the caller falls
        back to a default.

        Round 10, small fix from the gate: a HOSTILE or broken `Retry-After`
        (`-5`, `nan`) passed the old `max(0.0, ...)` and returned a 0s COOLDOWN --
        a provider explicitly saying "stop" (a 429 is as unambiguous as a probe)
        ended up with NO punishment at all, hammered again immediately. A negative,
        non-finite (`nan`/`inf`) or outright unreadable value is treated the SAME
        as absent -- it falls back to the short default, never to zero."""
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

    def _clear_punishment(self, clave: str) -> None:
        """A success erases EVERY trace of previous punishment -- 429s and failed
        probes alike. Factored out so complete() and complete_stream() (with its
        _registrar_exito_una_vez) cannot drift apart on which dictionaries they
        clear."""
        self._punishments.pop(clave, None)
        self.cooldowns.pop(clave, None)
        self._cooldown_generation[clave] = self._cooldown_generation.get(clave, 0) + 1

    def _react_to_non_429_failure(self, ruta: Route, ahora: float, ahora_del_castigo: float,
                                    is_probe: bool) -> None:
        """The SINGLE decision point for a NON-429 failure (see
        _is_client_error) -- shared by complete() and complete_stream() so the two
        branches cannot drift apart in how they treat the same situation: that was
        exactly round 8's vector (complete_stream's `if emitido:` branch had a
        shortcut of its own)."""
        if is_probe:
            self._punish(ruta.key, ahora_del_castigo)
        elif ruta.tier == "pago":
            self._suspect_paid(ruta, ahora_del_castigo)
        else:
            self._suspect(ruta, ahora)

    def _suspect(self, ruta: Route, ahora: float) -> None:
        """Round 8/9: a REAL-TRAFFIC failure never excludes a route by itself --
        it only accumulates suspicion. On crossing SUSPICION_THRESHOLD CONSECUTIVE
        failures (round 9, HIGH 3: no longer "within a time window" -- see
        SUSPICION_THRESHOLD's header comment), the route is RECORDED as a
        candidate for an ON-DEMAND probe (the same fixed `PING` payload as the
        periodic probe); `_admit_pending_probes` decides, on FAIRNESS grounds,
        which of the candidates gets the next slot. It is that probe -- never this
        method, never the client's request -- that decides whether the route is
        punished (`complete(..., is_probe=True)`, in `_probe_on_demand`).

        FREE routes only -- paid ones go through `_suspect_paid` (see
        `_react_to_non_429_failure`), which punishes DIRECTLY without a probe: an
        on-demand probe would spend REAL MONEY against a paid provider with no
        natural owner for that charge (the daily cap is PER KEY, not per route)."""
        self._suspicions[ruta.key] = min(self._suspicions.get(ruta.key, 0) + 1, SUSPICION_THRESHOLD)
        if self._suspicions[ruta.key] < SUSPICION_THRESHOLD:
            return
        if ruta.key in self._probes_in_flight:
            return  # a probe is already in flight for this route -- do not duplicate
        ultimo = self._last_on_demand_probe.get(ruta.key, float("-inf"))
        if ahora - ultimo < ON_DEMAND_PROBE_LIMIT_S:
            return  # rate-limited per route -- the suspicion waits for the next chance
        # Whether there is quota is NOT decided HERE -- the route is recorded as a
        # candidate and `_admit_pending_probes` is asked to share the available
        # quota FAIRLY among ALL current candidates, not just this one.
        self._awaiting_probe[ruta.key] = ruta
        self._admit_pending_probes(ahora)

    def _admit_pending_probes(self, ahora: float) -> None:
        """Round 10, HIGH from the gate: share the global quota (round 9,
        MEDIUM 6) by FAIRNESS, not by arrival order. `complete()` walks the chain
        in the SAME order on every request (priority, then reliability) -- so with
        first-come-first-served admission, the routes at the FRONT of the chain
        ask for a probe first ALWAYS, and a genuinely broken route (whose
        collapsed reliability sends it to the END of the chain) never got to ask
        before the minute's quota ran out. Measured: with 6 routes and traffic
        suspecting all of them equally, the route in position 5 never obtained a
        probe across 60 simulated minutes -- exactly the route that MOST needed
        confirming, indefinitely starved.

        Priority: the candidate with the LONGEST time since an on-demand probe
        (never probed = infinity, always wins) takes the next free slot -- an
        effective round-robin by staleness, re-evaluated on EVERY call (not a
        fixed FIFO queue): as soon as a route gets its probe, no longer being the
        stalest makes the NEXT stalest win the following time, regardless of the
        order in which they asked within a single chain. The global CAP is
        unchanged (round 9): this changes WHO is admitted, not HOW MANY probes per
        minute."""
        corte_global = ahora - GLOBAL_PROBE_WINDOW_S
        self._recent_probes[:] = [m for m in self._recent_probes if m >= corte_global]
        candidatas = sorted(
            self._awaiting_probe.values(),
            key=lambda r: self._last_on_demand_probe.get(r.key, float("-inf")))
        for ruta in candidatas:
            if len(self._recent_probes) >= GLOBAL_PROBE_LIMIT_PER_MINUTE:
                break  # quota exhausted -- the rest wait for the next chance
            if ruta.key in self._probes_in_flight:
                self._awaiting_probe.pop(ruta.key, None)
                continue
            self._awaiting_probe.pop(ruta.key, None)
            self._recent_probes.append(ahora)
            self._last_on_demand_probe[ruta.key] = ahora
            tarea = asyncio.create_task(self._probe_on_demand(ruta, ahora))
            self._probes_in_flight[ruta.key] = tarea
            tarea.add_done_callback(
                lambda _t, clave=ruta.key: self._probes_in_flight.pop(clave, None))

    def _suspect_paid(self, ruta: Route, ahora: float) -> None:
        """HIGH 4 (round 9): paid routes stay out of suspicion+probe (see
        _suspect) because an on-demand probe would spend real money with no owner
        -- but that CANNOT mean nothing excludes them: without this, a broken paid
        route bills every request, forever, with no mechanism taking it out of
        rotation (measured: 40/40 billable calls, `/v1/uso` at `pago_hoy: 0`
        because only SUCCESS was counted, `TOPE_PAGO_DIARIO` never acting -- see
        api.py for the other side of this fix, counting what was BILLED). A DIRECT
        punishment is reintroduced (no probe, as in round 7), with the SAME
        threshold as free-route suspicion, scoped to paid routes specifically:
        they go LAST in the chain (section 7), so they only come into play once
        everything free is exhausted -- the cost of a false positive here is far
        lower than for a free route (all of round 8), and the cost of NOT
        excluding (unbounded spending) is strictly worse.

        Round 10, MEDIUM from the gate: that direct punishment's duration CANNOT
        reuse `_punish` (the exponential backoff exists for CONFIRMED PROBES) --
        exactly the same defect the 429 had before round 9, in another place.
        Measured through the real API: 60->120->...->3600s in 24 requests from one
        key, with the paid fallback out for ALL keys for an hour. Flat, capped
        (`PAID_DIRECT_COOLDOWN_S`), no escalation -- same as `_punish_429`."""
        n = min(self._paid_failures.get(ruta.key, 0) + 1, SUSPICION_THRESHOLD)
        self._paid_failures[ruta.key] = n
        if n >= SUSPICION_THRESHOLD:
            self.cooldowns[ruta.key] = ahora + PAID_DIRECT_COOLDOWN_S
            self._cooldown_generation[ruta.key] = self._cooldown_generation.get(ruta.key, 0) + 1
            self._paid_failures.pop(ruta.key, None)

    async def _probe_on_demand(self, ruta: Route, ahora: float) -> None:
        """The probe `_suspect` schedules. It reuses complete() with
        `is_probe=True` over a chain of a SINGLE route -- the same path that
        already punishes unambiguously -- and additionally records a row in
        `sondas` (the table feeding both reliability and
        `Storage.has_liveness_evidence`), exactly like `probing.probe_health` does
        for the periodic one: to the rest of the system, an on-demand probe and a
        periodic one are indistinguishable, only who fired it differs.

        MEDIUM 5 (round 9): a success of THIS probe must never cancel a 429 newer
        than itself (a client cancelling the provider's "back off" via an
        on-demand probe would be a path to getting the shared key banned). An
        ALREADY active cooldown is preserved (skipped entirely, the probe is not
        even spent) along with the punishment/cooldown GENERATION taken before
        starting; it is only cleared if nothing touched it while the probe was in
        flight.

        LOW 8 (round 9): this runs in a background `asyncio.Task` -- a non-HTTP
        exception (e.g. `sqlite3.OperationalError` under WAL contention with the
        probing scheduler writing in parallel) is not caught by `complete()`
        (which only expects `httpx.HTTPError`) and would go unhandled (silently,
        and with `_suspicions` uncleared: the route would be stuck, unable to be
        suspected again until the rate limit decayed, with a stale suspicion
        attached). It is caught, logged, and `_suspicions` is ALWAYS cleared so
        the route is never left half-way."""
        if self.cooldowns.get(ruta.key, 0.0) > ahora:
            self._suspicions.pop(ruta.key, None)
            return
        generacion_antes = self._cooldown_generation.get(ruta.key, 0)
        try:
            t0 = time.monotonic()
            r = await self.complete([ruta], dict(PING), ahora, is_probe=True)
            ms = int((time.monotonic() - t0) * 1000)
            # Round 10, small fix: it is stamped with NOW + how long the probe
            # took, not with the `ahora` from when it was SCHEDULED -- the same
            # class of bug as round 9's HIGH 2, one level up: for a hung route (up
            # to TIMEOUT_S=90s) the `sondas` row was dated up to 90s in the past,
            # which could mis-order the `ORDER BY momento DESC` that
            # `has_liveness_evidence` relies on.
            ahora_resuelto = ahora + ms / 1000.0
            if r.upstream_code != 429:
                # Small fix: a 429 against the PROBE already has its own
                # proportional punishment (_punish_429, inside complete()) -- it
                # is evidence the route is rate-limited NOW, not that it is
                # broken. Recording it ALSO as a failed health probe would confuse
                # it with a genuinely downed route: two consecutive 429s against
                # the probe would be enough for `has_liveness_evidence` (round 9)
                # to declare it dead -- a momentary capacity signal is not
                # evidence of death.
                self.store.record_probe(ruta.key, "salud", r.status == 200, ms, 0,
                                             r.upstream_code, 0, 0, ahora_resuelto)
            if r.status == 200 and self._cooldown_generation.get(ruta.key, 0) == generacion_antes:
                self._clear_punishment(ruta.key)
        except Exception:
            log.exception(
                "on-demand probe of %s failed with a non-HTTP exception "
                "(possibly SQLite contention under WAL) -- no probe is "
                "recorded and nothing is punished or cleared: the route "
                "is left WITHOUT A VERDICT, not dead by accident.", ruta.key)
        finally:
            self._suspicions.pop(ruta.key, None)

    async def wait_for_pending_probes(self) -> None:
        """Tests only: on-demand probes run in the background (see _suspect) so
        as not to add latency to the client request that triggered them. Awaiting
        this lets the ones in flight RIGHT NOW finish before asserting on
        cooldowns/probes -- without it, a test checking `p.cooldowns` right after
        crossing the threshold can run before the background probe has."""
        tareas = list(self._probes_in_flight.values())
        if tareas:
            await asyncio.gather(*tareas)

    @staticmethod
    def _clean(datos: dict, desenvolver_canvas: bool) -> str:
        razon_total = ""
        for eleccion in datos.get("choices", []):
            msg = eleccion.get("message") or {}
            contenido = msg.get("content")
            if isinstance(contenido, str):
                limpio, razon = trim(contenido, desenvolver_canvas)
                msg["content"] = limpio
                razon_total += razon
        return razon_total
