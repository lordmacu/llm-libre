import asyncio
import json
from pathlib import Path

import httpx
import pytest

from llm_libre.models import Capabilities, Route
from llm_libre.storage import Storage
from llm_libre.providers import Provider, load
from llm_libre.proxy import (COOLDOWN_429_DEFAULT_S, COOLDOWN_429_MAX_S,
                             COOLDOWN_429_STATED_MAX_S,
                             COOLDOWN_BASE_S, PAID_DIRECT_COOLDOWN_S,
                             ON_DEMAND_PROBE_LIMIT_S,
                             GLOBAL_PROBE_LIMIT_PER_MINUTE, SUSPICION_THRESHOLD,
                             GLOBAL_PROBE_WINDOW_S, Proxy, _is_client_error)

YAML_REAL = str(Path(__file__).resolve().parents[1] / "providers.yaml")

BODY = {"model": "auto", "messages": [{"role": "user", "content": "hi"}]}


def _route(model, provider="kilo", tier="free"):
    return Route(provider, model, tier, Capabilities(True, False, 100000, 4096))


def _prov(pid="kilo", tier="free", unwraps_canvas=False, strips_xai_cards=False):
    return Provider(pid, tier, "openai", f"https://{pid}.test", "", "/models", {}, [],
                     unwraps_canvas=unwraps_canvas, strips_xai_cards=strips_xai_cards)


def _ok(contenido="hi"):
    return {"choices": [{"message": {"role": "assistant", "content": contenido}}]}


def _proxy(handler, providers=("kilo",), canvas=frozenset(), cards=frozenset()):
    store = Storage(":memory:")
    store.create_schema()
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return Proxy({p: _prov(p, unwraps_canvas=p in canvas, strips_xai_cards=p in cards)
                  for p in providers}, store, client)


async def test_it_returns_the_first_route_that_answers():
    p = _proxy(lambda req: httpx.Response(200, json=_ok()))
    r = await p.complete([_route("a:free"), _route("b:free")], BODY, now=0.0)
    assert r.status == 200
    assert r.route.model_id == "a:free"
    assert r.attempts == 1


async def test_a_429_sends_the_route_to_cooldown_and_moves_to_the_next():
    calls = []

    def handler(req):
        calls.append(req.url)
        return httpx.Response(429) if len(calls) == 1 else httpx.Response(200, json=_ok())

    p = _proxy(handler)
    r = await p.complete([_route("a:free"), _route("b:free")], BODY, now=100.0)
    assert r.status == 200
    assert r.route.model_id == "b:free"
    assert r.attempts == 2
    assert p.cooldowns["kilo/a:free"] > 100.0


# --- Round 9, MEDIUM 7 from the gate ("the 429 is the only lever the client
#     has left"): the 429 stopped reusing _punish's exponential backoff (which
#     escalated up to COOLDOWN_CAP_S=3600s) -- measured, 12 requests from one key
#     were enough to cool 3 routes via real 429s, far beyond what the provider's
#     own rate-limit window justifies. It now respects `Retry-After` when the
#     provider sends one, and otherwise uses a short, FLAT default (it does not
#     escalate on repeated 429s) capped at COOLDOWN_429_MAX_S. ---

async def test_a_429_without_retry_after_does_not_escalate_on_repeated_hits():
    p = _proxy(lambda req: httpx.Response(429))
    await p.complete([_route("a:free")], BODY, now=0.0)
    first = p.cooldowns["kilo/a:free"]
    await p.complete([_route("a:free")], BODY, now=first)
    second = p.cooldowns["kilo/a:free"]
    assert second - first == COOLDOWN_429_DEFAULT_S  # flat, not exponential


async def test_a_429_respects_the_providers_retry_after():
    # abs=0.5: `_punish_429` stamps `now + the measured real latency`, not the
    # raw `now` (see the comment in test_a_negative_or_non_finite_retry_after...
    # below, where this same pattern is documented in detail) -- a strict `==`
    # comparison is flaky under load for the same reason.
    p = _proxy(lambda req: httpx.Response(429, headers={"Retry-After": "45"}))
    await p.complete([_route("a:free")], BODY, now=0.0)
    assert p.cooldowns["kilo/a:free"] == pytest.approx(45.0, abs=0.5)


async def test_a_429_caps_an_absurd_retry_after():
    # The ceiling moved from COOLDOWN_429_MAX_S to COOLDOWN_429_STATED_MAX_S when
    # the two caps were split: a Retry-After the provider actually SENT is a
    # statement about itself, not a guess by the gateway, and 300s was folding a
    # measured 23.7-hour DeepSeek mute into ~288 retries a day. Still bounded --
    # that is what this test is really pinning -- just at the hour this deployment
    # already accepts elsewhere (COOLDOWN_CAP_S).
    p = _proxy(lambda req: httpx.Response(429, headers={"Retry-After": "999999"}))
    await p.complete([_route("a:free")], BODY, now=0.0)
    assert p.cooldowns["kilo/a:free"] == pytest.approx(COOLDOWN_429_STATED_MAX_S, abs=0.5)


async def test_a_429_without_retry_after_punishes_with_no_probe_at_all():
    p = _proxy(lambda req: httpx.Response(429))
    await p.complete([_route("a:free")], BODY, now=0.0)
    await p.wait_for_pending_probes()
    assert p.store._con.execute("SELECT COUNT(*) FROM probes").fetchone()[0] == 0


async def test_a_success_clears_the_accumulated_punishment():
    state = {"should_fail": True}

    def handler(req):
        return httpx.Response(429) if state["should_fail"] else httpx.Response(200, json=_ok())

    p = _proxy(handler)
    await p.complete([_route("a:free")], BODY, now=0.0)
    state["should_fail"] = False
    await p.complete([_route("a:free")], BODY, now=1000.0)
    assert "kilo/a:free" not in p.cooldowns


async def test_exhausting_every_route_returns_a_503():
    p = _proxy(lambda req: httpx.Response(500))
    r = await p.complete([_route("a:free"), _route("b:free")], BODY, now=0.0)
    assert r.status == 503
    assert r.route is None
    assert r.attempts == 2


async def test_no_routes_returns_a_503_without_trying():
    p = _proxy(lambda req: httpx.Response(200, json=_ok()))
    r = await p.complete([], BODY, now=0.0)
    assert r.status == 503
    assert r.attempts == 0


async def test_it_trims_the_reasoning_from_the_response():
    p = _proxy(lambda req: httpx.Response(200, json=_ok("<think>mmm</think>hi")))
    r = await p.complete([_route("a:free")], BODY, now=0.0)
    assert r.json["choices"][0]["message"]["content"] == "hi"
    assert r.reasoning == "mmm"


async def test_it_unwraps_the_canvas_fence_on_the_non_streaming_path():
    # Only a provider declaring unwraps_canvas=True (chatgpt-proxy) unwraps it --
    # see finding 1 of the review, below.
    fence = (':::writing{title="x"}\nhi\n:::')
    p = _proxy(lambda req: httpx.Response(200, json=_ok(fence)),
              providers=("chatgpt",), canvas={"chatgpt"})
    r = await p.complete([_route("a:free", provider="chatgpt")], BODY, now=0.0)
    assert r.json["choices"][0]["message"]["content"] == "hi\n"


async def test_it_strips_xai_cards_on_the_non_streaming_path():
    # Only a provider declaring strips_xai_cards=True (grok-proxy) strips them.
    card = "a<xai:tool_usage_card>web_search</xai:tool_usage_card>b"
    p = _proxy(lambda req: httpx.Response(200, json=_ok(card)),
              providers=("grok",), cards={"grok"})
    r = await p.complete([_route("a:free", provider="grok")], BODY, now=0.0)
    assert r.json["choices"][0]["message"]["content"] == "ab"


async def test_a_provider_without_card_stripping_leaves_the_tags_alone():
    # A model quoting grok's wire format is answering the question it was asked.
    card = "escribe <xai:card>asi</xai:card> para la tarjeta"
    p = _proxy(lambda req: httpx.Response(200, json=_ok(card)))   # kilo, cards={}
    r = await p.complete([_route("a:free")], BODY, now=0.0)
    assert r.json["choices"][0]["message"]["content"] == card


# --- Finding 1 of the Task 13 review: canvas unwrapping was GLOBAL, but
#     ':::nota{...}' / ':::tip{...}' is also standard Docusaurus/MDX syntax -- it
#     was reproduced live against a Kilo route asking for documentation. A provider
#     that does NOT declare unwraps_canvas (Kilo, OpenRouter, MiniMax) has to leave
#     those markers intact. ---

async def test_a_provider_without_canvas_unwrapping_leaves_docusaurus_markers_alone():
    note = ":::note\nGuarda el token en el .env.\n:::"
    p = _proxy(lambda req: httpx.Response(200, json=_ok(note)))   # kilo, sin canvas={}
    r = await p.complete([_route("a:free")], BODY, now=0.0)
    assert r.json["choices"][0]["message"]["content"] == note


async def test_raw_mode_does_not_touch_the_content():
    p = _proxy(lambda req: httpx.Response(200, json=_ok("<think>mmm</think>hi")))
    r = await p.complete([_route("a:free")], BODY, now=0.0, raw=True)
    assert r.json["choices"][0]["message"]["content"] == "<think>mmm</think>hi"
    assert r.reasoning == ""


async def test_it_sends_the_models_real_id_not_the_alias():
    seen = []

    def handler(req):
        import json
        seen.append(json.loads(req.content)["model"])
        return httpx.Response(200, json=_ok())

    p = _proxy(handler)
    await p.complete([_route("poolside/x:free")], BODY, now=0.0)
    assert seen == ["poolside/x:free"]


async def test_it_records_one_event_per_attempt():
    p = _proxy(lambda req: httpx.Response(200, json=_ok()))
    await p.complete([_route("a:free")], BODY, now=0.0)
    rows = p.store._con.execute("SELECT key, ok FROM events").fetchall()
    assert rows == [("kilo/a:free", 1)]


async def test_a_200_with_an_invalid_body_does_not_blow_up_and_falls_to_503():
    p = _proxy(lambda req: httpx.Response(200, content=b"not json{{{"))
    r = await p.complete([_route("a:free")], BODY, now=0.0)
    assert r.status == 503
    # This is not rate limiting: the broken route must not end up punished.
    assert "kilo/a:free" not in p.cooldowns


async def test_a_200_with_an_invalid_body_moves_to_the_next_route():
    calls = []

    def handler(req):
        calls.append(req.url)
        if len(calls) == 1:
            return httpx.Response(200, content=b"not json{{{")
        return httpx.Response(200, json=_ok())

    p = _proxy(handler)
    r = await p.complete([_route("a:free"), _route("b:free")], BODY, now=0.0)
    assert r.status == 200
    assert r.route.model_id == "b:free"
    assert r.attempts == 2


# --- Fix round 3, B1 (Blocking): a 200 with no answer inside is not a success.
#     Most free models are reasoning models: they burn the budget thinking and
#     return 200 with finish_reason "length" and "content": null. Counting that as
#     a success RAISES that route's reliability, leaves /health at "ok" and skips
#     failover: the client receives an empty response as if it were the answer. ---

def _empty(finish="length"):
    """The real 200 a reasoning model returns when it runs out of budget:
    content null, no tool_calls."""
    return {"choices": [{"message": {"role": "assistant", "content": None},
                         "finish_reason": finish}]}


async def test_a_200_without_content_does_not_count_as_a_success():
    p = _proxy(lambda req: httpx.Response(200, json=_empty()))
    r = await p.complete([_route("a:free")], BODY, now=0.0)
    assert r.status == 503
    assert r.route is None
    # This is not rate limiting: the route must not end up punished, same as with a
    # body no-JSON.
    assert "kilo/a:free" not in p.cooldowns


async def test_a_200_without_content_moves_to_the_next_route():
    calls = []

    def handler(req):
        calls.append(req.url)
        if len(calls) == 1:
            return httpx.Response(200, json=_empty())
        return httpx.Response(200, json=_ok())

    p = _proxy(handler)
    r = await p.complete([_route("a:free"), _route("b:free")], BODY, now=0.0)
    assert r.status == 200
    assert r.route.model_id == "b:free"
    assert r.attempts == 2


async def test_a_200_without_content_is_recorded_as_a_failed_event():
    # The heart of the finding: if this is recorded with ok=1, the route that
    # returns empty RAISES its reliability every time it fails.
    p = _proxy(lambda req: httpx.Response(200, json=_empty()))
    await p.complete([_route("a:free")], BODY, now=0.0)
    rows = p.store._con.execute("SELECT key, ok FROM events").fetchall()
    assert rows == [("kilo/a:free", 0)]


async def test_a_200_with_blank_content_does_not_count_as_a_success_either():
    p = _proxy(lambda req: httpx.Response(200, json=_ok("   \n ")))
    r = await p.complete([_route("a:free")], BODY, now=0.0)
    assert r.status == 503


async def test_a_200_with_only_tool_calls_is_still_a_success():
    # A legitimate case that must NOT break: a function-calling response carries
    # content null and all the useful payload in tool_calls.
    data = {"choices": [{"message": {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "call_1", "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"}}]}}]}
    p = _proxy(lambda req: httpx.Response(200, json=data))
    r = await p.complete([_route("a:free")], BODY, now=0.0)
    assert r.status == 200
    assert r.route.model_id == "a:free"


async def test_a_200_that_is_all_reasoning_does_not_count_as_a_success():
    # What the client sees is what decides: if nothing is left after trimming the
    # <think>, the route answered nothing.
    p = _proxy(lambda req: httpx.Response(200, json=_ok("<think>thinking and thinking</think>")))
    r = await p.complete([_route("a:free")], BODY, now=0.0)
    assert r.status == 503


async def test_in_raw_mode_a_pure_reasoning_200_is_still_a_success():
    # With x_raw the client asked for the content verbatim: there IS an answer there.
    p = _proxy(lambda req: httpx.Response(200, json=_ok("<think>thinking</think>")))
    r = await p.complete([_route("a:free")], BODY, now=0.0, raw=True)
    assert r.status == 200
    assert r.json["choices"][0]["message"]["content"] == "<think>thinking</think>"


async def test_next_release_excludes_cooldowns_from_another_request():
    import json as jsonlib

    def handler(req):
        model = jsonlib.loads(req.content)["model"]
        return httpx.Response(429) if model == "z:free" else httpx.Response(500)

    p = _proxy(handler)
    # An earlier request, over completely different routes, punishes z:free.
    await p.complete([_route("z:free")], BODY, now=0.0)
    assert p.cooldowns["kilo/z:free"] > 0.0

    r = await p.complete([_route("a:free"), _route("b:free")], BODY, now=10.0)
    assert r.status == 503
    assert r.json["error"]["next_release"] is None


async def test_next_release_reports_the_soonest_one_of_this_chain():
    # Round 9: a 429 no longer escalates on repeated hits (MEDIUM 7), so two
    # routes only end up with different cooldowns if the PROVIDER asks for
    # different durations via Retry-After -- exactly the source of truth that is
    # now respected.
    def handler(req):
        model = json.loads(req.content)["model"]
        retry = "100" if model == "a:free" else "20"
        return httpx.Response(429, headers={"Retry-After": retry})

    p = _proxy(handler)
    r = await p.complete([_route("a:free"), _route("b:free")], BODY, now=0.0)
    assert r.status == 503
    assert p.cooldowns["kilo/b:free"] < p.cooldowns["kilo/a:free"]
    assert r.json["error"]["next_release"] == p.cooldowns["kilo/b:free"]


# --- Fix round 3, I5: the NON-streaming path cannot measure a
#     time-to-first-token (the response arrives all at once), so it stops
#     writing its round-trip into the ttft column and stores it in latencia_ms,
#     which is what it actually measured. ---

async def test_the_non_streaming_path_stores_latency_not_ttft():
    p = _proxy(lambda req: httpx.Response(200, json=_ok()))
    await p.complete([_route("a:free")], BODY, now=0.0)
    row = p.store._con.execute(
        "SELECT ttft_ms, latency_ms FROM events").fetchone()
    assert row[0] == 0                # no ttft is invented
    assert row[1] is not None         # but the real latency is recorded


# --- Fix round 4, Minor: a 200 whose JSON is valid but NOT an object (a list)
#     reached `_clean`, which does data.get(...) -> an uncaught AttributeError ->
#     500. Pre-existing, but `has_answer`'s defence sat one line AFTER where it was
#     needed. A passthrough gateway cannot return a 500 because the provider sent
#     something odd. ---

async def test_a_200_whose_json_is_not_an_object_does_not_blow_up():
    p = _proxy(lambda req: httpx.Response(200, json=[1, 2, 3]))
    r = await p.complete([_route("a:free")], BODY, now=0.0)
    assert r.status == 503
    assert "kilo/a:free" not in p.cooldowns     # no es rate-limit, esta rota


async def test_a_200_with_non_object_json_moves_to_the_next_route():
    calls = []

    def handler(req):
        calls.append(req.url)
        if len(calls) == 1:
            return httpx.Response(200, json=["this is not a response"])
        return httpx.Response(200, json=_ok())

    p = _proxy(handler)
    r = await p.complete([_route("a:free"), _route("b:free")], BODY, now=0.0)
    assert r.status == 200 and r.route.model_id == "b:free"


# --- Finding 2 of the Task 13 review, and its final redesign in round 8. Only a
#     429 punished (with exponential backoff). Everything else -- a 500,
#     a timeout, a network error, a 200 with no content -- NEVER left a cooldown,
#     and with TIMEOUT_S=90 a hung route (verified: `blog` is a saturated machine)
#     costs the client up to 5*90s=450s per request, indefinitely, while /health
#     stays "ok" because another route is alive.
#
#     Rounds 6 and 7 tried to solve it with predicates over the CLIENT'S TRAFFIC
#     (a list of codes, the inverted default, chain-level attribution) -- and each
#     one fell to a new vector, the last two hidden inside the very exceptions
#     round 7 had written (a single-route chain, forceable by the client with
#     `model` or `x_min_context`; complete_stream's cut via `if emitido:`). When
#     the leaks are in the exceptions you wrote yourself, the axis is wrong, not
#     under-enumerated.
#
#     Round 8 changes the axis: a real client's traffic can NO LONGER exclude a
#     route directly, ever -- it only accumulates SUSPICION (`Proxy._suspect`).
#     Crossing `SUSPICION_THRESHOLD` CONSECUTIVE failures (round 9: no longer
#     "within a time window", see below) schedules, in the background, OUR OWN
#     PROBE with the same fixed payload (`PING`) the periodic probe already uses --
#     and it is that probe, never the client's request, that decides whether the
#     route is punished (`_punish`, the same backoff as always). The 429 stays
#     intact: it punishes on the FIRST hit, without going through suspicion. ---

async def test_below_the_suspicion_threshold_no_probe_fires_and_nothing_is_punished():
    p = _proxy(lambda req: httpx.Response(500))
    for i in range(SUSPICION_THRESHOLD - 1):
        await p.complete([_route("a:free")], BODY, now=float(i))
    await p.wait_for_pending_probes()
    assert "kilo/a:free" not in p.cooldowns
    assert p.store._con.execute("SELECT COUNT(*) FROM probes").fetchone()[0] == 0


async def test_crossing_the_threshold_fires_a_probe_that_punishes_if_the_route_is_broken():
    p = _proxy(lambda req: httpx.Response(500))   # broken for ANY payload
    for i in range(SUSPICION_THRESHOLD):
        await p.complete([_route("a:free")], BODY, now=float(i))
    await p.wait_for_pending_probes()
    assert p.cooldowns["kilo/a:free"] > 0.0
    # The probe left its own record in `probes` -- the same evidence
    # Storage.has_liveness_evidence reads for /health.
    row = p.store._con.execute(
        "SELECT kind, ok FROM probes WHERE key = 'kilo/a:free'").fetchone()
    assert row == ("health", 0)


async def test_a_route_in_probe_cooldown_is_skipped_on_the_next_request():
    p = _proxy(lambda req: httpx.Response(500))
    for i in range(SUSPICION_THRESHOLD):
        await p.complete([_route("a:free")], BODY, now=float(i))
    await p.wait_for_pending_probes()
    now = float(SUSPICION_THRESHOLD)
    assert p.cooldowns["kilo/a:free"] > now
    await p.complete([_route("a:free")], BODY, now=now)
    # complete() does not filter by cooldown (router.order_routes does that over
    # the merged metrics, see test_router.py) -- what is tested here is that the
    # cooldown is STILL active, without this last attempt resetting it.
    assert p.cooldowns["kilo/a:free"] > now


async def test_a_success_clears_the_accumulated_suspicion():
    state = {"failures": 0}

    def handler(req):
        state["failures"] += 1
        if state["failures"] <= SUSPICION_THRESHOLD - 1:
            return httpx.Response(500)
        return httpx.Response(200, json=_ok())

    p = _proxy(handler)
    for i in range(SUSPICION_THRESHOLD - 1):
        await p.complete([_route("a:free")], BODY, now=float(i))
    # The success on attempt number SUSPICION_THRESHOLD clears the accumulated
    # suspicion: it never crosses the threshold, and no probe is fired.
    await p.complete([_route("a:free")], BODY, now=float(SUSPICION_THRESHOLD))
    await p.wait_for_pending_probes()
    assert "kilo/a:free" not in p.cooldowns
    assert p.store._con.execute("SELECT COUNT(*) FROM probes").fetchone()[0] == 0

    # And SUSPICION_THRESHOLD NEW failures are needed to accumulate again --
    # one alone is not enough, which is precisely what would prove the suspicion
    # was NOT cleared.
    state["failures"] = 0

    def handler_2(req):
        state["failures"] += 1
        return httpx.Response(500)
    p.http = httpx.AsyncClient(transport=httpx.MockTransport(handler_2))
    await p.complete([_route("a:free")], BODY, now=100.0)
    await p.wait_for_pending_probes()
    assert "kilo/a:free" not in p.cooldowns


async def test_a_429_punishes_on_the_first_hit_without_going_through_suspicion():
    # A direct contrast: ONE 429 (not SUSPICION_THRESHOLD) already punishes, and
    # without going through any probe. The 429 path was left untouched.
    p = _proxy(lambda req: httpx.Response(429))
    await p.complete([_route("a:free")], BODY, now=0.0)
    assert "kilo/a:free" in p.cooldowns
    assert p.store._con.execute("SELECT COUNT(*) FROM probes").fetchone()[0] == 0


async def test_it_uses_the_global_timeout_when_the_provider_declares_none():
    seen = []

    def handler(req):
        seen.append(req.extensions.get("timeout"))
        return httpx.Response(200, json=_ok())

    p = _proxy(handler)   # kilo, sin timeout_s declarado
    await p.complete([_route("a:free")], BODY, now=0.0)
    assert seen[0]["read"] == 90.0   # TIMEOUT_S


async def test_it_uses_the_providers_own_timeout_when_declared():
    seen = []

    def handler(req):
        seen.append(req.extensions.get("timeout"))
        return httpx.Response(200, json=_ok())

    store = Storage(":memory:")
    store.create_schema()
    lento = Provider("lento", "free", "openai", "https://lento.test", "", "/models",
                      {}, [], timeout_s=20.0)
    p = Proxy({"lento": lento}, store, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    await p.complete([_route("a:free", provider="lento")], BODY, now=0.0)
    assert seen[0]["read"] == 20.0


# --- Task 14: chatgpt's REAL config (providers.yaml, not a synthetic provider)
#     now declares timeout_s -- see the justification for the number in the YAML
#     itself. It loads the real file with providers.load (the same path as
#     production) so this test goes red if someone changes the value in the YAML
#     without touching this test, or if _timeout_for's wiring breaks -- not a
#     hand-invented timeout_s. ---

async def test_chatgpt_uses_its_own_timeout_from_the_real_yaml():
    seen = []

    def handler(req):
        seen.append(req.extensions.get("timeout"))
        return httpx.Response(200, json=_ok())

    chatgpt = next(p for p in load(YAML_REAL, {}) if p.id == "chatgpt")
    assert chatgpt.timeout_s is not None   # if this fails, the YAML lost timeout_s
    store = Storage(":memory:")
    store.create_schema()
    p = Proxy({"chatgpt": chatgpt}, store,
             httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    await p.complete([_route("gpt-5-3-mini", provider="chatgpt")], BODY, now=0.0)
    assert seen[0]["read"] == chatgpt.timeout_s


async def test_a_single_invalid_200_does_not_punish_but_n_consecutive_ones_do():
    # The suspicion count also includes a 200 with a broken or contentless body
    # -- not only 5xx/network errors -- because neither served the client. One hit
    # alone triggers nothing (already covered by
    # test_a_200_with_an_invalid_body_does_not_blow_up_and_falls_to_503); N in a
    # row, with the probe confirming, do punish.
    p = _proxy(lambda req: httpx.Response(200, content=b"not json{{{"))
    for i in range(SUSPICION_THRESHOLD):
        await p.complete([_route("a:free")], BODY, now=float(i))
    await p.wait_for_pending_probes()
    assert "kilo/a:free" in p.cooldowns


# --- Re-review: a HIGH finding. A 4xx (other than 429) is a DETERMINISTIC CLIENT
#     error -- an invalid payload, an unsupported parameter, an invalid role
#     sequence -- that the provider returns to ANYONE sending that same request,
#     healthy or not. Counting it toward a cooldown (directly or via suspicion)
#     would turn ONE client's mistake into a blackout for EVERYONE: verified
#     against the real 5-route registry, three consecutive malformed requests are
#     enough to leave all five in cooldown if they count. A 400 must only hurt the
#     client that sent it. ---

async def test_three_consecutive_400s_neither_trigger_suspicion_nor_punish():
    p = _proxy(lambda req: httpx.Response(400, json={"error": "bad request"}))
    for i in range(SUSPICION_THRESHOLD):
        await p.complete([_route("a:free")], BODY, now=float(i))
    await p.wait_for_pending_probes()
    assert "kilo/a:free" not in p.cooldowns
    # Not even a confirming probe was fired -- a 400 never counts as suspicion;
    # it is not that the probe came back "healthy".
    assert p.store._con.execute("SELECT COUNT(*) FROM probes").fetchone()[0] == 0


async def test_after_consecutive_400s_a_valid_request_from_another_key_is_still_served():
    # "Another key" is not a Proxy concept (that lives in api.py); what is tested
    # here is the root cause: with no cooldown triggered by the 400, a SUBSEQUENT
    # call (from whoever) still tries the route normally, instead of finding it
    # "exhausted by punishment" from the start.
    state = {"should_fail": True}

    def handler(req):
        if state["should_fail"]:
            return httpx.Response(400, json={"error": "context_length_exceeded"})
        return httpx.Response(200, json=_ok())

    p = _proxy(handler)
    for i in range(SUSPICION_THRESHOLD):
        await p.complete([_route("a:free")], BODY, now=float(i))
    await p.wait_for_pending_probes()
    assert "kilo/a:free" not in p.cooldowns

    state["should_fail"] = False
    r = await p.complete([_route("a:free")], BODY, now=100.0)
    assert r.status == 200


async def test_three_500s_still_punish_via_a_probe_as_before():
    # A direct regression: the 4xx fix must not touch the 5xx path.
    p = _proxy(lambda req: httpx.Response(500))
    for i in range(SUSPICION_THRESHOLD):
        await p.complete([_route("a:free")], BODY, now=float(i))
    await p.wait_for_pending_probes()
    assert "kilo/a:free" in p.cooldowns


async def test_a_mix_of_4xx_and_5xx_counts_only_the_5xx():
    # 400, 500, 400, 500, 400, 500: if the 400 counted toward suspicion, it would
    # fire the probe early. With SUSPICION_THRESHOLD=3, all THREE 500 responses
    # are needed (calls 2, 4 and 6) -- the interleaved 400s
    # interleaved ones do not count (they neither add up nor reset the window).
    # The probe (call 7, beyond the list) also sees a 500: it confirms the route
    # is genuinely broken.
    codes = [400, 500, 400, 500, 400, 500]
    calls = []

    def handler(req):
        code = codes[len(calls)] if len(calls) < len(codes) else 500
        calls.append(code)
        return httpx.Response(code)

    p = _proxy(handler)
    for i in range(5):   # the first 5 (400,500,400,500,400): only two 500s
        await p.complete([_route("a:free")], BODY, now=float(i))
    await p.wait_for_pending_probes()
    assert "kilo/a:free" not in p.cooldowns

    await p.complete([_route("a:free")], BODY, now=5.0)   # el tercer 500
    await p.wait_for_pending_probes()
    assert "kilo/a:free" in p.cooldowns


# --- Re-review (round 4), a HIGH finding: round 3's fix took the 4xx out of the
#     cooldown COUNTER, but it was still being written as an ordinary failed event
#     -- and that feeds reliability, which /health uses to declare a route dead. 26
#     malformed requests from ONE key were enough
#     to sink EVERY route's reliability, with /health at "down" while a DIFFERENT
#     key kept receiving 200s. Worse than the previous round's 503: the persistent
#     /datos volume means a container restart (which Coolify triggers ONLY because
#     /health failed) clears nothing -- a restart loop against a healthy
#     service. ---

async def test_a_400_is_recorded_but_flagged_as_a_client_error():
    p = _proxy(lambda req: httpx.Response(400, json={"error": "bad request"}))
    await p.complete([_route("a:free")], BODY, now=0.0)
    row = p.store._con.execute(
        "SELECT ok, is_client_error FROM events WHERE key = 'kilo/a:free'").fetchone()
    assert row == (0, 1)


async def test_a_500_is_recorded_without_the_client_error_flag():
    p = _proxy(lambda req: httpx.Response(500))
    await p.complete([_route("a:free")], BODY, now=0.0)
    row = p.store._con.execute(
        "SELECT ok, is_client_error FROM events WHERE key = 'kilo/a:free'").fetchone()
    assert row == (0, 0)


# --- Round 6: the AXIS (attribution) was still the right one, but the
#     IMPLEMENTATION inverted it -- "every 4xx is evidence about the request
#     EXCEPT these seven codes" is a default that hides any code nobody has
#     thought of yet. The proof: adding 405 to the set of
#     "route evidence" (or simply not thinking about it, which is what happened
#     with 401/403/404 in round 5) leaves the suite green anyway. The cost,
#     measured: all 5 routes returning 405 (or 409/415/418/431/451,
#     any 4xx nobody anticipated) left the client with a 503 on 100% of requests,
#     ZERO cooldowns, and /health at 200 "ok" --
#     round 3, verbatim, with a different code.
#
#     PRINCIPLE (it goes in the code and in spec S7): when it cannot be known
#     whose fault it is, IT MUST BE COUNTED. A false alarm recovers on its own --
#     someone looks at the ranking or /health, sees the route is fine, moves on. A
#     silent failure does NOT -- nobody ever looks. The costs are asymmetric, and
#     the default has to lean toward noticing.
#
#     The default is inverted: a 4xx is evidence about THE ROUTE unless it is in a
#     SHORT, justified list of codes genuinely about the payload. 429/408/425 NO
#     longer need to be in any list: under the inverted default they land on the
#     route's side by themselves -- a good sign the shape is right. ---

def test_400_413_422_are_evidence_about_the_request():
    assert _is_client_error(400) is True    # Bad Request: it could not even be parsed
    assert _is_client_error(413) is True    # Payload Too Large: THIS request's SIZE
    assert _is_client_error(422) is True    # Unprocessable: invalid for THIS request


def test_the_default_is_route_evidence_across_the_whole_4xx_range():
    # Pins the AXIS, not a list: the ENTIRE 4xx range is walked (not a sample of
    # codes somebody thought of today) against an INDEPENDENT copy of the expected
    # short list -- it deliberately does not import
    # _REQUEST_EVIDENCE_CODES for the comparison. If someone adds a code (405,
    # the one the reviewer used to test the previous round's mutant; or any other,
    # known or not) to the real set without this test also changing, it goes red:
    # it forces ANY widening of the short list through a deliberate decision
    # documented here, not a silent change in proxy.py.
    expected_short_list = {400, 413, 422}
    for code in range(400, 500):
        is_request_evidence = code in expected_short_list
        assert _is_client_error(code) is is_request_evidence, code


def test_401_402_403_404_408_425_429_remain_route_evidence():
    # A regression from rounds 4/5: these seven are NO longer in any explicit set
    # (the inverted default covers them on its own), but the behaviour has to stay
    # the same.
    for code in (401, 402, 403, 404, 408, 425, 429):
        assert _is_client_error(code) is False, code


async def test_three_consecutive_405s_punish_via_a_probe():
    # The exact code the reviewer used to test round 6's mutant: nobody thought
    # about 405 explicitly, and under the inverted default that no longer matters
    # -- any unlisted code counts as suspicion, and the probe (which also receives
    # a 405 from this same handler) confirms.
    p = _proxy(lambda req: httpx.Response(405))
    for i in range(SUSPICION_THRESHOLD):
        await p.complete([_route("a:free")], BODY, now=float(i))
    await p.wait_for_pending_probes()
    assert "kilo/a:free" in p.cooldowns


async def test_a_405_is_recorded_without_the_client_error_flag():
    p = _proxy(lambda req: httpx.Response(405))
    await p.complete([_route("a:free")], BODY, now=0.0)
    row = p.store._con.execute(
        "SELECT ok, is_client_error FROM events WHERE key = 'kilo/a:free'").fetchone()
    assert row == (0, 0)


async def test_three_consecutive_409_415_418_431_451s_punish_via_a_probe():
    # More codes "nobody thought of" -- 409 Conflict, 415 Unsupported Media Type,
    # 418 (the teapot), 431 Request Header Fields Too Large, 451 Unavailable For
    # Legal Reasons. None is in any list, and all of them have to count as
    # suspicion under the inverted default.
    for code in (409, 415, 418, 431, 451):
        p = _proxy(lambda req, c=code: httpx.Response(c))
        for i in range(SUSPICION_THRESHOLD):
            await p.complete([_route("a:free")], BODY, now=float(i))
        await p.wait_for_pending_probes()
        assert "kilo/a:free" in p.cooldowns, code


async def test_three_consecutive_400_422s_do_not_trigger_suspicion():
    for code in (400, 422):
        p = _proxy(lambda req, c=code: httpx.Response(c, json={"error": "x"}))
        for i in range(SUSPICION_THRESHOLD):
            await p.complete([_route("a:free")], BODY, now=float(i))
        await p.wait_for_pending_probes()
        assert "kilo/a:free" not in p.cooldowns, code


async def test_three_consecutive_413s_do_not_trigger_suspicion():
    p = _proxy(lambda req: httpx.Response(413, json={"error": "payload too large"}))
    for i in range(SUSPICION_THRESHOLD):
        await p.complete([_route("a:free")], BODY, now=float(i))
    await p.wait_for_pending_probes()
    assert "kilo/a:free" not in p.cooldowns


def _multi(*modelos):
    return [_route(m) for m in modelos]


def _ping(body: bytes) -> bool:
    """True if the request that reached the mock is the probe -- the same fixed
    `PING` payload (proxy.py), never one a client could write."""
    return json.loads(body)["messages"][0]["content"] == "ping"


# --- Round 8. Rounds 6 and 7 solved attribution by CODE and by CHAIN, and the
#     gate still found TWO more vectors -- both of them escape hatches of
#     propio diseno de round 7:
#
#     1. A chain of a SINGLE route. The previous round left it committing
#        immediately because "it has nothing to compare against" -- but the CLIENT
#        can force that chain themselves, with an explicit `model` or with
#        `x_min_context` (which /v1/ranking already publishes per route, with no
#        internal knowledge needed). 15 identical requests were enough to cool all
#        five routes, one by one.
#     2. complete_stream's `if emitido:` branch, which committed with no chain
#        check at all -- see test_proxy_stream.py.
#
#     Round 8 removes the "how many routes are in the chain" axis entirely: REAL
#     TRAFFIC NEVER EXCLUDES A ROUTE, whatever the chain's shape. It only
#     accumulates suspicion; crossing the threshold schedules OUR OWN probe (fixed
#     payload, written by the gateway) and that probe is the only thing that
#     decides. The tests below reproduce vector 1 exactly -- a single-route chain
#     -- with the mock giving the PROBE a different response from the one it gives
#     the CLIENT: if the route is genuinely healthy (the "ping" succeeds even
#     though the real request fails) the probe saves it; if it is genuinely broken
#     (any payload fails, the "ping" included), the probe confirms and punishes --
#     quickly, without waiting for the 5h cycle. ---

async def test_an_identical_failure_in_a_single_route_chain_does_not_punish_a_healthy_route():
    # The gate's exact vector 1: an explicit `model` narrows it to ONE route.
    def handler(req):
        if _ping(req.content):
            return httpx.Response(200, json=_ok())
        return httpx.Response(403, json={"error": "contenido flageado"})

    p = _proxy(handler)
    for i in range(15):
        r = await p.complete(_multi("a:free"), BODY, now=float(i))
        assert r.status == 503
    await p.wait_for_pending_probes()
    assert p.cooldowns == {}


@pytest.mark.parametrize("build_handler", [
    lambda: (lambda req: httpx.Response(200, json=_ok()) if _ping(req.content)
             else httpx.Response(451, json={"error": "no disponible por razones legales"})),
    lambda: (lambda req: httpx.Response(200, json=_ok()) if _ping(req.content)
             else httpx.Response(200, json=_ok(contenido=None))),
])
async def test_more_identical_failure_vectors_in_a_single_route_chain_do_not_punish(build_handler):
    p = _proxy(build_handler())
    for i in range(15):
        await p.complete(_multi("a:free"), BODY, now=float(i))
    await p.wait_for_pending_probes()
    assert p.cooldowns == {}


async def test_an_identical_timeout_in_a_single_route_chain_does_not_punish_a_healthy_route():
    def handler(req):
        if _ping(req.content):
            return httpx.Response(200, json=_ok())
        raise httpx.ReadTimeout("prompt gigante", request=req)

    p = _proxy(handler)
    for i in range(15):
        await p.complete(_multi("a:free"), BODY, now=float(i))
    await p.wait_for_pending_probes()
    assert p.cooldowns == {}


async def test_an_identical_failure_in_a_multi_route_chain_punishes_no_healthy_route_either():
    # Continuity with round 7: the multi-route vector stays covered too,
    # now via the same mechanism (a chain-length check is no longer needed
    # de cadena aparte).
    def handler(req):
        if _ping(req.content):
            return httpx.Response(200, json=_ok())
        return httpx.Response(403, json={"error": "contenido flageado"})

    routes = _multi("m0:free", "m1:free", "m2:free", "m3:free", "m4:free")
    p = _proxy(handler)
    for i in range(SUSPICION_THRESHOLD):
        await p.complete(routes, BODY, now=float(i))
    await p.wait_for_pending_probes()
    assert p.cooldowns == {}


async def test_the_event_is_still_recorded_even_though_suspicion_does_not_punish():
    # The fix is ONLY about cooldown. `events`/reliability (a measurement, not an
    # exclusion) keep counting every attempt unchanged -- Part 1 is untouched.
    routes = _multi("m0:free", "m1:free")
    p = _proxy(lambda req: httpx.Response(403, json={"error": "contenido flageado"}))
    await p.complete(routes, BODY, now=0.0)
    rows = p.store._con.execute(
        "SELECT key, ok, is_client_error FROM events ORDER BY key").fetchall()
    assert rows == [("kilo/m0:free", 0, 0), ("kilo/m1:free", 0, 0)]


async def test_a_genuinely_broken_route_with_a_healthy_sibling_cools_down_quickly_via_a_probe():
    # The contrast: when the route is GENUINELY broken (the probe fails too), it
    # cools down -- with a healthy sibling in the chain or without one (next
    # test). "Quickly" means within SUSPICION_THRESHOLD requests plus one probe,
    # not the 5h cycle.
    def handler(req):
        body = json.loads(req.content)
        if _ping(req.content):
            return httpx.Response(500)   # the probe sees it broken too
        return httpx.Response(500) if body["model"] == "a:free" else httpx.Response(200, json=_ok())

    p = _proxy(handler)
    for i in range(SUSPICION_THRESHOLD):
        r = await p.complete(_multi("a:free", "b:free"), BODY, now=float(i))
        assert r.status == 200
    await p.wait_for_pending_probes()
    assert "kilo/a:free" in p.cooldowns
    assert "kilo/b:free" not in p.cooldowns


async def test_a_genuinely_broken_route_in_a_single_route_chain_cools_down_quickly():
    p = _proxy(lambda req: httpx.Response(500))   # broken for ANY payload
    for i in range(SUSPICION_THRESHOLD):
        await p.complete(_multi("a:free"), BODY, now=float(i))
    await p.wait_for_pending_probes()
    assert "kilo/a:free" in p.cooldowns


async def test_a_genuinely_down_pool_is_excluded_without_waiting_hours():
    # An explicit requirement from the gate: a genuinely down pool cannot be left
    # waiting for the 5h cycle -- all three routes cool down within roughly
    # SUSPICION_THRESHOLD requests, not hours.
    routes = _multi("m0:free", "m1:free", "m2:free")
    p = _proxy(lambda req: httpx.Response(500))
    for i in range(SUSPICION_THRESHOLD):
        await p.complete(routes, BODY, now=float(i))
    await p.wait_for_pending_probes()
    for route in routes:
        assert route.key in p.cooldowns, route.key


async def test_a_429_in_a_fully_failing_chain_still_punishes_immediately():
    # A 429 does not go through suspicion -- it still punishes on the first hit,
    # regardless of whether the rest of the chain failed too, and without firing
    # any probe. It is an unambiguous signal about THAT route, not about the
    # request.
    routes = _multi("m0:free", "m1:free", "m2:free")
    p = _proxy(lambda req: httpx.Response(429))
    r = await p.complete(routes, BODY, now=0.0)
    assert r.status == 503
    assert "kilo/m0:free" in p.cooldowns
    assert "kilo/m1:free" in p.cooldowns
    assert "kilo/m2:free" in p.cooldowns
    assert p.store._con.execute("SELECT COUNT(*) FROM probes").fetchone()[0] == 0


# --- The three remaining requirements the gate asked for explicitly: rate
#     limiting of on-demand probes, suspicion decay, and paid routes staying
#     outside the mechanism. ---

async def test_the_per_route_on_demand_probe_limit_is_respected():
    ping_calls = []

    def handler(req):
        if _ping(req.content):
            ping_calls.append(1)
            return httpx.Response(200, json=_ok())
        return httpx.Response(500)

    p = _proxy(handler)
    for i in range(SUSPICION_THRESHOLD):
        await p.complete([_route("a:free")], BODY, now=float(i))
    await p.wait_for_pending_probes()
    assert len(ping_calls) == 1   # the first streak already fired its probe

    # More failures, STILL within ON_DEMAND_PROBE_LIMIT_S of the
    # first probe: they cross the threshold again, but the rate limit absorbs the
    # request -- no second probe is fired.
    assert ON_DEMAND_PROBE_LIMIT_S > SUSPICION_THRESHOLD + 1  # the test's assumption
    for i in range(SUSPICION_THRESHOLD, 2 * SUSPICION_THRESHOLD):
        await p.complete([_route("a:free")], BODY, now=float(i))
    await p.wait_for_pending_probes()
    assert len(ping_calls) == 1


# --- Round 9, HIGH 3 from the gate: round 8's 10-minute window meant traffic
#     slower than ~1 failure every 200s NEVER gathered three inside the window --
#     measured, 80 consecutive failures spaced 301s apart fired no probe at all;
#     at 300s they did. A dead route in a low-traffic deployment stayed first in
#     `order_routes` forever. Suspicion is now a CONSECUTIVE counter, with no
#     window: it does not evaporate "because the service is quiet", it only resets
#     on a real success. ---

async def test_suspicion_does_not_evaporate_even_when_traffic_is_slow():
    p = _proxy(lambda req: httpx.Response(500))
    # Three failures separated by MUCH more than the old 10-minute window --
    # exactly the scenario the gate measured (301s apart) and that used to leave
    # suspicion at zero forever.
    await p.complete([_route("a:free")], BODY, now=0.0)
    await p.complete([_route("a:free")], BODY, now=10_000.0)
    await p.complete([_route("a:free")], BODY, now=20_000.0)
    await p.wait_for_pending_probes()
    assert p.cooldowns["kilo/a:free"] > 0.0


async def test_a_success_resets_the_suspicion_counter_not_a_clock():
    # The real protection against "two unrelated incidents adding up" is the
    # reset-on-success, not a time window: two failures, one real success in
    # between, and two more failures (even if they arrive quickly) must not add up
    # to four -- they have to start over from zero.
    state = {"code": 500}

    def handler(req):
        return httpx.Response(state["code"])

    p = _proxy(handler)
    await p.complete([_route("a:free")], BODY, now=0.0)
    await p.complete([_route("a:free")], BODY, now=1.0)
    state["code"] = 200

    def handler_ok(req):
        return httpx.Response(200, json=_ok())
    p.http = httpx.AsyncClient(transport=httpx.MockTransport(handler_ok))
    await p.complete([_route("a:free")], BODY, now=2.0)  # success: resets

    p.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await p.complete([_route("a:free")], BODY, now=3.0)
    await p.complete([_route("a:free")], BODY, now=4.0)
    await p.wait_for_pending_probes()
    assert "kilo/a:free" not in p.cooldowns  # only two NEW failures: not enough


# --- Round 9, HIGH 4 from the gate: round 8 excluded paid routes from the entire
#     suspicion+probe mechanism (they are never probed, because an on-
#     demand probe would spend money with no owner) -- but that alone left a
#     broken paid route billing every request forever with NOTHING excluding it
#     (round 7 DID cool it down). A DIRECT punishment is reintroduced, with no
#     probe, at the same threshold -- bounded by being the chain's last tier. ---

async def test_paid_routes_punish_directly_with_no_probe_at_all():
    p = _proxy(lambda req: httpx.Response(500), providers=("minimax",))
    paid_route = _route("m1", provider="minimax", tier="paid")
    for i in range(SUSPICION_THRESHOLD):
        await p.complete([paid_route], BODY, now=float(i))
    await p.wait_for_pending_probes()
    assert p.cooldowns["minimax/m1"] > 0.0
    # Directly: no probe was ever fired (or spent) to get there.
    assert p.store._con.execute("SELECT COUNT(*) FROM probes").fetchone()[0] == 0


# --- Round 10, MEDIUM from the gate: the direct paid punishment reused _punish's
#     exponential backoff (which exists for CONFIRMED PROBES, something a paid
#     punishment never has) -- the SAME defect the 429 had before round 9, in
#     another place. Measured through the real API:
#     60->120->240->480->960->1920->3600s in 24 requests from one key. Flat,
#     capped -- same as _punish_429. ---

async def test_the_direct_paid_punishment_is_flat_and_does_not_escalate():
    # A small tolerance (not exact): `punish_at` (HIGH 2, round 9) is
    # stamped with NOW + the attempt's REAL latency, not with the raw `now` --
    # with MockTransport that is 0-1ms of jitter depending on machine load. It
    # makes no difference to what this test proves: if the punishment escalated
    # (round 9's _punish), the second round would come out at 120s, well
    # outside a half-second tolerance.
    p = _proxy(lambda req: httpx.Response(500), providers=("minimax",))
    paid_route = _route("m1", provider="minimax", tier="paid")
    now = 0.0
    for i in range(SUSPICION_THRESHOLD):
        await p.complete([paid_route], BODY, now=now + i)
    first = p.cooldowns["minimax/m1"] - (now + (SUSPICION_THRESHOLD - 1))
    assert first == pytest.approx(PAID_DIRECT_COOLDOWN_S, abs=0.5)

    now = 1000.0
    for i in range(SUSPICION_THRESHOLD):
        await p.complete([paid_route], BODY, now=now + i)
    second = p.cooldowns["minimax/m1"] - (now + (SUSPICION_THRESHOLD - 1))
    assert second == pytest.approx(PAID_DIRECT_COOLDOWN_S, abs=0.5)  # the SAME flat value


async def test_below_the_threshold_a_paid_route_is_not_punished():
    p = _proxy(lambda req: httpx.Response(500), providers=("minimax",))
    paid_route = _route("m1", provider="minimax", tier="paid")
    for i in range(SUSPICION_THRESHOLD - 1):
        await p.complete([paid_route], BODY, now=float(i))
    await p.wait_for_pending_probes()
    assert "minimax/m1" not in p.cooldowns


async def test_a_paid_success_clears_the_paid_failure_counter():
    state = {"failures": 0}

    def handler(req):
        state["failures"] += 1
        if state["failures"] <= SUSPICION_THRESHOLD - 1:
            return httpx.Response(500)
        return httpx.Response(200, json=_ok())

    p = _proxy(handler, providers=("minimax",))
    paid_route = _route("m1", provider="minimax", tier="paid")
    for i in range(SUSPICION_THRESHOLD - 1):
        await p.complete([paid_route], BODY, now=float(i))
    await p.complete([paid_route], BODY, now=float(SUSPICION_THRESHOLD))  # exito
    assert "minimax/m1" not in p.cooldowns


# --- Round 9, HIGH 2 from the gate: the cooldown was stamped with the `now` from
#     when the attempt started, not with how long it took. Measured in production
#     (TIMEOUT_S=90): effective exclusion max(0, 60*2^(n-1) - 90) = 0s, 30s, 150s,
#     390s over the first four punishments -- a HUNG route was born with its
#     cooldown already expired. It is tested with a small real delay (not a real
#     90s) to verify the arithmetic without making the test slow. ---

async def test_the_cooldown_of_a_slow_attempt_is_not_born_already_eaten():
    delay_s = 0.05

    async def handler(req):
        await asyncio.sleep(delay_s)
        return httpx.Response(500)

    p = _proxy(handler)
    now = 1000.0
    r = await p.complete([_route("a:free")], BODY, now=now, is_probe=True)
    assert r.status == 503
    # Without the fix: cooldowns["kilo/a:free"] == now + COOLDOWN_BASE_S exactly.
    # With the fix: now + delay_s + COOLDOWN_BASE_S -- more than that.
    assert p.cooldowns["kilo/a:free"] > now + COOLDOWN_BASE_S
    assert p.cooldowns["kilo/a:free"] >= now + delay_s + COOLDOWN_BASE_S


# --- Round 9, MEDIUM 5 from the gate: a successful on-demand probe could erase a
#     real 429 NEWER than the probe itself, if the 429 arrived while the probe was
#     still in flight -- a path for a client to cancel the provider's "back off"
#     via probes and keep hammering the shared key. ---

async def test_a_successful_probe_does_not_erase_a_429_newer_than_itself():
    probe_in_flight = asyncio.Event()
    release_probe = asyncio.Event()
    traffic = {"code": 500}

    async def handler(req):
        body = json.loads(req.content)
        if body["messages"][0]["content"] == "ping":
            probe_in_flight.set()
            await release_probe.wait()
            return httpx.Response(200, json=_ok())
        return httpx.Response(traffic["code"])

    p = _proxy(handler)
    for i in range(SUSPICION_THRESHOLD):
        await p.complete([_route("a:free")], BODY, now=float(i))
    await probe_in_flight.wait()  # the probe started and is frozen mid-flight

    # Meanwhile a REAL 429 arrives through normal traffic -- newer than the probe
    # that has not finished yet.
    traffic["code"] = 429
    await p.complete([_route("a:free")], BODY, now=1000.0)
    assert p.cooldowns["kilo/a:free"] == pytest.approx(1000.0 + COOLDOWN_429_DEFAULT_S, abs=0.5)

    # The probe is released: it resolves SUCCESSFULLY -- but it must not overwrite
    # the 429 that arrived after it started.
    release_probe.set()
    await p.wait_for_pending_probes()
    assert p.cooldowns["kilo/a:free"] == pytest.approx(1000.0 + COOLDOWN_429_DEFAULT_S, abs=0.5)


async def test_a_probe_does_not_start_if_the_route_is_already_in_cooldown():
    # A corollary of the same fix: if by the time it gets to run the route is
    # ALREADY in cooldown (e.g. a 429 that arrived before the scheduler gave the
    # task any time), the probe is not even spent.
    ping_calls = []

    async def handler(req):
        body = json.loads(req.content)
        if body["messages"][0]["content"] == "ping":
            ping_calls.append(1)
        return httpx.Response(200, json=_ok())

    p = _proxy(handler)
    p.cooldowns["kilo/a:free"] = 10_000.0  # already punished, well into the future
    await p._probe_on_demand(_route("a:free"), now=100.0)
    assert ping_calls == []


# --- Round 10, HIGH from the gate: the global quota (round 9) was shared by
#     arrival order -- but `complete()` walks the chain ALWAYS in the same order
#     (priority, reliability), so the chain's first N routes took the quota EVERY
#     time, and a route with collapsed reliability (which `sort_key` sends to the
#     END) could NEVER get a probe. Measured: a victim in position 5 of 6 or 11 of
#     12, zero probes across 60 simulated minutes -- the periodic 5h cycle as the
#     only backstop, degrading detection from ~2s to hours. ---

async def test_a_route_at_the_end_of_an_11_route_catalogue_is_not_starved_by_the_global_quota():
    def handler(req):
        body = json.loads(req.content)
        model = body["model"]
        is_ping = body["messages"][0]["content"] == "ping"
        if model == "victim:free":
            return httpx.Response(500)  # broken for ANY payload, the probe included
        if is_ping:
            return httpx.Response(200, json=_ok())  # the rest: healthy against their own probe
        return httpx.Response(500)  # but real traffic keeps failing for all of them

    p = _proxy(handler)
    # The victim at the END of the chain/catalogue -- exactly where a route with
    # collapsed reliability lands.
    routes = _multi(*[f"m{i}:free" for i in range(10)]) + [_route("victim:free")]

    now = 0.0
    for _ in range(4):   # ceil(11/5)=3 admission rounds + margin
        for i in range(SUSPICION_THRESHOLD):
            r = await p.complete(routes, BODY, now=now)
            assert r.status == 503  # nothing served -- all 11 always fail for real traffic
            now += 1.0
        await p.wait_for_pending_probes()
        if "kilo/victim:free" in p.cooldowns:
            break
        now += GLOBAL_PROBE_WINDOW_S + 5.0  # let the global quota free up

    assert p.cooldowns.get("kilo/victim:free", 0.0) > 0.0


# --- Round 9, MEDIUM 6 from the gate: the on-demand probe limit was PER ROUTE --
#     the AGGREGATE was unbounded. Measured: 11 routes, 15,840 extra requests a
#     day. A global cap, independent of how many routes the catalogue has. ---

async def test_the_global_on_demand_probe_limit_is_respected():
    ping_calls = []

    def handler(req):
        body = json.loads(req.content)
        if body["messages"][0]["content"] == "ping":
            ping_calls.append(1)
            return httpx.Response(200, json=_ok())
        return httpx.Response(500)

    p = _proxy(handler)
    routes = _multi(*[f"m{i}:free" for i in range(GLOBAL_PROBE_LIMIT_PER_MINUTE + 3)])
    for route in routes:
        for i in range(SUSPICION_THRESHOLD):
            await p.complete([route], BODY, now=0.0)
    await p.wait_for_pending_probes()
    assert len(ping_calls) == GLOBAL_PROBE_LIMIT_PER_MINUTE


async def test_the_global_limit_frees_up_once_the_window_passes():
    ping_calls = []

    def handler(req):
        body = json.loads(req.content)
        if body["messages"][0]["content"] == "ping":
            ping_calls.append(1)
            return httpx.Response(200, json=_ok())
        return httpx.Response(500)

    p = _proxy(handler)
    routes = _multi(*[f"m{i}:free" for i in range(GLOBAL_PROBE_LIMIT_PER_MINUTE + 1)])
    for route in routes:
        for i in range(SUSPICION_THRESHOLD):
            await p.complete([route], BODY, now=0.0)
    await p.wait_for_pending_probes()
    assert len(ping_calls) == GLOBAL_PROBE_LIMIT_PER_MINUTE

    # The route that ran out of quota keeps its suspicion at the threshold (it is
    # reset) -- beyond the global window, the next failure tries again and this
    # time the quota is free.
    missing_route = routes[-1]
    await p.complete([missing_route], BODY, now=GLOBAL_PROBE_WINDOW_S + 10.0)
    await p.wait_for_pending_probes()
    assert len(ping_calls) == GLOBAL_PROBE_LIMIT_PER_MINUTE + 1


# --- Round 9, LOW 8 from the gate: the on-demand probe runs in a background
#     asyncio.Task -- a NON-HTTP exception (complete() only catches
#     httpx.HTTPError) went unhandled, silently, with `_suspicions` uncleared: the
#     route was left stuck. ---

async def test_a_non_http_exception_in_the_probe_neither_blows_up_nor_punishes_blindly(caplog):
    p = _proxy(lambda req: httpx.Response(500))
    # The exception has to happen INSIDE the PROBE's attempt (the
    # (SUSPICION_THRESHOLD+1)-th record_event -- the first SUSPICION_THRESHOLD are
    # the real traffic that builds the suspicion), and BEFORE complete() reaches
    # its own punishment decision -- so the verdict is left genuinely unresolved,
    # not already taken.
    original = p.store.record_event
    counter = {"n": 0}

    def _record_event_that_sometimes_blows_up(*a, **kw):
        counter["n"] += 1
        if counter["n"] > SUSPICION_THRESHOLD:
            raise RuntimeError("contencion simulada de sqlite bajo WAL")
        return original(*a, **kw)
    p.store.record_event = _record_event_that_sometimes_blows_up

    for i in range(SUSPICION_THRESHOLD):
        await p.complete([_route("a:free")], BODY, now=float(i))
    await p.wait_for_pending_probes()  # no cuelga ni propaga

    assert "kilo/a:free" not in p.cooldowns  # sin veredicto, no muerta por accidente
    assert "on-demand probe" in caplog.text  # it was logged, not silent
    assert "kilo/a:free" not in p._suspicions  # no queda trabada esperando para siempre


# --- Round 10, small fixes from the gate. ---

async def test_a_negative_or_non_finite_retry_after_falls_back_to_the_default():
    # A hostile or broken `Retry-After` (-5, nan) must NOT produce a 0s cooldown
    # -- a provider explicitly saying "stop" (a 429 is as unambiguous as a probe)
    # would end up hammered again immediately.
    #
    # Post-Task-14 review (gate): `_punish_429` stamps
    # `punish_at = now + measured_real_latency/1000.0`, not the raw `now` (HIGH 2,
    # round 9, see the header comment in proxy.py) -- so a strict `==` comparison
    # against COOLDOWN_429_DEFAULT_S failed every time the MOCKED round-trip
    # crossed 1ms under load (0/20 in isolation, 10/10 running the full suite in
    # parallel with other things). `pytest.approx(..., abs=0.5)` is the same margin
    # the flat paid-cooldown test already uses for the same problem -- generous
    # against real jitter (it will never come close to half a second with a
    # MockTransport) but seconds smaller than any real behaviour change (e.g.
    # going back to exponential escalation instead of flat).
    for value in ("-5", "nan", "inf", "-inf", "no-es-un-numero"):
        p = _proxy(lambda req, v=value: httpx.Response(429, headers={"Retry-After": v}))
        await p.complete([_route("a:free")], BODY, now=0.0)
        assert p.cooldowns["kilo/a:free"] == pytest.approx(COOLDOWN_429_DEFAULT_S, abs=0.5), value


async def test_a_429_against_the_probe_is_not_recorded_as_a_failed_health_probe():
    # A rate limit against the probe ALREADY has its own proportional punishment
    # (_punish_429, inside complete()) -- it is not evidence that the route is
    # BROKEN, it is evidence that it is rate-limited RIGHT NOW. Recording it also
    # as a failed health probe would confuse it with a genuinely downed route.
    p = _proxy(lambda req: httpx.Response(429))
    await p._probe_on_demand(_route("a:free"), now=100.0)
    assert p.store._con.execute("SELECT COUNT(*) FROM probes").fetchone()[0] == 0
    # But it DID punish -- the 429 has its own path, bypassing probes entirely.
    assert "kilo/a:free" in p.cooldowns


async def test_the_probe_row_is_stamped_at_resolution_not_at_scheduling():
    # The same class of bug as HIGH 2 (round 9), one level up: for a hung route,
    # the `probes` row was dated up to TIMEOUT_S=90s in the past, which could
    # mis-order the `ORDER BY momento DESC` that has_liveness_evidence relies on.
    delay_s = 0.05

    async def handler(req):
        await asyncio.sleep(delay_s)
        return httpx.Response(500)

    p = _proxy(handler)
    now = 1000.0
    await p._probe_on_demand(_route("a:free"), now=now)
    row = p.store._con.execute(
        "SELECT at FROM probes WHERE key = 'kilo/a:free'").fetchone()
    assert row[0] >= now + delay_s


# --- A timed-out route must not 503 with an empty reason, added 2026-08-18 ----
#
# Reported symptom: pinning `model` to a real id and sending a long prompt gave a
# 503 after exactly 60s (deepseek, whose provider declares timeout_s: 60) or
# exactly 90s (grok, which declares none and so takes the global TIMEOUT_S). The
# same ids with a short prompt answered 200 in ~2s.
#
# Those numbers are the configured ceilings, not a coincidence -- but the body the
# client got back said only `"message": "no routes available"` with `"detail": ""`,
# because httpx raises ReadTimeout with an empty message and `str(e)` is "". A
# timeout and a DNS failure and a connection refusal were indistinguishable, and
# none of them said "this route was still generating when the clock ran out".


async def test_a_timeout_is_named_in_the_503_instead_of_an_empty_detail():
    def handler(req):
        raise httpx.ReadTimeout("")

    p = _proxy(handler)
    r = await p.complete([_route("a:free")], BODY, now=0.0)
    assert r.status == 503
    detail = r.json["error"]["detail"]
    assert detail, "an empty detail tells the client nothing"
    assert "ReadTimeout" in detail


async def test_a_network_error_that_does_describe_itself_keeps_its_message():
    def handler(req):
        raise httpx.ConnectError("nodename nor servname provided")

    p = _proxy(handler)
    r = await p.complete([_route("a:free")], BODY, now=0.0)
    assert "nodename nor servname provided" in r.json["error"]["detail"]
    assert "ConnectError" in r.json["error"]["detail"]


async def test_the_503_says_how_many_routes_were_actually_tried():
    """With an explicit model id the chain is ONE route, so there is no failover
    at all -- every model id in the production catalogue is served by exactly one
    provider. The client has to be able to tell that apart from "we tried 40
    routes and all of them failed"."""
    def handler(req):
        raise httpx.ReadTimeout("")

    p = _proxy(handler)
    r = await p.complete([_route("a:free")], BODY, now=0.0)
    assert r.json["error"]["routes_tried"] == 1
    r2 = await p.complete([_route("a:free"), _route("b:free")], BODY, now=0.0)
    assert r2.json["error"]["routes_tried"] == 2


# --- A provider that STATES how long to back off, added 2026-08-19 ------------
#
# COOLDOWN_429_MAX_S (300s) capped every 429 alike, whether the provider had said
# how long to wait or the gateway was guessing. Those are not the same claim, and
# conflating them makes the cap wrong in one direction or the other.
#
# The tight cap exists for the GUESSED case, and the measurement behind it is
# real: 12 requests from one key cooled a whole small catalogue through 429s that
# carried no Retry-After, over windows the provider resets in seconds.
#
# But a provider that sends `Retry-After` is not being guessed at -- it is stating
# a fact about itself. Measured live on 2026-08-19: DeepSeek muted the anonymous
# account for 23.7 HOURS and said so. Folded into 300s, the gateway would retry a
# muted account roughly 288 times a day, which is plausibly how the mute got there.


async def test_a_stated_retry_after_is_honoured_far_past_the_guessed_cap():
    p = _proxy(lambda req: httpx.Response(429, headers={"Retry-After": "1800"}))
    await p.complete([_route("a:free")], BODY, now=0.0)
    assert p.cooldowns["kilo/a:free"] == pytest.approx(1800.0)


async def test_a_stated_retry_after_is_still_bounded():
    """Bounded, because a compromised provider could send an absurd one."""
    p = _proxy(lambda req: httpx.Response(429, headers={"Retry-After": "99999999"}))
    await p.complete([_route("a:free")], BODY, now=0.0)
    assert p.cooldowns["kilo/a:free"] == pytest.approx(COOLDOWN_429_STATED_MAX_S)


async def test_without_a_retry_after_the_short_guessed_cap_still_applies():
    """The original measurement must keep holding: a 429 the gateway has to guess
    about cannot cool a route for an hour."""
    p = _proxy(lambda req: httpx.Response(429))
    await p.complete([_route("a:free")], BODY, now=0.0)
    assert p.cooldowns["kilo/a:free"] == pytest.approx(COOLDOWN_429_DEFAULT_S)
    assert p.cooldowns["kilo/a:free"] <= COOLDOWN_429_MAX_S


async def test_the_guessed_cap_stays_tighter_than_the_stated_one():
    assert COOLDOWN_429_MAX_S < COOLDOWN_429_STATED_MAX_S
