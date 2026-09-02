import asyncio
import json
import time
from datetime import datetime, timezone

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from llm_libre.storage import Storage
from llm_libre.api import State, create_app, parse_request
from llm_libre.auth import RateLimiter
from llm_libre.models import Capabilities, Route
from llm_libre.providers import Provider
from llm_libre.proxy import (ON_DEMAND_PROBE_LIMIT_S, PENDING_CAP,
                             SUSPICION_THRESHOLD, Proxy)


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _sse(*chunks: str) -> bytes:
    lines = [f'data: {{"choices":[{{"delta":{{"content":"{t}"}}}}]}}\n\n' for t in chunks]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def test_auto_is_balanced():
    p = parse_request({"model": "auto"})
    assert p.model is None and p.profile == "balanced"


def test_the_profile_aliases():
    assert parse_request({"model": "auto:fast"}).profile == "fast"
    assert parse_request({"model": "auto:strong"}).profile == "strong"


def test_the_capability_aliases_translate_into_requirements():
    p = parse_request({"model": "auto:tools"})
    assert p.needs_tools is True and p.profile == "balanced"
    assert parse_request({"model": "auto:vision"}).needs_vision is True


def test_a_real_model_is_preserved():
    p = parse_request({"model": "poolside/laguna-s-2.1:free"})
    assert p.model == "poolside/laguna-s-2.1:free"


def test_sending_tools_requires_tool_support_even_without_asking():
    p = parse_request({"model": "auto", "tools": [{"type": "function"}]})
    assert p.needs_tools is True


def test_the_x_extensions_are_honoured():
    p = parse_request({"model": "auto", "x_requires": ["tools", "vision"],
                            "x_min_context": 200000, "x_allow_paid": False})
    assert p.needs_tools and p.needs_vision
    assert p.min_context == 200000
    assert p.allow_paid is False


def test_a_whitespace_only_model_is_treated_as_absent():
    # Fix round 1, finding 3 (Minor): "   " is truthy, so it slipped past the
    # "or auto" and came out empty after the strip -- a confusing 404 about the
    # model ''. It must be treated like "" / None / absent: it falls back to "auto".
    p = parse_request({"model": "   "})
    assert p.model is None and p.profile == "balanced"


# --- Post-Task-14 review (gate): three real defects the reviewer found by
#     reading parse_request, not by running it -- all three were malformed CLIENT
#     input falling into a silent hole (a degradation with no warning, or an
#     outright 500). ---

def test_auto_with_an_unknown_suffix_returns_400():
    # "auto:turbo" (a typical typo for "auto:tools") fell through all three suffix
    # branches without touching anything -- silently identical to asking for plain
    # "auto". Dangerous for a client that genuinely wanted to require a capability:
    # ends up without it, with no warning at all.
    with pytest.raises(HTTPException) as exc:
        parse_request({"model": "auto:turbo"})
    assert exc.value.status_code == 400
    assert "auto:turbo" in exc.value.detail["message"]


def test_plain_auto_still_works_after_the_unknown_suffix_fix():
    # A direct regression of the fix above: suffix == "" (that is, "auto" with no
    # ":") must NOT enter the rejection branch.
    p = parse_request({"model": "auto"})
    assert p.model is None and p.profile == "balanced"


def test_auto_balanceado_is_still_a_valid_alias():
    # "balanced" IS in PROFILES -- "auto:balanced" is redundant with plain
    # "auto", but valid, and must not fall into the rejection branch.
    p = parse_request({"model": "auto:balanced"})
    assert p.profile == "balanced"


def test_x_requires_as_a_bare_string_is_accepted_as_a_single_value():
    # `set("tools")` iterates CHARACTERS ({'t','o','l','s'}), so
    # "tools" in exigidas was False and the requirement was ignored entirely,
    # without an error. A bare string (instead of a one-element list) is accepted
    # just the same.
    p = parse_request({"model": "auto", "x_requires": "tools"})
    assert p.needs_tools is True
    assert p.needs_vision is False


def test_x_requires_as_a_list_still_works_as_before():
    p = parse_request({"model": "auto", "x_requires": ["vision"]})
    assert p.needs_vision is True
    assert p.needs_tools is False


def test_a_non_numeric_x_min_context_returns_400_naming_the_field():
    with pytest.raises(HTTPException) as exc:
        parse_request({"model": "auto", "x_min_context": "cien mil"})
    assert exc.value.status_code == 400
    assert exc.value.detail["field"] == "x_min_context"
    assert exc.value.detail["received_value"] == "cien mil"


def test_a_numeric_x_min_context_as_a_string_still_works():
    # int("100000") is valid -- the fix must not become stricter than it already
    # was for the case that DID work.
    p = parse_request({"model": "auto", "x_min_context": "100000"})
    assert p.min_context == 100000


# --- Post-Task-14 review (third gate): the SAME family of bug as x_min_context
#     (an uncaught cast over a client field blows up with
#     TypeError/AttributeError and escapes as a 500) got through twice more --
#     x_requires with a value that is neither a string nor a list (set() over an
#     int/bool/float/list-of-lists) and model with anything that is not a string
#     (.strip() over a number, a list, a dict). The fix generalises with
#     _read_field instead of patching the third site by hand -- these tests cover
#     the two new ones and confirm the SAME error shape
#     (message/campo/valor_recibido) x_min_context already established. ---

@pytest.mark.parametrize("value", [5, True, 3.5, [["tools"]]])
def test_x_requires_that_is_neither_string_nor_list_returns_400_naming_the_field(value):
    with pytest.raises(HTTPException) as exc:
        parse_request({"model": "auto", "x_requires": value})
    assert exc.value.status_code == 400
    assert exc.value.detail["field"] == "x_requires"
    assert exc.value.detail["received_value"] == value


@pytest.mark.parametrize("value", [5, True, 3.5, ["a"], {"a": 1}])
def test_a_non_string_model_returns_400_naming_the_field(value):
    with pytest.raises(HTTPException) as exc:
        parse_request({"model": value})
    assert exc.value.status_code == 400
    assert exc.value.detail["field"] == "model"
    assert exc.value.detail["received_value"] == value


def test_an_absent_or_null_model_still_falls_back_to_auto_without_a_400():
    # A direct regression of the fix above: None (absent) is still valid -- only a
    # PRESENT value of the wrong type should return a 400.
    assert parse_request({}).model is None
    assert parse_request({"model": None}).model is None


@pytest.fixture
def client():
    store = Storage(":memory:")
    store.create_schema()
    store.upsert_routes(
        [Route("kilo", "a:free", "free", Capabilities(True, False, 100000, 4096))], 1.0)
    prov = {"kilo": Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])}
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "hi"}}]})))
    state = State(store=store, proxy=Proxy(prov, store, http),
                    api_keys={"buena"}, daily_paid_cap=200)
    return TestClient(create_app(state))


@pytest.fixture
def state_client():
    """Like `client`, but also exposing the `State`: the tests for
    /health need to seed events/cooldowns directly into the store and
    the proxy, which the `client` fixture does not allow touching from outside."""
    store = Storage(":memory:")
    store.create_schema()
    store.upsert_routes(
        [Route("kilo", "a:free", "free", Capabilities(True, False, 100000, 4096))], 1.0)
    prov = {"kilo": Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])}
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "hi"}}]})))
    state = State(store=store, proxy=Proxy(prov, store, http),
                    api_keys={"buena"}, daily_paid_cap=200)
    return state, TestClient(create_app(state))


def test_without_a_key_it_returns_401(client):
    r = client.post("/v1/chat/completions", json={"model": "auto", "messages": []})
    assert r.status_code == 401


def test_with_a_bad_key_it_returns_401(client):
    r = client.post("/v1/chat/completions", headers={"X-API-Key": "mala"},
                     json={"model": "auto", "messages": []})
    assert r.status_code == 401


def test_completions_answers_and_marks_the_route_used(client):
    r = client.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert r.headers["X-Route-Used"] == "kilo/a:free"
    assert r.headers["X-Tier"] == "free"
    assert r.json()["choices"][0]["message"]["content"] == "hi"


# --- Post-Task-14 review (gate): the same three defects as above
#     (test_auto_with_an_unknown_suffix_returns_400 and company), but exercised
#     through the full HTTP client -- to confirm parse_request is genuinely wired
#     into the real /v1/chat/completions path, not merely tested in isolation. ---

def test_completions_with_an_unknown_alias_returns_400_not_500(client):
    r = client.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto:turbo", "messages": []})
    assert r.status_code == 400
    assert "auto:turbo" in r.json()["detail"]["message"]


def test_completions_with_a_non_numeric_x_min_context_returns_400_not_500(client):
    r = client.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "x_min_context": "cien mil", "messages": []})
    assert r.status_code == 400
    assert r.json()["detail"]["field"] == "x_min_context"


def test_completions_with_x_requires_as_a_string_applies_the_requirement(client):
    # kilo/a:free (the only route in the `client` fixture) declares tools=True, so
    # "x_requires": "tools" (a bare string) has to keep working -- if the bug
    # (set() over a string) came back, this would still return 200 because kilo
    # DOES have tools, so the real proof that the requirement was applied lives in
    # the unit test above
    # (test_x_requires_as_a_bare_string_is_accepted_as_a_single_value); this one
    # only confirms the request reaches the proxy intact without blowing up on the
    # way.
    r = client.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "x_requires": "tools", "messages": []})
    assert r.status_code == 200


# --- Post-Task-14 review (third gate): the same two defects as above (x_requires
#     that is neither string nor list, a non-string model), through the full HTTP
#     client -- it confirms parse_request is wired into the real path, not merely
#     tested in isolation. ---

def test_completions_with_an_invalid_x_requires_type_returns_400_not_500(client):
    r = client.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "x_requires": 5, "messages": []})
    assert r.status_code == 400
    assert r.json()["detail"]["field"] == "x_requires"


def test_completions_with_an_invalid_model_type_returns_400_not_500(client):
    r = client.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": 5, "messages": []})
    assert r.status_code == 400
    assert r.json()["detail"]["field"] == "model"


# --- Task 14 (documentation): the governing rule is that ENRICHING
#     /openapi.json cannot touch the endpoint's real behaviour -- completions()
#     still reads `await request.json()` by hand, WITHOUT a Pydantic model binding
#     the body (that would make FastAPI drop any field it does not declare,
#     breaking the passthrough contract this project exists to provide).
#     `test_client.py` already tests this at the level of `build_request` (the
#     shallow copy that strips the x_* extensions from the body); this test extends
#     it to the full HTTP level -- client -> FastAPI -> proxy -> provider -- so it
#     stays pinned that a field neither the gateway nor any OpenAI SDK knows about
#     still reaches the provider VERBATIM, with the exact value the client
#     sent. ---

def test_an_unknown_field_reaches_the_provider_verbatim():
    received = {}

    def handler(req):
        received.update(json.loads(req.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "hi"}}]})

    store = Storage(":memory:")
    store.create_schema()
    store.upsert_routes(
        [Route("kilo", "a:free", "free", Capabilities(True, False, 100000, 4096))], 1.0)
    prov = {"kilo": Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])}
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    state = State(store=store, proxy=Proxy(prov, store, http),
                    api_keys={"buena"}, daily_paid_cap=200)
    c = TestClient(create_app(state))

    r = c.post("/v1/chat/completions", headers={"X-API-Key": "buena"}, json={
        "model": "auto",
        "messages": [{"role": "user", "content": "hi"}],
        # Neither a standard OpenAI field the gateway does not list in its
        # documentation (reasoning) nor one invented by whichever provider
        # (safety_identifier) is a GATEWAY EXTENSION (x_*, see
        # GATEWAY_EXTENSIONS) -- the contract is "pass through anything you do
        # not recognise", not an allow-list.
        "reasoning": {"enabled": False},
        "safety_identifier": "something-the-gateway-will-never-know",
    })

    assert r.status_code == 200
    assert received["reasoning"] == {"enabled": False}
    assert received["safety_identifier"] == "something-the-gateway-will-never-know"
    # And the GATEWAY's own extensions, if they came in the same request, still
    # would not travel -- see
    # test_it_does_not_forward_the_gateway_extensions_to_the_provider in
    # test_client.py for that half of the contract.
    assert "x_raw" not in received


def test_models_lists_the_catalogue_and_the_aliases(client):
    r = client.get("/v1/models", headers={"X-API-Key": "buena"})
    ids = [m["id"] for m in r.json()["data"]]
    assert "a:free" in ids
    assert "auto" in ids and "auto:fast" in ids


def test_asking_for_impossible_capabilities_returns_400(client):
    r = client.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "messages": [], "x_min_context": 99999999})
    assert r.status_code == 400


def test_an_explicit_model_that_no_longer_exists_returns_404_with_suggestions(client):
    # It is the bug this project exists to prevent: a hardcoded id that died.
    #
    # DEVIATION from the brief: the brief asserts `"a:free" in str(r.json())`, but
    # that depends on difflib.get_close_matches (cutoff=0.3) considering "a:free"
    # similar enough to "poolside/laguna-m.1:free" -- a detail of the similarity
    # metric, not of the contract this test wants to protect. What is asserted is
    # the real contract: that the response carries the "suggestions" key.
    r = client.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "poolside/laguna-m.1:free", "messages": []})
    assert r.status_code == 404
    assert "suggestions" in str(r.json())


# --- ALSO from the round 6 review: the case above covers an id that is NO LONGER
#     in the catalogue. This one is different and is, literally, "the project's
#     reason to exist" -- an id that IS STILL in the catalogue but which the real
#     provider no longer serves (a genuine 404, live). The route already takes the
#     reliability hit (a 404 is route evidence by default, Part 1), but until this
#     fix the client only saw a generic 503 ("detail": "HTTP 404") --
#     indistinguishable from any other transient unavailability, throughout the
#     window of up to 5h before the next catalogue sync (never for paid routes,
#     which are not probed). ---

def test_an_explicit_model_gone_upstream_returns_404_not_503(state_client):
    state, client = state_client
    state.proxy.http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(404, json={"error": {"message": "model not found"}})))
    r = client.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "a:free", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 404
    assert "a:free" in str(r.json())
    assert "suggestions" in str(r.json())


# --- Round 7, LOW from the gate: `_similar_ids` ran against `active_routes()`,
#     which STILL carries the id just declared dead -- the client read
#     `"the model 'a:free' no longer exists"` with `suggestions: ['a:free',
#     ...]`. The request's own `model` is excluded from the list before
#     buscar parecidos. ---

def test_the_live_404_suggestions_exclude_the_model_just_declared_dead(state_client):
    state, client = state_client
    state.store.upsert_routes(
        [Route("kilo", "a:free", "free", Capabilities(True, False, 100000, 4096)),
         Route("kilo", "a:freebie", "free", Capabilities(True, False, 100000, 4096))],
        1.0)
    state.proxy.http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(404, json={"error": {"message": "model not found"}})))
    r = client.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "a:free", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 404
    suggestions = r.json()["detail"]["suggestions"]
    assert "a:free" not in suggestions
    assert "a:freebie" in suggestions


def test_the_auto_model_with_an_upstream_404_is_still_503(state_client):
    # In "auto" mode there is no EXPLICIT id to name -- request.model is None --
    # so this case keeps the usual generic 503, not the new 404 (which requires a
    # specific model to make suggestions about).
    state, client = state_client
    state.proxy.http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(404, json={"error": {"message": "model not found"}})))
    r = client.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 503


def test_health_says_ok_when_a_live_route_exists(client):
    assert client.get("/health").json()["status"] == "ok"


# --- Fix round 1, finding 1 (Critical): /health must ALSO fail when
#     the free route is genuinely broken (a 500 on every attempt), not only when
#     it is in cooldown from a 429.
#
#     Rewritten in round 6, Part 2: the previous version seeded 8 failures + 2
#     successes and expected "down" -- under the "evidence of life" redesign that
#     case CORRECTLY becomes "ok" (there is a real recent success; see
#     test_health_stays_ok_with_30_403s_but_one_recent_success below for the case
#     this replaces). This test now checks what the coordinator asked for
#     explicitly: a GENUINELY dead provider -- zero successes, and the health probe
#     (the most reliable signal there is, because the gateway controls its own
#     payload) failing too. ---

def test_health_is_not_ok_when_the_free_route_genuinely_fails(state_client):
    state, client = state_client
    now = time.time()
    # Real traffic, always failing, and the health probe fails too. This NEVER
    # goes through Proxy._punish (only a 429 triggers that), so proxy.cooldowns
    # stays empty -- the old version of /health would say "ok" anyway if it only
    # looked at cooldowns.
    for _ in range(10):
        state.store.record_event("kilo/a:free", False, 0, 500, now)
    # Round 9: a single failed probe is no longer enough for /health (see
    # Storage.has_liveness_evidence) -- two consecutive ones, with no success in
    # between, are.
    state.store.record_probe("kilo/a:free", "health", False, 100, 0, 500, 0, 0, now - 1)
    state.store.record_probe("kilo/a:free", "health", False, 100, 0, 500, 0, 0, now)
    assert state.proxy.cooldowns == {}  # confirms it is not because of a cooldown

    r = client.get("/health")
    assert r.status_code != 200
    assert r.json()["status"] != "ok"


def test_health_is_ok_when_the_free_route_has_no_telemetry_yet(state_client):
    # A freshly synced route, with no events yet, carries the NEUTRAL reliability
    # (not zero) and must keep counting as alive.
    state, client = state_client
    rows = state.store._con.execute("SELECT COUNT(*) FROM events").fetchone()
    assert rows[0] == 0
    assert client.get("/health").json()["status"] == "ok"


def test_health_is_ok_when_the_free_route_is_healthy(state_client):
    state, client = state_client
    now = time.time()
    for _ in range(9):
        state.store.record_event("kilo/a:free", True, 200, 200, now)
    for _ in range(1):
        state.store.record_event("kilo/a:free", False, 0, 500, now)
    assert client.get("/health").json()["status"] == "ok"


def test_health_still_excludes_on_a_429_cooldown(state_client):
    # The cooldown exclusion (the one that already existed) must not have broken
    # when the reliability one was added.
    state, client = state_client

    async def _force_429():
        routes = state.store.active_routes()
        state.proxy.http = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(429, json={"error": "rate limited"})))
        await state.proxy.complete(routes, {"model": "a:free", "messages": []}, time.time())

    asyncio.run(_force_429())
    assert state.proxy.cooldowns  # confirms it DID end up in cooldown this time

    r = client.get("/health")
    assert r.status_code != 200
    assert r.json()["status"] != "ok"


# --- Round 7, MEDIUM (test quality) from the gate: the test above does not pin
#     the cooldown leg of `_viva()` in api.py -- deleting the
#     `m.cooldown_until <= now` condition outright leaves the whole suite green,
#     because that test's route has no liveness evidence of its own (there was
#     never a real success): `has_liveness_evidence()` drops it on its own, without
#     the cooldown having to intervene. Under round 6's redesign ("evidence of
#     life, not absence of failures"), that condition is precisely the one a future
#     cleanup would delete with nothing noticing. This test isolates the cooldown
#     from the other leg: the route DOES have liveness evidence (a recent success),
#     and /health still has to say not-ok while the cooldown is active. ---

def test_health_excludes_on_cooldown_even_with_liveness_evidence(state_client):
    state, client = state_client
    now = time.time()
    state.store.record_event("kilo/a:free", True, 50, 200, now)

    async def _force_429():
        routes = state.store.active_routes()
        state.proxy.http = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(429, json={"error": "rate limited"})))
        await state.proxy.complete(routes, {"model": "a:free", "messages": []}, now)

    asyncio.run(_force_429())
    assert state.proxy.cooldowns  # confirma que quedo en cooldown

    # Isolated: without the cooldown, this route DOES count as alive on its own.
    assert state.store.has_liveness_evidence("kilo/a:free", now) is True

    r = client.get("/health")
    assert r.status_code != 200
    assert r.json()["status"] != "ok"


# --- Round 6 of Task 13, Part 2. `403` is GENUINELY ambiguous: a suspended
#     account (route evidence, correctly counted -- Part 1) versus content
#     moderated on the provider's side (evidence about ONE particular client's
#     REQUEST). The gateway cannot tell them apart without parsing each provider's
#     specific body, so classifying it correctly (Part 1) is not enough: 30
#     requests with flagged content from a single key must not be able to switch
#     /health off for EVERY key if the route has already proven, with a valid
#     request, that it serves. This is what "evidence of life" buys that "average
#     reliability" could not. ---

def test_health_stays_ok_with_30_403s_but_one_recent_success(state_client):
    state, client = state_client
    now = time.time()
    state.store.record_event("kilo/a:free", True, 50, 200, now)
    for _ in range(30):
        state.store.record_event("kilo/a:free", False, 0, 403, now)
    assert client.get("/health").json()["status"] == "ok"


def test_health_after_a_process_restart_stays_ok_with_403s_and_one_success(tmp_path):
    # A restart of the case above: the success and the 403s live in the /datos
    # file, so a fresh process against the SAME database has to read "ok" just the
    # same, without generating traffic again.
    db_path = str(tmp_path / "salud_403.sqlite3")
    now = time.time()

    store1 = Storage(db_path)
    store1.create_schema()
    store1.upsert_routes(
        [Route("kilo", "a:free", "free", Capabilities(True, False, 100000, 4096))], 1.0)
    store1.record_event("kilo/a:free", True, 50, 200, now)
    for _ in range(30):
        store1.record_event("kilo/a:free", False, 0, 403, now)
    proxy1 = Proxy(
        {"kilo": Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])},
        store1, httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(403, json={"error": "flagged"}))))
    estado1 = State(store=store1, proxy=proxy1, api_keys={"buena"}, daily_paid_cap=200)
    client1 = TestClient(create_app(estado1))
    assert client1.get("/health").json()["status"] == "ok"

    store2 = Storage(db_path)
    store2.create_schema()
    proxy2 = Proxy(
        {"kilo": Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])},
        store2, httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(403, json={"error": "flagged"}))))
    estado2 = State(store=store2, proxy=proxy2, api_keys={"buena"}, daily_paid_cap=200)
    client2 = TestClient(create_app(estado2))
    assert client2.get("/health").json()["status"] == "ok"


def test_health_after_a_process_restart_stays_down_with_no_successes_or_probe(tmp_path):
    # A restart of the "genuinely dead" case: zero successes and the health probe
    # failed too, persisted in /datos -- a fresh process has to keep reading
    # "down", not "ok for lack of evidence to the contrary".
    db_path = str(tmp_path / "salud_muerta.sqlite3")
    now = time.time()

    store1 = Storage(db_path)
    store1.create_schema()
    store1.upsert_routes(
        [Route("kilo", "a:free", "free", Capabilities(True, False, 100000, 4096))], 1.0)
    for _ in range(10):
        store1.record_event("kilo/a:free", False, 0, 500, now)
    # Round 9: TWO consecutive failed probes are required, not one.
    store1.record_probe("kilo/a:free", "health", False, 100, 0, 500, 0, 0, now - 1)
    store1.record_probe("kilo/a:free", "health", False, 100, 0, 500, 0, 0, now)
    proxy1 = Proxy(
        {"kilo": Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])},
        store1, httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(500))))
    estado1 = State(store=store1, proxy=proxy1, api_keys={"buena"}, daily_paid_cap=200)
    client1 = TestClient(create_app(estado1))
    r1 = client1.get("/health")
    assert r1.status_code != 200
    assert r1.json()["status"] != "ok"

    # A second process with a HEALTHY transport, to prove that "down" comes from
    # the ALREADY PERSISTED telemetry, not from new traffic this process
    # genere.
    store2 = Storage(db_path)
    store2.create_schema()
    proxy2 = Proxy(
        {"kilo": Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])},
        store2, httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json={"choices": [
                {"message": {"role": "assistant", "content": "hi"}}]}))))
    estado2 = State(store=store2, proxy=proxy2, api_keys={"buena"}, daily_paid_cap=200)
    client2 = TestClient(create_app(estado2))
    r2 = client2.get("/health")
    assert r2.status_code != 200
    assert r2.json()["status"] != "ok"


def test_health_after_a_process_restart_stays_ok_with_no_telemetry(tmp_path):
    # A restart of the "fresh install" case: zero events, zero probes -- a route
    # with no evidence yet is not born dead, neither in the first process nor after
    # a restart against the same empty database.
    db_path = str(tmp_path / "salud_fresca.sqlite3")

    store1 = Storage(db_path)
    store1.create_schema()
    store1.upsert_routes(
        [Route("kilo", "a:free", "free", Capabilities(True, False, 100000, 4096))], 1.0)
    estado1 = State(store=store1, proxy=Proxy({}, store1, httpx.AsyncClient()),
                     api_keys={"buena"}, daily_paid_cap=200)
    client1 = TestClient(create_app(estado1))
    assert client1.get("/health").json()["status"] == "ok"

    store2 = Storage(db_path)
    store2.create_schema()
    estado2 = State(store=store2, proxy=Proxy({}, store2, httpx.AsyncClient()),
                     api_keys={"buena"}, daily_paid_cap=200)
    client2 = TestClient(create_app(estado2))
    assert client2.get("/health").json()["status"] == "ok"


# --- Round 4 of Task 13, a HIGH finding. Round 3's fix took the 4xx out of the
#     cooldown COUNTER, but it was still being written as an ordinary failed event
#     -- and that feeds reliability, which _viva() uses for the floor of
#     /health. Reproduced: 26 malformed requests IN A ROW from ONE key
#     (a client retrying the same error, something OpenAI's SDKs do on their own)
#     are enough to sink EVERY route's reliability, with /health at "down"/503
#     while a DIFFERENT key with a
#     VALID request keeps receiving 200s the whole time. Worse than the previous
#     round's 503: Coolify uses /health as its health check and RESTARTS the
#     container when it fails -- but `events` lives on the
#     persistent /datos volume, so a fresh process against the SAME database keeps
#     seeing the same 26 failures and keeps reporting "down". A restart loop that
#     restarting cannot break, against a service that
#     responde bien. ---

def test_health_stays_ok_after_30_consecutive_400s(state_client):
    state, client = state_client
    now = time.time()

    async def _mandar_400_treinta_veces():
        routes = state.store.active_routes()
        state.proxy.http = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(400, json={"error": "bad request"})))
        for _ in range(30):
            await state.proxy.complete(routes, {"model": "a:free", "messages": []}, now)

    asyncio.run(_mandar_400_treinta_veces())
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_falls_with_30_consecutive_500s_as_before(state_client):
    # A direct contrast: a failure that IS evidence about the route still brings
    # /health down, exactly as before this fix.
    state, client = state_client
    now = time.time()

    async def _mandar_500_treinta_veces():
        routes = state.store.active_routes()
        state.proxy.http = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(500)))
        for _ in range(30):
            await state.proxy.complete(routes, {"model": "a:free", "messages": []}, now)

    asyncio.run(_mandar_500_treinta_veces())
    r = client.get("/health")
    assert r.status_code != 200
    assert r.json()["status"] != "ok"


def test_health_after_a_process_restart_stays_ok_with_consecutive_400s(tmp_path):
    # The restart-loop case: `events` lives in a real file (the
    # /datos volume), not in process memory. A fresh Storage/Proxy/State
    # SECOND one, against the SAME database, has to read the same result as the
    # first -- if the fix depended on some in-memory state of the old process,
    # this test would catch it and the one above would not.
    db_path = str(tmp_path / "salud.sqlite3")

    store1 = Storage(db_path)
    store1.create_schema()
    store1.upsert_routes(
        [Route("kilo", "a:free", "free", Capabilities(True, False, 100000, 4096))], 1.0)
    proxy1 = Proxy(
        {"kilo": Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])},
        store1, httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(400, json={"error": "bad request"}))))
    estado1 = State(store=store1, proxy=proxy1, api_keys={"buena"}, daily_paid_cap=200)

    async def _mandar_400_treinta_veces():
        routes = store1.active_routes()
        for _ in range(30):
            await proxy1.complete(routes, {"model": "a:free", "messages": []}, time.time())

    asyncio.run(_mandar_400_treinta_veces())
    client1 = TestClient(create_app(estado1))
    assert client1.get("/health").json()["status"] == "ok"

    # "Container restart": a fresh process, a fresh Storage, the SAME database.
    store2 = Storage(db_path)
    store2.create_schema()
    proxy2 = Proxy(
        {"kilo": Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])},
        store2, httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json={"choices": [
                {"message": {"role": "assistant", "content": "hi"}}]}))))
    estado2 = State(store=store2, proxy=proxy2, api_keys={"buena"}, daily_paid_cap=200)
    client2 = TestClient(create_app(estado2))
    r = client2.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_the_ranking_does_not_move_on_consecutive_400s_but_does_on_500s(state_client):
    state, client = state_client
    now = time.time()
    before = client.get("/v1/ranking", headers={"X-API-Key": "buena"}).json()["routes"][0]

    async def _mandar(http_code, veces):
        routes = state.store.active_routes()
        state.proxy.http = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(http_code, json={"error": "x"} if http_code < 500 else None)))
        for _ in range(veces):
            await state.proxy.complete(routes, {"model": "a:free", "messages": []}, now)

    asyncio.run(_mandar(400, 30))
    after_400 = client.get("/v1/ranking", headers={"X-API-Key": "buena"}).json()["routes"][0]
    assert after_400["reliability"] == before["reliability"]
    assert after_400["score"] == before["score"]

    asyncio.run(_mandar(500, 30))
    after_500 = client.get("/v1/ranking", headers={"X-API-Key": "buena"}).json()["routes"][0]
    assert after_500["reliability"] < before["reliability"]
    assert after_500["score"] < before["score"]


# --- Round 5 of Task 13, a HIGH finding. Round 4's classification
#     ("is it retryable?" decides) filed 401/402/403/404 on the client's side --
#     but they are not evidence about the REQUEST, they are evidence about the
#     ROUTE: the key expired (401), the account ran out of credit (402,
#     OpenRouter's "insufficient credits"), the account is suspended or there is
#     moderation on the provider's side (403), or the model no longer exists (404
#     -- literally the central problem this project exists to detect). Measured:
#     all 5 routes returning 401 left the client with a 503 on 100% of requests
#     while /health stayed "ok" -- the green-light blackout /health exists to
#     prevent, and with no backstop for paid routes (never probed). Redrawn around
#     ATTRIBUTION (see proxy._is_client_error / _REQUEST_EVIDENCE_CODES): these
#     four now count the same as a
#     500, en /health y en /v1/ranking. ---

@pytest.mark.parametrize("http_code", [401, 402, 403, 404])
def test_health_falls_with_30_consecutive_route_evidence_codes(state_client, http_code):
    state, client = state_client
    now = time.time()

    async def _mandar_treinta_veces():
        routes = state.store.active_routes()
        state.proxy.http = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(http_code, json={"error": "x"})))
        for _ in range(30):
            await state.proxy.complete(routes, {"model": "a:free", "messages": []}, now)

    asyncio.run(_mandar_treinta_veces())
    r = client.get("/health")
    assert r.status_code != 200
    assert r.json()["status"] != "ok"


def test_health_after_a_process_restart_stays_down_with_consecutive_401s(tmp_path):
    # The flip side of round 4's restart-loop test: a code that IS route evidence
    # (401, an expired key) has to keep
    # reporting "down" even in a FRESH process against the SAME database --
    # this is the test that would have caught the original bug (a 401 wrongly
    # filed as a client error left /health saying "ok" with all 5 routes genuinely
    # down, even after a restart).
    db_path = str(tmp_path / "salud_401.sqlite3")

    store1 = Storage(db_path)
    store1.create_schema()
    store1.upsert_routes(
        [Route("kilo", "a:free", "free", Capabilities(True, False, 100000, 4096))], 1.0)
    proxy1 = Proxy(
        {"kilo": Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])},
        store1, httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(401, json={"error": "invalid api key"}))))
    estado1 = State(store=store1, proxy=proxy1, api_keys={"buena"}, daily_paid_cap=200)

    async def _mandar_401_treinta_veces():
        routes = store1.active_routes()
        for _ in range(30):
            await proxy1.complete(routes, {"model": "a:free", "messages": []}, time.time())
        await proxy1.wait_for_pending_probes()

    asyncio.run(_mandar_401_treinta_veces())
    # Round 9: the suspicion (30 real 401s) fires ONE on-demand probe
    # -- but /health now requires TWO consecutive failures (see
    # Storage.has_liveness_evidence). The confirming one is added
    # directly: the mechanism by which the FIRST probe fires on its own is
    # already covered in test_proxy.py; this test is about PERSISTENCE after
    # a restart, not about the real 60s rate limit between probes.
    store1.record_probe("kilo/a:free", "health", False, 100, 0, 401, 0, 0, time.time())
    client1 = TestClient(create_app(estado1))
    r1 = client1.get("/health")
    assert r1.status_code != 200
    assert r1.json()["status"] != "ok"

    # "Container restart": a fresh process, a fresh Storage, the SAME database --
    # with a HEALTHY transport in the second process, to prove the outage comes
    # from the ALREADY PERSISTED telemetry (events with es_error_cliente=0), not
    # from any traffic this second process generates again.
    store2 = Storage(db_path)
    store2.create_schema()
    proxy2 = Proxy(
        {"kilo": Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])},
        store2, httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json={"choices": [
                {"message": {"role": "assistant", "content": "hi"}}]}))))
    estado2 = State(store=store2, proxy=proxy2, api_keys={"buena"}, daily_paid_cap=200)
    client2 = TestClient(create_app(estado2))
    r2 = client2.get("/health")
    assert r2.status_code != 200
    assert r2.json()["status"] != "ok"


# --- Round 8. The gate found two vectors round 7 did not close, both escape
#     hatches of its own design: a chain of a SINGLE route (forceable by the
#     client with an explicit `model` or `x_min_context`, with no
#     internal knowledge) and complete_stream's `if emitido:` branch
#     (with no chain check). Round 8 removes the "how many routes are there"
#     axis: real traffic NEVER excludes a route directly, it only accumulates
#     suspicion; crossing the threshold schedules OUR OWN probe (the fixed `PING`
#     payload, the same one probing.py uses) and that probe alone decides. Verified
#     end to end via /health, not only in proxy.py -- see
#     test_proxy.py/test_proxy_stream.py for the finer-grained coverage. ---

def _ping(body: bytes) -> bool:
    messages = json.loads(body).get("messages") or []
    return bool(messages) and messages[0].get("content") == "ping"


def test_health_stays_ok_under_a_single_route_chain_attack(state_client):
    # state_client already has a single free route -- exactly the gate's vector 1
    # (an explicit `model`, or x_min_context, narrows down to this).
    state, client = state_client
    now = time.time()

    def handler(req):
        if _ping(req.content):
            return httpx.Response(200, json={"choices": [
                {"message": {"role": "assistant", "content": "pong"}}]})
        return httpx.Response(403, json={"error": "contenido flageado"})

    async def _quince_pedidos_identicos():
        routes = state.store.active_routes()
        state.proxy.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        for i in range(15):
            await state.proxy.complete(routes, {"model": "a:free", "messages": []},
                                         now + i)
        await state.proxy.wait_for_pending_probes()

    asyncio.run(_quince_pedidos_identicos())
    assert client.get("/health").json()["status"] == "ok"


def test_health_stays_ok_under_a_flood_of_contentless_streaming_chunks(state_client):
    # The gate's vector 2: the `if emitido:` branch (PENDING_CAP's force-flush)
    # with no chain narrowing -- "auto", no extensions.
    state, client = state_client
    now = time.time()
    lines = ['data: {"choices":[{"index":0,"delta":{"content":""},'
             '"finish_reason":null}]}\n\n' for _ in range(PENDING_CAP + 6)]
    lines.append("data: [DONE]\n\n")
    payload_sin_contenido = "".join(lines).encode()

    def handler(req):
        if _ping(req.content):
            return httpx.Response(200, json={"choices": [
                {"message": {"role": "assistant", "content": "pong"}}]})
        return httpx.Response(200, content=payload_sin_contenido)

    async def _quince_streams_identicos():
        routes = state.store.active_routes()
        state.proxy.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        body = {"model": "auto", "stream": True,
                 "messages": [{"role": "user", "content": "think hard"}]}
        for i in range(15):
            [ln async for ln in state.proxy.complete_stream(routes, body, now + i)]
        await state.proxy.wait_for_pending_probes()

    asyncio.run(_quince_streams_identicos())
    assert client.get("/health").json()["status"] == "ok"


def test_health_falls_quickly_for_a_genuinely_broken_route_via_an_on_demand_probe(state_client):
    # The contrast: a genuinely broken route (the probe fails too) cools down
    # within SUSPICION_THRESHOLD requests plus one probe -- not in 5h.
    state, client = state_client
    now = time.time()

    async def _umbral_pedidos():
        routes = state.store.active_routes()
        state.proxy.http = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(500)))
        for i in range(SUSPICION_THRESHOLD):
            await state.proxy.complete(routes, {"model": "a:free", "messages": []},
                                         now + i)
        await state.proxy.wait_for_pending_probes()

    asyncio.run(_umbral_pedidos())
    assert state.proxy.cooldowns  # the probe confirmed it and punished
    r = client.get("/health")
    assert r.status_code != 200
    assert r.json()["status"] != "ok"


def test_health_after_a_restart_stays_ok_following_a_single_route_chain_attack(tmp_path):
    db_path = str(tmp_path / "salud_sospecha_ok.sqlite3")
    now = time.time()

    def handler(req):
        if _ping(req.content):
            return httpx.Response(200, json={"choices": [
                {"message": {"role": "assistant", "content": "pong"}}]})
        return httpx.Response(403, json={"error": "contenido flageado"})

    store1 = Storage(db_path)
    store1.create_schema()
    store1.upsert_routes(
        [Route("kilo", "a:free", "free", Capabilities(True, False, 100000, 4096))], 1.0)
    proxy1 = Proxy(
        {"kilo": Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])},
        store1, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    estado1 = State(store=store1, proxy=proxy1, api_keys={"buena"}, daily_paid_cap=200)

    async def _quince_pedidos():
        routes = store1.active_routes()
        for i in range(15):
            await proxy1.complete(routes, {"model": "a:free", "messages": []}, now + i)
        await proxy1.wait_for_pending_probes()

    asyncio.run(_quince_pedidos())
    client1 = TestClient(create_app(estado1))
    assert client1.get("/health").json()["status"] == "ok"

    # "Container restart": a fresh process, a fresh Storage, the SAME database --
    # without the original proxy1 (with its already-resolved probe) intervening.
    store2 = Storage(db_path)
    store2.create_schema()
    proxy2 = Proxy(
        {"kilo": Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])},
        store2, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    estado2 = State(store=store2, proxy=proxy2, api_keys={"buena"}, daily_paid_cap=200)
    client2 = TestClient(create_app(estado2))
    assert client2.get("/health").json()["status"] == "ok"


def test_health_after_a_restart_stays_down_following_a_failed_on_demand_probe(tmp_path):
    # The flip side: the ON-DEMAND probe (not the periodic one) wrote the row
    # in `probes` that declares the route dead -- it has to persist all the same.
    db_path = str(tmp_path / "salud_sospecha_caida.sqlite3")
    now = time.time()

    store1 = Storage(db_path)
    store1.create_schema()
    store1.upsert_routes(
        [Route("kilo", "a:free", "free", Capabilities(True, False, 100000, 4096))], 1.0)
    proxy1 = Proxy(
        {"kilo": Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])},
        store1, httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(500))))   # broken for ANY payload, the probe included
    estado1 = State(store=store1, proxy=proxy1, api_keys={"buena"}, daily_paid_cap=200)

    async def _dos_rachas_de_umbral_pedidos():
        routes = store1.active_routes()
        for i in range(SUSPICION_THRESHOLD):
            await proxy1.complete(routes, {"model": "a:free", "messages": []}, now + i)
        await proxy1.wait_for_pending_probes()
        # A second streak, beyond the on-demand probe rate limit: it fires a
        # SECOND probe through the real mechanism. Round 9 requires two
        # consecutive failures (see Storage.has_liveness_evidence) for /health to
        # treat it as dead -- one alone is no longer enough.
        ahora2 = now + ON_DEMAND_PROBE_LIMIT_S + SUSPICION_THRESHOLD + 10
        for i in range(SUSPICION_THRESHOLD):
            await proxy1.complete(routes, {"model": "a:free", "messages": []}, ahora2 + i)
        await proxy1.wait_for_pending_probes()

    asyncio.run(_dos_rachas_de_umbral_pedidos())
    client1 = TestClient(create_app(estado1))
    r1 = client1.get("/health")
    assert r1.status_code != 200
    assert r1.json()["status"] != "ok"

    rows = store1._con.execute(
        "SELECT kind, ok FROM probes WHERE key = 'kilo/a:free'").fetchall()
    assert len(rows) == 2
    assert all(f == ("health", 0) for f in rows)

    # A second process, with a HEALTHY transport -- to prove that "down" comes
    # from the ALREADY PERSISTED probe, not from new traffic this process
    # genere.
    store2 = Storage(db_path)
    store2.create_schema()
    proxy2 = Proxy(
        {"kilo": Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])},
        store2, httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json={"choices": [
                {"message": {"role": "assistant", "content": "hi"}}]}))))
    estado2 = State(store=store2, proxy=proxy2, api_keys={"buena"}, daily_paid_cap=200)
    client2 = TestClient(create_app(estado2))
    r2 = client2.get("/health")
    assert r2.status_code != 200
    assert r2.json()["status"] != "ok"


def test_the_ranking_falls_on_consecutive_401s_just_as_on_500s(state_client):
    state, client = state_client
    now = time.time()
    before = client.get("/v1/ranking", headers={"X-API-Key": "buena"}).json()["routes"][0]

    async def _mandar_401_treinta_veces():
        routes = state.store.active_routes()
        state.proxy.http = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(401, json={"error": "invalid api key"})))
        for _ in range(30):
            await state.proxy.complete(routes, {"model": "a:free", "messages": []}, now)

    asyncio.run(_mandar_401_treinta_veces())
    after = client.get("/v1/ranking", headers={"X-API-Key": "buena"}).json()["routes"][0]
    assert after["reliability"] < before["reliability"]
    assert after["score"] < before["score"]


def test_the_ranking_breaks_down_the_components(client):
    row = client.get("/v1/ranking", headers={"X-API-Key": "buena"}).json()["routes"][0]
    for field in ("key", "score", "quality", "reliability", "ttft_p50_ms", "tier",
                 "priority"):
        assert field in row


# --- Finding 3 of the Task 13 review: /v1/ranking ordered ONLY by score and did
#     not carry `priority`, so it could show kilo/k:free
#     arriba de todo mientras X-Route-Used decia chatgpt/gpt-5-0 -- el
#     endpoint the README describes as the place to audit WHY the router chose
#     what it chose stopped explaining it. It now orders with the SAME key as
#     router.order_routes (via router.sort_key). ---

def test_the_ranking_orders_by_priority_not_only_by_score(state_client):
    state, client = state_client
    state.store.upsert_routes([
        Route("chatgpt", "gpt-5:free", "free", Capabilities(True, False, 100000, 4096),
             priority=0),
    ], 2.0, deactivate_missing=False)
    # chatgpt: top priority (0) but a BAD score.
    state.store.record_probe("chatgpt/gpt-5:free", "quality", True, 0, 0, 200, 1, 5, 10.0)
    state.store.record_event("chatgpt/gpt-5:free", False, 0, 500, 20.0)
    # kilo/a:free (priority 100, the fixture's default): a BETTER score.
    state.store.record_probe("kilo/a:free", "quality", True, 0, 0, 200, 5, 5, 10.0)
    state.store.record_event("kilo/a:free", True, 50, 200, 20.0)

    rows = client.get("/v1/ranking", headers={"X-API-Key": "buena"}).json()["routes"]
    claves = [f["key"] for f in rows]
    puntajes = {f["key"]: f["score"] for f in rows}
    # Confirms it genuinely scores worse -- otherwise the test proves nothing.
    assert puntajes["chatgpt/gpt-5:free"] < puntajes["kilo/a:free"]
    # And it still goes first: priority rules, as in the real router.
    assert claves[0] == "chatgpt/gpt-5:free"
    assert {f["key"]: f["priority"] for f in rows} == {
        "chatgpt/gpt-5:free": 0, "kilo/a:free": 100}


def test_the_ranking_sends_a_route_in_cooldown_last_even_if_it_scores_better(state_client):
    # Re-review: /v1/ranking still did not model the cooldown, so a punished
    # route -- one the router would NEVER pick right now -- could
    # head the table anyway. en_cooldown_hasta was already in the row (it
    # can diagnose), but the ORDER has to match the router's: a route in cooldown
    # goes last, regardless of priority or
    # score.
    state, client = state_client
    state.store.upsert_routes([
        Route("chatgpt", "gpt-5:free", "free", Capabilities(True, False, 100000, 4096),
             priority=0),
    ], 2.0, deactivate_missing=False)
    # chatgpt: the best priority AND the best score -- but it is punished.
    state.store.record_probe("chatgpt/gpt-5:free", "quality", True, 0, 0, 200, 5, 5, 10.0)
    state.store.record_event("chatgpt/gpt-5:free", True, 50, 200, 20.0)
    state.proxy.cooldowns["chatgpt/gpt-5:free"] = time.time() + 500

    rows = client.get("/v1/ranking", headers={"X-API-Key": "buena"}).json()["routes"]
    claves = [f["key"] for f in rows]
    assert claves[-1] == "chatgpt/gpt-5:free"
    assert claves[0] == "kilo/a:free"


# --- Fix round 1, finding 2 (Critical): the daily paid cap also has to
#     bind on the streaming branch, not only on the synchronous one. ---

def _free_and_paid_state(daily_paid_cap, make_free_response, make_paid_response):
    """Two routes -- one free, one paid. `make_free_response`/`make_paid_response`
    are zero-argument callables that build a FRESH `httpx.Response` on
    each call (not a shared object): the responses travel via `.stream()`, whose
    internal state can only be consumed once, and this
    helper may be invoked more than once per route within a single test."""
    store = Storage(":memory:")
    store.create_schema()
    store.upsert_routes([
        Route("free_prov", "f:free", "free", Capabilities(True, False, 100000, 4096)),
        Route("paid_prov", "p:paid", "paid", Capabilities(True, False, 100000, 4096)),
    ], 1.0)
    prov = {
        "free_prov": Provider("free_prov", "free", "openai", "https://f.test", "", "/models", {}, []),
        "paid_prov": Provider("paid_prov", "paid", "openai", "https://p.test", "", "/models", {}, []),
    }

    def responder(req):
        return make_free_response() if "f.test" in str(req.url) else make_paid_response()

    http = httpx.AsyncClient(transport=httpx.MockTransport(responder))
    state = State(store=store, proxy=Proxy(prov, store, http),
                    api_keys={"buena"}, daily_paid_cap=daily_paid_cap)
    return state, TestClient(create_app(state))


def test_paid_streaming_counts_usage_and_the_cap_binds():
    # (a) with a tiny cap, a streaming request that IS served by the paid route
    # counts usage -- and a second one, with the cap already exhausted, stops
    # offering the paid route (the only provider answering correctly is the paid
    # one; with the cap exhausted the chain is left with no viable candidates and
    # the stream falls to the "no routes available" body without paying again).
    state, client = _free_and_paid_state(
        daily_paid_cap=1,
        make_free_response=lambda: httpx.Response(500, json={"error": "free caida"}),
        make_paid_response=lambda: httpx.Response(200, content=_sse("de", " pago")))
    day = _today()
    assert state.store.paid_usage("buena", day) == 0

    r1 = client.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                      json={"model": "auto", "messages": [], "stream": True})
    assert r1.status_code == 200
    assert "de" in r1.text and "pago" in r1.text
    assert state.store.paid_usage("buena", day) == 1  # (a) conto exactamente 1

    r2 = client.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                      json={"model": "auto", "messages": [], "stream": True})
    assert state.store.paid_usage("buena", day) == 1  # el tope realmente ato
    assert "error" in r2.text  # no viable route: free is still down, paid was excluded


# --- Round 9, HIGH 4 from the gate: round 8 counted paid usage only on SUCCESS
#     (`r.route`/`on_route_committed`) -- but a 200 with empty content (a reasoning
#     model that burns its budget) is BILLED by the provider anyway, even though
#     the gateway treats it as failed and continues the chain. Measured: 40/40
#     billable calls with
#     `pago_hoy: 0`, DAILY_PAID_CAP never acting. Now every
#     attempt with status 200 against a paid route, whether it serves or not. ---

def test_paid_streaming_bills_an_empty_200_even_though_it_serves_nothing():
    empty_body = b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\ndata: [DONE]\n\n'
    state, client = _free_and_paid_state(
        daily_paid_cap=5,
        make_free_response=lambda: httpx.Response(500, json={"error": "free caida"}),
        make_paid_response=lambda: httpx.Response(200, content=empty_body))
    day = _today()
    r = client.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "messages": [], "stream": True})
    assert r.status_code == 200
    assert state.store.paid_usage("buena", day) == 1  # facturable, aunque no sirvio


def test_paid_non_streaming_bills_an_empty_200_even_though_it_serves_nothing():
    store = Storage(":memory:")
    store.create_schema()
    store.upsert_routes([
        Route("free_prov", "f:free", "free", Capabilities(True, False, 100000, 4096)),
        Route("paid_prov", "p:paid", "paid", Capabilities(True, False, 100000, 4096)),
    ], 1.0)
    prov = {
        "free_prov": Provider("free_prov", "free", "openai", "https://f.test", "", "/models", {}, []),
        "paid_prov": Provider("paid_prov", "paid", "openai", "https://p.test", "", "/models", {}, []),
    }
    empty_body = {"choices": [{"message": {"role": "assistant", "content": None}}]}

    def responder(req):
        if "f.test" in str(req.url):
            return httpx.Response(500, json={"error": "free caida"})
        return httpx.Response(200, json=empty_body)

    http = httpx.AsyncClient(transport=httpx.MockTransport(responder))
    state = State(store=store, proxy=Proxy(prov, store, http),
                    api_keys={"buena"}, daily_paid_cap=5)
    client = TestClient(create_app(state))
    day = _today()
    r = client.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "messages": []})
    assert r.status_code == 503  # nothing actually served the client
    assert state.store.paid_usage("buena", day) == 1  # pero SI se factura


def test_streaming_served_by_a_free_route_counts_no_paid_usage():
    # (b) if the winner is the FREE route, it must not be counted as paid usage
    # even though a paid route is available in the chain.
    state, client = _free_and_paid_state(
        daily_paid_cap=5,
        make_free_response=lambda: httpx.Response(200, content=_sse("gra", "tis")),
        make_paid_response=lambda: httpx.Response(200, content=_sse("de", " pago")))
    day = _today()

    r = client.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "messages": [], "stream": True})
    assert r.status_code == 200
    assert "gra" in r.text and "tis" in r.text
    assert state.store.paid_usage("buena", day) == 0


def test_streaming_with_no_live_route_counts_no_paid_usage():
    # (c) if EVERY route fails, no paid usage must be counted either.
    state, client = _free_and_paid_state(
        daily_paid_cap=5,
        make_free_response=lambda: httpx.Response(500, json={"error": "free caida"}),
        make_paid_response=lambda: httpx.Response(500, json={"error": "pago caido"}))
    day = _today()

    r = client.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "messages": [], "stream": True})
    assert r.status_code == 200  # los headers ya salieron; el error va en el body SSE
    assert "error" in r.text
    assert state.store.paid_usage("buena", day) == 0


def test_paid_streaming_counts_once_not_per_chunk():
    # (d) a paid stream with SEVERAL chunks must increment paid_usage by exactly 1,
    # not once per chunk.
    state, client = _free_and_paid_state(
        daily_paid_cap=5,
        make_free_response=lambda: httpx.Response(500, json={"error": "free caida"}),
        make_paid_response=lambda: httpx.Response(
            200, content=_sse("u", "n", "o", "dos", "tres", "cuatro")))
    day = _today()

    r = client.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "messages": [], "stream": True})
    assert r.status_code == 200
    assert state.store.paid_usage("buena", day) == 1


# --- Fix round 2 (final), change 1: `require_api_key` also accepts the key
#     via `Authorization: Bearer <key>`, not only via `X-API-Key`. It is what lets
#     `OpenAI(base_url=..., api_key="<key>")` authenticate with no extra
#     configuration -- the contract's central promise ("change only base_url"),
#     which before this change was false: the SDK sends the key via
#     `Authorization`, and the gateway only read `X-API-Key`. `X-API-Key`
#     still exists (`arkiv-api`, the sibling gateway, uses it) and still
#     winning when both headers arrive together.

def test_it_authorises_with_bearer_and_no_x_api_key(client):
    r = client.post("/v1/chat/completions", headers={"Authorization": "Bearer buena"},
                     json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "hi"


def test_a_bearer_with_a_bad_key_still_returns_401(client):
    r = client.post("/v1/chat/completions", headers={"Authorization": "Bearer mala"},
                     json={"model": "auto", "messages": []})
    assert r.status_code == 401


def test_x_api_key_alone_still_works_as_before(client):
    r = client.get("/v1/models", headers={"X-API-Key": "buena"})
    assert r.status_code == 200


def test_when_both_arrive_and_disagree_x_api_key_wins(client):
    # X-API-Key carries the good one; Authorization carries a key that is not even
    # exist. If Authorization won, this would be a 401 -- it confirms the
    # precedencia declarada.
    r = client.get("/v1/models", headers={
        "X-API-Key": "buena", "Authorization": "Bearer ni-existe"})
    assert r.status_code == 200


def test_a_malformed_authorization_does_not_blow_up_and_returns_401(client):
    # None of these broken shapes may raise an uncaught exception: they are
    # treated the same as "no key was sent at all".
    for header in ("buena", "Bearer", "Bearer   ", "Basic buena", "buena sin bearer"):
        r = client.get("/v1/models", headers={"Authorization": header})
        assert r.status_code == 401, header


def test_the_per_minute_limit_counts_the_same_whichever_header_is_used():
    store = Storage(":memory:")
    store.create_schema()
    store.upsert_routes(
        [Route("kilo", "a:free", "free", Capabilities(True, False, 100000, 4096))], 1.0)
    prov = {"kilo": Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])}
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "hi"}}]})))
    state = State(store=store, proxy=Proxy(prov, store, http),
                    api_keys={"buena"}, daily_paid_cap=200,
                    rate_limiter=RateLimiter(2))
    client = TestClient(create_app(state))

    r1 = client.get("/v1/models", headers={"X-API-Key": "buena"})
    r2 = client.get("/v1/models", headers={"Authorization": "Bearer buena"})
    r3 = client.get("/v1/models", headers={"X-API-Key": "buena"})
    assert r1.status_code == 200 and r2.status_code == 200
    # The limit (2/min) was already exhausted between the two previous requests,
    # regardless of each having used a different header: it is the same resolved
    # key, so it counts against the same counter.
    assert r3.status_code == 429


# --- Fix round 3, I3: /v1/ranking has to carry the date of the last probe
#     (design section 6) and tell "measured quality of 0.6" from "never measured"
#     -- exactly the datum that was missing to diagnose B2. ---

def test_the_ranking_marks_a_route_without_a_quality_probe_as_unmeasured(client):
    row = client.get("/v1/ranking", headers={"X-API-Key": "buena"}).json()["routes"][0]
    assert row["quality_measured"] is False
    assert row["quality"] is None            # no se muestra el neutro como medicion
    assert row["quality_assumed"] == 0.6     # pero se dice cual se uso para puntuar
    assert row["last_quality_probe"] is None
    assert row["last_probe"] is None


def test_the_ranking_carries_the_date_of_the_last_probe(state_client):
    state, client = state_client
    # 2026-08-17T12:00:00Z y 2026-08-17T18:00:00Z
    state.store.record_probe("kilo/a:free", "quality", True, 0, 0, 200, 3, 5,
                                   1786968000.0)
    state.store.record_probe("kilo/a:free", "health", True, 120, 0, 200, 0, 0,
                                   1786989600.0)
    row = client.get("/v1/ranking", headers={"X-API-Key": "buena"}).json()["routes"][0]
    assert row["quality_measured"] is True
    assert row["quality"] == 0.6             # 3/5, esta vez SI medido
    assert row["quality_assumed"] is None
    assert row["last_quality_probe"] == "2026-08-17T12:00:00Z"
    assert row["last_probe"] == "2026-08-17T18:00:00Z"


def test_a_route_in_cooldown_does_not_lose_its_measured_quality_mark(state_client):
    # `_metrics` rebuilt Metrics positionally to inject the cooldown, and so lost
    # the newer fields: a punished route showed up as "never measured" and the
    # router sent it to the bottom twice over.
    state, client = state_client
    state.store.record_probe("kilo/a:free", "quality", True, 0, 0, 200, 5, 5,
                                   1786968000.0)
    state.proxy.cooldowns["kilo/a:free"] = time.time() + 600
    row = client.get("/v1/ranking", headers={"X-API-Key": "buena"}).json()["routes"][0]
    assert row["quality_measured"] is True
    assert row["cooldown_until"] > time.time()


# --- Fix round 3, B3 (Blocking): section 9's 503 was delivered as a 400 on every
#     outage. `order_routes` filters out the cooldowns, the list arrives empty and
#     the api shouted "no route satisfies the request" -- a 400, which every SDK and
#     every alerting layer reads as "your request is malformed": they do not retry
#     and they wake nobody. The free tiers rate-limiting at once is
#     the EXPECTED failure, not an exotic one. ---

def test_every_candidate_in_cooldown_returns_503_not_400(state_client):
    state, client = state_client
    until = time.time() + 600
    state.proxy.cooldowns["kilo/a:free"] = until
    r = client.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "messages": []})
    assert r.status_code == 503
    assert r.json()["detail"]["next_release"] == pytest.approx(until)


def test_every_candidate_in_cooldown_returns_503_when_streaming_too(state_client):
    state, client = state_client
    state.proxy.cooldowns["kilo/a:free"] = time.time() + 600
    r = client.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "messages": [], "stream": True})
    assert r.status_code == 503


def test_capabilities_nobody_satisfies_is_still_a_400(state_client):
    # The other side of the coin: this IS the client's fault and has to stay a
    # 400, with what it asked for and how many routes exist.
    state, client = state_client
    r = client.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "messages": [], "x_min_context": 99999999})
    assert r.status_code == 400
    assert r.json()["detail"]["active_routes"] == 1


def test_the_daily_paid_cap_returns_503_not_400():
    # Section 9: "a key exceeded its daily paid cap -> 503, never a silent
    # charge". With the free route in cooldown and the cap exhausted, the chain is
    # empty -- but the paid route EXISTS and could serve: this is unavailability,
    # not a malformed request.
    state, client = _free_and_paid_state(
        daily_paid_cap=1,
        make_free_response=lambda: httpx.Response(200, json={"choices": [
            {"message": {"role": "assistant", "content": "hi"}}]}),
        make_paid_response=lambda: httpx.Response(200, json={"choices": [
            {"message": {"role": "assistant", "content": "pago"}}]}))
    state.store.add_paid_usage("buena", _today())          # tope agotado
    state.proxy.cooldowns["free_prov/f:free"] = time.time() + 300

    r = client.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "messages": []})
    assert r.status_code == 503
    assert r.json()["detail"]["paid_cap_reached"] is True


def test_the_unavailability_503_reports_no_release_when_there_is_no_cooldown():
    # There is only a paid route and the client forbade it: there is nothing to
    # wait for, so proxima_liberacion is null instead of an invented number.
    state, client = _free_and_paid_state(
        daily_paid_cap=9,
        make_free_response=lambda: httpx.Response(500),
        make_paid_response=lambda: httpx.Response(500))
    state.store.upsert_routes([], 1.0, deactivate_missing=False)
    state.proxy.cooldowns["free_prov/f:free"] = time.time() + 300
    r = client.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "messages": [], "x_allow_paid": False})
    assert r.status_code == 503
    assert r.json()["detail"]["next_release"] == pytest.approx(
        state.proxy.cooldowns["free_prov/f:free"])


# --- Fix round 3, I2: section 6.1 promises to return the trimmed reasoning in a
#     separate field, `x_reasoning`. It was trimmed out of `content` and thrown
#     away: a
#     client with the default `x_raw: false` had no way to recover it. ---

def _client_that_thinks(contenido):
    store = Storage(":memory:")
    store.create_schema()
    store.upsert_routes(
        [Route("kilo", "a:free", "free", Capabilities(True, False, 100000, 4096))], 1.0)
    prov = {"kilo": Provider("kilo", "free", "openai", "https://k.test", "",
                              "/models", {}, [])}
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={"choices": [
            {"message": {"role": "assistant", "content": contenido}}]})))
    state = State(store=store, proxy=Proxy(prov, store, http),
                    api_keys={"buena"}, daily_paid_cap=200)
    return TestClient(create_app(state))


def test_it_returns_the_trimmed_reasoning_in_x_reasoning():
    client = _client_that_thinks("<think>2+2 son 4</think>La response es 4.")
    r = client.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "messages": []})
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "La response es 4."
    assert r.json()["x_reasoning"] == "2+2 son 4"


def test_without_reasoning_it_does_not_add_the_field():
    client = _client_that_thinks("La response es 4.")
    assert "x_reasoning" not in client.post(
        "/v1/chat/completions", headers={"X-API-Key": "buena"},
        json={"model": "auto", "messages": []}).json()


def test_in_raw_mode_there_is_no_x_reasoning_because_it_stays_in_the_content():
    client = _client_that_thinks("<think>mmm</think>hi")
    body = client.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                          json={"model": "auto", "messages": [], "x_raw": True}).json()
    assert body["choices"][0]["message"]["content"] == "<think>mmm</think>hi"
    assert "x_reasoning" not in body


# --- Fix round 3, ALSO: the coupling between the neutral reliability value and
#     /health's floor was load-bearing and lived in two different files, with
#     nothing testing it. If it ever inverted, a FRESH install -- with not one
#     event yet -- reported "down" and Coolify never marked the container
#     healthy: the service never started, because of a constant.
#
#     Round 6, Part 2: the mechanism this test protected (comparing
#     `NEUTRAL_RELIABILITY` contra `UMBRAL_CONFIABILIDAD_SALUD`) desaparecio
#     along with the average-based `/health` -- `UMBRAL_CONFIABILIDAD_SALUD` no
#     longer exists. The contract it protected ("a route with no telemetry counts
#     as alive") is still in force, now in `Storage.has_liveness_evidence` (see
#     test_tiene_evidencia_de_vida_sin_ninguna_telemetria en test_almacen.py)
#     y verificado end-to-end en test_health_is_ok_when_the_free_route_has_no_telemetry_yet
#     and test_health_after_a_process_restart_stays_ok_with_no_telemetry
#     arriba. ---


# --- /v1/traffic: what the gateway did, end to end over the wire


def test_a_real_request_is_traced_from_the_wire_to_the_report(state_client):
    """The whole chain in one test: a client asks, the attempt is recorded with
    what was asked, and /v1/traffic reports where it landed.

    Route-keyed rows alone could never answer this -- see Storage.traffic.
    """
    state, c = state_client
    r = c.post("/v1/chat/completions",
               headers={"Authorization": "Bearer buena"},
               json={"model": "auto:strong",
                     "messages": [{"role": "user", "content": "hola"}]})
    assert r.status_code == 200
    assert r.headers["X-Route-Used"] == "kilo/a:free"

    t = c.get("/v1/traffic", headers={"Authorization": "Bearer buena"}).json()
    assert t["requests"] == 1
    assert t["needed_failover"] == 0
    # The client asked for auto:strong; the report says which route served it --
    # the same pairing X-Route-Used gives at request time, but persisted.
    assert t["by_requested"]["auto:strong"] == {"kilo/a:free": 1}
    assert t["served"]["kilo/a:free"] == {"ok": 1, "failed": 0}


def test_traffic_needs_an_api_key(state_client):
    _, c = state_client
    assert c.get("/v1/traffic").status_code == 401


def test_traffic_window_is_clamped_not_trusted(state_client):
    """`hours` arrives from the query string; an absurd value must not turn into
    an unbounded scan on a machine that is already saturated."""
    _, c = state_client
    for hours in ("999999", "-5"):
        r = c.get(f"/v1/traffic?hours={hours}",
                  headers={"Authorization": "Bearer buena"})
        assert r.status_code == 200


def _register(state, provider_id):
    """Put `provider_id` in the process's registry.

    The /health provider block is filtered to the providers this process loaded
    from providers.yaml: a contract row is never deleted (the history is worth
    keeping, exactly as it is for routes), so an entry for a provider that has
    since been taken out of the YAML would keep reporting its last plan as
    though it were still being swept.
    """
    state.providers.append(Provider(provider_id, "free", "openai",
                                    "https://p.test/v1", "", "/models", {}, []))


def test_health_reports_each_providers_contract(state_client):
    state, client = state_client
    _register(state, "chatgpt")
    state.store.put_contract("chatgpt", {
        "contract": 1,
        "auth": {"mode": "account", "plan": "go", "subscription_active": True,
                 "expires_at": "2026-09-06T00:28:46Z"},
        "capabilities": {"images": True, "vision": True},
    }, 100.0)
    body = client.get("/health", headers={"X-API-Key": "buena"}).json()
    entry = body["providers"]["chatgpt"]
    assert entry["contract"] == 1
    assert entry["auth_mode"] == "account"
    assert entry["plan"] == "go"
    assert entry["expires_at"] == "2026-09-06T00:28:46Z"
    assert entry["reported_capabilities"]["images"] is True
    assert entry["seen_at"] == 100.0


def test_health_calls_them_reported_capabilities_not_effective_ones(state_client):
    """Design 4.4 asked for "the effective capability set" and this endpoint
    cannot honestly give one: it echoes the stored document, while what a route
    is actually served with has been through `exceptions` and `emulates_tools`
    since. Naming the key for what it holds is the fix; the effective per-route
    values are in `GET /v1/ranking`, and this key must not read like them.
    """
    state, client = state_client
    _register(state, "chatgpt")
    state.store.put_contract(
        "chatgpt", {"contract": 1, "capabilities": {"tools": True}}, 100.0)
    body = client.get("/health", headers={"X-API-Key": "buena"}).json()
    entry = body["providers"]["chatgpt"]
    assert "capabilities" not in entry
    assert entry["reported_capabilities"] == {"tools": True}


def test_health_hides_the_provider_block_from_a_keyless_caller(state_client):
    """`plan` and `expires_at` are the operator's ChatGPT tier and renewal date,
    and this gateway faces the internet. The container health check needs the
    status, not the subscription."""
    state, client = state_client
    _register(state, "chatgpt")
    state.store.put_contract(
        "chatgpt", {"contract": 1,
                    "auth": {"mode": "account", "plan": "go",
                             "expires_at": "2026-09-06T00:28:46Z"}}, 100.0)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "providers" not in body
    # ...and everything the health check and the dashboards read is still there.
    assert set(body) == {"status", "active_routes", "available_routes",
                         "free_available"}
    assert "go" not in r.text


def test_health_hides_the_provider_block_from_a_wrong_key(state_client):
    state, client = state_client
    _register(state, "chatgpt")
    state.store.put_contract("chatgpt", {"contract": 1}, 100.0)
    r = client.get("/health", headers={"X-API-Key": "mala"})
    assert r.status_code == 200
    assert "providers" not in r.json()


def test_health_does_not_report_a_provider_that_left_the_registry(state_client):
    state, client = state_client
    state.store.put_contract("openrouter", {"contract": 1}, 100.0)
    body = client.get("/health", headers={"X-API-Key": "buena"}).json()
    assert body["providers"] == {}


def test_health_reports_a_provider_without_a_contract_as_null(state_client):
    _, client = state_client
    body = client.get("/health", headers={"X-API-Key": "buena"}).json()
    assert body["providers"] == {}


def test_ranking_rows_carry_the_new_capability_axes(state_client):
    state, client = state_client
    caps = Capabilities(tools=False, vision=True, context=52815, max_output=8192,
                        audio_speech=True, translate=True, search=True)
    state.store.upsert_routes([Route("chatgpt", "gpt-5-6", "free", caps)], time.time())
    rows = client.get("/v1/ranking", headers={"X-API-Key": "buena"}).json()["routes"]
    row = [r for r in rows if r["key"] == "chatgpt/gpt-5-6"][0]
    assert row["audio_speech"] is True
    assert row["audio_transcription"] is False
    assert row["translate"] is True
    assert row["search"] is True


def test_health_reports_a_contract_with_no_auth_or_capabilities_block(state_client):
    """A minimal document -- just the version -- must not blow up the field
    lookups: every field degrades to None/empty rather than raising."""
    state, client = state_client
    _register(state, "x")
    state.store.put_contract("x", {"contract": 1}, 100.0)
    body = client.get("/health", headers={"X-API-Key": "buena"}).json()
    entry = body["providers"]["x"]
    assert entry["auth_mode"] is None
    assert entry["plan"] is None
    assert entry["expires_at"] is None
    assert entry["reported_capabilities"] == {}


def test_health_degrades_a_malformed_auth_block_instead_of_raising(state_client):
    """`parse_health` deliberately tolerates a non-dict `auth` (nothing routes
    on it) and still persists the document, so a truthy non-dict really can
    reach this endpoint. /health is the container health check: a 500 here
    restarts the process, which re-reads the same row and 500s again."""
    state, client = state_client
    _register(state, "x")
    state.store.put_contract(
        "x", {"contract": 1, "auth": "pending", "capabilities": []}, 100.0)
    r = client.get("/health", headers={"X-API-Key": "buena"})
    assert r.status_code == 200
    entry = r.json()["providers"]["x"]
    assert entry["auth_mode"] is None
    assert entry["reported_capabilities"] == {}


# --- x_requires reached only two flags (2026-09-01) -------------------------
#
# `parse_request` consumed the parsed set for `tools` and `vision` and dropped
# everything else on the floor, in silence, even though RouteRequest already had
# the `needs_*` fields and `_satisfies` already had the branches -- they were only
# ever set by the capability ENDPOINTS. And `search` had no flag at all.
#
# Measured against the live gateway before the fix: `x_requires: ["translate"]`
# was served by grok/grok-plugins-4p6-powerpoint (translate=false), and
# `x_requires: ["images","translate","search"]` by
# kilo/nvidia/nemotron-3.5-lightning:free (false on all three). A requirement that
# is quietly ignored is worse than one that is rejected: the caller believes it
# was honoured.

def test_x_requires_search_reaches_its_flag():
    p = parse_request({"model": "auto", "x_requires": ["search"]})
    assert p.needs_search is True


@pytest.mark.parametrize("name", ["images", "audio_speech", "audio_transcription",
                                  "translate"])
def test_an_endpoint_only_capability_in_x_requires_returns_400(name):
    # These four are decided by the URL that was called and by nothing a client
    # can put in a body -- see the comment above `needs_images`. Asking for one
    # from a chat body is a category error, and saying so is the whole point:
    # the previous behaviour accepted the word and ignored it.
    with pytest.raises(HTTPException) as exc:
        parse_request({"model": "auto", "x_requires": [name]})
    assert exc.value.status_code == 400
    assert exc.value.detail["field"] == "x_requires"


def test_an_unknown_capability_in_x_requires_returns_400_naming_the_value():
    # A typo used to be indistinguishable from a satisfied requirement. And the
    # 400 has to say WHICH word was wrong: routing this through _read_field made
    # it answer "must be a string or a list of strings" for a value that was
    # already a list of strings, which sends the caller hunting the wrong bug.
    with pytest.raises(HTTPException) as exc:
        parse_request({"model": "auto", "x_requires": ["vison"]})
    assert exc.value.status_code == 400
    assert exc.value.detail["field"] == "x_requires"
    assert "vison" in exc.value.detail["message"]
    assert "string" not in exc.value.detail["message"]


def test_the_two_capabilities_that_already_worked_keep_working():
    p = parse_request({"model": "auto", "x_requires": ["tools", "vision"]})
    assert p.needs_tools and p.needs_vision
    assert p.needs_search is False


# --- The streamed response could not say WHO served it (2026-09-02) -----------
#
# `X-Route-Used` cannot travel on a stream: headers go out before the failover
# chain resolves. The per-chunk `model` the provider sends is not enough on its
# own, because the same model id can exist at several providers -- which is the
# whole reason routing is by model rather than by provider. So a streaming
# client could not attribute an answer at all.
#
# The same pass also drops provider-private fields: perplexity leaks a `_pplx`
# object into its final chunk, inside a payload that is otherwise the OpenAI
# contract.

def _stream_lines(text):
    return [l for l in text.split("\n\n") if l.strip()]


def test_a_streamed_answer_names_the_route_that_served_it():
    state, client = _free_and_paid_state(
        daily_paid_cap=0,
        make_free_response=lambda: httpx.Response(200, content=_sse("ho", "la")),
        make_paid_response=lambda: httpx.Response(500, json={"error": "x"}))
    r = client.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "messages": [], "stream": True})
    lines = _stream_lines(r.text)

    assert lines[-1] == "data: [DONE]"
    attribution = json.loads(lines[-2][6:])
    # provider/model, not just the model: that is the point of the field.
    assert attribution["x_route"] == "free_prov/f:free"
    assert attribution["x_tier"] == "free"
    # OpenAI's own convention for a final metadata-only chunk, so an SDK that
    # already tolerates its usage chunk tolerates this one.
    assert attribution["choices"] == []


def test_nothing_is_attributed_when_no_route_served():
    state, client = _free_and_paid_state(
        daily_paid_cap=0,
        make_free_response=lambda: httpx.Response(500, json={"error": "x"}),
        make_paid_response=lambda: httpx.Response(500, json={"error": "x"}))
    r = client.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "messages": [], "stream": True})
    assert "x_route" not in r.text


def test_a_provider_private_field_does_not_reach_the_client():
    leaky = ('data: {"choices":[{"delta":{"content":"hi"}}],'
             '"_pplx":{"display_model":"turbo"}}\n\n'
             'data: [DONE]\n\n').encode()
    state, client = _free_and_paid_state(
        daily_paid_cap=0,
        make_free_response=lambda: httpx.Response(200, content=leaky),
        make_paid_response=lambda: httpx.Response(500, json={"error": "x"}))
    r = client.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "messages": [], "stream": True})
    assert "hi" in r.text          # the answer still arrives
    assert "_pplx" not in r.text   # the provider's own metadata does not
