import json
import logging
import time

import httpx
import pytest

from llm_libre import probing
from llm_libre.contract import REQUIRED_CAPABILITIES
from llm_libre.storage import Storage
from llm_libre.api import State
from llm_libre.models import Capabilities, Route
from llm_libre.providers import Provider
from llm_libre.quality_suite import CASES
from llm_libre.proxy import Proxy
from llm_libre.quality_suite import SHORT_TOKEN_BUDGET
from llm_libre.probing import (HEALTH_FLOOR_S, PING, QUALITY_INTERVAL_S, cycle,
                               probe_health, probe_quality, sync_catalogue)

CATALOGUE = {"data": [
    {"id": "x:free", "pricing": {"prompt": "0"}, "context_length": 1000,
     "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
     "supported_parameters": ["tools"], "top_provider": {"max_completion_tokens": 100}}]}


def _store():
    a = Storage(":memory:")
    a.create_schema()
    return a


def _route(model="x:free", tier="free", provider="kilo", tools=True):
    return Route(provider, model, tier, Capabilities(tools, False, 1000, 100))


def _proxy(handler):
    prov = {"kilo": Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])}
    return Proxy(prov, _store(), httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def test_sync_stores_the_discovered_routes():
    store = _store()
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=CATALOGUE)))
    prov = [Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])]
    await sync_catalogue(http, prov, store, now=100.0)
    assert [r.key for r in store.active_routes()] == ["kilo/x:free"]


async def test_sync_does_not_splinter_a_query_string_in_base_url():
    # Round 5: the same concatenated-suffix bug, on the /models side -- the test
    # has to look at the URL actually requested, not at an intermediate field.
    store = _store()
    seen_urls = []

    def handler(req):
        seen_urls.append(str(req.url))
        return httpx.Response(200, json=CATALOGUE)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prov = [Provider("chatgpt", "free", "openai", "https://blog.test:8888?token=abc",
                      "", "/models", {}, [])]
    await sync_catalogue(http, prov, store, now=100.0)
    assert seen_urls == ["https://blog.test:8888/models?token=abc"]


async def test_sync_adds_the_paid_fixed_models():
    store = _store()
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=CATALOGUE)))
    prov = [Provider("minimax", "paid", "openai", "https://m.test", "k", "",
                      {}, [{"id": "MiniMax-M3", "tools": True, "vision": False,
                            "context": 128000, "max_output": 32768}])]
    await sync_catalogue(http, prov, store, now=100.0)
    routes = store.active_routes()
    assert [r.key for r in routes] == ["minimax/MiniMax-M3"]
    assert routes[0].tier == "paid"


async def test_sync_supports_fixed_models_and_dynamic_discovery_together():
    """A provider may declare fixed routes (e.g. dall-e-3) AND discover chat
    models via models_path. Both sets must be active after a single sync pass."""
    store = _store()
    bare_catalogue = {"data": [
        {"id": "gpt-5-3-mini", "object": "model", "description": "GPT-5.3 Mini"},
    ]}
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=bare_catalogue)))
    prov = [Provider("chatgpt", "free", "openai", "https://cg.test", "", "/models",
                     {}, [{"id": "dall-e-3", "tools": False, "vision": False,
                           "images": True, "context": 128000, "max_output": 0}],
                     priority=0,
                     default_capabilities=Capabilities(False, False, 128000, 8192))]
    await sync_catalogue(http, prov, store, now=100.0)
    routes = {r.model_id: r for r in store.active_routes()}
    assert "dall-e-3" in routes, "fixed image route missing"
    assert "gpt-5-3-mini" in routes, "dynamically discovered chat route missing"
    assert routes["dall-e-3"].capabilities.images is True
    assert routes["gpt-5-3-mini"].capabilities.images is False


async def test_sync_propagates_the_providers_priority_to_discovered_routes():
    store = _store()
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=CATALOGUE)))
    prov = [Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [],
                      priority=1)]
    await sync_catalogue(http, prov, store, now=100.0)
    routes = store.active_routes()
    assert routes[0].priority == 1


async def test_sync_propagates_the_providers_default_capabilities():
    # Simulates chatgpt-proxy: /models carries ids but NO capability metadata --
    # sync_catalogue has to apply the provider's declared capabilities to every
    # discovered id, tools:false included.
    store = _store()
    bare_catalogue = {"data": [
        {"id": "gpt-5-3-mini", "object": "model", "description": "GPT-5.3 Mini"},
    ]}
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=bare_catalogue)))
    prov = [Provider("chatgpt", "free", "openai", "https://cg.test", "", "/models", {}, [],
                      priority=0,
                      default_capabilities=Capabilities(False, False, 128000, 8192))]
    await sync_catalogue(http, prov, store, now=100.0)
    routes = store.active_routes()
    assert [r.key for r in routes] == ["chatgpt/gpt-5-3-mini"]
    assert routes[0].capabilities.tools is False
    assert routes[0].capabilities.context == 128000
    assert routes[0].priority == 0


async def test_a_down_provider_does_not_erase_the_others_catalogue():
    store = _store()
    store.upsert_routes([_route("previous:free")], timestamp=50.0)

    def handler(req):
        return httpx.Response(500)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prov = [Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])]
    await sync_catalogue(http, prov, store, now=100.0)
    # If /models fails nothing is deactivated: an old catalogue beats an empty one.
    assert len(store.active_routes()) == 1


async def test_a_partial_failure_does_not_corrupt_the_timestamp_of_what_was_discovered():
    # Deviation from the brief: `upsert_routes(discovered, now if not failed else 0.0)`
    # overwrote visto_por_ultima_vez with 0.0 for the routes of providers that DID
    # answer too, not only for the missing ones. upsert_routes' third parameter
    # (deactivate_missing) exists precisely to separate "do not switch off what
    # never arrived" from "corrupt the timestamp of what did".
    store = _store()

    def handler(req):
        if "k.test" in str(req.url):
            return httpx.Response(200, json=CATALOGUE)
        return httpx.Response(500)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prov = [
        Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, []),
        Provider("roto", "free", "openai", "https://roto.test", "", "/models", {}, []),
    ]
    await sync_catalogue(http, prov, store, now=100.0)
    row = store._con.execute(
        "SELECT last_seen FROM routes WHERE key='kilo/x:free'").fetchone()
    assert row[0] == 100.0


EMPTY_CATALOGUE = {"data": []}

# A genuine 200, but whose only model is paid: normalize() filters it out and the
# usable result is zero too, even though the response "has data".
CATALOGUE_ALL_FILTERED = {"data": [
    {"id": "pago:model", "pricing": {"prompt": "0.002"}, "context_length": 1000,
     "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
     "supported_parameters": [], "top_provider": {"max_completion_tokens": 100}}]}


async def test_a_200_with_empty_data_does_not_erase_the_previous_routes():
    # Finding 1: a 200 with zero usable models is more likely a broken provider
    # than a genuinely empty catalogue -- it must not authorise the UPDATE that
    # deactivates everything already known about it.
    store = _store()
    store.upsert_routes([_route("previous:free")], timestamp=50.0)
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=EMPTY_CATALOGUE)))
    prov = [Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])]
    await sync_catalogue(http, prov, store, now=100.0)
    assert [r.key for r in store.active_routes()] == ["kilo/previous:free"]


async def test_a_200_whose_models_are_all_filtered_out_does_not_erase_previous_routes():
    store = _store()
    store.upsert_routes([_route("previous:free")], timestamp=50.0)
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=CATALOGUE_ALL_FILTERED)))
    prov = [Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])]
    await sync_catalogue(http, prov, store, now=100.0)
    assert [r.key for r in store.active_routes()] == ["kilo/previous:free"]


async def test_un_proveedor_vacio_no_frena_la_actualizacion_del_que_si_respondio():
    store = _store()
    store.upsert_routes([_route("previous:free", provider="empty")], timestamp=50.0)

    def handler(req):
        if "k.test" in str(req.url):
            return httpx.Response(200, json=CATALOGUE)
        return httpx.Response(200, json=EMPTY_CATALOGUE)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prov = [
        Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, []),
        Provider("empty", "free", "openai", "https://empty.test", "", "/models", {}, []),
    ]
    await sync_catalogue(http, prov, store, now=100.0)
    keys_ = {r.key for r in store.active_routes()}
    assert "kilo/x:free" in keys_          # the provider that did answer was updated
    assert "empty/previous:free" in keys_    # the empty one kept what it had


async def test_a_healthy_provider_deactivates_its_own_own_stale_route_even_if_another_is_empty():
    # Finding 1, case (c) -- the one that separates a real fix from a blunt one.
    # kilo had an old route that no longer appears in its new catalogue; the empty
    # one answers 200 with no models. That must not hold back the removal
    # of kilo/old:free: kilo answered correctly and its own disappearance is real.
    store = _store()
    store.upsert_routes([_route("old:free", provider="kilo")], timestamp=50.0)
    store.upsert_routes([_route("previous:free", provider="empty")], timestamp=50.0)

    def handler(req):
        if "k.test" in str(req.url):
            return httpx.Response(200, json=CATALOGUE)   # ya no trae old:free, solo x:free
        return httpx.Response(200, json=EMPTY_CATALOGUE)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prov = [
        Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, []),
        Provider("empty", "free", "openai", "https://empty.test", "", "/models", {}, []),
    ]
    await sync_catalogue(http, prov, store, now=100.0)
    keys_ = {r.key for r in store.active_routes()}
    assert keys_ == {"kilo/x:free", "empty/previous:free"}
    assert "kilo/old:free" not in keys_    # deactivated: kilo answered, and no longer carries it


async def test_a_healthy_provider_does_not_deactivate_another_providers_routes():
    # kilo's deactivation must be SCOPED to kilo: an old route belonging to
    # "other" (much older than `now`) must not fall victim to kilo's UPDATE. "other"
    # is STILL registered (with neither modelos_path nor modelos_fijos, so the
    # per-provider loop skips it without touching it) -- if it were not in `prov`,
    # the orphaned-provider sweep (Storage.deactivate_unregistered_providers, see
    # the separate block of tests below) would switch it off for a DIFFERENT
    # reason, and this test would stop isolating what it claims to test.
    store = _store()
    store.upsert_routes([_route("old:free", provider="other")], timestamp=10.0)
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=CATALOGUE)))
    prov = [
        Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, []),
        Provider("other", "free", "openai", "https://other.test", "", "", {}, []),
    ]
    await sync_catalogue(http, prov, store, now=100.0)
    keys_ = {r.key for r in store.active_routes()}
    assert keys_ == {"kilo/x:free", "other/old:free"}


SHARED_CATALOGUE = {"data": [
    {"id": "shared:free", "pricing": {"prompt": "0"}, "context_length": 1000,
     "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
     "supported_parameters": [], "top_provider": {"max_completion_tokens": 100}}]}


async def test_the_same_model_in_two_providers_only_the_one_that_lost_it_is_switched_off():
    # The reason for "routes, not models": the same modelo_id can exist in two
    # providers as two independent rows (different keys). If one stops offering it
    # and the other keeps it, they must be treated separately.
    store = _store()
    store.upsert_routes([_route("shared:free", provider="kilo")], timestamp=50.0)
    store.upsert_routes([_route("shared:free", provider="other")], timestamp=50.0)

    def handler(req):
        if "k.test" in str(req.url):
            return httpx.Response(200, json=CATALOGUE)          # kilo ya no trae shared:free
        return httpx.Response(200, json=SHARED_CATALOGUE)   # the other provider still returns it

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prov = [
        Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, []),
        Provider("other", "free", "openai", "https://o.test", "", "/models", {}, []),
    ]
    await sync_catalogue(http, prov, store, now=100.0)
    keys_ = {r.key for r in store.active_routes()}
    assert "kilo/shared:free" not in keys_   # kilo lo perdio: se apaga
    assert "other/shared:free" in keys_       # the other provider keeps it: still active
    assert "kilo/x:free" in keys_


async def test_a_200_with_a_non_json_body_is_a_failure_and_does_not_stop_the_others():
    # Finding 2: r.json() can raise JSONDecodeError (a ValueError subclass) on a
    # non-JSON body; that must neither escape sync_catalogue nor stop the
    # processing of the providers that come after.
    store = _store()

    def handler(req):
        if "roto.test" in str(req.url):
            return httpx.Response(200, text="this is not json")
        return httpx.Response(200, json=CATALOGUE)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prov = [
        Provider("roto", "free", "openai", "https://roto.test", "", "/models", {}, []),
        Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, []),
    ]
    await sync_catalogue(http, prov, store, now=100.0)
    assert [r.key for r in store.active_routes()] == ["kilo/x:free"]


async def test_a_200_with_an_unexpected_shape_is_a_failure_and_does_not_stop_the_others():
    # Valid JSON but of a shape normalize() does not expect (e.g. an auth error
    # served with status 200): it raises AttributeError inside normalize(), not an
    # httpx.HTTPError.
    store = _store()

    def handler(req):
        if "roto.test" in str(req.url):
            return httpx.Response(200, json={"error": "unauthorized"})
        return httpx.Response(200, json=CATALOGUE)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prov = [
        Provider("roto", "free", "openai", "https://roto.test", "", "/models", {}, []),
        Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, []),
    ]
    await sync_catalogue(http, prov, store, now=100.0)
    assert [r.key for r in store.active_routes()] == ["kilo/x:free"]


# --- Removing a provider from providers.yaml (the real case: openrouter, with
#     no OPENROUTER_API_KEY, whose 16 routes 401'd every time and only took up
#     probe quota and ranking space proving they were dead) must not leave its
#     routes at `active=1` forever -- see
#     Storage.deactivate_unregistered_providers. This is the "through
#     sync_catalogue" half of that coverage; the other half (the Storage method in
#     isolation) lives in test_storage.py. ---

async def test_sync_deactivates_the_routes_of_a_provider_removed_from_the_registry():
    store = _store()
    store.upsert_routes([_route("previous:free", provider="openrouter")], timestamp=50.0)
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=CATALOGUE)))
    # openrouter is NO LONGER in the list providers.load() returns -- this
    # simulates having removed it from providers.yaml.
    prov = [Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])]
    await sync_catalogue(http, prov, store, now=100.0)
    keys_ = {r.key for r in store.active_routes()}
    assert keys_ == {"kilo/x:free"}
    row = store._con.execute(
        "SELECT active FROM routes WHERE key = 'openrouter/previous:free'").fetchone()
    assert row == (0,)   # switched off, not deleted


async def test_sync_does_not_deactivate_routes_of_still_registered_providers():
    # Deliberately isolated from the NORMAL per-provider removal logic (already
    # covered by the tests above, "whoever answered updates its own"): minimax
    # uses modelos_fijos, so its route is re-declared IDENTICALLY on every pass
    # regardless of any HTTP mock -- if it survives, that is proof the
    # orphan sweep (`deactivate_unregistered_providers`) did not touch it, not
    # that it "happened to reappear in the catalogue".
    store = _store()
    store.upsert_routes([_route("previous:free", provider="openrouter")], timestamp=50.0)
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=CATALOGUE)))
    prov = [
        Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, []),
        Provider("minimax", "paid", "openai", "https://m.test", "k", "", {},
                  [{"id": "MiniMax-M3", "tools": True, "vision": False,
                    "context": 128000, "max_output": 32768}]),
    ]
    await sync_catalogue(http, prov, store, now=100.0)
    keys_ = {r.key for r in store.active_routes()}
    # kilo was rediscovered, minimax (modelos_fijos, STILL registered) stays
    # active, and openrouter (removed from the registry) is gone -- all three at
    # once, with none of them confused for another.
    assert keys_ == {"kilo/x:free", "minimax/MiniMax-M3"}


async def test_sync_with_an_empty_registry_does_not_switch_off_the_whole_catalogue():
    # Gate review: `providers=[]` (a syntactically valid but truncated or badly
    # edited `providers.yaml`, more likely than a deliberate "no providers")
    # must not trigger the orphan sweep -- see the guard in
    # Storage.deactivate_unregistered_providers.
    store = _store()
    store.upsert_routes([_route("previous:free", provider="kilo")], timestamp=50.0)
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=CATALOGUE)))
    await sync_catalogue(http, [], store, now=100.0)
    assert {r.key for r in store.active_routes()} == {"kilo/previous:free"}


async def test_sync_catalogue_does_not_let_a_broken_providers_exception_escape():
    # Two different flavours of "broken provider" (a non-JSON body and JSON of an
    # unexpected shape) alongside a healthy one: if the exception escaped, this
    # await would not even finish and the test would fail with an error from
    # collection, not by an assert.
    store = _store()

    def handler(req):
        url = str(req.url)
        if "nojson.test" in url:
            return httpx.Response(200, text="<html>not json</html>")
        if "odd.test" in url:
            return httpx.Response(200, json={"error": "unauthorized"})
        return httpx.Response(200, json=CATALOGUE)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prov = [
        Provider("nojson", "free", "openai", "https://nojson.test", "", "/models", {}, []),
        Provider("odd", "free", "openai", "https://odd.test", "", "/models", {}, []),
        Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, []),
    ]
    total = await sync_catalogue(http, prov, store, now=100.0)
    assert total == 1
    assert [r.key for r in store.active_routes()] == ["kilo/x:free"]


async def test_the_health_probe_records_success():
    p = _proxy(lambda req: httpx.Response(200, json={
        "choices": [{"message": {"content": "ok"}}]}))
    await probe_health(p, p.store, [_route()], now=100.0)
    row = p.store._con.execute(
        "SELECT kind, ok FROM probes WHERE key='kilo/x:free'").fetchone()
    assert row == ("health", 1)


async def test_the_health_probe_records_the_failure_of_a_model_that_no_longer_exists():
    p = _proxy(lambda req: httpx.Response(404, json={"error": "model_not_found"}))
    await probe_health(p, p.store, [_route()], now=100.0)
    row = p.store._con.execute(
        "SELECT ok FROM probes WHERE kind='health'").fetchone()
    assert row[0] == 0


# --- Round 9, HIGH 1 from the gate: probe_health called proxy.complete WITHOUT
#     is_probe=True -- round 8 gated direct punishment behind that flag, and this
#     caller was never updated. A periodic probe runs ONCE every 5h per route;
#     without is_probe=True a failure only accumulated suspicion (which needs 3
#     consecutive ones), so 20 periodic probes against a dead route (100h) left 20
#     `probes ok=0` rows and ZERO cooldowns -- the probe had lost its authority to
#     exclude. With is_probe=True, ONE failed probe already punishes. ---

async def test_the_health_probe_punishes_immediately_on_a_single_failure():
    p = _proxy(lambda req: httpx.Response(500))
    await probe_health(p, p.store, [_route()], now=100.0)
    assert p.cooldowns["kilo/x:free"] > 0.0
    # And without going through suspicion -- there is no accumulated mark waiting
    # for a second or third failure.
    assert p.store._con.execute(
        "SELECT COUNT(*) FROM probes WHERE kind='health' AND ok=0").fetchone()[0] == 1


async def test_the_quality_probe_stores_passed_over_total_cases():
    p = _proxy(lambda req: httpx.Response(200, json={
        "choices": [{"message": {"content": "12"}}]}))
    await probe_quality(p, p.store, [_route()], now=100.0)
    row = p.store._con.execute(
        "SELECT cases_passed, cases_total FROM probes WHERE kind='quality'").fetchone()
    # POINTS, not a count of cases: a discriminating case is worth
    # DISCRIMINATING_WEIGHT and a liveness case 1, so the denominator is the sum
    # of the weights of the cases that ran, not len(CASES). See quality_suite.evaluate.
    assert row[1] == sum(c.weight for c in CASES)
    # This stub answers "12" to everything: it passes arithmetic and format and
    # fails json, tools and all three discriminating cases.
    assert 1 <= row[0] < row[1]


# --- Round 10, small fix from the gate: the same wiring hole as HIGH 1 (round
#     9), one function further along -- `probe_quality` called complete() without
#     is_probe=True, so a failed battery case fed `_suspect` (meant for CLIENT
#     traffic) and burned quota from the on-demand probe budget, which is scarce
#     and shared with real traffic. `case.body` is as gateway-authored as PING. ---

async def test_the_quality_probe_punishes_directly_without_going_through_suspicion():
    p = _proxy(lambda req: httpx.Response(500))
    await probe_quality(p, p.store, [_route()], now=100.0)
    assert p.cooldowns["kilo/x:free"] > 0.0
    # Directly -- never via suspicion (which would need SUSPICION_THRESHOLD
    # failures, and would additionally fire a separate 'salud' probe).
    assert p.store._con.execute(
        "SELECT COUNT(*) FROM probes WHERE kind='health'").fetchone()[0] == 0


async def test_quality_does_not_probe_paid_routes():
    p = _proxy(lambda req: httpx.Response(200, json={
        "choices": [{"message": {"content": "12"}}]}))
    await probe_quality(p, p.store, [_route(tier="paid")], now=100.0)
    assert p.store._con.execute(
        "SELECT COUNT(*) FROM probes WHERE kind='quality'").fetchone()[0] == 0


async def test_quality_skips_the_tools_case_without_counting_it_as_a_failure():
    # A route that does not declare tool support must not look as bad in the
    # score as one that declares it and gets it wrong: the case is skipped
    # entirely (counting toward neither passed nor total), not marked as failed.
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(200, json={"choices": [{"message": {"content": "12"}}]})

    p = _proxy(handler)
    await probe_quality(p, p.store, [_route(tools=False)], now=100.0)
    row = p.store._con.execute(
        "SELECT cases_passed, cases_total FROM probes WHERE kind='quality'").fetchone()
    # The skipped case leaves BOTH sides of the fraction alone: the denominator is
    # every weight EXCEPT the tools case's, not the full battery's.
    tools_weight = next(c.weight for c in CASES if c.name == "tools")
    assert row[1] == sum(c.weight for c in CASES) - tools_weight
    assert len(calls) == len(CASES) - 1   # and no quota was spent asking for it


async def test_cycle_syncs_probes_health_and_probes_quality_on_cycle_zero():
    store = _store()

    # The handler has to answer a CHAT request with a chat response. It used to
    # return the catalogue to everything, which the health probe reads as "no
    # answer" -- harmless while the battery ran unconditionally, and now decisive:
    # the battery only measures routes whose health probe just succeeded.
    def handler(req):
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json=CATALOGUE)
        return httpx.Response(200, json={"choices": [{"message": {"content": "12"}}]})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prov = [Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])]
    proxy = Proxy({"kilo": prov[0]}, store, http)
    state = State(store=store, proxy=proxy, api_keys=set(), daily_paid_cap=0,
                    providers=prov, http=http)
    await cycle(state, counter=0)
    assert [r.key for r in store.active_routes()] == ["kilo/x:free"]
    kinds = {t for (t,) in store._con.execute("SELECT kind FROM probes").fetchall()}
    assert kinds == {"health", "quality"}


async def test_cycle_does_not_probe_quality_outside_the_interval():
    store = _store()
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=CATALOGUE)))
    prov = [Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])]
    proxy = Proxy({"kilo": prov[0]}, store, http)
    state = State(store=store, proxy=proxy, api_keys=set(), daily_paid_cap=0,
                    providers=prov, http=http)
    await cycle(state, counter=1)
    kinds = {t for (t,) in store._con.execute("SELECT kind FROM probes").fetchall()}
    assert kinds == {"health"}


# --- Fix round 3, B4 (Blocking): section 8 says "paid routes are NOT probed".
#     probe_quality already filtered by tier; probe_health did not, and it receives
#     active_routes(), which includes minimax/MiniMax-M3. That was ~5 billable
#     calls a day, invisible to add_paid_usage, /v1/usage and DAILY_PAID_CAP. ---

async def test_the_health_probe_does_not_spend_money_on_paid_routes():
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    p = _proxy(handler)
    await probe_health(p, p.store, [_route(tier="paid")], now=100.0)
    assert calls == []
    assert p.store._con.execute("SELECT COUNT(*) FROM probes").fetchone()[0] == 0


async def test_the_health_probe_still_probes_the_free_routes_of_the_same_list():
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    p = _proxy(handler)
    await probe_health(p, p.store, [_route("g:free"), _route("P", tier="paid")],
                        now=100.0)
    assert len(calls) == 1
    keys_ = [c for (c,) in p.store._con.execute("SELECT key FROM probes")]
    assert keys_ == ["kilo/g:free"]


async def test_the_full_cycle_does_not_probe_the_paid_route():
    # The real case: `cycle` passes active_routes() to probe_health, and
    # minimax/MiniMax-M3 comes along in there from the YAML's modelos_fijos.
    store = _store()
    calls = []

    def handler(req):
        calls.append(str(req.url))
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json=CATALOGUE)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prov = [
        Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, []),
        Provider("minimax", "paid", "openai", "https://m.test", "k", "", {},
                  [{"id": "MiniMax-M3", "tools": True, "vision": False,
                    "context": 128000, "max_output": 32768}]),
    ]
    proxy = Proxy({p.id: p for p in prov}, store, http)
    state = State(store=store, proxy=proxy, api_keys=set(), daily_paid_cap=0,
                    providers=prov, http=http)
    await cycle(state, counter=1)   # counter 1: health yes, quality no
    assert {r.key for r in store.active_routes()} == {"kilo/x:free", "minimax/MiniMax-M3"}
    assert not any("m.test" in u for u in calls), "se le pego a la ruta de pago"


# --- Fix round 3, I4: `sync_catalogue` failed in absolute silence -- four
#     `continue` branches (network error, non-200 status, malformed body, empty
#     catalogue) and not one log line in the whole module. If a provider starts
#     failing, its catalogue freezes forever and nothing says so. This is exactly
#     the layer that exists to prevent stale catalogues. ---

def _prov_kilo():
    return [Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])]


async def _sincronizar_con(handler, caplog):
    store = _store()
    store.upsert_routes([_route("previous:free")], timestamp=50.0)
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with caplog.at_level(logging.WARNING, logger="llm_libre.probing"):
        await sync_catalogue(http, _prov_kilo(), store, now=100.0)
    return caplog.text


async def test_a_network_error_is_logged_with_the_provider_and_the_reason(caplog):
    def handler(req):
        raise httpx.ConnectError("no hay ruta al host")

    text = await _sincronizar_con(handler, caplog)
    assert "kilo" in text and "no hay ruta al host" in text


async def test_a_non_200_status_is_logged(caplog):
    text = await _sincronizar_con(lambda req: httpx.Response(503), caplog)
    assert "kilo" in text and "503" in text


async def test_a_malformed_body_is_logged(caplog):
    text = await _sincronizar_con(
        lambda req: httpx.Response(200, text="<html>mantenimiento</html>"), caplog)
    assert "kilo" in text
    assert "could not interpret" in text


async def test_an_empty_catalogue_is_logged(caplog):
    text = await _sincronizar_con(
        lambda req: httpx.Response(200, json=EMPTY_CATALOGUE), caplog)
    assert "kilo" in text
    assert "zero usable models" in text


async def test_a_healthy_sync_does_not_pollute_the_warning_logs(caplog):
    store = _store()
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=CATALOGUE)))
    with caplog.at_level(logging.WARNING, logger="llm_libre.probing"):
        await sync_catalogue(http, _prov_kilo(), store, now=100.0)
    assert caplog.text == ""


async def test_the_health_probe_does_not_write_a_ttft_it_never_measured():
    # I5: the health probe is non-streaming, so its number is a complete
    # round-trip, not a ttft. It goes to latencia_ms; the ttft column stays 0 and
    # the ttft p50 ignores it.
    p = _proxy(lambda req: httpx.Response(200, json={
        "choices": [{"message": {"content": "ok"}}]}))
    await probe_health(p, p.store, [_route()], now=100.0)
    row = p.store._con.execute(
        "SELECT latency_ms, ttft_ms FROM probes WHERE kind='health'").fetchone()
    assert row[0] >= 0
    assert row[1] == 0


# --- Fix round 4, N1 (Blocking): the health PING had max_tokens=8, four times
#     less than the 32 the battery had already proven insufficient. While an empty
#     200 counted as success it was harmless; since it became a FAILED attempt
#     (fix B1), a ping that leaves the model no room to think MANUFACTURES the
#     failure it claims to measure. Measured against Kilo: 5 of 11 free routes
#     "dead" with max_tokens=8, including the one that serves `auto` on a cold
#     start. ---

def _a_reasoning_model(min_budget=512, reply="pong"):
    """A provider that behaves like the real free models: when the budget is not
    enough, it burns it thinking and returns a 200 with
    finish_reason 'length' y content null."""
    def handler(req):
        body = json.loads(req.content)
        if body.get("max_tokens", 0) < min_budget:
            return httpx.Response(200, json={"choices": [
                {"message": {"role": "assistant", "content": None},
                 "finish_reason": "length"}]})
        return httpx.Response(200, json={"choices": [
            {"message": {"role": "assistant", "content": reply},
             "finish_reason": "stop"}]})
    return handler


async def test_the_health_probe_does_not_kill_a_model_by_denying_it_room_to_think():
    p = _proxy(_a_reasoning_model())
    await probe_health(p, p.store, [_route()], now=100.0)
    row = p.store._con.execute(
        "SELECT ok FROM probes WHERE kind='health'").fetchone()
    assert row[0] == 1, ("el ping fabrico el fallo que dice medir: no le dio "
                          "the model a big enough budget")


async def test_the_health_ping_cannot_lag_behind_the_batterys_cap():
    # Both caps exist for the same reason; if the battery goes up and the ping
    # does not, this exact bug comes back.
    assert PING["max_tokens"] >= SHORT_TOKEN_BUDGET


async def test_the_health_probe_stores_the_providers_code_not_the_gateways():
    # A 200 that arrives empty is STILL a failure (fix B1), but what has to be
    # stored in the table is the 200 the provider gave, not the 503 the gateway
    # synthesises: without that there is no way to tell "the provider is down"
    # from "it answered fine but with nothing inside".
    p = _proxy(lambda req: httpx.Response(200, json={"choices": [
        {"message": {"role": "assistant", "content": None},
         "finish_reason": "length"}]}))
    await probe_health(p, p.store, [_route()], now=100.0)
    row = p.store._con.execute(
        "SELECT ok, http_code FROM probes WHERE kind='health'").fetchone()
    assert row == (0, 200)


async def test_the_health_probe_stores_the_real_404_of_a_model_that_no_longer_exists():
    p = _proxy(lambda req: httpx.Response(404, json={"error": "model_not_found"}))
    await probe_health(p, p.store, [_route()], now=100.0)
    row = p.store._con.execute(
        "SELECT ok, http_code FROM probes WHERE kind='health'").fetchone()
    assert row == (0, 404)


# --- The battery does not measure a route that just failed its health probe ---
#
# `probe_quality` spent five requests per route unconditionally, including on
# routes that had just failed the one-request health probe moments earlier in the
# same cycle. That is expensive in two different ways, and both were observed in
# production on 2026-08-19.
#
# It BURNS scarce quota. DeepSeek muted the account, and the battery kept spending
# ten requests a run (five per route, two routes) proving a muted account is muted.
#
# And it POISONS the measurement. Those runs record 0/5, which is indistinguishable
# from "this model answers badly" -- so chatgpt read quality 0.333 and deepseek
# 0.00 while both were perfectly healthy models behind a dead container and a
# muted account. Recovery then takes as many runs as the average is wide.
#
# The health probe already runs FIRST in every cycle. One request now decides
# whether five more are worth spending.


async def test_the_battery_skips_a_route_whose_health_probe_just_failed():
    calls = []

    def handler(req):
        calls.append(req)
        return httpx.Response(200, json={"choices": [{"message": {"content": "12"}}]})

    p = _proxy(handler)
    await probe_quality(p, p.store, [_route()], now=100.0, answered=set())
    assert calls == []


async def test_a_skipped_route_records_no_quality_row():
    """The old measurement must survive untouched: a stale real number beats a
    fresh fabricated zero."""
    p = _proxy(lambda req: httpx.Response(200, json={"choices": [{"message": {"content": "12"}}]}))
    p.store.upsert_routes([_route()], timestamp=10.0)   # metrics() only sees active routes
    p.store.record_probe("kilo/x:free", "quality", True, 0, 0, 200, 5, 5, 50.0)
    await probe_quality(p, p.store, [_route()], now=100.0, answered=set())
    rows = p.store._con.execute(
        "SELECT cases_passed, cases_total FROM probes WHERE kind='quality'").fetchall()
    assert rows == [(5, 5)]
    assert p.store.metrics()["kilo/x:free"].quality == 1.0


async def test_a_route_that_answered_is_still_measured():
    p = _proxy(lambda req: httpx.Response(200, json={"choices": [{"message": {"content": "12"}}]}))
    await probe_quality(p, p.store, [_route()], now=100.0, answered={"kilo/x:free"})
    assert p.store._con.execute(
        "SELECT COUNT(*) FROM probes WHERE kind='quality'").fetchone()[0] == 1


async def test_without_health_results_everything_is_measured_as_before():
    """The 450-odd tests that predate this call probe_quality with four args."""
    p = _proxy(lambda req: httpx.Response(200, json={"choices": [{"message": {"content": "12"}}]}))
    await probe_quality(p, p.store, [_route()], now=100.0)
    assert p.store._con.execute(
        "SELECT COUNT(*) FROM probes WHERE kind='quality'").fetchone()[0] == 1


async def test_probe_health_reports_which_routes_answered():
    ok = _route("vive:free")
    dead = _route("muerta:free")

    def handler(req):
        body = req.content.decode()
        if "muerta" in body or "muerta" in str(req.url):
            return httpx.Response(500)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    p = _proxy(handler)
    answered = await probe_health(p, p.store, [ok, dead], now=100.0)
    assert answered == {"kilo/vive:free"}


async def test_the_cycle_feeds_the_health_result_into_the_battery():
    """Wiring: a dead provider must cost one request per route, not six."""
    calls = []

    def handler(req):
        calls.append(str(req.url))
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json=CATALOGUE)
        return httpx.Response(500)

    store = _store()
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prov = [Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])]
    state = State(store=store, proxy=Proxy({"kilo": prov[0]}, store, http),
                  api_keys=set(), daily_paid_cap=200, providers=prov, http=http)
    await cycle(state, counter=0)          # counter 0 == a quality cycle
    completions = [u for u in calls if "chat/completions" in u]
    assert len(completions) == 1, completions


# --- The battery paces itself on evidence, not on an in-memory counter --------
#
# `scheduler` starts at `counter = 0` on every process start, and `cycle` runs the
# battery when `counter % QUALITY_EVERY_N_CYCLES == 0`. Since 0 % 5 == 0, EVERY
# restart ran a full battery, and the counter never survived to apply the pacing
# it was written for.
#
# Measured 2026-08-19 against the live deployment: the battery is meant to run once
# every ~25h. On 2026-08-18 it ran 28 times against deepseek and produced 824
# quality rows across the catalogue -- roughly 4,120 requests to free providers in
# one day. DeepSeek muted the account at 01:25 the following morning.
#
# The elapsed time since the last battery run is IN THE DATABASE, so the decision
# is taken from there. That survives restarts, crashes and redeploys alike, which
# an in-memory counter cannot.


async def test_the_battery_is_skipped_when_it_ran_recently(monkeypatch):
    calls = []

    def handler(req):
        calls.append(str(req.url))
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json=CATALOGUE)
        return httpx.Response(200, json={"choices": [{"message": {"content": "12"}}]})

    store = _store()
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prov = [Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])]
    state = State(store=store, proxy=Proxy({"kilo": prov[0]}, store, http),
                  api_keys=set(), daily_paid_cap=200, providers=prov, http=http)
    await cycle(state, counter=0)
    first = len([u for u in calls if "chat/completions" in u])
    # Let the health sweep through. This test is about the BATTERY's interval,
    # and the sweep now has a floor of its own (HEALTH_FLOOR_S) that would
    # short-circuit the whole cycle before the battery is even considered.
    store.mark_health_sweep(time.time() - HEALTH_FLOOR_S - 60)
    calls.clear()
    await cycle(state, counter=0)          # a restart: counter is 0 again
    second = len([u for u in calls if "chat/completions" in u])
    assert first > second, (first, second)
    assert second == 1, "only the health probe should have run"


async def test_the_battery_runs_again_once_the_interval_has_passed(monkeypatch):
    calls = []

    def handler(req):
        calls.append(str(req.url))
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json=CATALOGUE)
        return httpx.Response(200, json={"choices": [{"message": {"content": "12"}}]})

    store = _store()
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prov = [Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])]
    state = State(store=store, proxy=Proxy({"kilo": prov[0]}, store, http),
                  api_keys=set(), daily_paid_cap=200, providers=prov, http=http)
    await cycle(state, counter=0)
    # Age every battery row well past the interval.
    store._con.execute("UPDATE probes SET at = at - ? WHERE kind='quality'",
                       (QUALITY_INTERVAL_S + 60,))
    store._con.commit()
    # ... and let the health sweep through too, for the same reason as the test
    # above: the floor is a separate gate that runs before the battery's.
    store.mark_health_sweep(time.time() - HEALTH_FLOOR_S - 60)
    calls.clear()
    await cycle(state, counter=0)
    assert len([u for u in calls if "chat/completions" in u]) > 1


# --- The health sweep paces itself on evidence too ---------------------------
#
# The battery got this guard (above); the health sweep did not, and ran
# unconditionally on every process start. Measured 2026-08-19 on the live
# deployment: four deployments between 22:34 and 22:41 produced four full sweeps
# in twelve minutes, Kilo's free tier took 33 requests in eight and a half of
# them, and kilo/poolside/laguna-xs-2.1:free answered the fourth with a 429 and
# left routing. See probing.HEALTH_FLOOR_S.
#
# The interval cannot fix this: a burst of redeploys never reaches a sleep.


async def _state_over(handler):
    store = _store()
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prov = [Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])]
    return State(store=store, proxy=Proxy({"kilo": prov[0]}, store, http),
                 api_keys=set(), daily_paid_cap=200, providers=prov, http=http)


def _answering(calls):
    def handler(req):
        calls.append(str(req.url))
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json=CATALOGUE)
        return httpx.Response(200, json={"choices": [{"message": {"content": "12"}}]})
    return handler


async def test_a_restart_does_not_repeat_the_health_sweep():
    calls = []
    state = await _state_over(_answering(calls))
    await cycle(state, counter=0)
    calls.clear()
    await cycle(state, counter=0)          # a redeploy, minutes later
    assert [u for u in calls if "chat/completions" in u] == []


async def test_the_catalogue_is_still_synced_on_a_restart():
    """Skipping the sweep must not skip discovery: new routes only appear when the
    process restarts, and /models costs no chat quota."""
    calls = []
    state = await _state_over(_answering(calls))
    await cycle(state, counter=0)
    calls.clear()
    await cycle(state, counter=0)
    assert [u for u in calls if u.endswith("/models")] != []


async def test_the_health_sweep_runs_again_once_the_floor_has_passed():
    calls = []
    state = await _state_over(_answering(calls))
    await cycle(state, counter=0)
    state.store.mark_health_sweep(time.time() - HEALTH_FLOOR_S - 60)
    calls.clear()
    await cycle(state, counter=0)
    assert [u for u in calls if "chat/completions" in u] != []


async def test_an_on_demand_probe_does_not_suppress_the_sweep():
    """The proxy's own suspicion probes write `kind='health'` rows. If the guard
    read those, one of them would pass for a sweep of the whole catalogue."""
    calls = []
    state = await _state_over(_answering(calls))
    state.store.record_probe("kilo/x:free", "health", True, 10, 0, 200, 0, 0, time.time())
    await cycle(state, counter=0)
    assert [u for u in calls if "chat/completions" in u] != []


async def test_a_sweep_that_dies_mid_catalogue_still_counts_against_the_floor(monkeypatch):
    """The 2026-08-19 burst OPENED with a sweep the container was killed inside:
    the telemetry stops at grok-*, alphabetically short of kilo. If only a
    COMPLETED sweep armed the floor, that one would have spent 19 routes' worth of
    quota and left the next restart free to sweep the whole catalogue again --
    which is exactly what happened, three times over, before Kilo refused.

    What the floor bounds is REQUEST VOLUME, so what it must record is the
    attempt, not the success.
    """
    calls = []
    state = await _state_over(_answering(calls))

    async def killed(*args, **kwargs):
        raise RuntimeError("the container was killed mid-sweep")

    monkeypatch.setattr(probing, "probe_health", killed)
    with pytest.raises(RuntimeError):
        await cycle(state, counter=0)
    monkeypatch.undo()

    calls.clear()
    await cycle(state, counter=0)
    assert [u for u in calls if "chat/completions" in u] == []


_CAPS = {k: False for k in REQUIRED_CAPABILITIES}
_HEALTH = {"status": "ok", "contract": 1, "provider": "chatgpt",
           "auth": {"mode": "account", "plan": "go", "subscription_active": True},
           "capabilities": {**_CAPS, "chat": True, "streaming": True,
                            "vision": True, "images": True}}
_MODELS = {"data": [{"id": "gpt-5-6", "context_window": 52815,
                     "max_output_tokens": 8192,
                     "capabilities": {"tools": False, "vision": True,
                                      "images": False}}]}


def _chatgpt(**kw):
    return Provider("chatgpt", "free", "openai", "https://c.test/v1", "",
                    "/models", {}, [], reads_capabilities=True, **kw)


def _routed(health=None, models=None, health_status=200):
    def handler(req):
        if req.url.path.endswith("/health"):
            return httpx.Response(health_status, json=health or _HEALTH)
        return httpx.Response(200, json=models or _MODELS)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_sync_applies_the_contract_to_the_discovered_routes():
    store = _store()
    await sync_catalogue(_routed(), [_chatgpt()], store, now=100.0)
    caps = store.active_routes()[0].capabilities
    assert caps.vision is True
    assert caps.context == 52815          # not the 128000 anyone declared
    assert caps.images is False           # narrowed per model


async def test_sync_persists_the_contract_document():
    store = _store()
    await sync_catalogue(_routed(), [_chatgpt()], store, now=100.0)
    assert store.get_contract("chatgpt")["auth"]["plan"] == "go"


async def test_a_failing_health_keeps_the_previous_catalogue(caplog):
    store = _store()
    await sync_catalogue(_routed(), [_chatgpt()], store, now=100.0)
    before = store.active_routes()
    with caplog.at_level(logging.WARNING):
        await sync_catalogue(_routed(health_status=500), [_chatgpt()], store, now=200.0)
    assert store.active_routes() == before
    assert "chatgpt" in caplog.text


async def test_a_provider_that_does_not_read_capabilities_never_requests_health():
    seen = []

    def handler(req):
        seen.append(req.url.path)
        return httpx.Response(200, json=CATALOGUE)

    store = _store()
    prov = [Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])]
    await sync_catalogue(httpx.AsyncClient(transport=httpx.MockTransport(handler)),
                         prov, store, now=100.0)
    assert not any(p.endswith("/health") for p in seen)


async def test_a_health_without_the_contract_falls_back_to_the_yaml():
    store = _store()
    defaults = Capabilities(tools=False, vision=False, context=128000, max_output=8192)
    provider = Provider("chatgpt", "free", "openai", "https://c.test/v1", "",
                        "/models", {}, [], reads_capabilities=True,
                        default_capabilities=defaults)
    await sync_catalogue(_routed(health={"status": "ok"}), [provider], store, now=100.0)
    assert store.active_routes()[0].capabilities.context == 128000
