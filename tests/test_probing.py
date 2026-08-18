import json
import logging

import httpx

from llm_libre.storage import Storage
from llm_libre.api import State
from llm_libre.models import Capabilities, Route
from llm_libre.providers import Provider
from llm_libre.proxy import Proxy
from llm_libre.quality_suite import SHORT_TOKEN_BUDGET
from llm_libre.probing import (PING, cycle, probe_health, probe_quality,
                               sync_catalogue)

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
    store.upsert_routes([_route("previa:free")], timestamp=50.0)

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
    store.upsert_routes([_route("previa:free")], timestamp=50.0)
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=EMPTY_CATALOGUE)))
    prov = [Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])]
    await sync_catalogue(http, prov, store, now=100.0)
    assert [r.key for r in store.active_routes()] == ["kilo/previa:free"]


async def test_a_200_whose_models_are_all_filtered_out_does_not_erase_previous_routes():
    store = _store()
    store.upsert_routes([_route("previa:free")], timestamp=50.0)
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=CATALOGUE_ALL_FILTERED)))
    prov = [Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])]
    await sync_catalogue(http, prov, store, now=100.0)
    assert [r.key for r in store.active_routes()] == ["kilo/previa:free"]


async def test_un_proveedor_vacio_no_frena_la_actualizacion_del_que_si_respondio():
    store = _store()
    store.upsert_routes([_route("previa:free", provider="vacio")], timestamp=50.0)

    def handler(req):
        if "k.test" in str(req.url):
            return httpx.Response(200, json=CATALOGUE)
        return httpx.Response(200, json=EMPTY_CATALOGUE)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prov = [
        Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, []),
        Provider("vacio", "free", "openai", "https://vacio.test", "", "/models", {}, []),
    ]
    await sync_catalogue(http, prov, store, now=100.0)
    keys_ = {r.key for r in store.active_routes()}
    assert "kilo/x:free" in keys_          # the provider that did answer was updated
    assert "vacio/previa:free" in keys_    # the empty one kept what it had


async def test_un_proveedor_sano_desactiva_su_propia_ruta_vieja_aunque_otro_este_vacio():
    # Finding 1, case (c) -- the one that separates a real fix from a blunt one.
    # kilo had an old route that no longer appears in its new catalogue; vacio
    # answers 200 with no models. vacio being empty must not hold back the removal
    # of kilo/old:free: kilo answered correctly and its own disappearance is real.
    store = _store()
    store.upsert_routes([_route("old:free", provider="kilo")], timestamp=50.0)
    store.upsert_routes([_route("previa:free", provider="vacio")], timestamp=50.0)

    def handler(req):
        if "k.test" in str(req.url):
            return httpx.Response(200, json=CATALOGUE)   # ya no trae old:free, solo x:free
        return httpx.Response(200, json=EMPTY_CATALOGUE)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prov = [
        Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, []),
        Provider("vacio", "free", "openai", "https://vacio.test", "", "/models", {}, []),
    ]
    await sync_catalogue(http, prov, store, now=100.0)
    keys_ = {r.key for r in store.active_routes()}
    assert keys_ == {"kilo/x:free", "vacio/previa:free"}
    assert "kilo/old:free" not in keys_    # deactivated: kilo answered, and no longer carries it


async def test_a_healthy_provider_does_not_deactivate_another_providers_routes():
    # kilo's deactivation must be SCOPED to kilo: an old route belonging to
    # "otro" (much older than `now`) must not fall victim to kilo's UPDATE. "otro"
    # is STILL registered (with neither modelos_path nor modelos_fijos, so the
    # per-provider loop skips it without touching it) -- if it were not in `prov`,
    # the orphaned-provider sweep (Storage.deactivate_unregistered_providers, see
    # the separate block of tests below) would switch it off for a DIFFERENT
    # reason, and this test would stop isolating what it claims to test.
    store = _store()
    store.upsert_routes([_route("vieja:free", provider="otro")], timestamp=10.0)
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=CATALOGUE)))
    prov = [
        Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, []),
        Provider("otro", "free", "openai", "https://otro.test", "", "", {}, []),
    ]
    await sync_catalogue(http, prov, store, now=100.0)
    keys_ = {r.key for r in store.active_routes()}
    assert keys_ == {"kilo/x:free", "otro/vieja:free"}


CATALOGUE_COMPARTIDO = {"data": [
    {"id": "shared:free", "pricing": {"prompt": "0"}, "context_length": 1000,
     "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
     "supported_parameters": [], "top_provider": {"max_completion_tokens": 100}}]}


async def test_the_same_model_in_two_providers_only_the_one_that_lost_it_is_switched_off():
    # The reason for "routes, not models": the same modelo_id can exist in two
    # providers as two independent rows (different keys). If one stops offering it
    # and the other keeps it, they must be treated separately.
    store = _store()
    store.upsert_routes([_route("shared:free", provider="kilo")], timestamp=50.0)
    store.upsert_routes([_route("shared:free", provider="otro")], timestamp=50.0)

    def handler(req):
        if "k.test" in str(req.url):
            return httpx.Response(200, json=CATALOGUE)          # kilo ya no trae shared:free
        return httpx.Response(200, json=CATALOGUE_COMPARTIDO)   # otro lo sigue trayendo

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prov = [
        Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, []),
        Provider("otro", "free", "openai", "https://o.test", "", "/models", {}, []),
    ]
    await sync_catalogue(http, prov, store, now=100.0)
    keys_ = {r.key for r in store.active_routes()}
    assert "kilo/shared:free" not in keys_   # kilo lo perdio: se apaga
    assert "otro/shared:free" in keys_       # otro lo conserva: sigue activa
    assert "kilo/x:free" in keys_


async def test_a_200_with_a_non_json_body_is_a_failure_and_does_not_stop_the_others():
    # Finding 2: r.json() can raise JSONDecodeError (a ValueError subclass) on a
    # non-JSON body; that must neither escape sync_catalogue nor stop the
    # processing of the providers that come after.
    store = _store()

    def handler(req):
        if "roto.test" in str(req.url):
            return httpx.Response(200, text="esto no es json")
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
#     routes at `activa=1` forever -- see
#     Storage.deactivate_unregistered_providers. This is the "through
#     sync_catalogue" half of that coverage; the other half (the Storage method in
#     isolation) lives in test_storage.py. ---

async def test_sync_deactivates_the_routes_of_a_provider_removed_from_the_registry():
    store = _store()
    store.upsert_routes([_route("previa:free", provider="openrouter")], timestamp=50.0)
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=CATALOGUE)))
    # openrouter is NO LONGER in the list providers.load() returns -- this
    # simulates having removed it from providers.yaml.
    prov = [Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])]
    await sync_catalogue(http, prov, store, now=100.0)
    keys_ = {r.key for r in store.active_routes()}
    assert keys_ == {"kilo/x:free"}
    row = store._con.execute(
        "SELECT active FROM routes WHERE key = 'openrouter/previa:free'").fetchone()
    assert row == (0,)   # apagada, no borrada


async def test_sync_does_not_deactivate_routes_of_still_registered_providers():
    # Deliberately isolated from the NORMAL per-provider removal logic (already
    # covered by the tests above, "whoever answered updates its own"): minimax
    # uses modelos_fijos, so its route is re-declared IDENTICALLY on every pass
    # regardless of any HTTP mock -- if it survives, that is proof the
    # orphan sweep (`deactivate_unregistered_providers`) did not touch it, not
    # that it "happened to reappear in the catalogue".
    store = _store()
    store.upsert_routes([_route("previa:free", provider="openrouter")], timestamp=50.0)
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
    store.upsert_routes([_route("previa:free", provider="kilo")], timestamp=50.0)
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=CATALOGUE)))
    await sync_catalogue(http, [], store, now=100.0)
    assert {r.key for r in store.active_routes()} == {"kilo/previa:free"}


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
        if "raro.test" in url:
            return httpx.Response(200, json={"error": "unauthorized"})
        return httpx.Response(200, json=CATALOGUE)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prov = [
        Provider("nojson", "free", "openai", "https://nojson.test", "", "/models", {}, []),
        Provider("raro", "free", "openai", "https://raro.test", "", "/models", {}, []),
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
    assert row[1] == 5
    assert 1 <= row[0] < 5   # pasa aritmetica y formato, falla json y tools


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
    assert row[1] == 4          # 5 cases minus the tools one, which was skipped
    assert len(calls) == 4    # y no se le gasto cuota pidiendoselo


async def test_cycle_syncs_probes_health_and_probes_quality_on_cycle_zero():
    store = _store()
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=CATALOGUE)))
    prov = [Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])]
    proxy = Proxy({"kilo": prov[0]}, store, http)
    estado = State(store=store, proxy=proxy, api_keys=set(), daily_paid_cap=0,
                    providers=prov, http=http)
    await cycle(estado, counter=0)
    assert [r.key for r in store.active_routes()] == ["kilo/x:free"]
    tipos = {t for (t,) in store._con.execute("SELECT kind FROM probes").fetchall()}
    assert tipos == {"health", "quality"}


async def test_cycle_does_not_probe_quality_outside_the_interval():
    store = _store()
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=CATALOGUE)))
    prov = [Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])]
    proxy = Proxy({"kilo": prov[0]}, store, http)
    estado = State(store=store, proxy=proxy, api_keys=set(), daily_paid_cap=0,
                    providers=prov, http=http)
    await cycle(estado, counter=1)
    tipos = {t for (t,) in store._con.execute("SELECT kind FROM probes").fetchall()}
    assert tipos == {"health"}


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
    estado = State(store=store, proxy=proxy, api_keys=set(), daily_paid_cap=0,
                    providers=prov, http=http)
    await cycle(estado, counter=1)   # contador 1: salud si, calidad no
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
    store.upsert_routes([_route("previa:free")], timestamp=50.0)
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with caplog.at_level(logging.WARNING, logger="llm_libre.probing"):
        await sync_catalogue(http, _prov_kilo(), store, now=100.0)
    return caplog.text


async def test_a_network_error_is_logged_with_the_provider_and_the_reason(caplog):
    def handler(req):
        raise httpx.ConnectError("no hay ruta al host")

    texto = await _sincronizar_con(handler, caplog)
    assert "kilo" in texto and "no hay ruta al host" in texto


async def test_a_non_200_status_is_logged(caplog):
    texto = await _sincronizar_con(lambda req: httpx.Response(503), caplog)
    assert "kilo" in texto and "503" in texto


async def test_a_malformed_body_is_logged(caplog):
    texto = await _sincronizar_con(
        lambda req: httpx.Response(200, text="<html>mantenimiento</html>"), caplog)
    assert "kilo" in texto
    assert "could not interpret" in texto


async def test_an_empty_catalogue_is_logged(caplog):
    texto = await _sincronizar_con(
        lambda req: httpx.Response(200, json=EMPTY_CATALOGUE), caplog)
    assert "kilo" in texto
    assert "zero usable models" in texto


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

def _modelo_que_razona(tope_minimo=512, respuesta="pong"):
    """Un proveedor que se comporta como los modelos gratis de verdad: si no le
    alcanza el presupuesto, se lo gasta pensando y devuelve 200 con
    finish_reason 'length' y content null."""
    def handler(req):
        cuerpo = json.loads(req.content)
        if cuerpo.get("max_tokens", 0) < tope_minimo:
            return httpx.Response(200, json={"choices": [
                {"message": {"role": "assistant", "content": None},
                 "finish_reason": "length"}]})
        return httpx.Response(200, json={"choices": [
            {"message": {"role": "assistant", "content": respuesta},
             "finish_reason": "stop"}]})
    return handler


async def test_the_health_probe_does_not_kill_a_model_by_denying_it_room_to_think():
    p = _proxy(_modelo_que_razona())
    await probe_health(p, p.store, [_route()], now=100.0)
    row = p.store._con.execute(
        "SELECT ok FROM probes WHERE kind='health'").fetchone()
    assert row[0] == 1, ("el ping fabrico el fallo que dice medir: no le dio "
                          "presupuesto suficiente al modelo")


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
