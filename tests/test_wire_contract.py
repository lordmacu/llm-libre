"""Frozen public contract: everything a client can observe.

Endpoint paths, JSON keys, response headers, request extensions, YAML keys,
profile and tier values, and the persisted schema. Nothing here can change
without breaking a deployed client, and none of it has a compiler or a type
checker watching it -- which is why the assertions below are deliberately
literal, spelling out every string rather than deriving it from the code they
are meant to constrain.

This file was written BEFORE the Spanish-to-English refactor began, to freeze
what the refactor must not touch. Two of the couplings it pins down are
invisible at the call site:

- `api.py` used to serialise `request.__dict__` straight into the 400 and 503
  error bodies, so the dataclass FIELD NAMES were wire format and renaming a
  field silently changed the HTTP contract. `RouteRequest.as_wire()` now makes
  that mapping explicit; this file asserts what it produces.
- `providers.load` reads YAML keys as string literals, so the `Provider`
  dataclass fields may be renamed freely but the keys in `providers.yaml` may
  not.

The surface has since been translated ON PURPOSE -- a deliberate, breaking
change made while no client had integrated yet, not a refactor that leaked. What
the file is for has not changed: if an assertion here fails during a rename, the
rename went too far. Fix the code, never this file. Changing anything here is a
versioned API change.
"""
import httpx
import pytest
from fastapi.testclient import TestClient

from llm_libre.storage import Storage
from llm_libre.api import State, create_app, parse_request
from llm_libre.models import GATEWAY_EXTENSIONS, Capabilities, Route
from llm_libre.providers import Provider, load
from llm_libre.proxy import Proxy

YAML = "providers.yaml"


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


AUTH = {"Authorization": "Bearer buena"}


# --- endpoint paths ---

def test_endpoint_paths(client):
    for path in ("/health", "/v1/models", "/v1/ranking", "/v1/usage"):
        assert client.get(path, headers=AUTH).status_code == 200, path


def test_the_image_generation_endpoint_exists_at_the_openai_path(client):
    """`POST /v1/images/generations`, so a client changes only base_url.

    Asserted through the app's route table rather than by calling it: the
    fixture has no image-capable route, so a call would 400 -- which proves the
    filter works, not that the PATH is the one clients expect."""
    assert "/v1/images/generations" in {r.path for r in client.app.routes}


# --- response bodies ---

def test_health_body_keys(client):
    body = client.get("/health").json()
    assert set(body) == {"status", "active_routes", "available_routes", "free_available"}
    assert body["status"] in {"ok", "degraded", "down"}


def test_ranking_row_keys(client):
    body = client.get("/v1/ranking", headers=AUTH).json()
    assert set(body) == {"routes"}
    assert set(body["routes"][0]) == {
        "key", "tier", "priority", "score", "quality", "quality_measured",
        "quality_assumed", "last_quality_probe", "last_probe", "reliability",
        "ttft_p50_ms", "latency_p50_ms", "cooldown_until",
        # Every live punishment by the capability it is evidence ABOUT. Added
        # 2026-08-19, when punishments became scoped: `cooldown_until` alone
        # cannot distinguish "out of image quota" from "route is down".
        "cooldowns",
        # The inferred CHAT quota, added 2026-08-19 for the providers that publish
        # no rate limit of their own. Same measured/assumed split as `quality`:
        # `rate_allowance` is null until an actual refusal has been seen, and
        # `rate_floor` carries the lower bound separately so the two can never be
        # confused.
        #
        # `rate_window_s` joined them the same day, when measuring the live data
        # showed most refusals are against DAILY quotas: an allowance is
        # meaningless without the window it belongs to, and reading a daily quota
        # through an hourly lens produced allowances below what the same routes
        # had already been observed sustaining. `rate_per_hour` stays as the
        # normalised form, for comparison against providers that advertise one.
        "rate_allowance", "rate_window_s", "rate_per_hour", "rate_measured",
        "rate_floor", "rate_used", "rate_remaining", "rate_exhausts_in_s",
        "rate_recovery_s", "rate_episodes",
        "tools", "vision",
        # image OUTPUT -- a separate axis from `vision`, which is image input
        "images",
        "context",
    }


def test_usage_body_keys(client):
    assert set(client.get("/v1/usage", headers=AUTH).json()) == {"day", "paid_today", "cap"}


def test_models_body_keys(client):
    body = client.get("/v1/models", headers=AUTH).json()
    assert body["object"] == "list"
    assert {"id", "object", "owned_by"} <= set(body["data"][0])


# --- response headers ---

def test_response_headers(client):
    r = client.post("/v1/chat/completions", headers=AUTH,
                     json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    for header in ("X-Route-Used", "X-Attempts", "X-Tier"):
        assert header in r.headers, header


# --- error bodies: `request` mirrors what RouteRequest.as_wire() produces ---

def test_error_body_exposes_the_request_fields(client):
    """400 and 503 bodies echo the normalised request -- these keys are wire."""
    r = client.post("/v1/chat/completions", headers=AUTH,
                     json={"model": "auto", "x_min_context": 99_000_000,
                           "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code in (400, 503)
    detail = r.json()["detail"]
    assert "message" in detail
    assert set(detail["request"]) == {
        "model", "needs_tools", "needs_vision", "needs_images", "min_context",
        "profile", "allow_paid",
    }


def test_error_body_diagnostic_keys(client):
    r = client.post("/v1/chat/completions", headers=AUTH,
                     json={"model": "auto", "x_min_context": 99_000_000,
                           "messages": [{"role": "user", "content": "hi"}]})
    detail = r.json()["detail"]
    assert "active_routes" in detail or "compatible_routes" in detail


# --- request-side vocabulary ---

def test_gateway_extension_names():
    assert GATEWAY_EXTENSIONS == frozenset(
        {"x_requires", "x_min_context", "x_allow_paid", "x_raw"})


def test_model_aliases_and_profiles():
    assert parse_request({"model": "auto"}).profile == "balanced"
    assert parse_request({"model": "auto:fast"}).profile == "fast"
    assert parse_request({"model": "auto:strong"}).profile == "strong"
    assert parse_request({"model": "auto:tools"}).needs_tools is True
    assert parse_request({"model": "auto:vision"}).needs_vision is True


def test_x_requires_values_are_capability_names():
    p = parse_request({"model": "auto", "x_requires": ["tools", "vision"]})
    assert p.needs_tools and p.needs_vision


# --- config file keys ---

def test_yaml_keys_are_still_understood():
    """proveedores.cargar reads these as string literals; the file must keep them."""
    provs = load(YAML, {})
    assert provs, "no providers loaded"
    assert {p.id for p in provs} >= {"chatgpt", "perplexity", "deepseek", "grok",
                                     "mistral", "kilo", "minimax"}
    ds = next(p for p in provs if p.id == "deepseek")
    assert ds.tier == "free"
    assert ds.dialect == "openai"
    assert ds.priority == 0   # branded proxies lead since the 2026-08-19 experiment
    assert ds.emulates_tools is True
    assert ds.timeout_s == 150.0
    cg = next(p for p in provs if p.id == "chatgpt")
    assert cg.default_capabilities is not None
    assert cg.unwraps_canvas is True
    assert cg.models_path == "/models"
    grok = next(p for p in provs if p.id == "grok")
    assert "imagine-agent-mode" in grok.exceptions


def test_tier_values_are_wire():
    provs = load(YAML, {})
    assert {p.tier for p in provs} <= {"free", "paid"}


def test_the_yaml_key_names_are_literal(tmp_path):
    """The complement to the test above, which only proves the CURRENT file
    parses. `providers.load` reads these keys as string literals, so a rename on
    either side alone -- the code or the file -- fails silently: `.get()` returns
    the default and the provider quietly loses its priority, its timeout or its
    tool emulation. Spelled out here so both sides move together or not at all."""
    yaml_file = tmp_path / "providers.yaml"
    yaml_file.write_text(
        "providers:\n"
        "  - id: p\n"
        "    tier: free\n"
        "    dialect: openai\n"
        "    base_url: https://p.test/v1\n"
        "    api_key_env: P_KEY\n"
        "    models_path: /models\n"
        "    priority: 7\n"
        "    timeout_s: 42\n"
        "    unwraps_canvas: true\n"
        "    emulates_tools: true\n"
        "    extra_headers:\n"
        "      X-Thing: yes\n"
        "    default_capabilities:\n"
        "      tools: false\n"
        "      vision: true\n"
        "      context: 128000\n"
        "      max_output: 4096\n"
        "    exceptions:\n"
        "      weird-model:\n"
        "        tools: false\n"
        "    fixed_models:\n"
        "      - id: m\n"
        "        tools: true\n"
        "        vision: false\n"
        "        context: 1000\n"
        "        max_output: 100\n", encoding="utf-8")
    p = load(str(yaml_file), {"P_KEY": "secret"})[0]
    assert (p.id, p.tier, p.dialect) == ("p", "free", "openai")
    assert p.api_key == "secret"
    assert p.models_path == "/models"
    assert p.priority == 7
    assert p.timeout_s == 42.0
    assert p.unwraps_canvas is True
    assert p.emulates_tools is True
    assert p.extra_headers == {"X-Thing": True}
    assert p.default_capabilities.context == 128000
    assert p.default_capabilities.max_output == 4096
    assert p.exceptions == {"weird-model": {"tools": False}}
    assert p.fixed_models[0]["id"] == "m"


# --- persisted schema ---

def test_sqlite_table_names_are_stable():
    """The DB lives on disk in production; renaming a table needs a migration."""
    store = Storage(":memory:")
    store.create_schema()
    tablas = {row[0] for row in
              store._con.execute(
                  "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {"routes", "probes", "events", "paid_usage"} <= tablas


def test_a_spanish_database_migrates_without_losing_a_single_row():
    """The counterpart to the test above: the names moved, so what proves the move
    was safe is that a database written under the OLD ones still arrives intact.

    Production's database sits on the /datos volume, which survives redeploys --
    it is the only copy of the telemetry the router scores routes with. A rename
    that silently dropped it would leave a gateway that looks healthy while
    routing blind. Every table, every column and both stored vocabularies (`tier`
    and the probe `kind`) are checked, plus idempotency: the same migration runs
    twice, because a container restart runs `create_schema` again."""
    store = Storage(":memory:")
    store._con.executescript("""
        CREATE TABLE rutas (clave TEXT PRIMARY KEY, proveedor TEXT, modelo_id TEXT,
            tier TEXT, tools INTEGER, vision INTEGER, contexto INTEGER,
            max_salida INTEGER, visto_por_ultima_vez REAL, activa INTEGER,
            prioridad INTEGER);
        CREATE TABLE sondas (clave TEXT, tipo TEXT, momento REAL, ok INTEGER,
            latencia_ms INTEGER, ttft_ms INTEGER, codigo_http INTEGER,
            casos_pasados INTEGER, casos_totales INTEGER);
        CREATE INDEX ix_sondas ON sondas(clave, tipo, momento DESC);
        CREATE TABLE eventos (clave TEXT, momento REAL, ok INTEGER, ttft_ms INTEGER,
            codigo_http INTEGER, latencia_ms INTEGER, es_error_cliente INTEGER);
        CREATE INDEX ix_eventos ON eventos(clave, momento DESC);
        CREATE TABLE uso_pago (llave TEXT, dia TEXT, peticiones INTEGER,
            PRIMARY KEY (llave, dia));
        INSERT INTO rutas VALUES ('kilo/a:free','kilo','a:free','gratis',1,0,128000,4096,10.0,1,1);
        INSERT INTO rutas VALUES ('minimax/M3','minimax','M3','pago',1,1,200000,8192,10.0,1,2);
        INSERT INTO sondas VALUES ('kilo/a:free','salud',11.0,1,300,0,200,NULL,NULL);
        INSERT INTO sondas VALUES ('kilo/a:free','quality',12.0,1,0,0,200,4,5);
        INSERT INTO eventos VALUES ('kilo/a:free',13.0,1,250,200,900,0);
        INSERT INTO uso_pago VALUES ('secreta','2026-08-18',7);
    """)
    store.create_schema()
    store.create_schema()   # a restart runs it again: it must be a no-op

    con = store._con
    assert {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}\
        == {"routes", "probes", "events", "paid_usage",
            # Added later; a legacy database gains them from CREATE TABLE IF NOT
            # EXISTS like any other, since they hold no pre-existing data.
            "assets", "sweeps"}
    # The old indexes are dropped rather than carried over: renaming a table keeps
    # its index NAMES, so leaving them would mean two identical indexes per table.
    assert {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'ix_%'")}\
        == {"ix_probes", "ix_events", "ix_assets_age"}
    assert [f[1] for f in con.execute("PRAGMA table_info(routes)")] == [
        "key", "provider", "model_id", "tier", "tools", "vision", "context",
        "max_output", "last_seen", "active", "priority",
        # Added after the rename: an old database gains it through the
        # add-column migration, defaulting to 0 -- "cannot generate" is the safe
        # direction for a row written before anyone measured it.
        "images"]
    assert [f[1] for f in con.execute("PRAGMA table_info(probes)")] == [
        "key", "kind", "at", "ok", "latency_ms", "ttft_ms", "http_code",
        "cases_passed", "cases_total"]
    assert [f[1] for f in con.execute("PRAGMA table_info(events)")] == [
        "key", "at", "ok", "ttft_ms", "http_code", "latency_ms", "is_client_error",
        # Also added after the rename, for Storage.rate_budgets: an allowance
        # belongs to the resource it was spent on, so an image refusal must not be
        # counted against chat traffic. Old rows default to 'chat', which is what
        # they were -- image generation arrived later.
        "capability",
        # For Storage.traffic: these describe the CLIENT REQUEST an attempt was
        # serving, which route-keyed rows alone cannot express -- five rows against
        # one route are five requests or one request failing over five times, and
        # nothing in the other columns tells those apart. They migrate to NULL, not
        # to a default: a row written before they existed carries no such
        # information, and inventing one would quietly mix fabricated rows into
        # every ratio computed off this table.
        "requested", "request_id", "attempt"]
    assert [f[1] for f in con.execute("PRAGMA table_info(paid_usage)")] == [
        "api_key", "day", "requests"]

    # Not one row lost, and the Spanish VOCABULARY rewritten too: leaving 'gratis'
    # behind while the router asks for 'free' matches zero routes and answers 503
    # to everything, with no error anywhere to say why.
    assert {r[0]: r[1] for r in con.execute("SELECT key, tier FROM routes")} == {
        "kilo/a:free": "free", "minimax/M3": "paid"}
    assert {r[0] for r in con.execute("SELECT kind FROM probes")} == {"health", "quality"}
    assert con.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
    assert store.paid_usage("secreta", "2026-08-18") == 7
    assert [r.key for r in store.active_routes()] == ["kilo/a:free", "minimax/M3"]
    assert store.metrics()["kilo/a:free"].quality == 4 / 5


# --- /v1/ranking exposes cooldowns per capability, 2026-08-19 ----------------
#
# `cooldown_until` keeps meaning what it always did -- when this route is free for
# CHAT -- because that is what `auto` uses and what every reader of this table has
# assumed. But punishments are now scoped (see proxy.ALL_CAPABILITIES), so that
# single number can no longer tell the whole story: a grok agent whose image
# bucket is empty is unavailable for images and perfectly healthy for chat, and an
# operator asking "why did my image request 503 while chat works" has to be able
# to see it.


def _state_and_client():
    """Like the `client` fixture, but hands back the State too so a test can
    punish a route directly."""
    store = Storage(":memory:")
    store.create_schema()
    store.upsert_routes([
        Route("kilo", "a:free", "free", Capabilities(True, False, 100000, 4096)),
        Route("grok", "imagine", "free",
              Capabilities(False, False, 100000, 4096, images=True)),
    ], 1.0)
    prov = {"kilo": Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, []),
            "grok": Provider("grok", "free", "openai", "https://g.test", "", "/models", {}, [])}
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "hi"}}]})))
    state = State(store=store, proxy=Proxy(prov, store, http),
                  api_keys={"buena"}, daily_paid_cap=200)
    return state, TestClient(create_app(state))


def _row(client, key):
    return next(x for x in client.get("/v1/ranking", headers=AUTH).json()["routes"]
                if x["key"] == key)


def test_the_ranking_reports_a_cooldown_that_only_affects_images():
    state, client = _state_and_client()
    state.proxy.cooldowns.punish("grok/imagine", 9e9, scope="images")
    row = _row(client, "grok/imagine")
    # The headline number is the CHAT one, and chat is unaffected.
    assert row["cooldown_until"] == 0.0
    assert row["cooldowns"]["images"] == 9e9


def test_a_route_wide_cooldown_shows_up_in_the_headline_number():
    state, client = _state_and_client()
    state.proxy.cooldowns["kilo/a:free"] = 9e9        # no scope == the whole route
    row = _row(client, "kilo/a:free")
    assert row["cooldown_until"] == 9e9
    assert row["cooldowns"]["*"] == 9e9


def test_a_route_with_no_punishment_reports_no_scopes():
    _, client = _state_and_client()
    row = _row(client, "kilo/a:free")
    assert row["cooldown_until"] == 0.0
    assert row["cooldowns"] == {}
