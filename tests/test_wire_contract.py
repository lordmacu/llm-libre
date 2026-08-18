"""Frozen public contract: everything a client can observe.

This file exists to make the Spanish-to-English refactor safe. The internals are
being renamed module by module, but the names below are the PUBLIC surface --
endpoint paths, JSON keys, response headers, request extensions, YAML keys and
profile values -- and renaming any of them breaks every deployed client with no
compiler, type checker or existing test to catch it.

Two of these couplings are invisible at the call site and caused this file to be
written before a single rename:

- `api.py` serialises `pedido.__dict__` straight into the 400 and 503 error
  bodies, so the `Pedido` dataclass FIELD NAMES are wire format. Renaming a
  field silently changes the HTTP contract.
- `providers.load` reads YAML keys as string literals, so the `Provider`
  dataclass fields may be renamed freely but the keys in `proveedores.yaml` may
  not.

So: these assertions are deliberately literal. If one fails during a refactor,
the rename went too far -- fix the code, never this file. Changing anything here
is a versioned API change, not a refactor.
"""
import httpx
import pytest
from fastapi.testclient import TestClient

from llm_libre.storage import Storage
from llm_libre.api import State, create_app, parse_request
from llm_libre.models import GATEWAY_EXTENSIONS, Capabilities, Route
from llm_libre.providers import Provider, load
from llm_libre.proxy import Proxy

YAML = "proveedores.yaml"


@pytest.fixture
def cliente():
    almacen = Storage(":memory:")
    almacen.create_schema()
    almacen.upsert_routes(
        [Route("kilo", "a:free", "free", Capabilities(True, False, 100000, 4096))], 1.0)
    prov = {"kilo": Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])}
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "hola"}}]})))
    estado = State(store=almacen, proxy=Proxy(prov, almacen, http),
                    api_keys={"buena"}, daily_paid_cap=200)
    return TestClient(create_app(estado))


AUTH = {"Authorization": "Bearer buena"}


# --- endpoint paths ---

def test_endpoint_paths(cliente):
    for path in ("/health", "/v1/models", "/v1/ranking", "/v1/uso"):
        assert cliente.get(path, headers=AUTH).status_code == 200, path


# --- response bodies ---

def test_health_body_keys(cliente):
    body = cliente.get("/health").json()
    assert set(body) == {"estado", "rutas_activas", "rutas_libres", "gratis_libres"}
    assert body["estado"] in {"ok", "degradado", "caido"}


def test_ranking_row_keys(cliente):
    body = cliente.get("/v1/ranking", headers=AUTH).json()
    assert set(body) == {"rutas"}
    assert set(body["rutas"][0]) == {
        "clave", "tier", "prioridad", "puntaje", "calidad", "calidad_medida",
        "calidad_asumida", "ultima_sonda_calidad", "ultima_sonda", "confiabilidad",
        "ttft_p50_ms", "latencia_p50_ms", "en_cooldown_hasta", "tools", "vision",
        "contexto",
    }


def test_uso_body_keys(cliente):
    assert set(cliente.get("/v1/uso", headers=AUTH).json()) == {"dia", "pago_hoy", "tope"}


def test_models_body_keys(cliente):
    body = cliente.get("/v1/models", headers=AUTH).json()
    assert body["object"] == "list"
    assert {"id", "object", "owned_by"} <= set(body["data"][0])


# --- response headers ---

def test_response_headers(cliente):
    r = cliente.post("/v1/chat/completions", headers=AUTH,
                     json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    for header in ("X-Ruta-Usada", "X-Intentos", "X-Tier"):
        assert header in r.headers, header


# --- error bodies: `pedido` mirrors the Pedido dataclass field names ---

def test_error_body_exposes_pedido_fields(cliente):
    """400 and 503 bodies serialise pedido.__dict__ -- these keys are wire."""
    r = cliente.post("/v1/chat/completions", headers=AUTH,
                     json={"model": "auto", "x_min_contexto": 99_000_000,
                           "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code in (400, 503)
    detail = r.json()["detail"]
    assert "message" in detail
    assert set(detail["pedido"]) == {
        "modelo", "requiere_tools", "requiere_vision", "min_contexto", "perfil",
        "permitir_pago",
    }


def test_error_body_diagnostic_keys(cliente):
    r = cliente.post("/v1/chat/completions", headers=AUTH,
                     json={"model": "auto", "x_min_contexto": 99_000_000,
                           "messages": [{"role": "user", "content": "hi"}]})
    detail = r.json()["detail"]
    assert "rutas_activas" in detail or "rutas_compatibles" in detail


# --- request-side vocabulary ---

def test_gateway_extension_names():
    assert GATEWAY_EXTENSIONS == frozenset(
        {"x_requiere", "x_min_contexto", "x_permitir_pago", "x_crudo"})


def test_model_aliases_and_profiles():
    assert parse_request({"model": "auto"}).profile == "balanceado"
    assert parse_request({"model": "auto:rapido"}).profile == "rapido"
    assert parse_request({"model": "auto:potente"}).profile == "potente"
    assert parse_request({"model": "auto:tools"}).needs_tools is True
    assert parse_request({"model": "auto:vision"}).needs_vision is True


def test_requiere_values_are_capability_names():
    p = parse_request({"model": "auto", "x_requiere": ["tools", "vision"]})
    assert p.needs_tools and p.needs_vision


# --- config file keys ---

def test_yaml_keys_are_still_understood():
    """proveedores.cargar reads these as string literals; the file must keep them."""
    provs = load(YAML, {})
    assert provs, "no providers loaded"
    assert {p.id for p in provs} >= {"chatgpt", "perplexity", "deepseek", "kilo", "minimax"}
    ds = next(p for p in provs if p.id == "deepseek")
    assert ds.tier == "free"
    assert ds.dialect == "openai"
    assert ds.priority == 1
    assert ds.emulates_tools is True
    assert ds.timeout_s == 60.0
    cg = next(p for p in provs if p.id == "chatgpt")
    assert cg.default_capabilities is not None
    assert cg.unwraps_canvas is True
    assert cg.models_path == "/models"
    grok = next(p for p in provs if p.id == "grok")
    assert "imagine-agent-mode" in grok.exceptions


def test_tier_values_are_wire():
    provs = load(YAML, {})
    assert {p.tier for p in provs} <= {"free", "paid"}


# --- persisted schema ---

def test_sqlite_table_names_are_stable():
    """The DB lives on disk in production; renaming a table needs a migration."""
    almacen = Storage(":memory:")
    almacen.create_schema()
    tablas = {row[0] for row in
              almacen._con.execute(
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
    almacen = Storage(":memory:")
    almacen._con.executescript("""
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
        INSERT INTO sondas VALUES ('kilo/a:free','calidad',12.0,1,0,0,200,4,5);
        INSERT INTO eventos VALUES ('kilo/a:free',13.0,1,250,200,900,0);
        INSERT INTO uso_pago VALUES ('secreta','2026-08-18',7);
    """)
    almacen.create_schema()
    almacen.create_schema()   # a restart runs it again: it must be a no-op

    con = almacen._con
    assert {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")} \
        == {"routes", "probes", "events", "paid_usage"}
    # The old indexes are dropped rather than carried over: renaming a table keeps
    # its index NAMES, so leaving them would mean two identical indexes per table.
    assert {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'ix_%'")} \
        == {"ix_probes", "ix_events"}
    assert [f[1] for f in con.execute("PRAGMA table_info(routes)")] == [
        "key", "provider", "model_id", "tier", "tools", "vision", "context",
        "max_output", "last_seen", "active", "priority"]
    assert [f[1] for f in con.execute("PRAGMA table_info(probes)")] == [
        "key", "kind", "at", "ok", "latency_ms", "ttft_ms", "http_code",
        "cases_passed", "cases_total"]
    assert [f[1] for f in con.execute("PRAGMA table_info(events)")] == [
        "key", "at", "ok", "ttft_ms", "http_code", "latency_ms", "is_client_error"]
    assert [f[1] for f in con.execute("PRAGMA table_info(paid_usage)")] == [
        "api_key", "day", "requests"]

    # Not one row lost, and the Spanish VOCABULARY rewritten too: leaving 'gratis'
    # behind while the router asks for 'free' matches zero routes and answers 503
    # to everything, with no error anywhere to say why.
    assert {r[0]: r[1] for r in con.execute("SELECT key, tier FROM routes")} == {
        "kilo/a:free": "free", "minimax/M3": "paid"}
    assert {r[0] for r in con.execute("SELECT kind FROM probes")} == {"health", "quality"}
    assert con.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
    assert almacen.paid_usage("secreta", "2026-08-18") == 7
    assert [r.key for r in almacen.active_routes()] == ["kilo/a:free", "minimax/M3"]
    assert almacen.metrics()["kilo/a:free"].quality == 4 / 5
