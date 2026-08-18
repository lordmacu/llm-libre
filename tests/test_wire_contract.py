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
from llm_libre.api import Estado, crear_app, interpretar_pedido
from llm_libre.models import GATEWAY_EXTENSIONS, Capabilities, Route
from llm_libre.providers import Provider, load
from llm_libre.proxy import Proxy

YAML = "proveedores.yaml"


@pytest.fixture
def cliente():
    almacen = Storage(":memory:")
    almacen.create_schema()
    almacen.upsert_routes(
        [Route("kilo", "a:free", "gratis", Capabilities(True, False, 100000, 4096))], 1.0)
    prov = {"kilo": Provider("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, [])}
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "hola"}}]})))
    estado = Estado(almacen=almacen, proxy=Proxy(prov, almacen, http),
                    llaves={"buena"}, tope_pago_diario=200)
    return TestClient(crear_app(estado))


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
    assert interpretar_pedido({"model": "auto"}).profile == "balanceado"
    assert interpretar_pedido({"model": "auto:rapido"}).profile == "rapido"
    assert interpretar_pedido({"model": "auto:potente"}).profile == "potente"
    assert interpretar_pedido({"model": "auto:tools"}).needs_tools is True
    assert interpretar_pedido({"model": "auto:vision"}).needs_vision is True


def test_requiere_values_are_capability_names():
    p = interpretar_pedido({"model": "auto", "x_requiere": ["tools", "vision"]})
    assert p.needs_tools and p.needs_vision


# --- config file keys ---

def test_yaml_keys_are_still_understood():
    """proveedores.cargar reads these as string literals; the file must keep them."""
    provs = load(YAML, {})
    assert provs, "no providers loaded"
    assert {p.id for p in provs} >= {"chatgpt", "perplexity", "deepseek", "kilo", "minimax"}
    ds = next(p for p in provs if p.id == "deepseek")
    assert ds.tier == "gratis"
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
    assert {p.tier for p in provs} <= {"gratis", "pago"}


# --- persisted schema ---

def test_sqlite_table_names_are_stable():
    """The DB lives on disk in production; renaming a table needs a migration."""
    almacen = Storage(":memory:")
    almacen.create_schema()
    tablas = {row[0] for row in
              almacen._con.execute(
                  "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {"rutas", "sondas", "eventos", "uso_pago"} <= tablas
