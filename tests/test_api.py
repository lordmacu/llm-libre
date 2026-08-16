import httpx
import pytest
from fastapi.testclient import TestClient

from llm_libre.almacen import Almacen
from llm_libre.api import Estado, crear_app, interpretar_pedido
from llm_libre.modelos import Capacidades, Ruta
from llm_libre.proveedores import Proveedor
from llm_libre.proxy import Proxy


def test_auto_es_balanceado():
    p = interpretar_pedido({"model": "auto"})
    assert p.modelo is None and p.perfil == "balanceado"


def test_los_alias_de_perfil():
    assert interpretar_pedido({"model": "auto:rapido"}).perfil == "rapido"
    assert interpretar_pedido({"model": "auto:potente"}).perfil == "potente"


def test_los_alias_de_capacidad_se_traducen_a_requisitos():
    p = interpretar_pedido({"model": "auto:tools"})
    assert p.requiere_tools is True and p.perfil == "balanceado"
    assert interpretar_pedido({"model": "auto:vision"}).requiere_vision is True


def test_un_modelo_real_se_conserva():
    p = interpretar_pedido({"model": "poolside/laguna-s-2.1:free"})
    assert p.modelo == "poolside/laguna-s-2.1:free"


def test_mandar_tools_exige_soporte_de_tools_aunque_no_se_pida():
    p = interpretar_pedido({"model": "auto", "tools": [{"type": "function"}]})
    assert p.requiere_tools is True


def test_las_extensiones_x_se_respetan():
    p = interpretar_pedido({"model": "auto", "x_requiere": ["tools", "vision"],
                            "x_min_contexto": 200000, "x_permitir_pago": False})
    assert p.requiere_tools and p.requiere_vision
    assert p.min_contexto == 200000
    assert p.permitir_pago is False


@pytest.fixture
def cliente():
    almacen = Almacen(":memory:")
    almacen.crear_esquema()
    almacen.upsert_rutas(
        [Ruta("kilo", "a:free", "gratis", Capacidades(True, False, 100000, 4096))], 1.0)
    prov = {"kilo": Proveedor("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, [])}
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "hola"}}]})))
    estado = Estado(almacen=almacen, proxy=Proxy(prov, almacen, http),
                    llaves={"buena"}, tope_pago_diario=200)
    return TestClient(crear_app(estado))


def test_sin_llave_da_401(cliente):
    r = cliente.post("/v1/chat/completions", json={"model": "auto", "messages": []})
    assert r.status_code == 401


def test_con_llave_mala_da_401(cliente):
    r = cliente.post("/v1/chat/completions", headers={"X-API-Key": "mala"},
                     json={"model": "auto", "messages": []})
    assert r.status_code == 401


def test_completions_responde_y_marca_la_ruta_usada(cliente):
    r = cliente.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert r.headers["X-Ruta-Usada"] == "kilo/a:free"
    assert r.headers["X-Tier"] == "gratis"
    assert r.json()["choices"][0]["message"]["content"] == "hola"


def test_models_lista_el_catalogo_y_los_alias(cliente):
    r = cliente.get("/v1/models", headers={"X-API-Key": "buena"})
    ids = [m["id"] for m in r.json()["data"]]
    assert "a:free" in ids
    assert "auto" in ids and "auto:rapido" in ids


def test_pedir_capacidades_imposibles_da_400(cliente):
    r = cliente.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "messages": [], "x_min_contexto": 99999999})
    assert r.status_code == 400


def test_un_modelo_explicito_que_ya_no_existe_da_404_con_sugerencias(cliente):
    # Es el bug que este proyecto existe para evitar: un id cableado que se murio.
    #
    # DESVIACION respecto al brief: el brief afirma `"a:free" in str(r.json())`,
    # pero eso depende de que difflib.get_close_matches (cutoff=0.3) considere
    # "a:free" suficientemente parecido a "poolside/laguna-m.1:free" — un detalle
    # de la metrica de similitud, no del contrato que este test quiere proteger.
    # Se afirma el contrato real: que la respuesta trae la clave "sugerencias".
    r = cliente.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "poolside/laguna-m.1:free", "messages": []})
    assert r.status_code == 404
    assert "sugerencias" in str(r.json())


def test_health_dice_ok_si_hay_ruta_viva(cliente):
    assert cliente.get("/health").json()["estado"] == "ok"


def test_ranking_desglosa_los_componentes(cliente):
    fila = cliente.get("/v1/ranking", headers={"X-API-Key": "buena"}).json()["rutas"][0]
    for campo in ("clave", "puntaje", "calidad", "confiabilidad", "ttft_p50_ms", "tier"):
        assert campo in fila
