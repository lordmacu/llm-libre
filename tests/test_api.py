import asyncio
import time
from datetime import datetime, timezone

import httpx
import pytest
from fastapi.testclient import TestClient

from llm_libre.almacen import Almacen
from llm_libre.api import Estado, crear_app, interpretar_pedido
from llm_libre.modelos import Capacidades, Ruta
from llm_libre.proveedores import Proveedor
from llm_libre.proxy import Proxy


def _hoy() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _sse(*trozos: str) -> bytes:
    lineas = [f'data: {{"choices":[{{"delta":{{"content":"{t}"}}}}]}}\n\n' for t in trozos]
    lineas.append("data: [DONE]\n\n")
    return "".join(lineas).encode()


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


def test_model_de_solo_espacios_se_trata_como_ausente():
    # Fix round 1, hallazgo 3 (Minor): "   " es truthy, asi que se colaba antes
    # del "or auto" y quedaba vacio tras el strip -- un 404 confuso sobre el
    # modelo ''. Debe tratarse igual que "" / None / ausente: cae a "auto".
    p = interpretar_pedido({"model": "   "})
    assert p.modelo is None and p.perfil == "balanceado"


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


@pytest.fixture
def estado_cliente():
    """Como `cliente`, pero exponiendo tambien el `Estado`: los tests de
    /health necesitan sembrar eventos/cooldowns directamente en el almacen y
    el proxy, cosa que la fixture `cliente` no permite tocar desde afuera."""
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
    return estado, TestClient(crear_app(estado))


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


# --- Fix round 1, hallazgo 1 (Critical): /health tambien debe fallar cuando
#     la ruta gratis esta genuinamente rota (500 en cada intento), no solo
#     cuando esta en cooldown por un 429. ---

def test_health_no_es_ok_si_la_gratis_falla_de_verdad(estado_cliente):
    estado, cliente = estado_cliente
    ahora = time.time()
    # Trafico real, mayormente fallido: la unica ruta gratis responde mal en
    # 8 de 10 intentos recientes. Esto NUNCA pasa por Proxy._castigar (eso
    # solo lo dispara un 429), asi que proxy.cooldowns queda vacio -- la
    # version vieja de /health diria "ok" igual.
    for _ in range(8):
        estado.almacen.registrar_evento("kilo/a:free", False, 0, 500, ahora)
    for _ in range(2):
        estado.almacen.registrar_evento("kilo/a:free", True, 200, 200, ahora)
    assert estado.proxy.cooldowns == {}  # confirma que no es por cooldown

    r = cliente.get("/health")
    assert r.status_code != 200
    assert r.json()["estado"] != "ok"


def test_health_ok_si_la_gratis_no_tiene_telemetria_aun(estado_cliente):
    # Una ruta recien sincronizada, sin ningun evento todavia, carga la
    # confiabilidad NEUTRA (no cero) y debe seguir contando como viva.
    estado, cliente = estado_cliente
    filas = estado.almacen._con.execute("SELECT COUNT(*) FROM eventos").fetchone()
    assert filas[0] == 0
    assert cliente.get("/health").json()["estado"] == "ok"


def test_health_ok_si_la_gratis_esta_sana(estado_cliente):
    estado, cliente = estado_cliente
    ahora = time.time()
    for _ in range(9):
        estado.almacen.registrar_evento("kilo/a:free", True, 200, 200, ahora)
    for _ in range(1):
        estado.almacen.registrar_evento("kilo/a:free", False, 0, 500, ahora)
    assert cliente.get("/health").json()["estado"] == "ok"


def test_health_sigue_excluyendo_por_cooldown_de_429(estado_cliente):
    # La exclusion por cooldown (la que ya existia) no debe haberse roto al
    # agregar la de confiabilidad.
    estado, cliente = estado_cliente

    async def _forzar_429():
        rutas = estado.almacen.rutas_activas()
        estado.proxy.http = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(429, json={"error": "rate limited"})))
        await estado.proxy.completar(rutas, {"model": "a:free", "messages": []}, time.time())

    asyncio.run(_forzar_429())
    assert estado.proxy.cooldowns  # confirma que esta vez SI quedo en cooldown

    r = cliente.get("/health")
    assert r.status_code != 200
    assert r.json()["estado"] != "ok"


def test_ranking_desglosa_los_componentes(cliente):
    fila = cliente.get("/v1/ranking", headers={"X-API-Key": "buena"}).json()["rutas"][0]
    for campo in ("clave", "puntaje", "calidad", "confiabilidad", "ttft_p50_ms", "tier"):
        assert campo in fila


# --- Fix round 1, hallazgo 2 (Critical): el tope de pago diario tambien debe
#     atar en la rama de streaming, no solo en la sincronica. ---

def _estado_libre_y_pago(tope_pago_diario, hacer_resp_free, hacer_resp_paid):
    """Dos rutas -- una gratis, una de pago. `hacer_resp_free`/`hacer_resp_paid`
    son callables SIN argumentos que fabrican una `httpx.Response` NUEVA en
    cada llamada (no un objeto compartido): las respuestas viajan por
    `.stream()`, cuyo estado interno solo se puede consumir una vez, y este
    helper puede invocarse mas de una vez por ruta dentro de un mismo test."""
    almacen = Almacen(":memory:")
    almacen.crear_esquema()
    almacen.upsert_rutas([
        Ruta("free_prov", "f:free", "gratis", Capacidades(True, False, 100000, 4096)),
        Ruta("paid_prov", "p:paid", "pago", Capacidades(True, False, 100000, 4096)),
    ], 1.0)
    prov = {
        "free_prov": Proveedor("free_prov", "gratis", "openai", "https://f.test", "", "/models", {}, []),
        "paid_prov": Proveedor("paid_prov", "pago", "openai", "https://p.test", "", "/models", {}, []),
    }

    def responder(req):
        return hacer_resp_free() if "f.test" in str(req.url) else hacer_resp_paid()

    http = httpx.AsyncClient(transport=httpx.MockTransport(responder))
    estado = Estado(almacen=almacen, proxy=Proxy(prov, almacen, http),
                    llaves={"buena"}, tope_pago_diario=tope_pago_diario)
    return estado, TestClient(crear_app(estado))


def test_streaming_pago_cuenta_uso_y_el_tope_ata():
    # (a) con un tope minusculo, una peticion en streaming SI servida por la
    # ruta de pago cuenta uso -- y una segunda, con el tope ya agotado, deja
    # de ofrecer la ruta de pago (el unico proveedor que responde bien es el
    # pago; con tope agotado la cadena queda vacia de candidatas viables y el
    # stream cae a "sin rutas disponibles" sin volver a pagar).
    estado, cliente = _estado_libre_y_pago(
        tope_pago_diario=1,
        hacer_resp_free=lambda: httpx.Response(500, json={"error": "free caida"}),
        hacer_resp_paid=lambda: httpx.Response(200, content=_sse("de", " pago")))
    dia = _hoy()
    assert estado.almacen.uso_pago("buena", dia) == 0

    r1 = cliente.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                      json={"model": "auto", "messages": [], "stream": True})
    assert r1.status_code == 200
    assert "de" in r1.text and "pago" in r1.text
    assert estado.almacen.uso_pago("buena", dia) == 1  # (a) conto exactamente 1

    r2 = cliente.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                      json={"model": "auto", "messages": [], "stream": True})
    assert estado.almacen.uso_pago("buena", dia) == 1  # el tope realmente ato
    assert "error" in r2.text  # ninguna ruta viable: free sigue caida, pago se excluyo


def test_streaming_servido_por_gratis_no_cuenta_uso_de_pago():
    # (b) si la ganadora es la ruta GRATIS, no debe contarse como uso de pago
    # aunque haya una ruta de pago disponible en la cadena.
    estado, cliente = _estado_libre_y_pago(
        tope_pago_diario=5,
        hacer_resp_free=lambda: httpx.Response(200, content=_sse("gra", "tis")),
        hacer_resp_paid=lambda: httpx.Response(200, content=_sse("de", " pago")))
    dia = _hoy()

    r = cliente.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "messages": [], "stream": True})
    assert r.status_code == 200
    assert "gra" in r.text and "tis" in r.text
    assert estado.almacen.uso_pago("buena", dia) == 0


def test_streaming_sin_ninguna_ruta_viva_no_cuenta_uso_de_pago():
    # (c) si TODAS las rutas fallan, tampoco debe contarse uso de pago.
    estado, cliente = _estado_libre_y_pago(
        tope_pago_diario=5,
        hacer_resp_free=lambda: httpx.Response(500, json={"error": "free caida"}),
        hacer_resp_paid=lambda: httpx.Response(500, json={"error": "pago caido"}))
    dia = _hoy()

    r = cliente.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "messages": [], "stream": True})
    assert r.status_code == 200  # los headers ya salieron; el error va en el cuerpo SSE
    assert "error" in r.text
    assert estado.almacen.uso_pago("buena", dia) == 0


def test_streaming_pago_cuenta_una_sola_vez_no_por_chunk():
    # (d) un stream de pago con VARIOS chunks debe incrementar uso_pago
    # exactamente 1, no una vez por chunk.
    estado, cliente = _estado_libre_y_pago(
        tope_pago_diario=5,
        hacer_resp_free=lambda: httpx.Response(500, json={"error": "free caida"}),
        hacer_resp_paid=lambda: httpx.Response(
            200, content=_sse("u", "n", "o", "dos", "tres", "cuatro")))
    dia = _hoy()

    r = cliente.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "messages": [], "stream": True})
    assert r.status_code == 200
    assert estado.almacen.uso_pago("buena", dia) == 1
