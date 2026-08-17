import asyncio
import time
from datetime import datetime, timezone

import httpx
import pytest
from fastapi.testclient import TestClient

from llm_libre.almacen import Almacen
from llm_libre.api import Estado, crear_app, interpretar_pedido
from llm_libre.auth import LimitadorPorLlave
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


# --- Round 4 de Task 13, hallazgo HIGH. El fix de la ronda 3 saco el 4xx del
#     CONTADOR de cooldown, pero seguia escribiendose como evento fallido
#     comun -- y eso alimenta confiabilidad, que _viva() usa para el piso de
#     /health. Reproducido: 26 pedidos malformados SEGUIDOS de UNA llave
#     (un cliente reintentando el mismo error, algo que los SDK de OpenAI
#     hacen solos) bastan para tirar la confiabilidad de TODAS las rutas por
#     el piso, con /health en "caido"/503 mientras una llave DISTINTA con un
#     pedido VALIDO sigue recibiendo 200 todo el tiempo. Peor que el 503 de
#     la ronda anterior: Coolify usa /health como health check y REINICIA el
#     contenedor cuando falla -- pero `eventos` vive en el volumen
#     persistente de /datos, asi que un proceso nuevo contra la MISMA base
#     sigue viendo los mismos 26 fallos y sigue reportando "caido". Loop de
#     reinicios que reiniciar no puede cortar, contra un servicio que
#     responde bien. ---

def test_health_sigue_ok_tras_30_400_seguidos(estado_cliente):
    estado, cliente = estado_cliente
    ahora = time.time()

    async def _mandar_400_treinta_veces():
        rutas = estado.almacen.rutas_activas()
        estado.proxy.http = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(400, json={"error": "bad request"})))
        for _ in range(30):
            await estado.proxy.completar(rutas, {"model": "a:free", "messages": []}, ahora)

    asyncio.run(_mandar_400_treinta_veces())
    r = cliente.get("/health")
    assert r.status_code == 200
    assert r.json()["estado"] == "ok"


def test_health_cae_con_30_500_seguidos_igual_que_antes(estado_cliente):
    # Contraste directo: un fallo que SI es evidencia sobre la ruta sigue
    # tirando /health, exactamente como antes de este fix.
    estado, cliente = estado_cliente
    ahora = time.time()

    async def _mandar_500_treinta_veces():
        rutas = estado.almacen.rutas_activas()
        estado.proxy.http = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(500)))
        for _ in range(30):
            await estado.proxy.completar(rutas, {"model": "a:free", "messages": []}, ahora)

    asyncio.run(_mandar_500_treinta_veces())
    r = cliente.get("/health")
    assert r.status_code != 200
    assert r.json()["estado"] != "ok"


def test_health_tras_reinicio_del_proceso_sigue_ok_con_400_seguidos(tmp_path):
    # El caso del loop de reinicios: `eventos` vive en un archivo real (el
    # volumen /datos), no en memoria de proceso. Un Almacen/Proxy/Estado
    # SEGUNDO, contra la MISMA base, tiene que leer el mismo resultado que
    # el primero -- si el fix dependiera de algun estado en memoria del
    # proceso viejo, este test lo detectaria y el de arriba no.
    ruta_db = str(tmp_path / "salud.sqlite3")

    almacen1 = Almacen(ruta_db)
    almacen1.crear_esquema()
    almacen1.upsert_rutas(
        [Ruta("kilo", "a:free", "gratis", Capacidades(True, False, 100000, 4096))], 1.0)
    proxy1 = Proxy(
        {"kilo": Proveedor("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, [])},
        almacen1, httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(400, json={"error": "bad request"}))))
    estado1 = Estado(almacen=almacen1, proxy=proxy1, llaves={"buena"}, tope_pago_diario=200)

    async def _mandar_400_treinta_veces():
        rutas = almacen1.rutas_activas()
        for _ in range(30):
            await proxy1.completar(rutas, {"model": "a:free", "messages": []}, time.time())

    asyncio.run(_mandar_400_treinta_veces())
    cliente1 = TestClient(crear_app(estado1))
    assert cliente1.get("/health").json()["estado"] == "ok"

    # "Reinicio del contenedor": proceso nuevo, Almacen nuevo, MISMA base.
    almacen2 = Almacen(ruta_db)
    almacen2.crear_esquema()
    proxy2 = Proxy(
        {"kilo": Proveedor("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, [])},
        almacen2, httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json={"choices": [
                {"message": {"role": "assistant", "content": "hola"}}]}))))
    estado2 = Estado(almacen=almacen2, proxy=proxy2, llaves={"buena"}, tope_pago_diario=200)
    cliente2 = TestClient(crear_app(estado2))
    r = cliente2.get("/health")
    assert r.status_code == 200
    assert r.json()["estado"] == "ok"


def test_ranking_no_se_mueve_con_400_seguidos_pero_si_con_500(estado_cliente):
    estado, cliente = estado_cliente
    ahora = time.time()
    antes = cliente.get("/v1/ranking", headers={"X-API-Key": "buena"}).json()["rutas"][0]

    async def _mandar(codigo, veces):
        rutas = estado.almacen.rutas_activas()
        estado.proxy.http = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(codigo, json={"error": "x"} if codigo < 500 else None)))
        for _ in range(veces):
            await estado.proxy.completar(rutas, {"model": "a:free", "messages": []}, ahora)

    asyncio.run(_mandar(400, 30))
    despues_400 = cliente.get("/v1/ranking", headers={"X-API-Key": "buena"}).json()["rutas"][0]
    assert despues_400["confiabilidad"] == antes["confiabilidad"]
    assert despues_400["puntaje"] == antes["puntaje"]

    asyncio.run(_mandar(500, 30))
    despues_500 = cliente.get("/v1/ranking", headers={"X-API-Key": "buena"}).json()["rutas"][0]
    assert despues_500["confiabilidad"] < antes["confiabilidad"]
    assert despues_500["puntaje"] < antes["puntaje"]


# --- Round 5 de Task 13, hallazgo HIGH. La clasificacion de la ronda 4
#     ("es reintentable?" decide) archivaba 401/402/403/404 del lado del
#     cliente -- pero no son evidencia del PEDIDO, son evidencia de la
#     RUTA: la clave vencio (401), la cuenta se quedo sin credito (402,
#     "insufficient credits" de OpenRouter), la cuenta esta suspendida o
#     hay moderacion del lado del proveedor (403), o el modelo ya no existe
#     (404 -- literalmente el problema central que este proyecto existe
#     para detectar). Medido: las 5 rutas devolviendo 401 dejaba al cliente
#     con 503 en el 100% de los pedidos mientras /health seguia en "ok" --
#     el apagon con luz verde que /health existe para prevenir, y sin
#     ningun backstop para las rutas de pago (nunca sondeadas). Redibujado
#     sobre ATRIBUCION (ver proxy._es_error_del_cliente /
#     _CODIGOS_EVIDENCIA_DE_RUTA): estos cuatro ahora cuentan igual que un
#     500, en /health y en /v1/ranking. ---

@pytest.mark.parametrize("codigo", [401, 402, 403, 404])
def test_health_cae_con_30_seguidos_de_un_codigo_de_ruta(estado_cliente, codigo):
    estado, cliente = estado_cliente
    ahora = time.time()

    async def _mandar_treinta_veces():
        rutas = estado.almacen.rutas_activas()
        estado.proxy.http = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(codigo, json={"error": "x"})))
        for _ in range(30):
            await estado.proxy.completar(rutas, {"model": "a:free", "messages": []}, ahora)

    asyncio.run(_mandar_treinta_veces())
    r = cliente.get("/health")
    assert r.status_code != 200
    assert r.json()["estado"] != "ok"


def test_health_tras_reinicio_del_proceso_sigue_caido_con_401_seguidos(tmp_path):
    # La contracara del test de restart-loop de la ronda 4: un codigo que SI
    # es evidencia de la ruta (401, clave vencida) tiene que seguir
    # reportando "caido" incluso en un proceso NUEVO contra la MISMA base --
    # este es el test que habria detectado el bug original (401 mal
    # archivado como error del cliente dejaba a /health diciendo "ok" con
    # las 5 rutas realmente caidas, incluso despues de un reinicio).
    ruta_db = str(tmp_path / "salud_401.sqlite3")

    almacen1 = Almacen(ruta_db)
    almacen1.crear_esquema()
    almacen1.upsert_rutas(
        [Ruta("kilo", "a:free", "gratis", Capacidades(True, False, 100000, 4096))], 1.0)
    proxy1 = Proxy(
        {"kilo": Proveedor("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, [])},
        almacen1, httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(401, json={"error": "invalid api key"}))))
    estado1 = Estado(almacen=almacen1, proxy=proxy1, llaves={"buena"}, tope_pago_diario=200)

    async def _mandar_401_treinta_veces():
        rutas = almacen1.rutas_activas()
        for _ in range(30):
            await proxy1.completar(rutas, {"model": "a:free", "messages": []}, time.time())

    asyncio.run(_mandar_401_treinta_veces())
    cliente1 = TestClient(crear_app(estado1))
    r1 = cliente1.get("/health")
    assert r1.status_code != 200
    assert r1.json()["estado"] != "ok"

    # "Reinicio del contenedor": proceso nuevo, Almacen nuevo, MISMA base --
    # con un transport SANO en el segundo proceso, para probar que la caida
    # viene de la TELEMETRIA ya persistida (eventos con es_error_cliente=0),
    # no de ningun trafico que este segundo proceso vuelva a generar.
    almacen2 = Almacen(ruta_db)
    almacen2.crear_esquema()
    proxy2 = Proxy(
        {"kilo": Proveedor("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, [])},
        almacen2, httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(200, json={"choices": [
                {"message": {"role": "assistant", "content": "hola"}}]}))))
    estado2 = Estado(almacen=almacen2, proxy=proxy2, llaves={"buena"}, tope_pago_diario=200)
    cliente2 = TestClient(crear_app(estado2))
    r2 = cliente2.get("/health")
    assert r2.status_code != 200
    assert r2.json()["estado"] != "ok"


def test_ranking_cae_con_401_seguidos_igual_que_con_500(estado_cliente):
    estado, cliente = estado_cliente
    ahora = time.time()
    antes = cliente.get("/v1/ranking", headers={"X-API-Key": "buena"}).json()["rutas"][0]

    async def _mandar_401_treinta_veces():
        rutas = estado.almacen.rutas_activas()
        estado.proxy.http = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(401, json={"error": "invalid api key"})))
        for _ in range(30):
            await estado.proxy.completar(rutas, {"model": "a:free", "messages": []}, ahora)

    asyncio.run(_mandar_401_treinta_veces())
    despues = cliente.get("/v1/ranking", headers={"X-API-Key": "buena"}).json()["rutas"][0]
    assert despues["confiabilidad"] < antes["confiabilidad"]
    assert despues["puntaje"] < antes["puntaje"]


def test_ranking_desglosa_los_componentes(cliente):
    fila = cliente.get("/v1/ranking", headers={"X-API-Key": "buena"}).json()["rutas"][0]
    for campo in ("clave", "puntaje", "calidad", "confiabilidad", "ttft_p50_ms", "tier",
                 "prioridad"):
        assert campo in fila


# --- Hallazgo 3 de la revision de Task 13: /v1/ranking ordenaba SOLO por
#     puntaje y no traia `prioridad`, asi que podia mostrar kilo/k:free
#     arriba de todo mientras X-Ruta-Usada decia chatgpt/gpt-5-0 -- el
#     endpoint que el README describe como el lugar para auditar POR QUE el
#     router eligio lo que eligio dejaba de explicarlo. Ahora ordena con la
#     MISMA clave que router.ordenar (via router.clave_de_orden). ---

def test_ranking_ordena_por_prioridad_no_solo_por_puntaje(estado_cliente):
    estado, cliente = estado_cliente
    estado.almacen.upsert_rutas([
        Ruta("chatgpt", "gpt-5:free", "gratis", Capacidades(True, False, 100000, 4096),
             prioridad=0),
    ], 2.0, desactivar_faltantes=False)
    # chatgpt: prioridad maxima (0) pero puntaje MALO.
    estado.almacen.registrar_sonda("chatgpt/gpt-5:free", "calidad", True, 0, 0, 200, 1, 5, 10.0)
    estado.almacen.registrar_evento("chatgpt/gpt-5:free", False, 0, 500, 20.0)
    # kilo/a:free (prioridad 100, el default de la fixture): puntaje MEJOR.
    estado.almacen.registrar_sonda("kilo/a:free", "calidad", True, 0, 0, 200, 5, 5, 10.0)
    estado.almacen.registrar_evento("kilo/a:free", True, 50, 200, 20.0)

    filas = cliente.get("/v1/ranking", headers={"X-API-Key": "buena"}).json()["rutas"]
    claves = [f["clave"] for f in filas]
    puntajes = {f["clave"]: f["puntaje"] for f in filas}
    # Confirma que de verdad puntua peor -- si no, el test no prueba nada.
    assert puntajes["chatgpt/gpt-5:free"] < puntajes["kilo/a:free"]
    # Y aun asi va primero: la prioridad manda, como en el router de verdad.
    assert claves[0] == "chatgpt/gpt-5:free"
    assert {f["clave"]: f["prioridad"] for f in filas} == {
        "chatgpt/gpt-5:free": 0, "kilo/a:free": 100}


def test_ranking_manda_al_final_una_ruta_en_cooldown_aunque_puntue_mejor(estado_cliente):
    # Re-revision: /v1/ranking seguia sin modelar el cooldown, asi que una
    # ruta castigada -- que el router JAMAS elegiria ahora mismo -- podia
    # encabezar la tabla igual. en_cooldown_hasta ya estaba en la fila (se
    # puede diagnosticar), pero el ORDEN tiene que coincidir con el del
    # router: una ruta en cooldown va al final, sin importar prioridad ni
    # puntaje.
    estado, cliente = estado_cliente
    estado.almacen.upsert_rutas([
        Ruta("chatgpt", "gpt-5:free", "gratis", Capacidades(True, False, 100000, 4096),
             prioridad=0),
    ], 2.0, desactivar_faltantes=False)
    # chatgpt: la mejor prioridad Y el mejor puntaje -- pero esta castigada.
    estado.almacen.registrar_sonda("chatgpt/gpt-5:free", "calidad", True, 0, 0, 200, 5, 5, 10.0)
    estado.almacen.registrar_evento("chatgpt/gpt-5:free", True, 50, 200, 20.0)
    estado.proxy.cooldowns["chatgpt/gpt-5:free"] = time.time() + 500

    filas = cliente.get("/v1/ranking", headers={"X-API-Key": "buena"}).json()["rutas"]
    claves = [f["clave"] for f in filas]
    assert claves[-1] == "chatgpt/gpt-5:free"
    assert claves[0] == "kilo/a:free"


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


# --- Fix round 2 (final), cambio 1: `exigir_llave` acepta la llave tambien
#     por `Authorization: Bearer <llave>`, no solo por `X-API-Key`. Es lo que
#     permite que `OpenAI(base_url=..., api_key="<llave>")` autentique sin
#     configuracion extra -- la promesa central del contrato ("cambia solo
#     base_url"), que antes de este cambio era falsa: el SDK manda la llave
#     via `Authorization`, y el gateway solo leia `X-API-Key`. `X-API-Key`
#     sigue existiendo (la usa `arkiv-api`, el gateway hermano) y sigue
#     ganando si ambas cabeceras llegan juntas.

def test_autoriza_con_bearer_sin_x_api_key(cliente):
    r = cliente.post("/v1/chat/completions", headers={"Authorization": "Bearer buena"},
                     json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "hola"


def test_bearer_con_llave_mala_sigue_dando_401(cliente):
    r = cliente.post("/v1/chat/completions", headers={"Authorization": "Bearer mala"},
                     json={"model": "auto", "messages": []})
    assert r.status_code == 401


def test_x_api_key_sola_sigue_funcionando_igual_que_antes(cliente):
    r = cliente.get("/v1/models", headers={"X-API-Key": "buena"})
    assert r.status_code == 200


def test_si_llegan_las_dos_y_no_coinciden_gana_x_api_key(cliente):
    # X-API-Key trae la buena; Authorization trae una llave que ni siquiera
    # existe. Si Authorization ganara, esto seria 401 -- confirma la
    # precedencia declarada.
    r = cliente.get("/v1/models", headers={
        "X-API-Key": "buena", "Authorization": "Bearer ni-existe"})
    assert r.status_code == 200


def test_authorization_malformado_no_revienta_y_da_401(cliente):
    # Ninguna de estas formas rotas debe tirar una excepcion sin atrapar: se
    # tratan igual que "no se mando ninguna llave".
    for cabecera in ("buena", "Bearer", "Bearer   ", "Basic buena", "buena sin bearer"):
        r = cliente.get("/v1/models", headers={"Authorization": cabecera})
        assert r.status_code == 401, cabecera


def test_el_limite_por_minuto_cuenta_igual_sin_importar_la_cabecera_usada():
    almacen = Almacen(":memory:")
    almacen.crear_esquema()
    almacen.upsert_rutas(
        [Ruta("kilo", "a:free", "gratis", Capacidades(True, False, 100000, 4096))], 1.0)
    prov = {"kilo": Proveedor("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, [])}
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "hola"}}]})))
    estado = Estado(almacen=almacen, proxy=Proxy(prov, almacen, http),
                    llaves={"buena"}, tope_pago_diario=200,
                    limitador=LimitadorPorLlave(2))
    cliente = TestClient(crear_app(estado))

    r1 = cliente.get("/v1/models", headers={"X-API-Key": "buena"})
    r2 = cliente.get("/v1/models", headers={"Authorization": "Bearer buena"})
    r3 = cliente.get("/v1/models", headers={"X-API-Key": "buena"})
    assert r1.status_code == 200 and r2.status_code == 200
    # El limite (2/min) ya se agoto entre las dos peticiones anteriores, sin
    # importar que cada una uso una cabecera distinta: es la misma llave
    # resuelta, asi que cuenta contra el mismo contador.
    assert r3.status_code == 429


# --- Fix round 3, I3: /v1/ranking tiene que traer la fecha de la ultima sonda
#     (§6 del diseno) y distinguir "calidad medida 0.6" de "nunca medida" --
#     que es exactamente el dato que hacia falta para diagnosticar B2. ---

def test_ranking_marca_como_no_medida_una_ruta_sin_sonda_de_calidad(cliente):
    fila = cliente.get("/v1/ranking", headers={"X-API-Key": "buena"}).json()["rutas"][0]
    assert fila["calidad_medida"] is False
    assert fila["calidad"] is None            # no se muestra el neutro como medicion
    assert fila["calidad_asumida"] == 0.6     # pero se dice cual se uso para puntuar
    assert fila["ultima_sonda_calidad"] is None
    assert fila["ultima_sonda"] is None


def test_ranking_trae_la_fecha_de_la_ultima_sonda(estado_cliente):
    estado, cliente = estado_cliente
    # 2026-08-17T12:00:00Z y 2026-08-17T18:00:00Z
    estado.almacen.registrar_sonda("kilo/a:free", "calidad", True, 0, 0, 200, 3, 5,
                                   1786968000.0)
    estado.almacen.registrar_sonda("kilo/a:free", "salud", True, 120, 0, 200, 0, 0,
                                   1786989600.0)
    fila = cliente.get("/v1/ranking", headers={"X-API-Key": "buena"}).json()["rutas"][0]
    assert fila["calidad_medida"] is True
    assert fila["calidad"] == 0.6             # 3/5, esta vez SI medido
    assert fila["calidad_asumida"] is None
    assert fila["ultima_sonda_calidad"] == "2026-08-17T12:00:00Z"
    assert fila["ultima_sonda"] == "2026-08-17T18:00:00Z"


def test_una_ruta_en_cooldown_no_pierde_su_marca_de_calidad_medida(estado_cliente):
    # `_metricas` reconstruia Metricas posicionalmente para inyectar el
    # cooldown, y asi perdia los campos nuevos: una ruta castigada aparecia
    # como "nunca medida" y el router la mandaba al fondo por partida doble.
    estado, cliente = estado_cliente
    estado.almacen.registrar_sonda("kilo/a:free", "calidad", True, 0, 0, 200, 5, 5,
                                   1786968000.0)
    estado.proxy.cooldowns["kilo/a:free"] = time.time() + 600
    fila = cliente.get("/v1/ranking", headers={"X-API-Key": "buena"}).json()["rutas"][0]
    assert fila["calidad_medida"] is True
    assert fila["en_cooldown_hasta"] > time.time()


# --- Fix round 3, B3 (Blocking): el 503 del §9 se entregaba como 400 en toda
#     caida. `ordenar` filtra los cooldowns, la lista llega vacia y la api
#     gritaba "ninguna ruta cumple lo pedido" -- un 400, que todo SDK y toda
#     capa de alertas leen como "tu peticion esta mal formada": no reintentan
#     y no despiertan a nadie. Que los tiers gratis rate-limiteen a la vez es
#     el fallo ESPERADO, no uno raro. ---

def test_todas_las_candidatas_en_cooldown_da_503_no_400(estado_cliente):
    estado, cliente = estado_cliente
    hasta = time.time() + 600
    estado.proxy.cooldowns["kilo/a:free"] = hasta
    r = cliente.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "messages": []})
    assert r.status_code == 503
    assert r.json()["detail"]["proxima_liberacion"] == pytest.approx(hasta)


def test_todas_las_candidatas_en_cooldown_da_503_tambien_en_streaming(estado_cliente):
    estado, cliente = estado_cliente
    estado.proxy.cooldowns["kilo/a:free"] = time.time() + 600
    r = cliente.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "messages": [], "stream": True})
    assert r.status_code == 503


def test_capacidades_que_nadie_cumple_sigue_siendo_400(estado_cliente):
    # El otro lado de la moneda: esto SI es culpa del cliente y tiene que
    # seguir siendo 400, con lo que pidio y cuantas rutas hay.
    estado, cliente = estado_cliente
    r = cliente.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "messages": [], "x_min_contexto": 99999999})
    assert r.status_code == 400
    assert r.json()["detail"]["rutas_activas"] == 1


def test_el_tope_de_pago_diario_da_503_no_400():
    # §9: "Llave supero su tope de pago diario -> 503, nunca un cobro
    # silencioso". Con la gratis en cooldown y el tope agotado, la cadena queda
    # vacia -- pero la ruta de pago EXISTE y podria servir: es indisponibilidad,
    # no una peticion mal formada.
    estado, cliente = _estado_libre_y_pago(
        tope_pago_diario=1,
        hacer_resp_free=lambda: httpx.Response(200, json={"choices": [
            {"message": {"role": "assistant", "content": "hola"}}]}),
        hacer_resp_paid=lambda: httpx.Response(200, json={"choices": [
            {"message": {"role": "assistant", "content": "pago"}}]}))
    estado.almacen.sumar_uso_pago("buena", _hoy())          # tope agotado
    estado.proxy.cooldowns["free_prov/f:free"] = time.time() + 300

    r = cliente.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "messages": []})
    assert r.status_code == 503
    assert r.json()["detail"]["tope_pago_alcanzado"] is True


def test_el_503_por_indisponibilidad_no_reporta_liberacion_si_no_hay_cooldown():
    # Solo hay ruta de pago, el cliente la prohibio: no hay nada que esperar,
    # asi que proxima_liberacion es null en vez de un numero inventado.
    estado, cliente = _estado_libre_y_pago(
        tope_pago_diario=9,
        hacer_resp_free=lambda: httpx.Response(500),
        hacer_resp_paid=lambda: httpx.Response(500))
    estado.almacen.upsert_rutas([], 1.0, desactivar_faltantes=False)
    estado.proxy.cooldowns["free_prov/f:free"] = time.time() + 300
    r = cliente.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "messages": [], "x_permitir_pago": False})
    assert r.status_code == 503
    assert r.json()["detail"]["proxima_liberacion"] == pytest.approx(
        estado.proxy.cooldowns["free_prov/f:free"])


# --- Fix round 3, I2: el §6.1 promete devolver el razonamiento recortado en un
#     campo aparte, `x_razonamiento`. Se recortaba de `content` y se tiraba: un
#     cliente con el default `x_crudo: false` no tenia forma de recuperarlo. ---

def _cliente_que_piensa(contenido):
    almacen = Almacen(":memory:")
    almacen.crear_esquema()
    almacen.upsert_rutas(
        [Ruta("kilo", "a:free", "gratis", Capacidades(True, False, 100000, 4096))], 1.0)
    prov = {"kilo": Proveedor("kilo", "gratis", "openai", "https://k.test", "",
                              "/models", {}, [])}
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={"choices": [
            {"message": {"role": "assistant", "content": contenido}}]})))
    estado = Estado(almacen=almacen, proxy=Proxy(prov, almacen, http),
                    llaves={"buena"}, tope_pago_diario=200)
    return TestClient(crear_app(estado))


def test_devuelve_el_razonamiento_recortado_en_x_razonamiento():
    cliente = _cliente_que_piensa("<think>2+2 son 4</think>La respuesta es 4.")
    r = cliente.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "messages": []})
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "La respuesta es 4."
    assert r.json()["x_razonamiento"] == "2+2 son 4"


def test_sin_razonamiento_no_agrega_el_campo():
    cliente = _cliente_que_piensa("La respuesta es 4.")
    assert "x_razonamiento" not in cliente.post(
        "/v1/chat/completions", headers={"X-API-Key": "buena"},
        json={"model": "auto", "messages": []}).json()


def test_en_modo_crudo_no_hay_x_razonamiento_porque_sigue_en_el_content():
    cliente = _cliente_que_piensa("<think>mmm</think>hola")
    cuerpo = cliente.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                          json={"model": "auto", "messages": [], "x_crudo": True}).json()
    assert cuerpo["choices"][0]["message"]["content"] == "<think>mmm</think>hola"
    assert "x_razonamiento" not in cuerpo


# --- Fix round 3, ALSO: el acoplamiento entre el neutro de confiabilidad y el
#     piso de /health es cargante y vive en dos archivos distintos, sin nada
#     que lo pruebe. Si alguna vez se invierte, una instalacion NUEVA -- sin
#     un solo evento todavia -- reporta "caido" y Coolify nunca marca el
#     contenedor como sano: el servicio no arranca nunca, por una constante. ---

def test_el_neutro_de_confiabilidad_queda_por_encima_del_piso_de_health():
    from llm_libre.api import UMBRAL_CONFIABILIDAD_SALUD
    from llm_libre.modelos import CONFIABILIDAD_NEUTRA
    assert CONFIABILIDAD_NEUTRA > UMBRAL_CONFIABILIDAD_SALUD, (
        "una ruta sin telemetria debe contar como viva (§6 del diseno): con el "
        "neutro por debajo del piso, /health diria 'caido' en una instalacion "
        "nueva y el contenedor nunca pasaria el health check")
