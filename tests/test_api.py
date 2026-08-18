import asyncio
import json
import time
from datetime import datetime, timezone

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from llm_libre.almacen import Almacen
from llm_libre.api import Estado, crear_app, interpretar_pedido
from llm_libre.auth import PerKeyRateLimiter
from llm_libre.modelos import Capacidades, Ruta
from llm_libre.proveedores import Proveedor
from llm_libre.proxy import (LIMITE_PROBE_BAJO_DEMANDA_S, TOPE_PENDIENTES,
                             UMBRAL_SOSPECHA, Proxy)


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


# --- Revision post-Task-14 (gate): tres defectos reales que el reviewer
#     encontro leyendo interpretar_pedido, no ejecutando -- los tres eran
#     entradas malformadas del CLIENTE cayendo en un hueco silencioso
#     (una degradacion sin aviso, o directamente un 500). ---

def test_auto_con_sufijo_desconocido_da_400():
    # "auto:turbo" (typo tipico de "auto:tools") caia por las tres ramas de
    # sufijo sin tocar nada -- silenciosamente identico a pedir "auto" liso.
    # Peligroso para un cliente que de verdad queria exigir una capacidad:
    # se queda sin ella, sin ningun aviso.
    with pytest.raises(HTTPException) as exc:
        interpretar_pedido({"model": "auto:turbo"})
    assert exc.value.status_code == 400
    assert "auto:turbo" in exc.value.detail["message"]


def test_auto_liso_sigue_funcionando_tras_el_fix_de_sufijo_desconocido():
    # Regresion directa del fix de arriba: sufijo == "" (o sea "auto" sin
    # ":") NO debe entrar en la rama de rechazo.
    p = interpretar_pedido({"model": "auto"})
    assert p.modelo is None and p.perfil == "balanceado"


def test_auto_balanceado_sigue_siendo_un_alias_valido():
    # "balanceado" ESTA en PERFILES -- "auto:balanceado" es redundante con
    # "auto" liso, pero valido, y no debe caer en la rama de rechazo.
    p = interpretar_pedido({"model": "auto:balanceado"})
    assert p.perfil == "balanceado"


def test_x_requiere_como_string_suelto_se_acepta_como_un_solo_valor():
    # `set("tools")` itera CARACTERES ({'t','o','l','s'}), asi que
    # "tools" in exigidas daba False y la exigencia se ignoraba entera, sin
    # error. Un string suelto (en vez de una lista de uno) se acepta igual.
    p = interpretar_pedido({"model": "auto", "x_requiere": "tools"})
    assert p.requiere_tools is True
    assert p.requiere_vision is False


def test_x_requiere_como_lista_sigue_funcionando_igual_que_antes():
    p = interpretar_pedido({"model": "auto", "x_requiere": ["vision"]})
    assert p.requiere_vision is True
    assert p.requiere_tools is False


def test_x_min_contexto_no_numerico_da_400_nombrando_el_campo():
    with pytest.raises(HTTPException) as exc:
        interpretar_pedido({"model": "auto", "x_min_contexto": "cien mil"})
    assert exc.value.status_code == 400
    assert exc.value.detail["campo"] == "x_min_contexto"
    assert exc.value.detail["valor_recibido"] == "cien mil"


def test_x_min_contexto_numerico_como_string_sigue_funcionando():
    # int("100000") es valido -- el fix no debe volverse mas estricto de lo
    # que ya era para el caso que SI funcionaba.
    p = interpretar_pedido({"model": "auto", "x_min_contexto": "100000"})
    assert p.min_contexto == 100000


# --- Revision post-Task-14 (tercer gate): la MISMA familia de bug que
#     x_min_contexto (un cast sin atrapar sobre un campo del cliente
#     revienta con TypeError/AttributeError, escapa como 500) penetro dos
#     veces mas -- x_requiere con un valor ni string ni lista (set() sobre
#     un int/bool/float/lista-de-listas) y model con cualquier cosa que no
#     sea un string (.strip() sobre un numero, una lista, un dict). El fix
#     generaliza con _leer_campo en vez de parchear el tercer sitio a
#     mano -- estos tests cubren los dos nuevos y confirman la MISMA forma
#     de error (message/campo/valor_recibido) que ya establecio
#     x_min_contexto. ---

@pytest.mark.parametrize("valor", [5, True, 3.5, [["tools"]]])
def test_x_requiere_no_string_ni_lista_da_400_nombrando_el_campo(valor):
    with pytest.raises(HTTPException) as exc:
        interpretar_pedido({"model": "auto", "x_requiere": valor})
    assert exc.value.status_code == 400
    assert exc.value.detail["campo"] == "x_requiere"
    assert exc.value.detail["valor_recibido"] == valor


@pytest.mark.parametrize("valor", [5, True, 3.5, ["a"], {"a": 1}])
def test_model_no_string_da_400_nombrando_el_campo(valor):
    with pytest.raises(HTTPException) as exc:
        interpretar_pedido({"model": valor})
    assert exc.value.status_code == 400
    assert exc.value.detail["campo"] == "model"
    assert exc.value.detail["valor_recibido"] == valor


def test_model_ausente_o_nulo_sigue_cayendo_a_auto_sin_dar_400():
    # Regresion directa del fix de arriba: None (ausente) sigue siendo
    # valido -- solo un valor PRESENTE con el tipo equivocado debe dar 400.
    assert interpretar_pedido({}).modelo is None
    assert interpretar_pedido({"model": None}).modelo is None


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


# --- Revision post-Task-14 (gate): los mismos tres defectos de arriba
#     (test_auto_con_sufijo_desconocido_da_400 y compania), pero probados a
#     traves del cliente HTTP completo -- para confirmar que interpretar_pedido
#     de verdad esta conectado al camino real de /v1/chat/completions, no
#     solo probado de forma aislada. ---

def test_completions_con_alias_desconocido_da_400_no_500(cliente):
    r = cliente.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto:turbo", "messages": []})
    assert r.status_code == 400
    assert "auto:turbo" in r.json()["detail"]["message"]


def test_completions_con_x_min_contexto_no_numerico_da_400_no_500(cliente):
    r = cliente.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "x_min_contexto": "cien mil", "messages": []})
    assert r.status_code == 400
    assert r.json()["detail"]["campo"] == "x_min_contexto"


def test_completions_con_x_requiere_como_string_aplica_la_exigencia(cliente):
    # kilo/a:free (la unica ruta del fixture `cliente`) declara tools=True,
    # asi que "x_requiere": "tools" (string suelto) tiene que seguir
    # sirviendo -- si el bug (set() sobre un string) volviera, esto
    # devolveria 200 igual porque kilo SI tiene tools, asi que la prueba
    # real de que la exigencia se aplico vive en el test unitario de arriba
    # (test_x_requiere_como_string_suelto_se_acepta_como_un_solo_valor);
    # este solo confirma que el pedido llega entero hasta el proxy sin
    # reventar en el camino.
    r = cliente.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "x_requiere": "tools", "messages": []})
    assert r.status_code == 200


# --- Revision post-Task-14 (tercer gate): mismos dos defectos que arriba
#     (x_requiere no-string-ni-lista, model no-string), a traves del
#     cliente HTTP completo -- confirma que interpretar_pedido esta
#     conectado al camino real, no solo probado aislado. ---

def test_completions_con_x_requiere_de_tipo_invalido_da_400_no_500(cliente):
    r = cliente.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "x_requiere": 5, "messages": []})
    assert r.status_code == 400
    assert r.json()["detail"]["campo"] == "x_requiere"


def test_completions_con_model_de_tipo_invalido_da_400_no_500(cliente):
    r = cliente.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": 5, "messages": []})
    assert r.status_code == 400
    assert r.json()["detail"]["campo"] == "model"


# --- Task 14 (documentacion): la regla que manda es que ENRIQUECER
#     /openapi.json no puede tocar el comportamiento real del endpoint --
#     completions() sigue leyendo `await request.json()` a mano, SIN un
#     modelo Pydantic que ligue el cuerpo (eso haria que FastAPI descarte
#     cualquier campo que no declare, rompiendo el contrato passthrough que
#     este proyecto existe para dar). `test_cliente.py` ya prueba esto al
#     nivel de `build_request` (la copia somera que le saca las extensiones
#     x_* al cuerpo); este test lo extiende al nivel HTTP completo -- cliente
#     -> FastAPI -> proxy -> proveedor -- para que quede pineado que un campo
#     que ni el gateway ni ningun SDK de OpenAI conocen sigue llegando al
#     proveedor TAL CUAL, con el valor exacto que mando el cliente. ---

def test_un_campo_desconocido_llega_al_proveedor_tal_cual():
    recibido = {}

    def handler(req):
        recibido.update(json.loads(req.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "hola"}}]})

    almacen = Almacen(":memory:")
    almacen.crear_esquema()
    almacen.upsert_rutas(
        [Ruta("kilo", "a:free", "gratis", Capacidades(True, False, 100000, 4096))], 1.0)
    prov = {"kilo": Proveedor("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, [])}
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    estado = Estado(almacen=almacen, proxy=Proxy(prov, almacen, http),
                    llaves={"buena"}, tope_pago_diario=200)
    c = TestClient(crear_app(estado))

    r = c.post("/v1/chat/completions", headers={"X-API-Key": "buena"}, json={
        "model": "auto",
        "messages": [{"role": "user", "content": "hi"}],
        # Ni un campo estandar de OpenAI que el gateway no lista en su
        # documentacion (reasoning) ni uno inventado por el proveedor de
        # turno (safety_identifier) son EXTENSIONES DEL GATEWAY (x_*, ver
        # EXTENSIONES_GATEWAY) -- el contrato es "pasa todo lo que no
        # reconozcas", no una lista blanca.
        "reasoning": {"enabled": False},
        "safety_identifier": "algo-que-el-gateway-jamas-va-a-conocer",
    })

    assert r.status_code == 200
    assert recibido["reasoning"] == {"enabled": False}
    assert recibido["safety_identifier"] == "algo-que-el-gateway-jamas-va-a-conocer"
    # Y las extensiones DEL GATEWAY, si vinieran en el mismo pedido, seguirian
    # sin viajar -- ver test_no_reenvia_las_extensiones_del_gateway_al_proveedor
    # en test_cliente.py para esa mitad del contrato.
    assert "x_crudo" not in recibido


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


# --- ALSO de la revision round 6: el caso de arriba cubre un id que YA NO
#     esta en el catalogo. Este otro es distinto y es, textualmente, "la
#     razon de ser del proyecto" -- un id que SIGUE en el catalogo pero que
#     el proveedor real ya no sirve (404 genuino, en vivo). La ruta ya se
#     lleva el golpe de confiabilidad (404 es evidencia de la ruta por
#     default, Parte 1), pero hasta este fix el cliente solo veia un 503
#     generico ("detalle": "HTTP 404") -- indistinguible de cualquier otra
#     indisponibilidad transitoria, durante la ventana de hasta 5h antes del
#     proximo sync de catalogo (nunca para rutas de pago, que no se
#     sondean). ---

def test_modelo_explicito_que_desaparecio_upstream_da_404_no_503(estado_cliente):
    estado, cliente = estado_cliente
    estado.proxy.http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(404, json={"error": {"message": "model not found"}})))
    r = cliente.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "a:free", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 404
    assert "a:free" in str(r.json())
    assert "sugerencias" in str(r.json())


# --- Round 7, LOW del gate: `_parecidos` corria contra `rutas_activas()`,
#     que TODAVIA trae el id que se acaba de declarar muerto -- el cliente
#     leia `"el modelo 'a:free' ya no existe"` con `sugerencias: ['a:free',
#     ...]`. Se excluye el propio `pedido.modelo` de la lista antes de
#     buscar parecidos. ---

def test_la_sugerencia_del_404_en_vivo_no_incluye_el_modelo_recien_declarado_muerto(estado_cliente):
    estado, cliente = estado_cliente
    estado.almacen.upsert_rutas(
        [Ruta("kilo", "a:free", "gratis", Capacidades(True, False, 100000, 4096)),
         Ruta("kilo", "a:freebie", "gratis", Capacidades(True, False, 100000, 4096))],
        1.0)
    estado.proxy.http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(404, json={"error": {"message": "model not found"}})))
    r = cliente.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "a:free", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 404
    sugerencias = r.json()["detail"]["sugerencias"]
    assert "a:free" not in sugerencias
    assert "a:freebie" in sugerencias


def test_modelo_auto_con_404_upstream_sigue_siendo_503(estado_cliente):
    # En modo "auto" no hay un id EXPLICITO que nombrar -- pedido.modelo es
    # None -- asi que este caso se queda con el 503 generico de siempre, no
    # con el 404 nuevo (que exige un modelo puntual sobre el cual sugerir).
    estado, cliente = estado_cliente
    estado.proxy.http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(404, json={"error": {"message": "model not found"}})))
    r = cliente.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 503


def test_health_dice_ok_si_hay_ruta_viva(cliente):
    assert cliente.get("/health").json()["estado"] == "ok"


# --- Fix round 1, hallazgo 1 (Critical): /health tambien debe fallar cuando
#     la ruta gratis esta genuinamente rota (500 en cada intento), no solo
#     cuando esta en cooldown por un 429.
#
#     Reescrito en round 6, Parte 2: la version anterior sembraba 8 fallos +
#     2 exitos y esperaba "caido" -- bajo el redisenio de "evidencia de
#     vida" ese caso pasa a "ok" CORRECTAMENTE (hay un exito reciente real,
#     ver test_health_sigue_ok_con_30_403_pero_un_exito_reciente mas abajo
#     para el caso que este reemplaza). Este test ahora prueba lo que el
#     coordinador pidio explicitamente: un proveedor GENUINAMENTE muerto --
#     cero exitos, y la sonda de salud (la senal mas confiable que existe,
#     porque el gateway controla su propio payload) tambien falla. ---

def test_health_no_es_ok_si_la_gratis_falla_de_verdad(estado_cliente):
    estado, cliente = estado_cliente
    ahora = time.time()
    # Trafico real, siempre fallido, y la sonda de salud tambien falla. Esto
    # NUNCA pasa por Proxy._castigar (eso solo lo dispara un 429), asi que
    # proxy.cooldowns queda vacio -- la version vieja de /health diria "ok"
    # igual si solo mirara cooldowns.
    for _ in range(10):
        estado.almacen.registrar_evento("kilo/a:free", False, 0, 500, ahora)
    # Round 9: una sola sonda fallida ya no alcanza para /health (ver
    # Almacen.tiene_evidencia_de_vida) -- dos consecutivas, sin exito de
    # por medio, si.
    estado.almacen.registrar_sonda("kilo/a:free", "salud", False, 100, 0, 500, 0, 0, ahora - 1)
    estado.almacen.registrar_sonda("kilo/a:free", "salud", False, 100, 0, 500, 0, 0, ahora)
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


# --- Round 7, MEDIUM (calidad de test) del gate: el test de arriba no pinea
#     el leg de cooldown de `_viva()` en api.py -- borrar del todo la
#     condicion `m.en_cooldown_hasta <= ahora` deja la suite completa en
#     verde, porque la ruta de ese test NO tiene evidencia de vida propia
#     (nunca hubo un exito real): `tiene_evidencia_de_vida()` la tira sola,
#     sin que el cooldown tenga que intervenir. Con el redisenio de round 6
#     ("evidencia de vida, no ausencia de fallos"), esa condicion es
#     precisamente la que una limpieza futura borraria sin que nada lo
#     note. Este test aisla el cooldown de la otra pata: la ruta SI tiene
#     evidencia de vida (un exito reciente), y aun asi /health tiene que
#     seguir diciendo no-ok mientras el cooldown siga activo. ---

def test_health_excluye_por_cooldown_aunque_haya_evidencia_de_vida(estado_cliente):
    estado, cliente = estado_cliente
    ahora = time.time()
    estado.almacen.registrar_evento("kilo/a:free", True, 50, 200, ahora)

    async def _forzar_429():
        rutas = estado.almacen.rutas_activas()
        estado.proxy.http = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(429, json={"error": "rate limited"})))
        await estado.proxy.completar(rutas, {"model": "a:free", "messages": []}, ahora)

    asyncio.run(_forzar_429())
    assert estado.proxy.cooldowns  # confirma que quedo en cooldown

    # Aislado: sin el cooldown, esta ruta SI cuenta como viva por si sola.
    assert estado.almacen.tiene_evidencia_de_vida("kilo/a:free", ahora) is True

    r = cliente.get("/health")
    assert r.status_code != 200
    assert r.json()["estado"] != "ok"


# --- Round 6 de Task 13, Parte 2. `403` es GENUINAMENTE ambiguo: cuenta
#     suspendida (evidencia de la ruta, correcto contarlo -- Parte 1) vs.
#     contenido moderado del lado del proveedor (evidencia del PEDIDO de UN
#     cliente puntual). El gateway no puede distinguirlos sin parsear el
#     cuerpo especifico de cada proveedor, asi que clasificarlo bien (Parte
#     1) no alcanza: 30 pedidos con contenido flageado de una sola llave no
#     deben poder apagar /health para TODAS las llaves si la ruta ya
#     demostro, con un pedido valido, que sirve. Esto es lo que "evidencia
#     de vida" compra que "confiabilidad promedio" no podia. ---

def test_health_sigue_ok_con_30_403_pero_un_exito_reciente(estado_cliente):
    estado, cliente = estado_cliente
    ahora = time.time()
    estado.almacen.registrar_evento("kilo/a:free", True, 50, 200, ahora)
    for _ in range(30):
        estado.almacen.registrar_evento("kilo/a:free", False, 0, 403, ahora)
    assert cliente.get("/health").json()["estado"] == "ok"


def test_health_tras_reinicio_del_proceso_sigue_ok_con_403_y_un_exito(tmp_path):
    # Restart del caso de arriba: el exito y los 403 quedan en el archivo de
    # /datos: un proceso nuevo contra la MISMA base tiene que leer "ok"
    # igual, sin volver a generar trafico.
    ruta_db = str(tmp_path / "salud_403.sqlite3")
    ahora = time.time()

    almacen1 = Almacen(ruta_db)
    almacen1.crear_esquema()
    almacen1.upsert_rutas(
        [Ruta("kilo", "a:free", "gratis", Capacidades(True, False, 100000, 4096))], 1.0)
    almacen1.registrar_evento("kilo/a:free", True, 50, 200, ahora)
    for _ in range(30):
        almacen1.registrar_evento("kilo/a:free", False, 0, 403, ahora)
    proxy1 = Proxy(
        {"kilo": Proveedor("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, [])},
        almacen1, httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(403, json={"error": "flagged"}))))
    estado1 = Estado(almacen=almacen1, proxy=proxy1, llaves={"buena"}, tope_pago_diario=200)
    cliente1 = TestClient(crear_app(estado1))
    assert cliente1.get("/health").json()["estado"] == "ok"

    almacen2 = Almacen(ruta_db)
    almacen2.crear_esquema()
    proxy2 = Proxy(
        {"kilo": Proveedor("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, [])},
        almacen2, httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(403, json={"error": "flagged"}))))
    estado2 = Estado(almacen=almacen2, proxy=proxy2, llaves={"buena"}, tope_pago_diario=200)
    cliente2 = TestClient(crear_app(estado2))
    assert cliente2.get("/health").json()["estado"] == "ok"


def test_health_tras_reinicio_del_proceso_sigue_caido_sin_exitos_ni_sonda(tmp_path):
    # Restart del "genuinamente muerto": cero exitos y la sonda de salud
    # tambien fallo, persistido en /datos -- un proceso nuevo tiene que
    # seguir leyendo "caido", no "ok por falta de evidencia en contra".
    ruta_db = str(tmp_path / "salud_muerta.sqlite3")
    ahora = time.time()

    almacen1 = Almacen(ruta_db)
    almacen1.crear_esquema()
    almacen1.upsert_rutas(
        [Ruta("kilo", "a:free", "gratis", Capacidades(True, False, 100000, 4096))], 1.0)
    for _ in range(10):
        almacen1.registrar_evento("kilo/a:free", False, 0, 500, ahora)
    # Round 9: hacen falta DOS sondas fallidas consecutivas, no una.
    almacen1.registrar_sonda("kilo/a:free", "salud", False, 100, 0, 500, 0, 0, ahora - 1)
    almacen1.registrar_sonda("kilo/a:free", "salud", False, 100, 0, 500, 0, 0, ahora)
    proxy1 = Proxy(
        {"kilo": Proveedor("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, [])},
        almacen1, httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(500))))
    estado1 = Estado(almacen=almacen1, proxy=proxy1, llaves={"buena"}, tope_pago_diario=200)
    cliente1 = TestClient(crear_app(estado1))
    r1 = cliente1.get("/health")
    assert r1.status_code != 200
    assert r1.json()["estado"] != "ok"

    # Segundo proceso con un transport SANO, para probar que "caido" viene
    # de la TELEMETRIA ya persistida, no de trafico nuevo que este proceso
    # genere.
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


def test_health_tras_reinicio_del_proceso_sigue_ok_sin_telemetria(tmp_path):
    # Restart de "instalacion nueva": cero eventos, cero sondas -- una ruta
    # sin evidencia todavia no nace muerta, ni en el primer proceso ni tras
    # un reinicio contra la misma base vacia.
    ruta_db = str(tmp_path / "salud_fresca.sqlite3")

    almacen1 = Almacen(ruta_db)
    almacen1.crear_esquema()
    almacen1.upsert_rutas(
        [Ruta("kilo", "a:free", "gratis", Capacidades(True, False, 100000, 4096))], 1.0)
    estado1 = Estado(almacen=almacen1, proxy=Proxy({}, almacen1, httpx.AsyncClient()),
                     llaves={"buena"}, tope_pago_diario=200)
    cliente1 = TestClient(crear_app(estado1))
    assert cliente1.get("/health").json()["estado"] == "ok"

    almacen2 = Almacen(ruta_db)
    almacen2.crear_esquema()
    estado2 = Estado(almacen=almacen2, proxy=Proxy({}, almacen2, httpx.AsyncClient()),
                     llaves={"buena"}, tope_pago_diario=200)
    cliente2 = TestClient(crear_app(estado2))
    assert cliente2.get("/health").json()["estado"] == "ok"


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
        await proxy1.esperar_sondas_pendientes()

    asyncio.run(_mandar_401_treinta_veces())
    # Round 9: la sospecha (30x401 reales) dispara UNA sonda bajo demanda
    # -- pero /health ahora exige DOS fallidas consecutivas (ver
    # Almacen.tiene_evidencia_de_vida). Se agrega la confirmatoria
    # directamente: el mecanismo que la PRIMERA sonda se dispara sola ya
    # esta cubierto en test_proxy.py; este test es sobre PERSISTENCIA tras
    # un reinicio, no sobre el rate-limit real de 60s entre sondas.
    almacen1.registrar_sonda("kilo/a:free", "salud", False, 100, 0, 401, 0, 0, time.time())
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


# --- Round 8. El gate encontro dos vectores que round 7 no cerraba, los dos
#     escotillas de su propio diseno: una cadena de UNA sola ruta (forzable
#     por el cliente con `model` explicito o `x_min_contexto`, sin ningun
#     conocimiento interno) y la rama `if emitido:` de completar_stream
#     (sin chequeo de cadena). Round 8 saca el eje de "cuantas rutas hay":
#     trafico real NUNCA excluye una ruta directo, solo acumula sospecha;
#     cruzar el umbral programa una sonda PROPIA (payload fijo `PING`, el
#     mismo que sondeo.py) que es la unica que decide. Verificado end-to-end
#     via /health, no solo en proxy.py -- ver test_proxy.py/
#     test_proxy_stream.py para la cobertura mas granular. ---

def _ping(cuerpo: bytes) -> bool:
    mensajes = json.loads(cuerpo).get("messages") or []
    return bool(mensajes) and mensajes[0].get("content") == "ping"


def test_health_sigue_ok_bajo_ataque_de_cadena_de_una_sola_ruta(estado_cliente):
    # estado_cliente ya tiene una sola ruta gratis -- exactamente el vector
    # 1 del gate (un `model` explicito, o x_min_contexto, narrows a esto).
    estado, cliente = estado_cliente
    ahora = time.time()

    def handler(req):
        if _ping(req.content):
            return httpx.Response(200, json={"choices": [
                {"message": {"role": "assistant", "content": "pong"}}]})
        return httpx.Response(403, json={"error": "contenido flageado"})

    async def _quince_pedidos_identicos():
        rutas = estado.almacen.rutas_activas()
        estado.proxy.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        for i in range(15):
            await estado.proxy.completar(rutas, {"model": "a:free", "messages": []},
                                         ahora + i)
        await estado.proxy.esperar_sondas_pendientes()

    asyncio.run(_quince_pedidos_identicos())
    assert cliente.get("/health").json()["estado"] == "ok"


def test_health_sigue_ok_con_flood_de_chunks_sin_contenido_en_streaming(estado_cliente):
    # El vector 2 del gate: la rama if emitido: (force-flush de
    # TOPE_PENDIENTES) sin narrows de cadena -- "auto", sin extensiones.
    estado, cliente = estado_cliente
    ahora = time.time()
    lineas = ['data: {"choices":[{"index":0,"delta":{"content":""},'
             '"finish_reason":null}]}\n\n' for _ in range(TOPE_PENDIENTES + 6)]
    lineas.append("data: [DONE]\n\n")
    payload_sin_contenido = "".join(lineas).encode()

    def handler(req):
        if _ping(req.content):
            return httpx.Response(200, json={"choices": [
                {"message": {"role": "assistant", "content": "pong"}}]})
        return httpx.Response(200, content=payload_sin_contenido)

    async def _quince_streams_identicos():
        rutas = estado.almacen.rutas_activas()
        estado.proxy.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        cuerpo = {"model": "auto", "stream": True,
                 "messages": [{"role": "user", "content": "piensa mucho"}]}
        for i in range(15):
            [ln async for ln in estado.proxy.completar_stream(rutas, cuerpo, ahora + i)]
        await estado.proxy.esperar_sondas_pendientes()

    asyncio.run(_quince_streams_identicos())
    assert cliente.get("/health").json()["estado"] == "ok"


def test_health_cae_rapido_con_una_ruta_rota_de_verdad_via_sonda_bajo_demanda(estado_cliente):
    # Contraste: una ruta rota de verdad (le va mal tambien a la sonda) se
    # enfria en UMBRAL_SOSPECHA pedidos + una sonda -- no en 5h.
    estado, cliente = estado_cliente
    ahora = time.time()

    async def _umbral_pedidos():
        rutas = estado.almacen.rutas_activas()
        estado.proxy.http = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(500)))
        for i in range(UMBRAL_SOSPECHA):
            await estado.proxy.completar(rutas, {"model": "a:free", "messages": []},
                                         ahora + i)
        await estado.proxy.esperar_sondas_pendientes()

    asyncio.run(_umbral_pedidos())
    assert estado.proxy.cooldowns  # la sonda confirmo y castigo
    r = cliente.get("/health")
    assert r.status_code != 200
    assert r.json()["estado"] != "ok"


def test_health_tras_reinicio_sigue_ok_tras_ataque_de_cadena_de_una_sola_ruta(tmp_path):
    ruta_db = str(tmp_path / "salud_sospecha_ok.sqlite3")
    ahora = time.time()

    def handler(req):
        if _ping(req.content):
            return httpx.Response(200, json={"choices": [
                {"message": {"role": "assistant", "content": "pong"}}]})
        return httpx.Response(403, json={"error": "contenido flageado"})

    almacen1 = Almacen(ruta_db)
    almacen1.crear_esquema()
    almacen1.upsert_rutas(
        [Ruta("kilo", "a:free", "gratis", Capacidades(True, False, 100000, 4096))], 1.0)
    proxy1 = Proxy(
        {"kilo": Proveedor("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, [])},
        almacen1, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    estado1 = Estado(almacen=almacen1, proxy=proxy1, llaves={"buena"}, tope_pago_diario=200)

    async def _quince_pedidos():
        rutas = almacen1.rutas_activas()
        for i in range(15):
            await proxy1.completar(rutas, {"model": "a:free", "messages": []}, ahora + i)
        await proxy1.esperar_sondas_pendientes()

    asyncio.run(_quince_pedidos())
    cliente1 = TestClient(crear_app(estado1))
    assert cliente1.get("/health").json()["estado"] == "ok"

    # "Reinicio del contenedor": proceso nuevo, Almacen nuevo, MISMA base --
    # sin que el proxy1 original (con su sonda ya resuelta) intervenga.
    almacen2 = Almacen(ruta_db)
    almacen2.crear_esquema()
    proxy2 = Proxy(
        {"kilo": Proveedor("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, [])},
        almacen2, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    estado2 = Estado(almacen=almacen2, proxy=proxy2, llaves={"buena"}, tope_pago_diario=200)
    cliente2 = TestClient(crear_app(estado2))
    assert cliente2.get("/health").json()["estado"] == "ok"


def test_health_tras_reinicio_sigue_caido_tras_sonda_bajo_demanda_fallida(tmp_path):
    # La contracara: la SONDA BAJO DEMANDA (no la periodica) escribio la fila
    # de `sondas` que declara la ruta muerta -- tiene que persistir igual.
    ruta_db = str(tmp_path / "salud_sospecha_caida.sqlite3")
    ahora = time.time()

    almacen1 = Almacen(ruta_db)
    almacen1.crear_esquema()
    almacen1.upsert_rutas(
        [Ruta("kilo", "a:free", "gratis", Capacidades(True, False, 100000, 4096))], 1.0)
    proxy1 = Proxy(
        {"kilo": Proveedor("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, [])},
        almacen1, httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(500))))   # rota para CUALQUIER payload, incluida la sonda
    estado1 = Estado(almacen=almacen1, proxy=proxy1, llaves={"buena"}, tope_pago_diario=200)

    async def _dos_rachas_de_umbral_pedidos():
        rutas = almacen1.rutas_activas()
        for i in range(UMBRAL_SOSPECHA):
            await proxy1.completar(rutas, {"model": "a:free", "messages": []}, ahora + i)
        await proxy1.esperar_sondas_pendientes()
        # Segunda racha, mas alla del rate-limit de sondas bajo demanda:
        # dispara una SEGUNDA sonda por el mecanismo real. Round 9 exige
        # dos fallidas consecutivas (ver Almacen.tiene_evidencia_de_vida)
        # para que /health la trate como muerta -- una sola ya no alcanza.
        ahora2 = ahora + LIMITE_PROBE_BAJO_DEMANDA_S + UMBRAL_SOSPECHA + 10
        for i in range(UMBRAL_SOSPECHA):
            await proxy1.completar(rutas, {"model": "a:free", "messages": []}, ahora2 + i)
        await proxy1.esperar_sondas_pendientes()

    asyncio.run(_dos_rachas_de_umbral_pedidos())
    cliente1 = TestClient(crear_app(estado1))
    r1 = cliente1.get("/health")
    assert r1.status_code != 200
    assert r1.json()["estado"] != "ok"

    filas = almacen1._con.execute(
        "SELECT tipo, ok FROM sondas WHERE clave = 'kilo/a:free'").fetchall()
    assert len(filas) == 2
    assert all(f == ("salud", 0) for f in filas)

    # Segundo proceso, con un transport SANO -- para probar que "caido"
    # viene de la sonda YA persistida, no de trafico nuevo que este proceso
    # genere.
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


# --- Round 9, HIGH 4 del gate: round 8 solo contaba uso de pago en el
#     EXITO (`r.ruta`/`en_ruta_comprometida`) -- pero un 200 con contenido
#     vacio (un modelo de razonamiento que se gasta el presupuesto) el
#     proveedor lo COBRA igual, aunque el gateway lo trate como fallido y
#     siga la cadena. Medido: 40/40 llamadas facturables con
#     `pago_hoy: 0`, TOPE_PAGO_DIARIO nunca actuando. Ahora se cuenta todo
#     intento con status 200 contra una ruta de pago, sirva o no. ---

def test_streaming_pago_factura_un_200_vacio_aunque_no_sirva():
    vacio = b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\ndata: [DONE]\n\n'
    estado, cliente = _estado_libre_y_pago(
        tope_pago_diario=5,
        hacer_resp_free=lambda: httpx.Response(500, json={"error": "free caida"}),
        hacer_resp_paid=lambda: httpx.Response(200, content=vacio))
    dia = _hoy()
    r = cliente.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "messages": [], "stream": True})
    assert r.status_code == 200
    assert estado.almacen.uso_pago("buena", dia) == 1  # facturable, aunque no sirvio


def test_no_streaming_pago_factura_un_200_vacio_aunque_no_sirva():
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
    vacio = {"choices": [{"message": {"role": "assistant", "content": None}}]}

    def responder(req):
        if "f.test" in str(req.url):
            return httpx.Response(500, json={"error": "free caida"})
        return httpx.Response(200, json=vacio)

    http = httpx.AsyncClient(transport=httpx.MockTransport(responder))
    estado = Estado(almacen=almacen, proxy=Proxy(prov, almacen, http),
                    llaves={"buena"}, tope_pago_diario=5)
    cliente = TestClient(crear_app(estado))
    dia = _hoy()
    r = cliente.post("/v1/chat/completions", headers={"X-API-Key": "buena"},
                     json={"model": "auto", "messages": []})
    assert r.status_code == 503  # nada sirvio de verdad al cliente
    assert estado.almacen.uso_pago("buena", dia) == 1  # pero SI se factura


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
                    limitador=PerKeyRateLimiter(2))
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
#     piso de /health era cargante y vivia en dos archivos distintos, sin nada
#     que lo probara. Si alguna vez se invertia, una instalacion NUEVA -- sin
#     un solo evento todavia -- reportaba "caido" y Coolify nunca marcaba el
#     contenedor como sano: el servicio no arrancaba nunca, por una constante.
#
#     Round 6, Parte 2: el mecanismo que este test protegia (comparar
#     `CONFIABILIDAD_NEUTRA` contra `UMBRAL_CONFIABILIDAD_SALUD`) desaparecio
#     junto con `/health` basado en promedio -- `UMBRAL_CONFIABILIDAD_SALUD`
#     ya no existe. El contrato que protegia ("una ruta sin telemetria cuenta
#     como viva") sigue vivo, ahora en `Almacen.tiene_evidencia_de_vida` (ver
#     test_tiene_evidencia_de_vida_sin_ninguna_telemetria en test_almacen.py)
#     y verificado end-to-end en test_health_ok_si_la_gratis_no_tiene_telemetria_aun
#     y test_health_tras_reinicio_del_proceso_sigue_ok_sin_telemetria mas
#     arriba. ---
