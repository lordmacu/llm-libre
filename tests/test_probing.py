import json
import logging

import httpx

from llm_libre.storage import Storage
from llm_libre.api import Estado
from llm_libre.models import Capabilities, Route
from llm_libre.providers import Provider
from llm_libre.proxy import Proxy
from llm_libre.quality_suite import SHORT_TOKEN_BUDGET
from llm_libre.probing import (PING, cycle, probe_health, probe_quality,
                               sync_catalogue)

CATALOGO = {"data": [
    {"id": "x:free", "pricing": {"prompt": "0"}, "context_length": 1000,
     "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
     "supported_parameters": ["tools"], "top_provider": {"max_completion_tokens": 100}}]}


def _almacen():
    a = Storage(":memory:")
    a.create_schema()
    return a


def _ruta(modelo="x:free", tier="gratis", provider="kilo", tools=True):
    return Route(provider, modelo, tier, Capabilities(tools, False, 1000, 100))


def _proxy(handler):
    prov = {"kilo": Provider("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, [])}
    return Proxy(prov, _almacen(), httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def test_sincronizar_guarda_las_rutas_descubiertas():
    almacen = _almacen()
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=CATALOGO)))
    prov = [Provider("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, [])]
    await sync_catalogue(http, prov, almacen, now=100.0)
    assert [r.key for r in almacen.active_routes()] == ["kilo/x:free"]


async def test_sincronizar_no_astilla_un_query_string_en_base_url():
    # Round 5: el mismo bug del sufijo concatenado, del lado de /models --
    # el test tiene que mirar la URL que de verdad se pidio, no un campo
    # intermedio.
    almacen = _almacen()
    vistas = []

    def handler(req):
        vistas.append(str(req.url))
        return httpx.Response(200, json=CATALOGO)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prov = [Provider("chatgpt", "gratis", "openai", "https://blog.test:8888?token=abc",
                      "", "/models", {}, [])]
    await sync_catalogue(http, prov, almacen, now=100.0)
    assert vistas == ["https://blog.test:8888/models?token=abc"]


async def test_sincronizar_agrega_los_modelos_fijos_de_pago():
    almacen = _almacen()
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=CATALOGO)))
    prov = [Provider("minimax", "pago", "openai", "https://m.test", "k", "",
                      {}, [{"id": "MiniMax-M3", "tools": True, "vision": False,
                            "contexto": 128000, "max_salida": 32768}])]
    await sync_catalogue(http, prov, almacen, now=100.0)
    rutas = almacen.active_routes()
    assert [r.key for r in rutas] == ["minimax/MiniMax-M3"]
    assert rutas[0].tier == "pago"


async def test_sincronizar_propaga_la_prioridad_del_proveedor_a_las_rutas_descubiertas():
    almacen = _almacen()
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=CATALOGO)))
    prov = [Provider("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, [],
                      priority=1)]
    await sync_catalogue(http, prov, almacen, now=100.0)
    rutas = almacen.active_routes()
    assert rutas[0].priority == 1


async def test_sincronizar_propaga_las_capacidades_por_defecto_del_proveedor():
    # Simula chatgpt-proxy: /models trae ids pero NINGUN metadato de
    # capacidad -- sync_catalogue tiene que aplicar las capacidades
    # declaradas del proveedor a cada id descubierto, tools:false incluido.
    almacen = _almacen()
    catalogo_desnudo = {"data": [
        {"id": "gpt-5-3-mini", "object": "model", "description": "GPT-5.3 Mini"},
    ]}
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=catalogo_desnudo)))
    prov = [Provider("chatgpt", "gratis", "openai", "https://cg.test", "", "/models", {}, [],
                      priority=0,
                      default_capabilities=Capabilities(False, False, 128000, 8192))]
    await sync_catalogue(http, prov, almacen, now=100.0)
    rutas = almacen.active_routes()
    assert [r.key for r in rutas] == ["chatgpt/gpt-5-3-mini"]
    assert rutas[0].capabilities.tools is False
    assert rutas[0].capabilities.context == 128000
    assert rutas[0].priority == 0


async def test_un_proveedor_caido_no_borra_el_catalogo_de_los_demas():
    almacen = _almacen()
    almacen.upsert_routes([_ruta("previa:free")], timestamp=50.0)

    def handler(req):
        return httpx.Response(500)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prov = [Provider("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, [])]
    await sync_catalogue(http, prov, almacen, now=100.0)
    # Si /models falla no se desactiva nada: mejor catalogo viejo que catalogo vacio.
    assert len(almacen.active_routes()) == 1


async def test_un_fallo_parcial_no_corrompe_el_visto_por_ultima_vez_de_lo_que_si_se_descubrio():
    # Deviation del brief: `upsert_rutas(descubiertas, ahora if not fallo else 0.0)`
    # pisaba visto_por_ultima_vez con 0.0 tambien en las rutas de los proveedores
    # que SI respondieron, no solo en las que faltaron. El tercer parametro de
    # upsert_rutas (desactivar_faltantes) existe justo para separar "no apagues
    # lo que no llego" de "corrompe el momento de lo que si llego".
    almacen = _almacen()

    def handler(req):
        if "k.test" in str(req.url):
            return httpx.Response(200, json=CATALOGO)
        return httpx.Response(500)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prov = [
        Provider("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, []),
        Provider("roto", "gratis", "openai", "https://roto.test", "", "/models", {}, []),
    ]
    await sync_catalogue(http, prov, almacen, now=100.0)
    fila = almacen._con.execute(
        "SELECT visto_por_ultima_vez FROM rutas WHERE clave='kilo/x:free'").fetchone()
    assert fila[0] == 100.0


CATALOGO_VACIO = {"data": []}

# Un 200 real, pero cuyo unico modelo es de pago: normalize() lo filtra y el
# resultado utilizable tambien queda en cero, aunque la respuesta "tenga data".
CATALOGO_TODO_FILTRADO = {"data": [
    {"id": "pago:model", "pricing": {"prompt": "0.002"}, "context_length": 1000,
     "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
     "supported_parameters": [], "top_provider": {"max_completion_tokens": 100}}]}


async def test_un_200_con_data_vacia_no_borra_las_rutas_previas():
    # Finding 1: un 200 con cero modelos utilizables es mas probablemente un
    # proveedor roto que un catalogo genuinamente vacio -- no debe autorizar
    # el UPDATE que desactiva todo lo que ya se conocia de el.
    almacen = _almacen()
    almacen.upsert_routes([_ruta("previa:free")], timestamp=50.0)
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=CATALOGO_VACIO)))
    prov = [Provider("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, [])]
    await sync_catalogue(http, prov, almacen, now=100.0)
    assert [r.key for r in almacen.active_routes()] == ["kilo/previa:free"]


async def test_un_200_cuyos_modelos_quedan_todos_filtrados_no_borra_las_rutas_previas():
    almacen = _almacen()
    almacen.upsert_routes([_ruta("previa:free")], timestamp=50.0)
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=CATALOGO_TODO_FILTRADO)))
    prov = [Provider("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, [])]
    await sync_catalogue(http, prov, almacen, now=100.0)
    assert [r.key for r in almacen.active_routes()] == ["kilo/previa:free"]


async def test_un_proveedor_vacio_no_frena_la_actualizacion_del_que_si_respondio():
    almacen = _almacen()
    almacen.upsert_routes([_ruta("previa:free", provider="vacio")], timestamp=50.0)

    def handler(req):
        if "k.test" in str(req.url):
            return httpx.Response(200, json=CATALOGO)
        return httpx.Response(200, json=CATALOGO_VACIO)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prov = [
        Provider("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, []),
        Provider("vacio", "gratis", "openai", "https://vacio.test", "", "/models", {}, []),
    ]
    await sync_catalogue(http, prov, almacen, now=100.0)
    claves = {r.key for r in almacen.active_routes()}
    assert "kilo/x:free" in claves          # el proveedor que si respondio se actualizo
    assert "vacio/previa:free" in claves    # el vacio conservo lo que ya tenia


async def test_un_proveedor_sano_desactiva_su_propia_ruta_vieja_aunque_otro_este_vacio():
    # Finding 1, caso (c) -- el que distingue un fix real de uno a lo bruto.
    # kilo tenia una ruta vieja que ya no aparece en su catalogo nuevo; vacio
    # responde 200 sin modelos. Que vacio este vacio no debe frenar la baja de
    # kilo/old:free: kilo respondio bien y su propia desaparicion es real.
    almacen = _almacen()
    almacen.upsert_routes([_ruta("old:free", provider="kilo")], timestamp=50.0)
    almacen.upsert_routes([_ruta("previa:free", provider="vacio")], timestamp=50.0)

    def handler(req):
        if "k.test" in str(req.url):
            return httpx.Response(200, json=CATALOGO)   # ya no trae old:free, solo x:free
        return httpx.Response(200, json=CATALOGO_VACIO)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prov = [
        Provider("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, []),
        Provider("vacio", "gratis", "openai", "https://vacio.test", "", "/models", {}, []),
    ]
    await sync_catalogue(http, prov, almacen, now=100.0)
    claves = {r.key for r in almacen.active_routes()}
    assert claves == {"kilo/x:free", "vacio/previa:free"}
    assert "kilo/old:free" not in claves    # se desactivo: kilo respondio, y ya no la trae


async def test_un_proveedor_sano_no_desactiva_rutas_de_otro_proveedor():
    # La desactivacion de kilo debe quedar ACOTADA a kilo: una ruta vieja de
    # "otro" (mucho mas vieja que `ahora`) no debe caer victima del UPDATE
    # de kilo. "otro" SIGUE registrado (sin modelos_path ni modelos_fijos,
    # asi que el loop por-proveedor lo saltea sin tocarlo) -- si no
    # estuviera en `prov`, el barrido de proveedores huerfanos
    # (Storage.desactivar_proveedores_no_registrados, ver el bloque de
    # tests aparte mas abajo) la apagaria por una razon DISTINTA, y este
    # test dejaria de aislar lo que dice probar.
    almacen = _almacen()
    almacen.upsert_routes([_ruta("vieja:free", provider="otro")], timestamp=10.0)
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=CATALOGO)))
    prov = [
        Provider("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, []),
        Provider("otro", "gratis", "openai", "https://otro.test", "", "", {}, []),
    ]
    await sync_catalogue(http, prov, almacen, now=100.0)
    claves = {r.key for r in almacen.active_routes()}
    assert claves == {"kilo/x:free", "otro/vieja:free"}


CATALOGO_COMPARTIDO = {"data": [
    {"id": "shared:free", "pricing": {"prompt": "0"}, "context_length": 1000,
     "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
     "supported_parameters": [], "top_provider": {"max_completion_tokens": 100}}]}


async def test_el_mismo_modelo_en_dos_proveedores_el_que_lo_pierde_se_apaga_el_que_lo_tiene_sigue():
    # Razon de ser de "rutas, no modelos": el mismo modelo_id puede existir en
    # dos proveedores como dos filas independientes (claves distintas). Si uno
    # lo deja de ofrecer y el otro lo mantiene, deben tratarse por separado.
    almacen = _almacen()
    almacen.upsert_routes([_ruta("shared:free", provider="kilo")], timestamp=50.0)
    almacen.upsert_routes([_ruta("shared:free", provider="otro")], timestamp=50.0)

    def handler(req):
        if "k.test" in str(req.url):
            return httpx.Response(200, json=CATALOGO)          # kilo ya no trae shared:free
        return httpx.Response(200, json=CATALOGO_COMPARTIDO)   # otro lo sigue trayendo

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prov = [
        Provider("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, []),
        Provider("otro", "gratis", "openai", "https://o.test", "", "/models", {}, []),
    ]
    await sync_catalogue(http, prov, almacen, now=100.0)
    claves = {r.key for r in almacen.active_routes()}
    assert "kilo/shared:free" not in claves   # kilo lo perdio: se apaga
    assert "otro/shared:free" in claves       # otro lo conserva: sigue activa
    assert "kilo/x:free" in claves


async def test_un_200_con_cuerpo_no_json_se_trata_como_fallo_y_no_frena_a_los_demas():
    # Finding 2: r.json() puede tirar JSONDecodeError (subclase de ValueError)
    # con un cuerpo no-JSON; eso no debe escapar de sync_catalogue ni
    # frenar el procesamiento de los proveedores que vienen despues.
    almacen = _almacen()

    def handler(req):
        if "roto.test" in str(req.url):
            return httpx.Response(200, text="esto no es json")
        return httpx.Response(200, json=CATALOGO)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prov = [
        Provider("roto", "gratis", "openai", "https://roto.test", "", "/models", {}, []),
        Provider("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, []),
    ]
    await sync_catalogue(http, prov, almacen, now=100.0)
    assert [r.key for r in almacen.active_routes()] == ["kilo/x:free"]


async def test_un_200_con_forma_inesperada_se_trata_como_fallo_y_no_frena_a_los_demas():
    # JSON valido pero de una forma que normalize() no espera (p.ej. un error
    # de autenticacion servido con status 200): dispara AttributeError dentro
    # de normalize(), no un httpx.HTTPError.
    almacen = _almacen()

    def handler(req):
        if "roto.test" in str(req.url):
            return httpx.Response(200, json={"error": "unauthorized"})
        return httpx.Response(200, json=CATALOGO)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prov = [
        Provider("roto", "gratis", "openai", "https://roto.test", "", "/models", {}, []),
        Provider("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, []),
    ]
    await sync_catalogue(http, prov, almacen, now=100.0)
    assert [r.key for r in almacen.active_routes()] == ["kilo/x:free"]


# --- Sacar un proveedor de proveedores.yaml (caso real: openrouter, sin
#     OPENROUTER_API_KEY, cuyas 16 rutas 401-eaban siempre y solo ocupaban
#     cupo de sonda y espacio en el ranking probando estar muertas) no debe
#     dejar sus rutas `activa=1` para siempre -- ver
#     Storage.desactivar_proveedores_no_registrados. Esto es la mitad
#     "a traves de sync_catalogue" de esa cobertura; la otra mitad
#     (el metodo de Almacen aislado) esta en test_almacen.py. ---

async def test_sincronizar_desactiva_las_rutas_de_un_proveedor_que_se_saco_del_registro():
    almacen = _almacen()
    almacen.upsert_routes([_ruta("previa:free", provider="openrouter")], timestamp=50.0)
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=CATALOGO)))
    # openrouter YA NO esta en la lista que carga proveedores.load() --
    # simula haberlo sacado de proveedores.yaml.
    prov = [Provider("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, [])]
    await sync_catalogue(http, prov, almacen, now=100.0)
    claves = {r.key for r in almacen.active_routes()}
    assert claves == {"kilo/x:free"}
    fila = almacen._con.execute(
        "SELECT activa FROM rutas WHERE clave = 'openrouter/previa:free'").fetchone()
    assert fila == (0,)   # apagada, no borrada


async def test_sincronizar_no_desactiva_rutas_de_proveedores_que_siguen_registrados():
    # Aislado a proposito de la logica NORMAL de baja por-proveedor (que ya
    # cubren los tests de arriba, "el que si respondio actualiza lo suyo"):
    # minimax usa modelos_fijos, asi que su ruta se re-declara IDENTICA en
    # cada pasada sin importar ningun mock HTTP -- si sobrevive, es prueba
    # de que el barrido de huerfanos (`desactivar_proveedores_no_registrados`)
    # no la toco, no de que "justo volvio a aparecer en el catalogo".
    almacen = _almacen()
    almacen.upsert_routes([_ruta("previa:free", provider="openrouter")], timestamp=50.0)
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=CATALOGO)))
    prov = [
        Provider("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, []),
        Provider("minimax", "pago", "openai", "https://m.test", "k", "", {},
                  [{"id": "MiniMax-M3", "tools": True, "vision": False,
                    "contexto": 128000, "max_salida": 32768}]),
    ]
    await sync_catalogue(http, prov, almacen, now=100.0)
    claves = {r.key for r in almacen.active_routes()}
    # kilo se descubrio de nuevo, minimax (modelos_fijos, SIGUE registrado)
    # sigue activa, y openrouter (sacado del registro) desaparecio -- las
    # tres cosas a la vez, sin que ninguna se confunda con la otra.
    assert claves == {"kilo/x:free", "minimax/MiniMax-M3"}


async def test_sincronizar_con_registro_vacio_no_apaga_el_catalogo_entero():
    # Revision de gate: `proveedores=[]` (un `proveedores.yaml` sintacticamente
    # valido pero truncado/mal editado, mas probable que "sin proveedores" a
    # proposito) no debe disparar el barrido de huerfanos -- ver el guard en
    # Storage.desactivar_proveedores_no_registrados.
    almacen = _almacen()
    almacen.upsert_routes([_ruta("previa:free", provider="kilo")], timestamp=50.0)
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=CATALOGO)))
    await sync_catalogue(http, [], almacen, now=100.0)
    assert {r.key for r in almacen.active_routes()} == {"kilo/previa:free"}


async def test_sync_catalogue_no_deja_escapar_la_excepcion_de_un_proveedor_roto():
    # Dos formas distintas de "proveedor roto" (cuerpo no-JSON y JSON con forma
    # inesperada) conviviendo con uno sano: si la excepcion escapara, este
    # await ni siquiera terminaria y la prueba fallaria por un error de
    # coleccion, no por un assert.
    almacen = _almacen()

    def handler(req):
        url = str(req.url)
        if "nojson.test" in url:
            return httpx.Response(200, text="<html>not json</html>")
        if "raro.test" in url:
            return httpx.Response(200, json={"error": "unauthorized"})
        return httpx.Response(200, json=CATALOGO)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prov = [
        Provider("nojson", "gratis", "openai", "https://nojson.test", "", "/models", {}, []),
        Provider("raro", "gratis", "openai", "https://raro.test", "", "/models", {}, []),
        Provider("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, []),
    ]
    total = await sync_catalogue(http, prov, almacen, now=100.0)
    assert total == 1
    assert [r.key for r in almacen.active_routes()] == ["kilo/x:free"]


async def test_la_sonda_de_salud_registra_exito():
    p = _proxy(lambda req: httpx.Response(200, json={
        "choices": [{"message": {"content": "ok"}}]}))
    await probe_health(p, p.almacen, [_ruta()], now=100.0)
    fila = p.almacen._con.execute(
        "SELECT tipo, ok FROM sondas WHERE clave='kilo/x:free'").fetchone()
    assert fila == ("salud", 1)


async def test_la_sonda_de_salud_registra_el_fallo_de_un_modelo_que_ya_no_existe():
    p = _proxy(lambda req: httpx.Response(404, json={"error": "model_not_found"}))
    await probe_health(p, p.almacen, [_ruta()], now=100.0)
    fila = p.almacen._con.execute(
        "SELECT ok FROM sondas WHERE tipo='salud'").fetchone()
    assert fila[0] == 0


# --- Round 9, HIGH 1 del gate: probe_health llamaba proxy.completar SIN
#     es_sonda=True -- round 8 gateo el castigo directo detras de esa
#     bandera, y este llamador nunca se actualizo. Una sonda periodica
#     corre UNA vez cada 5h por ruta; sin es_sonda=True, un fallo solo
#     acumulaba sospecha (que necesita 3 consecutivos), asi que 20 sondas
#     periodicas contra una ruta muerta (100h) dejaban 20 filas
#     `sondas ok=0` y CERO cooldowns -- la sonda perdio su autoridad para
#     excluir. Con es_sonda=True, UNA sola sonda fallida ya castiga. ---

async def test_la_sonda_de_salud_castiga_de_inmediato_con_un_solo_fallo():
    p = _proxy(lambda req: httpx.Response(500))
    await probe_health(p, p.almacen, [_ruta()], now=100.0)
    assert p.cooldowns["kilo/x:free"] > 0.0
    # Y sin pasar por sospecha -- no hay ninguna marca acumulada esperando
    # un segundo o tercer fallo.
    assert p.almacen._con.execute(
        "SELECT COUNT(*) FROM sondas WHERE tipo='salud' AND ok=0").fetchone()[0] == 1


async def test_la_sonda_de_calidad_guarda_casos_pasados_sobre_totales():
    p = _proxy(lambda req: httpx.Response(200, json={
        "choices": [{"message": {"content": "12"}}]}))
    await probe_quality(p, p.almacen, [_ruta()], now=100.0)
    fila = p.almacen._con.execute(
        "SELECT casos_pasados, casos_totales FROM sondas WHERE tipo='calidad'").fetchone()
    assert fila[1] == 5
    assert 1 <= fila[0] < 5   # pasa aritmetica y formato, falla json y tools


# --- Round 10, fix chico del gate: mismo hueco de wiring que HIGH 1
#     (round 9), una funcion mas alla -- `probe_quality` llamaba
#     completar() sin es_sonda=True, asi que un caso de bateria fallido
#     alimentaba `_sospechar` (pensada para trafico de CLIENTE) y gastaba
#     cupo del presupuesto de sondas bajo demanda, escaso y compartido con
#     el trafico real. `caso.body` es tan gateway-autor como PING. ---

async def test_la_sonda_de_calidad_castiga_directo_sin_pasar_por_sospecha():
    p = _proxy(lambda req: httpx.Response(500))
    await probe_quality(p, p.almacen, [_ruta()], now=100.0)
    assert p.cooldowns["kilo/x:free"] > 0.0
    # Directo -- nunca via sospecha (que necesitaria UMBRAL_SOSPECHA
    # fallos, y ademas dispararia una sonda de tipo 'salud' aparte).
    assert p.almacen._con.execute(
        "SELECT COUNT(*) FROM sondas WHERE tipo='salud'").fetchone()[0] == 0


async def test_la_calidad_no_sondea_rutas_de_pago():
    p = _proxy(lambda req: httpx.Response(200, json={
        "choices": [{"message": {"content": "12"}}]}))
    await probe_quality(p, p.almacen, [_ruta(tier="pago")], now=100.0)
    assert p.almacen._con.execute(
        "SELECT COUNT(*) FROM sondas WHERE tipo='calidad'").fetchone()[0] == 0


async def test_la_calidad_omite_el_caso_de_tools_sin_contarlo_como_fallo():
    # Una ruta que no declara soporte de tools no debe verse igual de mal en el
    # puntaje que una que lo declara y lo hace mal: el caso se omite entero
    # (no cuenta ni para pasados ni para totales), no se marca como fallido.
    llamadas = []

    def handler(req):
        llamadas.append(req)
        return httpx.Response(200, json={"choices": [{"message": {"content": "12"}}]})

    p = _proxy(handler)
    await probe_quality(p, p.almacen, [_ruta(tools=False)], now=100.0)
    fila = p.almacen._con.execute(
        "SELECT casos_pasados, casos_totales FROM sondas WHERE tipo='calidad'").fetchone()
    assert fila[1] == 4          # 5 casos menos el de tools, que se omitio
    assert len(llamadas) == 4    # y no se le gasto cuota pidiendoselo


async def test_ciclo_sincroniza_sondea_salud_y_sondea_calidad_en_el_ciclo_cero():
    almacen = _almacen()
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=CATALOGO)))
    prov = [Provider("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, [])]
    proxy = Proxy({"kilo": prov[0]}, almacen, http)
    estado = Estado(almacen=almacen, proxy=proxy, llaves=set(), tope_pago_diario=0,
                    proveedores=prov, http=http)
    await cycle(estado, counter=0)
    assert [r.key for r in almacen.active_routes()] == ["kilo/x:free"]
    tipos = {t for (t,) in almacen._con.execute("SELECT tipo FROM sondas").fetchall()}
    assert tipos == {"salud", "calidad"}


async def test_ciclo_no_sondea_calidad_fuera_del_intervalo():
    almacen = _almacen()
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=CATALOGO)))
    prov = [Provider("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, [])]
    proxy = Proxy({"kilo": prov[0]}, almacen, http)
    estado = Estado(almacen=almacen, proxy=proxy, llaves=set(), tope_pago_diario=0,
                    proveedores=prov, http=http)
    await cycle(estado, counter=1)
    tipos = {t for (t,) in almacen._con.execute("SELECT tipo FROM sondas").fetchall()}
    assert tipos == {"salud"}


# --- Fix round 3, B4 (Blocking): §8 dice "las rutas de pago NO se sondean".
#     probe_quality ya filtraba por tier; probe_health no, y recibe
#     rutas_activas(), que incluye minimax/MiniMax-M3. Eran ~5 llamadas
#     facturables por dia, invisibles para sumar_uso_pago, /v1/uso y
#     TOPE_PAGO_DIARIO. ---

async def test_la_sonda_de_salud_no_gasta_plata_en_las_rutas_de_pago():
    llamadas = []

    def handler(req):
        llamadas.append(req)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    p = _proxy(handler)
    await probe_health(p, p.almacen, [_ruta(tier="pago")], now=100.0)
    assert llamadas == []
    assert p.almacen._con.execute("SELECT COUNT(*) FROM sondas").fetchone()[0] == 0


async def test_la_sonda_de_salud_sigue_sondeando_las_gratis_de_la_misma_lista():
    llamadas = []

    def handler(req):
        llamadas.append(req)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    p = _proxy(handler)
    await probe_health(p, p.almacen, [_ruta("g:free"), _ruta("P", tier="pago")],
                        now=100.0)
    assert len(llamadas) == 1
    claves = [c for (c,) in p.almacen._con.execute("SELECT clave FROM sondas")]
    assert claves == ["kilo/g:free"]


async def test_el_ciclo_completo_no_sondea_la_ruta_de_pago():
    # El caso real: `cycle` le pasa active_routes() a probe_health, y ahi
    # adentro viene minimax/MiniMax-M3 desde los modelos_fijos del YAML.
    almacen = _almacen()
    llamadas = []

    def handler(req):
        llamadas.append(str(req.url))
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json=CATALOGO)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prov = [
        Provider("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, []),
        Provider("minimax", "pago", "openai", "https://m.test", "k", "", {},
                  [{"id": "MiniMax-M3", "tools": True, "vision": False,
                    "contexto": 128000, "max_salida": 32768}]),
    ]
    proxy = Proxy({p.id: p for p in prov}, almacen, http)
    estado = Estado(almacen=almacen, proxy=proxy, llaves=set(), tope_pago_diario=0,
                    proveedores=prov, http=http)
    await cycle(estado, counter=1)   # contador 1: salud si, calidad no
    assert {r.key for r in almacen.active_routes()} == {"kilo/x:free", "minimax/MiniMax-M3"}
    assert not any("m.test" in u for u in llamadas), "se le pego a la ruta de pago"


# --- Fix round 3, I4: `sync_catalogue` fallaba en silencio absoluto --
#     cuatro `continue` (error de red, status != 200, cuerpo mal formado,
#     catalogo vacio) y ni una linea de log en todo el modulo. Si un proveedor
#     empieza a fallar, su catalogo se congela para siempre y nada lo dice.
#     Esta es justo la capa que existe para evitar catalogos rancios. ---

def _prov_kilo():
    return [Provider("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, [])]


async def _sincronizar_con(handler, caplog):
    almacen = _almacen()
    almacen.upsert_routes([_ruta("previa:free")], timestamp=50.0)
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with caplog.at_level(logging.WARNING, logger="llm_libre.probing"):
        await sync_catalogue(http, _prov_kilo(), almacen, now=100.0)
    return caplog.text


async def test_un_error_de_red_se_loguea_con_el_proveedor_y_la_razon(caplog):
    def handler(req):
        raise httpx.ConnectError("no hay ruta al host")

    texto = await _sincronizar_con(handler, caplog)
    assert "kilo" in texto and "no hay ruta al host" in texto


async def test_un_status_distinto_de_200_se_loguea(caplog):
    texto = await _sincronizar_con(lambda req: httpx.Response(503), caplog)
    assert "kilo" in texto and "503" in texto


async def test_un_cuerpo_mal_formado_se_loguea(caplog):
    texto = await _sincronizar_con(
        lambda req: httpx.Response(200, text="<html>mantenimiento</html>"), caplog)
    assert "kilo" in texto
    assert "could not interpret" in texto


async def test_un_catalogo_vacio_se_loguea(caplog):
    texto = await _sincronizar_con(
        lambda req: httpx.Response(200, json=CATALOGO_VACIO), caplog)
    assert "kilo" in texto
    assert "zero usable models" in texto


async def test_una_sincronizacion_sana_no_ensucia_los_logs_de_warning(caplog):
    almacen = _almacen()
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=CATALOGO)))
    with caplog.at_level(logging.WARNING, logger="llm_libre.probing"):
        await sync_catalogue(http, _prov_kilo(), almacen, now=100.0)
    assert caplog.text == ""


async def test_la_sonda_de_salud_no_escribe_un_ttft_que_no_midio():
    # I5: la sonda de salud es no-streaming, asi que su numero es un round-trip
    # completo, no un ttft. Va a latencia_ms; la columna de ttft queda en 0 y
    # el p50 de ttft la ignora.
    p = _proxy(lambda req: httpx.Response(200, json={
        "choices": [{"message": {"content": "ok"}}]}))
    await probe_health(p, p.almacen, [_ruta()], now=100.0)
    fila = p.almacen._con.execute(
        "SELECT latencia_ms, ttft_ms FROM sondas WHERE tipo='salud'").fetchone()
    assert fila[0] >= 0
    assert fila[1] == 0


# --- Fix round 4, N1 (Blocking): el PING de salud tenia max_tokens=8, cuatro
#     veces menos que los 32 que la bateria ya demostro insuficientes. Mientras
#     un 200 vacio contaba como exito era inofensivo; desde que es un intento
#     FALLIDO (fix B1), un ping que no deja pensar al modelo FABRICA el fallo
#     que dice medir. Medido contra Kilo: 5 de 11 rutas gratis "muertas" con
#     max_tokens=8, incluida la que sirve `auto` en el arranque en frio. ---

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


async def test_la_sonda_de_salud_no_mata_a_un_modelo_por_no_dejarlo_pensar():
    p = _proxy(_modelo_que_razona())
    await probe_health(p, p.almacen, [_ruta()], now=100.0)
    fila = p.almacen._con.execute(
        "SELECT ok FROM sondas WHERE tipo='salud'").fetchone()
    assert fila[0] == 1, ("el ping fabrico el fallo que dice medir: no le dio "
                          "presupuesto suficiente al modelo")


async def test_el_ping_de_salud_no_puede_quedarse_atras_del_tope_de_la_bateria():
    # Los dos topes existen por la misma razon; si la bateria sube y el ping no,
    # vuelve exactamente este bug.
    assert PING["max_tokens"] >= SHORT_TOKEN_BUDGET


async def test_la_sonda_de_salud_guarda_el_codigo_del_proveedor_no_el_del_gateway():
    # Un 200 que llega vacio SIGUE siendo un fallo (fix B1), pero en la tabla
    # tiene que quedar el 200 que dio el proveedor, no el 503 que sintetiza el
    # gateway: sin eso no hay forma de distinguir "el proveedor esta caido" de
    # "respondio bien pero sin nada adentro".
    p = _proxy(lambda req: httpx.Response(200, json={"choices": [
        {"message": {"role": "assistant", "content": None},
         "finish_reason": "length"}]}))
    await probe_health(p, p.almacen, [_ruta()], now=100.0)
    fila = p.almacen._con.execute(
        "SELECT ok, codigo_http FROM sondas WHERE tipo='salud'").fetchone()
    assert fila == (0, 200)


async def test_la_sonda_de_salud_guarda_el_404_real_de_un_modelo_que_ya_no_existe():
    p = _proxy(lambda req: httpx.Response(404, json={"error": "model_not_found"}))
    await probe_health(p, p.almacen, [_ruta()], now=100.0)
    fila = p.almacen._con.execute(
        "SELECT ok, codigo_http FROM sondas WHERE tipo='salud'").fetchone()
    assert fila == (0, 404)
