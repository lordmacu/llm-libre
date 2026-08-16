import httpx

from llm_libre.almacen import Almacen
from llm_libre.api import Estado
from llm_libre.modelos import Capacidades, Ruta
from llm_libre.proveedores import Proveedor
from llm_libre.proxy import Proxy
from llm_libre.sondeo import ciclo, sincronizar_catalogo, sondear_calidad, sondear_salud

CATALOGO = {"data": [
    {"id": "x:free", "pricing": {"prompt": "0"}, "context_length": 1000,
     "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
     "supported_parameters": ["tools"], "top_provider": {"max_completion_tokens": 100}}]}


def _almacen():
    a = Almacen(":memory:")
    a.crear_esquema()
    return a


def _ruta(modelo="x:free", tier="gratis", proveedor="kilo", tools=True):
    return Ruta(proveedor, modelo, tier, Capacidades(tools, False, 1000, 100))


def _proxy(handler):
    prov = {"kilo": Proveedor("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, [])}
    return Proxy(prov, _almacen(), httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def test_sincronizar_guarda_las_rutas_descubiertas():
    almacen = _almacen()
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=CATALOGO)))
    prov = [Proveedor("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, [])]
    await sincronizar_catalogo(http, prov, almacen, ahora=100.0)
    assert [r.clave for r in almacen.rutas_activas()] == ["kilo/x:free"]


async def test_sincronizar_agrega_los_modelos_fijos_de_pago():
    almacen = _almacen()
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=CATALOGO)))
    prov = [Proveedor("minimax", "pago", "openai", "https://m.test", "k", "",
                      {}, [{"id": "MiniMax-M3", "tools": True, "vision": False,
                            "contexto": 128000, "max_salida": 32768}])]
    await sincronizar_catalogo(http, prov, almacen, ahora=100.0)
    rutas = almacen.rutas_activas()
    assert [r.clave for r in rutas] == ["minimax/MiniMax-M3"]
    assert rutas[0].tier == "pago"


async def test_un_proveedor_caido_no_borra_el_catalogo_de_los_demas():
    almacen = _almacen()
    almacen.upsert_rutas([_ruta("previa:free")], momento=50.0)

    def handler(req):
        return httpx.Response(500)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    prov = [Proveedor("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, [])]
    await sincronizar_catalogo(http, prov, almacen, ahora=100.0)
    # Si /models falla no se desactiva nada: mejor catalogo viejo que catalogo vacio.
    assert len(almacen.rutas_activas()) == 1


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
        Proveedor("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, []),
        Proveedor("roto", "gratis", "openai", "https://roto.test", "", "/models", {}, []),
    ]
    await sincronizar_catalogo(http, prov, almacen, ahora=100.0)
    fila = almacen._con.execute(
        "SELECT visto_por_ultima_vez FROM rutas WHERE clave='kilo/x:free'").fetchone()
    assert fila[0] == 100.0


async def test_la_sonda_de_salud_registra_exito():
    p = _proxy(lambda req: httpx.Response(200, json={
        "choices": [{"message": {"content": "ok"}}]}))
    await sondear_salud(p, p.almacen, [_ruta()], ahora=100.0)
    fila = p.almacen._con.execute(
        "SELECT tipo, ok FROM sondas WHERE clave='kilo/x:free'").fetchone()
    assert fila == ("salud", 1)


async def test_la_sonda_de_salud_registra_el_fallo_de_un_modelo_que_ya_no_existe():
    p = _proxy(lambda req: httpx.Response(404, json={"error": "model_not_found"}))
    await sondear_salud(p, p.almacen, [_ruta()], ahora=100.0)
    fila = p.almacen._con.execute(
        "SELECT ok FROM sondas WHERE tipo='salud'").fetchone()
    assert fila[0] == 0


async def test_la_sonda_de_calidad_guarda_casos_pasados_sobre_totales():
    p = _proxy(lambda req: httpx.Response(200, json={
        "choices": [{"message": {"content": "12"}}]}))
    await sondear_calidad(p, p.almacen, [_ruta()], ahora=100.0)
    fila = p.almacen._con.execute(
        "SELECT casos_pasados, casos_totales FROM sondas WHERE tipo='calidad'").fetchone()
    assert fila[1] == 5
    assert 1 <= fila[0] < 5   # pasa aritmetica y formato, falla json y tools


async def test_la_calidad_no_sondea_rutas_de_pago():
    p = _proxy(lambda req: httpx.Response(200, json={
        "choices": [{"message": {"content": "12"}}]}))
    await sondear_calidad(p, p.almacen, [_ruta(tier="pago")], ahora=100.0)
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
    await sondear_calidad(p, p.almacen, [_ruta(tools=False)], ahora=100.0)
    fila = p.almacen._con.execute(
        "SELECT casos_pasados, casos_totales FROM sondas WHERE tipo='calidad'").fetchone()
    assert fila[1] == 4          # 5 casos menos el de tools, que se omitio
    assert len(llamadas) == 4    # y no se le gasto cuota pidiendoselo


async def test_ciclo_sincroniza_sondea_salud_y_sondea_calidad_en_el_ciclo_cero():
    almacen = _almacen()
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=CATALOGO)))
    prov = [Proveedor("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, [])]
    proxy = Proxy({"kilo": prov[0]}, almacen, http)
    estado = Estado(almacen=almacen, proxy=proxy, llaves=set(), tope_pago_diario=0,
                    proveedores=prov, http=http)
    await ciclo(estado, contador=0)
    assert [r.clave for r in almacen.rutas_activas()] == ["kilo/x:free"]
    tipos = {t for (t,) in almacen._con.execute("SELECT tipo FROM sondas").fetchall()}
    assert tipos == {"salud", "calidad"}


async def test_ciclo_no_sondea_calidad_fuera_del_intervalo():
    almacen = _almacen()
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json=CATALOGO)))
    prov = [Proveedor("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, [])]
    proxy = Proxy({"kilo": prov[0]}, almacen, http)
    estado = Estado(almacen=almacen, proxy=proxy, llaves=set(), tope_pago_diario=0,
                    proveedores=prov, http=http)
    await ciclo(estado, contador=1)
    tipos = {t for (t,) in almacen._con.execute("SELECT tipo FROM sondas").fetchall()}
    assert tipos == {"salud"}
