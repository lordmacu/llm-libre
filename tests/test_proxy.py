import httpx
import pytest

from llm_libre.modelos import Capacidades, Ruta
from llm_libre.almacen import Almacen
from llm_libre.proveedores import Proveedor
from llm_libre.proxy import Proxy

CUERPO = {"model": "auto", "messages": [{"role": "user", "content": "hola"}]}


def _ruta(modelo, proveedor="kilo", tier="gratis"):
    return Ruta(proveedor, modelo, tier, Capacidades(True, False, 100000, 4096))


def _prov(pid="kilo", tier="gratis"):
    return Proveedor(pid, tier, "openai", f"https://{pid}.test", "", "/models", {}, [])


def _ok(contenido="hola"):
    return {"choices": [{"message": {"role": "assistant", "content": contenido}}]}


def _proxy(handler, proveedores=("kilo",)):
    almacen = Almacen(":memory:")
    almacen.crear_esquema()
    cliente = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return Proxy({p: _prov(p) for p in proveedores}, almacen, cliente)


async def test_devuelve_la_primera_ruta_que_responde():
    p = _proxy(lambda req: httpx.Response(200, json=_ok()))
    r = await p.completar([_ruta("a:free"), _ruta("b:free")], CUERPO, ahora=0.0)
    assert r.estado == 200
    assert r.ruta.modelo_id == "a:free"
    assert r.intentos == 1


async def test_un_429_manda_la_ruta_a_cooldown_y_pasa_a_la_siguiente():
    llamadas = []

    def handler(req):
        llamadas.append(req.url)
        return httpx.Response(429) if len(llamadas) == 1 else httpx.Response(200, json=_ok())

    p = _proxy(handler)
    r = await p.completar([_ruta("a:free"), _ruta("b:free")], CUERPO, ahora=100.0)
    assert r.estado == 200
    assert r.ruta.modelo_id == "b:free"
    assert r.intentos == 2
    assert p.cooldowns["kilo/a:free"] > 100.0


async def test_el_cooldown_crece_con_cada_429_seguido():
    p = _proxy(lambda req: httpx.Response(429))
    await p.completar([_ruta("a:free")], CUERPO, ahora=0.0)
    primero = p.cooldowns["kilo/a:free"]
    await p.completar([_ruta("a:free")], CUERPO, ahora=primero)
    assert p.cooldowns["kilo/a:free"] - primero > primero


async def test_un_exito_limpia_el_castigo_acumulado():
    estado = {"fallar": True}

    def handler(req):
        return httpx.Response(429) if estado["fallar"] else httpx.Response(200, json=_ok())

    p = _proxy(handler)
    await p.completar([_ruta("a:free")], CUERPO, ahora=0.0)
    estado["fallar"] = False
    await p.completar([_ruta("a:free")], CUERPO, ahora=1000.0)
    assert "kilo/a:free" not in p.cooldowns


async def test_agotadas_todas_las_rutas_devuelve_503():
    p = _proxy(lambda req: httpx.Response(500))
    r = await p.completar([_ruta("a:free"), _ruta("b:free")], CUERPO, ahora=0.0)
    assert r.estado == 503
    assert r.ruta is None
    assert r.intentos == 2


async def test_sin_rutas_devuelve_503_sin_intentar():
    p = _proxy(lambda req: httpx.Response(200, json=_ok()))
    r = await p.completar([], CUERPO, ahora=0.0)
    assert r.estado == 503
    assert r.intentos == 0


async def test_recorta_el_razonamiento_de_la_respuesta():
    p = _proxy(lambda req: httpx.Response(200, json=_ok("<think>mmm</think>hola")))
    r = await p.completar([_ruta("a:free")], CUERPO, ahora=0.0)
    assert r.json["choices"][0]["message"]["content"] == "hola"
    assert r.razonamiento == "mmm"


async def test_en_modo_crudo_no_toca_el_contenido():
    p = _proxy(lambda req: httpx.Response(200, json=_ok("<think>mmm</think>hola")))
    r = await p.completar([_ruta("a:free")], CUERPO, ahora=0.0, crudo=True)
    assert r.json["choices"][0]["message"]["content"] == "<think>mmm</think>hola"
    assert r.razonamiento == ""


async def test_manda_el_id_real_del_modelo_no_el_alias():
    vistos = []

    def handler(req):
        import json
        vistos.append(json.loads(req.content)["model"])
        return httpx.Response(200, json=_ok())

    p = _proxy(handler)
    await p.completar([_ruta("poolside/x:free")], CUERPO, ahora=0.0)
    assert vistos == ["poolside/x:free"]


async def test_registra_un_evento_por_intento():
    p = _proxy(lambda req: httpx.Response(200, json=_ok()))
    await p.completar([_ruta("a:free")], CUERPO, ahora=0.0)
    filas = p.almacen._con.execute("SELECT clave, ok FROM eventos").fetchall()
    assert filas == [("kilo/a:free", 1)]
