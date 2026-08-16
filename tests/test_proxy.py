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


async def test_un_200_con_cuerpo_invalido_no_revienta_y_cae_a_503():
    p = _proxy(lambda req: httpx.Response(200, content=b"not json{{{"))
    r = await p.completar([_ruta("a:free")], CUERPO, ahora=0.0)
    assert r.estado == 503
    # No es rate-limit: la ruta rota no debe quedar castigada.
    assert "kilo/a:free" not in p.cooldowns


async def test_un_200_con_cuerpo_invalido_pasa_a_la_siguiente_ruta():
    llamadas = []

    def handler(req):
        llamadas.append(req.url)
        if len(llamadas) == 1:
            return httpx.Response(200, content=b"not json{{{")
        return httpx.Response(200, json=_ok())

    p = _proxy(handler)
    r = await p.completar([_ruta("a:free"), _ruta("b:free")], CUERPO, ahora=0.0)
    assert r.estado == 200
    assert r.ruta.modelo_id == "b:free"
    assert r.intentos == 2


# --- Fix round 3, B1 (Blocking): un 200 que no trae respuesta adentro no es un
#     exito. La mayoria de los modelos gratis son de razonamiento: se gastan el
#     presupuesto pensando y devuelven 200 con finish_reason "length" y
#     "content": null. Contarlo como exito SUBE la confiabilidad de esa ruta,
#     deja /health en "ok" y no hace failover: el cliente recibe una respuesta
#     vacia como si fuera la respuesta. ---

def _vacia(finish="length"):
    """El 200 real que devuelve un modelo de razonamiento que se quedo sin
    presupuesto: content null, sin tool_calls."""
    return {"choices": [{"message": {"role": "assistant", "content": None},
                         "finish_reason": finish}]}


async def test_un_200_sin_contenido_no_cuenta_como_exito():
    p = _proxy(lambda req: httpx.Response(200, json=_vacia()))
    r = await p.completar([_ruta("a:free")], CUERPO, ahora=0.0)
    assert r.estado == 503
    assert r.ruta is None
    # No es rate-limit: la ruta no debe quedar castigada, igual que con un
    # cuerpo no-JSON.
    assert "kilo/a:free" not in p.cooldowns


async def test_un_200_sin_contenido_pasa_a_la_siguiente_ruta():
    llamadas = []

    def handler(req):
        llamadas.append(req.url)
        if len(llamadas) == 1:
            return httpx.Response(200, json=_vacia())
        return httpx.Response(200, json=_ok())

    p = _proxy(handler)
    r = await p.completar([_ruta("a:free"), _ruta("b:free")], CUERPO, ahora=0.0)
    assert r.estado == 200
    assert r.ruta.modelo_id == "b:free"
    assert r.intentos == 2


async def test_un_200_sin_contenido_se_registra_como_evento_fallido():
    # El corazon del hallazgo: si esto se registra con ok=1, la ruta que
    # devuelve vacio SUBE su confiabilidad cada vez que falla.
    p = _proxy(lambda req: httpx.Response(200, json=_vacia()))
    await p.completar([_ruta("a:free")], CUERPO, ahora=0.0)
    filas = p.almacen._con.execute("SELECT clave, ok FROM eventos").fetchall()
    assert filas == [("kilo/a:free", 0)]


async def test_un_200_con_contenido_en_blanco_tampoco_cuenta_como_exito():
    p = _proxy(lambda req: httpx.Response(200, json=_ok("   \n ")))
    r = await p.completar([_ruta("a:free")], CUERPO, ahora=0.0)
    assert r.estado == 503


async def test_un_200_con_solo_tool_calls_sigue_siendo_exito():
    # Caso legitimo que NO debe romperse: una respuesta de function calling
    # trae content null y toda la carga util en tool_calls.
    datos = {"choices": [{"message": {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "call_1", "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"}}]}}]}
    p = _proxy(lambda req: httpx.Response(200, json=datos))
    r = await p.completar([_ruta("a:free")], CUERPO, ahora=0.0)
    assert r.estado == 200
    assert r.ruta.modelo_id == "a:free"


async def test_un_200_que_es_todo_razonamiento_no_cuenta_como_exito():
    # Lo que el cliente ve es lo que decide: si tras recortar el <think> no
    # queda nada, la ruta no respondio nada.
    p = _proxy(lambda req: httpx.Response(200, json=_ok("<think>pienso y pienso</think>")))
    r = await p.completar([_ruta("a:free")], CUERPO, ahora=0.0)
    assert r.estado == 503


async def test_en_modo_crudo_un_200_de_puro_razonamiento_sigue_siendo_exito():
    # Con x_crudo el cliente pidio el contenido tal cual: ahi SI hay respuesta.
    p = _proxy(lambda req: httpx.Response(200, json=_ok("<think>pienso</think>")))
    r = await p.completar([_ruta("a:free")], CUERPO, ahora=0.0, crudo=True)
    assert r.estado == 200
    assert r.json["choices"][0]["message"]["content"] == "<think>pienso</think>"


async def test_proxima_liberacion_no_incluye_cooldowns_de_otro_pedido():
    import json as jsonlib

    def handler(req):
        modelo = jsonlib.loads(req.content)["model"]
        return httpx.Response(429) if modelo == "z:free" else httpx.Response(500)

    p = _proxy(handler)
    # Un pedido anterior, por rutas totalmente distintas, castiga a z:free.
    await p.completar([_ruta("z:free")], CUERPO, ahora=0.0)
    assert p.cooldowns["kilo/z:free"] > 0.0

    r = await p.completar([_ruta("a:free"), _ruta("b:free")], CUERPO, ahora=10.0)
    assert r.estado == 503
    assert r.json["error"]["proxima_liberacion"] is None


async def test_proxima_liberacion_reporta_la_mas_cercana_de_esta_cadena():
    p = _proxy(lambda req: httpx.Response(429))
    # kilo/a:free ya trae un 429 previo, asi que en la proxima ronda su cooldown
    # crece mas que el de kilo/b:free, que apenas cae por primera vez.
    await p.completar([_ruta("a:free")], CUERPO, ahora=0.0)
    primero_de_a = p.cooldowns["kilo/a:free"]

    r = await p.completar([_ruta("a:free"), _ruta("b:free")], CUERPO, ahora=primero_de_a)
    assert r.estado == 503
    assert p.cooldowns["kilo/b:free"] < p.cooldowns["kilo/a:free"]
    assert r.json["error"]["proxima_liberacion"] == p.cooldowns["kilo/b:free"]
