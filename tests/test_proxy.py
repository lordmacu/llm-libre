import httpx
import pytest

from llm_libre.modelos import Capacidades, Ruta
from llm_libre.almacen import Almacen
from llm_libre.proveedores import Proveedor
from llm_libre.proxy import TOPE_FALLOS_SEGUIDOS, Proxy

CUERPO = {"model": "auto", "messages": [{"role": "user", "content": "hola"}]}


def _ruta(modelo, proveedor="kilo", tier="gratis"):
    return Ruta(proveedor, modelo, tier, Capacidades(True, False, 100000, 4096))


def _prov(pid="kilo", tier="gratis", desenvuelve_canvas=False):
    return Proveedor(pid, tier, "openai", f"https://{pid}.test", "", "/models", {}, [],
                     desenvuelve_canvas=desenvuelve_canvas)


def _ok(contenido="hola"):
    return {"choices": [{"message": {"role": "assistant", "content": contenido}}]}


def _proxy(handler, proveedores=("kilo",), canvas=frozenset()):
    almacen = Almacen(":memory:")
    almacen.crear_esquema()
    cliente = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return Proxy({p: _prov(p, desenvuelve_canvas=p in canvas) for p in proveedores},
                 almacen, cliente)


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


async def test_desenvuelve_la_cerca_de_canvas_en_el_camino_no_streaming():
    # Solo un proveedor que declara desenvuelve_canvas=True (chatgpt-proxy)
    # la desenvuelve -- ver el hallazgo 1 de la revision, mas abajo.
    cerca = (':::writing{title="x"}\nhola\n:::')
    p = _proxy(lambda req: httpx.Response(200, json=_ok(cerca)),
              proveedores=("chatgpt",), canvas={"chatgpt"})
    r = await p.completar([_ruta("a:free", proveedor="chatgpt")], CUERPO, ahora=0.0)
    assert r.json["choices"][0]["message"]["content"] == "hola\n"


# --- Hallazgo 1 de la revision de Task 13: el desenvuelto de canvas era
#     GLOBAL, pero ':::nota{...}' / ':::tip{...}' tambien es sintaxis
#     Docusaurus/MDX estandar -- se reprodujo en vivo contra una ruta de
#     Kilo pidiendo documentacion. Un proveedor que NO declara
#     desenvuelve_canvas (Kilo, OpenRouter, MiniMax) tiene que dejar esas
#     marcas intactas. ---

async def test_un_proveedor_sin_desenvuelve_canvas_no_toca_las_marcas_docusaurus():
    nota = ":::note\nGuarda el token en el .env.\n:::"
    p = _proxy(lambda req: httpx.Response(200, json=_ok(nota)))   # kilo, sin canvas={}
    r = await p.completar([_ruta("a:free")], CUERPO, ahora=0.0)
    assert r.json["choices"][0]["message"]["content"] == nota


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


# --- Fix round 3, I5: el camino NO streaming no puede medir un
#     time-to-first-token (la respuesta llega entera de una vez), asi que deja
#     de escribir su round-trip en la columna de ttft y lo guarda en
#     latencia_ms, que es lo que de verdad midio. ---

async def test_el_camino_no_streaming_guarda_latencia_no_ttft():
    p = _proxy(lambda req: httpx.Response(200, json=_ok()))
    await p.completar([_ruta("a:free")], CUERPO, ahora=0.0)
    fila = p.almacen._con.execute(
        "SELECT ttft_ms, latencia_ms FROM eventos").fetchone()
    assert fila[0] == 0                # no se inventa un ttft
    assert fila[1] is not None         # pero la latencia real si queda registrada


# --- Fix round 4, Minor: un 200 cuyo JSON es valido pero NO es un objeto (una
#     lista) llegaba a `_limpiar`, que hace datos.get(...) -> AttributeError
#     sin atrapar -> 500. Preexistente, pero la defensa de `hay_respuesta`
#     quedo una linea DESPUES de donde hacia falta. Un gateway de passthrough
#     no puede devolver 500 porque el proveedor mando algo raro. ---

async def test_un_200_cuyo_json_no_es_un_objeto_no_revienta():
    p = _proxy(lambda req: httpx.Response(200, json=[1, 2, 3]))
    r = await p.completar([_ruta("a:free")], CUERPO, ahora=0.0)
    assert r.estado == 503
    assert "kilo/a:free" not in p.cooldowns     # no es rate-limit, esta rota


async def test_un_200_con_json_que_no_es_objeto_pasa_a_la_siguiente_ruta():
    llamadas = []

    def handler(req):
        llamadas.append(req.url)
        if len(llamadas) == 1:
            return httpx.Response(200, json=["esto no es una respuesta"])
        return httpx.Response(200, json=_ok())

    p = _proxy(handler)
    r = await p.completar([_ruta("a:free"), _ruta("b:free")], CUERPO, ahora=0.0)
    assert r.estado == 200 and r.ruta.modelo_id == "b:free"


# --- Hallazgo 2 de la revision de Task 13: solo un 429 castiga (con backoff
#     exponencial). Todo lo demas -- 500, timeout, error de red, 200 sin
#     contenido -- no dejaba NUNCA cooldown, y con TIMEOUT_S=90 una ruta
#     colgada (verificado: `blog` es una maquina saturada) le cuesta al
#     cliente hasta 5*90s=450s por pedido, indefinidamente, mientras /health
#     sigue en "ok" porque otra ruta esta viva. Un hiccup aislado no debe
#     sacar una ruta sana de la rotacion -- por eso recien despues de
#     TOPE_FALLOS_SEGUIDOS fallos SEGUIDOS (sin exito en el medio) se
#     castiga, con el MISMO backoff que ya usa el 429 (_castigar). El 429
#     sigue intacto: castiga en el PRIMER golpe, no despues de N. ---

async def test_menos_de_n_fallos_duros_no_castiga():
    p = _proxy(lambda req: httpx.Response(500))
    for i in range(TOPE_FALLOS_SEGUIDOS - 1):
        await p.completar([_ruta("a:free")], CUERPO, ahora=float(i))
    assert "kilo/a:free" not in p.cooldowns


async def test_n_fallos_duros_seguidos_pone_la_ruta_en_cooldown():
    p = _proxy(lambda req: httpx.Response(500))
    for i in range(TOPE_FALLOS_SEGUIDOS):
        await p.completar([_ruta("a:free")], CUERPO, ahora=float(i))
    assert p.cooldowns["kilo/a:free"] > float(TOPE_FALLOS_SEGUIDOS - 1)


async def test_una_ruta_en_cooldown_por_fallos_duros_se_salta_en_el_siguiente_pedido():
    p = _proxy(lambda req: httpx.Response(500))
    for i in range(TOPE_FALLOS_SEGUIDOS):
        await p.completar([_ruta("a:free")], CUERPO, ahora=float(i))
    ahora = float(TOPE_FALLOS_SEGUIDOS)
    assert p.cooldowns["kilo/a:free"] > ahora
    # sin la ruta castigada disponible, la cadena queda vacia: agotadas_todas
    r = await p.completar([_ruta("a:free")], CUERPO, ahora=ahora)
    # completar() no filtra por cooldown (eso lo hace router.ordenar sobre
    # las metricas fusionadas, ver test_router.py) -- lo que se prueba aca es
    # que el cooldown SIGUE activo, sin que este ultimo intento lo reinicie.
    assert p.cooldowns["kilo/a:free"] > ahora


async def test_un_exito_limpia_los_fallos_duros_seguidos():
    estado = {"fallos": 0}

    def handler(req):
        estado["fallos"] += 1
        if estado["fallos"] <= TOPE_FALLOS_SEGUIDOS - 1:
            return httpx.Response(500)
        return httpx.Response(200, json=_ok())

    p = _proxy(handler)
    for i in range(TOPE_FALLOS_SEGUIDOS - 1):
        await p.completar([_ruta("a:free")], CUERPO, ahora=float(i))
    # El exito en el intento numero TOPE_FALLOS_SEGUIDOS limpia el contador.
    await p.completar([_ruta("a:free")], CUERPO, ahora=float(TOPE_FALLOS_SEGUIDOS))
    assert "kilo/a:free" not in p.cooldowns

    # Y hacen falta TOPE_FALLOS_SEGUIDOS fallos NUEVOS para volver a castigar
    # -- no alcanza con uno solo, que es justo lo que probaria que el
    # contador NO se reinicio.
    estado["fallos"] = 0

    def handler_2(req):
        estado["fallos"] += 1
        return httpx.Response(500)
    p.http = httpx.AsyncClient(transport=httpx.MockTransport(handler_2))
    await p.completar([_ruta("a:free")], CUERPO, ahora=100.0)
    assert "kilo/a:free" not in p.cooldowns


async def test_codigo_429_castiga_en_el_primer_golpe_no_despues_de_n():
    # Contraste directo: un SOLO 429 (no TOPE_FALLOS_SEGUIDOS) ya castiga.
    # El path del 429 no se toco.
    p = _proxy(lambda req: httpx.Response(429))
    await p.completar([_ruta("a:free")], CUERPO, ahora=0.0)
    assert "kilo/a:free" in p.cooldowns


async def test_usa_el_timeout_global_si_el_proveedor_no_declara_el_suyo():
    vistos = []

    def handler(req):
        vistos.append(req.extensions.get("timeout"))
        return httpx.Response(200, json=_ok())

    p = _proxy(handler)   # kilo, sin timeout_s declarado
    await p.completar([_ruta("a:free")], CUERPO, ahora=0.0)
    assert vistos[0]["read"] == 90.0   # TIMEOUT_S


async def test_usa_el_timeout_propio_del_proveedor_si_lo_declara():
    vistos = []

    def handler(req):
        vistos.append(req.extensions.get("timeout"))
        return httpx.Response(200, json=_ok())

    almacen = Almacen(":memory:")
    almacen.crear_esquema()
    lento = Proveedor("lento", "gratis", "openai", "https://lento.test", "", "/models",
                      {}, [], timeout_s=20.0)
    p = Proxy({"lento": lento}, almacen, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    await p.completar([_ruta("a:free", proveedor="lento")], CUERPO, ahora=0.0)
    assert vistos[0]["read"] == 20.0


async def test_un_solo_200_invalido_no_castiga_pero_n_seguidos_si():
    # La cuenta de "fallo duro" tambien incluye un 200 con cuerpo roto o sin
    # contenido -- no solo 5xx/red -- porque ninguno de los dos sirvio al
    # cliente. Un solo golpe no castiga (ya cubierto por
    # test_un_200_con_cuerpo_invalido_no_revienta_y_cae_a_503); N seguidos si.
    p = _proxy(lambda req: httpx.Response(200, content=b"not json{{{"))
    for i in range(TOPE_FALLOS_SEGUIDOS):
        await p.completar([_ruta("a:free")], CUERPO, ahora=float(i))
    assert "kilo/a:free" in p.cooldowns


# --- Re-revision: hallazgo HIGH. Un 4xx (que no sea 429) es un error
#     DETERMINISTA del CLIENTE -- payload invalido, parametro no soportado,
#     secuencia de roles invalida -- que el proveedor le devuelve a
#     CUALQUIERA que mande ese mismo pedido, sano o no. Contarlo hacia el
#     cooldown convierte el error de UN cliente en un apagon para TODOS:
#     verificado contra el registro real de 5 rutas, tres pedidos
#     malformados seguidos bastan para dejar las cinco en cooldown, y una
#     llave DISTINTA con un pedido valido recibe 503 mientras tanto. Antes de
#     este fix un 400 solo perjudicaba al cliente que lo mando -- ahora tiene
#     que seguir siendo asi. ---

async def test_tres_400_seguidos_no_castigan():
    p = _proxy(lambda req: httpx.Response(400, json={"error": "bad request"}))
    for i in range(TOPE_FALLOS_SEGUIDOS):
        await p.completar([_ruta("a:free")], CUERPO, ahora=float(i))
    assert "kilo/a:free" not in p.cooldowns


async def test_tras_400_seguidos_un_pedido_valido_de_otra_llave_sigue_sirviendose():
    # "Otra llave" no es un concepto de Proxy (eso vive en api.py); lo que se
    # prueba aca es la causa raiz: sin cooldown activado por el 400, una
    # llamada SIGUIENTE (de quien sea) sigue intentando la ruta normalmente,
    # no la encuentra "agotada por castigo" de entrada.
    estado = {"fallar": True}

    def handler(req):
        if estado["fallar"]:
            return httpx.Response(400, json={"error": "context_length_exceeded"})
        return httpx.Response(200, json=_ok())

    p = _proxy(handler)
    for i in range(TOPE_FALLOS_SEGUIDOS):
        await p.completar([_ruta("a:free")], CUERPO, ahora=float(i))
    assert "kilo/a:free" not in p.cooldowns

    estado["fallar"] = False
    r = await p.completar([_ruta("a:free")], CUERPO, ahora=100.0)
    assert r.estado == 200


async def test_tres_500_siguen_castigando_igual_que_antes():
    # Regresion directa: el fix de 4xx no debe tocar el camino de 5xx.
    p = _proxy(lambda req: httpx.Response(500))
    for i in range(TOPE_FALLOS_SEGUIDOS):
        await p.completar([_ruta("a:free")], CUERPO, ahora=float(i))
    assert "kilo/a:free" in p.cooldowns


async def test_mezcla_de_4xx_y_5xx_solo_cuenta_los_5xx():
    # 400, 500, 400, 500, 400, 500: si el 400 contara, castigaria antes de
    # tiempo. Con TOPE_FALLOS_SEGUIDOS=3, hacen falta las TRES respuestas 500
    # (llamadas 2, 4 y 6) para castigar -- los 400 intercalados no cuentan
    # (ni suman ni reinician el contador).
    codigos = [400, 500, 400, 500, 400, 500]
    llamadas = []

    def handler(req):
        codigo = codigos[len(llamadas)]
        llamadas.append(codigo)
        return httpx.Response(codigo)

    p = _proxy(handler)
    for i in range(5):   # las primeras 5 (400,500,400,500,400): solo dos 500
        await p.completar([_ruta("a:free")], CUERPO, ahora=float(i))
    assert "kilo/a:free" not in p.cooldowns

    await p.completar([_ruta("a:free")], CUERPO, ahora=5.0)   # el tercer 500
    assert "kilo/a:free" in p.cooldowns
