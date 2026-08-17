import asyncio
import json

import httpx
import pytest

from llm_libre.modelos import Capacidades, Ruta
from llm_libre.almacen import Almacen
from llm_libre.proveedores import Proveedor
from llm_libre.proxy import (COOLDOWN_429_DEFAULT_S, COOLDOWN_429_MAXIMO_S,
                             COOLDOWN_BASE_S, LIMITE_PROBE_BAJO_DEMANDA_S,
                             LIMITE_PROBE_GLOBAL_POR_MINUTO, UMBRAL_SOSPECHA,
                             VENTANA_PROBE_GLOBAL_S, Proxy, _es_error_del_cliente)

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


# --- Round 9, MEDIUM 7 del gate ("el 429 es el unico resorte que le queda
#     al cliente"): el 429 dejo de reutilizar el backoff exponencial de
#     _castigar (que escalaba hasta COOLDOWN_TOPE_S=3600s) -- medido, 12
#     pedidos de una llave alcanzaban para enfriar 3 rutas via 429 real,
#     mucho mas alla de lo que la propia ventana de rate-limit del
#     proveedor justifica. Ahora respeta `Retry-After` cuando el proveedor
#     lo manda, y si no, usa un default corto y FLAT (no escala con 429s
#     repetidos) topeado a COOLDOWN_429_MAXIMO_S. ---

async def test_429_sin_retry_after_no_escala_con_golpes_repetidos():
    p = _proxy(lambda req: httpx.Response(429))
    await p.completar([_ruta("a:free")], CUERPO, ahora=0.0)
    primero = p.cooldowns["kilo/a:free"]
    await p.completar([_ruta("a:free")], CUERPO, ahora=primero)
    segundo = p.cooldowns["kilo/a:free"]
    assert segundo - primero == COOLDOWN_429_DEFAULT_S  # flat, no exponencial


async def test_429_respeta_el_retry_after_del_proveedor():
    p = _proxy(lambda req: httpx.Response(429, headers={"Retry-After": "45"}))
    await p.completar([_ruta("a:free")], CUERPO, ahora=0.0)
    assert p.cooldowns["kilo/a:free"] == 45.0


async def test_429_topea_un_retry_after_absurdo():
    p = _proxy(lambda req: httpx.Response(429, headers={"Retry-After": "999999"}))
    await p.completar([_ruta("a:free")], CUERPO, ahora=0.0)
    assert p.cooldowns["kilo/a:free"] == COOLDOWN_429_MAXIMO_S


async def test_429_sin_retry_after_castiga_sin_ninguna_sonda():
    p = _proxy(lambda req: httpx.Response(429))
    await p.completar([_ruta("a:free")], CUERPO, ahora=0.0)
    await p.esperar_sondas_pendientes()
    assert p.almacen._con.execute("SELECT COUNT(*) FROM sondas").fetchone()[0] == 0


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
    # Round 9: el 429 ya no escala con golpes repetidos (MEDIUM 7), asi que
    # dos rutas solo terminan con cooldowns distintos si el PROVEEDOR pide
    # duraciones distintas via Retry-After -- exactamente la fuente de
    # verdad que ahora se respeta.
    def handler(req):
        modelo = json.loads(req.content)["model"]
        retry = "100" if modelo == "a:free" else "20"
        return httpx.Response(429, headers={"Retry-After": retry})

    p = _proxy(handler)
    r = await p.completar([_ruta("a:free"), _ruta("b:free")], CUERPO, ahora=0.0)
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


# --- Hallazgo 2 de la revision de Task 13, y su rediseno final en round 8.
#     Solo un 429 castigaba (con backoff exponencial). Todo lo demas -- 500,
#     timeout, error de red, 200 sin contenido -- no dejaba NUNCA cooldown, y
#     con TIMEOUT_S=90 una ruta colgada (verificado: `blog` es una maquina
#     saturada) le cuesta al cliente hasta 5*90s=450s por pedido,
#     indefinidamente, mientras /health sigue en "ok" porque otra ruta esta
#     viva.
#
#     Rondas 6 y 7 intentaron resolverlo con predicados sobre el TRAFICO DEL
#     CLIENTE (una lista de codigos, el default invertido, atribucion a
#     nivel de cadena) -- y cada uno cayo por un vector nuevo, los dos
#     ultimos escondidos en las propias excepciones que round 7 escribio
#     (una cadena de una sola ruta, forzable por el cliente con `model` o
#     `x_min_contexto`; el corte de completar_stream por `if emitido:`).
#     Cuando las fugas estan en las excepciones que uno mismo escribio, el
#     eje esta mal, no sub-enumerado.
#
#     Round 8 cambia el eje: el trafico de un cliente real YA NO PUEDE
#     excluir una ruta directamente, nunca -- solo acumula SOSPECHA
#     (`Proxy._sospechar`). Cruzar `UMBRAL_SOSPECHA` fallos CONSECUTIVOS
#     (round 9: ya no "dentro de una ventana de tiempo", ver mas abajo)
#     programa, en segundo plano, una SONDA PROPIA con el mismo payload
#     fijo (`PING`) que ya usa la sonda periodica -- y es esa sonda, nunca
#     el pedido del cliente, la que decide si la ruta se
#     castiga (`_castigar`, el mismo backoff de siempre). El 429 sigue
#     intacto: castiga en el PRIMER golpe, sin pasar por sospecha. ---

async def test_menos_del_umbral_de_sospecha_no_dispara_sonda_ni_castiga():
    p = _proxy(lambda req: httpx.Response(500))
    for i in range(UMBRAL_SOSPECHA - 1):
        await p.completar([_ruta("a:free")], CUERPO, ahora=float(i))
    await p.esperar_sondas_pendientes()
    assert "kilo/a:free" not in p.cooldowns
    assert p.almacen._con.execute("SELECT COUNT(*) FROM sondas").fetchone()[0] == 0


async def test_al_cruzar_el_umbral_se_dispara_una_sonda_que_castiga_si_la_ruta_esta_rota():
    p = _proxy(lambda req: httpx.Response(500))   # rota para CUALQUIER payload
    for i in range(UMBRAL_SOSPECHA):
        await p.completar([_ruta("a:free")], CUERPO, ahora=float(i))
    await p.esperar_sondas_pendientes()
    assert p.cooldowns["kilo/a:free"] > 0.0
    # La sonda dejo su propia constancia en `sondas` -- la misma evidencia
    # que lee Almacen.tiene_evidencia_de_vida para /health.
    fila = p.almacen._con.execute(
        "SELECT tipo, ok FROM sondas WHERE clave = 'kilo/a:free'").fetchone()
    assert fila == ("salud", 0)


async def test_una_ruta_en_cooldown_por_sonda_se_salta_en_el_siguiente_pedido():
    p = _proxy(lambda req: httpx.Response(500))
    for i in range(UMBRAL_SOSPECHA):
        await p.completar([_ruta("a:free")], CUERPO, ahora=float(i))
    await p.esperar_sondas_pendientes()
    ahora = float(UMBRAL_SOSPECHA)
    assert p.cooldowns["kilo/a:free"] > ahora
    await p.completar([_ruta("a:free")], CUERPO, ahora=ahora)
    # completar() no filtra por cooldown (eso lo hace router.ordenar sobre
    # las metricas fusionadas, ver test_router.py) -- lo que se prueba aca es
    # que el cooldown SIGUE activo, sin que este ultimo intento lo reinicie.
    assert p.cooldowns["kilo/a:free"] > ahora


async def test_un_exito_limpia_la_sospecha_acumulada():
    estado = {"fallos": 0}

    def handler(req):
        estado["fallos"] += 1
        if estado["fallos"] <= UMBRAL_SOSPECHA - 1:
            return httpx.Response(500)
        return httpx.Response(200, json=_ok())

    p = _proxy(handler)
    for i in range(UMBRAL_SOSPECHA - 1):
        await p.completar([_ruta("a:free")], CUERPO, ahora=float(i))
    # El exito en el intento numero UMBRAL_SOSPECHA limpia la sospecha
    # acumulada: nunca llega a cruzar el umbral, no se dispara ninguna sonda.
    await p.completar([_ruta("a:free")], CUERPO, ahora=float(UMBRAL_SOSPECHA))
    await p.esperar_sondas_pendientes()
    assert "kilo/a:free" not in p.cooldowns
    assert p.almacen._con.execute("SELECT COUNT(*) FROM sondas").fetchone()[0] == 0

    # Y hacen falta UMBRAL_SOSPECHA fallos NUEVOS para volver a acumular --
    # no alcanza con uno solo, que es justo lo que probaria que la sospecha
    # NO se limpio.
    estado["fallos"] = 0

    def handler_2(req):
        estado["fallos"] += 1
        return httpx.Response(500)
    p.http = httpx.AsyncClient(transport=httpx.MockTransport(handler_2))
    await p.completar([_ruta("a:free")], CUERPO, ahora=100.0)
    await p.esperar_sondas_pendientes()
    assert "kilo/a:free" not in p.cooldowns


async def test_codigo_429_castiga_en_el_primer_golpe_sin_pasar_por_sospecha():
    # Contraste directo: un SOLO 429 (no UMBRAL_SOSPECHA) ya castiga, y sin
    # pasar por ninguna sonda. El path del 429 no se toco.
    p = _proxy(lambda req: httpx.Response(429))
    await p.completar([_ruta("a:free")], CUERPO, ahora=0.0)
    assert "kilo/a:free" in p.cooldowns
    assert p.almacen._con.execute("SELECT COUNT(*) FROM sondas").fetchone()[0] == 0


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
    # La cuenta de sospecha tambien incluye un 200 con cuerpo roto o sin
    # contenido -- no solo 5xx/red -- porque ninguno de los dos sirvio al
    # cliente. Un solo golpe no dispara nada (ya cubierto por
    # test_un_200_con_cuerpo_invalido_no_revienta_y_cae_a_503); N seguidos,
    # con la sonda confirmando, si castigan.
    p = _proxy(lambda req: httpx.Response(200, content=b"not json{{{"))
    for i in range(UMBRAL_SOSPECHA):
        await p.completar([_ruta("a:free")], CUERPO, ahora=float(i))
    await p.esperar_sondas_pendientes()
    assert "kilo/a:free" in p.cooldowns


# --- Re-revision: hallazgo HIGH. Un 4xx (que no sea 429) es un error
#     DETERMINISTA del CLIENTE -- payload invalido, parametro no soportado,
#     secuencia de roles invalida -- que el proveedor le devuelve a
#     CUALQUIERA que mande ese mismo pedido, sano o no. Contarlo hacia
#     cooldown (directo o via sospecha) convertiria el error de UN cliente
#     en un apagon para TODOS: verificado contra el registro real de 5
#     rutas, tres pedidos malformados seguidos bastan para dejar las cinco
#     en cooldown si se cuentan. Un 400 solo debe perjudicar al cliente que
#     lo mando. ---

async def test_tres_400_seguidos_no_disparan_sospecha_ni_castigan():
    p = _proxy(lambda req: httpx.Response(400, json={"error": "bad request"}))
    for i in range(UMBRAL_SOSPECHA):
        await p.completar([_ruta("a:free")], CUERPO, ahora=float(i))
    await p.esperar_sondas_pendientes()
    assert "kilo/a:free" not in p.cooldowns
    # Ni siquiera se disparo una sonda a confirmar -- el 400 nunca cuenta
    # como sospecha, no es que la sonda haya dado "sana".
    assert p.almacen._con.execute("SELECT COUNT(*) FROM sondas").fetchone()[0] == 0


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
    for i in range(UMBRAL_SOSPECHA):
        await p.completar([_ruta("a:free")], CUERPO, ahora=float(i))
    await p.esperar_sondas_pendientes()
    assert "kilo/a:free" not in p.cooldowns

    estado["fallar"] = False
    r = await p.completar([_ruta("a:free")], CUERPO, ahora=100.0)
    assert r.estado == 200


async def test_tres_500_siguen_castigando_via_sonda_igual_que_antes():
    # Regresion directa: el fix de 4xx no debe tocar el camino de 5xx.
    p = _proxy(lambda req: httpx.Response(500))
    for i in range(UMBRAL_SOSPECHA):
        await p.completar([_ruta("a:free")], CUERPO, ahora=float(i))
    await p.esperar_sondas_pendientes()
    assert "kilo/a:free" in p.cooldowns


async def test_mezcla_de_4xx_y_5xx_solo_cuenta_los_5xx():
    # 400, 500, 400, 500, 400, 500: si el 400 contara para sospecha,
    # dispararia la sonda antes de tiempo. Con UMBRAL_SOSPECHA=3, hacen
    # falta las TRES respuestas 500 (llamadas 2, 4 y 6) -- los 400
    # intercalados no cuentan (ni suman ni reinician la ventana). La sonda
    # (llamada 7, mas alla de la lista) tambien ve 500: confirma que la
    # ruta esta rota de verdad.
    codigos = [400, 500, 400, 500, 400, 500]
    llamadas = []

    def handler(req):
        codigo = codigos[len(llamadas)] if len(llamadas) < len(codigos) else 500
        llamadas.append(codigo)
        return httpx.Response(codigo)

    p = _proxy(handler)
    for i in range(5):   # las primeras 5 (400,500,400,500,400): solo dos 500
        await p.completar([_ruta("a:free")], CUERPO, ahora=float(i))
    await p.esperar_sondas_pendientes()
    assert "kilo/a:free" not in p.cooldowns

    await p.completar([_ruta("a:free")], CUERPO, ahora=5.0)   # el tercer 500
    await p.esperar_sondas_pendientes()
    assert "kilo/a:free" in p.cooldowns


# --- Re-revision (round 4), hallazgo HIGH: el fix de la ronda 3 saco el 4xx
#     del CONTADOR de cooldown, pero seguia escribiendose como evento
#     fallido comun -- y eso alimenta confiabilidad, que /health usa para
#     declarar una ruta muerta. 26 pedidos malformados de UNA llave bastaban
#     para tirar la confiabilidad de TODAS las rutas, con /health en "caido"
#     mientras una llave DISTINTA seguia recibiendo 200. Peor que el 503 de
#     la ronda anterior: el volumen persistente de /datos hace que un
#     reinicio del contenedor (que Coolify dispara SOLO porque /health
#     fallo) no limpie nada -- loop de reinicios contra un servicio sano. ---

async def test_un_400_se_registra_pero_marcado_como_error_del_cliente():
    p = _proxy(lambda req: httpx.Response(400, json={"error": "bad request"}))
    await p.completar([_ruta("a:free")], CUERPO, ahora=0.0)
    fila = p.almacen._con.execute(
        "SELECT ok, es_error_cliente FROM eventos WHERE clave = 'kilo/a:free'").fetchone()
    assert fila == (0, 1)


async def test_un_500_se_registra_sin_marca_de_error_del_cliente():
    p = _proxy(lambda req: httpx.Response(500))
    await p.completar([_ruta("a:free")], CUERPO, ahora=0.0)
    fila = p.almacen._con.execute(
        "SELECT ok, es_error_cliente FROM eventos WHERE clave = 'kilo/a:free'").fetchone()
    assert fila == (0, 0)


# --- Round 6: el EJE (atribucion) seguia siendo el correcto, pero la
#     IMPLEMENTACION lo invertia -- "todo 4xx es evidencia del pedido SALVO
#     estos siete codigos" es un default que oculta cualquier codigo en el
#     que nadie penso todavia. La prueba: agregar 405 al conjunto de
#     "evidencia de ruta" (o simplemente no pensarlo, que es lo que paso
#     con 401/403/404 en la ronda 5) deja la suite en verde igual. Medido
#     el costo: las 5 rutas devolviendo 405 (o 409/415/418/431/451,
#     cualquier 4xx que nadie anticipo) dejaba al cliente con 503 en el
#     100% de los pedidos, CERO cooldowns, y /health en 200 "ok" -- la
#     ronda 3, verbatim, con otro codigo.
#
#     PRINCIPIO (va en el codigo y en el spec S7): cuando no se puede saber
#     de quien es la culpa, HAY QUE CONTARLO. Una falsa alarma se recupera
#     sola -- alguien mira el ranking o /health, ve que la ruta esta bien,
#     sigue. Una salida silenciosa NO -- nadie mira nunca. Los costos son
#     asimetricos, y el default tiene que inclinarse hacia notar.
#
#     El default se invierte: un 4xx es evidencia de LA RUTA salvo que este
#     en una lista CORTA y justificada de codigos genuinamente sobre el
#     payload. 429/408/425 ya NO necesitan estar en ninguna lista: bajo el
#     default invertido caen del lado de la ruta solos -- buena senal de
#     que la forma es la correcta. ---

def test_400_413_422_son_evidencia_del_pedido():
    assert _es_error_del_cliente(400) is True    # Bad Request: ni se pudo interpretar
    assert _es_error_del_cliente(413) is True    # Payload Too Large: el TAMAÑO de este pedido
    assert _es_error_del_cliente(422) is True    # Unprocessable: invalido para ESTE pedido


def test_el_default_es_evidencia_de_ruta_para_todo_el_rango_4xx():
    # Pin del EJE, no de una lista: se recorre el rango 4xx COMPLETO (no una
    # muestra de codigos que alguien penso hoy) contra una copia
    # INDEPENDIENTE de la lista corta esperada -- no se importa
    # _CODIGOS_EVIDENCIA_DE_PEDIDO para la comparacion. Si alguien agrega un
    # codigo (405, el que uso el reviewer para probar el mutante de la
    # ronda anterior; o cualquier otro, conocido o no) al conjunto real sin
    # que este test tambien cambie, se pone rojo: fuerza que CUALQUIER
    # ampliacion de la lista corta pase por una decision deliberada y
    # documentada aca, no un cambio silencioso en proxy.py.
    lista_corta_esperada = {400, 413, 422}
    for codigo in range(400, 500):
        si_es_pedido = codigo in lista_corta_esperada
        assert _es_error_del_cliente(codigo) is si_es_pedido, codigo


def test_401_402_403_404_408_425_429_siguen_como_evidencia_de_ruta():
    # Regresion de las rondas 4/5: estos siete ya NO estan en ningun
    # conjunto explicito (el default invertido los cubre solos), pero el
    # comportamiento tiene que seguir siendo el mismo.
    for codigo in (401, 402, 403, 404, 408, 425, 429):
        assert _es_error_del_cliente(codigo) is False, codigo


async def test_tres_405_seguidos_castigan_via_sonda():
    # El codigo exacto que el reviewer uso para probar el mutante de round
    # 6: nadie penso en 405 explicitamente, y bajo el default invertido eso
    # ya no importa -- cualquier codigo no listado cuenta como sospecha, y
    # la sonda (que tambien recibe 405 de este mismo handler) confirma.
    p = _proxy(lambda req: httpx.Response(405))
    for i in range(UMBRAL_SOSPECHA):
        await p.completar([_ruta("a:free")], CUERPO, ahora=float(i))
    await p.esperar_sondas_pendientes()
    assert "kilo/a:free" in p.cooldowns


async def test_405_se_registra_sin_marca_de_error_del_cliente():
    p = _proxy(lambda req: httpx.Response(405))
    await p.completar([_ruta("a:free")], CUERPO, ahora=0.0)
    fila = p.almacen._con.execute(
        "SELECT ok, es_error_cliente FROM eventos WHERE clave = 'kilo/a:free'").fetchone()
    assert fila == (0, 0)


async def test_tres_409_415_418_431_451_seguidos_castigan_via_sonda():
    # Mas codigos que "nadie penso" -- 409 Conflict, 415 Unsupported Media
    # Type, 418 (el teapot), 431 Request Header Fields Too Large, 451
    # Unavailable For Legal Reasons. Ninguno esta en ninguna lista, y todos
    # tienen que contar como sospecha bajo el default invertido.
    for codigo in (409, 415, 418, 431, 451):
        p = _proxy(lambda req, c=codigo: httpx.Response(c))
        for i in range(UMBRAL_SOSPECHA):
            await p.completar([_ruta("a:free")], CUERPO, ahora=float(i))
        await p.esperar_sondas_pendientes()
        assert "kilo/a:free" in p.cooldowns, codigo


async def test_tres_400_422_seguidos_no_disparan_sospecha():
    for codigo in (400, 422):
        p = _proxy(lambda req, c=codigo: httpx.Response(c, json={"error": "x"}))
        for i in range(UMBRAL_SOSPECHA):
            await p.completar([_ruta("a:free")], CUERPO, ahora=float(i))
        await p.esperar_sondas_pendientes()
        assert "kilo/a:free" not in p.cooldowns, codigo


async def test_tres_413_seguidos_no_disparan_sospecha():
    p = _proxy(lambda req: httpx.Response(413, json={"error": "payload too large"}))
    for i in range(UMBRAL_SOSPECHA):
        await p.completar([_ruta("a:free")], CUERPO, ahora=float(i))
    await p.esperar_sondas_pendientes()
    assert "kilo/a:free" not in p.cooldowns


def _multi(*modelos):
    return [_ruta(m) for m in modelos]


def _ping(cuerpo: bytes) -> bool:
    """True si el pedido que le llego al mock es la sonda -- el mismo
    payload fijo `PING` (proxy.py), nunca uno que un cliente pueda escribir."""
    return json.loads(cuerpo)["messages"][0]["content"] == "ping"


# --- Round 8. Rondas 6 y 7 resolvieron la atribucion de CODIGOS y de CADENA,
#     y aun asi el gate encontro DOS vectores mas -- los dos, escotillas del
#     propio diseno de round 7:
#
#     1. Una cadena de UNA SOLA ruta. El round anterior la dejaba
#        commiteando de inmediato porque "no tiene con que compararse" --
#        pero el CLIENTE puede forzar esa cadena el mismo, con un `model`
#        explicito o con `x_min_contexto` (que /v1/ranking ya publica por
#        ruta, sin hacer falta ningun conocimiento interno). 15 pedidos
#        identicos bastaban para enfriar las cinco rutas, una por una.
#     2. La rama `if emitido:` de completar_stream, que commiteaba sin
#        ningun chequeo de cadena -- ver test_proxy_stream.py.
#
#     Round 8 saca el eje de "cuantas rutas hay en la cadena" por completo:
#     TRAFICO REAL NUNCA EXCLUYE UNA RUTA, sin importar la forma de la
#     cadena. Solo acumula sospecha; cruzar el umbral programa una sonda
#     PROPIA (payload fijo, el gateway lo escribe) que es la unica que
#     decide. Los tests de abajo reproducen el vector 1 exacto -- una cadena
#     de una sola ruta -- con el mock devolviendole a la SONDA una respuesta
#     distinta de la que le da al CLIENTE: si la ruta esta sana de verdad
#     (le va bien al "ping" aunque el pedido real falle) la sonda la salva;
#     si esta rota de verdad (le va mal a cualquier payload, incluido el
#     "ping"), la sonda confirma y castiga -- rapido, sin esperar el ciclo
#     de 5h. ---

async def test_una_falla_identica_en_cadena_de_una_sola_ruta_no_castiga_una_ruta_sana():
    # El vector 1 exacto del gate: `model` explicito narrows a UNA ruta.
    def handler(req):
        if _ping(req.content):
            return httpx.Response(200, json=_ok())
        return httpx.Response(403, json={"error": "contenido flageado"})

    p = _proxy(handler)
    for i in range(15):
        r = await p.completar(_multi("a:free"), CUERPO, ahora=float(i))
        assert r.estado == 503
    await p.esperar_sondas_pendientes()
    assert p.cooldowns == {}


@pytest.mark.parametrize("armar_handler", [
    lambda: (lambda req: httpx.Response(200, json=_ok()) if _ping(req.content)
             else httpx.Response(451, json={"error": "no disponible por razones legales"})),
    lambda: (lambda req: httpx.Response(200, json=_ok()) if _ping(req.content)
             else httpx.Response(200, json=_ok(contenido=None))),
])
async def test_mas_vectores_de_falla_identica_en_cadena_de_una_sola_ruta_no_castigan(armar_handler):
    p = _proxy(armar_handler())
    for i in range(15):
        await p.completar(_multi("a:free"), CUERPO, ahora=float(i))
    await p.esperar_sondas_pendientes()
    assert p.cooldowns == {}


async def test_un_timeout_identico_en_cadena_de_una_sola_ruta_no_castiga_una_ruta_sana():
    def handler(req):
        if _ping(req.content):
            return httpx.Response(200, json=_ok())
        raise httpx.ReadTimeout("prompt gigante", request=req)

    p = _proxy(handler)
    for i in range(15):
        await p.completar(_multi("a:free"), CUERPO, ahora=float(i))
    await p.esperar_sondas_pendientes()
    assert p.cooldowns == {}


async def test_una_falla_identica_en_cadena_multi_ruta_tampoco_castiga_ninguna_sana():
    # Continuidad con round 7: el vector multi-ruta tambien sigue cubierto,
    # ahora por el mismo mecanismo (ya no hace falta un chequeo de longitud
    # de cadena aparte).
    def handler(req):
        if _ping(req.content):
            return httpx.Response(200, json=_ok())
        return httpx.Response(403, json={"error": "contenido flageado"})

    rutas = _multi("m0:free", "m1:free", "m2:free", "m3:free", "m4:free")
    p = _proxy(handler)
    for i in range(UMBRAL_SOSPECHA):
        await p.completar(rutas, CUERPO, ahora=float(i))
    await p.esperar_sondas_pendientes()
    assert p.cooldowns == {}


async def test_el_evento_se_sigue_registrando_aunque_la_sospecha_no_castigue():
    # El fix es SOLO sobre cooldown. `eventos`/confiabilidad (una medicion,
    # no una exclusion) siguen contando cada intento sin cambios -- Parte 1
    # no se toca.
    rutas = _multi("m0:free", "m1:free")
    p = _proxy(lambda req: httpx.Response(403, json={"error": "contenido flageado"}))
    await p.completar(rutas, CUERPO, ahora=0.0)
    filas = p.almacen._con.execute(
        "SELECT clave, ok, es_error_cliente FROM eventos ORDER BY clave").fetchall()
    assert filas == [("kilo/m0:free", 0, 0), ("kilo/m1:free", 0, 0)]


async def test_una_ruta_rota_de_verdad_con_hermana_sana_se_enfria_rapido_via_sonda():
    # Contraste: cuando la ruta esta rota DE VERDAD (le va mal tambien a la
    # sonda), se enfria -- con una hermana sana en la cadena o sin ella
    # (siguiente test). "Rapido" quiere decir dentro de UMBRAL_SOSPECHA
    # pedidos + una sonda, no el ciclo de 5h.
    def handler(req):
        cuerpo = json.loads(req.content)
        if _ping(req.content):
            return httpx.Response(500)   # la sonda tambien la ve rota
        return httpx.Response(500) if cuerpo["model"] == "a:free" else httpx.Response(200, json=_ok())

    p = _proxy(handler)
    for i in range(UMBRAL_SOSPECHA):
        r = await p.completar(_multi("a:free", "b:free"), CUERPO, ahora=float(i))
        assert r.estado == 200
    await p.esperar_sondas_pendientes()
    assert "kilo/a:free" in p.cooldowns
    assert "kilo/b:free" not in p.cooldowns


async def test_una_ruta_rota_de_verdad_en_cadena_de_una_sola_ruta_se_enfria_rapido():
    p = _proxy(lambda req: httpx.Response(500))   # rota para CUALQUIER payload
    for i in range(UMBRAL_SOSPECHA):
        await p.completar(_multi("a:free"), CUERPO, ahora=float(i))
    await p.esperar_sondas_pendientes()
    assert "kilo/a:free" in p.cooldowns


async def test_un_pool_genuinamente_caido_se_excluye_sin_esperar_horas():
    # Requisito explicito del gate: un pool de verdad caido no puede quedar
    # esperando el ciclo de 5h -- las tres rutas se enfrian en el orden de
    # UMBRAL_SOSPECHA pedidos, no horas.
    rutas = _multi("m0:free", "m1:free", "m2:free")
    p = _proxy(lambda req: httpx.Response(500))
    for i in range(UMBRAL_SOSPECHA):
        await p.completar(rutas, CUERPO, ahora=float(i))
    await p.esperar_sondas_pendientes()
    for ruta in rutas:
        assert ruta.clave in p.cooldowns, ruta.clave


async def test_429_en_una_cadena_que_falla_entera_sigue_castigando_de_inmediato():
    # 429 no pasa por sospecha -- sigue castigando en el primer golpe, sin
    # importar si el resto de la cadena tambien fallo y sin disparar
    # ninguna sonda. Es una senal inequivoca de ESA ruta, no del pedido.
    rutas = _multi("m0:free", "m1:free", "m2:free")
    p = _proxy(lambda req: httpx.Response(429))
    r = await p.completar(rutas, CUERPO, ahora=0.0)
    assert r.estado == 503
    assert "kilo/m0:free" in p.cooldowns
    assert "kilo/m1:free" in p.cooldowns
    assert "kilo/m2:free" in p.cooldowns
    assert p.almacen._con.execute("SELECT COUNT(*) FROM sondas").fetchone()[0] == 0


# --- Los tres requisitos restantes que el gate pidio explicitamente:
#     rate-limit de sondas bajo demanda, decaimiento de sospecha, y rutas de
#     pago afuera del mecanismo. ---

async def test_el_limite_de_sondas_bajo_demanda_por_ruta_se_respeta():
    llamadas_ping = []

    def handler(req):
        if _ping(req.content):
            llamadas_ping.append(1)
            return httpx.Response(200, json=_ok())
        return httpx.Response(500)

    p = _proxy(handler)
    for i in range(UMBRAL_SOSPECHA):
        await p.completar([_ruta("a:free")], CUERPO, ahora=float(i))
    await p.esperar_sondas_pendientes()
    assert len(llamadas_ping) == 1   # la primera racha ya disparo su sonda

    # Mas fallos, TODAVIA dentro de LIMITE_PROBE_BAJO_DEMANDA_S de la
    # primera sonda: vuelven a cruzar el umbral, pero el rate limit absorbe
    # el pedido -- no se dispara una segunda sonda.
    assert LIMITE_PROBE_BAJO_DEMANDA_S > UMBRAL_SOSPECHA + 1  # supuesto del test
    for i in range(UMBRAL_SOSPECHA, 2 * UMBRAL_SOSPECHA):
        await p.completar([_ruta("a:free")], CUERPO, ahora=float(i))
    await p.esperar_sondas_pendientes()
    assert len(llamadas_ping) == 1


# --- Round 9, HIGH 3 del gate: la ventana de 10 min de round 8 hacia que
#     trafico mas lento que ~1 fallo cada 200s NUNCA juntara tres dentro de
#     la ventana -- medido, 80 fallos consecutivos espaciados 301s aparte
#     no disparaban ninguna sonda; a 300s si. Una ruta muerta en un
#     despliegue de trafico bajo se quedaba primera en `ordenar` para
#     siempre. La sospecha ahora es un CONTADOR consecutivo, sin ventana:
#     no evapora "porque el servicio esta tranquilo", solo se reinicia en
#     un exito real. ---

async def test_la_sospecha_no_evapora_aunque_el_trafico_sea_lento():
    p = _proxy(lambda req: httpx.Response(500))
    # Tres fallos separados por MUCHO mas que la vieja ventana de 10 min --
    # exactamente el escenario que el gate midio (301s de por medio) y que
    # antes dejaba la sospecha en cero para siempre.
    await p.completar([_ruta("a:free")], CUERPO, ahora=0.0)
    await p.completar([_ruta("a:free")], CUERPO, ahora=10_000.0)
    await p.completar([_ruta("a:free")], CUERPO, ahora=20_000.0)
    await p.esperar_sondas_pendientes()
    assert p.cooldowns["kilo/a:free"] > 0.0


async def test_un_exito_reinicia_el_contador_de_sospecha_no_el_reloj():
    # La proteccion real contra "dos incidentes no relacionados que se
    # suman" es el reinicio-en-exito, no una ventana de tiempo: dos fallos,
    # un exito real de por medio, y otros dos fallos (aunque lleguen
    # rapido) no deben sumar cuatro -- tienen que volver a empezar de cero.
    estado = {"codigo": 500}

    def handler(req):
        return httpx.Response(estado["codigo"])

    p = _proxy(handler)
    await p.completar([_ruta("a:free")], CUERPO, ahora=0.0)
    await p.completar([_ruta("a:free")], CUERPO, ahora=1.0)
    estado["codigo"] = 200

    def handler_ok(req):
        return httpx.Response(200, json=_ok())
    p.http = httpx.AsyncClient(transport=httpx.MockTransport(handler_ok))
    await p.completar([_ruta("a:free")], CUERPO, ahora=2.0)  # exito: reinicia

    p.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await p.completar([_ruta("a:free")], CUERPO, ahora=3.0)
    await p.completar([_ruta("a:free")], CUERPO, ahora=4.0)
    await p.esperar_sondas_pendientes()
    assert "kilo/a:free" not in p.cooldowns  # solo dos fallos NUEVOS, no alcanza


# --- Round 9, HIGH 4 del gate: round 8 excluyo a las rutas de pago del
#     mecanismo de sospecha+sonda entero (nunca se sondean, una sonda bajo
#     demanda gastaria plata sin dueno) -- pero eso, solo, dejaba una ruta
#     de pago rota facturando cada pedido para siempre sin que NADA la
#     excluyera (round 7 SI la enfriaba). Se reintroduce un castigo
#     DIRECTO, sin sonda, con el mismo umbral -- bounded por ser el ultimo
#     escalon de la cadena. ---

async def test_las_rutas_de_pago_castigan_directo_sin_ninguna_sonda():
    p = _proxy(lambda req: httpx.Response(500), proveedores=("minimax",))
    ruta_pago = _ruta("m1", proveedor="minimax", tier="pago")
    for i in range(UMBRAL_SOSPECHA):
        await p.completar([ruta_pago], CUERPO, ahora=float(i))
    await p.esperar_sondas_pendientes()
    assert p.cooldowns["minimax/m1"] > 0.0
    # Directo: nunca se disparo (ni se gasto) ninguna sonda para llegar ahi.
    assert p.almacen._con.execute("SELECT COUNT(*) FROM sondas").fetchone()[0] == 0


async def test_menos_del_umbral_no_castiga_una_ruta_de_pago():
    p = _proxy(lambda req: httpx.Response(500), proveedores=("minimax",))
    ruta_pago = _ruta("m1", proveedor="minimax", tier="pago")
    for i in range(UMBRAL_SOSPECHA - 1):
        await p.completar([ruta_pago], CUERPO, ahora=float(i))
    await p.esperar_sondas_pendientes()
    assert "minimax/m1" not in p.cooldowns


async def test_un_exito_de_pago_limpia_el_contador_de_fallos_de_pago():
    estado = {"fallos": 0}

    def handler(req):
        estado["fallos"] += 1
        if estado["fallos"] <= UMBRAL_SOSPECHA - 1:
            return httpx.Response(500)
        return httpx.Response(200, json=_ok())

    p = _proxy(handler, proveedores=("minimax",))
    ruta_pago = _ruta("m1", proveedor="minimax", tier="pago")
    for i in range(UMBRAL_SOSPECHA - 1):
        await p.completar([ruta_pago], CUERPO, ahora=float(i))
    await p.completar([ruta_pago], CUERPO, ahora=float(UMBRAL_SOSPECHA))  # exito
    assert "minimax/m1" not in p.cooldowns


# --- Round 9, HIGH 2 del gate: el cooldown se estampaba con el `ahora` de
#     cuando arranco el intento, no con cuanto tardo. Medido en produccion
#     (TIMEOUT_S=90): exclusion efectiva max(0, 60*2^(n-1) - 90) = 0s, 30s,
#     150s, 390s en los primeros cuatro castigos -- una ruta COLGADA nacia
#     con su cooldown ya vencido. Se prueba con una demora real chica (no
#     90s reales) para verificar la aritmetica sin volver el test lento. ---

async def test_el_cooldown_de_un_intento_lento_no_nace_ya_comido():
    demora_s = 0.05

    async def handler(req):
        await asyncio.sleep(demora_s)
        return httpx.Response(500)

    p = _proxy(handler)
    ahora = 1000.0
    r = await p.completar([_ruta("a:free")], CUERPO, ahora=ahora, es_sonda=True)
    assert r.estado == 503
    # Sin el fix: cooldowns["kilo/a:free"] == ahora + COOLDOWN_BASE_S exacto.
    # Con el fix: ahora + demora_s + COOLDOWN_BASE_S -- mas que eso.
    assert p.cooldowns["kilo/a:free"] > ahora + COOLDOWN_BASE_S
    assert p.cooldowns["kilo/a:free"] >= ahora + demora_s + COOLDOWN_BASE_S


# --- Round 9, MEDIUM 5 del gate: un exito de una sonda bajo demanda podia
#     borrar un 429 real MAS NUEVO que la propia sonda, si el 429 llegaba
#     mientras la sonda seguia en vuelo -- un camino para que un cliente
#     cancelara el "pariate" del proveedor via sondas y siguiera martillando
#     la llave compartida. ---

async def test_una_sonda_exitosa_no_borra_un_429_mas_nuevo_que_ella():
    sonda_en_curso = asyncio.Event()
    continuar_sonda = asyncio.Event()
    trafico = {"codigo": 500}

    async def handler(req):
        cuerpo = json.loads(req.content)
        if cuerpo["messages"][0]["content"] == "ping":
            sonda_en_curso.set()
            await continuar_sonda.wait()
            return httpx.Response(200, json=_ok())
        return httpx.Response(trafico["codigo"])

    p = _proxy(handler)
    for i in range(UMBRAL_SOSPECHA):
        await p.completar([_ruta("a:free")], CUERPO, ahora=float(i))
    await sonda_en_curso.wait()  # la sonda arranco y quedo congelada en vuelo

    # Mientras tanto, un 429 REAL llega por trafico normal -- mas nuevo que
    # la sonda que todavia no termino.
    trafico["codigo"] = 429
    await p.completar([_ruta("a:free")], CUERPO, ahora=1000.0)
    assert p.cooldowns["kilo/a:free"] == 1000.0 + COOLDOWN_429_DEFAULT_S

    # Se libera la sonda: resuelve EXITOSA -- pero no debe pisar el 429 que
    # llego despues de que arranco.
    continuar_sonda.set()
    await p.esperar_sondas_pendientes()
    assert p.cooldowns["kilo/a:free"] == 1000.0 + COOLDOWN_429_DEFAULT_S


async def test_una_sonda_no_arranca_si_la_ruta_ya_esta_en_cooldown():
    # Corolario del mismo fix: si para cuando le toca correr la ruta YA
    # esta en cooldown (p.ej. un 429 que llego antes de que el scheduler le
    # diera tiempo a la tarea), ni se gasta la sonda.
    llamadas_ping = []

    async def handler(req):
        cuerpo = json.loads(req.content)
        if cuerpo["messages"][0]["content"] == "ping":
            llamadas_ping.append(1)
        return httpx.Response(200, json=_ok())

    p = _proxy(handler)
    p.cooldowns["kilo/a:free"] = 10_000.0  # ya castigada, bien en el futuro
    await p._probar_bajo_demanda(_ruta("a:free"), ahora=100.0)
    assert llamadas_ping == []


# --- Round 9, MEDIUM 6 del gate: el limite de sondas bajo demanda era POR
#     RUTA -- el AGREGADO no estaba acotado. Medido: 11 rutas, 15.840
#     pedidos extra por dia. Tope global, independiente de cuantas rutas
#     tenga el catalogo. ---

async def test_el_limite_global_de_sondas_bajo_demanda_se_respeta():
    llamadas_ping = []

    def handler(req):
        cuerpo = json.loads(req.content)
        if cuerpo["messages"][0]["content"] == "ping":
            llamadas_ping.append(1)
            return httpx.Response(200, json=_ok())
        return httpx.Response(500)

    p = _proxy(handler)
    rutas = _multi(*[f"m{i}:free" for i in range(LIMITE_PROBE_GLOBAL_POR_MINUTO + 3)])
    for ruta in rutas:
        for i in range(UMBRAL_SOSPECHA):
            await p.completar([ruta], CUERPO, ahora=0.0)
    await p.esperar_sondas_pendientes()
    assert len(llamadas_ping) == LIMITE_PROBE_GLOBAL_POR_MINUTO


async def test_el_limite_global_se_libera_pasada_la_ventana():
    llamadas_ping = []

    def handler(req):
        cuerpo = json.loads(req.content)
        if cuerpo["messages"][0]["content"] == "ping":
            llamadas_ping.append(1)
            return httpx.Response(200, json=_ok())
        return httpx.Response(500)

    p = _proxy(handler)
    rutas = _multi(*[f"m{i}:free" for i in range(LIMITE_PROBE_GLOBAL_POR_MINUTO + 1)])
    for ruta in rutas:
        for i in range(UMBRAL_SOSPECHA):
            await p.completar([ruta], CUERPO, ahora=0.0)
    await p.esperar_sondas_pendientes()
    assert len(llamadas_ping) == LIMITE_PROBE_GLOBAL_POR_MINUTO

    # La ruta que se quedo sin cupo sigue con la sospecha en el tope (no se
    # resetea) -- mas alla de la ventana global, la proxima falla la
    # vuelve a intentar y esta vez el cupo esta libre.
    faltante = rutas[-1]
    await p.completar([faltante], CUERPO, ahora=VENTANA_PROBE_GLOBAL_S + 10.0)
    await p.esperar_sondas_pendientes()
    assert len(llamadas_ping) == LIMITE_PROBE_GLOBAL_POR_MINUTO + 1


# --- Round 9, LOW 8 del gate: la sonda bajo demanda corre en un
#     asyncio.Task en segundo plano -- una excepcion NO-HTTP (completar()
#     solo atrapa httpx.HTTPError) quedaba sin recuperar, silenciosa, con
#     `_sospechas` sin limpiar: la ruta quedaba trabada. ---

async def test_una_excepcion_no_http_en_la_sonda_no_revienta_ni_castiga_a_ciegas(caplog):
    p = _proxy(lambda req: httpx.Response(500))
    # La excepcion tiene que pasar DENTRO del intento de la SONDA (el
    # (UMBRAL_SOSPECHA+1)-esimo registrar_evento -- los primeros
    # UMBRAL_SOSPECHA son el trafico real que arma la sospecha), y ANTES
    # de que completar() llegue a su propia decision de castigo -- para
    # que el veredicto quede genuinamente sin resolver, no ya tomado.
    original = p.almacen.registrar_evento
    contador = {"n": 0}

    def _registrar_evento_que_a_veces_revienta(*a, **kw):
        contador["n"] += 1
        if contador["n"] > UMBRAL_SOSPECHA:
            raise RuntimeError("contencion simulada de sqlite bajo WAL")
        return original(*a, **kw)
    p.almacen.registrar_evento = _registrar_evento_que_a_veces_revienta

    for i in range(UMBRAL_SOSPECHA):
        await p.completar([_ruta("a:free")], CUERPO, ahora=float(i))
    await p.esperar_sondas_pendientes()  # no cuelga ni propaga

    assert "kilo/a:free" not in p.cooldowns  # sin veredicto, no muerta por accidente
    assert "sonda bajo demanda" in caplog.text  # quedo logueado, no silencioso
    assert "kilo/a:free" not in p._sospechas  # no queda trabada esperando para siempre
