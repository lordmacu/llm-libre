import asyncio
import json
import logging
import time

import httpx

from llm_libre.almacen import Almacen
from llm_libre.modelos import Capacidades, Ruta
from llm_libre.proveedores import Proveedor
from llm_libre.proxy import Proxy

CUERPO = {"model": "auto", "messages": [], "stream": True}


def _ruta(modelo="a:free"):
    return Ruta("kilo", modelo, "gratis", Capacidades(True, False, 100000, 4096))


def _sse(*trozos):
    lineas = []
    for t in trozos:
        lineas.append('data: {"choices":[{"delta":{"content":"%s"}}]}\n\n' % t)
    lineas.append("data: [DONE]\n\n")
    return "".join(lineas).encode()


def _proxy(handler):
    almacen = Almacen(":memory:")
    almacen.crear_esquema()
    prov = {"kilo": Proveedor("kilo", "gratis", "openai", "https://k.test", "", "/models", {}, [])}
    return Proxy(prov, almacen, httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def _juntar(gen):
    texto = ""
    async for linea in gen:
        if not linea.startswith("data: ") or "[DONE]" in linea:
            continue
        obj = json.loads(linea[6:])
        texto += (obj.get("choices", [{}])[0].get("delta", {}) or {}).get("content", "")
    return texto


class _FlujoQuebrado(httpx.AsyncByteStream):
    """Simula una conexion que entrega un trozo real y despues se corta."""

    def __init__(self, trozo: bytes):
        self._trozo = trozo

    async def __aiter__(self):
        yield self._trozo
        raise httpx.ReadError("conexion cortada a mitad de stream")


class _FlujoConDemora(httpx.AsyncByteStream):
    """Deja pasar tiempo real antes del primer chunk y de nuevo antes de [DONE],
    para poder distinguir en un test si el ttft se mide en el primer token o en
    el cierre del stream."""

    def __init__(self, demora_antes: float, demora_despues: float):
        self._demora_antes = demora_antes
        self._demora_despues = demora_despues

    async def __aiter__(self):
        await asyncio.sleep(self._demora_antes)
        yield b'data: {"choices":[{"delta":{"content":"hola"}}]}\n\n'
        await asyncio.sleep(self._demora_despues)
        yield b"data: [DONE]\n\n"


async def test_pasa_el_contenido_completo():
    p = _proxy(lambda req: httpx.Response(200, content=_sse("ho", "la")))
    assert await _juntar(p.completar_stream([_ruta()], CUERPO, 0.0)) == "hola"


async def test_recorta_el_razonamiento_partido_entre_chunks():
    p = _proxy(lambda req: httpx.Response(200, content=_sse("ho", "<thi", "nk>zz</think>", "la")))
    assert await _juntar(p.completar_stream([_ruta()], CUERPO, 0.0)) == "hola"


async def test_una_etiqueta_que_nunca_cierra_no_cuelga_el_stream():
    p = _proxy(lambda req: httpx.Response(200, content=_sse("ok", "<think>", "sin cerrar")))
    assert await _juntar(p.completar_stream([_ruta()], CUERPO, 0.0)) == "ok"


async def test_hace_failover_si_la_primera_ruta_falla_antes_de_emitir():
    llamadas = []

    def handler(req):
        llamadas.append(1)
        if len(llamadas) == 1:
            return httpx.Response(429)
        return httpx.Response(200, content=_sse("bien"))

    p = _proxy(handler)
    assert await _juntar(p.completar_stream([_ruta("a:free"), _ruta("b:free")], CUERPO, 0.0)) == "bien"


async def test_termina_siempre_con_done():
    p = _proxy(lambda req: httpx.Response(200, content=_sse("x")))
    lineas = [l async for l in p.completar_stream([_ruta()], CUERPO, 0.0)]
    assert lineas[-1].strip() == "data: [DONE]"


async def test_sin_rutas_emite_un_error_y_cierra():
    p = _proxy(lambda req: httpx.Response(200, content=_sse("x")))
    lineas = [l async for l in p.completar_stream([], CUERPO, 0.0)]
    assert any("error" in l for l in lineas)
    assert lineas[-1].strip() == "data: [DONE]"


async def test_crudo_no_recorta_el_contenido():
    p = _proxy(lambda req: httpx.Response(200, content=_sse("<think>mmm</think>hola")))
    texto = await _juntar(p.completar_stream([_ruta()], CUERPO, 0.0, crudo=True))
    assert texto == "<think>mmm</think>hola"


async def test_no_descarta_un_chunk_de_tool_calls_con_contenido_vacio():
    # En streaming estilo OpenAI un chunk de tool_calls suele viajar con
    # content="". El bug del brief original lo descartaba por completo con un
    # `continue` que solo miraba si el contenido recortado quedaba vacio.
    cuerpo = (b'data: {"choices":[{"delta":{"content":"",'
             b'"tool_calls":[{"index":0,"id":"call_1","function":{"name":"buscar"}}]}}]}\n\n'
             b'data: [DONE]\n\n')
    p = _proxy(lambda req: httpx.Response(200, content=cuerpo))
    lineas = [l async for l in p.completar_stream([_ruta()], CUERPO, 0.0)]
    utiles = [l for l in lineas if "[DONE]" not in l]
    assert len(utiles) == 1
    obj = json.loads(utiles[0][len("data: "):])
    assert obj["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "buscar"


async def test_no_descarta_un_chunk_con_finish_reason_y_contenido_vacio():
    # El chunk final de un stream normal trae content vacio y solo
    # finish_reason: no debe perderse. (Fix round 3, B1: antes este test
    # mandaba SOLO ese chunk, un stream sin ninguna respuesta adentro, que hoy
    # cuenta como intento fallido -- por eso ahora va precedido de contenido
    # real, que es la forma en que el caso ocurre de verdad.)
    #
    # Fix round 4, N2: ademas usaba una forma que el protocolo real de OpenAI
    # NO produce -- `finish_reason` DENTRO del delta. En el protocolo real es
    # HERMANO de `delta`, y con esa forma el chunk se estaba perdiendo. Por eso
    # el bug paso desapercibido: el unico test que lo cubria no usaba la forma
    # de verdad. Ahora si.
    cuerpo = (b'data: {"choices":[{"index":0,"delta":{"content":"hola"}}]}\n\n'
             b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
             b'data: [DONE]\n\n')
    p = _proxy(lambda req: httpx.Response(200, content=cuerpo))
    lineas = [l async for l in p.completar_stream([_ruta()], CUERPO, 0.0)]
    utiles = [l for l in lineas if "[DONE]" not in l]
    assert len(utiles) == 2
    obj = json.loads(utiles[-1][len("data: "):])
    assert obj["choices"][0]["finish_reason"] == "stop"   # hermano de delta


async def test_no_hace_failover_si_la_conexion_se_corta_despues_de_emitir():
    llamadas = []

    def handler(req):
        llamadas.append(1)
        trozo = b'data: {"choices":[{"delta":{"content":"real"}}]}\n\n'
        return httpx.Response(200, stream=_FlujoQuebrado(trozo))

    p = _proxy(handler)
    lineas = [l async for l in p.completar_stream([_ruta("a:free"), _ruta("b:free")], CUERPO, 0.0)]
    # Solo se intento la primera ruta: una vez que le llego contenido real al
    # cliente no se puede saltar a la segunda ruta sin mezclar dos respuestas.
    assert len(llamadas) == 1
    assert any("real" in l for l in lineas)
    assert lineas[-1].strip() == "data: [DONE]"
    # Y la falla posterior a la emision no debe sumar un segundo evento encima
    # del que ya se registro cuando salio el primer chunk util.
    filas = p.almacen._con.execute("SELECT ok FROM eventos WHERE clave = ?",
                                   ("kilo/a:free",)).fetchall()
    assert filas == [(1,)]


async def test_no_descarta_un_chunk_con_tool_calls_vacio_pero_presente():
    # "tool_calls": [] es un valor falsy, pero la CLAVE esta presente: filtrar
    # por verdad del valor (en vez de por presencia de la clave) lo tiraria
    # igual que si no llevara nada, perdiendo la senal de que hay una tool call
    # en curso.
    cuerpo = (b'data: {"choices":[{"delta":{"content":"","tool_calls":[]}}]}\n\n'
             b'data: [DONE]\n\n')
    p = _proxy(lambda req: httpx.Response(200, content=cuerpo))
    lineas = [l async for l in p.completar_stream([_ruta()], CUERPO, 0.0)]
    utiles = [l for l in lineas if "[DONE]" not in l]
    assert len(utiles) == 1
    obj = json.loads(utiles[0][len("data: "):])
    assert obj["choices"][0]["delta"]["tool_calls"] == []


async def test_stream_de_puro_razonamiento_registra_un_evento_fallido():
    # Fix round 3, B1 (Blocking). Este test afirmaba lo contrario: que un
    # stream que no entrega NADA util igual cuenta como exito porque "la
    # llamada HTTP si funciono". Eso es justo el agujero: el cliente se queda
    # sin respuesta mientras la confiabilidad de la ruta SUBE, /health sigue en
    # "ok" y no hay failover. Lo que cuenta como exito es haber entregado
    # contenido o tool_calls, no haber recibido un 200.
    p = _proxy(lambda req: httpx.Response(
        200, content=_sse("<think>solo razonamiento</think>")))
    texto = await _juntar(p.completar_stream([_ruta()], CUERPO, 0.0))
    assert texto == ""
    filas = p.almacen._con.execute("SELECT ok FROM eventos").fetchall()
    assert filas == [(0,)]


async def test_un_stream_sin_contenido_util_hace_failover():
    # El 200 vacio de un modelo de razonamiento en streaming: chunks de role y
    # de finish_reason, ni una letra de contenido. Debe caer a la ruta
    # siguiente, no cerrarse con [DONE] como si hubiera respondido.
    llamadas = []
    vacio = (b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n'
             b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
             b'data: [DONE]\n\n')

    def handler(req):
        llamadas.append(1)
        if len(llamadas) == 1:
            return httpx.Response(200, content=vacio)
        return httpx.Response(200, content=_sse("bien"))

    p = _proxy(handler)
    texto = await _juntar(p.completar_stream([_ruta("a:free"), _ruta("b:free")], CUERPO, 0.0))
    assert texto == "bien"
    assert len(llamadas) == 2


async def test_un_stream_sin_contenido_util_registra_fallo_en_esa_ruta():
    vacio = (b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n'
             b'data: [DONE]\n\n')

    def handler(req):
        if "a:free" in req.content.decode():
            return httpx.Response(200, content=vacio)
        return httpx.Response(200, content=_sse("bien"))

    p = _proxy(handler)
    await _juntar(p.completar_stream([_ruta("a:free"), _ruta("b:free")], CUERPO, 0.0))
    filas = p.almacen._con.execute(
        "SELECT clave, ok FROM eventos ORDER BY clave").fetchall()
    assert filas == [("kilo/a:free", 0), ("kilo/b:free", 1)]


async def test_un_stream_de_solo_tool_calls_sigue_contando_como_exito():
    # Caso legitimo que NO debe romperse: un stream de function calling puede
    # no traer ni una letra de content.
    cuerpo = (b'data: {"choices":[{"delta":{"role":"assistant","content":"",'
              b'"tool_calls":[{"index":0,"id":"c1","function":{"name":"buscar"}}]}}]}\n\n'
              b'data: [DONE]\n\n')
    llamadas = []

    def handler(req):
        llamadas.append(1)
        return httpx.Response(200, content=cuerpo)

    p = _proxy(handler)
    lineas = [l async for l in p.completar_stream([_ruta("a:free"), _ruta("b:free")],
                                                  CUERPO, 0.0)]
    assert len(llamadas) == 1          # no hubo failover
    assert any("buscar" in l for l in lineas)
    filas = p.almacen._con.execute("SELECT ok FROM eventos").fetchall()
    assert filas == [(1,)]


async def test_un_stream_crudo_de_puro_razonamiento_sigue_siendo_exito():
    # Con x_crudo el cliente pidio el texto tal cual: el <think> ES la respuesta.
    p = _proxy(lambda req: httpx.Response(200, content=_sse("<think>mmm</think>")))
    texto = await _juntar(p.completar_stream([_ruta()], CUERPO, 0.0, crudo=True))
    assert texto == "<think>mmm</think>"
    filas = p.almacen._con.execute("SELECT ok FROM eventos").fetchall()
    assert filas == [(1,)]


async def test_falla_antes_de_emitir_registra_el_evento_una_sola_vez():
    llamadas = []

    def handler(req):
        llamadas.append(1)
        if len(llamadas) == 1:
            return httpx.Response(429)
        return httpx.Response(200, content=_sse("ok"))

    p = _proxy(handler)
    await _juntar(p.completar_stream([_ruta("a:free"), _ruta("b:free")], CUERPO, 0.0))
    filas = p.almacen._con.execute(
        "SELECT clave, ok FROM eventos ORDER BY clave").fetchall()
    assert filas == [("kilo/a:free", 0), ("kilo/b:free", 1)]


async def test_ttft_mide_el_primer_token_no_el_fin_del_stream():
    demora_antes, demora_despues = 0.05, 0.2

    def handler(req):
        return httpx.Response(200, stream=_FlujoConDemora(demora_antes, demora_despues))

    p = _proxy(handler)
    t0 = time.monotonic()
    lineas = [l async for l in p.completar_stream([_ruta()], CUERPO, 0.0)]
    duracion_total_ms = (time.monotonic() - t0) * 1000

    assert any("hola" in l for l in lineas)
    assert lineas[-1].strip() == "data: [DONE]"

    filas = p.almacen._con.execute("SELECT ok, ttft_ms FROM eventos").fetchall()
    assert filas == [(1, filas[0][1])]  # exactamente un evento
    ttft_ms = filas[0][1]

    # El stream completo tarda ~(demora_antes + demora_despues) = 250ms, pero el
    # primer token sale a los ~50ms. Si el ttft se hubiera medido al final del
    # stream (la regresion que este test detecta) quedaria pegado a
    # duracion_total_ms en vez de quedarse cerca de demora_antes.
    assert ttft_ms < duracion_total_ms - 100
    assert ttft_ms < (demora_antes * 1000) + 100


async def test_el_camino_streaming_si_escribe_ttft():
    # La contracara de I5: aca el ttft SI se puede medir, y es el unico camino
    # que escribe esa columna.
    p = _proxy(lambda req: httpx.Response(200, content=_sse("hola")))
    await _juntar(p.completar_stream([_ruta()], CUERPO, 0.0))
    fila = p.almacen._con.execute("SELECT ttft_ms, latencia_ms FROM eventos").fetchone()
    assert fila[0] >= 0 and fila[1] is None


async def test_un_chunk_de_solo_espacios_no_se_pierde():
    # Los deltas de streaming vienen partidos y muchos son " " o "\n" sueltos.
    # No cuentan como "respuesta util" para decidir el exito, pero SI son texto
    # del cliente: no se pueden tirar. Se retienen y salen, en orden, junto al
    # primer chunk con contenido de verdad.
    p = _proxy(lambda req: httpx.Response(200, content=_sse(" ", "hola", " ", "mundo")))
    assert await _juntar(p.completar_stream([_ruta()], CUERPO, 0.0)) == " hola mundo"


async def test_un_stream_de_puros_espacios_no_es_una_respuesta():
    llamadas = []

    def handler(req):
        llamadas.append(1)
        if len(llamadas) == 1:
            return httpx.Response(200, content=_sse(" ", "\n", "  "))
        return httpx.Response(200, content=_sse("bien"))

    p = _proxy(handler)
    assert await _juntar(p.completar_stream([_ruta("a:free"), _ruta("b:free")],
                                            CUERPO, 0.0)) == "bien"


# --- Fix round 4, N2: el guard de "chunk sin nada util" solo miraba dentro de
#     `delta`. En el protocolo real de OpenAI, `finish_reason` es HERMANO de
#     `delta` y el chunk de `usage` viene con `choices: []`, asi que los dos
#     se estaban descartando en silencio. No mordia con Kilo ni OpenRouter
#     porque ambos meten `role` en cada delta, pero si muerde con cualquier
#     proveedor estricto -- el dialecto OpenAI de MiniMax, o los Groq/Cerebras
#     que el diseno planea sumar. Perdida silenciosa de datos en un contrato
#     cuya premisa entera es "cambia solo base_url". ---

SOBRE = '"id":"c1","object":"chat.completion.chunk","created":1,"model":"m"'


async def test_no_pierde_el_chunk_final_real_de_openai():
    cuerpo = (b'data: {' + SOBRE.encode() +
              b',"choices":[{"index":0,"delta":{"role":"assistant","content":"hola"}}]}\n\n'
              b'data: {' + SOBRE.encode() +
              b',"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
              b'data: [DONE]\n\n')
    p = _proxy(lambda req: httpx.Response(200, content=cuerpo))
    lineas = [l async for l in p.completar_stream([_ruta()], CUERPO, 0.0)]
    assert any('"finish_reason": "stop"' in l or '"finish_reason":"stop"' in l
               for l in lineas), "se perdio el chunk de finish_reason"


async def test_no_pierde_el_chunk_de_usage():
    # stream_options.include_usage manda un chunk final con choices vacio.
    cuerpo = (b'data: {' + SOBRE.encode() +
              b',"choices":[{"index":0,"delta":{"content":"hola"}}]}\n\n'
              b'data: {' + SOBRE.encode() +
              b',"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1,'
              b'"total_tokens":4}}\n\n'
              b'data: [DONE]\n\n')
    p = _proxy(lambda req: httpx.Response(200, content=cuerpo))
    lineas = [l async for l in p.completar_stream([_ruta()], CUERPO, 0.0)]
    assert any("total_tokens" in l for l in lineas), "se perdio el chunk de usage"


async def test_el_sobre_repetido_no_convierte_en_util_a_un_chunk_de_razonamiento():
    # La contracara: si "trae algo mas que content" se midiera sobre el chunk
    # entero, las claves de sobre (id/object/created/model/index) -- que se
    # repiten IDENTICAS en cada chunk -- harian que todo chunk de razonamiento
    # ya recortado pareciera util. El stream de puro razonamiento dejaria de
    # hacer failover (regresion de B1) y el buffer se llenaria al pedo.
    razonar = b"".join(
        b'data: {' + SOBRE.encode() +
        b',"choices":[{"index":0,"delta":{"content":"' + t + b'"}}]}\n\n'
        for t in (b"<think>", b"pienso ", b"y pienso", b"</think>"))
    llamadas = []

    def handler(req):
        llamadas.append(1)
        if len(llamadas) == 1:
            return httpx.Response(200, content=razonar + b"data: [DONE]\n\n")
        return httpx.Response(200, content=_sse("bien"))

    p = _proxy(handler)
    texto = await _juntar(p.completar_stream([_ruta("a:free"), _ruta("b:free")],
                                             CUERPO, 0.0))
    assert texto == "bien"
    assert len(llamadas) == 2
    filas = p.almacen._con.execute(
        "SELECT clave, ok FROM eventos ORDER BY clave").fetchall()
    assert filas == [("kilo/a:free", 0), ("kilo/b:free", 1)]


# --- Fix round 4, Minor: descartar chunks retenidos deja de ser silencioso. ---

async def test_avisa_cuando_descarta_los_chunks_retenidos_de_un_intento_fallido(caplog):
    vacio = (b'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n'
             b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}\n\n'
             b'data: [DONE]\n\n')
    llamadas = []

    def handler(req):
        llamadas.append(1)
        if len(llamadas) == 1:
            return httpx.Response(200, content=vacio)
        return httpx.Response(200, content=_sse("bien"))

    p = _proxy(handler)
    with caplog.at_level(logging.INFO, logger="llm_libre.proxy"):
        texto = await _juntar(p.completar_stream([_ruta("a:free"), _ruta("b:free")],
                                                 CUERPO, 0.0))
    assert texto == "bien"
    assert "kilo/a:free" in caplog.text
    assert "descarta" in caplog.text


async def test_avisa_cuando_la_retencion_se_desborda(caplog):
    # Mas de TOPE_PENDIENTES chunks sin contenido: se sueltan y el intento ya
    # no puede hacer failover limpio. Eso tiene que quedar dicho.
    from llm_libre.proxy import TOPE_PENDIENTES
    ruido = b"".join(
        b'data: {"choices":[{"index":0,"delta":{"reasoning":"x"}}]}\n\n'
        for _ in range(TOPE_PENDIENTES + 5))
    p = _proxy(lambda req: httpx.Response(200, content=ruido + b"data: [DONE]\n\n"))
    with caplog.at_level(logging.INFO, logger="llm_libre.proxy"):
        [l async for l in p.completar_stream([_ruta("a:free")], CUERPO, 0.0)]
    assert "kilo/a:free" in caplog.text
