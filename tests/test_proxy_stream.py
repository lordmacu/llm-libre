import asyncio
import json
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
    cuerpo = (b'data: {"choices":[{"delta":{"content":"","finish_reason":"stop"}}]}\n\n'
             b'data: [DONE]\n\n')
    p = _proxy(lambda req: httpx.Response(200, content=cuerpo))
    lineas = [l async for l in p.completar_stream([_ruta()], CUERPO, 0.0)]
    utiles = [l for l in lineas if "[DONE]" not in l]
    assert len(utiles) == 1
    obj = json.loads(utiles[0][len("data: "):])
    assert obj["choices"][0]["delta"]["finish_reason"] == "stop"


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


async def test_stream_de_puro_razonamiento_registra_un_evento_exitoso():
    # Ningun chunk sobrevive al filtro (todo es <think>...</think> cerrado, sin
    # tool_calls/finish_reason/role) asi que el `if not evento_registrado` de
    # dentro del bucle nunca se dispara. Igual debe quedar UN evento con ok=1:
    # la llamada HTTP si funciono.
    p = _proxy(lambda req: httpx.Response(
        200, content=_sse("<think>solo razonamiento</think>")))
    texto = await _juntar(p.completar_stream([_ruta()], CUERPO, 0.0))
    assert texto == ""
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
