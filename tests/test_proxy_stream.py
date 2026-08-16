import json

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
