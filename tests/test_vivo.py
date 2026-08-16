import os

import httpx
import pytest

pytestmark = pytest.mark.vivo


async def test_chatgpt_proxy_responde_un_chat_real_si_esta_configurado():
    # Se salta limpio si CHATGPT_PROXY_URL no esta seteada: este proxy es un
    # servicio propio (blog), no siempre esta arriba, y a diferencia de Kilo
    # /OpenRouter no hay una URL publica fija contra la que pegar siempre.
    url = os.getenv("CHATGPT_PROXY_URL")
    if not url:
        pytest.skip("CHATGPT_PROXY_URL no esta configurada")
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(url.rstrip("/") + "/v1/chat/completions",
                         json={"model": "gpt-5-3-mini",
                               "messages": [{"role": "user", "content": "di hola"}]})
    assert r.status_code == 200, "chatgpt-proxy dejo de responder chat anonimo"
    datos = r.json()
    contenido = datos["choices"][0]["message"]["content"]
    assert isinstance(contenido, str) and contenido.strip()


async def test_kilo_sigue_aceptando_peticiones_anonimas():
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post("https://api.kilo.ai/api/gateway/chat/completions",
                         json={"model": "kilo-auto/free", "max_tokens": 8,
                               "messages": [{"role": "user", "content": "ping"}]})
    assert r.status_code == 200, "el tier anonimo de Kilo dejo de funcionar"


async def test_el_catalogo_de_kilo_trae_modelos_gratis_con_tools():
    from llm_libre.catalogo import normalizar
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.get("https://api.kilo.ai/api/gateway/models")
    rutas = normalizar("kilo", r.json())
    assert any(x.capacidades.tools for x in rutas)
