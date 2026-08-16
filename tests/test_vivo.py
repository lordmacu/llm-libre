import httpx
import pytest

pytestmark = pytest.mark.vivo


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
