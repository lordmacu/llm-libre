import os
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.vivo

YAML = str(Path(__file__).resolve().parents[1] / "proveedores.yaml")

# Ids que NUNCA deben aparecer en el catalogo descubierto: los alias legacy
# que chatgpt-proxy agrega para compatibilidad, y "auto", reservado por
# llm-libre mismo (colisiona con su propio alias en interpretar_pedido).
_IDS_QUE_NO_DEBERIAN_APARECER = {"gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo", "auto"}


async def test_chatgpt_proxy_responde_un_chat_real_si_esta_configurado():
    # Se salta limpio si CHATGPT_PROXY_URL no esta seteada: este proxy es un
    # servicio propio (blog), no siempre esta arriba, y a diferencia de Kilo
    # /OpenRouter no hay una URL publica fija contra la que pegar siempre.
    #
    # `base_url` se resuelve igual que en produccion (proveedores.cargar),
    # no se reconstruye la URL a mano aca: asi este test usa exactamente el
    # mismo camino (incluido el /v1 de base_url) que el gateway real.
    url = os.getenv("CHATGPT_PROXY_URL")
    if not url:
        pytest.skip("CHATGPT_PROXY_URL no esta configurada")
    from llm_libre.catalogo import normalizar
    from llm_libre.proveedores import cargar

    chatgpt = next(p for p in cargar(YAML, {"CHATGPT_PROXY_URL": url}) if p.id == "chatgpt")

    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(chatgpt.base_url.rstrip("/") + "/chat/completions",
                         json={"model": "gpt-5-3-mini",
                               "messages": [{"role": "user", "content": "di hola"}]})
    assert r.status_code == 200, "chatgpt-proxy dejo de responder chat anonimo"
    datos = r.json()
    contenido = datos["choices"][0]["message"]["content"]
    assert isinstance(contenido, str) and contenido.strip()

    # El catalogo ahora es DESCUBIERTO (ver follow-up de Task 13): confirma
    # contra el proxy real que sigue trayendo algo utilizable y que los
    # alias/el id reservado siguen sin colarse.
    async with httpx.AsyncClient(timeout=60) as c:
        r_modelos = await c.get(chatgpt.base_url.rstrip("/") + chatgpt.modelos_path)
    assert r_modelos.status_code == 200, "chatgpt-proxy dejo de responder /v1/models"
    rutas = normalizar("chatgpt", r_modelos.json(), prioridad=chatgpt.prioridad,
                       capacidades_por_defecto=chatgpt.capacidades_por_defecto)
    assert rutas, "el catalogo descubierto de chatgpt-proxy vino vacio"
    ids = {x.modelo_id for x in rutas}
    assert not ids & _IDS_QUE_NO_DEBERIAN_APARECER
    assert all(x.capacidades.tools is False for x in rutas)


async def test_chatgpt_proxy_no_hace_function_calling_de_verdad():
    # El hecho que sostiene tools:false en proveedores.yaml: el usuario
    # reporto "ya tenemos los tools habilitados" y esto se verifico
    # ejecutando (no releyendo) -- el proxy ya NO devuelve HTTP 500 al
    # mandarle tools, pero con tool_choice:"required" sigue devolviendo
    # tool_calls:None y prosa. Hasta ahora ese hecho solo se habia
    # verificado a mano; este test lo deja ejecutable contra el proxy real,
    # asi no se desactualiza en silencio si el backend cambia de
    # comportamiento.
    url = os.getenv("CHATGPT_PROXY_URL")
    if not url:
        pytest.skip("CHATGPT_PROXY_URL no esta configurada")
    from llm_libre.bateria import HERRAMIENTA
    from llm_libre.proveedores import cargar

    chatgpt = next(p for p in cargar(YAML, {"CHATGPT_PROXY_URL": url}) if p.id == "chatgpt")

    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(chatgpt.base_url.rstrip("/") + "/chat/completions",
                         json={"model": "gpt-5-3-mini",
                               "messages": [{"role": "user",
                                            "content": "Que clima hace en Bogota?"}],
                               "tools": [HERRAMIENTA], "tool_choice": "required"})
    # Mandarle tools ya no revienta (eso cambio, y esta bien): lo que se
    # verifica es que la RESPUESTA sigue sin ser function calling de
    # verdad -- si esto alguna vez empieza a devolver tool_calls, hay que
    # revisar si tools:false sigue siendo necesario.
    assert r.status_code == 200, "chatgpt-proxy revento al mandarle tools (volvio el 500 viejo?)"
    msg = r.json()["choices"][0]["message"]
    assert not msg.get("tool_calls"), (
        "chatgpt-proxy devolvio tool_calls: si esto paso, el backend anonimo "
        "empezo a soportar function calling de verdad y tools:false en "
        "proveedores.yaml deberia reconsiderarse")
    assert isinstance(msg.get("content"), str) and msg["content"].strip()


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
