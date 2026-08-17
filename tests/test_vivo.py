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


def _proveedor_chatgpt(url: str):
    # `base_url` se resuelve igual que en produccion (proveedores.cargar), no
    # se reconstruye la URL a mano: asi los tests usan exactamente el mismo
    # camino (incluida la normalizacion de base_url_env) que el gateway real.
    from llm_libre.proveedores import cargar
    return next(p for p in cargar(YAML, {"CHATGPT_PROXY_URL": url}) if p.id == "chatgpt")


async def _rutas_chatgpt_descubiertas(chatgpt) -> list:
    """Descubre el catalogo real de chatgpt-proxy -- NUNCA un id cableado a
    mano, que es justo la regla que esta Task existe para hacer cumplir.
    Compartido por los dos tests de abajo para que ninguno tenga que
    adivinar un modelo: la revision encontro que un id hardcodeado
    ("gpt-5-3-mini") que un dia deja de existir hace fallar el test con un
    diagnostico ERRADO ("revento al mandarle tools, ¿volvio el 500 viejo?")
    en vez de decir lo que de verdad paso (el id ya no existe)."""
    from llm_libre.catalogo import normalizar
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.get(chatgpt.base_url.rstrip("/") + chatgpt.modelos_path)
    assert r.status_code == 200, "chatgpt-proxy dejo de responder /v1/models"
    rutas = normalizar("chatgpt", r.json(), prioridad=chatgpt.prioridad,
                       capacidades_por_defecto=chatgpt.capacidades_por_defecto)
    assert rutas, "el catalogo descubierto de chatgpt-proxy vino vacio"
    return rutas


async def test_chatgpt_proxy_responde_un_chat_real_si_esta_configurado():
    # Se salta limpio si CHATGPT_PROXY_URL no esta seteada: este proxy es un
    # servicio propio (blog), no siempre esta arriba, y a diferencia de Kilo
    # /OpenRouter no hay una URL publica fija contra la que pegar siempre.
    url = os.getenv("CHATGPT_PROXY_URL")
    if not url:
        pytest.skip("CHATGPT_PROXY_URL no esta configurada")
    chatgpt = _proveedor_chatgpt(url)

    # El catalogo es DESCUBIERTO (ver follow-up de Task 13): confirma contra
    # el proxy real que sigue trayendo algo utilizable y que los alias/el id
    # reservado siguen sin colarse.
    rutas = await _rutas_chatgpt_descubiertas(chatgpt)
    ids = {x.modelo_id for x in rutas}
    assert not ids & _IDS_QUE_NO_DEBERIAN_APARECER
    assert all(x.capacidades.tools is False for x in rutas)

    # El id para el chat sale del catalogo QUE ACABA DE DESCUBRIR, no de una
    # constante en este archivo -- el mismo principio de "nada de ids
    # cableados" que rige todo proveedores.yaml.
    modelo_id = sorted(ids)[0]
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(chatgpt.base_url.rstrip("/") + "/chat/completions",
                         json={"model": modelo_id,
                               "messages": [{"role": "user", "content": "di hola"}]})
    assert r.status_code == 200, f"chatgpt-proxy dejo de responder chat anonimo ({modelo_id})"
    datos = r.json()
    contenido = datos["choices"][0]["message"]["content"]
    assert isinstance(contenido, str) and contenido.strip()


async def test_chatgpt_proxy_no_hace_function_calling_de_verdad():
    # El hecho que sostiene tools:false en proveedores.yaml: el usuario
    # reporto "ya tenemos los tools habilitados" y esto se verifico
    # ejecutando (no releyendo) -- el proxy ya NO devuelve HTTP 500 al
    # mandarle tools, pero con tool_choice:"required" sigue devolviendo
    # tool_calls:None y prosa.
    #
    # tools:false es una afirmacion sobre TODAS las rutas descubiertas de
    # chatgpt, no sobre un modelo puntual (hallazgo de la revision): un
    # backend que ganara function calling en un modelo distinto del
    # probado dejaria este test en verde con el YAML ya mintiendo. Se
    # prueban TODOS los ids que el catalogo real trajo, no uno solo.
    url = os.getenv("CHATGPT_PROXY_URL")
    if not url:
        pytest.skip("CHATGPT_PROXY_URL no esta configurada")
    from llm_libre.bateria import HERRAMIENTA

    chatgpt = _proveedor_chatgpt(url)
    rutas = await _rutas_chatgpt_descubiertas(chatgpt)

    for ruta in rutas:
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(chatgpt.base_url.rstrip("/") + "/chat/completions",
                             json={"model": ruta.modelo_id,
                                   "messages": [{"role": "user",
                                                "content": "Que clima hace en Bogota?"}],
                                   "tools": [HERRAMIENTA], "tool_choice": "required"})
        # Mandarle tools ya no revienta (eso cambio, y esta bien): lo que se
        # verifica es que la RESPUESTA sigue sin ser function calling de
        # verdad. El mensaje distingue las dos formas de fallar -- el status
        # code real dice si volvio el 500 viejo, si el id dejo de existir
        # (404), o algo distinto -- en vez de asumir una sola causa.
        assert r.status_code == 200, (
            f"chatgpt-proxy devolvio HTTP {r.status_code} para {ruta.modelo_id} al "
            "mandarle tools (revisar si volvio el 500 viejo o si el id ya no existe)")
        msg = r.json()["choices"][0]["message"]
        assert not msg.get("tool_calls"), (
            f"chatgpt-proxy devolvio tool_calls para {ruta.modelo_id}: si esto paso, el "
            "backend anonimo empezo a soportar function calling de verdad en ese modelo "
            "y tools:false en proveedores.yaml deberia reconsiderarse")
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
