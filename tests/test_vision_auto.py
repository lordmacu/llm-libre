"""Que una imagen en el cuerpo baste para exigir una ruta con vision, sin que
el cliente tenga que avisar con x_requiere ni con el alias auto:vision."""
from llm_libre.api import _has_image, parse_request
from llm_libre.models import Capabilities, RouteRequest, Route
from llm_libre.router import order_routes

IMG = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR"}}


def r(modelo, vision, provider="kilo"):
    return Route(provider=provider, model_id=modelo, tier="gratis", priority=1,
                capabilities=Capabilities(tools=True, vision=vision, context=100000,
                                        max_output=4096))


def test_texto_suelto_no_pide_vision():
    assert _has_image({"messages": [{"role": "user", "content": "hola"}]}) is False


def test_una_parte_image_url_pide_vision():
    assert _has_image({"messages": [{"role": "user", "content": [
        {"type": "text", "text": "que ves?"}, IMG]}]}) is True


def test_tambien_el_formato_input_image():
    assert _has_image({"messages": [{"role": "user", "content": [
        {"type": "input_image", "image_url": "http://x/y.png"}]}]}) is True


def test_la_imagen_en_un_turno_viejo_tambien_cuenta():
    assert _has_image({"messages": [
        {"role": "user", "content": [IMG]},
        {"role": "assistant", "content": "veo un gato"},
        {"role": "user", "content": "de que color es?"}]}) is True


def test_cuerpo_malformado_no_revienta():
    for c in ({}, {"messages": "hola"}, {"messages": [None, 3]},
              {"messages": [{"content": [None, "x"]}]}, {"messages": None}):
        assert _has_image(c) is False


def test_la_imagen_saca_las_rutas_sin_vision():
    rutas = [r("ciego", vision=False), r("ve", vision=True)]
    pedido = parse_request({"model": "auto", "messages": [
        {"role": "user", "content": [{"type": "text", "text": "?"}, IMG]}]})
    assert pedido.needs_vision is True
    assert [x.key for x in order_routes(rutas, {}, pedido, 0.0)] == ["kilo/ve"]


def test_sin_imagen_las_ciegas_siguen_sirviendo():
    rutas = [r("ciego", vision=False), r("ve", vision=True)]
    pedido = parse_request({"model": "auto",
                                 "messages": [{"role": "user", "content": "hola"}]})
    assert pedido.needs_vision is False
    assert len(order_routes(rutas, {}, pedido, 0.0)) == 2


def test_tools_en_el_cuerpo_ya_exigia_tools():
    """Ya funcionaba; se fija para que no se rompa al tocar lo de al lado."""
    pedido = parse_request({"model": "auto", "messages": [{"role": "user", "content": "x"}],
                                 "tools": [{"type": "function", "function": {"name": "f"}}]})
    assert pedido.needs_tools is True
