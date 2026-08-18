"""That an image in the body is enough to require a vision-capable route, with
no need for the client to announce it via x_requiere or the auto:vision alias."""
from llm_libre.api import _has_image, parse_request
from llm_libre.models import Capabilities, Route
from llm_libre.router import order_routes

IMG = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR"}}


def r(model, vision, provider="kilo"):
    return Route(provider=provider, model_id=model, tier="free", priority=1,
                 capabilities=Capabilities(tools=True, vision=vision, context=100000,
                                           max_output=4096))


def test_plain_text_does_not_require_vision():
    assert _has_image({"messages": [{"role": "user", "content": "hola"}]}) is False


def test_an_image_url_part_requires_vision():
    assert _has_image({"messages": [{"role": "user", "content": [
        {"type": "text", "text": "que ves?"}, IMG]}]}) is True


def test_the_input_image_format_counts_too():
    assert _has_image({"messages": [{"role": "user", "content": [
        {"type": "input_image", "image_url": "http://x/y.png"}]}]}) is True


def test_an_image_in_an_older_turn_counts_as_well():
    assert _has_image({"messages": [
        {"role": "user", "content": [IMG]},
        {"role": "assistant", "content": "veo un gato"},
        {"role": "user", "content": "de que color es?"}]}) is True


def test_a_malformed_body_does_not_blow_up():
    for c in ({}, {"messages": "hola"}, {"messages": [None, 3]},
              {"messages": [{"content": [None, "x"]}]}, {"messages": None}):
        assert _has_image(c) is False


def test_an_image_rules_out_the_routes_without_vision():
    routes = [r("ciego", vision=False), r("ve", vision=True)]
    request = parse_request({"model": "auto", "messages": [
        {"role": "user", "content": [{"type": "text", "text": "?"}, IMG]}]})
    assert request.needs_vision is True
    assert [x.key for x in order_routes(routes, {}, request, 0.0)] == ["kilo/ve"]


def test_without_an_image_the_blind_routes_still_serve():
    routes = [r("ciego", vision=False), r("ve", vision=True)]
    request = parse_request({"model": "auto",
                             "messages": [{"role": "user", "content": "hola"}]})
    assert request.needs_vision is False
    assert len(order_routes(routes, {}, request, 0.0)) == 2


def test_tools_in_the_body_already_required_tools():
    """This already worked; it is pinned so it does not break while the code
    next to it is being touched."""
    request = parse_request({"model": "auto", "messages": [{"role": "user", "content": "x"}],
                             "tools": [{"type": "function", "function": {"name": "f"}}]})
    assert request.needs_tools is True
