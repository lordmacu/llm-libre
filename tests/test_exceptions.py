"""Per-model exceptions over default_capabilities.

A discovered catalogue need not be homogeneous: grok publishes 31 ids of which
25 make tool calls and 6 do not (measured 2026-08-18).
"""
import os

from llm_libre.catalog import normalize
from llm_libre.models import Capabilities
from llm_libre.providers import load

DEFAULTS = Capabilities(tools=True, vision=False, context=128000, max_output=8192)
CATALOGUE = {"data": [{"id": "bueno"}, {"id": "sin-tools"}, {"id": "otro"}]}


def _by_id(routes):
    return {r.model_id: r.capabilities for r in routes}


def test_without_exceptions_everything_inherits_the_defaults():
    caps = _by_id(normalize("grok", CATALOGUE, 1, DEFAULTS))
    assert all(c.tools for c in caps.values())


def test_an_exception_overrides_only_the_declared_field():
    caps = _by_id(normalize("grok", CATALOGUE, 1, DEFAULTS,
                            {"sin-tools": {"tools": False}}))
    assert caps["sin-tools"].tools is False
    assert caps["bueno"].tools is True
    # whatever is NOT declared is inherited whole
    assert caps["sin-tools"].context == 128000
    assert caps["sin-tools"].max_output == 8192
    assert caps["sin-tools"].vision is False


def test_an_exception_for_an_id_that_does_not_exist_is_harmless():
    caps = _by_id(normalize("grok", CATALOGUE, 1, DEFAULTS,
                            {"fantasma": {"tools": False}}))
    assert len(caps) == 3 and all(c.tools for c in caps.values())


def test_several_fields_can_be_overridden_at_once():
    caps = _by_id(normalize("grok", CATALOGUE, 1, DEFAULTS,
                            {"otro": {"tools": False, "vision": True}}))
    assert caps["otro"].tools is False and caps["otro"].vision is True


def test_the_yaml_declares_groks_exceptions():
    provs = load("proveedores.yaml", dict(os.environ))
    grok = [p for p in provs if p.id == "grok"][0]
    assert grok.default_capabilities.tools is True
    # vision True: the proxy resolves the image on its own (resolve_vision_model
    # sends whatever cannot see to the VISION_POOL), and 30 of 31 routes read a
    # 4-digit code from a 488x232 image.
    assert grok.default_capabilities.vision is True
    # Only the imagine-agent-mode family is left out: they are image-generation
    # agents, 0/3 on tool_calls when repeated, and grok_backend itself documents
    # that they have no vision.
    assert set(grok.exceptions) == {
        "imagine-agent-mode", "imagine-agent-mode-dev", "imagine-agent-mode-grok-4-5"}
    assert all(v == {"tools": False, "vision": False} for v in grok.exceptions.values())


def test_minimax_declares_measured_vision():
    from llm_libre.providers import fixed_routes
    provs = load("proveedores.yaml", {"MINIMAX_API_KEY": "x"})
    mm = [r for p in provs if p.id == "minimax" for r in fixed_routes(p)]
    assert mm and mm[0].capabilities.vision is True
