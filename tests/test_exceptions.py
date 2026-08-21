"""Per-model exceptions over default_capabilities.

A discovered catalogue need not be homogeneous: grok publishes 31 ids of which
25 make tool calls and 6 do not (measured 2026-08-18).
"""
import os

from llm_libre.catalog import normalize
from llm_libre.models import Capabilities
from llm_libre.providers import load

DEFAULTS = Capabilities(tools=True, vision=False, context=128000, max_output=8192)
CATALOGUE = {"data": [{"id": "good"}, {"id": "no-tools"}, {"id": "other"}]}


def _by_id(routes):
    return {r.model_id: r.capabilities for r in routes}


def test_without_exceptions_everything_inherits_the_defaults():
    caps = _by_id(normalize("grok", CATALOGUE, 1, DEFAULTS))
    assert all(c.tools for c in caps.values())


def test_an_exception_overrides_only_the_declared_field():
    caps = _by_id(normalize("grok", CATALOGUE, 1, DEFAULTS,
                            {"no-tools": {"tools": False}}))
    assert caps["no-tools"].tools is False
    assert caps["good"].tools is True
    # whatever is NOT declared is inherited whole
    assert caps["no-tools"].context == 128000
    assert caps["no-tools"].max_output == 8192
    assert caps["no-tools"].vision is False


def test_an_exception_for_an_id_that_does_not_exist_is_harmless():
    caps = _by_id(normalize("grok", CATALOGUE, 1, DEFAULTS,
                            {"fantasma": {"tools": False}}))
    assert len(caps) == 3 and all(c.tools for c in caps.values())


def test_several_fields_can_be_overridden_at_once():
    caps = _by_id(normalize("grok", CATALOGUE, 1, DEFAULTS,
                            {"other": {"tools": False, "vision": True}}))
    assert caps["other"].tools is False and caps["other"].vision is True


def test_the_yaml_declares_groks_defaults_and_no_longer_pins_exceptions():
    provs = load("providers.yaml", dict(os.environ))
    grok = [p for p in provs if p.id == "grok"][0]
    assert grok.default_capabilities.tools is True
    # vision True: the proxy resolves the image on its own (resolve_vision_model
    # sends whatever cannot see to the VISION_POOL), and 30 of 31 routes read a
    # 4-digit code from a 488x232 image.
    assert grok.default_capabilities.vision is True
    # The imagine-agent-mode family (image-generation agents, 0/3 on tool_calls
    # when repeated, no vision per grok_backend's own docs) used to be pinned
    # here by hand with an `imagine-agent-mode*: {tools, vision, images}`
    # exception. Retired 2026-08-20: grok reads the capability contract now
    # (reads_capabilities is True, below), and its /v1/models publishes
    # exactly that {tools, vision, images} per model, so the override is gone
    # rather than merely redundant -- `exceptions` outranks the contract, and
    # a hand-written entry here would have survived it if grok ever changed
    # what these models do. See the retirement note on `exceptions` in
    # providers.yaml.
    assert grok.reads_capabilities is True
    assert grok.exceptions == {}


def test_minimax_declares_measured_vision():
    from llm_libre.providers import fixed_routes
    provs = load("providers.yaml", {"MINIMAX_API_KEY": "x"})
    mm = [r for p in provs if p.id == "minimax" for r in fixed_routes(p)]
    assert mm and mm[0].capabilities.vision is True
