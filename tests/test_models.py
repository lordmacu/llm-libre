from llm_libre.models import NEUTRAL_METRICS, Capabilities, Route, RouteRequest


def test_a_routes_key_joins_provider_and_model():
    r = Route("kilo", "poolside/laguna-s-2.1:free", "free",
              Capabilities(tools=True, vision=False, context=262144, max_output=32768))
    assert r.key == "kilo/poolside/laguna-s-2.1:free"


def test_a_route_without_a_declared_priority_lands_last_by_default():
    r = Route("kilo", "model:free", "free",
              Capabilities(tools=True, vision=False, context=1000, max_output=100))
    assert r.priority == 100


def test_the_priority_can_be_declared_explicitly():
    r = Route("chatgpt", "gpt-5-3-mini", "free",
              Capabilities(tools=False, vision=False, context=128000, max_output=8192),
              priority=0)
    assert r.priority == 0


def test_a_default_request_is_balanced_and_allows_paid():
    p = RouteRequest()
    assert p.profile == "balanced"
    assert p.allow_paid is True
    assert p.model is None


def test_the_neutral_metrics_are_not_in_cooldown():
    assert NEUTRAL_METRICS.cooldown_until == 0.0
    assert 0.0 < NEUTRAL_METRICS.quality <= 1.0


def test_as_wire_emits_the_spanish_keys_the_error_bodies_use():
    """The 400/503 bodies serialise this, so these keys are the HTTP contract --
    see tests/test_wire_contract.py, which asserts them end to end."""
    wire = RouteRequest(model="x", needs_tools=True, min_context=1000).as_wire()
    assert set(wire) == {"model", "needs_tools", "needs_vision", "needs_images",
                         "needs_audio_speech", "needs_audio_transcription",
                         "needs_translate", "needs_search", "min_context", "profile",
                         "allow_paid"}
    assert wire["model"] == "x"
    assert wire["needs_tools"] is True
    assert wire["min_context"] == 1000


def test_the_new_capability_axes_default_to_false():
    # Same shape `images` took: every existing construction site keeps working
    # unchanged, and a provider has to CLAIM a capability to get it.
    c = Capabilities(tools=True, vision=False, context=1000, max_output=100)
    assert c.audio_speech is False
    assert c.audio_transcription is False
    assert c.translate is False
    assert c.search is False
