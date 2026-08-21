import json
import logging
from pathlib import Path

from llm_libre.catalog import SUSTAINED_RATE_FLOOR, normalize
from llm_libre.contract import (REQUIRED_CAPABILITIES, Auth, ProviderContract,
                                parse_health)
from llm_libre.models import Capabilities
from llm_libre.providers import load

FIXTURES = Path(__file__).parent / "fixtures"
PROVIDERS_YAML = str(Path(__file__).resolve().parents[1] / "providers.yaml")

_CHATGPT_DEFAULTS = Capabilities(tools=False, vision=False, context=128000, max_output=8192)


def _load(name):
    return json.loads((FIXTURES / name).read_text())


def test_paid_models_are_discarded():
    data = {"data": [
        {"id": "expensive/model", "pricing": {"prompt": "0.0000015"},
         "architecture": {"output_modalities": ["text"]}},
    ]}
    assert normalize("kilo", data) == []


def test_free_models_that_do_not_return_text_only_are_discarded():
    # google/lyria-* is priced at 0 but is a MUSIC model:
    # output_modalities = ["text", "audio"]. Filtering by price is not enough.
    data = {"data": [
        {"id": "google/lyria-3-pro-preview", "pricing": {"prompt": "0"},
         "context_length": 1048576,
         "architecture": {"input_modalities": ["text", "image"],
                          "output_modalities": ["text", "audio"]},
         "supported_parameters": ["max_tokens"]},
    ]}
    assert normalize("kilo", data) == []


def test_normalize_stamps_the_providers_priority():
    data = {"data": [
        {"id": "without/tools:free", "pricing": {"prompt": "0"}, "context_length": 4096,
         "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
         "supported_parameters": ["max_tokens"]},
    ]}
    routes = normalize("kilo", data, priority=1)
    assert routes[0].priority == 1


def test_normalize_without_a_declared_priority_uses_the_default_of_one_hundred():
    data = {"data": [
        {"id": "without/tools:free", "pricing": {"prompt": "0"}, "context_length": 4096,
         "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
         "supported_parameters": ["max_tokens"]},
    ]}
    assert normalize("kilo", data)[0].priority == 100


def test_multimodal_input_is_accepted_as_long_as_output_is_text():
    data = {"data": [
        {"id": "nvidia/nemotron-omni:free", "pricing": {"prompt": "0"},
         "context_length": 256000,
         "architecture": {"input_modalities": ["text", "audio", "image", "video"],
                          "output_modalities": ["text"]},
         "supported_parameters": ["tools", "tool_choice"],
         "top_provider": {"max_completion_tokens": 8192}},
    ]}
    routes = normalize("kilo", data)
    assert len(routes) == 1
    assert routes[0].capabilities.vision is True
    assert routes[0].capabilities.tools is True
    assert routes[0].capabilities.context == 256000
    assert routes[0].capabilities.max_output == 8192
    assert routes[0].tier == "free"


def test_absence_of_tools_is_detected():
    data = {"data": [
        {"id": "without/tools:free", "pricing": {"prompt": "0"}, "context_length": 4096,
         "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
         "supported_parameters": ["max_tokens", "temperature"]},
    ]}
    assert normalize("kilo", data)[0].capabilities.tools is False


def test_an_unreadable_price_is_treated_as_paid():
    data = {"data": [{"id": "odd/model", "pricing": {},
                      "architecture": {"output_modalities": ["text"]}}]}
    assert normalize("kilo", data) == []


def test_against_the_real_kilo_catalogue():
    routes = normalize("kilo", _load("kilo_models.json"))
    assert len(routes) > 5
    assert all(r.tier == "free" for r in routes)
    assert all(r.provider == "kilo" for r in routes)
    assert not any("lyria" in r.model_id for r in routes)
    assert any(r.capabilities.tools for r in routes)


def test_against_the_real_openrouter_catalogue():
    routes = normalize("openrouter", _load("openrouter_models.json"))
    assert len(routes) > 5
    assert not any("lyria" in r.model_id for r in routes)


# --- Fix round 3, B2 (Blocking): in a fresh install EVERY route starts at
#     neutral quality, so the only discriminator is ttft -- and the fastest
#     thing in the pool is a safety classifier that answers "User Safety: safe"
#     to anything. It is discarded by what the PROVIDER ITSELF publishes about
#     it in /models (name/description), which the normaliser used to throw away.
#     Still discovery: not a single hardcoded id. ---

def _model(mid, name="A Model", description="A general chat model."):
    return {"id": mid, "name": name, "description": description,
            "pricing": {"prompt": "0"}, "context_length": 4096,
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "supported_parameters": ["tools"],
            "top_provider": {"max_completion_tokens": 4096}}


def test_a_guardrail_is_discarded_by_its_own_description():
    data = {"data": [_model(
        "vendor/whatever:free", "Vendor: Whatever (free)",
        "A compact 4B-parameter multimodal guardrail model. It moderates both "
        "inputs to and responses from LLMs.")]}
    assert normalize("kilo", data) == []


def test_a_classifier_and_a_reranker_are_discarded_by_their_description():
    data = {"data": [
        _model("v/a:free", "V: A", "A safety classifier for prompts."),
        _model("v/b:free", "V: B", "A reranker for retrieval pipelines."),
        _model("v/c:free", "V: C", "A text embeddings model."),
    ]}
    assert normalize("kilo", data) == []


def test_meta_routers_are_discarded_by_their_own_description():
    # Not a model: a lottery between other models. Scoring them in the ranking
    # measures a roulette wheel, and they hide which route actually served.
    data = {"data": [
        _model("kilo-auto/free", "Auto Free",
               "Rotates through available free models. Limited capability."),
        _model("openrouter/free", "Free Models Router",
               "The simplest way to get free inference. openrouter/free is a "
               "router that selects free models at random."),
    ]}
    assert normalize("kilo", data) == []


def test_an_ordinary_chat_model_is_not_discarded():
    data = {"data": [_model(
        "nvidia/nemotron-3-super:free", "NVIDIA: Nemotron 3 Super (free)",
        "A 120B-parameter open hybrid MoE model, activating just 12B parameters "
        "for maximum compute efficiency and accuracy in complex multi-agent "
        "applications.")]}
    assert [r.model_id for r in normalize("kilo", data)] == ["nvidia/nemotron-3-super:free"]


def test_a_model_without_name_or_description_is_not_discarded():
    # Both fields are optional: a provider that does not publish them must not
    # lose its entire catalogue.
    data = {"data": [{"id": "pelado:free", "pricing": {"prompt": "0"},
                      "context_length": 4096,
                      "architecture": {"output_modalities": ["text"]}}]}
    assert [r.model_id for r in normalize("kilo", data)] == ["pelado:free"]


def test_the_real_kilo_catalogue_no_longer_carries_the_classifier_or_meta_routers():
    ids = {r.model_id for r in normalize("kilo", _load("kilo_models.json"))}
    assert "nvidia/nemotron-3.5-content-safety:free" not in ids
    assert "kilo-auto/free" not in ids
    assert "openrouter/free" not in ids
    # ...and the real chat models are still there.
    assert "nvidia/nemotron-3-super-120b-a12b:free" in ids
    assert "poolside/laguna-s-2.1:free" in ids
    assert len(ids) == 11


def test_the_real_openrouter_catalogue_no_longer_carries_the_classifier_or_meta_router():
    ids = {r.model_id for r in normalize("openrouter", _load("openrouter_models.json"))}
    assert "nvidia/nemotron-3.5-content-safety:free" not in ids
    assert "openrouter/free" not in ids
    assert "google/gemma-4-31b-it:free" in ids
    assert "openai/gpt-oss-20b:free" in ids
    assert len(ids) == 15


# --- Fix round 3, ALSO: `m["id"]` with no guard. A catalogue entry without an
#     id blew up with KeyError; probing.py swallowed the whole thing and THAT
#     provider's catalogue froze forever, without a single log line. ---

def test_an_entry_without_an_id_is_skipped_without_blowing_up_and_is_logged(caplog):
    data = {"data": [
        {"pricing": {"prompt": "0"}, "architecture": {"output_modalities": ["text"]}},
        {"id": "good:free", "pricing": {"prompt": "0"},
         "architecture": {"output_modalities": ["text"]}},
    ]}
    with caplog.at_level(logging.WARNING, logger="llm_libre.catalog"):
        routes = normalize("kilo", data)
    assert [r.model_id for r in routes] == ["good:free"]
    assert "without 'id'" in caplog.text


def test_an_entry_that_is_not_an_object_is_skipped_without_blowing_up(caplog):
    # An auth error served with 200 leaves normalize() iterating over strings.
    with caplog.at_level(logging.WARNING, logger="llm_libre.catalog"):
        assert normalize("kilo", {"error": "unauthorized"}) == []
    assert caplog.records


# --- Task 13 follow-up: chatgpt-proxy became a DISCOVERED catalogue (its
#     /v1/models is now dynamic, with a TTL cache), but it still carries no
#     capability metadata -- only id/object/created/owned_by/description.
#     `default_capabilities` is the GENERAL mechanism for that pattern:
#     discovered ids, capabilities declared once for all of them. ---

def test_chatgpt_discovers_the_five_real_ids_in_the_fixture():
    routes = normalize("chatgpt", _load("chatgpt_models.json"),
                       default_capabilities=_CHATGPT_DEFAULTS)
    assert len(routes) == 5
    assert {r.model_id for r in routes} == {
        "gpt-5-5", "gpt-5-6", "gpt-5-3-mini", "gpt-5-5-mini", "gpt-5-6-mini"}


def test_chatgpt_discards_the_legacy_aliases_by_their_own_description():
    routes = normalize("chatgpt", _load("chatgpt_models.json"),
                       default_capabilities=_CHATGPT_DEFAULTS)
    ids = {r.model_id for r in routes}
    assert not ids & {"gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"}


def test_chatgpt_discards_auto_because_it_is_a_reserved_id():
    routes = normalize("chatgpt", _load("chatgpt_models.json"),
                       default_capabilities=_CHATGPT_DEFAULTS)
    assert "auto" not in {r.model_id for r in routes}


# --- INFO from the round 6 review: RESERVED_IDS only excluded the LITERAL id
#     "auto", not the compound aliases api.py ALSO treats as reserved
#     ("auto:fast", "auto:strong", "auto:tools", "auto:vision" -- see ALIAS
#     in api.py and parse_request, which resolves ANY id starting with
#     "auto:" as an alias, never as a literal id). A provider publishing a real
#     model called, say, "auto:fast" created a PERMANENTLY unreachable route:
#     no request with `model: "auto:fast"` ever reaches
#     `request.model == "auto:fast"`, because parse_request resolves it
#     first as profile "fast" with model=None. ---

def test_the_compound_auto_aliases_are_discarded_too():
    routes = normalize("prov", [
        {"id": "auto:fast", "name": "Collides with the compound alias"},
        {"id": "auto:strong", "name": "Collides with the compound alias"},
        {"id": "auto:tools", "name": "Collides with the compound alias"},
        {"id": "auto:vision", "name": "Collides with the compound alias"},
        {"id": "a-real-model", "name": "Real model"},
    ], default_capabilities=_CHATGPT_DEFAULTS)
    ids = {r.model_id for r in routes}
    assert ids == {"a-real-model"}


def test_a_new_model_from_the_proxy_appears_without_touching_the_yaml():
    # The "con_modelo_nuevo" fixture simulates ChatGPT's real backend adding a
    # model tomorrow (gpt-5-7) that does not exist today: it has to show up on
    # its own, without anyone editing providers.yaml or this test.
    routes = normalize("chatgpt", _load("chatgpt_models_con_modelo_nuevo.json"),
                       default_capabilities=_CHATGPT_DEFAULTS)
    ids = {r.model_id for r in routes}
    assert "gpt-5-7" in ids
    assert len(ids) == 6


def test_the_default_capabilities_are_applied_to_every_discovered_id():
    routes = normalize("chatgpt", _load("chatgpt_models.json"),
                       default_capabilities=_CHATGPT_DEFAULTS)
    assert len(routes) == 5
    assert all(r.capabilities == _CHATGPT_DEFAULTS for r in routes)
    assert all(r.capabilities.tools is False for r in routes)   # mandatory: no function calling


def test_default_capabilities_skip_price_and_output_modality():
    # A provider declaring defaults is asserting what its catalogue CANNOT say:
    # an entry with no "pricing" (normally => "paid", discarded) and with music
    # output_modalities (normally discarded) is accepted anyway, because those
    # two checks are skipped in this mode.
    data = {"data": [
        {"id": "model-without-metadata", "description": "A Model"},
    ]}
    routes = normalize("chatgpt", data, default_capabilities=_CHATGPT_DEFAULTS)
    assert [r.model_id for r in routes] == ["model-without-metadata"]
    assert routes[0].capabilities == _CHATGPT_DEFAULTS


def test_default_capabilities_do_not_disable_the_speciality_filter():
    # Only price/modality are skipped -- what the brief asked for. The speciality
    # filter (guardrails, classifiers, meta-routers) stays active as a defence: a
    # provider with defaults can expose one of those too.
    data = {"data": [
        {"id": "guardrail-1", "description": "A content safety guardrail model."},
    ]}
    assert normalize("chatgpt", data, default_capabilities=_CHATGPT_DEFAULTS) == []


def test_a_provider_without_default_capabilities_keeps_the_original_behaviour():
    # Kilo/OpenRouter declare no defaults: normalize() without that argument (or
    # with an explicit None) has to give EXACTLY what it gave before this change
    # -- price, output modality and speciality still filter as always.
    data = {"data": [
        {"id": "expensive/model", "pricing": {"prompt": "0.0000015"},
         "architecture": {"output_modalities": ["text"]}},
    ]}
    assert normalize("kilo", data, default_capabilities=None) == []
    assert normalize("kilo", data) == []


def test_against_the_real_chatgpt_proxy_catalogue():
    # The full fixture, exactly as /v1/models returns it today (verified
    # 2026-08-16): 10 entries -- 5 real models, 4 legacy aliases, "auto".
    routes = normalize("chatgpt", _load("chatgpt_models.json"),
                       default_capabilities=_CHATGPT_DEFAULTS)
    assert len(routes) == 5
    assert all(r.provider == "chatgpt" for r in routes)
    assert all(r.tier == "free" for r in routes)


# --- Scarce routes go behind their provider's abundant ones, 2026-08-19 -------
#
# Measured against grok-proxy: it publishes 33 models that llm-libre treated as
# 33 interchangeable routes, and their real sustained capacity differs by three
# orders of magnitude. Seventeen of them (the `grok-plugins-*` file agents and
# `imagine-agent-mode*`) carry 999 requests/hour EACH, on independent windows --
# ~17,000/h between them -- while `grok-3` carries 30 per 24h and `grok-4` seven.
# All of them are the same Grok 4.5 underneath.
#
# Treating those alike is expensive in both directions. The quality battery costs
# five requests per route per run, which is 17% of grok-3's ENTIRE daily budget
# for one run, and real traffic then finds it exhausted. Meanwhile the abundant
# pool -- same model, effectively unmetered -- sits idle.
#
# So a route that cannot sustain a request a minute is a RESERVE, not a workhorse:
# it keeps its provider's band but sorts after the abundant ones in it.

def _rated_model(mid, rate=None):
    m = {"id": mid, "pricing": {"prompt": "0"}, "context_length": 1000,
         "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
         "supported_parameters": ["tools"], "top_provider": {"max_completion_tokens": 100}}
    if rate is not None:
        m["requests_per_hour"] = rate
    return m


def test_an_abundant_route_keeps_its_providers_priority():
    routes = normalize("grok", {"data": [_rated_model("grok-plugins-4p6-excel", rate=999)]},
                       priority=0)
    assert routes[0].priority == 0


def test_a_scarce_route_is_demoted_behind_them():
    routes = normalize("grok", {"data": [_rated_model("grok-3", rate=1.25)]}, priority=0)
    assert routes[0].priority == 1


def test_a_route_that_declares_no_rate_is_not_demoted():
    """Every provider that does not publish this field must behave exactly as
    before -- the demotion is opt-in evidence, not an assumption."""
    routes = normalize("kilo", {"data": [_rated_model("x:free")]}, priority=1)
    assert routes[0].priority == 1


def test_the_demotion_is_relative_so_a_scarce_premium_still_leads_the_next_band():
    """grok's scarce routes drop behind grok's abundant ones, NOT behind Kilo:
    they are still premium, just held in reserve."""
    scarce = normalize("grok", {"data": [_rated_model("grok-3", rate=1.25)]}, priority=0)[0]
    kilo = normalize("kilo", {"data": [_rated_model("x:free")]}, priority=2)[0]
    assert scarce.priority < kilo.priority


def test_the_threshold_sits_at_one_request_a_minute():
    """A route that can sustain a request a minute can carry real traffic; one
    that cannot is a reserve. The measured populations are nowhere near this line
    -- 999/h against 1.25/h -- so its exact value is not load-bearing."""
    assert SUSTAINED_RATE_FLOOR == 60
    assert normalize("p", {"data": [_rated_model("a", rate=60)]}, priority=0)[0].priority == 0
    assert normalize("p", {"data": [_rated_model("b", rate=59)]}, priority=0)[0].priority == 1


def test_a_malformed_rate_does_not_break_discovery():
    """The field comes from a provider's JSON: it can be anything."""
    for bad in ("mucho", None, [], {}):
        routes = normalize("grok", {"data": [_rated_model("m", rate=bad)]}, priority=0)
        assert routes and routes[0].priority == 0, bad


# --- scarcity from a MEASURED allowance, for providers that publish none


def _catalogue(model_id="m", **extra):
    return {"data": [{"id": model_id, "pricing": {"prompt": "0"},
                      "supported_parameters": ["tools"], **extra}]}


def test_a_measured_allowance_below_the_floor_holds_a_route_in_reserve():
    """The providers this exists for publish nothing, so the only evidence is ours."""
    routes = normalize("kilo", _catalogue(), priority=0,
                       measured_rates={"kilo/m": 4.0})
    assert routes[0].priority == 1, "a measured 4/h is below the floor"


def test_a_published_allowance_outranks_a_measured_one():
    """What the provider states is a claim about policy; ours is a sample of it.

    The published number covers the whole population and is available before a
    single request; the measurement only ever sees the slice we sent. When both
    exist the statement wins -- here an abundant 999/h keeps the route in front
    even though our own traffic happened to run into a refusal.
    """
    routes = normalize("grok", _catalogue(requests_per_hour=999), priority=0,
                       measured_rates={"grok/m": 4.0})
    assert routes[0].priority == 0


def test_a_route_that_was_never_refused_is_never_demoted():
    """The safety property: absence of evidence is not evidence of scarcity.

    Measured 2026-08-19, 60 of 69 live routes had never once been refused -- if
    an unmeasured route could be demoted, that would be nearly the whole
    catalogue, and the demotion would be self-reinforcing: less traffic, less
    chance of ever measuring it. `sync_catalogue` passes only measured
    allowances for exactly this reason.
    """
    routes = normalize("deepseek", _catalogue(), priority=0, measured_rates={})
    assert routes[0].priority == 0
    assert normalize("deepseek", _catalogue(), priority=0)[0].priority == 0


def _contract(**overrides):
    caps = {k: False for k in REQUIRED_CAPABILITIES}
    caps.update(chat=True, streaming=True, vision=True, images=True,
                translate=True, search=True)
    caps.update(overrides)
    return ProviderContract(version=1, provider="chatgpt",
                            auth=Auth(mode="account", plan="go",
                                      subscription_active=True),
                            capabilities=caps)


def test_the_contract_supplies_the_provider_level_capabilities():
    routes = normalize("chatgpt", {"data": [{"id": "gpt-5-6"}]},
                       default_capabilities=_CHATGPT_DEFAULTS,
                       contract=_contract())
    c = routes[0].capabilities
    assert c.vision is True
    assert c.images is True
    assert c.translate is True
    assert c.search is True
    assert c.tools is False


def test_per_model_metadata_supplies_the_real_context_window():
    # The whole reason this exists: 128000 declared, 52815 real.
    routes = normalize("chatgpt",
                       {"data": [{"id": "gpt-5-6", "context_window": 52815,
                                  "max_output_tokens": 8192}]},
                       default_capabilities=_CHATGPT_DEFAULTS,
                       contract=_contract())
    assert routes[0].capabilities.context == 52815
    assert routes[0].capabilities.max_output == 8192


def test_a_model_without_a_context_window_falls_back_to_the_yaml():
    routes = normalize("chatgpt", {"data": [{"id": "gpt-5-6"}]},
                       default_capabilities=_CHATGPT_DEFAULTS,
                       contract=_contract())
    assert routes[0].capabilities.context == _CHATGPT_DEFAULTS.context


def test_a_per_model_capability_may_narrow_the_provider_level_one():
    routes = normalize("chatgpt",
                       {"data": [{"id": "gpt-image-1",
                                  "capabilities": {"vision": False, "images": True}}]},
                       default_capabilities=_CHATGPT_DEFAULTS,
                       contract=_contract())
    assert routes[0].capabilities.vision is False
    assert routes[0].capabilities.images is True


def test_a_per_model_capability_may_not_widen_the_provider_level_one(caplog):
    with caplog.at_level(logging.WARNING):
        routes = normalize("chatgpt",
                           {"data": [{"id": "gpt-5-6",
                                      "capabilities": {"images": True}}]},
                           default_capabilities=_CHATGPT_DEFAULTS,
                           contract=_contract(images=False))
    assert routes[0].capabilities.images is False
    assert "gpt-5-6" in caplog.text


def test_exceptions_beat_the_contract():
    # Level 1 against level 3. `exceptions` is the strongest voice, and it must
    # stay that way: it is where a measurement that contradicts what a proxy
    # says about itself gets recorded, per model id.
    routes = normalize("chatgpt", {"data": [{"id": "gpt-5-6"}]},
                       default_capabilities=_CHATGPT_DEFAULTS,
                       exceptions={"gpt-5-6": {"vision": False}},
                       contract=_contract(vision=True))
    assert routes[0].capabilities.vision is False


def test_exceptions_beat_a_per_model_value():
    """Level 1 against level 2 -- the rung of the precedence chain nothing was
    asserting, and the one that decides whether an exception can be trusted at
    all: /v1/models is the source most likely to disagree with a hand-written
    correction, because it is the one the proxy regenerates on its own.
    """
    routes = normalize("chatgpt",
                       {"data": [{"id": "gpt-5-6",
                                  "capabilities": {"vision": True}}]},
                       default_capabilities=_CHATGPT_DEFAULTS,
                       exceptions={"gpt-5-6": {"vision": False}},
                       contract=_contract(vision=True))
    assert routes[0].capabilities.vision is False


def test_an_exception_that_stays_silent_lets_the_contract_decide():
    """Being applied LAST cuts both ways: an exception that names a capability
    pins it past every plan change. chatgpt's `gpt-image-1` therefore declares
    only `tools` and `vision` -- had it kept `images: true`, the one image route
    the proxy actually publishes would still be advertised at `priority: 0`
    after the Go plan lapses, absorbing every image request into a 503.
    """
    catalogue = {"data": [{"id": "gpt-image-1"}]}
    exceptions = {"gpt-image-1": {"tools": False, "vision": False}}
    lapsed = normalize("chatgpt", catalogue, default_capabilities=_CHATGPT_DEFAULTS,
                       exceptions=exceptions, contract=_contract(images=False))
    assert lapsed[0].capabilities.images is False
    # ...and nothing is lost while the plan holds: the contract says true.
    paid = normalize("chatgpt", catalogue, default_capabilities=_CHATGPT_DEFAULTS,
                     exceptions=exceptions, contract=_contract(images=True))
    assert paid[0].capabilities.images is True
    assert paid[0].capabilities.vision is False    # the exception still applies


def test_without_a_contract_nothing_changes():
    routes = normalize("chatgpt", {"data": [{"id": "gpt-5-6"}]},
                       default_capabilities=_CHATGPT_DEFAULTS)
    assert routes[0].capabilities == _CHATGPT_DEFAULTS


# ── grok's three image routes, end to end ─────────────────────────────────────
#
# Until 2026-08-20 this invariant was asserted in providers.yaml, as
# `all(e.get("images") is True for e in grok.exceptions.values())`: a
# CI-checked statement that the gateway ends up with grok image routes.
# Retiring `exceptions` removed the assertion along with the override, and the
# invariant moved from a YAML declaration to a CROSS-REPO RUNTIME FACT -- grok's
# /health says `images: true`, grok's /v1/models says which three models it
# applies to, and normalize puts the two together. Nothing checked that.
#
# The fixtures are grok-proxy's REAL published shape, captured from its
# /health and /v1/models on the grok-contract branch, so this also documents
# the wire shape the gateway expects of it.

def _grok_fixtures():
    return _load("grok_health.json"), _load("grok_models.json")


def _grok_routes(contract_doc):
    """grok's catalogue through the REAL providers.yaml entry -- priority,
    default_capabilities and exceptions included. Using the registry rather
    than a synthetic Provider is the point: an `images: true` exception added
    back, or a default_capabilities edit, has to show up here."""
    _, models = _grok_fixtures()
    grok = [p for p in load(PROVIDERS_YAML, {}) if p.id == "grok"][0]
    return normalize("grok", models, priority=grok.priority,
                     default_capabilities=grok.default_capabilities,
                     exceptions=grok.exceptions,
                     contract=parse_health("grok", contract_doc)
                     if contract_doc is not None else None)


IMAGINE = {"imagine-agent-mode", "imagine-agent-mode-dev",
           "imagine-agent-mode-grok-4-5"}


def test_groks_health_fixture_is_a_contract_this_gateway_can_read():
    health, _ = _grok_fixtures()
    c = parse_health("grok", health)
    assert c is not None, "the captured /health no longer parses as a contract"
    assert c.provider == "grok"
    assert c.capabilities["images"] is True
    assert c.capabilities["files"] is False


def test_grok_ends_up_with_exactly_three_image_routes():
    health, _ = _grok_fixtures()
    routes = _grok_routes(health)
    assert {r.model_id for r in routes if r.capabilities.images} == IMAGINE


def test_the_imagine_family_draws_and_does_nothing_else():
    """The per-model block's whole job: the provider level says tools, vision
    and images are all true, and /v1/models narrows the imagine family to
    images only. A regression here does not remove a route -- it advertises
    three image agents as chat models at priority 0."""
    health, _ = _grok_fixtures()
    imagine = [r for r in _grok_routes(health) if r.model_id in IMAGINE]
    assert len(imagine) == 3
    for r in imagine:
        assert (r.capabilities.images, r.capabilities.tools,
                r.capabilities.vision) == (True, False, False), r.model_id


def test_the_plugin_routes_keep_tools_and_vision_and_do_not_draw():
    health, _ = _grok_fixtures()
    plugins = [r for r in _grok_routes(health)
               if r.model_id.startswith("grok-plugins-")]
    assert len(plugins) == 14
    for r in plugins:
        assert (r.capabilities.tools, r.capabilities.vision,
                r.capabilities.images) == (True, True, False), r.model_id


def test_without_the_contract_grok_loses_every_image_route():
    """The exposure providers.yaml's fallback comment now names explicitly.
    `default_capabilities` covers tools, vision, context and max_output; it
    carries no `images`, so any state where grok-proxy ANSWERS /health but
    parse_health returns None -- a rolled-back proxy with no `contract` key, a
    version this gateway does not speak, a partial `capabilities` block --
    takes grok's only three image routes away, with one WARNING and no alert.
    (An UNREACHABLE /health is a different path and does not cost them:
    probing carries over the last stored contract, or skips the provider.)

    Asserted rather than fixed: adding `images` to default_capabilities would
    reintroduce exactly the stale hand-declaration the exceptions retirement
    removed."""
    without = _grok_routes(None)
    assert [r.model_id for r in without if r.capabilities.images] == []
    # ...and nothing ELSE is lost, which is why this degrades quietly: the
    # same 17 routes come back, tools and vision intact.
    assert len(without) == len(_grok_routes(_grok_fixtures()[0])) == 17
    assert all(r.capabilities.tools and r.capabilities.vision
               for r in without if r.model_id not in IMAGINE)


def test_emulates_tools_over_discovered_native_models_warns_once(caplog):
    """A discovered catalogue whose models already report native tool support
    plus emulates_tools is the same contradiction fixed_routes warns about --
    aggregated to one line, because a provider like grok discovers 25 of them."""
    import logging
    data = {"data": [
        {"id": "native-a:free", "pricing": {"prompt": "0"}, "context_length": 4096,
         "architecture": {"output_modalities": ["text"]},
         "supported_parameters": ["tools", "tool_choice"]},
        {"id": "native-b:free", "pricing": {"prompt": "0"}, "context_length": 4096,
         "architecture": {"output_modalities": ["text"]},
         "supported_parameters": ["tools", "tool_choice"]},
        {"id": "plain:free", "pricing": {"prompt": "0"}, "context_length": 4096,
         "architecture": {"output_modalities": ["text"]},
         "supported_parameters": ["max_tokens"]},
    ]}
    with caplog.at_level(logging.WARNING, logger="llm_libre.catalog"):
        routes = normalize("kilo", data, emulates_tools=True)
    assert caplog.text.count("native tool support") == 1
    assert "2" in caplog.text
    assert all(r.capabilities.tools for r in routes)


def test_emulates_tools_over_a_toolless_discovery_stays_silent(caplog):
    import logging
    data = {"data": [
        {"id": "plain:free", "pricing": {"prompt": "0"}, "context_length": 4096,
         "architecture": {"output_modalities": ["text"]},
         "supported_parameters": ["max_tokens"]},
    ]}
    with caplog.at_level(logging.WARNING, logger="llm_libre.catalog"):
        routes = normalize("kilo", data, emulates_tools=True)
    assert "native tool support" not in caplog.text
    assert routes[0].capabilities.tools is True
