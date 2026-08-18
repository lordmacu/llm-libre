import json
import logging
from pathlib import Path

from llm_libre.catalog import normalize
from llm_libre.models import Capabilities

FIXTURES = Path(__file__).parent / "fixtures"

_CHATGPT_DEFAULTS = Capabilities(tools=False, vision=False, context=128000, max_output=8192)


def _load(name):
    return json.loads((FIXTURES / name).read_text())


def test_paid_models_are_discarded():
    data = {"data": [
        {"id": "caro/modelo", "pricing": {"prompt": "0.0000015"},
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
        {"id": "sin/tools:free", "pricing": {"prompt": "0"}, "context_length": 4096,
         "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
         "supported_parameters": ["max_tokens"]},
    ]}
    routes = normalize("kilo", data, priority=1)
    assert routes[0].priority == 1


def test_normalize_without_a_declared_priority_uses_the_default_of_one_hundred():
    data = {"data": [
        {"id": "sin/tools:free", "pricing": {"prompt": "0"}, "context_length": 4096,
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
        {"id": "sin/tools:free", "pricing": {"prompt": "0"}, "context_length": 4096,
         "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
         "supported_parameters": ["max_tokens", "temperature"]},
    ]}
    assert normalize("kilo", data)[0].capabilities.tools is False


def test_an_unreadable_price_is_treated_as_paid():
    data = {"data": [{"id": "raro/modelo", "pricing": {},
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

def _model(mid, name="Un Modelo", description="Un modelo de chat general."):
    return {"id": mid, "name": name, "description": description,
            "pricing": {"prompt": "0"}, "context_length": 4096,
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "supported_parameters": ["tools"],
            "top_provider": {"max_completion_tokens": 4096}}


def test_a_guardrail_is_discarded_by_its_own_description():
    data = {"data": [_model(
        "vendor/lo-que-sea:free", "Vendor: Lo Que Sea (free)",
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
#     id blew up with KeyError; sondeo.py swallowed the whole thing and THAT
#     provider's catalogue froze forever, without a single log line. ---

def test_an_entry_without_an_id_is_skipped_without_blowing_up_and_is_logged(caplog):
    data = {"data": [
        {"pricing": {"prompt": "0"}, "architecture": {"output_modalities": ["text"]}},
        {"id": "bueno:free", "pricing": {"prompt": "0"},
         "architecture": {"output_modalities": ["text"]}},
    ]}
    with caplog.at_level(logging.WARNING, logger="llm_libre.catalog"):
        routes = normalize("kilo", data)
    assert [r.model_id for r in routes] == ["bueno:free"]
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
#     ("auto:rapido", "auto:potente", "auto:tools", "auto:vision" -- see ALIAS
#     in api.py and parse_request, which resolves ANY id starting with
#     "auto:" as an alias, never as a literal id). A provider publishing a real
#     model called, say, "auto:rapido" created a PERMANENTLY unreachable route:
#     no request with `model: "auto:rapido"` ever reaches
#     `pedido.model == "auto:rapido"`, because parse_request resolves it
#     first as profile "rapido" with modelo=None. ---

def test_the_compound_auto_aliases_are_discarded_too():
    routes = normalize("prov", [
        {"id": "auto:rapido", "name": "Collides with the compound alias"},
        {"id": "auto:potente", "name": "Collides with the compound alias"},
        {"id": "auto:tools", "name": "Collides with the compound alias"},
        {"id": "auto:vision", "name": "Collides with the compound alias"},
        {"id": "un-modelo-real", "name": "Real model"},
    ], default_capabilities=_CHATGPT_DEFAULTS)
    ids = {r.model_id for r in routes}
    assert ids == {"un-modelo-real"}


def test_a_new_model_from_the_proxy_appears_without_touching_the_yaml():
    # The "con_modelo_nuevo" fixture simulates ChatGPT's real backend adding a
    # model tomorrow (gpt-5-7) that does not exist today: it has to show up on
    # its own, without anyone editing proveedores.yaml or this test.
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
        {"id": "modelo-sin-metadatos", "description": "Un Modelo"},
    ]}
    routes = normalize("chatgpt", data, default_capabilities=_CHATGPT_DEFAULTS)
    assert [r.model_id for r in routes] == ["modelo-sin-metadatos"]
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
        {"id": "caro/modelo", "pricing": {"prompt": "0.0000015"},
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
