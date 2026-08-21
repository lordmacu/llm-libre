from pathlib import Path

from llm_libre.contract import REQUIRED_CAPABILITIES, Auth, ProviderContract
from llm_libre.models import Capabilities
from llm_libre.providers import Provider, contract_url, fixed_routes, load

YAML = str(Path(__file__).resolve().parents[1] / "providers.yaml")


def test_it_loads_the_registered_providers():
    # openrouter was removed from the registry (operator decision, 2026-08-17):
    # OPENROUTER_API_KEY was never configured, so its 16 routes 401'd every time
    # -- almost half the catalogue, burning probe quota and space in /v1/ranking
    # just to keep proving they were still dead. It stays documented in
    # docs/providers.md as an example of the "all discovered" pattern with an
    # optional key.
    # perplexity joined on 2026-08-17 with ONE declared route (`turbo`): its
    # /v1/models publishes 124 models but in the anonymous flow they all fall
    # back to turbo, so declaring the catalogue would mean measuring 124 clones.
    # grok joined on 2026-08-18: our own OpenAI-compatible proxy, ids discovered
    # from /models and capabilities declared (its catalogue does not carry them).
    # It is the only free route with tools AND vision, both verified live against
    # the deployed proxy.
    # mistral joined on 2026-08-18, also with ONE declared route: its /v1/models
    # publishes ten ids but the proxy never forwards `model` to Le Chat, so all
    # ten are the same backend -- the perplexity lesson, applied before it cost
    # anything.
    ps = load(YAML, {"MINIMAX_API_KEY": "mm"})
    assert [p.id for p in ps] == ["chatgpt", "perplexity", "deepseek", "grok",
                                  "mistral", "kilo", "minimax"]


def test_perplexity_declares_a_single_route_without_tools():
    # No native tool calling, and emulation does not work either: measured 0/3
    # tool_calls through the emulation layer (2026-08-18). Claiming the capability
    # would route tool requests here only to answer prose -- so a request with
    # tools must NEVER land on this provider, and this declaration is what
    # guarantees it.
    pplx = next(p for p in load(YAML, {}) if p.id == "perplexity")
    assert pplx.emulates_tools is False
    routes = fixed_routes(pplx)
    assert [r.key for r in routes] == ["perplexity/turbo"]
    assert routes[0].capabilities.tools is False
    assert routes[0].tier == "free"
    assert pplx.base_url.endswith("/v1")   # without the /v1 everything 404s


def test_kilo_without_a_key_keeps_an_empty_key_and_stays_valid():
    kilo = next(p for p in load(YAML, {}) if p.id == "kilo")
    assert kilo.api_key == ""
    assert kilo.tier == "free"


def test_it_resolves_the_keys_from_the_environment():
    ps = load(YAML, {"MINIMAX_API_KEY": "secreta"})
    assert next(p for p in ps if p.id == "minimax").api_key == "secreta"


def test_the_extra_headers_are_preserved(tmp_path):
    # Migrated to a synthetic YAML (post-Task-14 review): it used to assert on
    # `openrouter`, the only real provider declaring cabeceras_extra -- but that
    # one was removed from the registry (see test_it_loads_the_registered_providers)
    # and this mechanism (load() forwards cabeceras_extra verbatim) does not depend
    # on any particular provider using it today.
    yaml_with_headers = tmp_path / "con_cabeceras.yaml"
    yaml_with_headers.write_text(
        "providers:\n"
        "  - id: suelto\n"
        "    tier: gratis\n"
        "    dialect: openai\n"
        "    base_url: https://suelto.test\n"
        "    models_path: /models\n"
        "    extra_headers:\n"
        "      X-Title: llm-libre\n")
    p = load(str(yaml_with_headers), {})[0]
    assert p.extra_headers["X-Title"] == "llm-libre"


def test_fixed_models_become_paid_routes():
    minimax = next(p for p in load(YAML, {}) if p.id == "minimax")
    routes = fixed_routes(minimax)
    assert len(routes) == 1
    assert routes[0].key == "minimax/MiniMax-M3"
    assert routes[0].tier == "paid"
    assert routes[0].capabilities.tools is True
    # vision became True on 2026-08-18: it was False only because nobody had
    # measured it. The capability sweep showed it correctly names the colour of a
    # red PNG, so it was a real capability the router was discarding because of a
    # conservative declaration.
    assert routes[0].capabilities.vision is True
    assert routes[0].capabilities.context == 128000
    assert routes[0].capabilities.max_output == 32768
    assert routes[0].priority == 2   # minimax's own, from the YAML, not a constant


def test_a_free_provider_has_no_fixed_models():
    kilo = next(p for p in load(YAML, {}) if p.id == "kilo")
    assert fixed_routes(kilo) == []


def test_a_whitespace_only_key_is_normalised_to_empty():
    ps = load(YAML, {"KILO_API_KEY": "   ", "MINIMAX_API_KEY": "\t\n"})
    kilo = next(p for p in ps if p.id == "kilo")
    minimax = next(p for p in ps if p.id == "minimax")
    assert kilo.api_key == ""
    assert minimax.api_key == ""


# --- Task 13: chatgpt-proxy, priority and base_url_env ---

def test_chatgpt_has_priority_zero_and_is_free():
    chatgpt = next(p for p in load(YAML, {}) if p.id == "chatgpt")
    assert chatgpt.tier == "free"
    assert chatgpt.priority == 0


def test_kilo_sits_behind_the_branded_band_and_its_reserve():
    kilo = next(p for p in load(YAML, {}) if p.id == "kilo")
    # 2 since the 2026-08-19 experiment: 0 is the branded proxies, 1 is the band
    # their SCARCE routes are demoted into (catalog.SUSTAINED_RATE_FLOOR), and Kilo
    # is the safety net behind both.
    assert kilo.priority == 2


def test_minimax_sits_at_priority_two():
    minimax = next(p for p in load(YAML, {}) if p.id == "minimax")
    assert minimax.priority == 2


def test_a_provider_without_priority_in_the_yaml_uses_the_default_of_one_hundred(tmp_path):
    yaml_no_priority = tmp_path / "sin_prioridad.yaml"
    yaml_no_priority.write_text(
        "providers:\n"
        "  - id: suelto\n"
        "    tier: gratis\n"
        "    dialect: openai\n"
        "    base_url: https://suelto.test\n"
        "    models_path: /models\n")
    p = load(str(yaml_no_priority), {})[0]
    assert p.priority == 100


def test_base_url_env_uses_the_environment_variable_when_defined():
    chatgpt = next(p for p in load(YAML, {"CHATGPT_PROXY_URL": "https://blog.test:8888/v1"})
                   if p.id == "chatgpt")
    assert chatgpt.base_url == "https://blog.test:8888/v1"


def test_base_url_env_falls_back_to_the_yaml_default_when_the_variable_is_missing():
    chatgpt = next(p for p in load(YAML, {}) if p.id == "chatgpt")
    assert chatgpt.base_url == "http://127.0.0.1:8888/v1"


def test_base_url_env_falls_back_to_the_default_when_the_variable_is_blank():
    chatgpt = next(p for p in load(YAML, {"CHATGPT_PROXY_URL": "   "})
                   if p.id == "chatgpt")
    assert chatgpt.base_url == "http://127.0.0.1:8888/v1"


def test_a_provider_without_base_url_env_is_unaffected(tmp_path):
    # kilo does not declare base_url_env: an environment variable that happens to
    # carry its id must not override its url.
    kilo = next(p for p in load(YAML, {"KILO_URL": "https://another.test"}) if p.id == "kilo")
    assert kilo.base_url == "https://api.kilo.ai/api/gateway"


# --- Task 13 follow-up: chatgpt moved from modelos_fijos to DISCOVERED (its
#     /v1/models is now dynamic), but it still carries no capability metadata --
#     which is why it declares `default_capabilities` instead of
#     `fixed_models`. It is a GENERAL mechanism: any provider whose catalogue is
#     equally bare can use it, it is not special to chatgpt.
#
#     dall-e-3 is a SINGLE fixed_model added on top: the proxy exposes
#     /v1/images/generations and needs one declared image-capable route. It
#     coexists with dynamic discovery because probing.sync_catalogue no longer
#     `continue`s after the fixed-models pass (see that file). ---

def test_chatgpt_is_discovered_via_models_path_not_via_fixed_models():
    chatgpt = next(p for p in load(YAML, {}) if p.id == "chatgpt")
    assert chatgpt.models_path == "/models"
    # dall-e-3 is the one fixed image-generation route; chat models are discovered.
    assert [m["id"] for m in chatgpt.fixed_models] == ["dall-e-3"]


def test_chatgpt_declares_default_capabilities_with_tools_false():
    chatgpt = next(p for p in load(YAML, {}) if p.id == "chatgpt")
    assert chatgpt.default_capabilities == Capabilities(
        tools=False, vision=False, context=128000, max_output=8192)


def test_kilo_declares_no_default_capabilities():
    kilo = next(p for p in load(YAML, {}) if p.id == "kilo")
    assert kilo.default_capabilities is None


def test_minimax_declares_no_default_capabilities_either():
    # Still the old pattern: ids AND capabilities declared by hand
    # (fixed_routes), not discovery with defaults.
    minimax = next(p for p in load(YAML, {}) if p.id == "minimax")
    assert minimax.default_capabilities is None
    assert len(fixed_routes(minimax)) == 1


def test_a_provider_without_default_capabilities_in_the_yaml_gets_none(tmp_path):
    yaml_no_defaults = tmp_path / "sin_defaults.yaml"
    yaml_no_defaults.write_text(
        "providers:\n"
        "  - id: suelto\n"
        "    tier: gratis\n"
        "    dialect: openai\n"
        "    base_url: https://suelto.test\n"
        "    models_path: /models\n")
    p = load(str(yaml_no_defaults), {})[0]
    assert p.default_capabilities is None


# --- Follow-up review: unpinned rungs ---
#
# `fixed_routes` stamping `priority=p.priority` had NO test distinguishing "it
# takes the provider's real priority" from "it always writes the same constant"
# -- inert today because minimax is the only modelos_fijos provider and it is
# paid, but the registry invites future declared free providers. It is tested
# with a synthetic YAML and a deliberately distinctive priority (77) so that no
# coincidence with a default (100) or with the real minimax (2) can disguise a
# hardcoded constant.

# --- Finding 1 of the review: canvas unwrapping was GLOBAL, and ':::nota{...}'
#     is also standard Docusaurus/MDX syntax -- it was verified live that a Kilo
#     route lost those legitimate documentation markers. It becomes a PER-PROVIDER
#     declaration, the same shape as default_capabilities: off by default, on
#     only for chatgpt. ---

def test_chatgpt_declares_canvas_unwrapping():
    chatgpt = next(p for p in load(YAML, {}) if p.id == "chatgpt")
    assert chatgpt.unwraps_canvas is True


def test_kilo_does_not_unwrap_canvas():
    kilo = next(p for p in load(YAML, {}) if p.id == "kilo")
    assert kilo.unwraps_canvas is False


def test_a_provider_without_canvas_unwrapping_in_the_yaml_defaults_to_false(tmp_path):
    yaml_no_canvas = tmp_path / "sin_canvas.yaml"
    yaml_no_canvas.write_text(
        "providers:\n"
        "  - id: suelto\n"
        "    tier: gratis\n"
        "    dialect: openai\n"
        "    base_url: https://suelto.test\n"
        "    models_path: /models\n")
    p = load(str(yaml_no_canvas), {})[0]
    assert p.unwraps_canvas is False


# --- Finding 2 of the review (per-provider timeout): a clean addition that does
#     not complicate the design -- a default of None means "use proxy.py's global
#     TIMEOUT_S", exactly as today for anyone who does not declare it. ---

def test_a_provider_without_a_declared_timeout_gets_none(tmp_path):
    # kilo still declares no timeout_s -- Task 14 only gave one to chatgpt (see
    # the test below), on purpose: no other provider's timeout is touched.
    kilo = next(p for p in load(YAML, {}) if p.id == "kilo")
    assert kilo.timeout_s is None


def test_chatgpt_declares_its_own_timeout_in_the_real_yaml():
    # Task 14: chatgpt has priority:0 (it is tried first on EVERY request) and
    # runs on `blog`, a saturated machine -- without its own timeout_s, a hang
    # there cost up to the full TIMEOUT_S=90s per attempt. 45s (see the comment in
    # providers.yaml for the measurement justifying it) halves that worst case
    # without lowering anyone else's timeout.
    chatgpt = next(p for p in load(YAML, {}) if p.id == "chatgpt")
    assert chatgpt.timeout_s == 150.0   # 150 since the 2026-08-19 experiment: a branded proxy must not lose
                                  # because the gateway gave up on a generation that was
                                  # still coming. See test_no_branded_proxy_is_cut_off_*.


def test_a_provider_can_declare_its_own_timeout(tmp_path):
    yaml_with_timeout = tmp_path / "con_timeout.yaml"
    yaml_with_timeout.write_text(
        "providers:\n"
        "  - id: lento\n"
        "    tier: gratis\n"
        "    dialect: openai\n"
        "    base_url: https://lento.test\n"
        "    models_path: /models\n"
        "    timeout_s: 20\n")
    p = load(str(yaml_with_timeout), {})[0]
    assert p.timeout_s == 20.0


# --- Finding 6 of the review: CHATGPT_PROXY_URL replaces the WHOLE base_url, so
#     the operator has to remember to include the /v1 themselves -- the same
#     footgun already fixed on the YAML side, surviving on the environment side.
#     Choice: NORMALISE (append the path suffix the YAML already declares as the
#     default, if the variable does not carry one) rather than failing at startup
#     -- there is a single correct interpretation (the suffix the YAML itself
#     declares), so auto-correcting is more useful than taking down a running
#     deployment over a recoverable typo. It is logged either way, so it stays
#     visible in production. ---

def test_base_url_env_appends_the_suffix_when_the_variable_lacks_it():
    chatgpt = next(p for p in load(YAML, {"CHATGPT_PROXY_URL": "https://blog.test:8888"})
                   if p.id == "chatgpt")
    assert chatgpt.base_url == "https://blog.test:8888/v1"


def test_base_url_env_with_a_query_string_does_not_splinter_the_suffix_inside_it():
    # LOW from the review: the suffix was appended by text concatenation, so a URL
    # with an empty path but WITH a query string ended up with the suffix glued
    # INSIDE the query value ("...:8888?token=abc" -> "...:8888?token=abc/v1").
    # The URL has to be parsed and rebuilt, not string-concatenated.
    chatgpt = next(
        p for p in load(YAML, {"CHATGPT_PROXY_URL": "https://blog.test:8888?token=abc"})
        if p.id == "chatgpt")
    assert chatgpt.base_url == "https://blog.test:8888/v1?token=abc"


def test_base_url_env_does_not_duplicate_the_suffix_when_it_is_already_there():
    chatgpt = next(p for p in load(YAML, {"CHATGPT_PROXY_URL": "https://blog.test:8888/v1"})
                   if p.id == "chatgpt")
    assert chatgpt.base_url == "https://blog.test:8888/v1"


def test_base_url_env_with_a_trailing_slash_does_not_duplicate_the_suffix():
    chatgpt = next(p for p in load(YAML, {"CHATGPT_PROXY_URL": "https://blog.test:8888/"})
                   if p.id == "chatgpt")
    assert chatgpt.base_url == "https://blog.test:8888/v1"


def test_base_url_env_logs_when_it_normalises(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="llm_libre.providers"):
        load(YAML, {"CHATGPT_PROXY_URL": "https://blog.test:8888"})
    assert "chatgpt" in caplog.text
    assert "/v1" in caplog.text


def test_base_url_env_appends_nothing_when_the_default_has_no_suffix(tmp_path):
    # kilo (with no base_url_env today) is unaffected by this mechanism; a
    # provider WITH base_url_env whose default has no path (host only) must not
    # append anything out of nowhere either.
    yaml_no_suffix = tmp_path / "sin_sufijo.yaml"
    yaml_no_suffix.write_text(
        "providers:\n"
        "  - id: suelto\n"
        "    tier: gratis\n"
        "    dialect: openai\n"
        "    base_url_env: SUELTO_URL\n"
        "    base_url: https://suelto.test\n"
        "    models_path: /models\n")
    p = load(str(yaml_no_suffix), {"SUELTO_URL": "https://another.test"})[0]
    assert p.base_url == "https://another.test"


# --- Re-review: the normalisation rule was TOO eager -- it appended the suffix
#     unconditionally, so "...:8888/v2" (a path the operator chose, e.g. a reverse
#     proxy mount) ended up as ".../v2/v1/chat/completions". It is tightened: the
#     suffix is only appended when the environment's URL carries NO path of its
#     own (empty or "/"); if it already has one, it is used AS IS -- "they said
#     what they meant" -- with a warning in case it was accidental, not a silent
#     correction. ---

def test_base_url_env_with_its_own_path_does_not_get_the_suffix_forced_on_it():
    chatgpt = next(p for p in load(YAML, {"CHATGPT_PROXY_URL": "https://blog.test:8888/v2"})
                   if p.id == "chatgpt")
    assert chatgpt.base_url == "https://blog.test:8888/v2"


def test_base_url_env_with_its_own_path_logs_a_warning_without_modifying(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="llm_libre.providers"):
        chatgpt = next(
            p for p in load(YAML, {"CHATGPT_PROXY_URL": "https://blog.test:8888/v2"})
            if p.id == "chatgpt")
    assert chatgpt.base_url == "https://blog.test:8888/v2"
    assert "chatgpt" in caplog.text
    assert "/v2" in caplog.text


def test_base_url_env_with_the_correct_path_logs_nothing(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="llm_libre.providers"):
        load(YAML, {"CHATGPT_PROXY_URL": "https://blog.test:8888/v1"})
    assert caplog.text == ""


def test_fixed_routes_uses_the_providers_real_priority_not_a_constant(tmp_path):
    yaml_odd_priority = tmp_path / "prioridad_rara.yaml"
    yaml_odd_priority.write_text(
        "providers:\n"
        "  - id: pago_futuro\n"
        "    tier: pago\n"
        "    priority: 77\n"
        "    dialect: openai\n"
        "    base_url: https://pago-futuro.test\n"
        "    fixed_models:\n"
        "      - id: model-x\n"
        "        tools: true\n"
        "        vision: false\n"
        "        context: 1000\n"
        "        max_output: 100\n")
    p = load(str(yaml_odd_priority), {})[0]
    routes = fixed_routes(p)
    assert len(routes) == 1
    assert routes[0].priority == 77


def test_deepseek_declares_two_routes_with_emulated_tools():
    # deepseek-chat and deepseek-reasoner do not support native tool calling, but
    # emulates_tools:true makes fixed_routes() report tools=True in capabilities:
    # prompt-injection emulation (tool_emulator.py) is transparent to the router.
    # deepseek-vision is NOT declared: image input was never verified.
    ds = next(p for p in load(YAML, {}) if p.id == "deepseek")
    assert ds.emulates_tools is True
    routes = fixed_routes(ds)
    assert [r.model_id for r in routes] == ["deepseek-chat", "deepseek-reasoner"]
    assert all(r.capabilities.tools is True for r in routes)
    assert all(r.capabilities.vision is False for r in routes)
    assert ds.base_url.endswith("/v1")
    assert ds.timeout_s == 150.0  # was 60: the WASM proof-of-work adds variable time
                                  # on top, and 60 was measured cutting off a real
                                  # generation at exactly 60.1s (2026-08-19).


def test_mistral_declares_one_route_not_the_ten_its_catalogue_publishes():
    """The perplexity lesson, applied to a second provider.

    mistral-proxy's /v1/models advertises ten ids (mistral-small-latest,
    mistral-large-latest, codestral-latest, open-mistral-7b and six more). They
    are all the SAME model: the proxy echoes the requested `model` back in the
    response envelope but never forwards it -- Le Chat's tRPC newChat body has
    no model field at all. Measured 2026-08-18: four different ids asked
    "exactly which model are you?" all answered "Mistral Medium 3.5".

    Discovering that catalogue would put ten routes in the table, each probed
    and scored independently, all hitting one backend -- 10x the probe quota
    spent ranking a single model against itself, and a /v1/ranking that lies
    about what is behind each row. So: fixed_models with one honest id."""
    ms = next(p for p in load(YAML, {}) if p.id == "mistral")
    routes = fixed_routes(ms)
    assert [r.key for r in routes] == ["mistral/mistral-medium-latest"]
    assert not ms.models_path, "discovery must stay off: the ten ids are one model"
    assert routes[0].tier == "free"
    assert ms.base_url.endswith("/v1")   # without the /v1 everything 404s


def test_mistral_emulates_tools_and_really_sees_images():
    # tools:false natively (3/3 tool_calls:None with tool_choice:"required"),
    # but 6/6 through the emulation layer -- the only provider besides deepseek
    # to earn emulates_tools. fixed_routes() therefore reports tools=True:
    # emulation is transparent to the router.
    #
    # vision:true is MEASURED, not declared out of optimism: a 64x64 solid red
    # PNG came back named "Red". It is the only free route with vision besides
    # grok.
    ms = next(p for p in load(YAML, {}) if p.id == "mistral")
    assert ms.emulates_tools is True
    caps = fixed_routes(ms)[0].capabilities
    assert caps.tools is True
    assert caps.vision is True
    # 60s because of CONCURRENCY, not single-request latency: alone it answers
    # in 3.5-7.4s, but three concurrent requests measured 4.9/12.6/16.2s against
    # one shared account session. A 45s ceiling would be within ~3x of an
    # observed real request under load.
    assert ms.timeout_s == 150.0   # 150 since the 2026-08-19 experiment: a branded proxy must not lose
                                  # because the gateway gave up on a generation that was
                                  # still coming. See test_no_branded_proxy_is_cut_off_*.


# --- The same shape of declaration, for grok-proxy's '<xai:...>' UI cards. It
#     covers the MARKERS only: the plain-text status labels Grok interleaves with
#     the answer ("Compilando las 20 recomendaciones", measured leaking into a
#     JSON value on 2026-08-18) reach the gateway as ordinary prose, and are
#     dropped at the source instead. ---

def test_grok_declares_card_stripping():
    grok = next(p for p in load(YAML, {}) if p.id == "grok")
    assert grok.strips_xai_cards is True


def test_kilo_does_not_strip_cards():
    kilo = next(p for p in load(YAML, {}) if p.id == "kilo")
    assert kilo.strips_xai_cards is False


def test_a_provider_without_card_stripping_in_the_yaml_defaults_to_false(tmp_path):
    yaml_no_cards = tmp_path / "sin_cards.yaml"
    yaml_no_cards.write_text(
        "providers:\n"
        "  - id: suelto\n"
        "    tier: gratis\n"
        "    dialect: openai\n"
        "    base_url: https://suelto.test\n"
        "    models_path: /models\n")
    p = load(str(yaml_no_cards), {})[0]
    assert p.strips_xai_cards is False


# --- Experiment 2026-08-19: the branded proxies go first ----------------------
#
# The deployment reaches five "premium" model families (ChatGPT, Perplexity,
# DeepSeek, Grok, Mistral) through self-hosted proxies, and one aggregator (Kilo)
# that serves a broad free catalogue. All of them are `tier: free` -- none costs
# money -- so `tier` cannot express the preference between them; `priority` is the
# knob that can, and that is exactly what it is for.
#
# The intent: try the branded families FIRST, rotating among themselves by
# measured quality and reliability (which is what the score does inside a single
# priority band), and fall through to Kilo only when none of them can serve right
# now. "Cannot serve" already means something precise and automatic here -- a
# route in cooldown is filtered out before ordering even runs -- so the fallback
# needs no new machinery.

BRANDED = {"chatgpt", "perplexity", "deepseek", "grok", "mistral"}


def _providers():
    return {p.id: p for p in load(YAML, {})}


def test_every_branded_proxy_sits_at_priority_zero():
    for pid in BRANDED:
        assert _providers()[pid].priority == 0, pid


def test_kilo_sits_behind_all_of_them():
    """Kilo is the safety net, not a competitor: it must lose to every branded
    provider that is available, and win the moment none of them is."""
    providers = _providers()
    assert all(providers[pid].priority < providers["kilo"].priority
               for pid in BRANDED)


def test_the_paid_tier_is_untouched_by_the_experiment():
    """The one invariant this must not dent: paid still goes last, and it gets
    there through `tier`, never through `priority`."""
    providers = _providers()
    assert providers["minimax"].tier == "paid"
    # NOT via priority -- it now TIES with kilo's 2, and that is precisely the
    # point: `tier` is what keeps it last, and no priority number can change that.
    assert providers["minimax"].priority >= providers["kilo"].priority


def test_no_branded_proxy_is_cut_off_by_its_own_timeout_before_two_minutes():
    """The point of the experiment is that the ONLY reason to skip a branded
    provider is a temporary limit (a 429, a cooldown), never the gateway giving up
    on a generation that was still coming.

    Measured live 2026-08-19 on a realistic prompt (a 12-item history, asking for
    7 grounded JSON objects): routes that answered took 46-90s, and deepseek
    (timeout_s 60) and grok (no timeout_s, so the global 90) were cut off at
    exactly their ceilings -- 60.1s and 90.1s. Those ceilings were the reason they
    lost, not anything about the models.
    """
    for pid in BRANDED:
        timeout = _providers()[pid].timeout_s
        assert timeout is not None, f"{pid} must state its own ceiling"
        assert timeout >= 120, f"{pid} is cut off at {timeout}s"


def test_reads_capabilities_defaults_to_false(tmp_path):
    yaml_path = tmp_path / "p.yaml"
    yaml_path.write_text(
        "providers:\n"
        "  - id: kilo\n    tier: free\n    dialect: openai\n"
        "    base_url: https://k.test\n    models_path: /models\n")
    assert load(str(yaml_path), {})[0].reads_capabilities is False


def test_reads_capabilities_is_read_from_the_yaml(tmp_path):
    yaml_path = tmp_path / "p.yaml"
    yaml_path.write_text(
        "providers:\n"
        "  - id: chatgpt\n    tier: free\n    dialect: openai\n"
        "    base_url: https://c.test/v1\n    models_path: /models\n"
        "    reads_capabilities: true\n")
    assert load(str(yaml_path), {})[0].reads_capabilities is True


def test_the_only_real_image_route_does_not_pin_images_through_exceptions():
    """`exceptions` is applied LAST, after the contract and after the per-model
    narrowing, so anything it names is pinned past every plan change.

    `gpt-image-1` -- not dall-e-3 -- is the id chatgpt-proxy actually publishes
    in /v1/models. Declaring `images: true` for it would keep the gateway's only
    real image route advertised at `priority: 0` after the Go plan lapses on
    2026-09-06, absorbing every image request into a 503: verbatim the failure
    the capability contract exists to end. The contract supplies `images`
    instead, already resolved against the plan.
    """
    assert _providers()["chatgpt"].exceptions["gpt-image-1"] == {
        "tools": False, "vision": False}


def test_the_real_registry_names_exactly_the_proxies_that_publish_a_contract():
    # The rollout is per proxy. mistral has not adopted the contract -- turning
    # this on before a proxy publishes /health would cost it a sweep of skipped
    # syncs. perplexity joined on 2026-08-20, the fourth to do so.
    providers = load("providers.yaml", {})
    assert sorted(p.id for p in providers if p.reads_capabilities) == [
        "chatgpt", "deepseek", "grok", "perplexity"]


def test_the_contract_url_drops_the_v1_prefix():
    """Verified against the live deployment on 2026-08-20: /v1/health answers
    404 and /health answers 200. Asking for the wrong one raised, was caught,
    and skipped the whole provider on every sweep -- forever."""
    assert contract_url("http://127.0.0.1:8890/v1") == "http://127.0.0.1:8890/health"


def test_the_contract_url_keeps_an_operator_chosen_mount_prefix():
    """`_resolve_base_url` deliberately supports a reverse-proxy mount, so the
    prefix is part of the address, not decoration: stripping to the origin would
    request a path the reverse proxy does not serve."""
    assert contract_url("https://host/proxy/v1") == "https://host/proxy/health"


def test_a_base_url_without_v1_just_gains_health():
    assert contract_url("https://k.test") == "https://k.test/health"
    assert contract_url("https://k.test/gateway/") == "https://k.test/gateway/health"


def test_the_contract_url_does_not_splinter_a_query_string():
    # Same hazard join_path exists for: CHATGPT_PROXY_URL may carry a token.
    assert contract_url("https://c.test:8888?token=abc") == \
        "https://c.test:8888/health?token=abc"


def test_only_a_whole_v1_segment_is_stripped():
    assert contract_url("https://k.test/api/v10") == "https://k.test/api/v10/health"


def _contract(**overrides):
    caps = {k: False for k in REQUIRED_CAPABILITIES}
    caps.update(chat=True, streaming=True, images=True)
    caps.update(overrides)
    return ProviderContract(version=1, provider="chatgpt",
                            auth=Auth(mode="account"), capabilities=caps)


def _chatgpt():
    return Provider("chatgpt", "free", "openai", "https://c.test/v1", "", "/models",
                    {}, [{"id": "dall-e-3", "tools": False, "vision": False,
                          "images": True, "context": 128000, "max_output": 0}])


def test_a_fixed_route_survives_when_the_contract_confirms_it():
    routes = fixed_routes(_chatgpt(), contract=_contract(images=True))
    assert [r.model_id for r in routes] == ["dall-e-3"]
    assert routes[0].capabilities.images is True


def test_a_fixed_route_is_dropped_when_the_contract_contradicts_it():
    # The event the design exists for: the Go plan lapses, `images` goes false,
    # and dall-e-3 -- which can do nothing else -- leaves the catalogue instead
    # of staying on as a chat route that answers nothing.
    assert fixed_routes(_chatgpt(), contract=_contract(images=False)) == []


def test_without_a_contract_a_fixed_route_is_untouched():
    routes = fixed_routes(_chatgpt())
    assert [r.model_id for r in routes] == ["dall-e-3"]
    assert routes[0].capabilities.images is True


def test_a_fixed_route_gains_the_provider_level_axes():
    routes = fixed_routes(_chatgpt(), contract=_contract(images=True, translate=True))
    assert routes[0].capabilities.translate is True


def test_a_fixed_route_gains_all_four_provider_level_axes_even_when_unclaimed():
    # The brief's test above only proves `translate` comes through. The other
    # three (audio_speech, audio_transcription, search) must be applied to
    # EVERY surviving fixed route too, not only when the model itself declared
    # something -- fixed_models never mentions them, so this is the only way
    # they can reach a fixed route at all.
    routes = fixed_routes(_chatgpt(), contract=_contract(
        images=True, audio_speech=True, audio_transcription=True, search=True))
    caps = routes[0].capabilities
    assert caps.audio_speech is True
    assert caps.audio_transcription is True
    assert caps.search is True


def test_an_emulated_tool_capability_is_not_read_as_a_contradiction():
    """`emulates_tools` is a claim about the GATEWAY, not about the proxy.

    deepseek declares both its models through `fixed_models`, sets
    `emulates_tools: true`, and sits at `priority: 0`. A truthful deepseek
    contract reports `tools: false` -- the proxy has no native function calling,
    llm-libre supplies it by injecting the definitions into the prompt (see
    tool_emulator.py). Testing emulation against the contract would read every
    one of those routes as contradicted and empty the provider out of the
    catalogue the moment it starts publishing /health.

    The contradiction is decided on what `fixed_models` DECLARED; emulation is
    applied afterwards, which is the order `catalog.normalize` already uses.
    """
    deepseek = Provider("deepseek", "free", "openai", "https://d.test/v1", "", "",
                        {}, [{"id": "deepseek-chat", "tools": False, "vision": False,
                              "context": 64000, "max_output": 8192}],
                        emulates_tools=True)
    routes = fixed_routes(deepseek, contract=_contract(tools=False))
    assert [r.model_id for r in routes] == ["deepseek-chat"]
    assert routes[0].capabilities.tools is True


def test_a_declared_tool_capability_the_contract_denies_still_drops_the_route():
    """The other half: emulation moving after the check must not make the check
    unreachable. A route that declares `tools: true` on its own -- no emulation
    -- against a contract reporting `tools: false` is still a contradiction."""
    provider = Provider("x", "free", "openai", "https://x.test/v1", "", "",
                        {}, [{"id": "m", "tools": True, "vision": False,
                              "context": 1000, "max_output": 100}])
    assert fixed_routes(provider, contract=_contract(tools=False)) == []


def test_grok_no_longer_pins_images_through_exceptions():
    # The gpt-image-1 defect, in its other home: `exceptions` applies LAST, so
    # a hand-declared images:true would survive a contract that says otherwise.
    # The per-model block on grok's /v1/models supplies these now.
    grok = [p for p in load("providers.yaml", {}) if p.id == "grok"][0]
    for override in grok.exceptions.values():
        assert "images" not in override


def test_emulates_tools_over_a_native_fixed_model_warns_about_the_downgrade(caplog):
    """emulates_tools claims "this provider has no native function calling".
    A fixed model declaring tools:true claims the opposite -- and acting on the
    flag silently replaces real function calling with prompt injection. The
    contradiction is knowable at load time, so it must be said at load time."""
    import logging
    p = Provider("grokish", "free", "openai", "https://g.test", "", "/models",
                 {}, [{"id": "m1", "tools": True, "vision": False,
                       "context": 32000, "max_output": 4096}],
                 emulates_tools=True)
    with caplog.at_level(logging.WARNING, logger="llm_libre.providers"):
        routes = fixed_routes(p)
    assert "grokish" in caplog.text
    assert "native" in caplog.text
    assert routes[0].capabilities.tools is True   # behaviour unchanged: flag wins


def test_emulates_tools_over_a_non_native_fixed_model_stays_silent(caplog):
    import logging
    p = Provider("deepseekish", "free", "openai", "https://d.test", "", "/models",
                 {}, [{"id": "m1", "tools": False, "vision": False,
                       "context": 32000, "max_output": 4096}],
                 emulates_tools=True)
    with caplog.at_level(logging.WARNING, logger="llm_libre.providers"):
        routes = fixed_routes(p)
    assert caplog.text == ""
    assert routes[0].capabilities.tools is True
