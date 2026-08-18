import os
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.vivo

YAML = str(Path(__file__).resolve().parents[1] / "providers.yaml")

# Ids that must NEVER appear in the discovered catalogue: the legacy aliases
# chatgpt-proxy adds for compatibility, and "auto", reserved by llm-libre itself
# (it collides with its own alias in parse_request).
_IDS_THAT_MUST_NOT_APPEAR = {"gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo", "auto"}


def _chatgpt_provider(url: str):
    # `base_url` is resolved exactly as in production (providers.load), the URL is
    # not rebuilt by hand: this way the tests use exactly the same path (including
    # base_url_env normalisation) as the real gateway.
    from llm_libre.providers import load
    return next(p for p in load(YAML, {"CHATGPT_PROXY_URL": url}) if p.id == "chatgpt")


async def _discovered_chatgpt_routes(chatgpt) -> list:
    """Discover chatgpt-proxy's real catalogue -- NEVER a hand-wired id, which is
    precisely the rule this Task exists to enforce. Shared by the two tests below
    so neither has to guess a model: the review found that a hardcoded id
    ("gpt-5-3-mini") which one day stops existing makes the test fail with the
    WRONG diagnosis ("it blew up when sent tools, is the old 500 back?") instead
    of saying what actually happened (the id no longer exists)."""
    from llm_libre.catalog import normalize
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.get(chatgpt.base_url.rstrip("/") + chatgpt.models_path)
    assert r.status_code == 200, "chatgpt-proxy stopped answering /v1/models"
    routes = normalize("chatgpt", r.json(), priority=chatgpt.priority,
                       default_capabilities=chatgpt.default_capabilities)
    assert routes, "chatgpt-proxy's discovered catalogue came back empty"
    return routes


async def test_chatgpt_proxy_answers_a_real_chat_when_configured():
    # It skips cleanly if CHATGPT_PROXY_URL is unset: this proxy is an in-house
    # service (blog), it is not always up, and unlike Kilo/OpenRouter there is no
    # fixed public URL to hit every time.
    url = os.getenv("CHATGPT_PROXY_URL")
    if not url:
        pytest.skip("CHATGPT_PROXY_URL is not configured")
    chatgpt = _chatgpt_provider(url)

    # The catalogue is DISCOVERED (see the Task 13 follow-up): this confirms
    # against the real proxy that it still returns something usable and that the
    # aliases and the reserved id still do not slip through.
    routes = await _discovered_chatgpt_routes(chatgpt)
    ids = {x.model_id for x in routes}
    assert not ids & _IDS_THAT_MUST_NOT_APPEAR
    assert all(x.capabilities.tools is False for x in routes)

    # The id used for the chat comes from the catalogue JUST DISCOVERED, not from
    # a constant in this file -- the same "no hardcoded ids" principle that
    # governs all of providers.yaml.
    model_id = sorted(ids)[0]
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(chatgpt.base_url.rstrip("/") + "/chat/completions",
                         json={"model": model_id,
                               "messages": [{"role": "user", "content": "say hi"}]})
    assert r.status_code == 200, f"chatgpt-proxy stopped answering anonymous chat ({model_id})"
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    assert isinstance(content, str) and content.strip()


async def test_chatgpt_proxy_does_not_really_do_function_calling():
    # The fact that sustains tools:false in providers.yaml: the user reported
    # "we already have tools enabled" and this was verified by running it (not by
    # re-reading) -- the proxy no longer returns HTTP 500 when sent tools, but with
    # tool_choice:"required" it still returns tool_calls:None and prose.
    #
    # tools:false is a claim about ALL of chatgpt's discovered routes, not about
    # one particular model (a review finding): a backend that gained function
    # calling on a model other than the one tested would leave this test green
    # with the YAML already lying. EVERY id the real catalogue returned is
    # exercised, not just one.
    url = os.getenv("CHATGPT_PROXY_URL")
    if not url:
        pytest.skip("CHATGPT_PROXY_URL is not configured")
    from llm_libre.quality_suite import WEATHER_TOOL

    chatgpt = _chatgpt_provider(url)
    routes = await _discovered_chatgpt_routes(chatgpt)

    for route in routes:
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(chatgpt.base_url.rstrip("/") + "/chat/completions",
                             json={"model": route.model_id,
                                   "messages": [{"role": "user",
                                                 "content": "Que clima hace en Bogota?"}],
                                   "tools": [WEATHER_TOOL], "tool_choice": "required"})
        # Sending it tools no longer blows up (that changed, and it is fine): what
        # is verified is that the RESPONSE is still not real function calling. The
        # message distinguishes the two ways of failing -- the real status code
        # says whether the old 500 is back, whether the id stopped existing (404),
        # or something else -- instead of assuming a single cause.
        assert r.status_code == 200, (
            f"chatgpt-proxy returned HTTP {r.status_code} for {route.model_id} when sent "
            "tools (check whether the old 500 is back or the id no longer exists)")
        msg = r.json()["choices"][0]["message"]
        assert not msg.get("tool_calls"), (
            f"chatgpt-proxy returned tool_calls for {route.model_id}: if this happened, the "
            "anonymous backend started genuinely supporting function calling on that model "
            "and tools:false in providers.yaml should be reconsidered")
        assert isinstance(msg.get("content"), str) and msg["content"].strip()


async def test_kilo_still_accepts_anonymous_requests():
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post("https://api.kilo.ai/api/gateway/chat/completions",
                         json={"model": "kilo-auto/free", "max_tokens": 8,
                               "messages": [{"role": "user", "content": "ping"}]})
    assert r.status_code == 200, "Kilo's anonymous tier stopped working"


async def test_the_kilo_catalogue_carries_free_models_with_tools():
    from llm_libre.catalog import normalize
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.get("https://api.kilo.ai/api/gateway/models")
    routes = normalize("kilo", r.json())
    assert any(x.capabilities.tools for x in routes)
