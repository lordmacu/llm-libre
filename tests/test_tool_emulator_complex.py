"""
Complex live tests for tool-calling emulation with deepseek-chat.
All tests are @pytest.mark.vivo — require DEEPSEEK_PROXY_URL to be set.
"""
import json
import os
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.vivo

YAML = str(Path(__file__).resolve().parents[1] / "proveedores.yaml")

WEATHER_TOOL = {"type": "function", "function": {
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string", "description": "City name"},
                                  "units": {"type": "string", "enum": ["celsius", "fahrenheit"],
                                            "description": "Temperature unit"}},
                   "required": ["city"]},
}}

SEARCH_TOOL = {"type": "function", "function": {
    "name": "search_web",
    "description": "Search the internet for information on any topic",
    "parameters": {"type": "object",
                   "properties": {"query": {"type": "string", "description": "Search query"},
                                  "max_results": {"type": "integer", "description": "Max results (1-10)"}},
                   "required": ["query"]},
}}

CALC_TOOL = {"type": "function", "function": {
    "name": "calculate",
    "description": "Evaluate mathematical expressions and perform complex calculations",
    "parameters": {"type": "object",
                   "properties": {"expression": {"type": "string",
                                                 "description": "Math expression to evaluate"},
                                  "precision": {"type": "integer",
                                                "description": "Decimal places in result"}},
                   "required": ["expression"]},
}}

ALL_TOOLS = [WEATHER_TOOL, SEARCH_TOOL, CALC_TOOL]


def _build_proxy(url: str):
    from llm_libre.almacen import Almacen
    from llm_libre.modelos import Capacidades, Ruta
    from llm_libre.providers import load
    from llm_libre.proxy import Proxy

    providers = {p.id: p for p in load(YAML, {"DEEPSEEK_PROXY_URL": url})}
    route = Ruta("deepseek", "deepseek-chat", "gratis",
                 Capacidades(tools=True, vision=False, contexto=64000, max_salida=8192))
    store = Almacen(":memory:")
    store.crear_esquema()
    return providers, route, store


def _tool_calls(response) -> list:
    return ((response.json.get("choices") or [{}])[0]
            .get("message", {}).get("tool_calls") or [])


def _content(response) -> str:
    return ((response.json.get("choices") or [{}])[0]
            .get("message", {}).get("content") or "")


def _skip_if_no_url():
    url = os.getenv("DEEPSEEK_PROXY_URL")
    if not url:
        pytest.skip("DEEPSEEK_PROXY_URL not set")
    return url


async def test_selects_correct_tool_from_multiple():
    """With 3 tools available, the model should pick the right one for each question."""
    url = _skip_if_no_url()
    providers, route, store = _build_proxy(url)

    cases = [
        ("What is the square root of 144?", "calculate"),
        ("What's the weather in Medellín right now?", "get_weather"),
        ("Search for information about the DeepSeek V3 model", "search_web"),
    ]

    async with httpx.AsyncClient(timeout=90) as http:
        from llm_libre.proxy import Proxy
        proxy = Proxy(providers, store, http)

        for question, expected_tool in cases:
            r = await proxy.completar(
                [route],
                {"model": "deepseek-chat",
                 "messages": [{"role": "user", "content": question}],
                 "tools": ALL_TOOLS,
                 "tool_choice": "required"},
                ahora=0.0,
            )
            assert r.estado == 200, f"HTTP {r.estado} for: {question}"
            tcs = _tool_calls(r)
            assert tcs, f"No tool_calls for: {question!r}"
            name = tcs[0]["function"]["name"]
            assert name == expected_tool, (
                f"For {question!r} expected '{expected_tool}', got '{name}'"
            )
            args = json.loads(tcs[0]["function"]["arguments"])
            print(f"\n  {question!r} → {name}({args})")


async def test_includes_optional_arguments_when_relevant():
    """When the question mentions units, the model should include the optional 'units' param."""
    url = _skip_if_no_url()
    providers, route, store = _build_proxy(url)

    async with httpx.AsyncClient(timeout=90) as http:
        from llm_libre.proxy import Proxy
        proxy = Proxy(providers, store, http)

        r = await proxy.completar(
            [route],
            {"model": "deepseek-chat",
             "messages": [{"role": "user",
                           "content": "What temperature in Fahrenheit is it in New York?"}],
             "tools": [WEATHER_TOOL],
             "tool_choice": "required"},
            ahora=0.0,
        )
        assert r.estado == 200
        tcs = _tool_calls(r)
        assert tcs, "No tool_calls generated"
        args = json.loads(tcs[0]["function"]["arguments"])
        assert "city" in args
        print(f"\n  args received: {args}")
        if "units" in args:
            print(f"  included 'units': {args['units']}")
        else:
            print("  did not include 'units' (acceptable, it was optional)")


async def test_two_turn_agentic_loop():
    """
    Two-turn agentic loop:
    1. User asks about weather in Bogotá
    2. Model calls get_weather → inject fake result
    3. Model gives final text response
    """
    url = _skip_if_no_url()
    providers, route, store = _build_proxy(url)

    async with httpx.AsyncClient(timeout=90) as http:
        from llm_libre.proxy import Proxy
        proxy = Proxy(providers, store, http)

        messages = [{"role": "user", "content": "What's the weather like in Bogotá today?"}]

        r1 = await proxy.completar(
            [route],
            {"model": "deepseek-chat",
             "messages": messages,
             "tools": [WEATHER_TOOL],
             "tool_choice": "required"},
            ahora=0.0,
        )
        assert r1.estado == 200, f"Turn 1 failed: {r1.json}"
        tcs = _tool_calls(r1)
        assert tcs, "Turn 1: no tool_calls generated"

        tc = tcs[0]
        call_id = tc["id"]
        fn_name = tc["function"]["name"]
        args = json.loads(tc["function"]["arguments"])
        print(f"\n  Turn 1 — called: {fn_name}({args})")

        fake_result = json.dumps({
            "city": args.get("city", "Bogotá"),
            "temperature": 18,
            "condition": "partly cloudy",
            "humidity": 72,
            "wind_kmh": 15,
        })

        messages = messages + [
            {"role": "assistant", "content": None, "tool_calls": [tc]},
            {"role": "tool", "tool_call_id": call_id, "content": fake_result},
        ]

        r2 = await proxy.completar(
            [route],
            {"model": "deepseek-chat",
             "messages": messages,
             "tools": [WEATHER_TOOL]},
            ahora=0.0,
        )
        assert r2.estado == 200, f"Turn 2 failed: {r2.json}"
        final_text = _content(r2)
        assert final_text.strip(), "Turn 2: empty text response"

        tcs2 = _tool_calls(r2)
        print(f"  Turn 2 — final response: {final_text[:200]!r}")
        if not tcs2:
            assert ("18" in final_text or "cloudy" in final_text.lower()
                    or "bogot" in final_text.lower()), (
                f"Response does not mention weather data: {final_text!r}"
            )


async def test_three_turn_comparison_loop():
    """
    Multi-turn loop to compare weather in two cities:
    Model calls get_weather twice (one per city) then gives a text comparison.
    """
    url = _skip_if_no_url()
    providers, route, store = _build_proxy(url)

    async with httpx.AsyncClient(timeout=90) as http:
        from llm_libre.proxy import Proxy
        proxy = Proxy(providers, store, http)

        messages = [{"role": "user",
                     "content": "Compare the weather in Bogotá and Madrid. "
                                "I need temperature and conditions for both."}]

        fake_weather = {
            "bogotá": {"temperature": 18, "condition": "partly cloudy", "humidity": 72},
            "madrid": {"temperature": 34, "condition": "sunny and hot", "humidity": 25},
        }

        calls_made = []

        for turn in range(3):
            r = await proxy.completar(
                [route],
                {"model": "deepseek-chat",
                 "messages": messages,
                 "tools": [WEATHER_TOOL]},
                ahora=0.0,
            )
            # The free DeepSeek proxy intermittently answers 200 with an empty
            # body under consecutive calls -- its own session degrading, not an
            # emulation failure. Skipping keeps this test measuring what it is
            # for (multi-turn coherence) instead of the provider's uptime.
            if r.estado == 503 and "sin contenido" in str(r.json):
                pytest.skip("provider returned an empty 200; not an emulation failure")
            assert r.estado == 200, f"Turn {turn + 1} HTTP {r.estado}"

            tcs = _tool_calls(r)
            if not tcs:
                text = _content(r)
                print(f"\n  Turn {turn + 1} — final response ({len(text)} chars):")
                print(f"  {text[:300]!r}")
                assert text.strip(), "Final response is empty"
                print(f"  Calls made: {calls_made}")
                assert len(calls_made) >= 1, "No tool calls were made"
                return

            assistant_msg = {"role": "assistant", "content": None, "tool_calls": tcs}
            messages = messages + [assistant_msg]

            for tc in tcs:
                fn_name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"])
                city = args.get("city", "").lower()
                calls_made.append(f"{fn_name}({args})")
                print(f"  Turn {turn + 1} — called: {fn_name}({args})")

                result = next(
                    (v for k, v in fake_weather.items() if k in city or city in k),
                    {"temperature": 20, "condition": "unknown"}
                )
                result["city"] = args.get("city", city)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result),
                })

        text = _content(r)
        print(f"\n  Loop ended at turn {turn + 1}. Calls: {calls_made}")
        if text:
            print(f"  Last text: {text[:200]!r}")
