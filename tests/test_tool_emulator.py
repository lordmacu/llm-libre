import json
import os
from pathlib import Path

import httpx
import pytest

from llm_libre.tool_emulator import (
    detect_and_convert,
    inject_into_body,
    parse_tool_calls,
)

YAML = str(Path(__file__).resolve().parents[1] / "proveedores.yaml")

WEATHER_TOOL = {"type": "function", "function": {
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
}}

BODY_WITH_TOOLS = {
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "What's the weather in Bogotá?"}],
    "tools": [WEATHER_TOOL],
    "tool_choice": "auto",
}

# The allow-list of function names offered by the request. Detection is gated on
# it: a JSON object naming anything outside this set stays plain text.
VALID = {"get_weather"}
MULTI_VALID = {"fn1", "fn2"}


# --- inject_into_body ---

def test_inject_removes_tools_and_tool_choice():
    result = inject_into_body(BODY_WITH_TOOLS)
    assert "tools" not in result
    assert "tool_choice" not in result


def test_inject_creates_system_message_with_instructions():
    result = inject_into_body(BODY_WITH_TOOLS)
    assert result["messages"][0]["role"] == "system"
    system = result["messages"][0]["content"]
    assert "get_weather" in system
    assert "name" in system
    assert "arguments" in system


def test_inject_preserves_existing_system_message():
    body = {
        **BODY_WITH_TOOLS,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather?"},
        ],
    }
    result = inject_into_body(body)
    system = result["messages"][0]["content"]
    assert "You are a helpful assistant." in system
    assert "get_weather" in system


def test_inject_converts_tool_messages_to_user():
    body = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": "What's the weather in Bogotá?"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "get_weather",
                                          "arguments": '{"city": "Bogotá"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": '{"temp": 18}'},
        ],
        "tools": [WEATHER_TOOL],
    }
    result = inject_into_body(body)
    roles = [m["role"] for m in result["messages"]]
    assert "tool" not in roles
    last = result["messages"][-1]
    assert last["role"] == "user"
    assert "[Function result]" in last["content"]


def test_inject_no_op_when_no_tools():
    body = {"model": "deepseek-chat",
            "messages": [{"role": "user", "content": "hello"}]}
    assert inject_into_body(body) is body


def test_inject_tool_choice_required_adds_mandatory_instruction():
    body = {**BODY_WITH_TOOLS, "tool_choice": "required"}
    result = inject_into_body(body)
    system = result["messages"][0]["content"]
    assert "REQUIRED" in system or "MUST" in system


def test_inject_tool_choice_none_adds_prohibition():
    body = {**BODY_WITH_TOOLS, "tool_choice": "none"}
    result = inject_into_body(body)
    system = result["messages"][0]["content"]
    assert "PROHIBITED" in system or "NOT" in system


def test_inject_converts_multiple_tool_calls_in_assistant_message():
    body = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": "Compare weather in Bogotá and Madrid"},
            {"role": "assistant", "content": None,
             "tool_calls": [
                 {"id": "c1", "type": "function",
                  "function": {"name": "get_weather", "arguments": '{"city": "Bogotá"}'}},
                 {"id": "c2", "type": "function",
                  "function": {"name": "get_weather", "arguments": '{"city": "Madrid"}'}},
             ]},
        ],
        "tools": [WEATHER_TOOL],
    }
    result = inject_into_body(body)
    asst_msg = next(m for m in result["messages"] if m["role"] == "assistant")
    obj = json.loads(asst_msg["content"])
    assert isinstance(obj, list)
    assert len(obj) == 2
    assert obj[0]["name"] == "get_weather"
    assert obj[1]["name"] == "get_weather"


# --- parse_tool_calls ---

def test_parse_clean_json():
    text = '{"name": "get_weather", "arguments": {"city": "Bogotá"}}'
    result = parse_tool_calls(text, VALID)
    assert result is not None
    assert result[0]["name"] == "get_weather"
    assert result[0]["arguments"]["city"] == "Bogotá"


def test_parse_json_in_markdown_block():
    text = 'I need to call:\n```json\n{"name": "get_weather", "arguments": {"city": "Madrid"}}\n```'
    result = parse_tool_calls(text, VALID)
    assert result is not None
    assert result[0]["name"] == "get_weather"


def test_parse_json_embedded_in_prose():
    text = 'To answer this I need {"name": "get_weather", "arguments": {"city": "Lima"}} with real data.'
    result = parse_tool_calls(text, VALID)
    assert result is not None
    assert result[0]["name"] == "get_weather"


def test_parse_arguments_as_json_string():
    text = '{"name": "get_weather", "arguments": "{\\"city\\": \\"Quito\\"}"}'
    result = parse_tool_calls(text, VALID)
    assert result is not None
    assert result[0]["arguments"]["city"] == "Quito"


def test_parse_returns_none_for_plain_text():
    assert parse_tool_calls("The weather in Bogotá is sunny today.", VALID) is None


def test_parse_returns_none_for_json_without_name():
    assert parse_tool_calls('{"result": "sunny"}', VALID) is None


def test_parse_array_of_tool_calls():
    text = '[{"name": "fn1", "arguments": {"x": 1}}, {"name": "fn2", "arguments": {"y": 2}}]'
    result = parse_tool_calls(text, MULTI_VALID)
    assert result is not None
    assert len(result) == 2
    assert result[0]["name"] == "fn1"
    assert result[1]["name"] == "fn2"


def test_parse_function_call_wrapper_format():
    text = '{"function_call": {"name": "get_weather", "arguments": {"city": "Bogotá"}}}'
    result = parse_tool_calls(text, VALID)
    assert result is not None
    assert result[0]["name"] == "get_weather"


def test_parse_tool_call_wrapper_format():
    text = '{"tool_call": {"name": "get_weather", "arguments": {"city": "Bogotá"}}}'
    result = parse_tool_calls(text, VALID)
    assert result is not None
    assert result[0]["name"] == "get_weather"


def test_parse_input_key_instead_of_arguments():
    text = '{"name": "get_weather", "input": {"city": "Bogotá"}}'
    result = parse_tool_calls(text, VALID)
    assert result is not None
    assert result[0]["arguments"]["city"] == "Bogotá"


def test_parse_xml_tool_call_tags():
    text = 'Here is my answer:\n<tool_call>{"name": "get_weather", "arguments": {"city": "Lima"}}</tool_call>'
    result = parse_tool_calls(text, VALID)
    assert result is not None
    assert result[0]["name"] == "get_weather"


# --- detect_and_convert ---

def _response_with_content(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def test_detect_converts_to_tool_calls():
    data = _response_with_content(
        '{"name": "get_weather", "arguments": {"city": "Bogotá"}}'
    )
    result = detect_and_convert(data, [WEATHER_TOOL])
    msg = result["choices"][0]["message"]
    assert msg["content"] is None
    tcs = msg["tool_calls"]
    assert len(tcs) == 1
    assert tcs[0]["type"] == "function"
    assert tcs[0]["function"]["name"] == "get_weather"
    args = json.loads(tcs[0]["function"]["arguments"])
    assert args["city"] == "Bogotá"


def test_detect_sets_finish_reason_tool_calls():
    data = _response_with_content(
        '{"name": "get_weather", "arguments": {"city": "Cali"}}'
    )
    result = detect_and_convert(data, [WEATHER_TOOL])
    assert result["choices"][0]["finish_reason"] == "tool_calls"


def test_detect_does_not_touch_text_response():
    data = _response_with_content("The weather in Bogotá is sunny.")
    assert detect_and_convert(data, [WEATHER_TOOL]) is data


def test_detect_generates_unique_ids():
    data = _response_with_content(
        '{"name": "get_weather", "arguments": {"city": "Medellín"}}'
    )
    r1 = detect_and_convert(data, [WEATHER_TOOL])
    r2 = detect_and_convert(data, [WEATHER_TOOL])
    id1 = r1["choices"][0]["message"]["tool_calls"][0]["id"]
    id2 = r2["choices"][0]["message"]["tool_calls"][0]["id"]
    assert id1 != id2


def test_detect_multiple_tool_calls_from_array():
    fn1 = {"type": "function", "function": {"name": "fn1", "parameters": {}}}
    fn2 = {"type": "function", "function": {"name": "fn2", "parameters": {}}}
    data = _response_with_content(
        '[{"name": "fn1", "arguments": {"x": 1}}, {"name": "fn2", "arguments": {"y": 2}}]'
    )
    result = detect_and_convert(data, [fn1, fn2])
    tcs = result["choices"][0]["message"]["tool_calls"]
    assert len(tcs) == 2
    assert tcs[0]["function"]["name"] == "fn1"
    assert tcs[1]["function"]["name"] == "fn2"


# --- false positives: the most dangerous failure mode ---

def test_detect_ignores_json_naming_an_unoffered_function():
    """A JSON call for a function the client never offered must stay text."""
    data = _response_with_content(
        '{"name": "delete_everything", "arguments": {"confirm": true}}'
    )
    assert detect_and_convert(data, [WEATHER_TOOL]) is data


def test_detect_ignores_json_when_no_tools_were_requested():
    """Without an allow-list there is no way to tell a call from JSON data."""
    data = _response_with_content(
        '{"name": "get_weather", "arguments": {"city": "Bogotá"}}'
    )
    assert detect_and_convert(data, None) is data
    assert detect_and_convert(data, []) is data


def test_parse_rejects_unoffered_name():
    text = '{"name": "some_other_function", "arguments": {}}'
    assert parse_tool_calls(text, VALID) is None


def test_parse_rejects_mixed_array_of_calls_and_data():
    """A list where only some entries are valid calls is data, not a batch."""
    text = '[{"name": "get_weather", "arguments": {}}, {"temp": 18}]'
    assert parse_tool_calls(text, VALID) is None


# --- parser robustness ---

def test_parse_ignores_braces_inside_think_block():
    """Reasoning scratchpads contain braces that must not anchor the scan."""
    text = ('<think>Maybe I should emit {"name": "wrong_fn"} here? No.</think>\n'
            '{"name": "get_weather", "arguments": {"city": "Cali"}}')
    result = parse_tool_calls(text, VALID)
    assert result is not None
    assert result[0]["arguments"]["city"] == "Cali"


def test_parse_handles_trailing_prose_with_braces():
    """A greedy regex ran to the last brace anywhere; balanced scanning must not."""
    text = ('{"name": "get_weather", "arguments": {"city": "Lima"}}\n'
            'I will format the result as {city, temp} once it arrives.')
    result = parse_tool_calls(text, VALID)
    assert result is not None
    assert result[0]["arguments"]["city"] == "Lima"


def test_parse_handles_braces_inside_string_values():
    text = '{"name": "get_weather", "arguments": {"city": "a } tricky { name"}}'
    result = parse_tool_calls(text, VALID)
    assert result is not None
    assert result[0]["arguments"]["city"] == "a } tricky { name"


def test_parse_call_without_arguments_key():
    text = '{"name": "get_weather"}'
    result = parse_tool_calls(text, VALID)
    assert result is not None
    assert result[0]["arguments"] == {}


def test_parse_malformed_argument_string_keeps_call_with_empty_args():
    text = '{"name": "get_weather", "arguments": "{broken json"}'
    result = parse_tool_calls(text, VALID)
    assert result is not None
    assert result[0]["name"] == "get_weather"
    assert result[0]["arguments"] == {}


# --- history reconstruction ---

def test_inject_keeps_assistant_prose_alongside_tool_calls():
    """An assistant turn may carry both text and calls; the text must survive."""
    body = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": "Weather in Bogotá?"},
            {"role": "assistant", "content": "Let me look that up.",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "get_weather",
                                          "arguments": '{"city": "Bogotá"}'}}]},
        ],
        "tools": [WEATHER_TOOL],
    }
    result = inject_into_body(body)
    asst = next(m for m in result["messages"] if m["role"] == "assistant")
    assert "Let me look that up." in asst["content"]
    assert "get_weather" in asst["content"]


def test_inject_flattens_list_content_in_tool_message():
    body = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "tool", "tool_call_id": "c1",
             "content": [{"type": "text", "text": '{"temp": 18}'}]},
        ],
        "tools": [WEATHER_TOOL],
    }
    result = inject_into_body(body)
    last = result["messages"][-1]
    assert '{"temp": 18}' in last["content"]
    assert "type" not in last["content"]


def test_inject_does_not_mutate_original_body():
    body = {
        "model": "deepseek-chat",
        "messages": [{"role": "system", "content": "Be brief."},
                     {"role": "user", "content": "hi"}],
        "tools": [WEATHER_TOOL],
    }
    original = json.loads(json.dumps(body))
    inject_into_body(body)
    assert body == original


def test_inject_strips_tools_even_when_none_are_callable():
    body = {"model": "deepseek-chat",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "retrieval"}],
            "tool_choice": "auto"}
    result = inject_into_body(body)
    assert "tools" not in result
    assert "tool_choice" not in result


# --- malformed client input must degrade, never raise ---
# api.py does no schema validation, so these bodies reach the emulator verbatim.
# An exception here escapes build_request inside the route loop and aborts the
# whole chain before a single upstream attempt is made.

@pytest.mark.parametrize("tools", [
    [{"type": "function", "function": "not_a_dict"}],
    [{"type": "function", "function": {"name": "f", "parameters": {"properties": [1, 2]}}}],
    [{"type": "function", "function": {"name": "f", "parameters": {"required": 5}}}],
    [{"type": "function", "function": {"name": "f", "parameters": "nope"}}],
    [{"type": "function", "function": {"name": 42}}],
    ["a bare string", None, 7],
    "not even a list",
])
def test_inject_survives_malformed_tools(tools):
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "tools": tools}
    result = inject_into_body(body)
    assert "tools" not in result


@pytest.mark.parametrize("choice", [
    {"type": "function", "function": "not_a_dict"},
    {"type": "function"},
    12345,
])
def test_inject_survives_malformed_tool_choice(choice):
    body = {**BODY_WITH_TOOLS, "tool_choice": choice}
    result = inject_into_body(body)
    assert "tool_choice" not in result


@pytest.mark.parametrize("tools", [
    [{"type": "function", "function": "not_a_dict"}],
    "not a list",
    None,
    [None, 7],
])
def test_tool_names_survives_malformed_tools(tools):
    from llm_libre.tool_emulator import tool_names
    assert tool_names(tools) == set()


def test_inject_still_converts_tool_messages_when_no_callable_tools():
    """A non-callable `tools` array must not skip history rewriting: a `tool`
    role would reach a provider that has no such role."""
    body = {
        "model": "m",
        "messages": [{"role": "tool", "tool_call_id": "c1", "content": "result"}],
        "tools": [{"type": "custom", "name": "c"}],
    }
    result = inject_into_body(body)
    assert all(m["role"] != "tool" for m in result["messages"])
    assert "[Function result]" in result["messages"][-1]["content"]


def test_inject_forwards_non_dict_messages_untouched():
    """Rejecting a malformed message is the provider's call, not ours."""
    body = {"model": "m", "messages": ["hola", {"role": "user", "content": "hi"}],
            "tools": [WEATHER_TOOL]}
    result = inject_into_body(body)
    assert "hola" in result["messages"]


# --- tool_choice: "none" is a guarantee, not a request ---

def test_detect_respects_tool_choice_none():
    data = _response_with_content(
        '{"name": "get_weather", "arguments": {"city": "Bogotá"}}'
    )
    assert detect_and_convert(data, [WEATHER_TOOL], "none") is data
    # ...and still converts under any other choice
    assert detect_and_convert(data, [WEATHER_TOOL], "auto") is not data


# --- candidate ordering and prose discrimination ---

def test_parse_prefers_the_real_call_over_an_illustrative_example():
    """A demo fence followed by the actual call must yield the actual call."""
    text = ('Here is how I would call it:\n'
            '```json\n{"name": "get_weather", "arguments": {"city": "EXAMPLE"}}\n```\n'
            'Now the real call:\n{"name": "get_weather", "arguments": {"city": "Quito"}}')
    result = parse_tool_calls(text, VALID)
    assert result is not None
    assert result[0]["arguments"]["city"] == "Quito"


def test_parse_ignores_a_schema_quoted_inside_a_clarifying_question():
    """Prose that merely quotes the tool's own JSON is not an invocation."""
    text = ('You asked about the weather tool. Its schema is '
            '{"name": "get_weather", "arguments": {}} -- but I need to know which '
            'city you mean before I can look anything up for you.')
    assert parse_tool_calls(text, VALID) is None


def test_parse_strips_reasoning_and_thinking_tags_too():
    for tag in ("thinking", "reasoning"):
        text = (f'<{tag}>I could call {{"name": "get_weather", "arguments": {{"city": "X"}}}} '
                f'but the user only wants a definition.</{tag}>\n'
                'Weather is the state of the atmosphere at a given time.')
        assert parse_tool_calls(text, VALID) is None, f"<{tag}> leaked a false call"


# --- stream chunk envelope ---

def test_stream_chunk_preserves_provider_envelope():
    from llm_libre.tool_emulator import build_stream_chunk
    envelope = {"id": "chatcmpl-abc", "created": 123, "model": "deepseek-chat",
                "usage": {"total_tokens": 42}}
    chunk = build_stream_chunk([{"name": "get_weather", "arguments": {"city": "Lima"}}],
                               envelope)
    assert chunk["id"] == "chatcmpl-abc"
    assert chunk["created"] == 123
    assert chunk["model"] == "deepseek-chat"
    assert chunk["usage"]["total_tokens"] == 42
    assert chunk["choices"][0]["finish_reason"] == "tool_calls"
    assert chunk["choices"][0]["delta"]["tool_calls"][0]["index"] == 0


def test_stream_chunk_text_variant():
    from llm_libre.tool_emulator import build_stream_chunk
    chunk = build_stream_chunk(None, {"id": "x"}, "plain answer")
    assert chunk["choices"][0]["delta"]["content"] == "plain answer"
    assert chunk["choices"][0]["finish_reason"] == "stop"
    assert chunk["id"] == "x"


# --- live tests against the real DeepSeek proxy ---

@pytest.mark.vivo
async def test_deepseek_emulates_tool_call():
    """DeepSeek with emulation returns proper tool_calls for a weather question."""
    url = os.getenv("DEEPSEEK_PROXY_URL")
    if not url:
        pytest.skip("DEEPSEEK_PROXY_URL not set")

    from llm_libre.almacen import Almacen
    from llm_libre.modelos import Capacidades, Ruta
    from llm_libre.proveedores import cargar
    from llm_libre.proxy import Proxy

    providers = {p.id: p for p in cargar(YAML, {"DEEPSEEK_PROXY_URL": url})}
    assert providers["deepseek"].emula_tools, "deepseek.emula_tools not active in YAML"

    route = Ruta("deepseek", "deepseek-chat", "gratis",
                 Capacidades(tools=True, vision=False, contexto=64000, max_salida=8192))
    store = Almacen(":memory:")
    store.crear_esquema()

    async with httpx.AsyncClient(timeout=90) as http:
        proxy = Proxy(providers, store, http)
        r = await proxy.completar(
            [route],
            {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": "What's the weather in Bogotá?"}],
                "tools": [WEATHER_TOOL],
                "tool_choice": "required",
            },
            ahora=0.0,
        )

    assert r.estado == 200, f"deepseek returned HTTP {r.estado}: {r.json}"
    msg = (r.json.get("choices") or [{}])[0].get("message", {})
    tcs = msg.get("tool_calls") or []
    assert tcs, f"No tool_calls with emulation. Response: {json.dumps(r.json, ensure_ascii=False)[:400]}"
    assert tcs[0]["function"]["name"] == "get_weather"
    args = json.loads(tcs[0]["function"]["arguments"])
    assert "city" in args, f"arguments missing 'city': {args}"


@pytest.mark.vivo
async def test_deepseek_text_response_without_tools():
    """Emulation does not break plain text requests."""
    url = os.getenv("DEEPSEEK_PROXY_URL")
    if not url:
        pytest.skip("DEEPSEEK_PROXY_URL not set")

    from llm_libre.almacen import Almacen
    from llm_libre.modelos import Capacidades, Ruta
    from llm_libre.proveedores import cargar
    from llm_libre.proxy import Proxy

    providers = {p.id: p for p in cargar(YAML, {"DEEPSEEK_PROXY_URL": url})}
    route = Ruta("deepseek", "deepseek-chat", "gratis",
                 Capacidades(tools=False, vision=False, contexto=64000, max_salida=8192))
    store = Almacen(":memory:")
    store.crear_esquema()

    async with httpx.AsyncClient(timeout=90) as http:
        proxy = Proxy(providers, store, http)
        r = await proxy.completar(
            [route],
            {"model": "deepseek-chat",
             "messages": [{"role": "user", "content": "Say only: hello"}]},
            ahora=0.0,
        )

    assert r.estado == 200
    content = (r.json.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    assert content.strip(), "Empty text response"
    assert not (r.json.get("choices") or [{}])[0].get("message", {}).get("tool_calls")
