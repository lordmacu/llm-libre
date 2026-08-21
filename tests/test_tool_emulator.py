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

YAML = str(Path(__file__).resolve().parents[1] / "providers.yaml")

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
    assert last["content"].startswith("[Function result")


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
    body = {"model": "m", "messages": ["hi", {"role": "user", "content": "hi"}],
            "tools": [WEATHER_TOOL]}
    result = inject_into_body(body)
    assert "hi" in result["messages"]


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


# --- vendor emission formats the parser must recognise ---
# Every one of these is still gated on the allow-list: coverage of MORE shapes
# never loosens the false-positive discipline, it only widens which spellings of
# a genuine call are understood.

def test_parse_batches_multiple_tool_call_tags_as_parallel_calls():
    """Hermes/Qwen-style models emit one <tool_call> tag per parallel call."""
    text = ('<tool_call>{"name": "fn1", "arguments": {"x": 1}}</tool_call>\n'
            '<tool_call>{"name": "fn2", "arguments": {"y": 2}}</tool_call>')
    result = parse_tool_calls(text, MULTI_VALID)
    assert result is not None
    assert [c["name"] for c in result] == ["fn1", "fn2"]
    assert result[0]["arguments"] == {"x": 1}
    assert result[1]["arguments"] == {"y": 2}


def test_parse_keeps_valid_tags_and_drops_invalid_ones():
    text = ('<tool_call>{"name": "not_offered", "arguments": {}}</tool_call>\n'
            '<tool_call>{"name": "get_weather", "arguments": {"city": "Lima"}}</tool_call>')
    result = parse_tool_calls(text, VALID)
    assert result is not None
    assert len(result) == 1
    assert result[0]["name"] == "get_weather"


def test_parse_mistral_tool_calls_marker():
    """Mistral fine-tunes prefix their calls with a literal [TOOL_CALLS] token."""
    text = '[TOOL_CALLS][{"name": "get_weather", "arguments": {"city": "Paris"}}]'
    result = parse_tool_calls(text, VALID)
    assert result is not None
    assert result[0]["arguments"]["city"] == "Paris"


def test_parse_keeps_the_marker_when_it_is_argument_data():
    """[TOOL_CALLS] inside a string value is the user's data, not a marker."""
    text = '{"name": "get_weather", "arguments": {"city": "[TOOL_CALLS] plaza"}}'
    result = parse_tool_calls(text, VALID)
    assert result is not None
    assert result[0]["arguments"]["city"] == "[TOOL_CALLS] plaza"


def test_parse_finds_the_call_after_a_quoted_schema():
    """The scan must not stop at the first balanced span: a schema quoted early
    in the reply used to shadow the real call at the end."""
    text = ('The tool takes {"type": "object", "properties": {"city": '
            '{"type": "string"}}} as its schema, so I will now call it:\n'
            '{"name": "get_weather", "arguments": {"city": "Lima"}}')
    result = parse_tool_calls(text, VALID)
    assert result is not None
    assert result[0]["arguments"]["city"] == "Lima"


def test_parse_python_literal_dict():
    """Weaker models emit Python repr instead of JSON: single quotes."""
    text = "{'name': 'get_weather', 'arguments': {'city': 'Bogotá'}}"
    result = parse_tool_calls(text, VALID)
    assert result is not None
    assert result[0]["arguments"]["city"] == "Bogotá"


def test_parse_python_literal_with_brace_inside_single_quoted_string():
    text = "{'name': 'get_weather', 'arguments': {'city': 'a } tricky { name'}}"
    result = parse_tool_calls(text, VALID)
    assert result is not None
    assert result[0]["arguments"]["city"] == "a } tricky { name"


def test_parse_tolerates_trailing_commas():
    text = '{"name": "get_weather", "arguments": {"city": "Quito",},}'
    result = parse_tool_calls(text, VALID)
    assert result is not None
    assert result[0]["arguments"]["city"] == "Quito"


def test_parse_strips_functions_namespace_prefix():
    """OpenAI-tuned models sometimes emit the internal `functions.` namespace."""
    text = '{"name": "functions.get_weather", "arguments": {"city": "Cali"}}'
    result = parse_tool_calls(text, VALID)
    assert result is not None
    assert result[0]["name"] == "get_weather"


def test_parse_resolves_a_unique_case_insensitive_name():
    text = '{"name": "Get_Weather", "arguments": {"city": "Cali"}}'
    result = parse_tool_calls(text, VALID)
    assert result is not None
    assert result[0]["name"] == "get_weather"


def test_parse_rejects_a_case_ambiguous_name():
    """Two offered names differing only by case: a third spelling names neither."""
    text = '{"name": "GETDATA", "arguments": {}}'
    assert parse_tool_calls(text, {"getData", "getdata"}) is None


def test_parse_react_action_shape():
    """LangChain/ReAct-trained models emit action/action_input."""
    text = '{"action": "get_weather", "action_input": {"city": "Cali"}}'
    result = parse_tool_calls(text, VALID)
    assert result is not None
    assert result[0]["name"] == "get_weather"
    assert result[0]["arguments"]["city"] == "Cali"


def test_parse_tool_and_tool_input_shape():
    text = '{"tool": "get_weather", "tool_input": {"city": "Cali"}}'
    result = parse_tool_calls(text, VALID)
    assert result is not None
    assert result[0]["arguments"]["city"] == "Cali"


def test_parse_plural_tool_calls_wrapper():
    """Some models wrap a batch in {"tool_calls": [...]}, mirroring the reply
    shape they were fine-tuned on."""
    text = ('{"tool_calls": [{"name": "fn1", "arguments": {"x": 1}}, '
            '{"name": "fn2", "arguments": {}}]}')
    result = parse_tool_calls(text, MULTI_VALID)
    assert result is not None
    assert [c["name"] for c in result] == ["fn1", "fn2"]


def test_parse_plural_wrapper_is_all_or_nothing():
    text = ('{"tool_calls": [{"name": "fn1", "arguments": {}}, '
            '{"temp": 18}]}')
    assert parse_tool_calls(text, MULTI_VALID) is None


def test_parse_unclosed_think_block_is_still_scratchpad():
    """A reasoning block truncated before its closing tag is reasoning to the
    end of the text, not an answer."""
    text = '<think>I could call {"name": "get_weather", "arguments": {"city": "X"}}'
    assert parse_tool_calls(text, VALID) is None


def test_parse_tool_code_fence_is_an_explicit_marker():
    text = ('I looked at the available functions and the forecast endpoint fits '
            'this question much better than answering from memory would, so:\n'
            '```tool_code\n{"name": "get_weather", "arguments": {"city": "Cali"}}\n```\n'
            'Once the result arrives I will summarise the conditions, compare them '
            'with the seasonal averages for the region, and suggest what to wear.')
    result = parse_tool_calls(text, VALID)
    assert result is not None
    assert result[0]["arguments"]["city"] == "Cali"


def test_parse_batches_jsonl_style_adjacent_calls():
    """Two bare calls separated only by whitespace are a parallel batch, the
    JSONL habit of models that ignore the documented array format."""
    text = ('{"name": "fn1", "arguments": {"x": 1}}\n'
            '{"name": "fn2", "arguments": {"y": 2}}')
    result = parse_tool_calls(text, MULTI_VALID)
    assert result is not None
    assert [c["name"] for c in result] == ["fn1", "fn2"]


def test_parse_prose_between_bare_objects_still_means_last_wins():
    """Prose between two bare objects is the demo-then-real pattern, not JSONL."""
    text = ('{"name": "get_weather", "arguments": {"city": "EXAMPLE"}}\n'
            'That was the format. Now the real call:\n'
            '{"name": "get_weather", "arguments": {"city": "Quito"}}')
    result = parse_tool_calls(text, VALID)
    assert result is not None
    assert len(result) == 1
    assert result[0]["arguments"]["city"] == "Quito"


# --- the new dialects stay behind the same allow-list ---

def test_parse_react_shape_rejects_unoffered_action():
    text = '{"action": "delete_everything", "action_input": {"confirm": true}}'
    assert parse_tool_calls(text, VALID) is None


def test_parse_python_literal_rejects_unoffered_name():
    text = "{'name': 'delete_everything', 'arguments': {}}"
    assert parse_tool_calls(text, VALID) is None


def test_parse_tools_namespace_prefix_also_resolves():
    text = '{"name": "tools.get_weather", "arguments": {"city": "Cali"}}'
    result = parse_tool_calls(text, VALID)
    assert result is not None
    assert result[0]["name"] == "get_weather"


def test_parse_adjacent_call_and_data_do_not_batch():
    """A JSONL run only groups when EVERY member is a call; here the valid
    call still converts alone because it opens the message."""
    text = ('{"name": "get_weather", "arguments": {"city": "Lima"}}\n'
            '{"temp": 18}')
    result = parse_tool_calls(text, VALID)
    assert result is not None
    assert len(result) == 1
    assert result[0]["name"] == "get_weather"


# --- tool_choice as an enforced contract, not a suggestion ---

def test_allowed_names_narrows_to_the_forced_function():
    from llm_libre.tool_emulator import allowed_names
    fn2 = {"type": "function", "function": {"name": "fn2", "parameters": {}}}
    tools = [WEATHER_TOOL, fn2]
    forced = {"type": "function", "function": {"name": "get_weather"}}
    assert allowed_names(tools, forced) == {"get_weather"}
    assert allowed_names(tools, "auto") == {"get_weather", "fn2"}
    assert allowed_names(tools, None) == {"get_weather", "fn2"}
    assert allowed_names(tools, "none") == set()
    unoffered = {"type": "function", "function": {"name": "ghost"}}
    assert allowed_names(tools, unoffered) == set()


def test_demands_call_variants():
    from llm_libre.tool_emulator import demands_call
    assert demands_call("required") is True
    assert demands_call({"type": "function", "function": {"name": "f"}}) is True
    assert demands_call("auto") is False
    assert demands_call("none") is False
    assert demands_call(None) is False
    assert demands_call({"type": "function", "function": {}}) is False


def test_detect_forced_function_ignores_calls_to_other_tools():
    """With tool_choice forcing fn1, a call to fn2 is not the demanded call --
    it must stay text so the gateway can treat the attempt as unmet."""
    fn1 = {"type": "function", "function": {"name": "fn1", "parameters": {}}}
    fn2 = {"type": "function", "function": {"name": "fn2", "parameters": {}}}
    data = _response_with_content('{"name": "fn2", "arguments": {}}')
    forced = {"type": "function", "function": {"name": "fn1"}}
    assert detect_and_convert(data, [fn1, fn2], forced) is data


def test_unmet_tool_demand():
    from llm_libre.tool_emulator import unmet_tool_demand
    prose = _response_with_content("It is sunny.")
    assert unmet_tool_demand(prose, [WEATHER_TOOL], "required") is True
    assert unmet_tool_demand(prose, [WEATHER_TOOL], "auto") is False
    assert unmet_tool_demand(prose, [], "required") is False
    called = detect_and_convert(
        _response_with_content('{"name": "get_weather", "arguments": {"city": "X"}}'),
        [WEATHER_TOOL], "required")
    assert unmet_tool_demand(called, [WEATHER_TOOL], "required") is False


# --- schema-aware argument coercion ---
# A prompted model returns every scalar as a string more often than a native
# one. Where the tool's own schema states the type and the conversion is
# lossless, the gateway repairs it; anything ambiguous is left exactly as sent.

TYPED_TOOL = {"type": "function", "function": {
    "name": "typed",
    "parameters": {"type": "object", "properties": {
        "count": {"type": "integer"},
        "flag": {"type": "boolean"},
        "ratio": {"type": "number"},
        "label": {"type": "string"},
        "items": {"type": "array"},
        "config": {"type": "object"},
    }},
}}


def test_detect_coerces_argument_types_per_schema():
    data = _response_with_content(
        '{"name": "typed", "arguments": {"count": "5", "flag": "true", '
        '"ratio": "0.5", "label": 7, "items": "[1, 2]", "config": "{\\"a\\": 1}"}}'
    )
    result = detect_and_convert(data, [TYPED_TOOL])
    args = json.loads(result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert args == {"count": 5, "flag": True, "ratio": 0.5,
                    "label": "7", "items": [1, 2], "config": {"a": 1}}


def test_detect_leaves_uncoercible_values_alone():
    data = _response_with_content(
        '{"name": "typed", "arguments": {"count": "five", "flag": "yes"}}'
    )
    result = detect_and_convert(data, [TYPED_TOOL])
    args = json.loads(result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert args == {"count": "five", "flag": "yes"}


def test_detect_does_not_coerce_undeclared_parameters():
    data = _response_with_content(
        '{"name": "typed", "arguments": {"extra": "5"}}'
    )
    result = detect_and_convert(data, [TYPED_TOOL])
    args = json.loads(result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert args == {"extra": "5"}


# --- parallel_tool_calls handling in injection ---

def test_inject_strips_parallel_tool_calls():
    """An emulated provider may 400 on unknown fields; the knob cannot travel."""
    body = {**BODY_WITH_TOOLS, "parallel_tool_calls": True}
    result = inject_into_body(body)
    assert "parallel_tool_calls" not in result


def test_inject_parallel_false_instructs_a_single_call():
    body = {**BODY_WITH_TOOLS, "parallel_tool_calls": False}
    result = inject_into_body(body)
    system = result["messages"][0]["content"]
    assert "one function" in system.lower()
    body_default = inject_into_body(BODY_WITH_TOOLS)
    assert "at most one" not in body_default["messages"][0]["content"].lower()


# --- tool results carry the function they answer ---

SEARCH_TOOL = {"type": "function", "function": {
    "name": "search_web",
    "description": "Search the web",
    "parameters": {"type": "object",
                   "properties": {"query": {"type": "string"}},
                   "required": ["query"]},
}}


def test_inject_labels_tool_results_with_their_function_name():
    """With parallel calls in history, an unlabeled result is ambiguous: the
    model cannot tell which answer belongs to which call."""
    body = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "Weather in Lima and news about it"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "get_weather", "arguments": '{"city": "Lima"}'}},
                {"id": "c2", "type": "function",
                 "function": {"name": "search_web", "arguments": '{"query": "Lima"}'}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": '{"temp": 18}'},
            {"role": "tool", "tool_call_id": "c2", "content": '{"hits": []}'},
        ],
        "tools": [WEATHER_TOOL, SEARCH_TOOL],
    }
    result = inject_into_body(body)
    tool_texts = [m["content"] for m in result["messages"]
                  if m["role"] == "user" and m["content"].startswith("[Function result")]
    assert len(tool_texts) == 2
    assert "get_weather" in tool_texts[0]
    assert "search_web" in tool_texts[1]


def test_inject_tool_result_with_unknown_id_keeps_generic_label():
    body = {
        "model": "m",
        "messages": [{"role": "tool", "tool_call_id": "ghost", "content": "data"}],
        "tools": [WEATHER_TOOL],
    }
    result = inject_into_body(body)
    last = result["messages"][-1]
    assert last["content"].startswith("[Function result]")


# --- live tests against the real DeepSeek proxy ---

@pytest.mark.vivo
async def test_deepseek_emulates_tool_call():
    """DeepSeek with emulation returns proper tool_calls for a weather question."""
    url = os.getenv("DEEPSEEK_PROXY_URL")
    if not url:
        pytest.skip("DEEPSEEK_PROXY_URL not set")

    from llm_libre.storage import Storage
    from llm_libre.models import Capabilities, Route
    from llm_libre.providers import load
    from llm_libre.proxy import Proxy

    providers = {p.id: p for p in load(YAML, {"DEEPSEEK_PROXY_URL": url})}
    assert providers["deepseek"].emulates_tools, "deepseek.emulates_tools not active in YAML"

    route = Route("deepseek", "deepseek-chat", "free",
                 Capabilities(tools=True, vision=False, context=64000, max_output=8192))
    store = Storage(":memory:")
    store.create_schema()

    async with httpx.AsyncClient(timeout=90) as http:
        proxy = Proxy(providers, store, http)
        r = await proxy.complete(
            [route],
            {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": "What's the weather in Bogotá?"}],
                "tools": [WEATHER_TOOL],
                "tool_choice": "required",
            },
            now=0.0,
        )

    assert r.status == 200, f"deepseek returned HTTP {r.status}: {r.json}"
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

    from llm_libre.storage import Storage
    from llm_libre.models import Capabilities, Route
    from llm_libre.providers import load
    from llm_libre.proxy import Proxy

    providers = {p.id: p for p in load(YAML, {"DEEPSEEK_PROXY_URL": url})}
    route = Route("deepseek", "deepseek-chat", "free",
                 Capabilities(tools=False, vision=False, context=64000, max_output=8192))
    store = Storage(":memory:")
    store.create_schema()

    async with httpx.AsyncClient(timeout=90) as http:
        proxy = Proxy(providers, store, http)
        r = await proxy.complete(
            [route],
            {"model": "deepseek-chat",
             "messages": [{"role": "user", "content": "Say only: hello"}]},
            now=0.0,
        )

    assert r.status == 200
    content = (r.json.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    assert content.strip(), "Empty text response"
    assert not (r.json.get("choices") or [{}])[0].get("message", {}).get("tool_calls")
