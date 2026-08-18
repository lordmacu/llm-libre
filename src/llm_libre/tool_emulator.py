import json
import re
import uuid


def _describe_params(params: dict) -> str:
    """Format function parameters as readable text for the injected prompt."""
    props = params.get("properties") or {}
    required = set(params.get("required") or [])
    lines = []
    for name, schema in props.items():
        kind = schema.get("type", "any")
        desc = schema.get("description", "")
        req_label = " (required)" if name in required else " (optional)"
        line = f"  - {name} ({kind}{req_label})"
        if desc:
            line += f": {desc}"
        if "enum" in schema:
            line += f" — allowed values: {schema['enum']}"
        lines.append(line)
    return "\n".join(lines) if lines else "  (no parameters)"


def _tool_choice_instruction(tool_choice) -> str:
    """Return an extra paragraph enforcing the requested tool_choice behaviour."""
    if tool_choice == "required":
        return (
            "\n\n**REQUIRED**: You MUST call one or more of the listed functions. "
            "Do NOT respond with plain text — output only the JSON call."
        )
    if tool_choice == "none":
        return "\n\n**PROHIBITED**: Do NOT call any function. Respond with plain text only."
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        name = (tool_choice.get("function") or {}).get("name", "")
        if name:
            return (
                f"\n\n**REQUIRED**: You MUST call specifically the function `{name}`. "
                "Do not call any other function."
            )
    return ""


def _build_tools_block(tools: list, tool_choice=None) -> str:
    """Build the system-prompt block that describes available functions."""
    functions = []
    for t in tools:
        if t.get("type") != "function":
            continue
        f = t.get("function") or {}
        name = f.get("name", "")
        desc = f.get("description", "")
        params = _describe_params(f.get("parameters") or {})
        functions.append(f"### {name}\n{desc}\nParameters:\n{params}")
    if not functions:
        return ""

    fn_bodies = "\n\n".join(functions)
    first_name = (tools[0].get("function") or {}).get("name", "function_name")
    choice_note = _tool_choice_instruction(tool_choice)

    return (
        "## Function call instructions\n\n"
        "When you need to call a function, respond ONLY with valid JSON in exactly "
        "this form — no text before or after, no code-block backticks:\n\n"
        '{"name": "<function_name>", "arguments": {<arguments>}}\n\n'
        f'Example: {{"name": "{first_name}", "arguments": {{"param": "value"}}}}\n\n'
        "To call multiple functions at once, use a JSON array:\n"
        '[{"name": "fn1", "arguments": {...}}, {"name": "fn2", "arguments": {...}}]\n\n'
        f"If you do not need to call any function, respond normally with plain text.{choice_note}\n\n"
        "## Available functions\n\n"
        f"{fn_bodies}"
    )


def inject_into_body(body: dict) -> dict:
    """Transform a request body for models that lack native tool-calling support.

    - Converts the ``tools`` list into text instructions appended to the system prompt.
    - Removes ``tools`` and ``tool_choice`` from the body.
    - Converts ``tool``-role messages (function results) to ``user`` messages.
    - Converts ``assistant`` messages that carry ``tool_calls`` to plain-text JSON
      so the conversation history stays coherent for a non-native model.

    Returns a new dict; the original is not mutated.
    """
    tools = body.get("tools")
    if not tools:
        return body

    block = _build_tools_block(tools, body.get("tool_choice"))
    new_messages = []

    for msg in body.get("messages") or []:
        role = msg.get("role")

        if role == "tool":
            content = msg.get("content") or ""
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            new_messages.append({"role": "user", "content": f"[Function result]: {content}"})

        elif role == "assistant" and msg.get("tool_calls"):
            tcs = msg["tool_calls"]
            if len(tcs) == 1:
                tc_fn = tcs[0]["function"]
                try:
                    args = json.loads(tc_fn.get("arguments") or "{}")
                except (json.JSONDecodeError, ValueError):
                    args = {}
                json_text = json.dumps(
                    {"name": tc_fn.get("name", ""), "arguments": args},
                    ensure_ascii=False,
                )
            else:
                calls = []
                for item in tcs:
                    tc_fn = item["function"]
                    try:
                        args = json.loads(tc_fn.get("arguments") or "{}")
                    except (json.JSONDecodeError, ValueError):
                        args = {}
                    calls.append({"name": tc_fn.get("name", ""), "arguments": args})
                json_text = json.dumps(calls, ensure_ascii=False)
            new_messages.append({"role": "assistant", "content": json_text})

        else:
            new_messages.append(msg)

    if new_messages and new_messages[0].get("role") == "system":
        existing = new_messages[0].get("content") or ""
        new_messages[0] = {
            **new_messages[0],
            "content": (existing + "\n\n" + block) if existing else block,
        }
    else:
        new_messages.insert(0, {"role": "system", "content": block})

    result = {k: v for k, v in body.items() if k not in ("tools", "tool_choice")}
    result["messages"] = new_messages
    return result


def detect_and_convert(data: dict) -> dict:
    """Scan response content for a tool-call JSON and convert to OpenAI tool_calls format.

    Returns ``data`` unchanged if no tool call is detected.
    """
    for i, choice in enumerate(data.get("choices") or []):
        msg = choice.get("message") or {}
        content = msg.get("content")
        if not isinstance(content, str):
            continue

        parsed = parse_tool_calls(content.strip())
        if not parsed:
            continue

        tool_calls = [
            {
                "id": f"call_emu_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
                },
            }
            for tc in parsed
        ]
        new_choice = {
            **choice,
            "message": {**msg, "content": None, "tool_calls": tool_calls},
            "finish_reason": "tool_calls",
        }
        new_choices = list(data.get("choices", []))
        new_choices[i] = new_choice
        return {**data, "choices": new_choices}

    return data


def build_synthetic_stream_chunk(parsed: list[dict]) -> dict:
    """Build an SSE chunk carrying tool_calls in OpenAI streaming delta format."""
    tool_calls = [
        {
            "index": i,
            "id": f"call_emu_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": tc["name"],
                "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
            },
        }
        for i, tc in enumerate(parsed)
    ]
    return {
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls,
            },
            "finish_reason": "tool_calls",
        }]
    }


def parse_tool_calls(text: str) -> list[dict] | None:
    """Parse model output as one or more tool calls.

    Tries multiple extraction strategies in order of specificity. Returns a list
    of ``{"name": str, "arguments": dict}`` dicts, or ``None`` if no recognisable
    tool call is found.
    """
    candidates = [text]

    # Strip leading prose before the first JSON opener
    stripped = re.sub(r"^[^{\[]*", "", text).strip()
    if stripped and stripped != text:
        candidates.append(stripped)

    # Markdown code blocks
    for block in re.findall(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL):
        candidates.append(block)

    # XML-style <tool_call> tags (used by some open-source models)
    for inner in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL):
        candidates.append(inner)

    # First JSON array in the text
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        candidates.append(m.group(0))

    # First JSON object in the text
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        candidates.append(m.group(0))

    for candidate in candidates:
        result = _try_parse_candidate(candidate.strip())
        if result:
            return result
    return None


def _try_parse_candidate(text: str) -> list[dict] | None:
    """Attempt to parse ``text`` as one or several tool calls."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None

    if isinstance(obj, list):
        results = [_extract_tool_call(item) for item in obj]
        results = [r for r in results if r is not None]
        return results or None

    if isinstance(obj, dict):
        tc = _extract_tool_call(obj)
        return [tc] if tc else None

    return None


def _extract_tool_call(obj: dict) -> dict | None:
    """Extract a normalised tool call from a parsed JSON object.

    Handles several format variants produced by different LLMs:

    - ``{"name": "...", "arguments": {...}}``                   canonical (our format)
    - ``{"name": "...", "input": {...}}``                       Anthropic format
    - ``{"name": "...", "parameters": {...}}``                  some open-source models
    - ``{"name": "...", "args": {...}}``                        shorthand variant
    - ``{"function_call": {"name": "...", "arguments": "..."}}`` OpenAI legacy
    - ``{"tool_call": {"name": "...", "arguments": {...}}}``
    - ``{"type": "function", "function": {"name": "...", "arguments": "..."}}``
    """
    if not isinstance(obj, dict):
        return None

    if "name" in obj:
        for args_key in ("arguments", "input", "parameters", "args"):
            if args_key in obj:
                return _normalize_tool_call(obj["name"], obj[args_key])
        return _normalize_tool_call(obj["name"], {})

    if "function_call" in obj:
        fc = obj["function_call"]
        if isinstance(fc, dict):
            return _normalize_tool_call(fc.get("name"), fc.get("arguments", {}))

    if "tool_call" in obj:
        tc = obj["tool_call"]
        if isinstance(tc, dict):
            return _normalize_tool_call(tc.get("name"), tc.get("arguments", {}))

    if obj.get("type") == "function" and "function" in obj:
        f = obj["function"]
        if isinstance(f, dict):
            return _normalize_tool_call(f.get("name"), f.get("arguments", {}))

    return None


def _normalize_tool_call(name, arguments) -> dict | None:
    """Validate and normalise a (name, arguments) pair."""
    if not isinstance(name, str) or not name:
        return None
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(arguments, dict):
        arguments = {}
    return {"name": name, "arguments": arguments}
