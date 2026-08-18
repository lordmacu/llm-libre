"""Tool-calling emulation for providers without native function-call support.

Some providers (chatgpt-proxy, perplexity, deepseek) accept an OpenAI-shaped
request but ignore the ``tools`` field entirely, answering in prose. That is
more dangerous than an outright error: an agentic client receives text where it
expects a structured call, and nothing signals the failure.

This module closes that gap in two halves, wired in at opposite ends of a request:

- :func:`inject_into_body` (called from ``cliente.armar_peticion``) rewrites the
  request so a non-native model can participate: ``tools`` becomes prose in the
  system prompt, and any tool-call history is replayed as plain text.
- :func:`detect_and_convert` (called from ``proxy.completar``) reads the model's
  text answer back and, when it is a tool call, rebuilds the OpenAI
  ``tool_calls`` shape the client is waiting for.

The emulation is transparent: the router sees ``tools=True`` for these routes and
clients see a normal OpenAI response either way.

**The central risk is a false positive.** Converting a genuine text answer into a
tool call is worse than missing a call: the client acts on a function invocation
the user never asked for. Every detection heuristic here is therefore gated on
``valid_names`` — the set of functions the client actually offered in THIS
request. A JSON object naming anything else stays text.
"""
import json
import re
import uuid

# Reasoning models (deepseek-reasoner and friends) emit a scratchpad before the
# answer. It routinely contains braces -- draft JSON, dict literals, prose about
# the schema -- so it has to come off before any brace scanning, or the scan
# locks onto a brace that belongs to the thinking, not the answer.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Some open-weights models wrap calls in an XML-ish tag instead of emitting bare
# JSON. The payload inside is still JSON, so this only strips the envelope.
_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)

_CODE_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)\s*```", re.DOTALL)

# Keys under which different vendors put the call arguments. Checked in this
# order; the first one present wins. `arguments` is our own documented format,
# `input` is Anthropic, `parameters`/`args` show up in open-weights fine-tunes.
_ARGUMENT_KEYS = ("arguments", "input", "parameters", "args")

# Envelope keys some models wrap a call in, e.g. {"function_call": {...}}.
_CALL_WRAPPER_KEYS = ("function_call", "tool_call", "function")


def _describe_params(params: dict) -> str:
    """Render a JSON-Schema parameter block as prose for the injected prompt."""
    props = params.get("properties") or {}
    required = set(params.get("required") or [])
    lines = []
    for name, schema in props.items():
        if not isinstance(schema, dict):
            schema = {}
        kind = schema.get("type", "any")
        line = f"  - {name} ({kind}{' , required' if name in required else ', optional'})"
        desc = schema.get("description")
        if desc:
            line += f": {desc}"
        if isinstance(schema.get("enum"), list):
            allowed = ", ".join(json.dumps(v, ensure_ascii=False) for v in schema["enum"])
            line += f" [allowed: {allowed}]"
        items = schema.get("items")
        if kind == "array" and isinstance(items, dict) and items.get("type"):
            line += f" (array of {items['type']})"
        lines.append(line)
    return "\n".join(lines) if lines else "  (no parameters)"


def _tool_choice_instruction(tool_choice) -> str:
    """Translate an OpenAI ``tool_choice`` value into an explicit prompt clause.

    A non-native model has no ``tool_choice`` knob, so the constraint only exists
    if it is spelled out in the prompt. Without this, ``required`` silently
    degraded to ``auto`` -- the model kept the option of answering in prose.
    """
    if tool_choice == "required":
        return ("\n\nIMPORTANT: You MUST call one of the functions listed below. "
                "Do NOT answer in prose. Output only the JSON call.")
    if tool_choice == "none":
        return ("\n\nIMPORTANT: Do NOT call any function on this turn. "
                "Answer in plain text only.")
    if isinstance(tool_choice, dict):
        name = (tool_choice.get("function") or {}).get("name")
        if isinstance(name, str) and name:
            return (f"\n\nIMPORTANT: You MUST call the function `{name}` specifically. "
                    "Do not call any other function, and do not answer in prose.")
    return ""


def tool_names(tools) -> set[str]:
    """Names of the function tools in an OpenAI ``tools`` array.

    This is the allow-list that keeps detection honest: only a JSON object whose
    ``name`` appears here may become a tool call. Anything else the model emits
    -- including well-formed JSON with a ``name`` key -- stays text.
    """
    names = set()
    for t in tools or []:
        if not isinstance(t, dict) or t.get("type") != "function":
            continue
        name = (t.get("function") or {}).get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


def _build_tools_block(tools: list, tool_choice=None) -> str:
    """Build the system-prompt section describing the callable functions."""
    described = []
    example_name = None
    for t in tools:
        if not isinstance(t, dict) or t.get("type") != "function":
            continue
        f = t.get("function") or {}
        name = f.get("name")
        if not isinstance(name, str) or not name:
            continue
        if example_name is None:
            example_name = name
        desc = f.get("description") or ""
        params = f.get("parameters")
        body = _describe_params(params if isinstance(params, dict) else {})
        described.append(f"### {name}\n{desc}\nParameters:\n{body}")

    if not described:
        return ""

    return (
        "## How to call a function\n\n"
        "To call a function, reply with ONLY a JSON object in exactly this shape -- "
        "no prose before or after it, no markdown code fence:\n\n"
        '{"name": "<function name>", "arguments": {<arguments>}}\n\n'
        f'Example: {{"name": "{example_name}", "arguments": {{"some_param": "some value"}}}}\n\n'
        "To call several functions at once, reply with a JSON array of those objects:\n"
        '[{"name": "first", "arguments": {}}, {"name": "second", "arguments": {}}]\n\n'
        "Use only the function names listed below, spelled exactly as shown. "
        "If no function is needed, just answer normally in plain text."
        f"{_tool_choice_instruction(tool_choice)}\n\n"
        "## Available functions\n\n"
        + "\n\n".join(described)
    )


def _tool_result_text(content) -> str:
    """Flatten a ``tool`` message's content into text.

    The OpenAI schema allows content to be a list of typed parts, not just a
    string. Interpolating the list straight into an f-string would hand the model
    a Python repr instead of the result it is supposed to reason about.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(p.get("text") or "")
            elif isinstance(p, str):
                parts.append(p)
        return " ".join(x for x in parts if x)
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def _calls_as_text(tool_calls: list) -> str:
    """Re-render assistant ``tool_calls`` as the JSON the emulation asked for.

    A non-native model does not understand the ``tool_calls`` field, so on a
    follow-up turn it sees an assistant message with no visible content and
    re-issues the call it already made. Replaying the calls as text in the exact
    format the system prompt documents keeps multi-turn history coherent.

    All calls are rendered, not just the first: dropping the rest of a parallel
    call left the model with one call in its history but several results.
    """
    rendered = []
    for item in tool_calls:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") or {}
        raw = fn.get("arguments")
        if isinstance(raw, dict):
            args = raw
        else:
            try:
                args = json.loads(raw or "{}")
            except (json.JSONDecodeError, ValueError, TypeError):
                args = {}
            if not isinstance(args, dict):
                args = {}
        rendered.append({"name": fn.get("name") or "", "arguments": args})
    if not rendered:
        return ""
    payload = rendered[0] if len(rendered) == 1 else rendered
    return json.dumps(payload, ensure_ascii=False)


def inject_into_body(body: dict) -> dict:
    """Rewrite a request so a model without native tool calling can answer it.

    Three transformations, all required for multi-turn agentic loops to work:

    1. ``tools`` (plus any ``tool_choice`` constraint) becomes prose appended to
       the system prompt, and both fields are dropped from the body -- the
       provider would ignore them, and some reject unknown fields outright.
    2. ``tool``-role messages become ``user`` messages carrying the result, since
       the provider has no ``tool`` role.
    3. ``assistant`` messages carrying ``tool_calls`` are replayed as the JSON
       text the emulation documents -- see :func:`_calls_as_text`.

    Returns a new dict. Neither the original body nor its message dicts are
    mutated: retries re-send the untouched original, and callers upstream still
    hold the client's own object.
    """
    tools = body.get("tools")
    if not tools:
        return body

    block = _build_tools_block(tools, body.get("tool_choice"))
    if not block:
        # `tools` was present but held nothing callable (empty list, or only
        # non-function entries). Still strip the fields -- a provider that
        # rejects unknown keys would 400 on a request we could have served.
        return {k: v for k, v in body.items() if k not in ("tools", "tool_choice")}

    messages = []
    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")

        if role == "tool":
            messages.append({
                "role": "user",
                "content": f"[Function result]: {_tool_result_text(msg.get('content'))}",
            })
            continue

        if role == "assistant" and msg.get("tool_calls"):
            as_text = _calls_as_text(msg["tool_calls"])
            # An assistant turn may carry BOTH prose and calls. Keeping the prose
            # preserves reasoning the model may refer back to later.
            prose = msg.get("content")
            content = f"{prose}\n{as_text}".strip() if isinstance(prose, str) and prose.strip() else as_text
            messages.append({"role": "assistant", "content": content})
            continue

        messages.append(msg)

    if messages and messages[0].get("role") == "system":
        head = messages[0]
        existing = head.get("content")
        existing = existing if isinstance(existing, str) else ""
        messages[0] = {**head, "content": f"{existing}\n\n{block}" if existing else block}
    else:
        messages.insert(0, {"role": "system", "content": block})

    result = {k: v for k, v in body.items() if k not in ("tools", "tool_choice")}
    result["messages"] = messages
    return result


def detect_and_convert(data: dict, tools=None) -> dict:
    """Convert a text answer that is really a tool call into ``tool_calls`` shape.

    ``tools`` is the array from the ORIGINAL client request. Only functions named
    there can be produced; see the module docstring on why this allow-list is not
    optional. When ``tools`` is absent or empty nothing is converted, because
    without it there is no way to tell a tool call from a model that merely
    answered with JSON.

    Returns ``data`` unchanged (same object) when no call is detected.
    """
    valid = tool_names(tools)
    if not valid:
        return data

    for i, choice in enumerate(data.get("choices") or []):
        if not isinstance(choice, dict):
            continue
        msg = choice.get("message") or {}
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            continue

        parsed = parse_tool_calls(content, valid)
        if not parsed:
            continue

        converted = {
            **choice,
            "message": {
                **msg,
                "content": None,
                "tool_calls": _to_openai_tool_calls(parsed),
            },
            "finish_reason": "tool_calls",
        }
        choices = list(data.get("choices") or [])
        choices[i] = converted
        return {**data, "choices": choices}

    return data


def _to_openai_tool_calls(parsed: list[dict]) -> list[dict]:
    """Wrap parsed calls in the OpenAI ``tool_calls`` structure."""
    return [{
        "id": f"call_emu_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {
            "name": call["name"],
            "arguments": json.dumps(call["arguments"], ensure_ascii=False),
        },
    } for call in parsed]


def build_stream_chunk(parsed: list[dict], model: str = "") -> dict:
    """Build the single SSE chunk that delivers emulated calls to a streaming client.

    Detection needs the whole answer, so the streaming path buffers the response
    and emits one complete chunk rather than incremental argument deltas. Clients
    accumulate ``tool_calls`` by ``index``, so a fully-formed entry is valid --
    just not progressive.
    """
    calls = _to_openai_tool_calls(parsed)
    chunk = {
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": None,
                      "tool_calls": [{"index": i, **c} for i, c in enumerate(calls)]},
            "finish_reason": "tool_calls",
        }],
    }
    if model:
        chunk["model"] = model
    return chunk


def _json_candidates(text: str):
    """Yield substrings of ``text`` that might be a JSON call, best guess first.

    Ordered most-specific to least so an explicit signal (a fenced block, a
    ``<tool_call>`` tag) is tried before a loose scan of the prose around it.

    The final two use :func:`_balanced_span`, not a greedy ``\\{.*\\}``. A greedy
    match runs from the first brace to the LAST one anywhere in the text, so a
    call followed by any prose containing a brace produced an unparseable blob.
    """
    stripped = text.strip()
    yield stripped

    for block in _CODE_FENCE.findall(text):
        yield block.strip()

    for tagged in _TOOL_CALL_TAG.findall(text):
        yield tagged.strip()

    array = _balanced_span(text, "[", "]")
    if array:
        yield array

    obj = _balanced_span(text, "{", "}")
    if obj:
        yield obj


def _balanced_span(text: str, opener: str, closer: str) -> str | None:
    """Return the first balanced ``opener``/``closer`` span, or None.

    Brace-counting rather than regex, and string-aware: a brace inside a JSON
    string value (``{"q": "use } carefully"}``) must not close the object, and a
    backslash-escaped quote must not end the string.
    """
    start = text.find(opener)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_tool_calls(text: str, valid_names) -> list[dict] | None:
    """Parse model output into tool calls, accepting only ``valid_names``.

    ``valid_names`` is the set of functions offered in the request. A call naming
    anything else is rejected outright -- that check is what separates "the model
    invoked a tool" from "the model wrote about JSON", and it is why an empty
    set can never produce a call.

    Returns a non-empty list of ``{"name", "arguments"}`` dicts, or None.
    """
    if not valid_names or not isinstance(text, str):
        return None

    # Strip reasoning scratchpads first: their braces would otherwise anchor the
    # balanced scan on content that is not the answer.
    cleaned = _THINK_BLOCK.sub("", text).strip()
    if not cleaned:
        return None

    # A response that is EXACTLY a JSON array is authoritative: the model chose
    # that structure deliberately, so `_parse_candidate`'s all-or-nothing rule
    # decides it outright. Without this, a mixed array like
    # `[{valid call}, {some data}]` was rejected as a batch and then rescued by
    # the single-object fallback below -- resurrecting the one call from what is
    # much more likely a list of data.
    if cleaned.startswith("["):
        try:
            if isinstance(json.loads(cleaned), list):
                return _parse_candidate(cleaned, valid_names)
        except (json.JSONDecodeError, ValueError):
            pass

    for candidate in _json_candidates(cleaned):
        if not candidate:
            continue
        calls = _parse_candidate(candidate, valid_names)
        if calls:
            return calls
    return None


def _parse_candidate(text: str, valid_names) -> list[dict] | None:
    """Parse one candidate substring as a call or list of calls."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None

    if isinstance(obj, list):
        calls = [c for c in (_extract_call(item, valid_names) for item in obj) if c]
        # Every element must be a valid call. A list where only some entries
        # qualify is far more likely to be data the model returned than a batch
        # of calls, and converting it would invent calls the model never made.
        return calls if calls and len(calls) == len(obj) else None

    if isinstance(obj, dict):
        call = _extract_call(obj, valid_names)
        return [call] if call else None

    return None


def _extract_call(obj, valid_names) -> dict | None:
    """Normalise one JSON object into a call, across known vendor shapes.

    Accepted shapes::

        {"name": ..., "arguments": {...}}                      documented format
        {"name": ..., "input": {...}}                          Anthropic
        {"name": ..., "parameters"|"args": {...}}              open-weights variants
        {"function_call": {"name": ..., "arguments": ...}}     OpenAI legacy
        {"tool_call": {...}} / {"type": "function", "function": {...}}

    Returns None unless the resolved name is in ``valid_names``.
    """
    if not isinstance(obj, dict):
        return None

    name = obj.get("name")
    if isinstance(name, str) and name:
        for key in _ARGUMENT_KEYS:
            if key in obj:
                return _normalize(name, obj[key], valid_names)
        # A name with no arguments key at all is a legitimate zero-argument call.
        return _normalize(name, {}, valid_names)

    for key in _CALL_WRAPPER_KEYS:
        inner = obj.get(key)
        if isinstance(inner, dict):
            call = _extract_call(inner, valid_names)
            if call:
                return call

    return None


def _normalize(name, arguments, valid_names) -> dict | None:
    """Validate a name/arguments pair and coerce arguments to a dict."""
    if not isinstance(name, str) or name not in valid_names:
        return None

    if isinstance(arguments, str):
        text = arguments.strip()
        if not text:
            arguments = {}
        else:
            try:
                arguments = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                # Malformed argument JSON. The name is valid and the model
                # clearly meant to call, so the call is kept with empty
                # arguments rather than dropped -- the client gets a call it can
                # reject, instead of prose it will misread as an answer.
                arguments = {}

    if not isinstance(arguments, dict):
        arguments = {}

    return {"name": name, "arguments": arguments}
