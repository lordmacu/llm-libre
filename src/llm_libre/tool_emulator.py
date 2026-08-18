"""Tool-calling emulation for providers without native function-call support.

Some providers (chatgpt-proxy, perplexity, deepseek) accept an OpenAI-shaped
request but ignore the ``tools`` field entirely, answering in prose. That is
more dangerous than an outright error: an agentic client receives text where it
expects a structured call, and nothing signals the failure.

This module closes that gap in two halves, wired in at opposite ends of a request:

- :func:`inject_into_body` (called from ``client.build_request``) rewrites the
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
#
# All three tag spellings that reasoning.py knows about are stripped here, not
# just <think>. On the normal path the trimmer has already removed them, but with
# `x_crudo: true` it is skipped entirely and detection still runs -- and a model
# that reasons "I could call {...}" then answers in prose had that discarded
# reasoning converted into a real call, destroying the actual answer.
_THINK_BLOCK = re.compile(
    r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)

# An embedded JSON object surrounded by much more prose is far more likely to be
# the model TALKING ABOUT a call ("its schema is {...}, but which city?") than
# making one. A call the model actually intends is the bulk of its reply, so a
# span below this share of the message is treated as prose.
_EMBEDDED_MIN_SHARE = 0.5

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
    """Render a JSON-Schema parameter block as prose for the injected prompt.

    Every field is type-checked before use. This runs on a body the client sent
    verbatim -- ``api.py`` does no schema validation -- so a malformed ``tools``
    entry must degrade to a thinner prompt, never raise: an exception here
    escapes ``build_request`` inside the route loop and aborts the whole chain
    before a single upstream attempt.
    """
    if not isinstance(params, dict):
        return "  (no parameters)"
    props = params.get("properties")
    props = props if isinstance(props, dict) else {}
    raw_required = params.get("required")
    required = set(raw_required) if isinstance(raw_required, (list, tuple, set)) else set()
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
        fn = tool_choice.get("function")
        name = fn.get("name") if isinstance(fn, dict) else None
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
    if not isinstance(tools, (list, tuple)):
        return names
    for t in tools:
        if not isinstance(t, dict) or t.get("type") != "function":
            continue
        fn = t.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


def _build_tools_block(tools: list, tool_choice=None) -> str:
    """Build the system-prompt section describing the callable functions."""
    described = []
    example_name = None
    for t in tools if isinstance(tools, (list, tuple)) else []:
        if not isinstance(t, dict) or t.get("type") != "function":
            continue
        f = t.get("function")
        if not isinstance(f, dict):
            continue
        name = f.get("name")
        if not isinstance(name, str) or not name:
            continue
        if example_name is None:
            example_name = name
        desc = f.get("description")
        desc = desc if isinstance(desc, str) else ""
        described.append(f"### {name}\n{desc}\nParameters:\n{_describe_params(f.get('parameters'))}")

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

    messages = []
    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            # Not our call to reject: forwarding it lets the provider answer with
            # its own 400, exactly as it would for any non-emulated route.
            # Dropping it silently altered the conversation instead.
            messages.append(msg)
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

    # A falsy block means `tools` was present but held nothing callable (an empty
    # list, or only non-function entries). The prompt gets nothing to say, but the
    # history rewriting above still had to happen -- a `tool`-role message left
    # unconverted goes to a provider that has no such role. Only the injection is
    # skipped; the fields are stripped either way, since a provider that rejects
    # unknown keys would 400 on a request we could otherwise have served.
    if block:
        if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
            head = messages[0]
            existing = head.get("content")
            existing = existing if isinstance(existing, str) else ""
            messages[0] = {**head, "content": f"{existing}\n\n{block}" if existing else block}
        else:
            messages.insert(0, {"role": "system", "content": block})

    result = {k: v for k, v in body.items() if k not in ("tools", "tool_choice")}
    result["messages"] = messages
    return result


def detect_and_convert(data: dict, tools=None, tool_choice=None) -> dict:
    """Convert a text answer that is really a tool call into ``tool_calls`` shape.

    ``tools`` is the array from the ORIGINAL client request. Only functions named
    there can be produced; see the module docstring on why this allow-list is not
    optional. When ``tools`` is absent or empty nothing is converted, because
    without it there is no way to tell a tool call from a model that merely
    answered with JSON.

    ``tool_choice="none"`` suppresses conversion outright. The injected prompt
    already asks the model not to call anything, but a prompt is a request, not a
    guarantee -- and OpenAI's contract promises the client that ``none`` yields
    no tool calls. Enforcing it here makes that true regardless of what the model
    decides to emit.

    Returns ``data`` unchanged (same object) when no call is detected.
    """
    if tool_choice == "none":
        return data
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


def build_stream_chunk(parsed: list[dict] | None, envelope: dict | None = None,
                       content: str | None = None) -> dict:
    """Build the single SSE chunk that closes an emulated streaming response.

    Detection needs the whole answer, so the streaming path buffers the response
    and emits one complete chunk instead of incremental deltas. Clients
    accumulate ``tool_calls`` by ``index``, so a fully-formed entry is valid --
    just not progressive.

    ``envelope`` carries ``id``/``created``/``model`` (and any ``usage``) captured
    from the provider's own chunks. Without it these emulated chunks were the only
    ones in the gateway missing fields the chunk schema marks required, and any
    client billing from ``usage`` silently got nothing on exactly these routes.
    ``model`` comes from the provider, not from the request, so it names the route
    that really served the call rather than the alias ``auto``.

    Pass ``parsed`` for a tool-call chunk, or ``content`` for a plain-text one.
    """
    envelope = envelope or {}
    if parsed:
        delta = {"role": "assistant", "content": None,
                 "tool_calls": [{"index": i, **c}
                                for i, c in enumerate(_to_openai_tool_calls(parsed))]}
        finish = "tool_calls"
    else:
        delta = {"role": "assistant", "content": content or ""}
        finish = "stop"

    chunk = {"object": "chat.completion.chunk",
             "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    for key in ("id", "created", "model", "system_fingerprint"):
        if envelope.get(key) is not None:
            chunk[key] = envelope[key]
    if envelope.get("usage") is not None:
        chunk["usage"] = envelope["usage"]
    return chunk


def _blank_out(text: str, pattern: re.Pattern) -> str:
    """Replace every match of ``pattern`` with spaces, preserving offsets.

    Lets the bare-JSON scan skip regions already claimed by an explicit marker
    (a fence or a tag) without shifting the positions of what remains.
    """
    return pattern.sub(lambda m: " " * len(m.group(0)), text)


def _json_candidates(text: str) -> list[tuple[int, str, bool]]:
    """Find every substring of ``text`` that might be a JSON call.

    Returns ``(position, candidate, is_explicit)`` triples. ``is_explicit`` marks
    candidates carrying a deliberate marker -- a ``<tool_call>`` tag or a code
    fence -- which are trusted regardless of how much prose surrounds them; a
    bare object found loose in a paragraph is not.

    Bare scanning uses :func:`_balanced_span`, not a greedy ``\\{.*\\}``: a greedy
    match runs from the first brace to the LAST one anywhere in the text, so a
    call followed by prose containing a brace became an unparseable blob. It also
    runs over a copy with fences and tags blanked out, so an illustrative example
    inside a fence is not re-discovered as if it were loose text.
    """
    found = []
    for m in _TOOL_CALL_TAG.finditer(text):
        found.append((m.start(), m.group(1).strip(), True))
    for m in _CODE_FENCE.finditer(text):
        found.append((m.start(), m.group(1).strip(), True))

    residual = _blank_out(_blank_out(text, _TOOL_CALL_TAG), _CODE_FENCE)
    for opener, closer in (("[", "]"), ("{", "}")):
        span = _balanced_span(residual, opener, closer)
        if span:
            found.append((residual.find(span), span, False))
    return found


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

    # Among several parseable candidates, the LAST one wins. A model that
    # illustrates the format before committing ("here is how I would call it:
    # ```…``` -- now the real call: {…}") puts the demo first and the real call
    # last, so taking the first match handed the client the example's arguments.
    best = None
    for position, candidate, is_explicit in _json_candidates(cleaned):
        if not candidate:
            continue
        if not is_explicit and not _looks_like_an_invocation(candidate, cleaned, position):
            continue
        calls = _parse_candidate(candidate, valid_names)
        if calls and (best is None or position >= best[0]):
            best = (position, calls)
    return best[1] if best else None


def _looks_like_an_invocation(span: str, whole: str, position: int) -> bool:
    """Decide whether a bare JSON object in ``whole`` is a call or just prose.

    A model that is actually calling opens with the JSON -- the injected prompt
    asks for exactly that and nothing else -- or works up to it and ends there
    ("...now the real call: {...}"). A model quoting the tool's own schema to ask
    a clarifying question ("its schema is {...}, but which city?") leaves a small
    object stranded mid-sentence with the question after it, and converting that
    destroys the question and issues a call the model never intended.

    So a bare span qualifies when it opens the message, closes it, or is most of
    it. Explicitly marked candidates skip this check entirely.
    """
    if position <= 0:
        return True
    if not whole:
        return False
    if not whole[position + len(span):].strip():
        return True
    return len(span) / len(whole) >= _EMBEDDED_MIN_SHARE


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
