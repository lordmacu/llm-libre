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

Invariants, each held by construction and pinned by tests:

1. SOUNDNESS — every call this module produces names a function in
   ``allowed_names(tools, tool_choice)``, the client's own offer narrowed by
   its own constraint. No input (model output, schema, history) can widen it.
2. ``tool_choice`` is ENFORCED, not relayed. ``"none"`` can never yield a
   call; a forced function (nested or flat spelling) authorises only itself;
   an ``allowed_tools`` subset authorises exactly its members; ``"required"``
   (or forced/``mode: "required"``) unmet is a failed attempt
   (:func:`unmet_tool_demand`) — the proxy fails over instead of handing an
   agentic client prose it will misread. ``parallel_tool_calls: false`` caps
   the emitted batch at one call.
3. COVERAGE — detection reads every dialect a prompted model actually emits:
   bare JSON, fenced JSON, ``<tool_call>`` tags (one per parallel call),
   Mistral's ``[TOOL_CALLS]`` marker, JSONL runs, Python-literal dicts, ReAct
   ``action``/``action_input``, wrapper envelopes, leaked ``functions.``
   namespaces — all behind the same allow-list. Native ``tool_calls`` a
   provider produces on its own pass through untouched, streaming included.
4. Argument repair is LOSSLESS OR IDENTITY, and idempotent: values are
   re-read against the tool's own JSON Schema by JSON's grammar (``"5"`` → 5
   for an ``integer``, unions and nested structures included) and anything
   that does not convert cleanly travels exactly as the model sent it
   (:func:`_apply_schemas`).
5. TOTALITY — the parser never raises on any string input, and its work is
   bounded linearly in the response size (``_MAX_UNCLOSED_SCANS``); when a
   bound trips, the failure direction is a missed call, never an invented one.

**The central risk is a false positive.** Converting a genuine text answer into a
tool call is worse than missing a call: the client acts on a function invocation
the user never asked for. Every detection heuristic here is therefore gated on
``valid_names`` — the set of functions the client actually offered in THIS
request. A JSON object naming anything else stays text.
"""
import ast
import json
import logging
import math
import re
import time
import uuid

log = logging.getLogger("llm_libre.tool_emulator")

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

# A scratchpad cut off before its closing tag (a truncated stream, a model that
# ran out of tokens mid-thought) is reasoning to the end of the text. Left in
# place, a draft call inside it is indistinguishable from an answer.
_THINK_UNCLOSED = re.compile(
    r"<(think|thinking|reasoning)>(?:(?!</\1>).)*$", re.DOTALL | re.IGNORECASE)

# Mistral fine-tunes prefix their calls with a literal control token; what
# follows it is the ordinary JSON array the rest of the parser already reads.
# Anchored to the start on purpose: appearing anywhere else, the same bytes are
# far more likely to be argument DATA (a string value quoting the token) than a
# marker, and an unanchored sub rewrote them inside the call it was enabling. A
# genuinely mid-text marker still works without stripping -- the balanced scan
# steps past the `[TOOL_CALLS]` span and finds the array behind it.
_TOOL_CALLS_MARKER = re.compile(r"^\s*\[/?TOOL_CALLS\]\s*")

# An embedded JSON object surrounded by much more prose is far more likely to be
# the model TALKING ABOUT a call ("its schema is {...}, but which city?") than
# making one. A call the model actually intends is the bulk of its reply, so a
# span below this share of the message is treated as prose.
_EMBEDDED_MIN_SHARE = 0.5

# Some open-weights models wrap calls in an XML-ish tag instead of emitting bare
# JSON. The payload inside is still JSON, so this only strips the envelope.
_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)

# Only fence labels that MARK a call are explicit markers: `json` (and bare)
# because the injected prompt asks for raw JSON, `tool_call`/`tool_code`
# because fine-tunes that learned those labels use them for nothing else. A
# ```python fence stays a bare-scan region -- code examples full of dict
# literals must keep having to earn conversion through the prose heuristics.
_CODE_FENCE = re.compile(r"```(?:json|JSON|tool_call|tool_code)?\s*(.*?)\s*```",
                         re.DOTALL)

# Keys under which different vendors put the call arguments. Checked in this
# order; the first one present wins. `arguments` is our own documented format,
# `input` is Anthropic, `parameters`/`args` show up in open-weights fine-tunes.
_ARGUMENT_KEYS = ("arguments", "input", "parameters", "args")

# Name-key spellings, each with the argument keys that accompany it. The first
# name-ish key PRESENT decides the shape; if its value fails the allow-list the
# object is data, and no further digging is allowed -- an object that names one
# thing and wraps another did not clearly call anything. `action`/`action_input`
# is the LangChain/ReAct dialect, `tool`/`tool_input` its sibling.
_NAME_KEY_TABLE = (
    ("name", _ARGUMENT_KEYS),
    ("action", ("action_input",) + _ARGUMENT_KEYS),
    ("tool", ("tool_input",) + _ARGUMENT_KEYS),
)

# Envelope keys some models wrap a call in, e.g. {"function_call": {...}}.
_CALL_WRAPPER_KEYS = ("function_call", "tool_call", "tool_use", "function")

# Namespace prefixes OpenAI-tuned models leak from their training format
# ("functions.get_weather"). Stripping one never invents a match: the tail
# still has to clear the same allow-list.
_NAMESPACE_PREFIXES = ("functions", "tools")

# The bare scan's work bound: each unclosed opener buys one O(n) walk, so this
# caps the scan at O(_MAX_UNCLOSED_SCANS * n) on any input. See _json_candidates.
_MAX_UNCLOSED_SCANS = 8


# How deep the prompt renders nested schemas. Past this, fields are announced
# as omitted rather than silently dropped: a model told a parameter exists but
# not its shape asks; a model never told invents.
_DESCRIBE_MAX_DEPTH = 3


def _describe_params(params, indent: str = "  ", depth: int = 0) -> str:
    """Render a JSON-Schema parameter block as prose for the injected prompt.

    Recursive: nested object properties and array-item fields are what complex
    tools are made of, and a flat rendering left the model guessing exactly the
    arguments it gets wrong most. Depth-capped by _DESCRIBE_MAX_DEPTH.

    Every field is type-checked before use. This runs on a body the client sent
    verbatim -- ``api.py`` does no schema validation -- so a malformed ``tools``
    entry must degrade to a thinner prompt, never raise: an exception here
    escapes ``build_request`` inside the route loop and aborts the whole chain
    before a single upstream attempt. (That includes ``required`` lists with
    unhashable members, which a bare ``set()`` turned into exactly such an
    exception.)
    """
    if not isinstance(params, dict):
        return "  (no parameters)" if depth == 0 else ""
    props = params.get("properties")
    props = props if isinstance(props, dict) else {}
    raw_required = params.get("required")
    required = ({r for r in raw_required if isinstance(r, str)}
                if isinstance(raw_required, (list, tuple, set)) else set())
    lines = []
    for name, schema in props.items():
        if not isinstance(schema, dict):
            schema = {}
        kind = schema.get("type", "any")
        if isinstance(kind, list):
            kind = "|".join(str(k) for k in kind if isinstance(k, str)) or "any"
        line = f"{indent}- {name} ({kind}{', required' if name in required else ', optional'})"
        desc = schema.get("description")
        if isinstance(desc, str) and desc:
            line += f": {desc}"
        if isinstance(schema.get("enum"), list):
            allowed = ", ".join(json.dumps(v, ensure_ascii=False) for v in schema["enum"])
            line += f" [allowed: {allowed}]"
        if "default" in schema:
            try:
                line += f" (default: {json.dumps(schema['default'], ensure_ascii=False)})"
            except (TypeError, ValueError):
                pass
        items = schema.get("items")
        if kind == "array" and isinstance(items, dict) and isinstance(items.get("type"), str):
            line += f" (array of {items['type']})"
        lines.append(line)

        for nested in (schema, items if isinstance(items, dict) else None):
            if nested is None or not isinstance(nested.get("properties"), dict):
                continue
            if depth >= _DESCRIBE_MAX_DEPTH:
                lines.append(f"{indent}  (deeper fields omitted)")
            else:
                rendered = _describe_params(nested, indent + "  ", depth + 1)
                if rendered:
                    lines.append(rendered)
    if lines:
        return "\n".join(lines)
    return "  (no parameters)" if depth == 0 else ""


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
    subset = _allowed_tools_names(tool_choice)
    if subset is not None:
        clause = ""
        if subset:
            listed = ", ".join(f"`{n}`" for n in sorted(subset))
            clause = (f"\n\nIMPORTANT: On this turn you may only call: {listed}. "
                      "Do not call any function outside that list.")
        if isinstance(tool_choice, dict) and tool_choice.get("mode") == "required":
            clause += (" You MUST call one of them -- do not answer in prose."
                       if clause else
                       "\n\nIMPORTANT: You MUST call one of the functions listed "
                       "below. Do NOT answer in prose.")
        return clause
    forced = _forced_function_name(tool_choice)
    if forced is not None:
        return (f"\n\nIMPORTANT: You MUST call the function `{forced}` specifically. "
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


def _forced_function_name(tool_choice) -> str | None:
    """The single function a dict-shaped ``tool_choice`` forces, or None.

    Reads both spellings in the wild: the Chat Completions nested form
    ``{"type": "function", "function": {"name": X}}`` and the Responses-style
    flat form ``{"type": "function", "name": X}``. This is THE definition of
    "forced" -- allowed_names, demands_call and the prompt instruction all read
    it here, so the three can never disagree about what was forced.
    """
    if not isinstance(tool_choice, dict):
        return None
    if tool_choice.get("type") == "allowed_tools":
        return None
    fn = tool_choice.get("function")
    name = fn.get("name") if isinstance(fn, dict) else None
    if isinstance(name, str) and name:
        return name
    name = tool_choice.get("name")
    return name if isinstance(name, str) and name else None


def _allowed_tools_names(tool_choice) -> set[str] | None:
    """The subset an ``allowed_tools`` choice authorises, or None if it is not
    one. Entries are read in both spellings, like :func:`_forced_function_name`.
    An empty set is a meaningful answer: the client authorised nothing."""
    if not (isinstance(tool_choice, dict)
            and tool_choice.get("type") == "allowed_tools"):
        return None
    names = set()
    entries = tool_choice.get("tools")
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function")
        name = fn.get("name") if isinstance(fn, dict) else None
        if not (isinstance(name, str) and name):
            name = entry.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


def allowed_names(tools, tool_choice) -> set[str]:
    """The names detection may produce once ``tool_choice`` has had its say.

    ``tool_choice`` does not only demand calls, it narrows them: a client that
    forced one specific function has NOT authorised any other, an
    ``allowed_tools`` subset authorises exactly its members, and ``"none"``
    authorises nothing at all. Detection gated on this set instead of the raw
    tool list is what turns ``tool_choice`` from a hint into a contract.
    """
    names = tool_names(tools)
    if tool_choice == "none":
        return set()
    subset = _allowed_tools_names(tool_choice)
    if subset is not None:
        return names & subset
    forced = _forced_function_name(tool_choice)
    if forced is not None:
        return names & {forced}
    return names


def demands_call(tool_choice) -> bool:
    """Whether ``tool_choice`` PROMISES the client a tool call.

    OpenAI's contract: ``"required"``, a forced specific function, and an
    ``allowed_tools`` choice with ``mode: "required"`` all guarantee the
    response carries ``tool_calls``. ``auto``/``none``/absent promise nothing.
    """
    if tool_choice == "required":
        return True
    if (isinstance(tool_choice, dict)
            and tool_choice.get("type") == "allowed_tools"):
        return tool_choice.get("mode") == "required"
    return _forced_function_name(tool_choice) is not None


def unmet_tool_demand(data, tools, tool_choice) -> bool:
    """True when ``tool_choice`` promised a call and ``data`` delivers none.

    Run AFTER :func:`detect_and_convert`: by then anything that could honestly
    be read as a call has been converted, so what remains is prose -- and prose
    where the client was guaranteed ``tool_calls`` is a failed attempt, not an
    answer. The caller (proxy.py) treats it exactly like a 200 with no content:
    fail over to the next route.

    An empty allow-list mutes the check: with no callable tools (or a forced
    function the client never offered) there is nothing the model could have
    called, and failing over would retry an impossible demand forever.
    """
    if not demands_call(tool_choice):
        return False
    if not allowed_names(tools, tool_choice):
        return False
    for choice in (data.get("choices") or []) if isinstance(data, dict) else []:
        if not isinstance(choice, dict):
            continue
        msg = choice.get("message")
        if isinstance(msg, dict) and msg.get("tool_calls"):
            return False
    return True


def _build_tools_block(tools: list, tool_choice=None, parallel_tool_calls=None) -> str:
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

    # `parallel_tool_calls: false` is a knob a non-native model does not have;
    # like `tool_choice`, the constraint only exists if the prompt spells it
    # out. Anything else (true, absent, malformed) gets the documented default:
    # batching stays available via the array format.
    if parallel_tool_calls is False:
        several = ("Call at most one function per reply -- never reply with "
                   "an array of several calls.\n\n")
    else:
        several = ("To call several functions at once, reply with a JSON array "
                   "of those objects:\n"
                   '[{"name": "first", "arguments": {}}, '
                   '{"name": "second", "arguments": {}}]\n\n')

    return (
        "## How to call a function\n\n"
        "To call a function, reply with ONLY a JSON object in exactly this shape -- "
        "no prose before or after it, no markdown code fence:\n\n"
        '{"name": "<function name>", "arguments": {<arguments>}}\n\n'
        f'Example: {{"name": "{example_name}", "arguments": {{"some_param": "some value"}}}}\n\n'
        + several +
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
       the provider has no ``tool`` role. Each result is labeled with the
       function it answers (resolved from its ``tool_call_id``): once a history
       carries parallel calls, an unlabeled result is ambiguous -- the model
       cannot tell which answer belongs to which call.
    3. ``assistant`` messages carrying ``tool_calls`` are replayed as the JSON
       text the emulation documents -- see :func:`_calls_as_text`.

    ``parallel_tool_calls`` is stripped alongside ``tools``/``tool_choice`` --
    same reasoning: the provider would at best ignore the unknown field and at
    worst reject the request over it. ``false`` becomes a prompt clause, the
    only place the constraint can live for a non-native model.

    Returns a new dict. Neither the original body nor its message dicts are
    mutated: retries re-send the untouched original, and callers upstream still
    hold the client's own object.
    """
    tools = body.get("tools")
    if not tools:
        return body

    block = _build_tools_block(tools, body.get("tool_choice"),
                               body.get("parallel_tool_calls"))

    names_by_call_id = {}
    for msg in body.get("messages") or []:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for item in msg.get("tool_calls") or []:
            if not isinstance(item, dict):
                continue
            fn = item.get("function")
            call_id = item.get("id")
            name = fn.get("name") if isinstance(fn, dict) else None
            if isinstance(call_id, str) and isinstance(name, str) and name:
                names_by_call_id[call_id] = name

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
            call_id = msg.get("tool_call_id")
            name = names_by_call_id.get(call_id) if isinstance(call_id, str) else None
            label = f"[Function result for {name}]" if name else "[Function result]"
            messages.append({
                "role": "user",
                "content": f"{label}: {_tool_result_text(msg.get('content'))}",
            })
            continue

        if role == "assistant" and msg.get("tool_calls"):
            as_text = _calls_as_text(msg["tool_calls"])
            # An assistant turn may carry BOTH prose and calls. Keeping the prose
            # preserves reasoning the model may refer back to later -- including
            # prose delivered as a list of typed parts, which flattens the same
            # way a tool result's does.
            prose = msg.get("content")
            if isinstance(prose, list):
                prose = _tool_result_text(prose)
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

    result = {k: v for k, v in body.items()
              if k not in ("tools", "tool_choice", "parallel_tool_calls")}
    result["messages"] = messages
    return result


def _content_as_text(content) -> str | None:
    """Flatten response content into detectable text, or None to leave it be.

    A plain string passes through. A list of parts flattens ONLY when every
    part is text (a bare string, or a dict whose payload is its ``text``):
    flattening past a non-text part (an image) would detect against a lossy
    projection of the answer and then discard the part on conversion.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content:
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif (isinstance(p, dict) and isinstance(p.get("text"), str)
                    and p.get("type") in (None, "text")):
                parts.append(p["text"])
            else:
                return None
        return "\n".join(parts)
    return None


def detect_and_convert(data: dict, tools=None, tool_choice=None,
                       parallel_tool_calls=None) -> dict:
    """Convert text answers that are really tool calls into ``tool_calls`` shape.

    ``tools`` is the array from the ORIGINAL client request. Only functions named
    there can be produced; see the module docstring on why this allow-list is not
    optional. When ``tools`` is absent or empty nothing is converted, because
    without it there is no way to tell a tool call from a model that merely
    answered with JSON.

    ``tool_choice`` is enforced, not relayed. ``"none"`` suppresses conversion
    outright: the injected prompt already asks the model not to call anything,
    but a prompt is a request, not a guarantee -- and OpenAI's contract promises
    the client that ``none`` yields no tool calls. A forced specific function
    (or an ``allowed_tools`` subset) narrows the allow-list (see
    :func:`allowed_names`): a call to any OTHER offered tool stays text, so the
    caller can treat the attempt as an unmet demand instead of handing the
    client a call it forbade. ``parallel_tool_calls=False`` caps a detected
    batch at its first call -- the prompt asks for one, this makes it true.

    Every choice converts independently (``n > 1`` sampling), and a choice that
    already carries native ``tool_calls`` is never touched: re-parsing its prose
    commentary must not overwrite the calls the provider actually made.

    Returns ``data`` unchanged (same object) when no call is detected.
    """
    valid = allowed_names(tools, tool_choice)
    if not valid:
        return data

    choices = list(data.get("choices") or [])
    converted_any = False
    for i, choice in enumerate(choices):
        if not isinstance(choice, dict):
            continue
        msg = choice.get("message") or {}
        if not isinstance(msg, dict) or msg.get("tool_calls"):
            continue
        content = _content_as_text(msg.get("content"))
        if not isinstance(content, str) or not content.strip():
            continue

        parsed = parse_tool_calls(content, valid, schemas=tools)
        if not parsed:
            continue
        if parallel_tool_calls is False:
            parsed = parsed[:1]
        log.debug("emulated tool call detected: %s",
                  [c["name"] for c in parsed])

        choices[i] = {
            **choice,
            "message": {
                **msg,
                "content": None,
                "tool_calls": _to_openai_tool_calls(parsed),
            },
            "finish_reason": "tool_calls",
        }
        converted_any = True

    return {**data, "choices": choices} if converted_any else data


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
                       content: str | None = None,
                       raw_tool_calls: list[dict] | None = None,
                       fallback_model: str | None = None) -> dict:
    """Build the single SSE chunk that closes an emulated streaming response.

    Detection needs the whole answer, so the streaming path buffers the response
    and emits one complete chunk instead of incremental deltas. Clients
    accumulate ``tool_calls`` by ``index``, so a fully-formed entry is valid --
    just not progressive.

    ``envelope`` carries ``id``/``created``/``model`` (and any ``usage``) captured
    from the provider's own chunks, plus the provider's own ``finish_reason``.
    ``model`` comes from the provider, not from the request, so it names the
    route that really served the call rather than the alias ``auto``; when the
    provider never sent one, ``fallback_model`` (the route's own id) fills in.
    ``id`` and ``created`` are synthesized when absent -- the chunk schema marks
    them required, and a nonconforming provider must not make THIS gateway
    nonconforming too.

    The text variant keeps the provider's ``finish_reason``: masking a
    ``"length"`` as ``"stop"`` hides from the client that its answer was cut --
    exactly the case where the buffered text may be a truncated call that could
    not be detected.

    Pass ``parsed`` for an emulated tool-call chunk, ``raw_tool_calls`` for
    already-shaped native calls travelling through verbatim, or ``content`` for
    a plain-text one.
    """
    envelope = envelope or {}
    if raw_tool_calls:
        # `content` may ride along: a native provider that streams prose AND
        # calls said both, and relaying only the calls drops half its answer.
        delta = {"role": "assistant", "content": content if content else None,
                 "tool_calls": [{"index": i, **c}
                                for i, c in enumerate(raw_tool_calls)]}
        finish = "tool_calls"
    elif parsed:
        delta = {"role": "assistant", "content": None,
                 "tool_calls": [{"index": i, **c}
                                for i, c in enumerate(_to_openai_tool_calls(parsed))]}
        finish = "tool_calls"
    else:
        delta = {"role": "assistant", "content": content or ""}
        # Only reasons that make sense WITHOUT tool_calls travel through: a
        # provider claiming "tool_calls" on what is being released as text
        # would hand the client a contract violation.
        provider_finish = envelope.get("finish_reason")
        finish = (provider_finish
                  if provider_finish in ("stop", "length", "content_filter")
                  else "stop")

    chunk = {"object": "chat.completion.chunk",
             "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    for key in ("id", "created", "model", "system_fingerprint"):
        if envelope.get(key) is not None:
            chunk[key] = envelope[key]
    if "id" not in chunk:
        chunk["id"] = f"chatcmpl-emu-{uuid.uuid4().hex[:16]}"
    if "created" not in chunk:
        chunk["created"] = int(time.time())
    if "model" not in chunk and fallback_model is not None:
        chunk["model"] = fallback_model
    if envelope.get("usage") is not None:
        chunk["usage"] = envelope["usage"]
    return chunk


def accumulate_tool_call_delta(acc: dict, fragments) -> None:
    """Fold one streaming delta's ``tool_calls`` list into ``acc``.

    The OpenAI streaming contract splits each call across chunks keyed by
    ``index``: the id/type/name usually arrive once, ``function.arguments``
    arrives as string fragments to concatenate. Concatenation is the correct
    accumulation for BOTH text fields -- a name that arrives whole concatenates
    onto the empty string, a name that arrives split still assembles.

    A provider that answers an "emulated" route with native tool_calls deltas
    is a provider doing the RIGHT thing; before this existed, the buffered
    emulation path read only ``delta.content`` and silently dropped its calls
    -- while the non-streaming path passed the same calls through. One request
    shape, two different answers: exactly the asymmetry this module exists to
    remove.
    """
    if not isinstance(fragments, list):
        return
    for frag in fragments:
        if not isinstance(frag, dict):
            continue
        index = frag.get("index")
        index = index if isinstance(index, int) and index >= 0 else 0
        slot = acc.setdefault(index, {"id": None, "name": "", "arguments": ""})
        if isinstance(frag.get("id"), str) and frag["id"]:
            slot["id"] = frag["id"]
        fn = frag.get("function")
        if isinstance(fn, dict):
            if isinstance(fn.get("name"), str):
                slot["name"] += fn["name"]
            if isinstance(fn.get("arguments"), str):
                slot["arguments"] += fn["arguments"]


def assembled_tool_calls(acc: dict) -> list[dict] | None:
    """Close an accumulator into OpenAI-shaped ``tool_calls``, or None if empty.

    Calls come out ordered by ``index`` -- the order the provider declared --
    and a call whose id never arrived gets one synthesized, because clients key
    their tool results by it.
    """
    if not acc:
        return None
    return [{"id": acc[index]["id"] or f"call_emu_{uuid.uuid4().hex[:8]}",
             "type": "function",
             "function": {"name": acc[index]["name"],
                          "arguments": acc[index]["arguments"] or "{}"}}
            for index in sorted(acc)]


def _blank_out(text: str, pattern: re.Pattern) -> str:
    """Replace every match of ``pattern`` with spaces, preserving offsets.

    Lets the bare-JSON scan skip regions already claimed by an explicit marker
    (a fence or a tag) without shifting the positions of what remains.
    """
    return pattern.sub(lambda m: " " * len(m.group(0)), text)


def _json_candidates(text: str) -> list[tuple[int, str, bool]]:
    """Find every substring of ``text`` that might be a JSON call.

    Returns ``(position, candidate, qualifies)`` triples, in the order the
    caller should consider them (later entries win position ties). ``qualifies``
    is the prose-vs-invocation verdict: candidates carrying a deliberate marker
    -- a ``<tool_call>`` tag or a code fence -- always qualify, a bare object
    found loose in a paragraph has to earn it (:func:`_looks_like_an_invocation`).

    Bare scanning walks EVERY balanced span, not just the first: a schema the
    model quoted early in its reply must not shadow the real call at the end.
    It uses :func:`_balanced_span`, not a greedy ``\\{.*\\}`` (a greedy match runs
    from the first brace to the LAST one anywhere in the text, so a call
    followed by prose containing a brace became an unparseable blob), and it
    runs over a copy with fences and tags blanked out, so an illustrative
    example inside a fence is not re-discovered as if it were loose text.

    Runs of bare spans separated by nothing but whitespace are ALSO offered
    joined into one synthetic array -- the JSONL habit of models that emit one
    object per line instead of the documented array. The joined candidate is
    appended after its members with the last member's position, so it wins the
    position tie exactly when every member parses as a call (all-or-nothing,
    like any array); a run with prose between objects never groups, keeping the
    demo-then-real-call pattern on the last-one-wins rule.
    """
    found = []
    for m in _TOOL_CALL_TAG.finditer(text):
        found.append((m.start(), m.group(1).strip(), True))
    for m in _CODE_FENCE.finditer(text):
        found.append((m.start(), m.group(1).strip(), True))

    residual = _blank_out(_blank_out(text, _TOOL_CALL_TAG), _CODE_FENCE)
    spans: list[tuple[int, int, str]] = []
    # Every unclosed opener costs one walk to the end of the text before the
    # scan can step past it. Unbudgeted, a hostile storm of bare openers makes
    # the whole scan quadratic in the response size -- a model (or whoever
    # controls one) must not be able to buy O(n²) of this gateway's CPU with
    # O(n) of output. The budget keeps the total work at O(k·n) and gives up
    # in the SAFE direction: a missed call behind 8+ stray unclosed openers,
    # never an invented one.
    unclosed_budget = _MAX_UNCLOSED_SCANS
    for opener, closer in (("[", "]"), ("{", "}")):
        pos = 0
        while unclosed_budget > 0:
            hit = _balanced_span(residual, opener, closer, pos)
            if hit is None:
                break
            span, start = hit
            if span is None:
                # An opener whose span never closes (an unpaired quote, a cut
                # answer). Step past it: a real call may still follow.
                unclosed_budget -= 1
                pos = start + 1
                continue
            spans.append((start, start + len(span), span))
            pos = start + len(span)
    spans.sort(key=lambda s: s[0])

    for start, end, span in spans:
        found.append((start, span,
                      _looks_like_an_invocation(span, residual, start)))

    for run in _whitespace_runs(spans, residual):
        if len(run) < 2:
            continue
        first_start, last_end = run[0][0], run[-1][1]
        joined = "[" + ", ".join(s[2] for s in run) + "]"
        qualifies = (first_start <= 0
                     or not residual[last_end:].strip()
                     or (last_end - first_start) / len(residual) >= _EMBEDDED_MIN_SHARE)
        found.append((run[-1][0], joined, qualifies))
    return found


def _whitespace_runs(spans: list[tuple[int, int, str]],
                     text: str) -> list[list[tuple[int, int, str]]]:
    """Group sorted spans into runs separated by whitespace only.

    Overlapping spans (an object already inside a scanned array) and spans with
    prose or punctuation between them each start a new run: only true
    side-by-side emission reads as one batch.
    """
    runs: list[list[tuple[int, int, str]]] = []
    for span in spans:
        if runs:
            prev_end = runs[-1][-1][1]
            if span[0] >= prev_end and not text[prev_end:span[0]].strip():
                runs[-1].append(span)
                continue
        runs.append([span])
    return runs


def _balanced_span(text: str, opener: str, closer: str,
                   begin: int = 0) -> tuple[str | None, int] | None:
    """Scan for the next balanced ``opener``/``closer`` span from ``begin``.

    Returns ``(span, start)``; ``span`` is None when an opener was found but
    never closed (the caller may resume past it), and the whole result is None
    when no opener remains.

    Brace-counting rather than regex, and string-aware for BOTH quote styles: a
    brace inside a JSON string value (``{"q": "use } carefully"}``) must not
    close the object, the same brace inside a Python-literal string
    (``{'q': 'use } carefully'}``) must not either, and a backslash-escaped
    quote must not end the string. An apostrophe inside a double-quoted string
    (``"it's"``) does not open a single-quoted one.
    """
    start = text.find(opener, begin)
    if start == -1:
        return None
    depth = 0
    quote: str | None = None
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if quote is not None:
            if ch == quote:
                quote = None
            continue
        if ch in ('"', "'"):
            quote = ch
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:i + 1], start
    return None, start


def parse_tool_calls(text: str, valid_names, schemas=None) -> list[dict] | None:
    """Parse model output into tool calls, accepting only ``valid_names``.

    ``valid_names`` is the set of functions offered in the request (narrowed by
    ``tool_choice`` when the caller enforces one -- see :func:`allowed_names`).
    A call naming anything else is rejected outright -- that check is what
    separates "the model invoked a tool" from "the model wrote about JSON", and
    it is why an empty set can never produce a call.

    ``schemas``, when given, is the original ``tools`` array; parsed arguments
    are then repaired against each function's declared parameter types
    (:func:`_apply_schemas`) -- losslessly or not at all.

    Returns a non-empty list of ``{"name", "arguments"}`` dicts, or None.
    """
    if not valid_names or not isinstance(text, str):
        return None

    # Strip reasoning scratchpads first (closed AND cut-off ones): their braces
    # would otherwise anchor the balanced scan on content that is not the
    # answer. The Mistral [TOOL_CALLS] marker comes off next -- what it prefixes
    # is the ordinary array format.
    cleaned = _THINK_BLOCK.sub("", text)
    cleaned = _THINK_UNCLOSED.sub("", cleaned)
    cleaned = _TOOL_CALLS_MARKER.sub("", cleaned).strip()
    if not cleaned:
        return None

    # Every <tool_call> tag is a deliberate, marked invocation, so ALL of them
    # together are the answer: models trained on that format emit one tag per
    # parallel call. A tag whose payload fails the allow-list is dropped, not
    # fatal -- the calls that did qualify are still the calls the model made.
    tag_calls = []
    for m in _TOOL_CALL_TAG.finditer(cleaned):
        calls = _parse_candidate(m.group(1).strip(), valid_names)
        if calls:
            tag_calls.extend(calls)
    if tag_calls:
        return _apply_schemas(tag_calls, schemas)

    # A response that is EXACTLY a JSON array is authoritative: the model chose
    # that structure deliberately, so `_parse_candidate`'s all-or-nothing rule
    # decides it outright. Without this, a mixed array like
    # `[{valid call}, {some data}]` was rejected as a batch and then rescued by
    # the single-object fallback below -- resurrecting the one call from what is
    # much more likely a list of data.
    if cleaned.startswith("[") and isinstance(_loads_tolerant(cleaned), list):
        return _apply_schemas(_parse_candidate(cleaned, valid_names), schemas)

    # Among several parseable candidates, the LAST one wins. A model that
    # illustrates the format before committing ("here is how I would call it:
    # ```…``` -- now the real call: {…}") puts the demo first and the real call
    # last, so taking the first match handed the client the example's arguments.
    best = None
    for position, candidate, qualifies in _json_candidates(cleaned):
        if not candidate or not qualifies:
            continue
        calls = _parse_candidate(candidate, valid_names)
        if calls and (best is None or position >= best[0]):
            best = (position, calls)
    return _apply_schemas(best[1], schemas) if best else None


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


def _loads_tolerant(text: str):
    """Read JSON, falling back to a GUARDED Python-literal read.

    Weaker models emit ``repr`` output as often as JSON: single quotes,
    trailing commas, ``True``/``None``. ``ast.literal_eval`` parses exactly
    that dialect and evaluates nothing (literals only, no names, no calls), so
    the fallback adds no execution surface. The result is round-tripped through
    ``json`` so downstream code only ever sees JSON-compatible values (tuples
    and sets become arrays, non-string keys become strings); anything that
    survives neither reading is None.
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        obj = ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None
    if not isinstance(obj, (dict, list)):
        return None
    try:
        return json.loads(json.dumps(obj, default=_as_jsonable))
    except (TypeError, ValueError):
        return None


def _as_jsonable(value):
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"not JSON-compatible: {type(value).__name__}")


def _parse_candidate(text: str, valid_names) -> list[dict] | None:
    """Parse one candidate substring as a call or list of calls."""
    obj = _loads_tolerant(text)
    if obj is None:
        return None
    return _calls_from_object(obj, valid_names)


def _calls_from_object(obj, valid_names) -> list[dict] | None:
    if isinstance(obj, list):
        calls = [c for c in (_extract_call(item, valid_names) for item in obj) if c]
        # Every element must be a valid call. A list where only some entries
        # qualify is far more likely to be data the model returned than a batch
        # of calls, and converting it would invent calls the model never made.
        return calls if calls and len(calls) == len(obj) else None

    if isinstance(obj, dict):
        call = _extract_call(obj, valid_names)
        if call:
            return [call]
        # {"tool_calls": [...]} mirrors the RESPONSE shape these models were
        # fine-tuned on. Same all-or-nothing rule as any other batch.
        batch = obj.get("tool_calls")
        if isinstance(batch, list) and batch:
            return _calls_from_object(batch, valid_names)
        return None

    return None


def _extract_call(obj, valid_names) -> dict | None:
    """Normalise one JSON object into a call, across known vendor shapes.

    Accepted shapes::

        {"name": ..., "arguments": {...}}                      documented format
        {"name": ..., "input": {...}}                          Anthropic
        {"name": ..., "parameters"|"args": {...}}              open-weights variants
        {"action": ..., "action_input": {...}}                 LangChain/ReAct
        {"tool": ..., "tool_input": {...}}                     ReAct sibling
        {"function_call": {"name": ..., "arguments": ...}}     OpenAI legacy
        {"tool_call"|"tool_use": {...}} / {"type": "function", "function": {...}}

    Returns None unless the resolved name is in ``valid_names``. The first
    name-ish key PRESENT decides the shape (see _NAME_KEY_TABLE): if its value
    fails the allow-list, the object is data and no further digging happens.
    """
    if not isinstance(obj, dict):
        return None

    for name_key, argument_keys in _NAME_KEY_TABLE:
        name = obj.get(name_key)
        if isinstance(name, str) and name:
            for key in argument_keys:
                if key in obj:
                    return _normalize(name, obj[key], valid_names)
            # A name with no arguments key at all is a legitimate
            # zero-argument call.
            return _normalize(name, {}, valid_names)

    for key in _CALL_WRAPPER_KEYS:
        inner = obj.get(key)
        if isinstance(inner, dict):
            call = _extract_call(inner, valid_names)
            if call:
                return call

    return None


def _resolve_name(name: str, valid_names) -> str | None:
    """Map an emitted name onto the offered one it clearly means, or None.

    Exact match first. Then the same name with a leaked vendor namespace
    stripped (``functions.get_weather``), then a case-insensitive match --
    accepted only when it is UNIQUE: if two offered names differ only by case,
    a third spelling names neither and stays text. Every path still ends inside
    ``valid_names``; nothing here can invent a function the client did not
    offer.
    """
    candidates = [name]
    head, dot, tail = name.partition(".")
    if dot and head in _NAMESPACE_PREFIXES and tail:
        candidates.append(tail)
    for candidate in candidates:
        if candidate in valid_names:
            return candidate
    for candidate in candidates:
        matches = [n for n in valid_names if n.lower() == candidate.lower()]
        if len(matches) == 1:
            return matches[0]
    return None


def _normalize(name, arguments, valid_names) -> dict | None:
    """Validate a name/arguments pair and coerce arguments to a dict."""
    if not isinstance(name, str):
        return None
    resolved = _resolve_name(name, valid_names)
    if resolved is None:
        return None

    if isinstance(arguments, str):
        text = arguments.strip()
        if not text:
            arguments = {}
        else:
            parsed = _loads_tolerant(text)
            # Malformed argument JSON degrades to {}: the name is valid and the
            # model clearly meant to call, so the call is kept with empty
            # arguments rather than dropped -- the client gets a call it can
            # reject, instead of prose it will misread as an answer.
            arguments = parsed if isinstance(parsed, dict) else {}

    if not isinstance(arguments, dict):
        arguments = {}

    return {"name": resolved, "arguments": arguments}


def _apply_schemas(calls: list[dict] | None, schemas) -> list[dict] | None:
    """Repair argument types against each function's declared schema.

    A prompted model returns scalars as strings far more often than a native
    tool-caller does ("5" where the schema says integer). Every coercion here
    is lossless or skipped: a value that does not convert CLEANLY to the
    declared type travels exactly as the model sent it, and parameters the
    schema does not declare are never touched.
    """
    if not calls or not isinstance(schemas, (list, tuple)):
        return calls
    params_by_name = {}
    for t in schemas:
        if not isinstance(t, dict) or t.get("type") != "function":
            continue
        fn = t.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            params_by_name[fn["name"]] = fn.get("parameters")
    repaired = []
    for call in calls:
        params = params_by_name.get(call["name"])
        props = params.get("properties") if isinstance(params, dict) else None
        if isinstance(props, dict):
            call = {**call, "arguments": {
                k: _coerce_value(v, props.get(k))
                for k, v in call["arguments"].items()}}
        repaired.append(call)
    return repaired


# Coercion reads numbers by JSON's own grammar, not Python's: int("1_000") and
# float("1_2.5") succeed in Python but "1_000" is not a JSON number, so
# accepting them would not be a lossless re-reading of what the model wrote.
_JSON_INT = re.compile(r"[+-]?[0-9]+\Z")
_JSON_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")

# Bounds recursion through nested objects/array items. Client schemas arrive as
# parsed JSON (acyclic by construction), so this is belt: a depth past it just
# stops repairing, never fails.
_COERCE_MAX_DEPTH = 8


def _satisfies(value, kind: str) -> bool:
    """Whether ``value`` already inhabits JSON-Schema type ``kind``.

    ``bool`` is deliberately NOT an integer here, although Python says it is:
    JSON Schema draws the same line, and folding True into 1 would rewrite a
    flag the model may have meant literally. A whole float (5.0) is
    deliberately NOT accepted as ``integer`` either, even though JSON Schema
    would validate it: rejecting it here routes it through the coercion step,
    which folds it to the canonical int that strict consumers (pydantic and
    friends) expect.
    """
    if kind == "string":
        return isinstance(value, str)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "array":
        return isinstance(value, list)
    if kind == "object":
        return isinstance(value, dict)
    if kind == "null":
        return value is None
    return False


def _coerce_scalar(value, kind: str, kinds: list[str]):
    """One lossless conversion attempt toward ``kind``; the input on failure."""
    if kind == "integer":
        if isinstance(value, str) and _JSON_INT.match(value.strip()):
            return int(value.strip())
        if isinstance(value, float) and math.isfinite(value) and value.is_integer():
            return int(value)
    elif kind == "number":
        if isinstance(value, str) and _JSON_NUMBER.match(value.strip()):
            return float(value.strip())
    elif kind == "boolean":
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
    elif kind == "string":
        # bool is excluded: json.dumps(True) is "true", which silently rewrites
        # a flag the model may have meant literally.
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return json.dumps(value)
    elif kind == "array" or kind == "object":
        if isinstance(value, str):
            parsed = _loads_tolerant(value.strip())
            if kind == "array" and isinstance(parsed, list):
                return parsed
            if kind == "object" and isinstance(parsed, dict):
                return parsed
    elif kind == "null":
        # Only when the union has no "string" member: with one, the literal
        # text "null" is at least as plausibly the string as the null.
        if (isinstance(value, str) and value.strip().lower() == "null"
                and "string" not in kinds):
            return None
    return value


def _coerce_value(value, schema, depth: int = 0):
    """Repair one value against its schema: losslessly, or not at all.

    The decision procedure, in order:

    1. Enum repair: a string differing from exactly ONE enum member only by
       case/padding becomes that member. Two members that collide
       case-insensitively make a third spelling ambiguous -- untouched.
    2. Identity: a value already satisfying ANY declared type (``type`` may be
       a union list) is never converted -- "5" under ["integer", "string"] IS
       the string. The one exception is a whole float under ``integer``, folded
       to the canonical int.
    3. Otherwise the declared types are tried in order and the first clean
       conversion wins; no clean conversion, no change.
    4. Structure recursion: dicts repair their declared ``properties``, lists
       repair every element against ``items`` -- bounded by _COERCE_MAX_DEPTH.

    The procedure is idempotent: every repaired value satisfies its type, and
    step 2 makes satisfying values fixed points.
    """
    if not isinstance(schema, dict) or depth > _COERCE_MAX_DEPTH:
        return value

    enum = schema.get("enum")
    if (isinstance(enum, list) and isinstance(value, str) and value not in enum):
        matches = [e for e in enum
                   if isinstance(e, str) and e.lower() == value.strip().lower()]
        if len(matches) == 1:
            value = matches[0]

    declared = schema.get("type")
    if isinstance(declared, str):
        kinds = [declared]
    elif isinstance(declared, list):
        kinds = [k for k in declared if isinstance(k, str)]
    else:
        kinds = []

    if not any(_satisfies(value, k) for k in kinds):
        for kind in kinds:
            converted = _coerce_scalar(value, kind, kinds)
            if converted is not value:
                value = converted
                break
    elif ("integer" in kinds and isinstance(value, float)
            and math.isfinite(value) and value.is_integer()):
        # A whole float in a union that also allows "number" satisfies the
        # union as-is; it is still folded to the canonical integer form.
        value = int(value)

    if isinstance(value, dict):
        props = schema.get("properties")
        if isinstance(props, dict):
            value = {k: _coerce_value(v, props.get(k), depth + 1)
                     for k, v in value.items()}
    elif isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, dict):
            value = [_coerce_value(v, items, depth + 1) for v in value]
    return value
