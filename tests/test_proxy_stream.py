import asyncio
import json
import logging
import time
from pathlib import Path

import httpx

from llm_libre.storage import Storage
from llm_libre.models import Capabilities, Route
from llm_libre.providers import Provider, load
from llm_libre.proxy import Proxy

YAML_REAL = str(Path(__file__).resolve().parents[1] / "providers.yaml")

BODY = {"model": "auto", "messages": [], "stream": True}


def _route(model="a:free", provider="kilo"):
    return Route(provider, model, "free", Capabilities(True, False, 100000, 4096))


def _sse(*chunks):
    lines = []
    for t in chunks:
        lines.append('data: {"choices":[{"delta":{"content":"%s"}}]}\n\n' % t)
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def _sse_json(*chunks):
    # Like _sse, but encoding the content with json.dumps: needed for chunks
    # carrying REAL quotes or newlines (e.g. a canvas fence marker), which _sse
    # breaks by interpolating them raw inside the JSON.
    lines = [f'data: {{"choices":[{{"delta":{{"content":{json.dumps(t)}}}}}]}}\n\n'
             for t in chunks]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def _proxy(handler, canvas=frozenset()):
    store = Storage(":memory:")
    store.create_schema()
    prov = {
        "kilo": Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [],
                          unwraps_canvas="kilo" in canvas),
        "chatgpt": Provider("chatgpt", "free", "openai", "https://cg.test", "", "/models",
                             {}, [], unwraps_canvas="chatgpt" in canvas),
    }
    return Proxy(prov, store, httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def _collect(gen):
    text = ""
    async for line in gen:
        if not line.startswith("data: ") or "[DONE]" in line:
            continue
        obj = json.loads(line[6:])
        text += (obj.get("choices", [{}])[0].get("delta", {}) or {}).get("content", "")
    return text


class _BrokenStream(httpx.AsyncByteStream):
    """Simulates a connection that delivers one real chunk and then drops."""

    def __init__(self, chunk: bytes):
        self._chunk = chunk

    async def __aiter__(self):
        yield self._chunk
        raise httpx.ReadError("conexion cortada a mitad de stream")


class _DelayedStream(httpx.AsyncByteStream):
    """Lets real time pass before the first chunk and again before [DONE], so a
    test can tell whether ttft is measured at the first token or at the stream's
    close."""

    def __init__(self, delay_before: float, delay_after: float):
        self._delay_before = delay_before
        self._delay_after = delay_after

    async def __aiter__(self):
        await asyncio.sleep(self._delay_before)
        yield b'data: {"choices":[{"delta":{"content":"hola"}}]}\n\n'
        await asyncio.sleep(self._delay_after)
        yield b"data: [DONE]\n\n"


async def test_it_passes_the_full_content_through():
    p = _proxy(lambda req: httpx.Response(200, content=_sse("ho", "la")))
    assert await _collect(p.complete_stream([_route()], BODY, 0.0)) == "hola"


async def test_it_trims_reasoning_split_across_chunks():
    p = _proxy(lambda req: httpx.Response(200, content=_sse("ho", "<thi", "nk>zz</think>", "la")))
    assert await _collect(p.complete_stream([_route()], BODY, 0.0)) == "hola"


async def test_a_tag_that_never_closes_does_not_hang_the_stream():
    p = _proxy(lambda req: httpx.Response(200, content=_sse("ok", "<think>", "sin cerrar")))
    assert await _collect(p.complete_stream([_route()], BODY, 0.0)) == "ok"


async def test_it_unwraps_a_canvas_fence_split_across_chunks():
    # chatgpt-proxy (Task 13) leaks ChatGPT's canvas mode: the opening/closing
    # marker travels split across chunks just like <think>, but here the content
    # INSIDE is the answer -- it is not lost. This only happens because the route
    # belongs to "chatgpt" (canvas={"chatgpt"}, see finding 1 of the review
    # below).
    p = _proxy(lambda req: httpx.Response(
        200, content=_sse_json(':::writing{title="x"}\n', 'ho', 'la\n', ':::')),
        canvas={"chatgpt"})
    text = await _collect(p.complete_stream([_route(provider="chatgpt")], BODY, 0.0))
    assert text == "hola\n"


# --- Finding 1 of the review: the same case, but streaming and with a route
#     WITHOUT canvas unwrapping (kilo, the default) -- the Docusaurus/MDX
#     documentation markers must survive intact, split across chunks included. ---

async def test_a_provider_without_canvas_unwrapping_leaves_markers_alone_when_streaming():
    p = _proxy(lambda req: httpx.Response(
        200, content=_sse_json(":::note\n", "Guarda el ", "token en el .env.\n", ":::")))
    text = await _collect(p.complete_stream([_route(provider="kilo")], BODY, 0.0))
    assert text == ":::note\nGuarda el token en el .env.\n:::"


async def test_it_fails_over_when_the_first_route_fails_before_emitting():
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429)
        return httpx.Response(200, content=_sse("bien"))

    p = _proxy(handler)
    assert await _collect(p.complete_stream([_route("a:free"), _route("b:free")], BODY, 0.0)) == "bien"


async def test_it_always_ends_with_done():
    p = _proxy(lambda req: httpx.Response(200, content=_sse("x")))
    lines = [l async for l in p.complete_stream([_route()], BODY, 0.0)]
    assert lines[-1].strip() == "data: [DONE]"


async def test_no_routes_emits_an_error_and_closes():
    p = _proxy(lambda req: httpx.Response(200, content=_sse("x")))
    lines = [l async for l in p.complete_stream([], BODY, 0.0)]
    assert any("error" in l for l in lines)
    assert lines[-1].strip() == "data: [DONE]"


async def test_raw_mode_does_not_trim_the_content():
    p = _proxy(lambda req: httpx.Response(200, content=_sse("<think>mmm</think>hola")))
    text = await _collect(p.complete_stream([_route()], BODY, 0.0, raw=True))
    assert text == "<think>mmm</think>hola"


async def test_it_does_not_discard_a_tool_calls_chunk_with_empty_content():
    # In OpenAI-style streaming a tool_calls chunk usually travels with
    # content="". The original brief's bug discarded it entirely with a `continue`
    # that only looked at whether the trimmed content came out empty.
    body = (b'data: {"choices":[{"delta":{"content":"",'
             b'"tool_calls":[{"index":0,"id":"call_1","function":{"name":"buscar"}}]}}]}\n\n'
             b'data: [DONE]\n\n')
    p = _proxy(lambda req: httpx.Response(200, content=body))
    lines = [l async for l in p.complete_stream([_route()], BODY, 0.0)]
    useful = [l for l in lines if "[DONE]" not in l]
    assert len(useful) == 1
    obj = json.loads(useful[0][len("data: "):])
    assert obj["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "buscar"


async def test_it_does_not_discard_a_chunk_with_finish_reason_and_empty_content():
    # The final chunk of a normal stream carries empty content and only
    # finish_reason: it must not be lost. (Fix round 3, B1: this test used to send
    # ONLY that chunk, a stream with no answer inside, which today counts as a
    # failed attempt -- which is why it is now preceded by real content, the way
    # the case actually occurs.)
    #
    # Fix round 4, N2: it also used a shape OpenAI's real protocol does NOT
    # produce -- `finish_reason` INSIDE the delta. In the real protocol it is a
    # SIBLING of `delta`, and with that shape the chunk was being lost. That is why
    # the bug went unnoticed: the only test covering it did not use the real
    # shape. Now it does.
    body = (b'data: {"choices":[{"index":0,"delta":{"content":"hola"}}]}\n\n'
             b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
             b'data: [DONE]\n\n')
    p = _proxy(lambda req: httpx.Response(200, content=body))
    lines = [l async for l in p.complete_stream([_route()], BODY, 0.0)]
    useful = [l for l in lines if "[DONE]" not in l]
    assert len(useful) == 2
    obj = json.loads(useful[-1][len("data: "):])
    assert obj["choices"][0]["finish_reason"] == "stop"   # hermano de delta


async def test_it_does_not_fail_over_when_the_connection_drops_after_emitting():
    calls = []

    def handler(req):
        calls.append(1)
        chunk = b'data: {"choices":[{"delta":{"content":"real"}}]}\n\n'
        return httpx.Response(200, stream=_BrokenStream(chunk))

    p = _proxy(handler)
    lines = [l async for l in p.complete_stream([_route("a:free"), _route("b:free")], BODY, 0.0)]
    # Only the first route was tried: once real content has reached the client,
    # switching to the second route would mix two responses.
    assert len(calls) == 1
    assert any("real" in l for l in lines)
    assert lines[-1].strip() == "data: [DONE]"
    # And the failure after emitting must not add a second event on top of the
    # one already recorded when the first useful chunk went out.
    rows = p.store._con.execute("SELECT ok FROM events WHERE key = ?",
                                   ("kilo/a:free",)).fetchall()
    assert rows == [(1,)]


async def test_it_does_not_discard_a_chunk_with_an_empty_but_present_tool_calls():
    # "tool_calls": [] is a falsy value, but the KEY is present: filtering by the
    # value's truthiness (instead of by the key's presence) would throw it away as
    # if it carried nothing, losing the signal that a tool call is in progress.
    body = (b'data: {"choices":[{"delta":{"content":"","tool_calls":[]}}]}\n\n'
             b'data: [DONE]\n\n')
    p = _proxy(lambda req: httpx.Response(200, content=body))
    lines = [l async for l in p.complete_stream([_route()], BODY, 0.0)]
    useful = [l for l in lines if "[DONE]" not in l]
    assert len(useful) == 1
    obj = json.loads(useful[0][len("data: "):])
    assert obj["choices"][0]["delta"]["tool_calls"] == []


async def test_a_pure_reasoning_stream_records_a_failed_event():
    # Fix round 3, B1 (Blocking). This test used to assert the opposite: that a
    # stream delivering NOTHING useful still counts as a success because "the HTTP
    # call did work". That is exactly the hole: the client is left with no answer
    # while the route's reliability GOES UP, /health stays "ok" and there is no
    # failover. What counts as success is having delivered content or tool_calls,
    # not having received a 200.
    p = _proxy(lambda req: httpx.Response(
        200, content=_sse("<think>solo razonamiento</think>")))
    text = await _collect(p.complete_stream([_route()], BODY, 0.0))
    assert text == ""
    rows = p.store._con.execute("SELECT ok FROM events").fetchall()
    assert rows == [(0,)]


async def test_a_stream_without_useful_content_fails_over():
    # A reasoning model's empty 200 when streaming: role and finish_reason
    # chunks, not one letter of content. It must fall through to the next route,
    # not close with [DONE] as if it had answered.
    calls = []
    empty_stream = (b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n'
             b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
             b'data: [DONE]\n\n')

    def handler(req):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(200, content=empty_stream)
        return httpx.Response(200, content=_sse("bien"))

    p = _proxy(handler)
    text = await _collect(p.complete_stream([_route("a:free"), _route("b:free")], BODY, 0.0))
    assert text == "bien"
    assert len(calls) == 2


async def test_a_stream_without_useful_content_records_a_failure_on_that_route():
    empty_stream = (b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n'
             b'data: [DONE]\n\n')

    def handler(req):
        if "a:free" in req.content.decode():
            return httpx.Response(200, content=empty_stream)
        return httpx.Response(200, content=_sse("bien"))

    p = _proxy(handler)
    await _collect(p.complete_stream([_route("a:free"), _route("b:free")], BODY, 0.0))
    rows = p.store._con.execute(
        "SELECT key, ok FROM events ORDER BY key").fetchall()
    assert rows == [("kilo/a:free", 0), ("kilo/b:free", 1)]


async def test_a_tool_calls_only_stream_still_counts_as_a_success():
    # A legitimate case that must NOT break: a function-calling stream may carry
    # not one letter of content.
    body = (b'data: {"choices":[{"delta":{"role":"assistant","content":"",'
              b'"tool_calls":[{"index":0,"id":"c1","function":{"name":"buscar"}}]}}]}\n\n'
              b'data: [DONE]\n\n')
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(200, content=body)

    p = _proxy(handler)
    lines = [l async for l in p.complete_stream([_route("a:free"), _route("b:free")],
                                                  BODY, 0.0)]
    assert len(calls) == 1          # no failover happened
    assert any("buscar" in l for l in lines)
    rows = p.store._con.execute("SELECT ok FROM events").fetchall()
    assert rows == [(1,)]


async def test_a_raw_pure_reasoning_stream_is_still_a_success():
    # With x_raw the client asked for the text verbatim: the <think> IS the answer.
    p = _proxy(lambda req: httpx.Response(200, content=_sse("<think>mmm</think>")))
    text = await _collect(p.complete_stream([_route()], BODY, 0.0, raw=True))
    assert text == "<think>mmm</think>"
    rows = p.store._con.execute("SELECT ok FROM events").fetchall()
    assert rows == [(1,)]


async def test_failing_before_emitting_records_the_event_only_once():
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429)
        return httpx.Response(200, content=_sse("ok"))

    p = _proxy(handler)
    await _collect(p.complete_stream([_route("a:free"), _route("b:free")], BODY, 0.0))
    rows = p.store._con.execute(
        "SELECT key, ok FROM events ORDER BY key").fetchall()
    assert rows == [("kilo/a:free", 0), ("kilo/b:free", 1)]


async def test_ttft_measures_the_first_token_not_the_end_of_the_stream():
    delay_before, delay_after = 0.05, 0.2

    def handler(req):
        return httpx.Response(200, stream=_DelayedStream(delay_before, delay_after))

    p = _proxy(handler)
    t0 = time.monotonic()
    lines = [l async for l in p.complete_stream([_route()], BODY, 0.0)]
    total_duration_ms = (time.monotonic() - t0) * 1000

    assert any("hola" in l for l in lines)
    assert lines[-1].strip() == "data: [DONE]"

    rows = p.store._con.execute("SELECT ok, ttft_ms FROM events").fetchall()
    assert rows == [(1, rows[0][1])]  # exactly one event
    ttft_ms = rows[0][1]

    # The whole stream takes ~(delay_before + delay_after) = 250ms, but the first
    # token comes out at ~50ms. If the ttft had been measured at the end of the
    # stream (the regression this test detects) would sit close to
    # total_duration_ms instead of staying near delay_before.
    assert ttft_ms < total_duration_ms - 100
    assert ttft_ms < (delay_before * 1000) + 100


async def test_the_streaming_path_does_write_a_ttft():
    # The flip side of I5: here the ttft CAN be measured, and it is the only path
    # that writes that column.
    p = _proxy(lambda req: httpx.Response(200, content=_sse("hola")))
    await _collect(p.complete_stream([_route()], BODY, 0.0))
    row = p.store._con.execute("SELECT ttft_ms, latency_ms FROM events").fetchone()
    assert row[0] >= 0 and row[1] is None


async def test_a_whitespace_only_chunk_is_not_lost():
    # Los deltas de streaming vienen partidos y muchos son " " o "\n" sueltos.
    # They do not count as a "useful response" when deciding success, but they
    # ARE client-facing text: they cannot be thrown away. They are held back and
    # go out, in order, alongside the first chunk with real content.
    p = _proxy(lambda req: httpx.Response(200, content=_sse(" ", "hola", " ", "mundo")))
    assert await _collect(p.complete_stream([_route()], BODY, 0.0)) == " hola mundo"


async def test_a_pure_whitespace_stream_is_not_a_response():
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(200, content=_sse(" ", "\n", "  "))
        return httpx.Response(200, content=_sse("bien"))

    p = _proxy(handler)
    assert await _collect(p.complete_stream([_route("a:free"), _route("b:free")],
                                            BODY, 0.0)) == "bien"


# --- Fix round 4, N2: the "chunk with nothing useful" guard only looked inside
#     `delta`. In OpenAI's real protocol, `finish_reason` is a SIBLING of `delta`
#     and the `usage` chunk arrives with `choices: []`, so both were being
#     discarded silently. It did not bite with Kilo or OpenRouter because both put
#     `role` in every delta, but it does bite with any strict provider -- MiniMax's
#     OpenAI dialect, or the Groq/Cerebras the design plans to add. Silent data
#     loss in a contract whose entire premise is "change only base_url". ---

ENVELOPE = '"id":"c1","object":"chat.completion.chunk","created":1,"model":"m"'


async def test_it_does_not_lose_openais_real_final_chunk():
    body = (b'data: {' + ENVELOPE.encode() +
              b',"choices":[{"index":0,"delta":{"role":"assistant","content":"hola"}}]}\n\n'
              b'data: {' + ENVELOPE.encode() +
              b',"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
              b'data: [DONE]\n\n')
    p = _proxy(lambda req: httpx.Response(200, content=body))
    lines = [l async for l in p.complete_stream([_route()], BODY, 0.0)]
    assert any('"finish_reason": "stop"' in l or '"finish_reason":"stop"' in l
               for l in lines), "se perdio el chunk de finish_reason"


async def test_it_does_not_lose_the_usage_chunk():
    # stream_options.include_usage sends a final chunk with empty choices.
    body = (b'data: {' + ENVELOPE.encode() +
              b',"choices":[{"index":0,"delta":{"content":"hola"}}]}\n\n'
              b'data: {' + ENVELOPE.encode() +
              b',"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1,'
              b'"total_tokens":4}}\n\n'
              b'data: [DONE]\n\n')
    p = _proxy(lambda req: httpx.Response(200, content=body))
    lines = [l async for l in p.complete_stream([_route()], BODY, 0.0)]
    assert any("total_tokens" in l for l in lines), "the usage chunk was lost"


async def test_the_repeated_envelope_does_not_make_a_reasoning_chunk_useful():
    # The flip side: if "carries something besides content" were measured on the
    # chunk
    # whole, the envelope keys (id/object/created/model/index) -- which repeat
    # IDENTICALLY in every chunk -- would make every already-trimmed reasoning
    # chunk look useful. A pure-reasoning stream would stop failing over (a
    # regression of B1) and the buffer would fill up for nothing.
    reasoning_stream = b"".join(
        b'data: {' + ENVELOPE.encode() +
        b',"choices":[{"index":0,"delta":{"content":"' + t + b'"}}]}\n\n'
        for t in (b"<think>", b"pienso ", b"y pienso", b"</think>"))
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(200, content=reasoning_stream + b"data: [DONE]\n\n")
        return httpx.Response(200, content=_sse("bien"))

    p = _proxy(handler)
    text = await _collect(p.complete_stream([_route("a:free"), _route("b:free")],
                                             BODY, 0.0))
    assert text == "bien"
    assert len(calls) == 2
    rows = p.store._con.execute(
        "SELECT key, ok FROM events ORDER BY key").fetchall()
    assert rows == [("kilo/a:free", 0), ("kilo/b:free", 1)]


# --- Fix round 4, Minor: descartar chunks retenidos deja de ser silencioso. ---

async def test_it_warns_when_discarding_the_held_chunks_of_a_failed_attempt(caplog):
    empty_stream = (b'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n'
             b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}\n\n'
             b'data: [DONE]\n\n')
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(200, content=empty_stream)
        return httpx.Response(200, content=_sse("bien"))

    p = _proxy(handler)
    with caplog.at_level(logging.INFO, logger="llm_libre.proxy"):
        text = await _collect(p.complete_stream([_route("a:free"), _route("b:free")],
                                                 BODY, 0.0))
    assert text == "bien"
    assert "kilo/a:free" in caplog.text
    assert "discarding" in caplog.text


async def test_it_warns_when_the_retention_buffer_overflows(caplog):
    # More than PENDING_CAP contentless chunks: they are released and the attempt
    # can no longer fail over cleanly. That has to be said out loud.
    from llm_libre.proxy import PENDING_CAP
    noise = b"".join(
        b'data: {"choices":[{"index":0,"delta":{"reasoning":"x"}}]}\n\n'
        for _ in range(PENDING_CAP + 5))
    p = _proxy(lambda req: httpx.Response(200, content=noise + b"data: [DONE]\n\n"))
    with caplog.at_level(logging.INFO, logger="llm_libre.proxy"):
        [l async for l in p.complete_stream([_route("a:free")], BODY, 0.0)]
    assert "kilo/a:free" in caplog.text


# --- Finding 2 of the review (streaming), and its final redesign in round 8 --
#     see the header comment of SUSPICION_THRESHOLD in proxy.py for the complete
#     rule. Exercised over ALL THREE failure paths of complete_stream (non-200
#     status, a stream with no useful content, a network error): no real-traffic
#     failure punishes directly, it only accumulates suspicion; crossing the
#     threshold fires our own probe in the background, and that probe decides. ---

def _ok():
    return {"choices": [{"message": {"role": "assistant", "content": "hola"}}]}


def _ping(body: bytes) -> bool:
    """True if the request that reached the mock is the probe -- the same fixed
    `PING` payload (proxy.py). The on-demand probe uses complete() (NOT
    streaming), so the response it deserves is ordinary JSON, not an SSE body."""
    messages = json.loads(body).get("messages") or []
    return bool(messages) and messages[0].get("content") == "ping"


async def test_n_consecutive_non_200_statuses_when_streaming_punish_via_a_probe():
    from llm_libre.proxy import SUSPICION_THRESHOLD
    p = _proxy(lambda req: httpx.Response(500))   # broken for ANY payload
    for i in range(SUSPICION_THRESHOLD):
        [l async for l in p.complete_stream([_route("a:free")], BODY, float(i))]
    await p.wait_for_pending_probes()
    assert p.cooldowns["kilo/a:free"] > 0.0


async def test_n_consecutive_streams_without_useful_content_punish_via_a_probe():
    from llm_libre.proxy import SUSPICION_THRESHOLD
    empty_stream = b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\ndata: [DONE]\n\n'
    p = _proxy(lambda req: httpx.Response(200, content=empty_stream))
    for i in range(SUSPICION_THRESHOLD):
        [l async for l in p.complete_stream([_route("a:free")], BODY, float(i))]
    await p.wait_for_pending_probes()
    assert p.cooldowns["kilo/a:free"] > 0.0


async def test_a_streaming_success_clears_the_accumulated_suspicion():
    from llm_libre.proxy import SUSPICION_THRESHOLD
    estado = {"n": 0}

    def handler(req):
        estado["n"] += 1
        if estado["n"] <= SUSPICION_THRESHOLD - 1:
            return httpx.Response(500)
        return httpx.Response(200, content=_sse("bien"))

    p = _proxy(handler)
    for i in range(SUSPICION_THRESHOLD):
        [l async for l in p.complete_stream([_route("a:free")], BODY, float(i))]
    await p.wait_for_pending_probes()
    assert "kilo/a:free" not in p.cooldowns


async def test_three_consecutive_400s_when_streaming_do_not_trigger_suspicion():
    # The same HIGH correction as in complete(): a 4xx (not 429) is a
    # DETERMINISTIC client error, not a signal that the route is broken.
    from llm_libre.proxy import SUSPICION_THRESHOLD
    p = _proxy(lambda req: httpx.Response(400, json={"error": "bad request"}))
    for i in range(SUSPICION_THRESHOLD):
        [l async for l in p.complete_stream([_route("a:free")], BODY, float(i))]
    await p.wait_for_pending_probes()
    assert "kilo/a:free" not in p.cooldowns


async def test_a_400_when_streaming_is_recorded_flagged_as_a_client_error():
    # Round 4: the same reliability/health hole on the streaming side.
    p = _proxy(lambda req: httpx.Response(400, json={"error": "bad request"}))
    [l async for l in p.complete_stream([_route("a:free")], BODY, 0.0)]
    row = p.store._con.execute(
        "SELECT ok, is_client_error FROM events WHERE key = 'kilo/a:free'").fetchone()
    assert row == (0, 1)


# --- Re-review: a MEDIUM finding. complete_stream() was still using the fixed
#     global TIMEOUT_S, ignoring Provider.timeout_s -- streaming is the default for
#     chat clients, and it is precisely the path of the "hung proxy" scenario that
#     motivated the per-provider timeout. A knob that does nothing is worse than no
#     knob at all. ---

async def test_it_uses_the_providers_own_timeout_when_streaming():
    seen = []

    def handler(req):
        seen.append(req.extensions.get("timeout"))
        return httpx.Response(200, content=_sse("bien"))

    store = Storage(":memory:")
    store.create_schema()
    lento = Provider("lento", "free", "openai", "https://lento.test", "", "/models",
                      {}, [], timeout_s=20.0)
    p = Proxy({"lento": lento}, store, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    [l async for l in p.complete_stream([_route(provider="lento")], BODY, 0.0)]
    assert seen[0]["read"] == 20.0


async def test_it_uses_the_global_timeout_when_streaming_if_the_provider_declares_none():
    seen = []

    def handler(req):
        seen.append(req.extensions.get("timeout"))
        return httpx.Response(200, content=_sse("bien"))

    p = _proxy(handler)   # kilo, sin timeout_s declarado
    [l async for l in p.complete_stream([_route(provider="kilo")], BODY, 0.0)]
    assert seen[0]["read"] == 90.0   # TIMEOUT_S


# --- Task 14: the same test as the one in test_proxy.py but on the streaming
#     path, with chatgpt's REAL config (providers.yaml, loaded with
#     providers.load -- the same path as production), not a synthetic provider. It
#     goes red if the YAML loses timeout_s or if complete_stream stops reading
#     Provider.timeout_s for that route. ---

async def test_chatgpt_uses_its_own_timeout_from_the_real_yaml_when_streaming():
    seen = []

    def handler(req):
        seen.append(req.extensions.get("timeout"))
        return httpx.Response(200, content=_sse("bien"))

    chatgpt = next(p for p in load(YAML_REAL, {}) if p.id == "chatgpt")
    assert chatgpt.timeout_s is not None   # si esto falla, el YAML perdio timeout_s
    store = Storage(":memory:")
    store.create_schema()
    p = Proxy({"chatgpt": chatgpt}, store,
             httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    [l async for l in p.complete_stream(
        [_route("gpt-5-3-mini", provider="chatgpt")], BODY, 0.0)]
    assert seen[0]["read"] == chatgpt.timeout_s


def _multi(*modelos):
    return [_route(m) for m in modelos]


# --- Round 8. The gate found TWO vectors round 7 did not close, both escape
#     hatches of that round's own design:
#
#     1. A chain of a SINGLE route -- the client forces it with an explicit
#        `model` or with `x_min_context` (published per route in /v1/ranking),
#        with no internal knowledge required.
#     2. The `if emitido:` branch below (PENDING_CAP's force-flush): it committed
#        immediately WITHOUT checking whether a sibling had succeeded and WITHOUT
#        looking at the chain's length -- `{"model":"auto","stream":true}` (no
#        extensions, no id) plus a prompt that burns the reasoning budget without
#        ever emitting text was enough: 15 requests, total blackout, across a
#        chain of 5 HEALTHY routes.
#
#     Round 8 removes both "how many routes are there" and "which branch did it
#     leave by" from the axis: ANY real-traffic failure, in ANY branch (including
#     `if emitido:`), only accumulates suspicion -- it never punishes directly.
#     The tests below reproduce both vectors with the mock returning the PROBE
#     (fixed payload, not streaming: the on-demand probe uses complete()) a
#     different response from the one it gives the client. ---

async def test_an_identical_non_200_in_a_single_route_chain_does_not_punish_a_healthy_route():
    def handler(req):
        if _ping(req.content):
            return httpx.Response(200, json=_ok())
        return httpx.Response(403, json={"error": "contenido flageado"})

    p = _proxy(handler)
    for i in range(15):
        [l async for l in p.complete_stream(_multi("a:free"), BODY, float(i))]
    await p.wait_for_pending_probes()
    assert p.cooldowns == {}


async def test_an_identical_useless_stream_in_a_single_route_chain_does_not_punish():
    empty_stream = b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\ndata: [DONE]\n\n'

    def handler(req):
        if _ping(req.content):
            return httpx.Response(200, json=_ok())
        return httpx.Response(200, content=empty_stream)

    p = _proxy(handler)
    for i in range(15):
        [l async for l in p.complete_stream(_multi("a:free"), BODY, float(i))]
    await p.wait_for_pending_probes()
    assert p.cooldowns == {}


async def test_an_identical_network_error_in_a_single_route_chain_does_not_punish_when_streaming():
    def handler(req):
        if _ping(req.content):
            return httpx.Response(200, json=_ok())
        raise httpx.ReadTimeout("prompt gigante", request=req)

    p = _proxy(handler)
    for i in range(15):
        [l async for l in p.complete_stream(_multi("a:free"), BODY, float(i))]
    await p.wait_for_pending_probes()
    assert p.cooldowns == {}


def _sse_structure_only(n: int) -> bytes:
    """`n` real streaming chunks carrying nothing but structural signal
    (`finish_reason` present, though null -- exactly as sent by a real provider in
    every intermediate chunk) and NEVER visible content: the
    pattern of a reasoning model that burns its budget thinking without ever
    getting to emit text. With `n > PENDING_CAP` it triggers the force-flush
    (`emitido=True`) without ever passing through `util=True` -- exactly the
    `if emitido:` branch this round stops treating as an exception."""
    lines = ['data: {"choices":[{"index":0,"delta":{"content":""},'
             '"finish_reason":null}]}\n\n' for _ in range(n)]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


async def test_a_flood_of_useless_chunks_does_not_punish_a_healthy_route():
    # The gate's exact vector 2: the chain does not even need narrowing --
    # "auto", no extensions, against 5 healthy routes.
    from llm_libre.proxy import PENDING_CAP
    contentless_payload = _sse_structure_only(PENDING_CAP + 6)

    def handler(req):
        if _ping(req.content):
            return httpx.Response(200, json=_ok())
        return httpx.Response(200, content=contentless_payload)

    routes = _multi("m0:free", "m1:free", "m2:free", "m3:free", "m4:free")
    p = _proxy(handler)
    cuerpo_auto = {"model": "auto", "stream": True,
                  "messages": [{"role": "user", "content": "piensa mucho antes de responder"}]}
    for i in range(15):
        [l async for l in p.complete_stream(routes, cuerpo_auto, float(i))]
    await p.wait_for_pending_probes()
    assert p.cooldowns == {}


async def test_a_genuinely_broken_route_with_a_healthy_sibling_is_still_punished_quickly():
    from llm_libre.proxy import SUSPICION_THRESHOLD

    def handler(req):
        if _ping(req.content):
            return httpx.Response(500)   # the probe sees it broken too
        body = json.loads(req.content)
        if body["model"] == "a:free":
            return httpx.Response(500)
        return httpx.Response(200, content=_sse("bien"))

    p = _proxy(handler)
    for i in range(SUSPICION_THRESHOLD):
        text = await _collect(p.complete_stream(_multi("a:free", "b:free"), BODY, float(i)))
        assert text == "bien"
    await p.wait_for_pending_probes()
    assert "kilo/a:free" in p.cooldowns
    assert "kilo/b:free" not in p.cooldowns


async def test_a_genuinely_broken_route_in_a_single_route_chain_cools_down_quickly():
    from llm_libre.proxy import SUSPICION_THRESHOLD
    p = _proxy(lambda req: httpx.Response(500))   # broken for ANY payload
    for i in range(SUSPICION_THRESHOLD):
        [l async for l in p.complete_stream(_multi("a:free"), BODY, float(i))]
    await p.wait_for_pending_probes()
    assert "kilo/a:free" in p.cooldowns
