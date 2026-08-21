"""The model-less capability endpoints: /v1/audio/speech, /v1/audio/transcriptions,
/v1/translate.

What these tests are really about is the FILTER. Every one of these endpoints
exists to route by a capability the provider published on its contract, so the
assertions that matter are "an incapable provider was never attempted" and "the
capability came from the URL, not from the body".
"""
import httpx
import pytest
from fastapi.testclient import TestClient

from llm_libre.api import State, create_app
from llm_libre.models import Capabilities, Route
from llm_libre.providers import Provider
from llm_libre.proxy import Proxy
from llm_libre.storage import Storage

AUTH = {"Authorization": "Bearer k"}
MP3 = b"ID3\x03\x00\x00\x00fake-mp3-bytes"


def _caps(**kw) -> Capabilities:
    base = dict(tools=False, vision=False, context=8000, max_output=1024)
    base.update(kw)
    return Capabilities(**base)


def _build(routes, handler):
    """An app whose upstream is a mock transport that records what it received."""
    store = Storage(":memory:")
    store.create_schema()
    store.upsert_routes(routes, 1.0)
    providers = {
        r.provider: Provider(r.provider, "free", "openai",
                             f"https://{r.provider}.test/v1", "", "/models", {}, [])
        for r in routes
    }
    seen = []

    def transport(request):
        seen.append(request)
        return handler(request)

    http = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    state = State(store=store, proxy=Proxy(providers, store, http),
                  api_keys={"k"}, daily_paid_cap=200)
    return TestClient(create_app(state)), seen


# ── text to speech ────────────────────────────────────────────────────────────

@pytest.fixture
def speech_app():
    routes = [
        Route("mute", "mute:free", "free", _caps()),
        Route("voice", "voice:free", "free", _caps(audio_speech=True)),
    ]
    return _build(routes, lambda req: httpx.Response(
        200, content=MP3, headers={"content-type": "audio/mpeg"}))


def test_speech_returns_audio_bytes_not_json(speech_app):
    client, _ = speech_app
    r = client.post("/v1/audio/speech", headers=AUTH,
                    json={"input": "hola", "voice": "alloy"})
    assert r.status_code == 200
    assert r.content == MP3
    assert r.headers["content-type"].startswith("audio/mpeg")


def test_speech_never_touches_a_provider_that_cannot_speak(speech_app):
    """The filter is the point: no request is spent discovering the obvious."""
    client, seen = speech_app
    client.post("/v1/audio/speech", headers=AUTH, json={"input": "hola"})
    assert [str(r.url) for r in seen] == ["https://voice.test/v1/audio/speech"]


def test_speech_reports_which_route_served_it(speech_app):
    client, _ = speech_app
    r = client.post("/v1/audio/speech", headers=AUTH, json={"input": "hola"})
    assert r.headers["X-Route-Used"] == "voice/voice:free"


def test_no_speaking_provider_is_a_400_and_costs_no_upstream_call():
    """400, not 503, and that is `_no_routes`'s documented split: nothing in the
    pool CAN speak, which is the same class of error as asking for vision no
    one has. 503 is reserved for routes that could serve but are down."""
    client, seen = _build([Route("mute", "mute:free", "free", _caps())],
                          lambda req: httpx.Response(200, json={}))
    r = client.post("/v1/audio/speech", headers=AUTH, json={"input": "hola"})
    assert r.status_code == 400
    assert seen == []


def test_an_empty_200_is_a_failure_not_a_silent_success():
    """Both proxies can answer 200 with nothing inside. Counting that as success
    is how a broken route stays at the head of the ranking."""
    client, _ = _build(
        [Route("voice", "voice:free", "free", _caps(audio_speech=True))],
        lambda req: httpx.Response(200, content=b"", headers={"content-type": "audio/mpeg"}))
    assert client.post("/v1/audio/speech", headers=AUTH,
                       json={"input": "hola"}).status_code == 503


def test_speech_fails_over_to_the_next_capable_provider():
    calls = []

    def handler(req):
        calls.append(str(req.url))
        if "first" in str(req.url):
            return httpx.Response(500, text="boom")
        return httpx.Response(200, content=MP3, headers={"content-type": "audio/mpeg"})

    client, _ = _build(
        [Route("first", "first:free", "free", _caps(audio_speech=True)),
         Route("second", "second:free", "free", _caps(audio_speech=True))],
        handler)
    r = client.post("/v1/audio/speech", headers=AUTH, json={"input": "hola"})
    assert r.status_code == 200 and r.content == MP3
    assert len(calls) == 2


# ── speech to text ────────────────────────────────────────────────────────────

def test_transcriptions_forwards_the_multipart_body_byte_for_byte():
    """Re-serialising would re-encode the audio on every failover attempt and
    drop any field this gateway has not heard of."""
    bodies = []

    def handler(req):
        bodies.append((req.content, req.headers.get("content-type")))
        return httpx.Response(200, json={"text": "hola"})

    client, _ = _build(
        [Route("ears", "ears:free", "free", _caps(audio_transcription=True))],
        handler)
    raw = b"--x\r\nContent-Disposition: form-data; name=\"file\"\r\n\r\nAUDIO\r\n--x--\r\n"
    r = client.post("/v1/audio/transcriptions", headers={
        **AUTH, "Content-Type": "multipart/form-data; boundary=x"}, content=raw)
    assert r.status_code == 200 and r.json() == {"text": "hola"}
    assert bodies[0][0] == raw
    assert bodies[0][1] == "multipart/form-data; boundary=x"


def test_transcriptions_skips_a_provider_that_cannot_listen():
    client, seen = _build(
        [Route("deaf", "deaf:free", "free", _caps()),
         Route("ears", "ears:free", "free", _caps(audio_transcription=True))],
        lambda req: httpx.Response(200, json={"text": "hola"}))
    client.post("/v1/audio/transcriptions", headers={
        **AUTH, "Content-Type": "multipart/form-data; boundary=x"}, content=b"--x--")
    assert [str(r.url) for r in seen] == ["https://ears.test/v1/audio/transcriptions"]


# ── translate ─────────────────────────────────────────────────────────────────

def test_translate_routes_by_its_own_capability():
    client, seen = _build(
        [Route("plain", "plain:free", "free", _caps(audio_speech=True)),
         Route("poly", "poly:free", "free", _caps(translate=True))],
        lambda req: httpx.Response(200, json={"text": "hello"}))
    r = client.post("/v1/translate", headers=AUTH, json={"text": "hola", "to": "en"})
    assert r.status_code == 200 and r.json() == {"text": "hello"}
    assert [str(u.url) for u in seen] == ["https://poly.test/v1/translate"]


def test_a_speaking_provider_does_not_satisfy_translate():
    """Each axis is independent: one capability never stands in for another."""
    client, seen = _build(
        [Route("plain", "plain:free", "free", _caps(audio_speech=True))],
        lambda req: httpx.Response(200, json={}))
    assert client.post("/v1/translate", headers=AUTH,
                       json={"text": "hola"}).status_code == 400
    assert seen == []


# ── the rule that keeps all of them honest ────────────────────────────────────

def test_the_capability_comes_from_the_url_not_from_the_body():
    """A chat body cannot talk its way into the speech endpoint's routing, and
    a speech body cannot reach a chat-only route."""
    client, seen = _build(
        [Route("chatty", "chatty:free", "free", _caps()),
         Route("voice", "voice:free", "free", _caps(audio_speech=True))],
        lambda req: httpx.Response(200, content=MP3,
                                   headers={"content-type": "audio/mpeg"}))
    client.post("/v1/audio/speech", headers=AUTH,
                json={"input": "hola", "needs_audio_speech": False,
                      "x_model": "chatty:free"})
    assert [str(r.url) for r in seen] == ["https://voice.test/v1/audio/speech"]
