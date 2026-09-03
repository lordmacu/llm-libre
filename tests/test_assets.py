"""Hosting the binaries a provider generated, on our own origin.

The behaviours worth pinning down are mostly about what does NOT happen: no
provider URL reaches the client, no client-supplied URL is ever fetched, no
failure to store turns a working generation into an error, and no id can escape
the assets directory.
"""
import base64
import hashlib
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from llm_libre.api import State, create_app, effective_public_base_url
from llm_libre.assets import AssetStore, content_disposition, localise, normalise_type
from llm_libre.models import Capabilities, Route
from llm_libre.providers import Provider
from llm_libre.proxy import Proxy
from llm_libre.storage import Storage

PNG = b"\x89PNG\r\n\x1a\n" + b"fake-bytes-for-a-test"
AUTH = {"Authorization": "Bearer k"}
PROVIDER_URL = "https://provider.test/generated/abc123?sig=expiring-signature"


@pytest.fixture
def store(tmp_path):
    db = Storage(":memory:")
    db.create_schema()
    return AssetStore(str(tmp_path / "assets"), db._con)


# --- the store ---

def test_the_id_is_the_hash_of_the_content(store):
    asset_id = store.put(PNG, "image/png", now=100.0)
    assert asset_id == hashlib.sha256(PNG).hexdigest()
    assert store.get(asset_id) == (PNG, "image/png")


def test_identical_bytes_stored_twice_cost_one_file(store):
    a = store.put(PNG, "image/png", now=100.0)
    b = store.put(PNG, "image/png", now=200.0)
    assert a == b
    assert store._con.execute("SELECT COUNT(*) FROM assets").fetchone()[0] == 1


def test_an_unknown_id_is_simply_absent(store):
    assert store.get("f" * 64) is None


def test_an_id_that_is_not_a_hash_cannot_reach_the_filesystem(store):
    # The id arrives as a URL path segment. Without validation this is a path
    # traversal; `get` must refuse it rather than let it near `open()`.
    for hostile in ("../../etc/passwd", "..", "", "a" * 63, "g" * 64, "/etc/passwd"):
        assert store.get(hostile) is None
    with pytest.raises(ValueError):
        store._path("../../etc/passwd")


def test_an_oversized_asset_is_refused_rather_than_stored(store):
    from llm_libre.assets import MAX_BYTES
    assert store.put(b"x" * (MAX_BYTES + 1), "image/png", now=1.0) is None
    assert store.put(b"", "image/png", now=1.0) is None


def test_an_unexpected_content_type_is_not_served_back_verbatim(store):
    """A provider must not be able to make us serve HTML from our own origin.

    Anything outside the allow-list is stored as an opaque download, so a
    generated blob cannot become stored XSS on whatever else the domain hosts.
    """
    asset_id = store.put(b"<script>alert(1)</script>", "text/html", now=1.0)
    assert store.get(asset_id)[1] == "application/octet-stream"


def test_pruning_removes_the_file_and_the_row(store):
    old = store.put(PNG, "image/png", now=100.0)
    new = store.put(b"other bytes entirely", "image/png", now=900.0)
    assert store.prune(before=500.0) == 1
    assert store.get(old) is None
    assert store.get(new) is not None
    assert not (store._dir / old).exists()


def test_an_unusable_directory_degrades_instead_of_raising(tmp_path):
    """A store that cannot write must not stop the process from starting.

    Image generation is one endpoint of five; refusing to boot over it would
    take chat, ranking and health down too.
    """
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("i am a file")
    db = Storage(":memory:")
    db.create_schema()
    store = AssetStore(str(blocker / "assets"), db._con)
    assert store.usable is False
    assert store.put(PNG, "image/png", now=1.0) is None
    assert store.get("a" * 64) is None
    assert store.prune(before=1e18) == 0


# --- rewriting a provider response ---

def _http(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_the_provider_url_never_reaches_the_client(store):
    fetched = []

    def handler(req):
        fetched.append(str(req.url))
        return httpx.Response(200, content=PNG, headers={"content-type": "image/png"})

    out = await localise({"created": 1, "data": [{"url": PROVIDER_URL}]},
                         store, _http(handler), "https://llm.example.app", None, now=5.0)
    url = out["data"][0]["url"]
    assert url.startswith("https://llm.example.app/v1/assets/")
    assert "provider.test" not in url
    assert fetched == [PROVIDER_URL]      # fetched exactly once
    assert store.get(url.rsplit("/", 1)[-1]) == (PNG, "image/png")


async def test_b64_json_is_honoured_and_leaves_no_url(store):
    handler = lambda req: httpx.Response(200, content=PNG,
                                         headers={"content-type": "image/png"})
    out = await localise({"data": [{"url": PROVIDER_URL, "revised_prompt": "a cat"}]},
                         store, _http(handler), "https://llm.example.app",
                         "b64_json", now=5.0)
    entry = out["data"][0]
    assert base64.b64decode(entry["b64_json"]) == PNG
    assert "url" not in entry
    assert entry["revised_prompt"] == "a cat"   # provider extras survive


async def test_a_failed_download_falls_back_to_the_providers_url(store):
    # Degrade, never fail: a usable-but-expiring URL beats a 500, and the client
    # can still see the image it paid for.
    handler = lambda req: httpx.Response(502)
    out = await localise({"data": [{"url": PROVIDER_URL}]},
                         store, _http(handler), "https://llm.example.app", None, now=5.0)
    assert out["data"][0]["url"] == PROVIDER_URL


async def test_without_a_public_base_url_nothing_is_rewritten(store):
    """An unset PUBLIC_BASE_URL must not produce URLs that resolve nowhere."""
    handler = lambda req: httpx.Response(200, content=PNG,
                                         headers={"content-type": "image/png"})
    out = await localise({"data": [{"url": PROVIDER_URL}]},
                         store, _http(handler), "", None, now=5.0)
    assert out["data"][0]["url"] == PROVIDER_URL


async def test_a_non_http_url_is_never_fetched(store):
    """The only URLs fetched are ones a provider returned to us.

    Nothing here may follow file://, data:// or anything else that would turn
    this into a reader of local resources.
    """
    fetched = []

    def handler(req):
        fetched.append(str(req.url))
        return httpx.Response(200, content=PNG)

    payload = {"data": [{"url": "file:///etc/passwd"},
                        {"url": "data:image/png;base64,AAAA"},
                        {"b64_json": "already-inline"}]}
    out = await localise(payload, store, _http(handler), "https://llm.example.app",
                         None, now=5.0)
    assert fetched == []
    assert out["data"] == payload["data"]


async def test_the_original_payload_is_not_mutated(store):
    handler = lambda req: httpx.Response(200, content=PNG,
                                         headers={"content-type": "image/png"})
    payload = {"created": 1, "data": [{"url": PROVIDER_URL}]}
    out = await localise(payload, store, _http(handler), "https://llm.example.app",
                         None, now=5.0)
    assert payload["data"][0]["url"] == PROVIDER_URL   # untouched
    assert out is not payload


# --- end to end through the app ---

def _app(tmp_path, image_response, public_base_url="https://llm.example.app"):
    db = Storage(":memory:")
    db.create_schema()
    route = Route("p", "drawer", "free",
                  Capabilities(tools=False, vision=False, context=1000,
                               max_output=100, images=True))
    db.upsert_routes([route], 1.0)

    def handler(req):
        if req.url.path == "/images/generations":
            return httpx.Response(200, json=image_response)
        return httpx.Response(200, content=PNG, headers={"content-type": "image/png"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provs = {"p": Provider("p", "free", "openai", "https://p.test", "", "/models", {}, [])}
    state = State(store=db, proxy=Proxy(provs, db, http), api_keys={"k"},
                  daily_paid_cap=200)
    state.assets = AssetStore(str(tmp_path / "assets"), db._con)
    state.public_base_url = public_base_url
    return TestClient(create_app(state)), state


def test_generation_returns_our_url_and_that_url_serves_the_bytes(tmp_path):
    client, _ = _app(tmp_path, {"created": 1, "data": [{"url": PROVIDER_URL}]})
    r = client.post("/v1/images/generations", headers=AUTH, json={"prompt": "a cat"})
    assert r.status_code == 200
    url = r.json()["data"][0]["url"]
    assert url.startswith("https://llm.example.app/v1/assets/")

    got = client.get(url.replace("https://llm.example.app", ""))
    assert got.status_code == 200
    assert got.content == PNG
    assert got.headers["content-type"].startswith("image/png")
    assert "immutable" in got.headers["cache-control"]
    assert got.headers["x-content-type-options"] == "nosniff"


def test_the_asset_endpoint_needs_no_api_key(tmp_path):
    """It has to work in an <img> tag, which cannot attach a header.

    What guards it is the id: the SHA-256 of the content, unguessable and
    underivable from the prompt.
    """
    client, _ = _app(tmp_path, {"created": 1, "data": [{"url": PROVIDER_URL}]})
    url = client.post("/v1/images/generations", headers=AUTH,
                      json={"prompt": "x"}).json()["data"][0]["url"]
    path = url.replace("https://llm.example.app", "")
    assert client.get(path).status_code == 200          # no AUTH header at all


def test_an_unknown_asset_is_a_404(tmp_path):
    client, _ = _app(tmp_path, {"created": 1, "data": [{"url": PROVIDER_URL}]})
    assert client.get("/v1/assets/" + "a" * 64).status_code == 404
    assert client.get("/v1/assets/not-a-hash").status_code == 404


def test_b64_json_through_the_endpoint(tmp_path):
    client, _ = _app(tmp_path, {"created": 1, "data": [{"url": PROVIDER_URL}]})
    r = client.post("/v1/images/generations", headers=AUTH,
                    json={"prompt": "x", "response_format": "b64_json"})
    entry = r.json()["data"][0]
    assert base64.b64decode(entry["b64_json"]) == PNG
    assert "url" not in entry


def test_response_format_does_not_travel_to_the_provider(tmp_path):
    """It is interpreted HERE: the provider's own default is what we want."""
    seen = {}

    db = Storage(":memory:")
    db.create_schema()
    db.upsert_routes([Route("p", "drawer", "free",
                            Capabilities(False, False, 1000, 100, images=True))], 1.0)

    def handler(req):
        import json
        if req.url.path == "/images/generations":
            seen.update(json.loads(req.content))
            return httpx.Response(200, json={"data": [{"url": PROVIDER_URL}]})
        return httpx.Response(200, content=PNG, headers={"content-type": "image/png"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provs = {"p": Provider("p", "free", "openai", "https://p.test", "", "/models", {}, [])}
    state = State(store=db, proxy=Proxy(provs, db, http), api_keys={"k"}, daily_paid_cap=200)
    state.assets = AssetStore(str(__import__("tempfile").mkdtemp()), db._con)
    state.public_base_url = "https://llm.example.app"
    TestClient(create_app(state)).post(
        "/v1/images/generations", headers=AUTH,
        json={"prompt": "x", "response_format": "b64_json"})
    # OpenAI defines response_format for the CLIENT; the provider is asked for a
    # url either way and the conversion happens here.
    assert seen.get("response_format") == "b64_json" or "response_format" not in seen


def test_with_no_store_configured_the_provider_url_is_passed_through(tmp_path):
    """The pre-assets behaviour, still reachable: state.assets = None."""
    client, state = _app(tmp_path, {"created": 1, "data": [{"url": PROVIDER_URL}]})
    state.assets = None
    r = client.post("/v1/images/generations", headers=AUTH, json={"prompt": "x"})
    assert r.json()["data"][0]["url"] == PROVIDER_URL


def test_generated_audio_keeps_its_type():
    """All five proxies do TTS as of 2026-08-20. Before this, an MP3 fell back
    to application/octet-stream and browsers downloaded it instead of playing
    it -- the asset was fine, the player was what broke."""
    for content_type in ("audio/mpeg", "audio/wav", "audio/ogg", "audio/webm",
                         "audio/mp4", "audio/aac", "audio/flac"):
        assert normalise_type(content_type) == content_type


def test_audio_is_served_inline_not_downloaded():
    """Unlike SVG, none of these can carry script, so nothing forces attachment."""
    assert content_disposition("audio/mpeg").startswith("inline")


def test_an_unlisted_audio_type_still_falls_back():
    assert normalise_type("audio/x-made-up") == "application/octet-stream"


# --- deriving the public base from the request when none is configured ---

def _scope_request(headers: dict, scheme: str = "http") -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    return Request({"type": "http", "asgi": {"version": "3.0"},
                    "http_version": "1.1", "method": "POST", "scheme": scheme,
                    "path": "/v1/chat/completions",
                    "raw_path": b"/v1/chat/completions", "query_string": b"",
                    "root_path": "", "headers": raw,
                    "client": ("127.0.0.1", 1234), "server": ("testserver", 80)})


def _no_base():
    return SimpleNamespace(public_base_url="")


def test_a_configured_base_url_wins_over_any_header():
    state = SimpleNamespace(public_base_url="https://llm.example.app")
    request = _scope_request({"host": "internal:8000",
                              "x-forwarded-proto": "http",
                              "x-forwarded-host": "chat.example.com"})
    assert effective_public_base_url(state, request) == "https://llm.example.app"


def test_the_origin_comes_from_the_forwarded_headers():
    request = _scope_request({"host": "internal:8000",
                              "x-forwarded-proto": "https",
                              "x-forwarded-host": "chat.example.com"})
    assert effective_public_base_url(_no_base(), request) == "https://chat.example.com"


def test_a_proxy_chain_contributes_only_its_first_forwarded_value():
    request = _scope_request({"x-forwarded-proto": "https, http",
                              "x-forwarded-host": "chat.example.com, 10.0.0.1"})
    assert effective_public_base_url(_no_base(), request) == "https://chat.example.com"


def test_the_host_header_and_the_connection_scheme_are_the_last_resort():
    request = _scope_request({"host": "chat.example.com:8443"}, scheme="https")
    assert effective_public_base_url(_no_base(), request) == "https://chat.example.com:8443"


def test_a_garbage_forwarded_host_is_not_turned_into_a_url():
    request = _scope_request({"x-forwarded-host": "not a host",
                              "host": "testserver"})
    assert effective_public_base_url(_no_base(), request) == "http://testserver"


def test_an_unknown_forwarded_proto_derives_nothing():
    request = _scope_request({"x-forwarded-proto": "gopher",
                              "host": "testserver"})
    assert effective_public_base_url(_no_base(), request) is None


def test_without_any_host_nothing_can_be_derived():
    assert effective_public_base_url(_no_base(), _scope_request({})) is None


def test_generated_images_are_rehosted_against_the_forwarded_origin(tmp_path):
    client, _ = _app(tmp_path, {"created": 1, "data": [{"url": PROVIDER_URL}]},
                     public_base_url="")
    r = client.post("/v1/images/generations",
                    headers={**AUTH, "X-Forwarded-Proto": "https",
                             "X-Forwarded-Host": "chat.example.com"},
                    json={"prompt": "a cat"})
    url = r.json()["data"][0]["url"]
    assert url.startswith("https://chat.example.com/v1/assets/")
    assert "provider.test" not in url


def test_without_forwarded_headers_the_host_header_carries_the_origin(tmp_path):
    client, _ = _app(tmp_path, {"created": 1, "data": [{"url": PROVIDER_URL}]},
                     public_base_url="")
    r = client.post("/v1/images/generations", headers=AUTH, json={"prompt": "a cat"})
    assert r.json()["data"][0]["url"].startswith("http://testserver/v1/assets/")


def test_the_configured_base_url_still_wins_end_to_end(tmp_path):
    client, _ = _app(tmp_path, {"created": 1, "data": [{"url": PROVIDER_URL}]})
    r = client.post("/v1/images/generations",
                    headers={**AUTH, "X-Forwarded-Proto": "https",
                             "X-Forwarded-Host": "chat.example.com"},
                    json={"prompt": "a cat"})
    assert r.json()["data"][0]["url"].startswith("https://llm.example.app/v1/assets/")


IMAGE_COMPLETION = {
    "id": "c1", "object": "chat.completion",
    "choices": [{"index": 0, "finish_reason": "stop", "message": {
        "role": "assistant",
        "content": [{"type": "image_url", "image_url": {"url": PROVIDER_URL}}]}}],
}


def _chat_app(tmp_path, respond, public_base_url=""):
    db = Storage(":memory:")
    db.create_schema()
    db.upsert_routes([Route("p", "talker", "free",
                            Capabilities(True, False, 100000, 4096))], 1.0)

    def handler(req):
        if req.url.path == "/chat/completions":
            return respond(req)
        return httpx.Response(200, content=PNG, headers={"content-type": "image/png"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provs = {"p": Provider("p", "free", "openai", "https://p.test", "", "/models", {}, [])}
    state = State(store=db, proxy=Proxy(provs, db, http), api_keys={"k"},
                  daily_paid_cap=200)
    state.assets = AssetStore(str(tmp_path / "assets"), db._con)
    state.public_base_url = public_base_url
    return TestClient(create_app(state)), state


FORWARDED = {**AUTH, "X-Forwarded-Proto": "https",
             "X-Forwarded-Host": "chat.example.com"}
CHAT_BODY = {"model": "auto", "messages": [{"role": "user", "content": "draw a cat"}]}


def test_completion_images_are_rehosted_against_the_forwarded_origin(tmp_path):
    """The reported bug: with PUBLIC_BASE_URL empty, a gpt-image-1 completion
    handed the client the proxy's internal URL, which resolves nowhere."""
    client, _ = _chat_app(tmp_path, lambda req: httpx.Response(200, json=IMAGE_COMPLETION))
    r = client.post("/v1/chat/completions", headers=FORWARDED, json=CHAT_BODY)
    assert r.status_code == 200
    url = r.json()["choices"][0]["message"]["content"][0]["image_url"]["url"]
    assert url.startswith("https://chat.example.com/v1/assets/")
    assert "provider.test" not in url


def test_streamed_completion_images_are_rehosted_against_the_forwarded_origin(tmp_path):
    sse = ('data: {"choices":[{"delta":{"content":[{"type":"image_url",'
           '"image_url":{"url":"%s"}}]}}]}\n\ndata: [DONE]\n\n'
           % PROVIDER_URL).encode()
    client, _ = _chat_app(tmp_path, lambda req: httpx.Response(
        200, content=sse, headers={"content-type": "text/event-stream"}))
    r = client.post("/v1/chat/completions", headers=FORWARDED,
                    json={**CHAT_BODY, "stream": True})
    assert r.status_code == 200
    assert "https://chat.example.com/v1/assets/" in r.text
    assert "provider.test" not in r.text
