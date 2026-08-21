"""Image generation as a routing capability.

The whole point of the feature is the FILTER, so most of what is worth asserting
is about routes NOT attempted. `vision` (image input) and `images` (image
output) are separate axes and a route can have either, both or neither -- the
tests below pin that down, because collapsing the two would send a "draw me a
cat" prompt to a model that can only look at pictures.
"""
import httpx
import pytest
from fastapi.testclient import TestClient

from llm_libre.api import State, create_app
from llm_libre.models import Capabilities, Route, RouteRequest
from llm_libre.providers import Provider
from llm_libre.proxy import Proxy
from llm_libre.router import compatible_routes
from llm_libre.storage import Storage

IMAGE_OK = {"created": 1, "data": [{"url": "https://x.test/a.png"}]}


def _route(model="m:free", provider="p1", images=False, tools=True, vision=False):
    return Route(provider, model, "free",
                 Capabilities(tools=tools, vision=vision, context=100000,
                              max_output=4096, images=images))


def _state(handler, routes, providers=None):
    store = Storage(":memory:")
    store.create_schema()
    store.upsert_routes(routes, 1.0)
    provs = providers or {
        r.provider: Provider(r.provider, "free", "openai", f"https://{r.provider}.test",
                             "", "/models", {}, [])
        for r in routes}
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return State(store=store, proxy=Proxy(provs, store, http),
                 api_keys={"k"}, daily_paid_cap=200)


AUTH = {"Authorization": "Bearer k"}


# --- the filter, at the router level ---

def test_a_route_without_images_can_never_serve_an_image_request():
    routes = [_route("chat-only"), _route("drawer", images=True)]
    out = compatible_routes(routes, RouteRequest(needs_images=True))
    assert [r.model_id for r in out] == ["drawer"]


def test_images_and_vision_are_separate_axes():
    """A model that SEES is not a model that DRAWS.

    Collapsing them is the obvious shortcut and it is wrong in both directions:
    grok's imagine-agent-mode generates but is declared vision:false, and most
    vision routes cannot generate anything.
    """
    sees = _route("sees", vision=True, images=False)
    draws = _route("draws", vision=False, images=True)
    both = _route("both", vision=True, images=True)
    pool = [sees, draws, both]
    assert {r.model_id for r in compatible_routes(pool, RouteRequest(needs_images=True))} \
        == {"draws", "both"}
    assert {r.model_id for r in compatible_routes(pool, RouteRequest(needs_vision=True))} \
        == {"sees", "both"}


def test_a_chat_request_is_never_narrowed_to_image_routes():
    # The reverse direction: needs_images defaults to False, so an ordinary chat
    # request still sees the whole pool, image generators included (they stay in
    # the chat pool on purpose -- nothing is excluded by declaration).
    pool = [_route("chat"), _route("drawer", images=True)]
    assert len(compatible_routes(pool, RouteRequest())) == 2


# --- the filter, through the endpoint ---

def test_the_endpoint_never_touches_a_route_that_cannot_generate():
    """The measurable claim: not 'it eventually succeeds' but 'it never asks'."""
    asked = []

    def handler(req):
        asked.append(str(req.url))
        return httpx.Response(200, json=IMAGE_OK)

    routes = [_route("chat-only", provider="p1"),
              _route("drawer", provider="p2", images=True)]
    client = TestClient(create_app(_state(handler, routes)))
    r = client.post("/v1/images/generations", headers=AUTH, json={"prompt": "a cat"})
    assert r.status_code == 200
    assert r.json() == IMAGE_OK
    assert len(asked) == 1
    assert "p2.test" in asked[0], asked
    assert all("p1.test" not in u for u in asked)


def test_it_posts_to_the_images_path_not_to_chat_completions():
    seen = []

    def handler(req):
        seen.append(req.url.path)
        return httpx.Response(200, json=IMAGE_OK)

    client = TestClient(create_app(_state(handler, [_route(images=True)])))
    client.post("/v1/images/generations", headers=AUTH, json={"prompt": "x"})
    assert seen == ["/images/generations"]


def test_the_route_that_served_is_reported_in_the_headers():
    client = TestClient(create_app(
        _state(lambda req: httpx.Response(200, json=IMAGE_OK),
               [_route("drawer", provider="p2", images=True)])))
    r = client.post("/v1/images/generations", headers=AUTH, json={"prompt": "x"})
    assert r.headers["X-Route-Used"] == "p2/drawer"
    assert r.headers["X-Tier"] == "free"
    assert r.headers["X-Attempts"] == "1"


def test_with_no_image_capable_route_it_is_a_400_not_a_503():
    # Design section 9's distinction, applied to this endpoint: "nothing can ever
    # do this" is a client error, "they are all busy" is unavailability. A 503
    # here would tell every SDK to retry forever against a pool that will never
    # be able to answer.
    client = TestClient(create_app(
        _state(lambda req: httpx.Response(200, json=IMAGE_OK), [_route("chat-only")])))
    r = client.post("/v1/images/generations", headers=AUTH, json={"prompt": "x"})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["request"]["needs_images"] is True


def test_it_fails_over_to_the_next_generator():
    calls = []

    def handler(req):
        calls.append(req.url.host)
        if req.url.host == "p1.test":
            return httpx.Response(500)
        return httpx.Response(200, json=IMAGE_OK)

    routes = [_route("a", provider="p1", images=True),
              _route("b", provider="p2", images=True)]
    client = TestClient(create_app(_state(handler, routes)))
    r = client.post("/v1/images/generations", headers=AUTH, json={"prompt": "x"})
    assert r.status_code == 200
    assert r.headers["X-Attempts"] == "2"
    assert calls == ["p1.test", "p2.test"]


def test_a_200_with_an_empty_data_array_is_not_a_success():
    """Both real upstreams can answer 200 with nothing in `data`.

    Accepting it would hand the client `{"data": []}` AND record a success for a
    route that generated nothing -- which is how a broken route keeps its place
    at the head of the ranking.
    """
    calls = []

    def handler(req):
        calls.append(req.url.host)
        if req.url.host == "p1.test":
            return httpx.Response(200, json={"created": 1, "data": []})
        return httpx.Response(200, json=IMAGE_OK)

    routes = [_route("a", provider="p1", images=True),
              _route("b", provider="p2", images=True)]
    state = _state(handler, routes)
    client = TestClient(create_app(state))
    r = client.post("/v1/images/generations", headers=AUTH, json={"prompt": "x"})
    assert r.status_code == 200
    assert r.headers["X-Route-Used"] == "p2/b"
    rows = state.store._con.execute(
        "SELECT key, ok FROM events ORDER BY key").fetchall()
    assert rows == [("p1/a", 0), ("p2/b", 1)]


def test_every_generator_failing_is_a_503_with_the_reason():
    client = TestClient(create_app(
        _state(lambda req: httpx.Response(200, json={"data": []}),
               [_route("a", provider="p1", images=True)])))
    r = client.post("/v1/images/generations", headers=AUTH, json={"prompt": "x"})
    assert r.status_code == 503
    assert r.json()["error"]["detail"] == "200 with no generated image"


def test_it_requires_a_key():
    client = TestClient(create_app(
        _state(lambda req: httpx.Response(200, json=IMAGE_OK), [_route(images=True)])))
    assert client.post("/v1/images/generations", json={"prompt": "x"}).status_code == 401


def test_the_gateway_extensions_do_not_travel_to_the_provider():
    sent = {}

    def handler(req):
        import json
        sent.update(json.loads(req.content))
        return httpx.Response(200, json=IMAGE_OK)

    client = TestClient(create_app(
        _state(handler, [_route("drawer", images=True)])))
    client.post("/v1/images/generations", headers=AUTH,
                json={"prompt": "x", "x_allow_paid": False, "size": "1024x1024"})
    assert "x_allow_paid" not in sent
    assert sent["size"] == "1024x1024"     # a provider parameter still travels
    assert sent["model"] == "drawer"       # rewritten to the route's real id


def test_a_paid_generator_is_counted_as_billable():
    paid = Route("pp", "draw", "paid",
                 Capabilities(tools=False, vision=False, context=1000,
                              max_output=100, images=True))
    state = _state(lambda req: httpx.Response(200, json=IMAGE_OK), [paid])
    client = TestClient(create_app(state))
    client.post("/v1/images/generations", headers=AUTH, json={"prompt": "x"})
    day = client.get("/v1/usage", headers=AUTH).json()
    assert day["paid_today"] == 1


def test_no_tool_emulation_prompt_is_injected_into_an_image_request():
    """A provider that emulates tools must not have its image prompt rewritten.

    build_image_request is a separate function from build_request precisely so
    this cannot happen by forgetting a flag.
    """
    sent = {}

    def handler(req):
        import json
        sent.update(json.loads(req.content))
        return httpx.Response(200, json=IMAGE_OK)

    provider = Provider("emu", "free", "openai", "https://emu.test", "", "/models",
                        {}, [], emulates_tools=True)
    route = Route("emu", "draw", "free",
                  Capabilities(tools=True, vision=False, context=1000,
                               max_output=100, images=True))
    client = TestClient(create_app(_state(handler, [route], {"emu": provider})))
    client.post("/v1/images/generations", headers=AUTH, json={"prompt": "a cat"})
    assert set(sent) == {"prompt", "model"}
    assert sent["prompt"] == "a cat"


# --- what the registry declares ---

def test_the_registry_declares_exactly_the_measured_generators():
    from llm_libre.providers import fixed_routes, load
    provs = load("providers.yaml", {})
    grok = next(p for p in provs if p.id == "grok")
    # grok's default_capabilities must NOT claim images, or all discovered
    # routes would advertise a capability only the imagine-agent-mode family
    # has. Those three ids used to get `images: true` through a per-id
    # `exceptions` entry; that was retired 2026-08-20 -- grok's /v1/models now
    # publishes {tools, vision, images} per model directly, so nothing needs
    # pinning here anymore. See the retirement note on `exceptions` in
    # providers.yaml.
    assert grok.default_capabilities.images is False
    assert grok.exceptions == {}
    # mistral claims BOTH: one route that reads images and generates them. It
    # generates intermittently (the model decides whether to call the tool) and
    # on a small shared quota -- see the note in providers.yaml -- but a real
    # request does return a real image, which is what a declaration requires.
    mistral = next(p for p in provs if p.id == "mistral")
    caps = fixed_routes(mistral)[0].capabilities
    assert caps.images is True
    assert caps.vision is True
    # chatgpt: default_capabilities has images:false (all discovered chat models
    # cannot generate), but dall-e-3 is declared as a fixed_model with images:true.
    # The proxy exposes POST /v1/images/generations and uses its internal DALL-E 3
    # tool via a conversation with force_use_tools=True.
    chatgpt = next(p for p in provs if p.id == "chatgpt")
    assert not (chatgpt.default_capabilities and chatgpt.default_capabilities.images)
    dalle = next(r for r in fixed_routes(chatgpt) if r.model_id == "dall-e-3")
    assert dalle.capabilities.images is True
    assert dalle.capabilities.tools is False
    # Everyone else: no claim. A provider gains the capability by claiming it.
    for pid in ("perplexity", "deepseek", "kilo", "minimax"):
        p = next(x for x in provs if x.id == pid)
        assert not (p.default_capabilities and p.default_capabilities.images), pid
        assert all(not r.capabilities.images for r in fixed_routes(p)), pid


def test_a_route_persists_and_reloads_its_images_capability():
    store = Storage(":memory:")
    store.create_schema()
    store.upsert_routes([_route("drawer", images=True), _route("chat")], 1.0)
    back = {r.model_id: r.capabilities.images for r in store.active_routes()}
    assert back == {"drawer": True, "chat": False}


@pytest.mark.parametrize("column", ["images"])
def test_an_old_database_gains_the_column_defaulting_to_cannot(tmp_path, column):
    """The safe direction for a row written before anyone measured it.

    Defaulting to 1 would advertise generation on 52 routes that cannot do it,
    and every image request would burn a full failover chain to discover that.
    """
    import sqlite3
    db = str(tmp_path / "old.sqlite3")
    con = sqlite3.connect(db)
    con.executescript("""
        CREATE TABLE routes (key TEXT PRIMARY KEY, provider TEXT, model_id TEXT,
            tier TEXT, tools INTEGER, vision INTEGER, context INTEGER,
            max_output INTEGER, last_seen REAL, active INTEGER, priority INTEGER);
        INSERT INTO routes VALUES ('k/a','k','a','free',1,0,1000,100,1.0,1,1);
    """)
    con.commit()
    con.close()
    store = Storage(db)
    store.create_schema()
    assert column in {f[1] for f in store._con.execute("PRAGMA table_info(routes)")}
    assert store.active_routes()[0].capabilities.images is False
