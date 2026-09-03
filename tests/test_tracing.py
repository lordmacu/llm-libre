"""One id, followed from the caller's phone to the provider and back.

The `events` table already groups a failover chain under a `request_id`, but
that id is minted HERE, so nothing in it can be matched against what the app
measured: three seconds of "the app is waiting" could be routing, the upstream,
or the network, and the rows cannot tell them apart. These tests pin the two
properties that make the id worth anything -- it is the CALLER's id when the
caller brought one, and it reaches the provider so the proxy logs the same
string.
"""
import httpx
import pytest
from fastapi.testclient import TestClient

from llm_libre.api import State, create_app
from llm_libre.models import Capabilities, Route
from llm_libre.providers import Provider
from llm_libre.proxy import Proxy
from llm_libre.storage import Storage

BODY = {"model": "auto", "messages": [{"role": "user", "content": "hola"}]}


def _client(seen: list | None = None):
    store = Storage(":memory:")
    store.create_schema()
    store.upsert_routes(
        [Route("kilo", "a:free", "free", Capabilities(True, False, 100000, 4096))], 1.0)
    prov = {"kilo": Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])}

    def handler(req: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(req)
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "hi"}}]})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    state = State(store=store, proxy=Proxy(prov, store, http),
                  api_keys={"buena"}, daily_paid_cap=200)
    return TestClient(create_app(state))


def test_the_callers_request_id_comes_back_on_the_response():
    c = _client()
    r = c.post("/v1/chat/completions", json=BODY,
               headers={"X-API-Key": "buena", "X-Request-Id": "app-abc123"})
    assert r.status_code == 200
    assert r.headers["X-Request-Id"] == "app-abc123"


def test_a_request_that_brings_no_id_is_given_one():
    # Nothing upstream of the gateway is obliged to trace. A request without an
    # id still has to be groupable in the events table, so one is minted.
    c = _client()
    r = c.post("/v1/chat/completions", json=BODY, headers={"X-API-Key": "buena"})
    assert r.status_code == 200
    assert r.headers.get("X-Request-Id")


def test_the_id_travels_on_to_the_provider():
    # The half that makes it a TRACE rather than a label: the proxy behind this
    # gateway logs the same string, so "the app waited 30s" can be split into
    # what the gateway spent and what the provider did.
    seen: list[httpx.Request] = []
    c = _client(seen)
    c.post("/v1/chat/completions", json=BODY,
           headers={"X-API-Key": "buena", "X-Request-Id": "app-abc123"})
    assert seen, "the provider was never called"
    assert seen[0].headers.get("x-request-id") == "app-abc123"


@pytest.mark.parametrize("bad", ["", "  ", "x" * 200, "id with spaces", "id\nnewline"])
def test_an_unusable_id_is_replaced_rather_than_trusted(bad):
    # The id is echoed into a response header and into log lines, so it is
    # attacker-controlled text: anything that is not a short, plain token is
    # dropped and a fresh one minted, rather than reflected.
    c = _client()
    r = c.post("/v1/chat/completions", json=BODY,
               headers={"X-API-Key": "buena", "X-Request-Id": bad})
    got = r.headers.get("X-Request-Id")
    assert got and got != bad
