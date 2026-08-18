"""Bounding what an UNAUTHENTICATED caller can cost.

Written the day the gateway went public. Until then the only limiter was keyed
by API key and ran AFTER the key was validated, which left two holes that do not
matter behind a firewall and matter a lot on the open internet:

  - a wrong key raises 401 without ever reaching a limiter, so guessing keys --
    or simply hammering the endpoint -- costs the caller nothing and is bounded
    by nothing;
  - /health and /v1/assets carry no key by design, so they had no limiter at
    all. /health is the most expensive endpoint in the service: it recalculates
    every route's metrics on every call.
"""
import httpx
from fastapi.testclient import TestClient

from llm_libre.api import State, create_app
from llm_libre.auth import RateLimiter, client_ip
from llm_libre.models import Capabilities, Route
from llm_libre.providers import Provider
from llm_libre.proxy import Proxy
from llm_libre.storage import Storage


def _client(per_ip=3, per_key=100):
    store = Storage(":memory:")
    store.create_schema()
    store.upsert_routes([Route("p", "m:free", "free",
                               Capabilities(True, False, 1000, 100))], 1.0)
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "hi"}}]})))
    provs = {"p": Provider("p", "free", "openai", "https://p.test", "", "/models", {}, [])}
    state = State(store=store, proxy=Proxy(provs, store, http), api_keys={"good"},
                  daily_paid_cap=200, rate_limiter=RateLimiter(per_key),
                  ip_rate_limiter=RateLimiter(per_ip))
    return TestClient(create_app(state)), state


# --- the limiter itself ---

def test_it_counts_a_sliding_minute():
    rl = RateLimiter(2)
    assert rl.allow("a", now=0.0)
    assert rl.allow("a", now=1.0)
    assert not rl.allow("a", now=2.0)
    assert rl.allow("a", now=61.5)      # the first two have aged out


def test_buckets_are_independent():
    rl = RateLimiter(1)
    assert rl.allow("a", now=0.0)
    assert not rl.allow("a", now=0.1)
    assert rl.allow("b", now=0.1)


def test_idle_buckets_are_evicted_so_the_limiter_is_not_the_leak():
    """Remembering every address that ever knocked IS a memory exhaustion.

    A limiter that grows one entry per source, forever, hands an attacker the
    exact resource it was added to protect.
    """
    from llm_libre.auth import MAX_TRACKED
    rl = RateLimiter(5)
    for i in range(MAX_TRACKED + 50):
        rl.allow(f"ip-{i}", now=0.0)
    rl.allow("fresh", now=10_000.0)     # everything else is long idle
    assert len(rl._hits) < MAX_TRACKED


def test_an_active_bucket_is_never_evicted():
    # Evicting one would hand a heavy caller a clean slate -- the opposite of
    # what the limiter is for.
    from llm_libre.auth import MAX_TRACKED
    rl = RateLimiter(1)
    rl.allow("heavy", now=0.0)
    for i in range(MAX_TRACKED + 50):
        rl.allow(f"ip-{i}", now=0.0)
    assert not rl.allow("heavy", now=1.0)


# --- which address is counted ---

class _Req:
    def __init__(self, headers, host="127.0.0.1"):
        self.headers = headers
        self.client = type("C", (), {"host": host})()


def test_the_real_address_comes_from_cloudflare_not_the_socket():
    """Behind the tunnel every socket says 127.0.0.1.

    Counting that would make ONE bucket for the whole internet -- a limiter that
    the first caller fills and everyone else is denied by.
    """
    assert client_ip(_Req({"cf-connecting-ip": "203.0.113.7"})) == "203.0.113.7"
    assert client_ip(_Req({"x-forwarded-for": "203.0.113.9, 10.0.0.1"})) == "203.0.113.9"
    assert client_ip(_Req({})) == "127.0.0.1"
    assert client_ip(_Req({}, host="")) == "unknown"


# --- through the app ---

def test_a_wrong_key_is_bounded_instead_of_free():
    client, _ = _client(per_ip=3)
    codes = [client.get("/v1/ranking",
                        headers={"Authorization": "Bearer wrong"}).status_code
             for _ in range(5)]
    assert codes[:3] == [401, 401, 401]
    assert codes[3:] == [429, 429], codes


def test_health_is_bounded_although_it_needs_no_key():
    client, _ = _client(per_ip=2)
    assert [client.get("/health").status_code for _ in range(4)] == [200, 200, 429, 429]


def test_the_asset_endpoint_is_bounded_too():
    client, _ = _client(per_ip=2)
    path = "/v1/assets/" + "a" * 64
    codes = [client.get(path).status_code for _ in range(4)]
    assert codes == [404, 404, 429, 429]


def test_the_ip_limit_applies_before_the_key_is_even_looked_at():
    """Order matters: it is the whole point of the change.

    With a valid key the caller is still bounded by address, so one machine
    holding a leaked key cannot use it without limit either.
    """
    client, _ = _client(per_ip=2, per_key=1000)
    good = {"Authorization": "Bearer good"}
    assert client.get("/v1/models", headers=good).status_code == 200
    assert client.get("/v1/models", headers=good).status_code == 200
    assert client.get("/v1/models", headers=good).status_code == 429


def test_different_addresses_do_not_share_an_allowance():
    client, _ = _client(per_ip=1)
    a = {"Authorization": "Bearer wrong", "CF-Connecting-IP": "203.0.113.1"}
    b = {"Authorization": "Bearer wrong", "CF-Connecting-IP": "203.0.113.2"}
    assert client.get("/v1/ranking", headers=a).status_code == 401
    assert client.get("/v1/ranking", headers=a).status_code == 429
    assert client.get("/v1/ranking", headers=b).status_code == 401   # unaffected


def test_with_no_ip_limiter_configured_nothing_changes():
    # Every existing test builds a State without one; they must keep passing.
    client, state = _client()
    state.ip_rate_limiter = None
    assert all(client.get("/health").status_code == 200 for _ in range(20))
