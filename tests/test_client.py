from llm_libre.client import build_request
from llm_libre.providers import Provider


def _prov(api_key="", extra=None):
    return Provider(id="kilo", tier="free", dialect="openai",
                     base_url="https://api.kilo.ai/api/gateway", api_key=api_key,
                     models_path="/models", extra_headers=extra or {}, fixed_models=[])


def test_without_a_key_no_authorization_is_sent():
    # Kilo's anonymous tier depends on this: sending an empty Bearer breaks it.
    url, headers, _ = build_request(_prov(), {"model": "x"})
    assert url == "https://api.kilo.ai/api/gateway/chat/completions"
    assert "Authorization" not in headers


def test_a_whitespace_only_key_sends_no_authorization():
    # load() normalises whitespace to ""; here we verify build_request does not
    # send Authorization for empty keys either.
    _, headers, _ = build_request(_prov(api_key="   "), {"model": "x"})
    assert "Authorization" not in headers


def test_with_a_key_it_sends_a_bearer():
    _, headers, _ = build_request(_prov(api_key="abc"), {"model": "x"})
    assert headers["Authorization"] == "Bearer abc"


def test_it_includes_the_extra_headers():
    _, headers, _ = build_request(_prov(extra={"X-Title": "llm-libre"}), {"model": "x"})
    assert headers["X-Title"] == "llm-libre"


def test_it_rewrites_the_model_to_the_providers_real_id():
    _, _, body = build_request(_prov(), {"model": "auto"}, real_model="poolside/x:free")
    assert body["model"] == "poolside/x:free"


def test_the_body_is_a_shallow_copy_and_does_not_mutate_the_original():
    """The returned body is a shallow copy of the original.

    Top-level keys (such as 'model') are independent after build_request, but
    nested structures (such as 'messages') are shared with the original on
    purpose and stay immutable during failover.
    """
    messages = [{"role": "user", "content": "hola"}]
    original = {"model": "auto", "messages": messages}
    url, headers, returned = build_request(_prov(), original, real_model="real")

    # (a) Top-level keys are independent: real_model rewrote only the returned one
    assert returned["model"] == "real"
    assert original["model"] == "auto"

    # (b) Nested structures are shared on purpose (identity)
    assert returned["messages"] is original["messages"]
    assert returned["messages"] is messages


# --- Fix round 3, I6: the gateway's extensions do not travel to the provider.
#     Kilo tolerates them; a stricter server has no reason to, and the ugliest
#     case is `x_permitir_pago: false` -- the field whose whole job is to AVOID
#     spending -- being the very one that makes the paid fallback reject the
#     request. ---

def test_it_does_not_forward_the_gateway_extensions_to_the_provider():
    original = {"model": "auto", "messages": [], "x_requiere": ["tools"],
                "x_min_contexto": 100000, "x_permitir_pago": False, "x_crudo": True}
    _, _, body = build_request(_prov(), original, real_model="real")
    assert "x_requiere" not in body
    assert "x_min_contexto" not in body
    assert "x_permitir_pago" not in body
    assert "x_crudo" not in body
    assert body["model"] == "real" and "messages" in body
    # And the original is untouched: the failover chain uses it again.
    assert "x_permitir_pago" in original


def test_it_lets_any_other_unknown_field_through():
    # Only THIS gateway's extensions are stripped. Whatever the client sends for
    # the provider (new parameters of whichever API) still travels: the contract
    # is passthrough.
    _, _, body = build_request(_prov(), {"model": "x", "reasoning": {"enabled": False},
                                         "provider": {"sort": "throughput"}})
    assert body["reasoning"] == {"enabled": False}
    assert body["provider"] == {"sort": "throughput"}


# --- Task 13, round 5: the previous round's fix only corrected where
#     _resolver_base_url STORES `Provider.base_url`, but build_request still
#     assembled the final URL by concatenating raw text
#     (`p.base_url.rstrip("/") + "/chat/completions"`). With a base_url carrying
#     a query string (via CHATGPT_PROXY_URL with no path of its own, e.g.
#     "https://blog.test:8888?token=abc"), that concatenation glues the suffix
#     INSIDE the query value -- the splinter had moved one layer down, to the URL
#     actually sent over the network. The test has to assert on that final URL,
#     not on `p.base_url`. ---

def test_a_query_string_in_base_url_does_not_splinter_the_final_url():
    prov = Provider(id="chatgpt", tier="free", dialect="openai",
                     base_url="https://blog.test:8888?token=abc", api_key="",
                     models_path="/models", extra_headers={}, fixed_models=[])
    url, _, _ = build_request(prov, {"model": "x"})
    assert url == "https://blog.test:8888/chat/completions?token=abc"
