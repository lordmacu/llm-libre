from llm_libre import tool_emulator as _emu
from llm_libre import tracing
from llm_libre.models import GATEWAY_EXTENSIONS
from llm_libre.providers import Provider, join_path



def _with_trace(headers: dict) -> dict:
    """Carry this request's trace id on to the provider.

    Added at BUILD time rather than at each POST site because there are four of
    those (chat, stream, images, capability endpoints) and a trace with one hole
    in it answers nothing: the missing hop is exactly the one an operator would
    have blamed. Absent outside a request -- the probing scheduler is the
    gateway asking, not a caller, and tagging its traffic with a caller's id
    would make the logs lie.
    """
    rid = tracing.current()
    if rid:
        headers[tracing.HEADER] = rid
    return headers


def build_image_request(p: Provider, body: dict,
                        real_model: str | None = None) -> tuple[str, dict, dict]:
    """The `/images/generations` counterpart of `build_request`.

    Deliberately NOT a flag on build_request. The chat builder does three things
    that are wrong here and would have to be switched off one by one: it injects
    the tool-emulation prompt (there are no `tools` in an image request, and a
    provider that emulates them must not have its prompt rewritten), and its URL
    and success criteria belong to a different endpoint. A second small function
    is cheaper to read than a builder with three "except when generating images"
    branches.

    What it keeps identical: the URL is assembled with `join_path` (see its
    docstring for the query-string bug plain concatenation causes), the
    Authorization header is omitted for an empty key (Kilo's anonymous tier
    depends on that, and the rule should not differ per endpoint), and this
    gateway's own `x_*` extensions are stripped while a provider's own
    parameters travel untouched.
    """
    url = join_path(p.base_url, "/images/generations")
    headers = {"Content-Type": "application/json", **p.extra_headers}
    if p.api_key.strip():
        headers["Authorization"] = "Bearer " + p.api_key
    out = {k: v for k, v in body.items() if k not in GATEWAY_EXTENSIONS}
    if real_model is not None:
        out["model"] = real_model
    return url, _with_trace(headers), out


def build_request(p: Provider, body: dict,
                  real_model: str | None = None) -> tuple[str, dict, dict]:
    """Return (url, headers, body) ready to POST to /chat/completions.

    The returned body is a SHALLOW copy of the original, minus this gateway's own
    extensions (`x_*`, see GATEWAY_EXTENSIONS). Only top-level keys may be
    reassigned by the caller without affecting the original; nested structures
    (such as `messages`) are shared with it and must be treated as read-only
    during retries.

    Only THIS gateway's extensions are stripped, not everything starting with
    `x_`: the contract is passthrough, and a provider's own new parameter has to
    be able to travel.
    """
    # join_path parses and rebuilds the URL rather than concatenating raw text:
    # see its docstring for the bug that avoids when base_url carries a query
    # string (via CHATGPT_PROXY_URL with no path of its own).
    url = join_path(p.base_url, "/chat/completions")
    headers = {"Content-Type": "application/json", **p.extra_headers}
    # Empty key => do NOT send Authorization. Kilo's anonymous tier depends on this.
    if p.api_key.strip():
        headers["Authorization"] = "Bearer " + p.api_key
    out = {k: v for k, v in body.items() if k not in GATEWAY_EXTENSIONS}
    if real_model is not None:
        out["model"] = real_model
    if p.emulates_tools and out.get("tools"):
        out = _emu.inject_into_body(out)
    return url, _with_trace(headers), out


def build_capability_request(p: Provider, path: str,
                             body: dict | None = None) -> tuple[str, dict]:
    """URL and headers for a capability endpoint that takes NO model.

    `/audio/speech`, `/audio/transcriptions` and `/translate` are all of this
    shape: the provider is chosen by capability, and there is nothing to
    rewrite into the payload because there is no model id in it. So unlike the
    two builders above this one returns no body -- the caller sends either the
    client's JSON (with this gateway's own `x_*` extensions stripped) or, for
    a multipart upload, the raw bytes it received.

    Identical to the others where it matters: `join_path` assembles the URL
    (see its docstring for the query-string bug plain concatenation causes) and
    the Authorization header is omitted for an empty key, because Kilo's
    anonymous tier depends on that and the rule must not differ per endpoint.
    """
    url = join_path(p.base_url, path)
    headers = dict(p.extra_headers)
    if p.api_key.strip():
        headers["Authorization"] = "Bearer " + p.api_key
    if body is not None:
        headers["Content-Type"] = "application/json"
    return url, _with_trace(headers)


def strip_extensions(body: dict) -> dict:
    """The client's JSON minus this gateway's own `x_*` extensions.

    Only THIS gateway's, not everything starting with `x_`: the contract is
    passthrough, and a provider's own new parameter has to be able to travel.
    """
    return {k: v for k, v in body.items() if k not in GATEWAY_EXTENSIONS}
