from llm_libre import tool_emulator as _emu
from llm_libre.models import GATEWAY_EXTENSIONS
from llm_libre.providers import Provider, join_path


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
    return url, headers, out
