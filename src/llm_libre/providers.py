import logging
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

import yaml

from llm_libre.models import Capabilities, Route

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Provider:
    """One entry of proveedores.yaml, already resolved against the environment.

    NOTE ON NAMES: these fields may be renamed freely, but the YAML KEYS they are
    read from may not -- `load` below reads them as string literals, and
    tests/test_wire_contract.py freezes them. `prioridad` is the one field kept
    in Spanish for now: `Ruta.priority` shares the name, and the two are renamed
    together when modelos.py is migrated.
    """
    id: str
    tier: str            # "free" | "paid"
    dialect: str         # "openai"
    base_url: str
    api_key: str         # "" = anonymous
    models_path: str = ""
    extra_headers: dict = field(default_factory=dict)
    fixed_models: list = field(default_factory=list)
    # A DIFFERENT concept from tier (free|paid) and from profile
    # (rapido|balanceado|potente): the manual order in which the router tries
    # providers before looking at score. Default 100 so a provider that does not
    # declare it lands last among its peers. See the long comment in modelos.Ruta.
    priority: int = 100
    # A THIRD registration pattern, alongside "all discovered" (Kilo/OpenRouter,
    # via models_path) and "all declared" (MiniMax, via fixed_models): a provider
    # whose /models carries IDS but NO capability metadata (chatgpt-proxy).
    # None => normal mode (capabilities are discovered from /models, as always).
    # When declared, catalog.normalize applies these capabilities to EVERY id it
    # discovers and skips the price/modality checks -- but the IDS still come
    # from /models, never from here: it is still discovery, not a hardcoded list
    # under another name.
    default_capabilities: Capabilities | None = None
    # Per-model-id overrides on `default_capabilities`: only the fields that
    # differ. See `_exceptions` for why.
    exceptions: dict = field(default_factory=dict)
    # Task 13 review, finding 1: unwrapping canvas fences (':::word{...}' ...
    # ':::') used to be GLOBAL, but ':::nota{...}' is also standard
    # Docusaurus/MDX syntax -- applying it blindly strips the markers from ANY
    # provider (verified live against Kilo). Default False: only chatgpt-proxy,
    # which genuinely leaks canvas mode into content, declares it true.
    unwraps_canvas: bool = False
    # Task 13 review, finding 2 (in passing): None = use proxy.py's global
    # TIMEOUT_S (the long-standing behaviour). When declared, it bounds the worst
    # case of ONE particular provider without lowering the timeout for everyone
    # -- useful for one that can hang (see the hard-failure cooldown in Proxy,
    # the reason this was added).
    timeout_s: float | None = None
    # Tool-calling emulation via prompt injection: True = this provider has no
    # native tools, but inject_into_body turns `tools` into system-prompt text
    # and proxy.py detects JSON tool-call responses and converts them to the
    # OpenAI tool_calls format. See tool_emulator.py.
    emulates_tools: bool = False


def join_path(base_url: str, suffix: str) -> str:
    """Join a path suffix (e.g. "/chat/completions", "/models") to `base_url` by
    PARSING the URL, not by concatenating raw text.

    `base_url` may carry a query string -- via `CHATGPT_PROXY_URL` with no path
    of its own, see `_resolve_base_url` -- and a raw concatenation
    (`base_url.rstrip("/") + suffix`) glues the suffix INSIDE the query value
    instead of appending it to the path: `"...:8888?token=abc" +
    "/chat/completions"` yields `"...:8888?token=abc/chat/completions"` instead
    of `"...:8888/chat/completions?token=abc"`. Round 5: this bug was first fixed
    only in `_resolve_base_url` (where `Provider.base_url` is STORED), but
    `client.build_request` and `probing.sync_catalogue` were still
    concatenating text when building the FINAL URL actually sent over the network
    -- the splinter had moved one layer down. Both now go through here."""
    parts = urlsplit(base_url)
    return urlunsplit(parts._replace(path=parts.path.rstrip("/") + suffix))


def _resolve_base_url(p: dict, env: dict) -> str:
    """`base_url` in the YAML is always the default; `base_url_env`, when
    declared, names an environment variable that overrides it whenever it carries
    something (blank or absent falls back to the default). It exists so an
    in-house provider (chatgpt-proxy) can point at an address that is not yet
    fixed without editing the YAML on every deployment.

    If the default `base_url` carries a PATH (e.g. chatgpt's "/v1") and the
    environment variable CARRIES NO PATH OF ITS OWN (it is empty or just "/",
    i.e. the operator supplied nothing but the host), we add it: the operator
    only has to remember the host, not the provider's internal path. Without
    this, "CHATGPT_PROXY_URL=https://blog.example" (no /v1) leaves every chat
    request hitting a URL that does not exist -- the SAME footgun already fixed
    on the YAML side (original base_url without /v1, Task 13), surviving on the
    environment side.

    If the environment variable DOES carry a path of its own (e.g. a reverse
    proxy mount serving chat under its own prefix) and that path does not match
    the expected suffix, it is NOT touched: adding it anyway would turn ".../v2"
    (a path the operator chose on purpose) into ".../v2/v1/chat/completions",
    with no way to say "no, leave it alone". The operator said what they meant;
    we only warn, in case it was accidental -- see the warning below."""
    env_var = p.get("base_url_env")
    if not env_var:
        return p["base_url"]
    from_env = (env.get(env_var, "") or "").strip().rstrip("/")
    if not from_env:
        return p["base_url"]
    suffix = urlsplit(p["base_url"]).path.rstrip("/")
    if not suffix:
        return from_env
    parts = urlsplit(from_env)
    env_path = parts.path.rstrip("/")
    if not env_path:
        log.warning(
            "%s: %s='%s' carries no path; appending the '%s' suffix that "
            "proveedores.yaml declares for this provider. Defining the variable "
            "with the suffix already included avoids this warning.",
            p["id"], env_var, from_env, suffix)
        # join_path parses and rebuilds (urlsplit/urlunsplit), it does not
        # concatenate raw text -- see its docstring for the bug that avoids.
        return join_path(from_env, suffix)
    if env_path != suffix:
        log.warning(
            "%s: %s='%s' carries a path ('%s') different from the '%s' suffix "
            "proveedores.yaml declares by default for this provider; it is used "
            "AS IS, unmodified -- if that was accidental, fix the variable.",
            p["id"], env_var, from_env, env_path, suffix)
    return from_env


def _default_capabilities(p: dict) -> Capabilities | None:
    data = p.get("capacidades_por_defecto")
    if not data:
        return None
    return Capabilities(tools=bool(data["tools"]), vision=bool(data["vision"]),
                       context=int(data["contexto"]), max_output=int(data["max_salida"]))


def _exceptions(p: dict) -> dict:
    """Per-model-id overrides on `capacidades_por_defecto`.

    `capacidades_por_defecto` declares ONE capability set for every id discovered
    from /models, and that is enough when the catalogue is homogeneous. Grok's is
    not: of its 31 models, 25 return a real tool_call and 6 do not (measured
    2026-08-18 against the proxy, not assumed). Without exceptions the choice
    would be between lying about 6 routes or giving up dynamic discovery and
    declaring all of them by hand.

    Only the differing fields are declared; the rest is inherited from the
    default. An id that does not appear here uses the whole default, which is the
    normal case.
    """
    return {str(k): dict(v or {}) for k, v in (p.get("excepciones") or {}).items()}


def load(yaml_path: str, env: dict) -> list[Provider]:
    """Read proveedores.yaml. The KEYS below are the config contract -- see the
    note in Provider and tests/test_wire_contract.py."""
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [Provider(
        id=p["id"], tier=p["tier"], dialect=p["dialecto"],
        base_url=_resolve_base_url(p, env),
        api_key=(env.get(p.get("clave_env", ""), "") or "").strip(),
        models_path=p.get("modelos_path", ""),
        extra_headers=p.get("cabeceras_extra", {}) or {},
        fixed_models=p.get("modelos_fijos", []) or [],
        priority=int(p.get("prioridad", 100)),
        default_capabilities=_default_capabilities(p),
        exceptions=_exceptions(p),
        unwraps_canvas=bool(p.get("desenvuelve_canvas", False)),
        timeout_s=(float(p["timeout_s"]) if p.get("timeout_s") is not None else None),
        emulates_tools=bool(p.get("emula_tools", False)),
    ) for p in data["proveedores"]]


def fixed_routes(p: Provider) -> list[Route]:
    return [Route(p.id, m["id"], p.tier,
                 Capabilities(tools=True if p.emulates_tools else bool(m["tools"]),
                             vision=bool(m["vision"]),
                             context=int(m["contexto"]), max_output=int(m["max_salida"])),
                 priority=p.priority)
            for m in p.fixed_models]
