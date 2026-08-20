import logging
from dataclasses import dataclass, field, replace
from urllib.parse import urlsplit, urlunsplit

import yaml

from llm_libre.models import Capabilities, Route

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Provider:
    """One entry of providers.yaml, already resolved against the environment.

    NOTE ON NAMES: these fields may be renamed freely, but the YAML KEYS they are
    read from may not -- `load` below reads them as string literals, and
    tests/test_wire_contract.py freezes them. `priority` is the one field kept
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
    # (fast|balanced|strong): the manual order in which the router tries
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
    # The same shape of decision as `unwraps_canvas`, for grok-proxy: strip its
    # '<xai:...>'/'<grok:...>' UI cards from `content`. Default False -- a model
    # asked to explain grok's wire format would legitimately quote those tags,
    # so only the provider that emits them for real declares it true. See
    # reasoning.XaiCardTrimmer, and note it does NOT cover grok's plain-text
    # status labels: those carry no marker here and are dropped at the source.
    strips_xai_cards: bool = False
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
    # Whether this provider's /health publishes the capability contract (spec
    # 2026-08-20). False -- the default -- is today's behaviour exactly:
    # capabilities come from default_capabilities/fixed_models and nothing extra
    # is requested. It is opt-in per provider precisely so the five in-house
    # proxies can adopt the contract one at a time, in any order, with no flag
    # day: a provider that has not adopted it must not have its sync skipped.
    reads_capabilities: bool = False


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


_API_PREFIX = "/v1"


def contract_url(base_url: str) -> str:
    """Where a proxy publishes its capability contract, derived from `base_url`.

    THE ASYMMETRY, because it is a fact about these proxies and not an
    arbitrary choice: their API lives under `/v1` (`/v1/chat/completions`,
    `/v1/models`) but `/health` sits BESIDE that prefix, at the root, because it
    is the container health check and predates the API surface. Measured against
    the live chatgpt-proxy on 2026-08-20: `/v1/health` -> 404, `/health` -> 200.
    `join_path(base_url, "/health")` therefore produced a URL no proxy serves,
    and the 404 skipped the whole provider on every sweep.

    So a trailing `/v1` is stripped from the path -- and NOTHING else is.
    Reducing to the origin instead would be wrong: `_resolve_base_url`
    deliberately supports an operator-chosen reverse-proxy mount, so
    `https://host/proxy/v1` must yield `https://host/proxy/health`; dropping the
    mount would ask for a path the reverse proxy does not serve. A `base_url`
    with no `/v1` suffix simply gains `/health`.

    Parses and rebuilds the URL for the same reason `join_path` does: `base_url`
    may carry a query string (`CHATGPT_PROXY_URL` with a token), and raw
    concatenation would splice the suffix into the query value.
    """
    parts = urlsplit(base_url)
    path = parts.path.rstrip("/")
    if path.endswith(_API_PREFIX):
        path = path[: -len(_API_PREFIX)]
    return urlunsplit(parts._replace(path=path + "/health"))


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
            "providers.yaml declares for this provider. Defining the variable "
            "with the suffix already included avoids this warning.",
            p["id"], env_var, from_env, suffix)
        # join_path parses and rebuilds (urlsplit/urlunsplit), it does not
        # concatenate raw text -- see its docstring for the bug that avoids.
        return join_path(from_env, suffix)
    if env_path != suffix:
        log.warning(
            "%s: %s='%s' carries a path ('%s') different from the '%s' suffix "
            "providers.yaml declares by default for this provider; it is used "
            "AS IS, unmodified -- if that was accidental, fix the variable.",
            p["id"], env_var, from_env, env_path, suffix)
    return from_env


def _default_capabilities(p: dict) -> Capabilities | None:
    data = p.get("default_capabilities")
    if not data:
        return None
    # `images` is OPTIONAL, unlike the other four: it was added after every
    # provider entry already existed, and a missing declaration must mean
    # "cannot generate" rather than a KeyError at startup. It is also the safe
    # default -- a provider gains the capability by claiming it, never by
    # omission.
    return Capabilities(tools=bool(data["tools"]), vision=bool(data["vision"]),
                       context=int(data["context"]), max_output=int(data["max_output"]),
                       images=bool(data.get("images", False)))


def _exceptions(p: dict) -> dict:
    """Per-model-id overrides on `default_capabilities`.

    `default_capabilities` declares ONE capability set for every id discovered
    from /models, and that is enough when the catalogue is homogeneous. Grok's is
    not: of its 31 models, 25 return a real tool_call and 6 do not (measured
    2026-08-18 against the proxy, not assumed). Without exceptions the choice
    would be between lying about 6 routes or giving up dynamic discovery and
    declaring all of them by hand.

    Only the differing fields are declared; the rest is inherited from the
    default. An id that does not appear here uses the whole default, which is the
    normal case.
    """
    return {str(k): dict(v or {}) for k, v in (p.get("exceptions") or {}).items()}


def load(yaml_path: str, env: dict) -> list[Provider]:
    """Read providers.yaml. The KEYS below are the config contract -- see the
    note in Provider and tests/test_wire_contract.py."""
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [Provider(
        id=p["id"], tier=p["tier"], dialect=p["dialect"],
        base_url=_resolve_base_url(p, env),
        api_key=(env.get(p.get("api_key_env", ""), "") or "").strip(),
        models_path=p.get("models_path", ""),
        extra_headers=p.get("extra_headers", {}) or {},
        fixed_models=p.get("fixed_models", []) or [],
        priority=int(p.get("priority", 100)),
        default_capabilities=_default_capabilities(p),
        exceptions=_exceptions(p),
        unwraps_canvas=bool(p.get("unwraps_canvas", False)),
        strips_xai_cards=bool(p.get("strips_xai_cards", False)),
        timeout_s=(float(p["timeout_s"]) if p.get("timeout_s") is not None else None),
        emulates_tools=bool(p.get("emulates_tools", False)),
        reads_capabilities=bool(p.get("reads_capabilities", False)),
    ) for p in data["providers"]]


def fixed_routes(p: Provider, contract=None) -> list[Route]:
    """The routes declared by hand in `fixed_models`.

    When the provider publishes the capability contract, a declaration here is
    checked against it and the route is DROPPED -- not silently downgraded --
    when the contract contradicts it. `chatgpt`'s `dall-e-3` is the case: it
    declares `images: true` and can do nothing else, so on a lapsed plan
    downgrading it to `images: false` would leave a route that looks like an
    ordinary chat model, gets picked for chat, and answers nothing. Dropping it
    is the honest outcome: the declaration asserted a shape the provider no
    longer has.

    `fixed_models` is not `exceptions`, and only `exceptions` outranks discovery.
    This block exists to name ids a catalogue does not publish, not to overrule
    live account state -- which is precisely what "the Go subscription expired"
    is.

    ORDER MATTERS, and it is the same order `catalog.normalize` uses:
    the contradiction is decided against what `fixed_models` DECLARED, and
    `emulates_tools` is applied AFTERWARDS. Emulation is a claim about this
    gateway, not about the proxy -- deepseek and mistral have no native function
    calling and llm-libre supplies it by injecting the tool definitions into the
    prompt (tool_emulator.py) -- so a truthful `tools: false` from their /health
    agrees with the declaration rather than contradicting it. Folding emulation
    in first made every one of deepseek's routes read as contradicted, which
    would have emptied a `priority: 0` provider out of the catalogue the day it
    adopted the contract.
    """
    routes = []
    for m in p.fixed_models:
        capabilities = Capabilities(
            tools=bool(m["tools"]),
            vision=bool(m["vision"]),
            context=int(m["context"]), max_output=int(m["max_output"]),
            images=bool(m.get("images", False)))
        if contract is not None:
            caps = contract.capabilities
            contradicted = [name for name in ("tools", "vision", "images")
                            if getattr(capabilities, name) and not caps[name]]
            if contradicted:
                log.warning(
                    "%s: the fixed model %s declares %s, which this provider's "
                    "/health reports it cannot do right now. The route is left "
                    "out of the catalogue for this sweep.",
                    p.id, m["id"], ", ".join(contradicted))
                continue
            capabilities = replace(
                capabilities,
                audio_speech=caps["audio_speech"],
                audio_transcription=caps["audio_transcription"],
                translate=caps["translate"],
                search=caps["search"])
        if p.emulates_tools:
            capabilities = replace(capabilities, tools=True)
        routes.append(Route(p.id, m["id"], p.tier, capabilities, priority=p.priority))
    return routes
