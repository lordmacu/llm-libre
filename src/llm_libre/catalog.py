import logging
import re

from dataclasses import replace

from llm_libre.contract import ProviderContract
from llm_libre.models import Capabilities, Route

log = logging.getLogger(__name__)

# The three capabilities that genuinely vary between models of the SAME
# provider: grok publishes 31 ids of which the three imagine-agent-mode ones
# draw but neither chat nor see, while the other 28 are the opposite. Everything
# else in the contract is a property of the provider's account, not of a model.
_PER_MODEL = ("tools", "vision", "images")

# Signals that a model, however free and however much text it returns, is NOT a
# general-purpose chat model and has no business in the `auto` rotation. They are
# read from the `name` and `description` fields the PROVIDER itself publishes in
# /models -- fields this normaliser used to throw away.
#
# This is still DISCOVERY, not hardcoded ids, and the distinction is the whole
# premise of the project (design section 1: "the catalogue is discovered from
# /models, always"): a blacklist of ids would rot exactly like the list of
# hardcoded ids this gateway exists to replace, whereas a new guardrail appearing
# tomorrow under another name will still describe itself as a guardrail.
#
# Why it matters: in a fresh install EVERY route starts at neutral quality, so
# until the first battery run the only discriminator is latency -- and the
# fastest thing in the free pool is
# `nvidia/nemotron-3.5-content-safety:free`, a classifier that answers
# "User Safety: safe" to anything. It ranked #1 in `auto`.
_SPECIALITY = (
    r"guardrail",
    r"content safety",
    r"\bmoderation\b",
    r"\bmoderates\b",
    r"\bclassifier\b",
    r"\breranker\b", r"\bre-ranker\b", r"\breranking\b",
    r"embeddings? model", r"text embeddings?\b",
    r"speech[- ]to[- ]text", r"text[- ]to[- ]speech",
)

# Meta-routers (`kilo-auto/free`, `openrouter/free`): not a model, a lottery
# between other models. Scoring one measures a roulette wheel -- its quality and
# latency are whoever came up that time -- and it also hides which route actually
# served, which is exactly what `X-Ruta-Usada` promises. They are discarded by
# the same means as everything else: what THEY say about themselves ("rotates
# through available free models", "a router that selects free models at random"),
# not their id.
_META_ROUTER = (
    r"\bmodels? router\b",
    r"\bis a router\b",
    r"rotates through",
    r"\brouter\b.{0,60}\bselects\b",
    r"selects .{0,40}\bmodels\b.{0,20}at random",
)

_DISCARD = re.compile("|".join(_SPECIALITY + _META_ROUTER), re.IGNORECASE)

# Ids reserved by llm-libre ITSELF, not by any provider: "auto" collides with
# parse_request's own alias (api.py). Asking for the model "auto" always
# resolves to the alias, never to a real route -- so a route with that literal
# modelo_id would be unreachable forever. It is filtered here, at discovery time,
# so that NO provider (present or future) can accidentally sneak in an invalid
# route by name collision.
#
# INFO from the round 6 review: the literal "auto" is not enough. `ALIAS` in
# api.py (and `parse_request`) also treats as a reserved alias ANY id of the
# form "auto:<suffix>" -- "auto:fast", "auto:strong", "auto:tools",
# "auto:vision" -- resolving it ALWAYS before comparing against `request.model`.
# A provider publishing a real model under one of those ids (or any other
# "auto:*" api.py adds tomorrow) created a permanently unreachable route, by
# exactly the mechanism that already justified reserving "auto". That is why
# `is_reserved_id` covers the whole PATTERN, not a list of ids known today.
RESERVED_IDS = frozenset({"auto"})

# Sustained requests per hour below which a route is a RESERVE, not a workhorse:
# it keeps its provider's band but sorts after the abundant routes in it.
#
# Measured against grok-proxy on 2026-08-19. It publishes 33 models that this
# gateway treated as 33 interchangeable routes, while their real sustained
# capacity differs by three orders of magnitude: the `grok-plugins-*` file agents
# and `imagine-agent-mode*` carry 999 requests/hour EACH on independent windows
# (~17,000/h between them), whereas `grok-3` carries 30 per 24h and `grok-4`
# seven. Underneath they are the same Grok 4.5.
#
# Treating them alike is expensive in both directions. The quality battery costs
# five requests per route per run -- 17% of grok-3's ENTIRE daily budget for a
# single run -- so real traffic arrives to find it exhausted, while the abundant
# pool of the same model sits idle. On the day this was measured the gateway's own
# probing had consumed 19 of grok-3's 30 daily requests.
#
# One request a minute is the line: above it a route can carry real traffic,
# below it it cannot. The measured populations are nowhere near that boundary
# (999/h against 1.25/h), so the exact value is not load-bearing.
SUSTAINED_RATE_FLOOR = 60


def _is_scarce(m: dict, measured: float | None = None) -> bool:
    """True if this route's sustained rate is known to be below the floor.

    Two sources, in order of authority. What the PROVIDER PUBLISHES
    (`requests_per_hour`) wins whenever it is there: it is a statement about
    policy, it covers the whole population rather than the slice we happened to
    send, and it is available before a single request is made.

    `measured` is Storage.rate_budgets' inference, and it exists for everyone
    else -- which is most of them. DeepSeek, the chatgpt proxy and Kilo's free
    pool publish nothing at all, so before this the floor could only ever demote
    grok routes, and a genuinely tiny allowance elsewhere stayed invisible until
    it started failing under real traffic.

    THE CALLER MUST PASS ONLY A MEASURED ALLOWANCE (RateBudget.measured), never
    the floor. `RateBudget.floor` is capped by how much traffic we sent, so an
    idle route reads low for want of demand -- feeding that here would demote
    routes for being unused, which is both wrong and self-reinforcing, since a
    demoted route receives even less traffic.

    Absent both -> False: a provider that publishes nothing AND has never been
    seen refusing keeps its previous behaviour exactly. Opt-in evidence, never an
    assumption -- the same principle as the rest of this module. The published
    value arrives as provider JSON, so anything non-numeric is treated as "did
    not say".
    """
    rate = m.get("requests_per_hour")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        rate = measured
    if rate is None:
        return False
    return rate < SUSTAINED_RATE_FLOOR


def is_reserved_id(id_: str) -> bool:
    """True if `id_` collides with "auto" or with any compound alias
    "auto:<suffix>" that parse_request (api.py) resolves before comparing
    against a literal id -- see the comment above RESERVED_IDS."""
    return id_ in RESERVED_IDS or id_.startswith("auto:")

# Providers that add legacy aliases to their own catalogue (chatgpt-proxy exposes
# "gpt-4o" as an alias of "auto", for instance) identify themselves with this
# prefix in their `description`. Keeping them would create a DUPLICATE route
# pointing at the same model under another name -- the ranking would end up
# measuring, and competing, the same model against itself.
_ALIAS_PREFIX = "Alias →"


def is_speciality_model(m: dict) -> bool:
    """True if the provider describes this model as something other than general chat."""
    return _DISCARD.search(f"{m.get('name') or ''} {m.get('description') or ''}") is not None


def _is_alias(m: dict) -> bool:
    return str(m.get("description") or "").startswith(_ALIAS_PREFIX)


def capabilities_from_contract(c: ProviderContract,
                               fallback: Capabilities | None) -> Capabilities:
    """The provider-level capabilities a contract asserts.

    The contract carries no sizes: `context` and `max_output` vary per model and
    arrive from /v1/models (see `apply_model_metadata`). Until one does, the YAML
    declaration is what there is -- and 0 when there is not even that, which is
    the honest value: `x_min_context` filtering on 0 excludes the route from
    requests that need a big window, rather than promising one it may not have.
    """
    caps = c.capabilities
    return Capabilities(
        tools=caps["tools"],
        vision=caps["vision"],
        context=fallback.context if fallback else 0,
        max_output=fallback.max_output if fallback else 0,
        images=caps["images"],
        audio_speech=caps["audio_speech"],
        audio_transcription=caps["audio_transcription"],
        translate=caps["translate"],
        search=caps["search"],
    )


def apply_model_metadata(base: Capabilities, m: dict, provider: str) -> Capabilities:
    """Narrow `base` with what /v1/models says about THIS model.

    A per-model value may only be NARROWER than the provider-level one. Widening
    is a contradiction, not extra information: the provider-level block is
    resolved against the ACCOUNT, so "this account cannot generate images" is a
    fact about the whole proxy, and a model claiming otherwise is describing what
    it could do on some other plan. It is logged and dropped, rather than
    refusing the whole document, because one confused entry should not cost the
    other twelve their real context windows.
    """
    fields = {}
    context = m.get("context_window")
    if isinstance(context, int) and not isinstance(context, bool) and context > 0:
        fields["context"] = context
    max_output = m.get("max_output_tokens")
    if isinstance(max_output, int) and not isinstance(max_output, bool) and max_output >= 0:
        fields["max_output"] = max_output
    per_model = m.get("capabilities")
    if isinstance(per_model, dict):
        for name in _PER_MODEL:
            value = per_model.get(name)
            if not isinstance(value, bool):
                continue
            if value and not getattr(base, name):
                log.warning(
                    "catalog %s: model %s claims %s=true while the provider "
                    "reports it false; the provider-level value wins.",
                    provider, m.get("id"), name)
                continue
            fields[name] = value
    return replace(base, **fields) if fields else base


def normalize(provider: str, data: dict | list, priority: int = 100,
              default_capabilities: Capabilities | None = None,
              exceptions: dict | None = None,
              emulates_tools: bool = False,
              measured_rates: dict[str, float] | None = None,
              contract: ProviderContract | None = None) -> list[Route]:
    """Turn a /models response into usable free chat routes.

    `priority` belongs to the PROVIDER (see Provider.priority), not to anything
    /models could carry: it is stamped identically onto every discovered route so
    the router can order them without consulting the registry again.

    `default_capabilities` (Provider.default_capabilities) marks a provider
    whose /models carries IDS but no capability metadata at all -- chatgpt-proxy
    is the case that motivated it: its catalogue is dynamic (discovered, not
    hardcoded) but only carries id/object/created/owned_by/description, never
    pricing nor modalities. When declared, those capabilities are applied
    IDENTICALLY to every discovered id and the price and output-modality checks
    are SKIPPED -- a provider declaring defaults is asserting what its catalogue
    cannot say. The speciality filter (guardrails, meta-routers) stays active in
    both modes: it has nothing to do with what the catalogue can or cannot
    report. This is still DISCOVERY in the sense that matters (design section 1):
    the IDS are read from /models, never hardcoded; only the capabilities come
    from the registry.

    `contract` (Provider.reads_capabilities) is the /health contract, and when
    present it takes over from `default_capabilities` as the provider-level
    source -- PRECEDENCE, strongest first, and the order is the whole design:
      1. `exceptions` (providers.yaml)  -- where a MEASURED lie is recorded
      2. /v1/models per-model values    -- narrowing only
      3. /health provider-level values  -- the contract
      4. `default_capabilities` (YAML)  -- the fallback for a proxy that
                                            has not adopted the contract
    """
    items = data.get("data", data) if isinstance(data, dict) else data
    routes: list[Route] = []
    native_tools_overridden = 0
    for m in items:
        # An entry with no `id` (or that is not even a dict) used to blow up with
        # KeyError/AttributeError; probing.py swallowed it and that provider's
        # ENTIRE catalogue froze forever, without a single log line saying so.
        # The broken entry is skipped and the fact is recorded.
        if not isinstance(m, dict):
            log.warning("catalog %s: entry is not an object, skipping: %.120r",
                        provider, m)
            continue
        if not m.get("id"):
            log.warning("catalog %s: entry without 'id', skipping: %.120r", provider, m)
            continue
        if is_reserved_id(m["id"]):
            continue
        if _is_alias(m):
            continue
        if is_speciality_model(m):
            log.info("catalog %s: %s discarded, the provider does not describe it "
                     "as a general chat model", provider, m["id"])
            continue
        # PRECEDENCE, strongest first -- and the order is the whole design:
        #   1. `exceptions` (providers.yaml)  -- where a MEASURED lie is recorded
        #   2. /v1/models per-model values    -- narrowing only
        #   3. /health provider-level values  -- the contract
        #   4. `default_capabilities` (YAML)  -- the fallback for a proxy that
        #                                        has not adopted the contract
        if contract is not None:
            capabilities = apply_model_metadata(
                capabilities_from_contract(contract, default_capabilities),
                m, provider)
        elif default_capabilities is not None:
            capabilities = default_capabilities
        else:
            if not _is_free(m):
                continue
            arch = m.get("architecture") or {}
            # Require text-ONLY output: music models (lyria) also list "text"
            # among their outputs, so "contains text" would let them through.
            outputs = set(arch.get("output_modalities") or ["text"])
            if outputs != {"text"}:
                continue
            supported = m.get("supported_parameters") or []
            top = m.get("top_provider") or {}
            capabilities = Capabilities(
                tools="tools" in supported,
                vision="image" in (arch.get("input_modalities") or []),
                context=int(m.get("context_length") or top.get("context_length") or 0),
                max_output=int(top.get("max_completion_tokens") or 0),
            )
        # `exceptions` applies to BOTH declared paths, and applies LAST: it is
        # the hand-written override, and the only thing that may contradict a
        # provider's own report of itself. A discovered catalogue need NOT be
        # homogeneous -- grok publishes 31 ids of which 25 make tool calls and 6
        # do not -- and the exception overrides ONLY the declared fields;
        # whatever is not named is inherited, so correcting one capability of
        # one model does not force repeating the other three.
        if contract is not None or default_capabilities is not None:
            override = (exceptions or {}).get(m["id"])
            if override:
                capabilities = replace(capabilities, **override)
        if emulates_tools:
            if capabilities.tools:
                native_tools_overridden += 1
            capabilities = replace(capabilities, tools=True)
        routes.append(Route(
            provider=provider,
            model_id=m["id"],
            tier="free",
            capabilities=capabilities,
            # A scarce route sorts one band behind its provider's abundant ones --
            # still ahead of the next provider, because it is held in reserve, not
            # demoted for being worse. See SUSTAINED_RATE_FLOOR.
            priority=priority + 1 if _is_scarce(
                m, (measured_rates or {}).get(f"{provider}/{m['id']}")) else priority,
        ))
    if native_tools_overridden:
        # Same contradiction fixed_routes warns about, aggregated: the flag
        # claims "no native function calling" while the discovered catalogue
        # reports models that HAVE it, and acting on the flag downgrades their
        # native calling to prompt injection. One line, not one per model -- a
        # provider like grok would print 25.
        log.warning(
            "%s: emulates_tools is set, but %d discovered models already "
            "report native tool support. Emulation will replace their native "
            "path with prompt injection.",
            provider, native_tools_overridden)
    return routes


def _is_free(m: dict) -> bool:
    price = (m.get("pricing") or {}).get("prompt")
    try:
        return float(price) == 0.0
    except (TypeError, ValueError):
        return False
