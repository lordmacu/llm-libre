import logging
import re

from dataclasses import replace

from llm_libre.modelos import Capacidades, Ruta

log = logging.getLogger(__name__)

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
# interpretar_pedido's own alias (api.py). Asking for the model "auto" always
# resolves to the alias, never to a real route -- so a route with that literal
# modelo_id would be unreachable forever. It is filtered here, at discovery time,
# so that NO provider (present or future) can accidentally sneak in an invalid
# route by name collision.
#
# INFO from the round 6 review: the literal "auto" is not enough. `ALIAS` in
# api.py (and `interpretar_pedido`) also treats as a reserved alias ANY id of the
# form "auto:<suffix>" -- "auto:rapido", "auto:potente", "auto:tools",
# "auto:vision" -- resolving it ALWAYS before comparing against `pedido.modelo`.
# A provider publishing a real model under one of those ids (or any other
# "auto:*" api.py adds tomorrow) created a permanently unreachable route, by
# exactly the mechanism that already justified reserving "auto". That is why
# `is_reserved_id` covers the whole PATTERN, not a list of ids known today.
RESERVED_IDS = frozenset({"auto"})


def is_reserved_id(id_: str) -> bool:
    """True if `id_` collides with "auto" or with any compound alias
    "auto:<suffix>" that interpretar_pedido (api.py) resolves before comparing
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


def normalize(provider: str, data: dict | list, priority: int = 100,
              default_capabilities: Capacidades | None = None,
              exceptions: dict | None = None,
              emulates_tools: bool = False) -> list[Ruta]:
    """Turn a /models response into usable free chat routes.

    `priority` belongs to the PROVIDER (see Proveedor.prioridad), not to anything
    /models could carry: it is stamped identically onto every discovered route so
    the router can order them without consulting the registry again.

    `default_capabilities` (Proveedor.capacidades_por_defecto) marks a provider
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
    """
    items = data.get("data", data) if isinstance(data, dict) else data
    routes: list[Ruta] = []
    for m in items:
        # An entry with no `id` (or that is not even a dict) used to blow up with
        # KeyError/AttributeError; sondeo.py swallowed it and that provider's
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
        if default_capabilities is not None:
            capabilities = default_capabilities
            # A discovered catalogue need NOT be homogeneous: grok publishes 31
            # ids of which 25 make tool calls and 6 do not. The exception
            # overrides ONLY the declared fields; whatever is not named is
            # inherited, so correcting one capability of one model does not force
            # repeating the other three.
            override = (exceptions or {}).get(m["id"])
            if override:
                capabilities = replace(capabilities, **override)
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
            capabilities = Capacidades(
                tools="tools" in supported,
                vision="image" in (arch.get("input_modalities") or []),
                contexto=int(m.get("context_length") or top.get("context_length") or 0),
                max_salida=int(top.get("max_completion_tokens") or 0),
            )
        if emulates_tools:
            capabilities = replace(capabilities, tools=True)
        routes.append(Ruta(
            proveedor=provider,
            modelo_id=m["id"],
            tier="gratis",
            capacidades=capabilities,
            prioridad=priority,
        ))
    return routes


def _is_free(m: dict) -> bool:
    price = (m.get("pricing") or {}).get("prompt")
    try:
        return float(price) == 0.0
    except (TypeError, ValueError):
        return False
