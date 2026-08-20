"""The proxy capability contract: what an in-house proxy says it can do NOW.

Spec: docs/superpowers/specs/2026-08-20-proxy-capability-contract-design.md

The gateway used to declare every provider's capabilities by hand in
providers.yaml, at one moment in time, and nothing detected when that snapshot
stopped matching reality: `context: 128000` against a real 52815, `images: true`
against an account whose paid plan expires on a date the gateway cannot see.
This module reads the replacement -- a versioned block the proxy itself
publishes on GET /health -- and refuses anything it cannot fully trust.

Refusing means returning None, and None is a NORMAL, supported answer, not an
error: it means "this proxy has not adopted the contract", and the caller falls
back to exactly the behaviour that exists today. That is what makes the rollout
incremental, one proxy at a time, with no flag day.

This module knows the WIRE FORMAT and nothing else. It does not know what a
Route is, it does not decide precedence, and it never talks to the network --
catalog.py and probing.py own those. Keeping it that narrow is what lets the
contract be tested against fixtures alone.
"""
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

# The schema version THIS gateway speaks. A proxy declaring anything else is
# refused rather than parsed optimistically: a contract whose meaning we are
# guessing at is worse than no contract, because the fallback path is known-good.
VERSION = 1

# Every key a compliant `capabilities` block must carry. A proxy that cannot do
# something says `false`; it never omits the key. That distinction is the whole
# point of requiring the full set: an omission is indistinguishable from a proxy
# that forgot to report a capability it HAS, and guessing either way is wrong.
REQUIRED_CAPABILITIES = frozenset({
    "chat", "streaming", "tools", "vision", "images",
    "audio_speech", "audio_transcription", "translate",
    "search", "files", "conversations",
})

_AUTH_MODES = frozenset({"anonymous", "account", "unknown"})


@dataclass(frozen=True)
class Auth:
    """Who the proxy is talking to the vendor as. INFORMATIONAL ONLY.

    The gateway must never branch on `plan`: plan names are vendor-specific
    ("go", "free", "plus", "pro", ...) and teaching the gateway to read them
    rebuilds exactly the coupling this contract removes. It exists for the
    operator's /health view and for the subscription-expiry alert; everything
    the gateway ACTS on lives in `ProviderContract.capabilities`.
    """
    mode: str                            # "anonymous" | "account" | "unknown"
    plan: str | None = None
    subscription_active: bool = False
    expires_at: str | None = None        # ISO 8601 UTC, or None
    # Whether the proxy actually TOLD us this, or whether `_auth` supplied
    # `mode="unknown"` because the block was absent or malformed. Two different
    # facts used to arrive identically: "I asked my vendor and could not tell"
    # (a transient condition worth refusing a sweep over) and "I have no account
    # concept and said nothing" (perfectly normal -- grok has no plan tiers).
    # Refusing the second freezes that provider's catalogue forever, which is the
    # failure this whole contract exists to prevent.
    resolved: bool = False


@dataclass(frozen=True)
class ProviderContract:
    version: int
    provider: str
    auth: Auth
    capabilities: dict


def parse_health(provider: str, doc: object) -> ProviderContract | None:
    """A parsed contract, or None when this is not a contract document.

    `provider` is the id from providers.yaml, used for log messages and as the
    fallback when the document does not name itself.

    Every refusal below is all-or-nothing on purpose. A half-read contract would
    mix discovered values with fallback ones, and the resulting capability set
    would belong to no single source -- impossible to reason about when a route
    starts failing. Refuse, log why, and let the caller use the known-good path.
    """
    if not isinstance(doc, dict):
        return None
    version = doc.get("contract")
    # A missing key is the pre-contract proxy: SILENT. During the rollout that is
    # the majority case, and warning once per provider per sweep would train the
    # operator to ignore this log.
    if not isinstance(version, int) or isinstance(version, bool):
        return None
    if version != VERSION:
        log.warning(
            "contract %s: the proxy speaks version %r, this gateway speaks %d. "
            "Ignored -- falling back to providers.yaml.", provider, version, VERSION)
        return None
    caps = doc.get("capabilities")
    if not isinstance(caps, dict):
        log.warning(
            "contract %s: declares version %d but carries no 'capabilities' "
            "object. Ignored -- falling back to providers.yaml.", provider, version)
        return None
    missing = sorted(REQUIRED_CAPABILITIES - caps.keys())
    if missing:
        log.warning(
            "contract %s: 'capabilities' is missing %s. Ignored -- a partial "
            "block cannot be told apart from a proxy that forgot to report a "
            "capability it has.", provider, missing)
        return None
    # `isinstance(True, int)` is True in Python, so this check is written the
    # strict way round. It is not pedantry: chatgpt-proxy's pre-contract block
    # carried English prose in these fields, and a non-empty string is truthy --
    # a loose check would read "automatic (override with web_search...)" as
    # "yes, this provider does search", which is the wrong direction.
    not_boolean = sorted(k for k in REQUIRED_CAPABILITIES
                         if not isinstance(caps[k], bool))
    if not_boolean:
        log.warning(
            "contract %s: these capabilities are not booleans: %s. Ignored.",
            provider, not_boolean)
        return None
    named = doc.get("provider")
    return ProviderContract(
        version=version,
        provider=str(named) if isinstance(named, str) and named else provider,
        auth=_auth(provider, doc.get("auth")),
        # Only the required keys are kept: an unknown key is a capability this
        # gateway does not understand yet, and carrying it further would let it
        # reach code that assumes the fixed set.
        capabilities={k: bool(caps[k]) for k in REQUIRED_CAPABILITIES},
    )


def _auth(provider: str, data: object) -> Auth:
    """The `auth` block, degrading to "unknown" rather than refusing.

    A malformed `auth` does NOT invalidate the document, unlike a malformed
    `capabilities`: nothing routes on `auth`. Losing it costs an operator a line
    in /health and one alert; losing the capabilities would cost correct routing.
    The two are treated differently because they are worth different amounts.

    `Auth.resolved` distinguishes WHY `mode` ended up "unknown". An absent or
    malformed block leaves it at the dataclass default (`False`): the proxy said
    nothing, which is the normal, permanent shape for one with no account
    concept at all (grok has no plan tiers). A well-formed dict sets it `True`
    even when `mode` itself was unrecognised and downgraded below -- the proxy
    DID speak, it just used a string this contract version does not know.
    """
    if not isinstance(data, dict):
        return Auth(mode="unknown")
    mode = data.get("mode")
    if mode not in _AUTH_MODES:
        log.warning(
            "contract %s: auth.mode=%r is not one of %s; reported as 'unknown'.",
            provider, mode, sorted(_AUTH_MODES))
        mode = "unknown"
    plan = data.get("plan")
    expires_at = data.get("expires_at")
    return Auth(
        mode=mode,
        plan=plan if isinstance(plan, str) and plan else None,
        subscription_active=bool(data.get("subscription_active", False)),
        expires_at=expires_at if isinstance(expires_at, str) and expires_at else None,
        resolved=True,
    )
