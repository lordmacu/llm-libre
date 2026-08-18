from dataclasses import dataclass

# An unevaluated route starts at a neutral value, not at 0: starting at 0 would
# mean the router never picks it and therefore it is never evaluated.
NEUTRAL_QUALITY = 0.6
NEUTRAL_RELIABILITY = 0.8
NEUTRAL_TTFT_MS = 1500.0


# This gateway's own extensions (design section 6). They are interpreted here and
# NOT forwarded to the provider: they are internal vocabulary, and a strict server
# may reject a body carrying fields it does not know. The ugliest case is
# `x_allow_paid: false`, the field whose whole job is to avoid spending, being
# the very one that makes the paid tier reject the request.
#
# The VALUES are wire format -- clients send them -- and are frozen by
# tests/test_wire_contract.py. Only the constant's name is internal.
GATEWAY_EXTENSIONS = frozenset({
    "x_requires", "x_min_context", "x_allow_paid", "x_raw",
})


@dataclass(frozen=True)
class Capabilities:
    tools: bool
    vision: bool
    context: int
    max_output: int
    # Image GENERATION -- a different axis from `vision`, which is image INPUT.
    # A route can have either, both or neither: grok's imagine-agent-mode family
    # generates but cannot see, mistral does both, and most free chat routes do
    # neither. Defaulted to False so every existing construction site (tests,
    # catalogue normalisation, the fixed-model YAML entries) keeps working
    # unchanged and a provider has to CLAIM the capability to get it.
    images: bool = False


@dataclass(frozen=True)
class Route:
    provider: str
    model_id: str
    tier: str  # "free" | "paid"
    capabilities: Capabilities
    # A DIFFERENT concept from tier and from profile (see RouteRequest.profile
    # below): the manual order in which the router tries routes before looking at
    # score, e.g. so an in-house provider (chatgpt-proxy) is tried before
    # third-party free ones. Default 100 -- deliberately high -- so a route
    # without a declared priority lands last among its peers instead of cutting
    # ahead of the ones that did declare it. It does NOT take part in the tier
    # invariant: a paid route with priority 0 still goes last (see
    # router.order_routes).
    priority: int = 100

    @property
    def key(self) -> str:
        return f"{self.provider}/{self.model_id}"


@dataclass(frozen=True)
class RouteRequest:
    """What the client asked for, normalised.

    Named RouteRequest rather than Request because api.py already imports
    FastAPI's `Request`.

    CAREFUL: these field names used to BE the wire format -- api.py serialised
    `__dict__` straight into the 400 and 503 error bodies, so renaming a field
    silently changed the HTTP contract with nothing to catch it. `as_wire()`
    below now makes that mapping explicit, and tests/test_wire_contract.py
    asserts the keys it produces.
    """
    model: str | None = None        # explicit id; None if an auto* alias arrived
    needs_tools: bool = False
    needs_vision: bool = False
    # Set by POST /v1/images/generations, never by a chat request: it is what
    # keeps that endpoint from ever attempting a route that cannot generate.
    needs_images: bool = False
    min_context: int = 0
    profile: str = "balanced"       # "fast" | "balanced" | "strong"
    allow_paid: bool = True

    def as_wire(self) -> dict:
        """The view that goes into the 400/503 error bodies.

        Explicit on purpose: this used to be `request.__dict__`, which meant every
        field rename was an undeclared API change. The wire keys live HERE, in one
        place, and they now happen to match the field names -- which is convenient
        and NOT a licence to go back to `__dict__`: the point of the method is that
        the two can be renamed independently, and that test_wire_contract.py has a
        single place to assert.
        """
        return {
            "model": self.model,
            "needs_tools": self.needs_tools,
            "needs_vision": self.needs_vision,
            "needs_images": self.needs_images,
            "min_context": self.min_context,
            "profile": self.profile,
            "allow_paid": self.allow_paid,
        }


@dataclass(frozen=True)
class Metrics:
    quality: float
    reliability: float
    ttft_p50_ms: float
    cooldown_until: float           # epoch seconds; 0 = not punished
    # Moment (epoch) of the last QUALITY probe, or None if never measured. None
    # means the `quality` above is the neutral assumption, not a measurement: the
    # router uses it so as not to prefer an unevaluated route over one with a real
    # score, and /v1/ranking uses it so as not to show 0.6 as if someone had
    # measured it.
    quality_measured_at: float | None = None
    # Moment of the last probe of any kind (health or quality). Design section 6
    # asks for it for /v1/ranking.
    last_probe_at: float | None = None
    # p50 of the COMPLETE round-trip, which is a different thing from
    # `ttft_p50_ms` (see the comment in storage.py). It does not enter the score:
    # it is exposed for diagnostics. None = never observed.
    latency_p50_ms: float | None = None


NEUTRAL_METRICS = Metrics(NEUTRAL_QUALITY, NEUTRAL_RELIABILITY, NEUTRAL_TTFT_MS, 0.0)
