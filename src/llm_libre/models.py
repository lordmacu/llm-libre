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
    # PROVIDER-LEVEL capabilities, stamped identically onto every route of a
    # provider: unlike tools/vision/images they do not vary per model, because
    # the endpoints behind them take no model. They live on Capabilities anyway
    # so a future /v1/audio/speech can filter routes with the same
    # `compatible_routes` machinery as everything else, instead of growing a
    # second, parallel notion of what a provider can do.
    #
    # All four default to False, exactly as `images` did: every existing
    # construction site keeps working unchanged, and a provider gains a
    # capability by claiming it, never by omission.
    audio_speech: bool = False           # POST /v1/audio/speech      (TTS)
    audio_transcription: bool = False    # POST /v1/audio/transcriptions (STT)
    translate: bool = False              # POST /v1/translate
    search: bool = False                 # web_search on chat completions


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
    # `ttft_p50_ms` (see the comment in storage.py). None = never observed.
    #
    # It used to be diagnostics-only. It now BACKS the score's latency factor for
    # any route whose ttft was never really measured -- see ranking.latency_signal_ms
    # and `ttft_measured` right below.
    latency_p50_ms: float | None = None
    # Whether `ttft_p50_ms` above is a real observation or the neutral default.
    #
    # This exists because the neutral default made the latency factor a CONSTANT.
    # Measured 2026-08-18 against the live deployment: 48 of 52 routes had
    # ttft_p50_ms == NEUTRAL_TTFT_MS, because only streaming traffic records a real
    # ttft and both the probes and this gateway's non-streaming path write 0. A
    # constant is common to every route, so raising it to the profile's exponent
    # cannot reorder anything -- `fast`, `balanced` and `strong` returned literally
    # identical orderings in production. `latency_p50_ms` was populated on 51 of
    # those same 52 routes: the signal existed, the score just never read it.
    #
    # Defaults to True so that every direct construction (tests, NEUTRAL_METRICS)
    # keeps meaning "use the ttft I am handing you". Only storage.metrics(), which
    # is the one caller that can actually tell, passes False.
    ttft_measured: bool = True


# `ttft_measured=False`: nothing has been observed for a brand-new route, so its
# ttft is the neutral default by definition, not a measurement.
NEUTRAL_METRICS = Metrics(NEUTRAL_QUALITY, NEUTRAL_RELIABILITY, NEUTRAL_TTFT_MS, 0.0,
                          ttft_measured=False)


@dataclass(frozen=True)
class RateBudget:
    """How much of a route's rate limit is left, inferred from our own history.

    Providers that publish their allowance (grok sends `requests_per_hour`) are
    already handled in catalog.py. This is for everyone else: DeepSeek, the
    chatgpt proxy, Kilo's free pool -- none of them say how much they will take
    before refusing, so the only way to know is to remember what happened.

    THE WINDOW IS MEASURED, NOT ASSUMED. The first version of this fixed it at one
    hour, on the reasoning that providers publish per-hour numbers (grok sends
    `requests_per_hour`) so a measured and an advertised figure would be
    comparable. Against 10,000 live events that produced arithmetic that cannot
    be true: mistral-medium was credited with 4/h on a route already observed
    sustaining 50 clean requests in one hour, and six of eight routes came out
    the same way.

    Counting the same exhaustions over longer windows showed why. At the moment
    of refusal these routes had sent almost NOTHING recently -- a median of 0 to
    1 requests in the trailing minute, 4 to 10 in the trailing hour -- so no
    short-window burst limit explains them. Over 24h the numbers become both
    stable and coherent: grok's three imagine agents land on 60, 62 and 64, all
    comfortably above their hourly floors. These are DAILY quotas, and an hourly
    lens cannot see a daily quota except as nonsense.

    So `window_s` is part of the measurement rather than a constant, and
    Storage._budget picks it by which window the evidence actually supports.
    `perplexity/turbo` remains genuinely hourly (16 successes inside 10 seconds
    before a refusal, an actual burst), which is why the shorter window is tried
    first: where both fit, the tighter constraint is the one that binds.

    The three fields answer three different questions, and conflating them is
    the mistake this class exists to prevent:

      allowance -- what we believe the quota IS, per `window_s`. Only ever set
                   from an observed exhaustion: the route accepted N requests and
                   then said 429. `None` means we have never pushed this route to
                   its limit, which is the common case (measured 2026-08-19: 60
                   of 69 live routes had never once been refused).
      floor     -- what we have SEEN it sustain in a clean HOUR -- always hourly,
                   whatever `window_s` turns out to be, because it answers a
                   different question. Always available, always a LOWER BOUND: it
                   is capped by how much traffic we happened to send, so a quiet
                   route reads low for want of demand, not for want of capacity.
      used      -- successes in the trailing `window_s`. The consumption side.

    `remaining` is only meaningful when `per_hour` is known, and returns None
    otherwise rather than guessing -- the same rule as `quality_measured_at`.
    """
    allowance: float | None     # measured allowance PER `window_s`; None = unknown
    window_s: float             # the window that allowance belongs to
    used: int                   # successful requests in the trailing `window_s`
    floor: int                  # most ever sustained in an HOUR with no refusal
    episodes: int               # exhaustions the estimate rests on (0 = none)
    # Median seconds from a refusal to the route's next SUCCESS. An UPPER BOUND,
    # never the reset itself: the gateway backs off after a 429
    # (proxy._punish_429), so what this measures is "it had recovered by the time
    # we asked again" -- the true reset happened at some unknown earlier moment.
    # Useful exactly as that ("it was usable again within X"), misleading if read
    # as the provider's window. None while no refusal has been followed by a
    # success. When a provider states `Retry-After` the proxy already honours it
    # directly, which is a real answer and does not need inferring.
    recovery_s: float | None = None

    @property
    def remaining(self) -> float | None:
        """How many requests are believed to be left this hour, or None while the
        allowance is still unknown. Never negative: once consumption passes the
        estimate the honest answer is "none left", not a negative budget -- the
        estimate is a median over episodes, so exceeding it is expected."""
        if self.allowance is None:
            return None
        return max(0.0, self.allowance - self.used)

    @property
    def exhausts_in_s(self) -> float | None:
        """Seconds until the allowance runs out AT THE CURRENT RATE, or None when
        that cannot be said honestly -- either the allowance was never measured,
        or nothing is being consumed right now (an idle route never runs out, and
        dividing by a zero rate would claim it runs out instantly).

        This is the forward-looking half of the pair: `remaining` says how much is
        left, this says how long that lasts if traffic keeps behaving as it has
        for the past hour. It is a projection from one hour of history, so it
        moves as traffic moves -- which is the point. It answers "this one is
        nearly done" before the 429 arrives, rather than after."""
        left = self.remaining
        if left is None or self.used <= 0:
            return None
        return left / (self.used / self.window_s)

    @property
    def measured(self) -> bool:
        """True when `per_hour` rests on an observed refusal rather than on
        nothing. Callers that change ROUTING must gate on this: `floor` looks
        like a small allowance for any route we simply never pushed, and
        demoting on that would punish routes for being idle."""
        return self.allowance is not None

    @property
    def per_hour(self) -> float | None:
        """The allowance expressed as an hourly rate, for comparison against what
        providers advertise (grok's `requests_per_hour`) and against
        catalog.SUSTAINED_RATE_FLOOR.

        READ IT AS A RATE, NOT AS A BUDGET. A daily quota of 60 is 2.5/h here,
        and that number is true on average and misleading in the moment: nothing
        stops all 60 being spent in one hour, leaving 23 with none. `allowance`
        and `window_s` are what routing decisions should use; this is for putting
        two providers side by side."""
        if self.allowance is None:
            return None
        return self.allowance * 3600.0 / self.window_s


UNKNOWN_BUDGET = RateBudget(allowance=None, window_s=3600.0, used=0, floor=0,
                            episodes=0)


@dataclass(frozen=True)
class RequestTrace:
    """What ONE client request was, carried down to every attempt it causes.

    Without this the `events` table records outcomes per ROUTE and nothing else,
    which answers "is this route healthy" and cannot answer any of the questions
    an operator actually asks: what fraction of `auto:strong` ends up where, how
    often a request fails over, whether five rows against one route were five
    client requests or one request bouncing five times. Those are indistinguishable
    from route-keyed rows alone.

    `requested` is the client's own string, verbatim -- "auto:strong",
    "deepseek-chat" -- and not the resolved profile. The whole point is to compare
    what was ASKED against what was SERVED, so normalising it here would erase the
    left-hand side of that comparison.

    `request_id` groups the failover chain. It is what makes an attempt countable
    as part of a request rather than as a request of its own.

    Probes carry no trace (None): they are the gateway asking, not a client, and
    counting them as client traffic would inflate exactly the ratios this exists
    to measure.
    """
    request_id: str
    requested: str | None = None
