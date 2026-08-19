from llm_libre.models import Metrics

WEIGHTS: dict[str, tuple[float, float, float]] = {
    "fast": (0.4, 1.0, 2.0),
    "balanced": (1.0, 1.0, 1.0),
    "strong": (2.0, 1.0, 0.25),
}
REFERENCE_MS = 1500.0


def latency_factor(ttft_ms: float) -> float:
    """Map time-to-first-token onto (0, 1]. 1 would be instantaneous."""
    if ttft_ms <= 0:
        return 1.0
    return REFERENCE_MS / (REFERENCE_MS + ttft_ms)


def latency_signal_ms(m: Metrics) -> float:
    """The best latency evidence this route actually has, in milliseconds.

    Preference order: a measured time-to-first-token, then the measured full
    round-trip, then the neutral default.

    The fallback is the whole point. `ttft_p50_ms` is only ever written by
    streaming traffic -- the periodic probes are non-streaming on purpose and
    write 0, and so does this gateway's own non-streaming path -- so a deployment
    whose clients do not stream leaves EVERY route sitting at the neutral default.
    Measured 2026-08-18 against the live deployment: 48 of 52 routes were at
    exactly NEUTRAL_TTFT_MS. That is not a mild inaccuracy, it silently disables
    the profiles: a constant factor is common to every route, so
    `factor ** weight` scales all scores by the same amount and cannot change
    anyone's position. `fast`, `balanced` and `strong` returned byte-identical
    orderings, and `auto:strong` was in practice `auto:whatever`.

    The round-trip is a coarser signal than a real ttft -- it includes the whole
    generation, not just the wait for the first token -- and it is deliberately
    NOT rescaled before being fed to `latency_factor`. Two reasons. It is
    monotonic in the same direction (a route that answers sooner has both a lower
    ttft and a lower round-trip), which is all an ORDERING needs. And the observed
    ranges overlap closely enough that mixing them does not distort the comparison:
    on that same snapshot the four real ttft values spanned 2666-4337ms while the
    round-trips spanned 1279-8469ms.
    """
    if m.ttft_measured:
        return m.ttft_p50_ms
    if m.latency_p50_ms is not None:
        return m.latency_p50_ms
    return m.ttft_p50_ms


def score(m: Metrics, profile: str) -> float:
    wq, wr, wl = WEIGHTS.get(profile, WEIGHTS["balanced"])
    return ((m.quality ** wq) * (m.reliability ** wr)
            * (latency_factor(latency_signal_ms(m)) ** wl))
