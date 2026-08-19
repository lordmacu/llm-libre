import pytest

from llm_libre.models import NEUTRAL_TTFT_MS, Metrics
from llm_libre.ranking import score

FAST_MEDIOCRE = Metrics(quality=0.5, reliability=1.0, ttft_p50_ms=200, cooldown_until=0)
SLOW_EXCELLENT = Metrics(quality=1.0, reliability=1.0, ttft_p50_ms=4000, cooldown_until=0)


def test_the_fast_profile_prefers_the_quick_mediocre_route():
    assert score(FAST_MEDIOCRE, "fast") > score(SLOW_EXCELLENT, "fast")


def test_the_powerful_profile_prefers_the_slow_excellent_route():
    assert score(SLOW_EXCELLENT, "strong") > score(FAST_MEDIOCRE, "strong")


def test_poor_reliability_sinks_every_profile():
    broken = Metrics(quality=1.0, reliability=0.05, ttft_p50_ms=100, cooldown_until=0)
    good = Metrics(quality=0.7, reliability=0.99, ttft_p50_ms=800, cooldown_until=0)
    for profile in ("fast", "balanced", "strong"):
        assert score(good, profile) > score(broken, profile)


def test_the_score_stays_between_zero_and_one():
    perfect = Metrics(quality=1.0, reliability=1.0, ttft_p50_ms=1, cooldown_until=0)
    assert 0.0 < score(perfect, "balanced") <= 1.0


def test_an_unknown_profile_falls_back_to_balanced():
    m = Metrics(0.8, 0.9, 700, 0)
    assert score(m, "inventado") == pytest.approx(score(m, "balanced"))


# --- Production reality, measured 2026-08-18 against the live deployment ------
#
# The tests above all hand `score()` a DIFFERENT ttft per route, so the latency
# factor discriminates and the profiles look like they work. Production does not
# look like that: 48 of the 52 active routes had `ttft_p50_ms` sitting at exactly
# NEUTRAL_TTFT_MS, because only streaming traffic ever records a real one and the
# probes are non-streaming by design. A constant latency factor is common to
# every route, so raising it to the profile exponent cannot reorder anything --
# `fast`, `balanced` and `strong` returned byte-identical orderings live.
#
# `latency_p50_ms` (the full round-trip) WAS populated on 51 of those 52 routes.
# The signal was there; the score just never looked at it.

UNMEASURED_TTFT = dict(ttft_p50_ms=NEUTRAL_TTFT_MS, ttft_measured=False, cooldown_until=0)


def test_the_profiles_diverge_when_only_the_round_trip_was_measured():
    """Two routes, neither with a real ttft: the slow-but-excellent one must win
    `strong` and lose `fast`. This is the production case."""
    quick_mediocre = Metrics(quality=0.5, reliability=1.0, latency_p50_ms=1300.0,
                             **UNMEASURED_TTFT)
    slow_excellent = Metrics(quality=1.0, reliability=1.0, latency_p50_ms=8000.0,
                             **UNMEASURED_TTFT)
    assert score(slow_excellent, "strong") > score(quick_mediocre, "strong")
    assert score(quick_mediocre, "fast") > score(slow_excellent, "fast")


def test_an_unmeasured_ttft_does_not_flatten_the_latency_factor():
    """Two routes identical except for their measured round-trip must not score
    the same. They did: both fell back to the same fabricated constant."""
    quick = Metrics(quality=1.0, reliability=1.0, latency_p50_ms=1300.0, **UNMEASURED_TTFT)
    slow = Metrics(quality=1.0, reliability=1.0, latency_p50_ms=8000.0, **UNMEASURED_TTFT)
    assert score(quick, "fast") > score(slow, "fast")


def test_a_real_ttft_still_wins_over_the_round_trip_fallback():
    """The fallback is a fallback: a route that genuinely measured its ttft is
    scored on that, not on its round-trip."""
    m = Metrics(quality=1.0, reliability=1.0, ttft_p50_ms=200.0, ttft_measured=True,
                latency_p50_ms=9000.0, cooldown_until=0)
    assert score(m, "fast") == pytest.approx(
        score(Metrics(quality=1.0, reliability=1.0, ttft_p50_ms=200.0,
                      ttft_measured=True, cooldown_until=0), "fast"))


def test_with_nothing_measured_at_all_the_neutral_value_is_used():
    """A brand new route has neither signal: it must still score, not crash."""
    m = Metrics(quality=1.0, reliability=1.0, latency_p50_ms=None, **UNMEASURED_TTFT)
    assert 0.0 < score(m, "balanced") <= 1.0
