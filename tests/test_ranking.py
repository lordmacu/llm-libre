import pytest

from llm_libre.models import Metrics
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
