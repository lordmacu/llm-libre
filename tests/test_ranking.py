import pytest

from llm_libre.modelos import Metricas
from llm_libre.ranking import score

FAST_MEDIOCRE = Metricas(calidad=0.5, confiabilidad=1.0, ttft_p50_ms=200, en_cooldown_hasta=0)
SLOW_EXCELLENT = Metricas(calidad=1.0, confiabilidad=1.0, ttft_p50_ms=4000, en_cooldown_hasta=0)


def test_the_fast_profile_prefers_the_quick_mediocre_route():
    assert score(FAST_MEDIOCRE, "rapido") > score(SLOW_EXCELLENT, "rapido")


def test_the_powerful_profile_prefers_the_slow_excellent_route():
    assert score(SLOW_EXCELLENT, "potente") > score(FAST_MEDIOCRE, "potente")


def test_poor_reliability_sinks_every_profile():
    broken = Metricas(calidad=1.0, confiabilidad=0.05, ttft_p50_ms=100, en_cooldown_hasta=0)
    good = Metricas(calidad=0.7, confiabilidad=0.99, ttft_p50_ms=800, en_cooldown_hasta=0)
    for profile in ("rapido", "balanceado", "potente"):
        assert score(good, profile) > score(broken, profile)


def test_the_score_stays_between_zero_and_one():
    perfect = Metricas(calidad=1.0, confiabilidad=1.0, ttft_p50_ms=1, en_cooldown_hasta=0)
    assert 0.0 < score(perfect, "balanceado") <= 1.0


def test_an_unknown_profile_falls_back_to_balanced():
    m = Metricas(0.8, 0.9, 700, 0)
    assert score(m, "inventado") == pytest.approx(score(m, "balanceado"))
