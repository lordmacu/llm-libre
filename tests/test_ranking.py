import pytest

from llm_libre.modelos import Metricas
from llm_libre.ranking import puntuar

RAPIDO_MEDIOCRE = Metricas(calidad=0.5, confiabilidad=1.0, ttft_p50_ms=200, en_cooldown_hasta=0)
LENTO_EXCELENTE = Metricas(calidad=1.0, confiabilidad=1.0, ttft_p50_ms=4000, en_cooldown_hasta=0)


def test_el_perfil_rapido_prefiere_al_veloz_mediocre():
    assert puntuar(RAPIDO_MEDIOCRE, "rapido") > puntuar(LENTO_EXCELENTE, "rapido")


def test_el_perfil_potente_prefiere_al_lento_excelente():
    assert puntuar(LENTO_EXCELENTE, "potente") > puntuar(RAPIDO_MEDIOCRE, "potente")


def test_la_falta_de_confiabilidad_hunde_cualquier_perfil():
    roto = Metricas(calidad=1.0, confiabilidad=0.05, ttft_p50_ms=100, en_cooldown_hasta=0)
    bueno = Metricas(calidad=0.7, confiabilidad=0.99, ttft_p50_ms=800, en_cooldown_hasta=0)
    for perfil in ("rapido", "balanceado", "potente"):
        assert puntuar(bueno, perfil) > puntuar(roto, perfil)


def test_el_puntaje_esta_acotado_entre_cero_y_uno():
    perfecto = Metricas(calidad=1.0, confiabilidad=1.0, ttft_p50_ms=1, en_cooldown_hasta=0)
    assert 0.0 < puntuar(perfecto, "balanceado") <= 1.0


def test_un_perfil_desconocido_cae_en_balanceado():
    m = Metricas(0.8, 0.9, 700, 0)
    assert puntuar(m, "inventado") == pytest.approx(puntuar(m, "balanceado"))
