from llm_libre.modelos import Metricas

# (peso_calidad, peso_confiabilidad, peso_latencia). Exponentes: mas peso = mas castigo
# cuando ese componente baja de 1.
PESOS: dict[str, tuple[float, float, float]] = {
    "rapido": (0.4, 1.0, 2.0),
    "balanceado": (1.0, 1.0, 1.0),
    "potente": (2.0, 1.0, 0.25),
}
REFERENCIA_MS = 1500.0


def factor_latencia(ttft_ms: float) -> float:
    """Mapea time-to-first-token a (0, 1]. 1 seria instantaneo."""
    if ttft_ms <= 0:
        return 1.0
    return REFERENCIA_MS / (REFERENCIA_MS + ttft_ms)


def puntuar(m: Metricas, perfil: str) -> float:
    wc, wr, wl = PESOS.get(perfil, PESOS["balanceado"])
    return (m.calidad ** wc) * (m.confiabilidad ** wr) * (factor_latencia(m.ttft_p50_ms) ** wl)
