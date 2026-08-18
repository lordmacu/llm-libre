from llm_libre.modelos import Metricas

# (quality_weight, reliability_weight, latency_weight). Higher exponent = harsher
# penalty when that component drops below 1.
#
# The KEYS are public API values -- they arrive from the client as the `auto:*`
# alias or `x_requiere`, and are frozen by tests/test_wire_contract.py. Only the
# constant's own name is internal.
PESOS: dict[str, tuple[float, float, float]] = {
    "rapido": (0.4, 1.0, 2.0),
    "balanceado": (1.0, 1.0, 1.0),
    "potente": (2.0, 1.0, 0.25),
}
REFERENCE_MS = 1500.0


def latency_factor(ttft_ms: float) -> float:
    """Map time-to-first-token onto (0, 1]. 1 would be instantaneous."""
    if ttft_ms <= 0:
        return 1.0
    return REFERENCE_MS / (REFERENCE_MS + ttft_ms)


def score(m: Metricas, profile: str) -> float:
    wq, wr, wl = PESOS.get(profile, PESOS["balanceado"])
    return (m.calidad ** wq) * (m.confiabilidad ** wr) * (latency_factor(m.ttft_p50_ms) ** wl)
