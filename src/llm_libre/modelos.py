from dataclasses import dataclass

# Una ruta sin evaluar arranca en un valor neutro, no en 0: si arrancara en 0
# el router jamas la elegiria y por lo tanto nunca se evaluaria.
CALIDAD_NEUTRA = 0.6
CONFIABILIDAD_NEUTRA = 0.8
TTFT_NEUTRO_MS = 1500.0


@dataclass(frozen=True)
class Capacidades:
    tools: bool
    vision: bool
    contexto: int
    max_salida: int


@dataclass(frozen=True)
class Ruta:
    proveedor: str
    modelo_id: str
    tier: str  # "gratis" | "pago"
    capacidades: Capacidades

    @property
    def clave(self) -> str:
        return f"{self.proveedor}/{self.modelo_id}"


@dataclass(frozen=True)
class Pedido:
    modelo: str | None = None       # id explicito; None si vino un alias auto*
    requiere_tools: bool = False
    requiere_vision: bool = False
    min_contexto: int = 0
    perfil: str = "balanceado"      # "rapido" | "balanceado" | "potente"
    permitir_pago: bool = True


@dataclass(frozen=True)
class Metricas:
    calidad: float
    confiabilidad: float
    ttft_p50_ms: float
    en_cooldown_hasta: float        # epoch en segundos; 0 = sin castigo


METRICAS_NEUTRAS = Metricas(CALIDAD_NEUTRA, CONFIABILIDAD_NEUTRA, TTFT_NEUTRO_MS, 0.0)
