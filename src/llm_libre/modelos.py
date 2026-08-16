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
    # Momento (epoch) de la ultima sonda de CALIDAD, o None si nunca se midio.
    # None significa que `calidad` de arriba es el supuesto neutro, no una
    # medicion: el router lo usa para no preferir una ruta sin evaluar por
    # encima de una con puntaje real, y /v1/ranking para no mostrar 0.6 como si
    # alguien lo hubiera medido.
    calidad_medida_en: float | None = None
    # Momento de la ultima sonda de cualquier tipo (salud o calidad). Lo pide
    # el §6 del diseno para /v1/ranking.
    ultima_sonda_en: float | None = None


METRICAS_NEUTRAS = Metricas(CALIDAD_NEUTRA, CONFIABILIDAD_NEUTRA, TTFT_NEUTRO_MS, 0.0)
