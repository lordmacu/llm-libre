from dataclasses import dataclass

# Una ruta sin evaluar arranca en un valor neutro, no en 0: si arrancara en 0
# el router jamas la elegiria y por lo tanto nunca se evaluaria.
CALIDAD_NEUTRA = 0.6
CONFIABILIDAD_NEUTRA = 0.8
TTFT_NEUTRO_MS = 1500.0


# Las extensiones propias de ESTE gateway (§6 del diseno). Se interpretan aca y
# NO se reenvian al proveedor: son vocabulario interno, y un servidor estricto
# puede rechazar un cuerpo con campos que no conoce. El caso mas feo es
# `x_permitir_pago: false`, el campo cuyo trabajo es evitar gasto, siendo justo
# el que haga que el escalon de pago rechace la peticion.
EXTENSIONES_GATEWAY = frozenset({
    "x_requiere", "x_min_contexto", "x_permitir_pago", "x_crudo",
})


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
    # Concepto DISTINTO de tier y de perfil (ver Pedido.perfil mas abajo): el
    # orden manual en el que el router prueba las rutas antes de mirar
    # puntaje, p.ej. para que un proveedor propio (chatgpt-proxy) se pruebe
    # antes que los proveedores gratis de terceros. Default 100 -- alto a
    # proposito -- para que una ruta sin prioridad declarada quede ultima
    # entre sus pares, en vez de colarse antes que las que si la declararon.
    # NO participa en el invariante de tier: una ruta de pago con prioridad 0
    # sigue yendo al final (ver router.ordenar).
    prioridad: int = 100

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
    # p50 del round-trip COMPLETO, que es otra cosa que `ttft_p50_ms` (ver el
    # comentario de almacen.py). No entra en el puntaje: se expone para poder
    # diagnosticar. None = nunca se observo.
    latencia_p50_ms: float | None = None


METRICAS_NEUTRAS = Metricas(CALIDAD_NEUTRA, CONFIABILIDAD_NEUTRA, TTFT_NEUTRO_MS, 0.0)
