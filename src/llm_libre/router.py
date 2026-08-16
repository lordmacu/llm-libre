from llm_libre.modelos import METRICAS_NEUTRAS, Metricas, Pedido, Ruta
from llm_libre.ranking import puntuar


def compatibles(rutas: list[Ruta], pedido: Pedido) -> list[Ruta]:
    """Las rutas que PODRIAN servir este pedido, ignorando si estan disponibles.

    Solo mira lo que el cliente pidio de forma inmutable -- capacidades,
    contexto, id explicito -- y no el cooldown ni el permiso de pago, que son
    estados del momento. Existe para separar los dos casos que el §9 del diseno
    separa y que la api venia mezclando en un 400: si esta lista queda vacia,
    NINGUNA ruta puede cumplir lo pedido nunca (400, error del cliente); si
    trae algo pero la cadena de `ordenar` sale vacia, hay rutas que podrian
    servir y estan caidas o en castigo (503, indisponibilidad).
    """
    candidatas = [r for r in rutas if _cumple(r, pedido)]
    if pedido.modelo is not None:
        candidatas = [r for r in candidatas if r.modelo_id == pedido.modelo]
    return candidatas


def ordenar(rutas: list[Ruta], metricas: dict[str, Metricas], pedido: Pedido,
            ahora: float) -> list[Ruta]:
    """Devuelve la cadena de intentos, mejor primero.

    Orden: `(tier == "pago", prioridad, no-medida, -puntaje)`.

    INVARIANTE que nada de lo que sigue puede romper: las rutas de PAGO van
    siempre al final, sin importar su `prioridad` ni su puntaje. Por eso
    `tier == "pago"` es el PRIMER criterio de la tupla (False < True ordena
    gratis antes que pago) y `prioridad` -- un concepto totalmente aparte,
    ver Ruta.prioridad -- entra recien despues: una ruta de pago con
    `prioridad: 0` no puede comprar un lugar antes que lo gratis. La plata es
    la razon.

    Dentro de un mismo tier, `prioridad` (menor primero) decide antes que el
    puntaje: es el orden manual declarado en el YAML (p.ej. un proveedor
    propio antes que los de terceros). A igual prioridad, el criterio previo
    a este cambio sigue intacto: una ruta nunca sondeada por la bateria de
    calidad (calidad_medida_en is None) va despues de una con medicion real,
    y recien ahi decide el puntaje.
    """
    candidatas = compatibles(rutas, pedido)
    if not pedido.permitir_pago:
        candidatas = [r for r in candidatas if r.tier == "gratis"]
    disponibles = [r for r in candidatas
                   if metricas.get(r.clave, METRICAS_NEUTRAS).en_cooldown_hasta <= ahora]

    def orden(r: Ruta) -> tuple[bool, int, int, float]:
        m = metricas.get(r.clave, METRICAS_NEUTRAS)
        return (r.tier == "pago", r.prioridad,
                1 if m.calidad_medida_en is None else 0,
                -puntuar(m, pedido.perfil))

    return sorted(disponibles, key=orden)


def _cumple(r: Ruta, p: Pedido) -> bool:
    c = r.capacidades
    if p.requiere_tools and not c.tools:
        return False
    if p.requiere_vision and not c.vision:
        return False
    if p.min_contexto and c.contexto < p.min_contexto:
        return False
    return True
