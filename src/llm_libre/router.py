from llm_libre.modelos import METRICAS_NEUTRAS, Metricas, Pedido, Ruta
from llm_libre.ranking import puntuar


def ordenar(rutas: list[Ruta], metricas: dict[str, Metricas], pedido: Pedido,
            ahora: float) -> list[Ruta]:
    """Devuelve la cadena de intentos, mejor primero. Las de pago van siempre al final."""
    candidatas = [r for r in rutas if _cumple(r, pedido)]
    if pedido.modelo is not None:
        candidatas = [r for r in candidatas if r.modelo_id == pedido.modelo]
    if not pedido.permitir_pago:
        candidatas = [r for r in candidatas if r.tier == "gratis"]
    disponibles = [r for r in candidatas
                   if metricas.get(r.clave, METRICAS_NEUTRAS).en_cooldown_hasta <= ahora]

    def orden(r: Ruta) -> tuple[int, float]:
        # Primer criterio: haber sido medida. Una ruta que nunca paso por la
        # bateria de calidad lleva el valor NEUTRO, que es un supuesto, no una
        # medicion -- y un supuesto no puede ganarle a un puntaje real, o un
        # modelo recien aparecido (rapido y sin evaluar) se sirve como si fuera
        # el mejor durante todo un ciclo de calidad. Sigue en la lista, solo
        # que despues: tiene que recibir trafico alguna vez o nunca se mediria.
        m = metricas.get(r.clave, METRICAS_NEUTRAS)
        return (1 if m.calidad_medida_en is None else 0, -puntuar(m, pedido.perfil))

    gratis = sorted([r for r in disponibles if r.tier == "gratis"], key=orden)
    pago = sorted([r for r in disponibles if r.tier == "pago"], key=orden)
    return gratis + pago


def _cumple(r: Ruta, p: Pedido) -> bool:
    c = r.capacidades
    if p.requiere_tools and not c.tools:
        return False
    if p.requiere_vision and not c.vision:
        return False
    if p.min_contexto and c.contexto < p.min_contexto:
        return False
    return True
