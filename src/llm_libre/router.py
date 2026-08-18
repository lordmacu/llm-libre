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


def clave_de_orden(r: Ruta, m: Metricas, perfil: str,
                   ahora: float) -> tuple[bool, bool, int, int, float]:
    """La clave de orden `(en-cooldown, tier == "pago", prioridad, no-medida,
    -puntaje)`, factorizada para que CUALQUIER lugar que quiera mostrar "el
    orden real del router" (hoy, /v1/ranking) use la MISMA logica en vez de
    inventar la suya -- /v1/ranking ordenaba antes solo por puntaje, sin
    `prioridad`, y podia mostrar una ruta arriba de todo mientras
    `X-Ruta-Usada` decia otra cosa. Ver el docstring de `ordenar` para el
    porque de cada posicion.

    `en-cooldown` (m.en_cooldown_hasta > ahora) es el PRIMER criterio,
    incluso antes de `tier`: una ruta castigada es algo que el router NUNCA
    va a elegir ahora mismo, sin importar tier/prioridad/puntaje, asi que en
    cualquier vista del "orden real" tiene que quedar al final de todo. En
    `ordenar()` esto es un no-op (las rutas en cooldown ya se filtraron
    ANTES de llegar aca, asi que este primer elemento siempre da False para
    todo lo que se ordena); es en /v1/ranking -- que muestra TODAS las rutas
    activas, cooldown incluido, por diagnostico -- donde este criterio hace
    el trabajo. `ahora` no tiene default a proposito: un default fijo (p.ej.
    0.0) haria que cualquier `en_cooldown_hasta > 0` -- incluido uno YA
    vencido hace rato -- se leyera como "todavia castigada" para siempre.
    """
    return (m.en_cooldown_hasta > ahora, r.tier == "pago", r.prioridad,
            1 if m.calidad_medida_en is None else 0,
            -puntuar(m, perfil))


def ordenar(rutas: list[Ruta], metricas: dict[str, Metricas], pedido: Pedido,
            ahora: float, aleatorio=None) -> list[Ruta]:
    """Devuelve la cadena de intentos, mejor primero.

    Orden: `(tier == "pago", prioridad, no-medida, -puntaje)` (ver
    `clave_de_orden`).

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

    El filtro de cooldown (mas abajo) va ANTES de mirar cualquiera de estos
    criterios: una ruta persistentemente rota o colgada no puede quedar
    primera solo por tener la prioridad mas alta (round 8: el cooldown lo
    dispara un 429 de inmediato o una SONDA que confirma que la ruta esta
    rota -- ver Proxy._sospechar en proxy.py -- nunca el trafico real solo).
    """
    candidatas = compatibles(rutas, pedido)
    if not pedido.permitir_pago:
        candidatas = [r for r in candidatas if r.tier == "gratis"]
    disponibles = [r for r in candidatas
                   if metricas.get(r.clave, METRICAS_NEUTRAS).en_cooldown_hasta <= ahora]

    def orden(r: Ruta) -> tuple[bool, bool, int, int, float]:
        return clave_de_orden(r, metricas.get(r.clave, METRICAS_NEUTRAS), pedido.perfil, ahora)

    ordenadas = sorted(disponibles, key=orden)
    if aleatorio is None:
        return ordenadas
    return rotar_empates(ordenadas, metricas, pedido.perfil, ahora, aleatorio)


def _cumple(r: Ruta, p: Pedido) -> bool:
    c = r.capacidades
    if p.requiere_tools and not c.tools:
        return False
    if p.requiere_vision and not c.vision:
        return False
    if p.min_contexto and c.contexto < p.min_contexto:
        return False
    return True


# Ancho de la banda de empate, como fraccion del mejor puntaje del grupo. Una
# ruta que puntua >= mejor * (1 - BANDA_EMPATE) se considera empatada con la
# mejor y entra al sorteo.
#
# 0.05 (5%) no es un numero magico sino una lectura del ranking real
# (2026-08-17): las 21 rutas activas puntuaban entre 0.48 y 0.50 -- un rango
# de 4% -- porque la bateria de calidad le da 1.00 a TODAS (un modelo de 2.6B
# pasa sus 5 casos igual que uno de 550B, verificado en vivo). Con esa
# realidad, 5% agrupa a practicamente todo el catalogo, que es exactamente lo
# que se quiere: si ninguna es mediblemente mejor, elegir siempre la misma
# solo quema la cuota de un proveedor mientras los otros miran.
#
# Es deliberadamente una BANDA y no un sorteo global: el dia que la bateria
# separe de verdad (o que las latencias reales se separen), las rutas peores
# se caen del grupo solas y el sorteo se angosta sin tocar este numero.
BANDA_EMPATE = 0.05


def _categoria(r: Ruta, m: Metricas, ahora: float) -> tuple[bool, bool, int, int]:
    """La parte CATEGORICA de la clave de orden -- todo menos el puntaje.

    Dos rutas solo pueden sortearse entre si dentro de la misma categoria, y
    eso es lo que protege los invariantes: una ruta de pago nunca comparte
    categoria con una gratis, ni una prioridad 0 con una prioridad 1, asi que
    el sorteo JAMAS puede subir una de pago por encima de lo gratis ni pasar
    por encima del orden manual del YAML. Lo unico que se sortea es el
    desempate por puntaje, que es justamente el criterio que hoy no
    discrimina.
    """
    return clave_de_orden(r, m, "balanceado", ahora)[:4]


def rotar_empates(ordenadas: list[Ruta], metricas: dict[str, Metricas],
                  perfil: str, ahora: float, aleatorio) -> list[Ruta]:
    """Baraja las rutas que estan empatadas con la mejor de su categoria.

    `ordenadas` ya viene ordenada por `ordenar`. Se recorre por categorias
    consecutivas; dentro de cada una, las que puntuan >= mejor * (1 -
    BANDA_EMPATE) se barajan entre si y el resto conserva su orden detras.

    `aleatorio` es cualquier objeto con `.shuffle` (p.ej. `random.Random`).
    Se inyecta en vez de usar el `random` global para que los tests puedan
    sembrarlo y para que `ordenar(...)` sin este argumento siga siendo
    completamente determinista -- que es lo que esperan los tests que ya
    existian antes de esta funcion.
    """
    salida: list[Ruta] = []
    i = 0
    while i < len(ordenadas):
        cat = _categoria(ordenadas[i], metricas.get(ordenadas[i].clave, METRICAS_NEUTRAS), ahora)
        j = i
        while j < len(ordenadas) and _categoria(
                ordenadas[j], metricas.get(ordenadas[j].clave, METRICAS_NEUTRAS), ahora) == cat:
            j += 1
        grupo = ordenadas[i:j]
        puntajes = [puntuar(metricas.get(r.clave, METRICAS_NEUTRAS), perfil) for r in grupo]
        mejor = puntajes[0]
        # `mejor <= 0` (todas las rutas del grupo puntuando cero) haria que el
        # piso fuera 0 y entrara TODO al sorteo, incluida una ruta que puntua
        # peor que otra: con puntajes no positivos la banda relativa no
        # significa nada, asi que no se sortea y manda el orden determinista.
        if mejor > 0:
            piso = mejor * (1 - BANDA_EMPATE)
            k = 0
            while k < len(grupo) and puntajes[k] >= piso:
                k += 1
            empatadas = grupo[:k]
            aleatorio.shuffle(empatadas)
            salida.extend(empatadas)
            salida.extend(grupo[k:])
        else:
            salida.extend(grupo)
        i = j
    return salida
