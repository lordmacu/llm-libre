from llm_libre.modelos import METRICAS_NEUTRAS, Metricas, Pedido, Ruta
from llm_libre.ranking import score


def compatible_routes(routes: list[Ruta], request: Pedido) -> list[Ruta]:
    """The routes that COULD serve this request, ignoring whether they are available.

    It only looks at what the client asked for immutably -- capabilities,
    context, explicit id -- not at cooldown or the paid-tier permission, which
    are momentary states. It exists to separate the two cases design section 9
    separates and the api used to conflate into one 400: if this list comes back
    empty, NO route can ever satisfy the request (400, client error); if it has
    entries but `order_routes` returns an empty chain, there are routes that
    could serve and they are down or being punished (503, unavailability).
    """
    candidates = [r for r in routes if _satisfies(r, request)]
    if request.modelo is not None:
        candidates = [r for r in candidates if r.modelo_id == request.modelo]
    return candidates


def sort_key(r: Ruta, m: Metricas, profile: str,
             now: float) -> tuple[bool, bool, int, int, float]:
    """The ordering key `(in-cooldown, tier == "pago", priority, unmeasured,
    -score)`, factored out so that ANY place wanting to show "the router's real
    order" (today, /v1/ranking) uses the SAME logic instead of inventing its own
    -- /v1/ranking used to sort by score alone, without `prioridad`, and could
    show one route at the top while `X-Ruta-Usada` said something else. See the
    docstring of `order_routes` for why each position is where it is.

    `in-cooldown` (m.en_cooldown_hasta > now) is the FIRST criterion, even before
    `tier`: a punished route is something the router will NEVER pick right now,
    regardless of tier/priority/score, so in any view of the "real order" it has
    to land at the very end. In `order_routes()` this is a no-op (cooldown routes
    were already filtered out BEFORE reaching here, so this first element is
    always False for everything being sorted); it is in /v1/ranking -- which
    shows ALL active routes, cooldown included, for diagnostics -- that this
    criterion does the work. `now` deliberately has no default: a fixed default
    (say 0.0) would make any `en_cooldown_hasta > 0` -- including one that
    expired long ago -- read as "still punished" forever.
    """
    return (m.en_cooldown_hasta > now, r.tier == "pago", r.prioridad,
            1 if m.calidad_medida_en is None else 0,
            -score(m, profile))


def order_routes(routes: list[Ruta], metrics: dict[str, Metricas], request: Pedido,
                 now: float, rng=None) -> list[Ruta]:
    """Return the chain of attempts, best first.

    Order: `(tier == "pago", priority, unmeasured, -score)` (see `sort_key`).

    An INVARIANT nothing below may break: PAID routes always go last, regardless
    of their `prioridad` or their score. That is why `tier == "pago"` is the
    FIRST criterion of the tuple (False < True orders free before paid) and
    `prioridad` -- an entirely separate concept, see Ruta.prioridad -- only comes
    after: a paid route with `prioridad: 0` cannot buy a place ahead of free
    ones. Money is the reason.

    Within the same tier, `prioridad` (lower first) decides before the score: it
    is the manual order declared in the YAML (e.g. an in-house provider before
    third-party ones). At equal priority, the criterion that predates this change
    stays intact: a route never probed by the quality battery
    (calidad_medida_en is None) goes after one with a real measurement, and only
    then does the score decide.

    The cooldown filter (below) runs BEFORE any of these criteria are consulted:
    a persistently broken or hung route cannot come first just by having the
    highest priority (round 8: cooldown is triggered by a 429 immediately or by a
    PROBE confirming the route is broken -- see Proxy._sospechar in proxy.py --
    never by real traffic alone).
    """
    candidates = compatible_routes(routes, request)
    if not request.permitir_pago:
        candidates = [r for r in candidates if r.tier == "gratis"]
    available = [r for r in candidates
                 if metrics.get(r.clave, METRICAS_NEUTRAS).en_cooldown_hasta <= now]

    def key(r: Ruta) -> tuple[bool, bool, int, int, float]:
        return sort_key(r, metrics.get(r.clave, METRICAS_NEUTRAS), request.perfil, now)

    ordered = sorted(available, key=key)
    if rng is None:
        return ordered
    return shuffle_ties(ordered, metrics, request.perfil, now, rng)


def _satisfies(r: Ruta, p: Pedido) -> bool:
    c = r.capacidades
    if p.requiere_tools and not c.tools:
        return False
    if p.requiere_vision and not c.vision:
        return False
    if p.min_contexto and c.contexto < p.min_contexto:
        return False
    return True


# Width of the tie band, as a fraction of the group's best score. A route
# scoring >= best * (1 - TIE_BAND) counts as tied with the best and enters the
# draw.
#
# 0.05 (5%) is not a magic number but a reading of the real ranking (2026-08-17):
# the 21 active routes scored between 0.48 and 0.50 -- a 4% range -- because the
# quality battery gives 1.00 to ALL of them (a 2.6B model passes its 5 cases just
# like a 550B one, verified live). Given that reality, 5% groups practically the
# whole catalogue, which is exactly what is wanted: if none is measurably better,
# always picking the same one just burns one provider's quota while the others
# watch.
#
# It is deliberately a BAND and not a global draw: the day the battery really
# separates routes (or real latencies do), the worse ones fall out of the group
# on their own and the draw narrows without touching this number.
TIE_BAND = 0.05


def _category(r: Ruta, m: Metricas, now: float) -> tuple[bool, bool, int, int]:
    """The CATEGORICAL part of the sort key -- everything except the score.

    Two routes can only be drawn against each other within the same category, and
    that is what protects the invariants: a paid route never shares a category
    with a free one, nor a priority 0 with a priority 1, so the draw can NEVER
    lift a paid route above the free ones nor override the YAML's manual order.
    The only thing being drawn is the score tiebreak, which is precisely the
    criterion that does not discriminate today.
    """
    return sort_key(r, m, "balanceado", now)[:4]


def shuffle_ties(ordered: list[Ruta], metrics: dict[str, Metricas],
                 profile: str, now: float, rng) -> list[Ruta]:
    """Shuffle the routes tied with the best of their category.

    `ordered` already comes sorted by `order_routes`. It is walked by consecutive
    categories; within each one, those scoring >= best * (1 - TIE_BAND) are
    shuffled among themselves and the rest keep their order behind them.

    `rng` is any object with `.shuffle` (e.g. `random.Random`). It is injected
    rather than using the global `random` so tests can seed it and so
    `order_routes(...)` without this argument stays completely deterministic --
    which is what the tests predating this function expect.
    """
    out: list[Ruta] = []
    i = 0
    while i < len(ordered):
        cat = _category(ordered[i], metrics.get(ordered[i].clave, METRICAS_NEUTRAS), now)
        j = i
        while j < len(ordered) and _category(
                ordered[j], metrics.get(ordered[j].clave, METRICAS_NEUTRAS), now) == cat:
            j += 1
        group = ordered[i:j]
        scores = [score(metrics.get(r.clave, METRICAS_NEUTRAS), profile) for r in group]
        best = scores[0]
        # `best <= 0` (every route in the group scoring zero) would put the floor
        # at 0 and let EVERYTHING into the draw, including a route scoring worse
        # than another: with non-positive scores the relative band means nothing,
        # so no draw happens and the deterministic order stands.
        if best > 0:
            floor = best * (1 - TIE_BAND)
            k = 0
            while k < len(group) and scores[k] >= floor:
                k += 1
            tied = group[:k]
            rng.shuffle(tied)
            out.extend(tied)
            out.extend(group[k:])
        else:
            out.extend(group)
        i = j
    return out
