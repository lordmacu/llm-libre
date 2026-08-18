import random

from llm_libre.models import Capabilities, Metrics, Route, RouteRequest
from llm_libre.router import TIE_BAND, order_routes


def r(model, provider="kilo", tier="gratis", priority=1):
    return Route(provider=provider, model_id=model, tier=tier, priority=priority,
                 capabilities=Capabilities(tools=True, vision=True, context=200000,
                                           max_output=4096))


def m(quality=0.8, reliability=1.0, ttft=1500, cooldown=0.0, measured_at=1000.0):
    return Metrics(quality=quality, reliability=reliability, ttft_p50_ms=ttft,
                   cooldown_until=cooldown, quality_measured_at=measured_at)


def test_without_an_rng_the_order_stays_deterministic():
    """The 447 tests that predate this feature call order_routes() with 4 args."""
    routes = [r("a"), r("b"), r("c")]
    met = {"kilo/a": m(), "kilo/b": m(), "kilo/c": m()}
    firsts = {tuple(x.key for x in order_routes(routes, met, RouteRequest(), 0.0))
              for _ in range(30)}
    assert len(firsts) == 1


def test_with_an_rng_the_tied_routes_rotate():
    routes = [r("a"), r("b"), r("c")]
    met = {"kilo/a": m(), "kilo/b": m(), "kilo/c": m()}   # identical: a perfect tie
    seen = {order_routes(routes, met, RouteRequest(), 0.0, random.Random(s))[0].key
            for s in range(40)}
    assert seen == {"kilo/a", "kilo/b", "kilo/c"}, seen


def test_a_route_outside_the_band_never_wins():
    # 0.30 is FAR below the 5% band: it must never enter the draw.
    routes = [r("buena1"), r("buena2"), r("mala")]
    met = {"kilo/buena1": m(quality=1.0), "kilo/buena2": m(quality=1.0),
           "kilo/mala": m(quality=0.30)}
    for s in range(60):
        out = order_routes(routes, met, RouteRequest(), 0.0, random.Random(s))
        assert out[-1].key == "kilo/mala"
        assert out[0].key in ("kilo/buena1", "kilo/buena2")


def test_the_draw_never_lifts_a_paid_route_above_the_free_ones():
    """The router's most important invariant."""
    routes = [r("gratis1"), r("gratis2"),
              r("cara", provider="minimax", tier="pago", priority=0)]
    met = {"kilo/gratis1": m(quality=0.5), "kilo/gratis2": m(quality=0.5),
           "minimax/cara": m(quality=1.0, ttft=50)}   # the paid one scores BETTER
    for s in range(60):
        out = order_routes(routes, met, RouteRequest(), 0.0, random.Random(s))
        assert out[-1].tier == "pago", [x.key for x in out]


def test_the_draw_respects_the_yamls_priority():
    routes = [r("propio", provider="chatgpt", priority=0), r("tercero1"), r("tercero2")]
    met = {"chatgpt/propio": m(quality=0.5), "kilo/tercero1": m(quality=1.0, ttft=10),
           "kilo/tercero2": m(quality=1.0, ttft=10)}   # the priority-1 ones score better
    for s in range(60):
        out = order_routes(routes, met, RouteRequest(), 0.0, random.Random(s))
        assert out[0].key == "chatgpt/propio"


def test_an_unmeasured_route_is_not_drawn_against_a_measured_one():
    routes = [r("medida"), r("nueva")]
    met = {"kilo/medida": m(quality=0.9, measured_at=1000.0),
           "kilo/nueva": m(quality=0.9, measured_at=None)}
    for s in range(40):
        out = order_routes(routes, met, RouteRequest(), 0.0, random.Random(s))
        assert out[0].key == "kilo/medida"


def test_zero_scores_do_not_break_the_band():
    routes = [r("a"), r("b")]
    met = {"kilo/a": m(quality=0.0, reliability=0.0),
           "kilo/b": m(quality=0.0, reliability=0.0)}
    out = order_routes(routes, met, RouteRequest(), 0.0, random.Random(1))
    assert len(out) == 2


def test_the_distribution_is_reasonably_even():
    """Rotating is not enough: it must not systematically favour one route."""
    routes = [r("a"), r("b"), r("c"), r("d")]
    met = {f"kilo/{x}": m() for x in "abcd"}
    counts = {}
    for s in range(4000):
        winner = order_routes(routes, met, RouteRequest(), 0.0, random.Random(s))[0].key
        counts[winner] = counts.get(winner, 0) + 1
    assert len(counts) == 4
    for key, n in counts.items():
        assert 800 < n < 1200, counts   # 1000 expected, +-20%
