import random
from llm_libre.models import Capabilities, Metrics, RouteRequest, Route
from llm_libre.router import order_routes, TIE_BAND


def r(modelo, provider="kilo", tier="gratis", priority=1):
    return Route(provider=provider, model_id=modelo, tier=tier, priority=priority,
                capabilities=Capabilities(tools=True, vision=True, context=200000, max_output=4096))


def m(quality=0.8, reliability=1.0, ttft=1500, cooldown=0.0, medida_en=1000.0):
    return Metrics(quality=quality, reliability=reliability, ttft_p50_ms=ttft,
                    cooldown_until=cooldown, quality_measured_at=medida_en)


def test_sin_aleatorio_el_orden_sigue_siendo_determinista():
    """Los 447 tests que ya existian llaman order_routes() con 4 argumentos."""
    rutas = [r("a"), r("b"), r("c")]
    met = {"kilo/a": m(), "kilo/b": m(), "kilo/c": m()}
    primeras = {tuple(x.key for x in order_routes(rutas, met, RouteRequest(), 0.0)) for _ in range(30)}
    assert len(primeras) == 1


def test_con_aleatorio_las_empatadas_rotan():
    rutas = [r("a"), r("b"), r("c")]
    met = {"kilo/a": m(), "kilo/b": m(), "kilo/c": m()}   # identicas: empate perfecto
    vistas = {order_routes(rutas, met, RouteRequest(), 0.0, random.Random(s))[0].key for s in range(40)}
    assert vistas == {"kilo/a", "kilo/b", "kilo/c"}, vistas


def test_una_ruta_fuera_de_la_banda_nunca_gana():
    # 0.30 esta MUY por debajo del 5%: no debe entrar al sorteo jamas.
    rutas = [r("buena1"), r("buena2"), r("mala")]
    met = {"kilo/buena1": m(quality=1.0), "kilo/buena2": m(quality=1.0),
           "kilo/mala": m(quality=0.30)}
    for s in range(60):
        salida = order_routes(rutas, met, RouteRequest(), 0.0, random.Random(s))
        assert salida[-1].key == "kilo/mala"
        assert salida[0].key in ("kilo/buena1", "kilo/buena2")


def test_el_sorteo_no_sube_una_ruta_de_pago_por_encima_de_lo_gratis():
    """El invariante mas importante del router."""
    rutas = [r("gratis1"), r("gratis2"), r("cara", provider="minimax", tier="pago", priority=0)]
    met = {"kilo/gratis1": m(quality=0.5), "kilo/gratis2": m(quality=0.5),
           "minimax/cara": m(quality=1.0, ttft=50)}   # la de pago puntua MEJOR
    for s in range(60):
        salida = order_routes(rutas, met, RouteRequest(), 0.0, random.Random(s))
        assert salida[-1].tier == "pago", [x.key for x in salida]


def test_el_sorteo_respeta_la_prioridad_del_yaml():
    rutas = [r("propio", provider="chatgpt", priority=0), r("tercero1"), r("tercero2")]
    met = {"chatgpt/propio": m(quality=0.5), "kilo/tercero1": m(quality=1.0, ttft=10),
           "kilo/tercero2": m(quality=1.0, ttft=10)}   # los de prioridad 1 puntuan mejor
    for s in range(60):
        salida = order_routes(rutas, met, RouteRequest(), 0.0, random.Random(s))
        assert salida[0].key == "chatgpt/propio"


def test_una_ruta_sin_medir_no_se_sortea_con_una_medida():
    rutas = [r("medida"), r("nueva")]
    met = {"kilo/medida": m(quality=0.9, medida_en=1000.0),
           "kilo/nueva": m(quality=0.9, medida_en=None)}
    for s in range(40):
        salida = order_routes(rutas, met, RouteRequest(), 0.0, random.Random(s))
        assert salida[0].key == "kilo/medida"


def test_puntajes_en_cero_no_rompen_la_banda():
    rutas = [r("a"), r("b")]
    met = {"kilo/a": m(quality=0.0, reliability=0.0), "kilo/b": m(quality=0.0, reliability=0.0)}
    salida = order_routes(rutas, met, RouteRequest(), 0.0, random.Random(1))
    assert len(salida) == 2


def test_el_reparto_es_razonablemente_parejo():
    """No basta con que rote: no debe favorecer sistematicamente a una."""
    rutas = [r("a"), r("b"), r("c"), r("d")]
    met = {f"kilo/{x}": m() for x in "abcd"}
    cuenta = {}
    for s in range(4000):
        g = order_routes(rutas, met, RouteRequest(), 0.0, random.Random(s))[0].key
        cuenta[g] = cuenta.get(g, 0) + 1
    assert len(cuenta) == 4
    for clave, n in cuenta.items():
        assert 800 < n < 1200, cuenta   # 1000 esperado, +-20%
