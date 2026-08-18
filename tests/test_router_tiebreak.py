import random
from llm_libre.modelos import Capacidades, Metricas, Pedido, Ruta
from llm_libre.router import order_routes, TIE_BAND


def r(modelo, proveedor="kilo", tier="gratis", prioridad=1):
    return Ruta(proveedor=proveedor, modelo_id=modelo, tier=tier, prioridad=prioridad,
                capacidades=Capacidades(tools=True, vision=True, contexto=200000, max_salida=4096))


def m(calidad=0.8, confiabilidad=1.0, ttft=1500, cooldown=0.0, medida_en=1000.0):
    return Metricas(calidad=calidad, confiabilidad=confiabilidad, ttft_p50_ms=ttft,
                    en_cooldown_hasta=cooldown, calidad_medida_en=medida_en)


def test_sin_aleatorio_el_orden_sigue_siendo_determinista():
    """Los 447 tests que ya existian llaman order_routes() con 4 argumentos."""
    rutas = [r("a"), r("b"), r("c")]
    met = {"kilo/a": m(), "kilo/b": m(), "kilo/c": m()}
    primeras = {tuple(x.clave for x in order_routes(rutas, met, Pedido(), 0.0)) for _ in range(30)}
    assert len(primeras) == 1


def test_con_aleatorio_las_empatadas_rotan():
    rutas = [r("a"), r("b"), r("c")]
    met = {"kilo/a": m(), "kilo/b": m(), "kilo/c": m()}   # identicas: empate perfecto
    vistas = {order_routes(rutas, met, Pedido(), 0.0, random.Random(s))[0].clave for s in range(40)}
    assert vistas == {"kilo/a", "kilo/b", "kilo/c"}, vistas


def test_una_ruta_fuera_de_la_banda_nunca_gana():
    # 0.30 esta MUY por debajo del 5%: no debe entrar al sorteo jamas.
    rutas = [r("buena1"), r("buena2"), r("mala")]
    met = {"kilo/buena1": m(calidad=1.0), "kilo/buena2": m(calidad=1.0),
           "kilo/mala": m(calidad=0.30)}
    for s in range(60):
        salida = order_routes(rutas, met, Pedido(), 0.0, random.Random(s))
        assert salida[-1].clave == "kilo/mala"
        assert salida[0].clave in ("kilo/buena1", "kilo/buena2")


def test_el_sorteo_no_sube_una_ruta_de_pago_por_encima_de_lo_gratis():
    """El invariante mas importante del router."""
    rutas = [r("gratis1"), r("gratis2"), r("cara", proveedor="minimax", tier="pago", prioridad=0)]
    met = {"kilo/gratis1": m(calidad=0.5), "kilo/gratis2": m(calidad=0.5),
           "minimax/cara": m(calidad=1.0, ttft=50)}   # la de pago puntua MEJOR
    for s in range(60):
        salida = order_routes(rutas, met, Pedido(), 0.0, random.Random(s))
        assert salida[-1].tier == "pago", [x.clave for x in salida]


def test_el_sorteo_respeta_la_prioridad_del_yaml():
    rutas = [r("propio", proveedor="chatgpt", prioridad=0), r("tercero1"), r("tercero2")]
    met = {"chatgpt/propio": m(calidad=0.5), "kilo/tercero1": m(calidad=1.0, ttft=10),
           "kilo/tercero2": m(calidad=1.0, ttft=10)}   # los de prioridad 1 puntuan mejor
    for s in range(60):
        salida = order_routes(rutas, met, Pedido(), 0.0, random.Random(s))
        assert salida[0].clave == "chatgpt/propio"


def test_una_ruta_sin_medir_no_se_sortea_con_una_medida():
    rutas = [r("medida"), r("nueva")]
    met = {"kilo/medida": m(calidad=0.9, medida_en=1000.0),
           "kilo/nueva": m(calidad=0.9, medida_en=None)}
    for s in range(40):
        salida = order_routes(rutas, met, Pedido(), 0.0, random.Random(s))
        assert salida[0].clave == "kilo/medida"


def test_puntajes_en_cero_no_rompen_la_banda():
    rutas = [r("a"), r("b")]
    met = {"kilo/a": m(calidad=0.0, confiabilidad=0.0), "kilo/b": m(calidad=0.0, confiabilidad=0.0)}
    salida = order_routes(rutas, met, Pedido(), 0.0, random.Random(1))
    assert len(salida) == 2


def test_el_reparto_es_razonablemente_parejo():
    """No basta con que rote: no debe favorecer sistematicamente a una."""
    rutas = [r("a"), r("b"), r("c"), r("d")]
    met = {f"kilo/{x}": m() for x in "abcd"}
    cuenta = {}
    for s in range(4000):
        g = order_routes(rutas, met, Pedido(), 0.0, random.Random(s))[0].clave
        cuenta[g] = cuenta.get(g, 0) + 1
    assert len(cuenta) == 4
    for clave, n in cuenta.items():
        assert 800 < n < 1200, cuenta   # 1000 esperado, +-20%
