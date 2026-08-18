from llm_libre.models import Capabilities, Route, RouteRequest, NEUTRAL_METRICS


def test_la_clave_de_una_ruta_une_proveedor_y_modelo():
    r = Route("kilo", "poolside/laguna-s-2.1:free", "gratis",
             Capabilities(tools=True, vision=False, context=262144, max_output=32768))
    assert r.key == "kilo/poolside/laguna-s-2.1:free"


def test_una_ruta_sin_prioridad_declarada_queda_ultima_por_defecto():
    # Default 100: un proveedor que no declara prioridad no debe colarse
    # antes de uno que si la declaro con un numero bajo.
    r = Route("kilo", "modelo:free", "gratis",
             Capabilities(tools=True, vision=False, context=1000, max_output=100))
    assert r.priority == 100


def test_la_prioridad_se_puede_declarar_explicita():
    r = Route("chatgpt", "gpt-5-3-mini", "gratis",
             Capabilities(tools=False, vision=False, context=128000, max_output=8192),
             priority=0)
    assert r.priority == 0


def test_el_pedido_por_defecto_es_balanceado_y_permite_pago():
    p = RouteRequest()
    assert p.profile == "balanceado"
    assert p.allow_paid is True
    assert p.model is None


def test_las_metricas_neutras_no_estan_en_cooldown():
    assert NEUTRAL_METRICS.cooldown_until == 0.0
    assert 0.0 < NEUTRAL_METRICS.quality <= 1.0
