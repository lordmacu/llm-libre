from llm_libre.modelos import Capacidades, Ruta, Pedido, METRICAS_NEUTRAS


def test_la_clave_de_una_ruta_une_proveedor_y_modelo():
    r = Ruta("kilo", "poolside/laguna-s-2.1:free", "gratis",
             Capacidades(tools=True, vision=False, contexto=262144, max_salida=32768))
    assert r.clave == "kilo/poolside/laguna-s-2.1:free"


def test_el_pedido_por_defecto_es_balanceado_y_permite_pago():
    p = Pedido()
    assert p.perfil == "balanceado"
    assert p.permitir_pago is True
    assert p.modelo is None


def test_las_metricas_neutras_no_estan_en_cooldown():
    assert METRICAS_NEUTRAS.en_cooldown_hasta == 0.0
    assert 0.0 < METRICAS_NEUTRAS.calidad <= 1.0
