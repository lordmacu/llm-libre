from llm_libre.api import interpretar_pedido


def test_auto_es_balanceado():
    p = interpretar_pedido({"model": "auto"})
    assert p.modelo is None and p.perfil == "balanceado"


def test_los_alias_de_perfil():
    assert interpretar_pedido({"model": "auto:rapido"}).perfil == "rapido"
    assert interpretar_pedido({"model": "auto:potente"}).perfil == "potente"


def test_los_alias_de_capacidad_se_traducen_a_requisitos():
    p = interpretar_pedido({"model": "auto:tools"})
    assert p.requiere_tools is True and p.perfil == "balanceado"
    assert interpretar_pedido({"model": "auto:vision"}).requiere_vision is True


def test_un_modelo_real_se_conserva():
    p = interpretar_pedido({"model": "poolside/laguna-s-2.1:free"})
    assert p.modelo == "poolside/laguna-s-2.1:free"


def test_mandar_tools_exige_soporte_de_tools_aunque_no_se_pida():
    p = interpretar_pedido({"model": "auto", "tools": [{"type": "function"}]})
    assert p.requiere_tools is True


def test_las_extensiones_x_se_respetan():
    p = interpretar_pedido({"model": "auto", "x_requiere": ["tools", "vision"],
                            "x_min_contexto": 200000, "x_permitir_pago": False})
    assert p.requiere_tools and p.requiere_vision
    assert p.min_contexto == 200000
    assert p.permitir_pago is False
