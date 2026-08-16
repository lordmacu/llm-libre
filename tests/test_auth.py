from llm_libre.auth import LimitadorPorLlave


def test_permite_hasta_el_tope():
    lim = LimitadorPorLlave(por_minuto=3)
    assert [lim.permitir("k", 0.0) for _ in range(4)] == [True, True, True, False]


def test_la_ventana_se_libera_al_minuto():
    lim = LimitadorPorLlave(por_minuto=1)
    assert lim.permitir("k", 0.0) is True
    assert lim.permitir("k", 30.0) is False
    assert lim.permitir("k", 61.0) is True


def test_cada_llave_tiene_su_propio_cupo():
    lim = LimitadorPorLlave(por_minuto=1)
    assert lim.permitir("a", 0.0) is True
    assert lim.permitir("b", 0.0) is True
