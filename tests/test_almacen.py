import pytest

from llm_libre.almacen import Almacen
from llm_libre.modelos import Capacidades, Ruta


def _ruta(modelo="a:free", proveedor="kilo", tools=True):
    return Ruta(proveedor, modelo, "gratis",
                Capacidades(tools=tools, vision=False, contexto=1000, max_salida=100))


@pytest.fixture
def almacen():
    a = Almacen(":memory:")
    a.crear_esquema()
    return a


def test_guarda_y_devuelve_rutas(almacen):
    almacen.upsert_rutas([_ruta()], momento=100.0)
    activas = almacen.rutas_activas()
    assert len(activas) == 1
    assert activas[0].clave == "kilo/a:free"
    assert activas[0].capacidades.tools is True


def test_una_ruta_que_desaparece_se_desactiva_pero_no_se_borra(almacen):
    almacen.upsert_rutas([_ruta("vieja:free"), _ruta("nueva:free")], momento=100.0)
    almacen.upsert_rutas([_ruta("nueva:free")], momento=200.0)
    activas = [r.modelo_id for r in almacen.rutas_activas()]
    assert activas == ["nueva:free"]
    # sigue en la tabla: el historico sirve para detectar renombres
    fila = almacen._con.execute(
        "SELECT activa FROM rutas WHERE modelo_id = 'vieja:free'").fetchone()
    assert fila[0] == 0


def test_una_ruta_que_vuelve_se_reactiva(almacen):
    almacen.upsert_rutas([_ruta("x:free")], momento=100.0)
    almacen.upsert_rutas([], momento=200.0)
    almacen.upsert_rutas([_ruta("x:free")], momento=300.0)
    assert len(almacen.rutas_activas()) == 1


def test_no_desactiva_nada_cuando_se_pide_conservar(almacen):
    almacen.upsert_rutas([_ruta("vieja:free"), _ruta("nueva:free")], momento=100.0)
    almacen.upsert_rutas([_ruta("nueva:free")], momento=200.0, desactivar_faltantes=False)
    activas = sorted(r.modelo_id for r in almacen.rutas_activas())
    assert activas == ["nueva:free", "vieja:free"]
    fila = almacen._con.execute(
        "SELECT visto_por_ultima_vez FROM rutas WHERE modelo_id = 'nueva:free'").fetchone()
    assert fila[0] == 200.0


def test_la_calidad_sale_de_la_ultima_sonda_de_calidad(almacen):
    almacen.upsert_rutas([_ruta()], momento=100.0)
    almacen.registrar_sonda("kilo/a:free", "calidad", True, 500, 200, 200, 2, 5, 100.0)
    almacen.registrar_sonda("kilo/a:free", "calidad", True, 500, 200, 200, 4, 5, 200.0)
    assert almacen.metricas()["kilo/a:free"].calidad == pytest.approx(0.8)


def test_la_confiabilidad_mezcla_sondas_y_eventos(almacen):
    almacen.upsert_rutas([_ruta()], momento=100.0)
    almacen.registrar_sonda("kilo/a:free", "salud", True, 100, 50, 200, 0, 0, 100.0)
    almacen.registrar_evento("kilo/a:free", False, 0, 500, 150.0)
    m = almacen.metricas()["kilo/a:free"]
    assert 0.0 < m.confiabilidad < 1.0


def test_una_ruta_sin_datos_recibe_metricas_neutras(almacen):
    almacen.upsert_rutas([_ruta()], momento=100.0)
    m = almacen.metricas()["kilo/a:free"]
    assert m.calidad == pytest.approx(0.6)
    assert m.en_cooldown_hasta == 0.0


def test_el_uso_de_pago_se_cuenta_por_llave_y_dia(almacen):
    assert almacen.uso_pago("k1", "2026-08-16") == 0
    assert almacen.sumar_uso_pago("k1", "2026-08-16") == 1
    assert almacen.sumar_uso_pago("k1", "2026-08-16") == 2
    assert almacen.uso_pago("k1", "2026-08-17") == 0
    assert almacen.uso_pago("k2", "2026-08-16") == 0
