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


def test_el_scope_de_proveedor_acota_la_baja_a_ese_proveedor(almacen):
    # kilo y otro tienen cada uno una ruta vieja. Al re-sincronizar SOLO kilo
    # (con proveedor="kilo"), su ruta vieja se apaga pero la de "otro" -- mas
    # vieja todavia, y ni siquiera mencionada en esta llamada -- debe
    # sobrevivir: sin el scope, el UPDATE sin filtrar por proveedor la
    # habria apagado igual, porque su visto_por_ultima_vez tambien es
    # anterior al `momento` nuevo.
    almacen.upsert_rutas([_ruta("vieja:free", proveedor="kilo")], momento=50.0)
    almacen.upsert_rutas([_ruta("vieja:free", proveedor="otro")], momento=50.0)
    almacen.upsert_rutas([_ruta("nueva:free", proveedor="kilo")], momento=200.0, proveedor="kilo")
    activas = {r.clave for r in almacen.rutas_activas()}
    assert activas == {"kilo/nueva:free", "otro/vieja:free"}


def test_el_scope_de_proveedor_no_cambia_el_comportamiento_por_defecto(almacen):
    # proveedor=None (el default) preserva el comportamiento historico: sin
    # scope, acota a toda la tabla -- exactamente lo que ya cubre
    # test_una_ruta_que_desaparece_se_desactiva_pero_no_se_borra. Este test
    # solo confirma que pasar proveedor=None explicito da lo mismo.
    almacen.upsert_rutas([_ruta("vieja:free", proveedor="kilo"),
                          _ruta("otra:free", proveedor="otro")], momento=100.0)
    almacen.upsert_rutas([_ruta("otra:free", proveedor="otro")], momento=200.0, proveedor=None)
    activas = {r.clave for r in almacen.rutas_activas()}
    assert activas == {"otro/otra:free"}   # kilo/vieja:free tambien se apaga, como antes


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


# --- Fix round 3, B2b/I3: distinguir "calidad medida 0.6" de "nunca medida". ---

def test_una_ruta_nunca_sondeada_no_declara_fecha_de_calidad(almacen):
    almacen.upsert_rutas([_ruta()], momento=100.0)
    m = almacen.metricas()["kilo/a:free"]
    assert m.calidad_medida_en is None
    assert m.ultima_sonda_en is None
    assert m.calidad == pytest.approx(0.6)   # el neutro sigue alimentando el puntaje


def test_una_ruta_sondeada_declara_el_momento_de_su_ultima_sonda_de_calidad(almacen):
    almacen.upsert_rutas([_ruta()], momento=100.0)
    almacen.registrar_sonda("kilo/a:free", "calidad", True, 0, 0, 200, 2, 5, 300.0)
    almacen.registrar_sonda("kilo/a:free", "calidad", True, 0, 0, 200, 4, 5, 900.0)
    m = almacen.metricas()["kilo/a:free"]
    assert m.calidad_medida_en == 900.0
    assert m.calidad == pytest.approx(0.8)


def test_la_ultima_sonda_cuenta_tambien_las_de_salud(almacen):
    almacen.upsert_rutas([_ruta()], momento=100.0)
    almacen.registrar_sonda("kilo/a:free", "calidad", True, 0, 0, 200, 4, 5, 300.0)
    almacen.registrar_sonda("kilo/a:free", "salud", True, 120, 0, 200, 0, 0, 800.0)
    m = almacen.metricas()["kilo/a:free"]
    assert m.ultima_sonda_en == 800.0        # la mas reciente de cualquier tipo
    assert m.calidad_medida_en == 300.0      # pero la de CALIDAD sigue siendo la suya


# --- Fix round 3, I5: `ttft_ms` mezclaba dos mediciones incompatibles en una
#     sola columna. El camino no-streaming guardaba el round-trip COMPLETO
#     (7-27 s en un modelo de razonamiento) y el de streaming el tiempo real
#     hasta el primer chunk (~200 ms). Mezclados en un mismo p50, el perfil
#     `rapido` ordenaba por un numero que no significa nada. ---

def test_el_ttft_p50_solo_cuenta_mediciones_de_ttft_de_verdad(almacen):
    almacen.upsert_rutas([_ruta()], momento=100.0)
    # Streaming: ttft real.
    almacen.registrar_evento("kilo/a:free", True, 200, 200, 150.0)
    # No streaming: no hay ttft que medir, va el round-trip a latencia_ms.
    almacen.registrar_evento("kilo/a:free", True, 0, 200, 160.0, latencia_ms=21000)
    almacen.registrar_evento("kilo/a:free", True, 0, 200, 170.0, latencia_ms=19000)
    assert almacen.metricas()["kilo/a:free"].ttft_p50_ms == 200.0


def test_la_latencia_total_se_guarda_aunque_no_alimente_el_ttft(almacen):
    almacen.upsert_rutas([_ruta()], momento=100.0)
    almacen.registrar_evento("kilo/a:free", True, 0, 200, 160.0, latencia_ms=21000)
    almacen.registrar_evento("kilo/a:free", True, 0, 200, 170.0, latencia_ms=19000)
    m = almacen.metricas()["kilo/a:free"]
    assert m.latencia_p50_ms == 21000.0        # p50 de las dos observaciones
    assert m.ttft_p50_ms == 1500.0             # el neutro: ttft nunca se midio


def test_sin_ninguna_observacion_de_ttft_se_usa_el_neutro(almacen):
    almacen.upsert_rutas([_ruta()], momento=100.0)
    m = almacen.metricas()["kilo/a:free"]
    assert m.ttft_p50_ms == 1500.0
    assert m.latencia_p50_ms is None
