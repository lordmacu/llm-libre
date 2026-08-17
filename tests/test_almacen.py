import sqlite3

import pytest

from llm_libre.almacen import Almacen
from llm_libre.modelos import Capacidades, Ruta


def _ruta(modelo="a:free", proveedor="kilo", tools=True, prioridad=100):
    return Ruta(proveedor, modelo, "gratis",
                Capacidades(tools=tools, vision=False, contexto=1000, max_salida=100),
                prioridad=prioridad)


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


# --- Re-revision (round 4) de Task 13: un 4xx del cliente ya no castiga la
#     ruta (round 3), pero SEGUIA escribiendose como evento fallido, y eso
#     alimenta confiabilidad -- que /health usa para declarar una ruta
#     muerta. Reproducido: 26 pedidos malformados seguidos de UNA llave
#     bastan para tirar la confiabilidad de TODAS las rutas por el piso, con
#     /health en "caido" mientras una llave DISTINTA sigue recibiendo 200.
#     `registrar_evento` gana `es_error_cliente`, y _confiabilidad excluye
#     esas filas de eventos POR COMPLETO -- ni suman como fallo, ni cuentan
#     para la ventana -- para que un 4xx sea evidencia sobre el PEDIDO, no
#     sobre la ruta. Se mantienen escritas (no se descartan) para que sigan
#     siendo diagnosticables. ---

def test_confiabilidad_ignora_eventos_marcados_como_error_del_cliente(almacen):
    almacen.upsert_rutas([_ruta()], momento=100.0)
    for i in range(30):
        almacen.registrar_evento("kilo/a:free", False, 0, 400, 100.0 + i,
                                 es_error_cliente=True)
    m = almacen.metricas()["kilo/a:free"]
    # Sin ninguna otra observacion, la ventana queda VACIA (no las 30 filas
    # contando como fallo): confiabilidad cae al neutro, no a 0.
    assert m.confiabilidad == pytest.approx(0.8)   # CONFIABILIDAD_NEUTRA


def test_confiabilidad_sigue_cayendo_con_fallos_que_no_son_del_cliente(almacen):
    # Regresion directa: un 500 (es_error_cliente=False, el default) tiene
    # que seguir contando como antes.
    almacen.upsert_rutas([_ruta()], momento=100.0)
    for i in range(30):
        almacen.registrar_evento("kilo/a:free", False, 0, 500, 100.0 + i)
    m = almacen.metricas()["kilo/a:free"]
    assert m.confiabilidad == pytest.approx(0.0)


def test_confiabilidad_mezcla_error_del_cliente_e_ignora_solo_esos(almacen):
    almacen.upsert_rutas([_ruta()], momento=100.0)
    almacen.registrar_evento("kilo/a:free", True, 50, 200, 100.0)
    for i in range(10):
        almacen.registrar_evento("kilo/a:free", False, 0, 400, 101.0 + i,
                                 es_error_cliente=True)
    m = almacen.metricas()["kilo/a:free"]
    # El unico evento que "cuenta" es el exito: los 10 de error del cliente
    # quedan completamente afuera de la ventana.
    assert m.confiabilidad == pytest.approx(1.0)


_ESQUEMA_VIEJO_SIN_ES_ERROR_CLIENTE = """
CREATE TABLE rutas (
    clave TEXT PRIMARY KEY, proveedor TEXT NOT NULL, modelo_id TEXT NOT NULL,
    tier TEXT NOT NULL, tools INTEGER NOT NULL, vision INTEGER NOT NULL,
    contexto INTEGER NOT NULL, max_salida INTEGER NOT NULL,
    visto_por_ultima_vez REAL NOT NULL, activa INTEGER NOT NULL DEFAULT 1,
    prioridad INTEGER NOT NULL DEFAULT 100);
CREATE TABLE eventos (
    clave TEXT NOT NULL, momento REAL NOT NULL, ok INTEGER NOT NULL,
    ttft_ms INTEGER, codigo_http INTEGER, latencia_ms INTEGER);
"""


def test_migra_una_base_vieja_sin_es_error_cliente_con_filas(tmp_path):
    ruta_db = str(tmp_path / "vieja_sin_flag.sqlite3")
    con = sqlite3.connect(ruta_db)
    con.executescript(_ESQUEMA_VIEJO_SIN_ES_ERROR_CLIENTE)
    con.execute(
        """INSERT INTO rutas (clave, proveedor, modelo_id, tier, tools, vision,
               contexto, max_salida, visto_por_ultima_vez, activa, prioridad)
           VALUES ('kilo/vieja:free','kilo','vieja:free','gratis',1,0,1000,100,50.0,1,100)""")
    con.execute(
        """INSERT INTO eventos (clave, momento, ok, ttft_ms, codigo_http, latencia_ms)
           VALUES ('kilo/vieja:free', 60.0, 0, 0, 400, 20)""")
    con.commit()
    con.close()

    almacen = Almacen(ruta_db)
    almacen.crear_esquema()   # no debe reventar (ALTER TABLE, no CREATE)

    # La fila vieja, escrita ANTES de que existiera es_error_cliente, migra
    # a 0 (comportamiento historico: SI cuenta como fallo) -- no se puede
    # reclasificar retroactivamente un evento que no distinguia la causa.
    fila = almacen._con.execute(
        "SELECT es_error_cliente FROM eventos WHERE clave = 'kilo/vieja:free'").fetchone()
    assert fila[0] == 0

    # Y la base migrada sigue siendo escribible con el flag nuevo.
    almacen.registrar_evento("kilo/vieja:free", False, 0, 400, 70.0, es_error_cliente=True)
    filas = almacen._con.execute(
        "SELECT es_error_cliente FROM eventos WHERE clave = 'kilo/vieja:free' "
        "ORDER BY momento").fetchall()
    assert [f[0] for f in filas] == [0, 1]


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


# --- Task 13: `prioridad` persiste y una base vieja migra sin perder datos. ---

def test_upsert_rutas_persiste_la_prioridad(almacen):
    almacen.upsert_rutas([_ruta("chatgpt:free", proveedor="chatgpt", prioridad=0)],
                         momento=100.0)
    activas = almacen.rutas_activas()
    assert len(activas) == 1
    assert activas[0].prioridad == 0


def test_upsert_rutas_sin_prioridad_declarada_persiste_el_default_cien(almacen):
    almacen.upsert_rutas([_ruta()], momento=100.0)
    assert almacen.rutas_activas()[0].prioridad == 100


def test_resincronizar_actualiza_la_prioridad_de_una_ruta_existente(almacen):
    # Un cambio de `prioridad` en el YAML (p.ej. subir a un proveedor de
    # lugar) tiene que propagarse en la proxima sincronizacion, no quedar
    # pegado al valor con el que la ruta se vio por primera vez.
    almacen.upsert_rutas([_ruta("a:free", prioridad=1)], momento=100.0)
    almacen.upsert_rutas([_ruta("a:free", prioridad=0)], momento=200.0)
    assert almacen.rutas_activas()[0].prioridad == 0


# `_migrar()` ya tiene un caso (eventos.latencia_ms, ver el comentario de
# cabecera de almacen.py) que agrega una columna a una tabla que YA existe con
# `ALTER TABLE ... ADD COLUMN` -- porque `CREATE TABLE IF NOT EXISTS` no toca
# una tabla existente. Este test reproduce el mismo riesgo para `rutas.
# prioridad`: la tabla `rutas` de produccion (el volumen de /datos) existe
# desde ANTES de esta feature y ya tiene filas. Si la migracion no fuera
# idempotente y compatible con datos existentes, un redeploy contra esa base
# reventaria al arrancar (o silenciosamente perderia la columna).
_ESQUEMA_VIEJO_SIN_PRIORIDAD = """
CREATE TABLE rutas (
    clave TEXT PRIMARY KEY, proveedor TEXT NOT NULL, modelo_id TEXT NOT NULL,
    tier TEXT NOT NULL, tools INTEGER NOT NULL, vision INTEGER NOT NULL,
    contexto INTEGER NOT NULL, max_salida INTEGER NOT NULL,
    visto_por_ultima_vez REAL NOT NULL, activa INTEGER NOT NULL DEFAULT 1);
"""


def test_migra_una_base_vieja_con_filas_sin_perder_datos(tmp_path):
    ruta_db = str(tmp_path / "vieja.sqlite3")
    # Simula la base de produccion: esquema PRE-prioridad, con una fila real
    # adentro (visto_por_ultima_vez, activa -- todo lo que la version vieja
    # del codigo ya escribia).
    con = sqlite3.connect(ruta_db)
    con.executescript(_ESQUEMA_VIEJO_SIN_PRIORIDAD)
    con.execute(
        """INSERT INTO rutas (clave, proveedor, modelo_id, tier, tools, vision,
               contexto, max_salida, visto_por_ultima_vez, activa)
           VALUES ('kilo/vieja:free','kilo','vieja:free','gratis',1,0,1000,100,50.0,1)""")
    con.commit()
    con.close()

    # Abrir con el codigo NUEVO no debe reventar (ALTER TABLE, no CREATE).
    almacen = Almacen(ruta_db)
    almacen.crear_esquema()

    activas = almacen.rutas_activas()
    assert len(activas) == 1
    assert activas[0].clave == "kilo/vieja:free"
    # La fila preexistente, sin ninguna prioridad en el momento en que se
    # escribio, migra al default (100), no a NULL ni a un valor inventado.
    assert activas[0].prioridad == 100

    # Y la base migrada sigue siendo escribible: una sincronizacion nueva
    # puede declarar prioridad para esa misma ruta o para una nueva.
    almacen.upsert_rutas([_ruta("vieja:free", proveedor="kilo", prioridad=0)], momento=200.0)
    almacen.upsert_rutas([_ruta("nueva:free", proveedor="chatgpt", prioridad=0)],
                         momento=200.0, desactivar_faltantes=False)
    activas = {r.clave: r.prioridad for r in almacen.rutas_activas()}
    assert activas == {"kilo/vieja:free": 0, "chatgpt/nueva:free": 0}


def test_migrar_una_base_vieja_es_idempotente(tmp_path):
    # Abrir la base migrada una SEGUNDA vez (el redeploy siguiente) no debe
    # reventar con "duplicate column name".
    ruta_db = str(tmp_path / "vieja2.sqlite3")
    con = sqlite3.connect(ruta_db)
    con.executescript(_ESQUEMA_VIEJO_SIN_PRIORIDAD)
    con.close()

    Almacen(ruta_db).crear_esquema()
    otra_vez = Almacen(ruta_db)
    otra_vez.crear_esquema()   # no debe reventar
    assert otra_vez.rutas_activas() == []
