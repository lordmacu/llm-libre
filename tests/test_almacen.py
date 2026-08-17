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


# --- Revision round 6 de Task 13, Parte 2. La clasificacion correcta de
#     codigos (Parte 1) no alcanza: `403` es GENUINAMENTE ambiguo -- cuenta
#     suspendida (evidencia de la ruta) o contenido moderado (evidencia del
#     PEDIDO) -- y el gateway no puede distinguirlos sin parsear el cuerpo
#     especifico de cada proveedor. Clasificarlo como evidencia de ruta
#     (correcto para el primer caso) lo deja vulnerable al segundo: 30
#     pedidos con contenido moderado de UN cliente bastan para tirar
#     confiabilidad (un PROMEDIO de las ultimas 50 observaciones) por el
#     piso para TODOS.
#
#     Redisenio: /health deja de usar confiabilidad. Pasa a ser "evidencia
#     de vida", no "ausencia de muerte" -- UN exito reciente prueba que la
#     ruta sirve; mil fallos de un mismo cliente no prueban que no puede.
#     `/v1/ranking` SIGUE usando confiabilidad exactamente como antes (no
#     se toca): una ruta mal puntuada solo pierde posicion y se
#     autocorrige, mientras que /health mal informado REINICIA EL
#     CONTENEDOR (Coolify) -- la asimetria es el punto. ---

def test_tiene_evidencia_de_vida_sin_ninguna_telemetria(almacen):
    # Ruta recien vista: no nacio muerta, todavia no tuvo su primera
    # oportunidad ni para bien ni para mal.
    almacen.upsert_rutas([_ruta()], momento=100.0)
    assert almacen.tiene_evidencia_de_vida("kilo/a:free", ahora=100.0) is True


def test_tiene_evidencia_de_vida_con_un_exito_reciente_pese_a_muchos_fallos(almacen):
    almacen.upsert_rutas([_ruta()], momento=100.0)
    almacen.registrar_evento("kilo/a:free", True, 50, 200, 100.0)
    for i in range(30):
        almacen.registrar_evento("kilo/a:free", False, 0, 403, 101.0 + i)
    assert almacen.tiene_evidencia_de_vida("kilo/a:free", ahora=200.0) is True


def test_tiene_evidencia_de_vida_con_una_sonda_de_salud_exitosa_reciente(almacen):
    almacen.upsert_rutas([_ruta()], momento=100.0)
    almacen.registrar_sonda("kilo/a:free", "salud", True, 100, 50, 200, 0, 0, 150.0)
    assert almacen.tiene_evidencia_de_vida("kilo/a:free", ahora=200.0) is True


def test_no_tiene_evidencia_de_vida_con_solo_fallos_reales(almacen):
    # El caso "genuinamente muerta": solo fallos que SI son evidencia de la
    # ruta (es_error_cliente=0, el default), sin exito en ningun lado.
    almacen.upsert_rutas([_ruta()], momento=100.0)
    for i in range(30):
        almacen.registrar_evento("kilo/a:free", False, 0, 500, 100.0 + i)
    assert almacen.tiene_evidencia_de_vida("kilo/a:free", ahora=200.0) is False


def test_tiene_evidencia_de_vida_si_los_fallos_son_todos_error_del_cliente(almacen):
    # Una ruta que SOLO recibio pedidos malformados (400/413/422, Parte 1)
    # todavia no tuvo su primera oportunidad de verdad -- se trata igual
    # que "sin telemetria", no como "muerta".
    almacen.upsert_rutas([_ruta()], momento=100.0)
    for i in range(30):
        almacen.registrar_evento("kilo/a:free", False, 0, 400, 100.0 + i,
                                 es_error_cliente=True)
    assert almacen.tiene_evidencia_de_vida("kilo/a:free", ahora=200.0) is True


def test_no_tiene_evidencia_de_vida_si_el_exito_quedo_fuera_de_la_ventana(almacen):
    from llm_libre.almacen import VENTANA_EVIDENCIA_VIDA_S
    almacen.upsert_rutas([_ruta()], momento=100.0)
    almacen.registrar_evento("kilo/a:free", True, 50, 200, 100.0)
    ahora = 100.0 + VENTANA_EVIDENCIA_VIDA_S + 1.0
    for i in range(30):
        almacen.registrar_evento("kilo/a:free", False, 0, 500, ahora - 10.0 + i * 0.1)
    assert almacen.tiene_evidencia_de_vida("kilo/a:free", ahora=ahora) is False


# --- Round 9, hallazgo del gate ("el camino indirecto de /health"): desde
#     que el trafico real puede disparar sondas BAJO DEMANDA (hasta 60/h por
#     ruta, ~300x mas seguido que el ciclo periodico de 5h), un cliente
#     controla CUANDO se muestrea una ruta -- mas muestras, mas chances de
#     agarrar por azar un problema transitorio del proveedor en UNA sola
#     sonda, y que /health lo trate como veredicto definitivo (sobrevive un
#     reinicio de contenedor). Decision: UNA sonda fallida sola YA NO
#     alcanza -- hacen falta DOS consecutivas, sin exito de por medio. ---

def test_una_sola_sonda_fallida_ya_no_alcanza_para_declarar_muerta(almacen):
    almacen.upsert_rutas([_ruta()], momento=100.0)
    almacen.registrar_sonda("kilo/a:free", "salud", False, 100, 0, 500, 0, 0, 150.0)
    assert almacen.tiene_evidencia_de_vida("kilo/a:free", ahora=200.0) is True


def test_dos_sondas_fallidas_consecutivas_si_declaran_muerta(almacen):
    almacen.upsert_rutas([_ruta()], momento=100.0)
    almacen.registrar_sonda("kilo/a:free", "salud", False, 100, 0, 500, 0, 0, 140.0)
    almacen.registrar_sonda("kilo/a:free", "salud", False, 100, 0, 500, 0, 0, 150.0)
    assert almacen.tiene_evidencia_de_vida("kilo/a:free", ahora=200.0) is False


def test_una_sonda_fallida_justo_despues_de_un_exito_no_alcanza(almacen):
    # Un solo fallo precedido por un exito NO es "dos consecutivos": la
    # señal mas vieja de las dos ultimas es un exito, no otro fallo.
    almacen.upsert_rutas([_ruta()], momento=100.0)
    almacen.registrar_evento("kilo/a:free", True, 50, 200, 140.0)
    almacen.registrar_sonda("kilo/a:free", "salud", False, 100, 0, 500, 0, 0, 150.0)
    assert almacen.tiene_evidencia_de_vida("kilo/a:free", ahora=200.0) is True


# --- Round 7, MEDIUM del gate: `tiene_evidencia_de_vida` solo miraba sondas
#     EXITOSAS -- una sonda con `ok=0` era invisible para la funcion.
#     Reproducido: las cinco rutas muertas, el ultimo exito real hace 20h
#     (adentro de la ventana de 24h), pero CUATRO sondas de salud fallidas
#     desde entonces (una cada 5h, el default de sondeo) -- la funcion
#     encontraba el exito viejo, se quedaba con el, y jamas miraba lo que
#     paso DESPUES. El argumento que ya justifica confiar en una sonda
#     EXITOSA (el gateway controla su propio payload, asi que no hay
#     ambiguedad posible de "esto es sobre el pedido") vale IGUAL para una
#     sonda FALLIDA: es evidencia inequivoca de que la ruta esta rota, no
#     solo de que esta viva. La ventana de 24h es defendible para una ruta
#     de PAGO que nunca se sondea; no lo es para una ruta gratis sondeada
#     cada 5h cuyos ultimos cuatro resultados son CONOCIDOS y se descartaban. ---

def test_una_sonda_fallida_mas_reciente_que_un_exito_viejo_declara_la_ruta_muerta(almacen):
    almacen.upsert_rutas([_ruta()], momento=0.0)
    ahora = 100_000.0
    veinte_horas = 20 * 3600.0
    cinco_horas = 5 * 3600.0
    almacen.registrar_evento("kilo/a:free", True, 50, 200, ahora - veinte_horas)
    # Cuatro sondas de salud FALLIDAS desde entonces, una cada 5h -- todas
    # mas recientes que el exito de arriba, la ULTIMA hace apenas 1h.
    for i in range(1, 5):
        almacen.registrar_sonda("kilo/a:free", "salud", False, 100, 0, 500, 0, 0,
                                ahora - veinte_horas + i * cinco_horas)
    assert almacen.tiene_evidencia_de_vida("kilo/a:free", ahora=ahora) is False


def test_un_exito_real_mas_reciente_que_una_sonda_fallida_declara_la_ruta_viva(almacen):
    # Simetrico al de arriba: si DESPUES de una sonda fallida hay un exito
    # real (un cliente de verdad recibio respuesta), esa es la senal mas
    # nueva y gana -- la ruta esta viva AHORA, sin importar el tropiezo
    # anterior de la sonda.
    almacen.upsert_rutas([_ruta()], momento=0.0)
    almacen.registrar_sonda("kilo/a:free", "salud", False, 100, 0, 500, 0, 0, 100.0)
    almacen.registrar_evento("kilo/a:free", True, 50, 200, 150.0)
    assert almacen.tiene_evidencia_de_vida("kilo/a:free", ahora=200.0) is True


def test_una_sonda_exitosa_mas_reciente_que_una_fallida_declara_la_ruta_viva(almacen):
    almacen.upsert_rutas([_ruta()], momento=0.0)
    almacen.registrar_sonda("kilo/a:free", "salud", False, 100, 0, 500, 0, 0, 100.0)
    almacen.registrar_sonda("kilo/a:free", "salud", True, 100, 50, 200, 0, 0, 150.0)
    assert almacen.tiene_evidencia_de_vida("kilo/a:free", ahora=200.0) is True
