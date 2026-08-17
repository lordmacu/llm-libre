import sqlite3

from llm_libre.modelos import (CALIDAD_NEUTRA, CONFIABILIDAD_NEUTRA, TTFT_NEUTRO_MS,
                               Capacidades, Metricas, Ruta)

ESQUEMA = """
CREATE TABLE IF NOT EXISTS rutas (
    clave TEXT PRIMARY KEY, proveedor TEXT NOT NULL, modelo_id TEXT NOT NULL,
    tier TEXT NOT NULL, tools INTEGER NOT NULL, vision INTEGER NOT NULL,
    contexto INTEGER NOT NULL, max_salida INTEGER NOT NULL,
    visto_por_ultima_vez REAL NOT NULL, activa INTEGER NOT NULL DEFAULT 1,
    prioridad INTEGER NOT NULL DEFAULT 100);

CREATE TABLE IF NOT EXISTS sondas (
    clave TEXT NOT NULL, tipo TEXT NOT NULL, momento REAL NOT NULL,
    ok INTEGER NOT NULL, latencia_ms INTEGER, ttft_ms INTEGER, codigo_http INTEGER,
    casos_pasados INTEGER, casos_totales INTEGER);
CREATE INDEX IF NOT EXISTS ix_sondas ON sondas(clave, tipo, momento DESC);

CREATE TABLE IF NOT EXISTS eventos (
    clave TEXT NOT NULL, momento REAL NOT NULL, ok INTEGER NOT NULL,
    ttft_ms INTEGER, codigo_http INTEGER, latencia_ms INTEGER,
    es_error_cliente INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS ix_eventos ON eventos(clave, momento DESC);

CREATE TABLE IF NOT EXISTS uso_pago (
    llave TEXT NOT NULL, dia TEXT NOT NULL, peticiones INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (llave, dia));
"""

VENTANA = 50  # cuantas observaciones recientes pesan en confiabilidad y latencia

# QUE SIGNIFICA CADA COLUMNA DE TIEMPO (decidido en el fix round 3, I5).
#
# `ttft_ms` significa UNA sola cosa: milisegundos hasta el primer token util
# que el llamador podria ver. Solo puede medirlo quien lee un stream, asi que
# solo el camino de streaming escribe ahi. Antes lo escribian los dos caminos:
# el no-streaming metia el round-trip COMPLETO (7-27 s en un modelo de
# razonamiento, porque hasta que no termina de generar no llega nada) y el de
# streaming el tiempo real al primer chunk (~200 ms). Mezclados en un mismo
# p50, el numero no significaba nada y el perfil `rapido` ordenaba por ruido.
#
# `latencia_ms` es el round-trip completo, que es lo que de verdad mide un
# camino no-streaming (y la sonda de salud). Se guarda -- el §5 del diseno ya
# lo pedia en `eventos` -- y se expone en /v1/ranking, pero NO alimenta el
# factor de latencia del puntaje, que esta calibrado en escala de ttft.
#
# Consecuencia asumida: en un despliegue que solo use el camino no-streaming,
# ningun ttft se mide y todas las rutas quedan con el neutro, o sea que la
# latencia deja de discriminar. Es preferible a discriminar por un numero
# inventado: se ve en /v1/ranking (ttft_p50_ms == el neutro para todas) y se
# arregla solo en cuanto haya trafico de streaming.


class Almacen:
    def __init__(self, ruta_db: str):
        self._con = sqlite3.connect(ruta_db, check_same_thread=False)
        self._con.execute("PRAGMA journal_mode=WAL")

    def crear_esquema(self) -> None:
        self._con.executescript(ESQUEMA)
        self._migrar()
        self._con.commit()

    def _migrar(self) -> None:
        """`CREATE TABLE IF NOT EXISTS` no agrega columnas a una tabla que ya
        existe: una base viva (la del volumen de /datos, que a proposito
        sobrevive a los redeploys) se quedaria sin `eventos.latencia_ms` --
        o, desde Task 13, sin `rutas.prioridad` ni `eventos.es_error_cliente`.
        Mismo patron para las tres: detectar la columna faltante y agregarla
        con un default que no rompa las filas que ya existen. Las filas
        VIEJAS de `eventos` (escritas antes de que existiera la distincion)
        migran a `es_error_cliente=0` -- siguen contando como fallo de la
        ruta, como contaban antes: no hay forma de reclasificar
        retroactivamente algo que nunca se distinguio al escribirlo."""
        columnas_eventos = {f[1] for f in self._con.execute("PRAGMA table_info(eventos)")}
        if "latencia_ms" not in columnas_eventos:
            self._con.execute("ALTER TABLE eventos ADD COLUMN latencia_ms INTEGER")
        if "es_error_cliente" not in columnas_eventos:
            self._con.execute(
                "ALTER TABLE eventos ADD COLUMN es_error_cliente INTEGER NOT NULL DEFAULT 0")
        columnas_rutas = {f[1] for f in self._con.execute("PRAGMA table_info(rutas)")}
        if "prioridad" not in columnas_rutas:
            self._con.execute(
                "ALTER TABLE rutas ADD COLUMN prioridad INTEGER NOT NULL DEFAULT 100")

    def upsert_rutas(self, rutas: list[Ruta], momento: float,
                     desactivar_faltantes: bool = True,
                     proveedor: str | None = None) -> None:
        for r in rutas:
            c = r.capacidades
            self._con.execute(
                """INSERT INTO rutas (clave, proveedor, modelo_id, tier, tools, vision,
                       contexto, max_salida, visto_por_ultima_vez, activa, prioridad)
                   VALUES (?,?,?,?,?,?,?,?,?,1,?)
                   ON CONFLICT(clave) DO UPDATE SET
                       tools=excluded.tools, vision=excluded.vision,
                       contexto=excluded.contexto, max_salida=excluded.max_salida,
                       visto_por_ultima_vez=excluded.visto_por_ultima_vez, activa=1,
                       prioridad=excluded.prioridad""",
                (r.clave, r.proveedor, r.modelo_id, r.tier, int(c.tools), int(c.vision),
                 c.contexto, c.max_salida, momento, r.prioridad))
        # Lo que no se vio en esta pasada se desactiva, no se borra: el historico
        # es lo que permite detectar un renombre de modelo. Se puede omitir este
        # paso cuando el llamador solo trae un subconjunto (p.ej. la sincronizacion
        # de un solo proveedor entre varios) y no quiere apagar las rutas de los demas.
        #
        # `proveedor`, si se pasa, ACOTA esa desactivacion a las rutas de ese
        # proveedor: sin esto, un UPDATE sin filtrar por proveedor apagaria
        # tambien las rutas de proveedores AJENOS a esta llamada (su
        # visto_por_ultima_vez siempre es mas vieja que `momento`, asi que
        # caerian igual). Es lo que permite que la sincronizacion de un
        # proveedor decida SUS bajas sin esperar a saber que paso con los
        # demas en la misma pasada. None (el default) preserva el
        # comportamiento historico: acota a nada, o sea a toda la tabla.
        if desactivar_faltantes:
            if proveedor is not None:
                self._con.execute(
                    "UPDATE rutas SET activa = 0 WHERE visto_por_ultima_vez < ? AND proveedor = ?",
                    (momento, proveedor))
            else:
                self._con.execute(
                    "UPDATE rutas SET activa = 0 WHERE visto_por_ultima_vez < ?", (momento,))
        self._con.commit()

    def rutas_activas(self) -> list[Ruta]:
        filas = self._con.execute(
            """SELECT proveedor, modelo_id, tier, tools, vision, contexto, max_salida,
                      prioridad
               FROM rutas WHERE activa = 1 ORDER BY clave""").fetchall()
        return [Ruta(p, m, t, Capacidades(bool(to), bool(vi), cx, ms), prioridad=pr)
                for p, m, t, to, vi, cx, ms, pr in filas]

    def registrar_sonda(self, clave: str, tipo: str, ok: bool, latencia_ms: int,
                        ttft_ms: int, codigo_http: int, casos_pasados: int,
                        casos_totales: int, momento: float) -> None:
        self._con.execute(
            "INSERT INTO sondas VALUES (?,?,?,?,?,?,?,?,?)",
            (clave, tipo, momento, int(ok), latencia_ms, ttft_ms, codigo_http,
             casos_pasados, casos_totales))
        self._con.commit()

    def registrar_evento(self, clave: str, ok: bool, ttft_ms: int,
                         codigo_http: int, momento: float,
                         latencia_ms: int | None = None,
                         es_error_cliente: bool = False) -> None:
        """`ttft_ms` solo lo escribe quien pudo medir un time-to-first-token de
        verdad (el camino de streaming); el resto pasa 0 y manda su round-trip
        en `latencia_ms`. Ver el comentario de arriba del archivo.

        `es_error_cliente` (revision de Task 13, ronda 4): un 4xx que no sea
        429 es evidencia sobre el PEDIDO, no sobre la ruta -- proxy.py ya
        no lo cuenta hacia el cooldown de fallos duros (ronda 3), pero
        SEGUIA escribiendose como evento fallido comun, y eso alimenta
        `_confiabilidad`, que `/health` usa para declarar una ruta muerta.
        Reproducido: 26 pedidos malformados seguidos de UNA llave bastaban
        para tirar la confiabilidad de TODAS las rutas por el piso, con
        /health en "caido" mientras una llave DISTINTA seguia recibiendo
        200 todo el tiempo -- peor que el 503 que la ronda 3 ya arreglo,
        porque en el volumen persistente de /datos un reinicio del proceso
        NO limpia el historial: sigue reportando "caido" contra la misma
        base. La fila se GUARDA igual (queda diagnosticable: un operador
        viendo `eventos` puede ver los 4xx) pero `_confiabilidad` la
        excluye de la ventana por completo -- ni cuenta como fallo, ni
        ocupa un lugar en las ultimas VENTANA observaciones."""
        self._con.execute(
            """INSERT INTO eventos (clave, momento, ok, ttft_ms, codigo_http, latencia_ms,
                   es_error_cliente)
               VALUES (?,?,?,?,?,?,?)""",
            (clave, momento, int(ok), ttft_ms, codigo_http, latencia_ms, int(es_error_cliente)))
        self._con.commit()

    def metricas(self) -> dict[str, Metricas]:
        salida: dict[str, Metricas] = {}
        for (clave,) in self._con.execute("SELECT clave FROM rutas WHERE activa = 1"):
            calidad, medida_en = self._calidad(clave)
            salida[clave] = Metricas(
                calidad=calidad,
                confiabilidad=self._confiabilidad(clave),
                ttft_p50_ms=self._ttft_p50(clave),
                en_cooldown_hasta=0.0,  # el cooldown vive en memoria del proxy
                calidad_medida_en=medida_en,
                ultima_sonda_en=self._ultima_sonda(clave),
                latencia_p50_ms=self._latencia_p50(clave),
            )
        return salida

    def _calidad(self, clave: str) -> tuple[float, float | None]:
        """(calidad, momento de la medicion). El momento es None si nunca se
        midio: ahi la calidad devuelta es el NEUTRO, un supuesto -- y quien la
        consuma tiene que poder distinguir un 0.6 medido de un 0.6 asumido."""
        fila = self._con.execute(
            """SELECT casos_pasados, casos_totales, momento FROM sondas
               WHERE clave = ? AND tipo = 'calidad' AND casos_totales > 0
               ORDER BY momento DESC LIMIT 1""", (clave,)).fetchone()
        if not fila:
            return CALIDAD_NEUTRA, None
        return fila[0] / fila[1], fila[2]

    def _ultima_sonda(self, clave: str) -> float | None:
        fila = self._con.execute(
            "SELECT MAX(momento) FROM sondas WHERE clave = ?", (clave,)).fetchone()
        return fila[0] if fila else None

    def _confiabilidad(self, clave: str) -> float:
        # `eventos.es_error_cliente = 0` excluye los 4xx del cliente (ver el
        # docstring de registrar_evento): no cuentan como fallo NI ocupan un
        # lugar en la ventana, como si ese pedido nunca le hubiera llegado a
        # esta ruta. `sondas` no tiene esta columna -- las sondas las genera
        # el propio gateway con un payload fijo, nunca son "culpa del
        # cliente" en este sentido -- asi que se incluyen todas, como antes.
        filas = self._con.execute(
            """SELECT ok FROM (
                   SELECT momento, ok FROM sondas WHERE clave = ?
                   UNION ALL
                   SELECT momento, ok FROM eventos
                   WHERE clave = ? AND es_error_cliente = 0
               ) ORDER BY momento DESC LIMIT ?""", (clave, clave, VENTANA)).fetchall()
        if not filas:
            return CONFIABILIDAD_NEUTRA
        return sum(f[0] for f in filas) / len(filas)

    def _ttft_p50(self, clave: str) -> float:
        """p50 de time-to-first-token. Solo entran observaciones que de verdad
        midieron un ttft (streaming): las demas escriben 0 y quedan fuera del
        `ttft_ms > 0`. Sin ninguna, se devuelve el neutro."""
        filas = self._con.execute(
            """SELECT ttft_ms FROM (
                   SELECT momento, ttft_ms FROM sondas WHERE clave = ? AND ok = 1
                   UNION ALL
                   SELECT momento, ttft_ms FROM eventos WHERE clave = ? AND ok = 1
               ) WHERE ttft_ms > 0 ORDER BY momento DESC LIMIT ?""",
            (clave, clave, VENTANA)).fetchall()
        if not filas:
            return TTFT_NEUTRO_MS
        valores = sorted(f[0] for f in filas)
        return float(valores[len(valores) // 2])

    def _latencia_p50(self, clave: str) -> float | None:
        """p50 del round-trip completo. No entra en el puntaje (el factor de
        latencia esta calibrado en escala de ttft); se expone para diagnostico,
        que es justo lo que faltaba cuando las dos mediciones compartian
        columna. None = nunca se observo."""
        filas = self._con.execute(
            """SELECT latencia_ms FROM (
                   SELECT momento, latencia_ms FROM sondas WHERE clave = ? AND ok = 1
                   UNION ALL
                   SELECT momento, latencia_ms FROM eventos WHERE clave = ? AND ok = 1
               ) WHERE latencia_ms > 0 ORDER BY momento DESC LIMIT ?""",
            (clave, clave, VENTANA)).fetchall()
        if not filas:
            return None
        valores = sorted(f[0] for f in filas)
        return float(valores[len(valores) // 2])

    def sumar_uso_pago(self, llave: str, dia: str) -> int:
        self._con.execute(
            """INSERT INTO uso_pago (llave, dia, peticiones) VALUES (?,?,1)
               ON CONFLICT(llave, dia) DO UPDATE SET peticiones = peticiones + 1""",
            (llave, dia))
        self._con.commit()
        return self.uso_pago(llave, dia)

    def uso_pago(self, llave: str, dia: str) -> int:
        fila = self._con.execute(
            "SELECT peticiones FROM uso_pago WHERE llave = ? AND dia = ?",
            (llave, dia)).fetchone()
        return fila[0] if fila else 0

    def podar(self, antes_de: float) -> None:
        self._con.execute("DELETE FROM sondas WHERE momento < ?", (antes_de,))
        self._con.execute("DELETE FROM eventos WHERE momento < ?", (antes_de,))
        self._con.commit()
