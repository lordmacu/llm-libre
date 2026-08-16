import sqlite3

from llm_libre.modelos import (CALIDAD_NEUTRA, CONFIABILIDAD_NEUTRA, TTFT_NEUTRO_MS,
                               Capacidades, Metricas, Ruta)

ESQUEMA = """
CREATE TABLE IF NOT EXISTS rutas (
    clave TEXT PRIMARY KEY, proveedor TEXT NOT NULL, modelo_id TEXT NOT NULL,
    tier TEXT NOT NULL, tools INTEGER NOT NULL, vision INTEGER NOT NULL,
    contexto INTEGER NOT NULL, max_salida INTEGER NOT NULL,
    visto_por_ultima_vez REAL NOT NULL, activa INTEGER NOT NULL DEFAULT 1);

CREATE TABLE IF NOT EXISTS sondas (
    clave TEXT NOT NULL, tipo TEXT NOT NULL, momento REAL NOT NULL,
    ok INTEGER NOT NULL, latencia_ms INTEGER, ttft_ms INTEGER, codigo_http INTEGER,
    casos_pasados INTEGER, casos_totales INTEGER);
CREATE INDEX IF NOT EXISTS ix_sondas ON sondas(clave, tipo, momento DESC);

CREATE TABLE IF NOT EXISTS eventos (
    clave TEXT NOT NULL, momento REAL NOT NULL, ok INTEGER NOT NULL,
    ttft_ms INTEGER, codigo_http INTEGER);
CREATE INDEX IF NOT EXISTS ix_eventos ON eventos(clave, momento DESC);

CREATE TABLE IF NOT EXISTS uso_pago (
    llave TEXT NOT NULL, dia TEXT NOT NULL, peticiones INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (llave, dia));
"""

VENTANA = 50  # cuantas observaciones recientes pesan en confiabilidad y latencia


class Almacen:
    def __init__(self, ruta_db: str):
        self._con = sqlite3.connect(ruta_db, check_same_thread=False)
        self._con.execute("PRAGMA journal_mode=WAL")

    def crear_esquema(self) -> None:
        self._con.executescript(ESQUEMA)
        self._con.commit()

    def upsert_rutas(self, rutas: list[Ruta], momento: float,
                     desactivar_faltantes: bool = True,
                     proveedor: str | None = None) -> None:
        for r in rutas:
            c = r.capacidades
            self._con.execute(
                """INSERT INTO rutas (clave, proveedor, modelo_id, tier, tools, vision,
                       contexto, max_salida, visto_por_ultima_vez, activa)
                   VALUES (?,?,?,?,?,?,?,?,?,1)
                   ON CONFLICT(clave) DO UPDATE SET
                       tools=excluded.tools, vision=excluded.vision,
                       contexto=excluded.contexto, max_salida=excluded.max_salida,
                       visto_por_ultima_vez=excluded.visto_por_ultima_vez, activa=1""",
                (r.clave, r.proveedor, r.modelo_id, r.tier, int(c.tools), int(c.vision),
                 c.contexto, c.max_salida, momento))
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
            """SELECT proveedor, modelo_id, tier, tools, vision, contexto, max_salida
               FROM rutas WHERE activa = 1 ORDER BY clave""").fetchall()
        return [Ruta(p, m, t, Capacidades(bool(to), bool(vi), cx, ms))
                for p, m, t, to, vi, cx, ms in filas]

    def registrar_sonda(self, clave: str, tipo: str, ok: bool, latencia_ms: int,
                        ttft_ms: int, codigo_http: int, casos_pasados: int,
                        casos_totales: int, momento: float) -> None:
        self._con.execute(
            "INSERT INTO sondas VALUES (?,?,?,?,?,?,?,?,?)",
            (clave, tipo, momento, int(ok), latencia_ms, ttft_ms, codigo_http,
             casos_pasados, casos_totales))
        self._con.commit()

    def registrar_evento(self, clave: str, ok: bool, ttft_ms: int,
                         codigo_http: int, momento: float) -> None:
        self._con.execute("INSERT INTO eventos VALUES (?,?,?,?,?)",
                          (clave, momento, int(ok), ttft_ms, codigo_http))
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
        filas = self._con.execute(
            """SELECT ok FROM (
                   SELECT momento, ok FROM sondas WHERE clave = ?
                   UNION ALL SELECT momento, ok FROM eventos WHERE clave = ?
               ) ORDER BY momento DESC LIMIT ?""", (clave, clave, VENTANA)).fetchall()
        if not filas:
            return CONFIABILIDAD_NEUTRA
        return sum(f[0] for f in filas) / len(filas)

    def _ttft_p50(self, clave: str) -> float:
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
