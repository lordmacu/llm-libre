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

# Cuanto hacia atras se busca EVIDENCIA POSITIVA de que una ruta sirve, para
# /health (Task 13, revision round 6, Parte 2). Mas grande que el intervalo
# default de sondeo de salud (SONDEO_SALUD_HORAS=5h) con margen para que UNA
# pasada de sondeo que falle o se demore no tire la ruta a "sin evidencia" --
# y generoso para una ruta de PAGO, que nunca se sondea (§8 del diseno) y
# solo puede probar que esta viva con trafico real, que puede ser esporadico
# (es el escalon de fallback, no el trafico principal). 24h = ~5 ciclos de
# sondeo default, y sigue acotado por `podar()` (30 dias de retencion) del
# lado de arriba.
VENTANA_EVIDENCIA_VIDA_S = 24 * 3600.0

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

    def desactivar_proveedores_no_registrados(self, proveedores_conocidos: set[str]) -> int:
        """Apaga (activa=0, NUNCA borra -- el historico sirve para detectar
        renombres de modelo, ver upsert_rutas) toda ruta cuyo `proveedor` ya
        no esta en `proveedores_conocidos`.

        Existe porque `upsert_rutas` (y por lo tanto la baja de rutas que ya
        no aparecen) esta ACOTADA al proveedor de cada llamada
        (`sincronizar_catalogo` la invoca una vez por proveedor, con su
        propio `proveedor=p.id`): un proveedor que se SACA por completo del
        registro (`proveedores.yaml`) nunca vuelve a sincronizarse, asi que
        sin este barrido aparte sus rutas quedarian `activa=1` para
        siempre -- visibles en `GET /v1/models`, en `GET /v1/ranking`, y
        elegibles como candidatas de ruteo que fallarian siempre (401 de
        una llave que ya no existe, por ejemplo), sin que nada las apague.

        Se llama una vez por ciclo de sondeo, desde `sondeo.sincronizar_catalogo`,
        con el conjunto de ids que el PROCESO tiene cargado de
        `proveedores.yaml` en ese momento -- no una lista fija dentro de esta
        funcion. Ojo con la precision de esa frase (revision de gate): esto
        NO significa que un edit al YAML se refleje solo, sin reiniciar --
        `proveedores.cargar()` se llama UNA sola vez, al arrancar el proceso
        (`principal.crear_estado`), asi que ese conjunto queda fijo en
        memoria hasta el proximo reinicio. Lo que SI es cierto: dado ese
        conjunto (fijo hasta reiniciar), el barrido corre en CADA ciclo de
        sondeo, asi que sacar (o volver a agregar) un proveedor se refleja
        sin tocar codigo -- solo hace falta reiniciar el proceso una vez para
        que cargue la lista nueva, despues de lo cual el barrido ya no
        necesita ninguna intervencion manual adicional. Devuelve cuantas
        filas se apagaron, para que el llamador pueda dejar un
        `log.warning` (un proveedor que desaparece del registro es una
        decision del operador, pero vale la pena que quede visible en los
        logs cuando de verdad tiene efecto).

        `proveedores_conocidos` vacio es un caso especial, a proposito
        (LOW de una revision de gate): un `proveedores.yaml` sintacticamente
        valido pero con `proveedores: []` (un archivo truncado o mal editado
        a mano, mas probable que una decision real de "sin proveedores") NO
        dispara el barrido -- se trata como "todavia no se sabe nada", no
        como "todos son huerfanos", porque lo segundo apagaria el catalogo
        ENTERO en el primer ciclo de sondeo despues de un typo de config,
        sin ningun otro sintoma previo que avisara. Alcanzable solo con un
        edit deliberado del YAML, y totalmente recuperable (las rutas se
        reactivan solas en el proximo ciclo si el registro vuelve a traerlas)
        -- pero el guard es gratis."""
        if not proveedores_conocidos:
            return 0
        filas = self._con.execute(
            "SELECT DISTINCT proveedor FROM rutas WHERE activa = 1").fetchall()
        huerfanos = [p for (p,) in filas if p not in proveedores_conocidos]
        if not huerfanos:
            return 0
        marcador = ",".join("?" * len(huerfanos))
        cur = self._con.execute(
            f"UPDATE rutas SET activa = 0 WHERE activa = 1 AND proveedor IN ({marcador})",
            huerfanos)
        self._con.commit()
        return cur.rowcount

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

    def tiene_evidencia_de_vida(self, clave: str, ahora: float) -> bool:
        """Para `/health` (Task 13, revision round 6, Parte 2; corregido en
        round 7) -- "evidencia de vida", no "ausencia de muerte".
        `confiabilidad` (arriba) es un PROMEDIO de las ultimas `VENTANA`
        observaciones, y un promedio puede arrastrarse a 0 por cualquier
        patron de trafico repetido de UN cliente. La clasificacion de
        codigos (Parte 1) no alcanza para cerrar esto del todo: `403` es
        GENUINAMENTE ambiguo -- cuenta suspendida (evidencia de la ruta) o
        contenido moderado (evidencia del PEDIDO) -- y el gateway no puede
        distinguirlos sin parsear el cuerpo especifico de cada proveedor,
        asi que clasificarlo como evidencia de ruta (correcto para el
        primer caso) lo deja vulnerable al segundo: 30 pedidos con
        contenido moderado de un cliente bastan para tirar el promedio por
        el piso para TODOS.

        La pregunta correcta no es "cuantos fallos tuvo" sino "cual es la
        SENAL MAS RECIENTE que se tiene sobre esta ruta": dentro de
        `VENTANA_EVIDENCIA_VIDA_S`, se junta el ultimo EXITO real
        (`eventos.ok=1`) con la ultima SONDA DE SALUD (`sondas.tipo='salud'`,
        cualquier resultado) y gana la que tenga el `momento` mas nuevo.

          - Si la senal mas nueva es un exito real, o una sonda EXITOSA:
            viva.
          - Si la senal mas nueva es una sonda FALLIDA: muerta. La sonda es
            la señal mas confiable que existe, porque el GATEWAY controla
            su propio payload -- un resultado contra una sonda NUNCA es
            ambiguo, en ningun sentido: si un 4xx contra una sonda es
            SIEMPRE evidencia de la ruta (round 6), una sonda que directamente
            fallo lo es igual, y round 7 corrige el bug de mirar solo la
            mitad de ese argumento (`ok=1`) e ignorar la otra (`ok=0`).
            Reproducido: exito real hace 20h (adentro de una ventana de
            24h) pero CUATRO sondas fallidas desde entonces (una cada 5h,
            el default de sondeo) -- la version anterior encontraba el
            exito viejo, se quedaba con el, y jamas miraba que paso
            DESPUES. La ventana de 24h es defendible para una ruta de PAGO
            que nunca se sondea; no lo es para una ruta gratis sondeada
            cada 5h cuyos ultimos resultados son CONOCIDOS y se
            descartaban.
          - Si no hay ninguna señal (ni exito ni sonda) dentro de la
            ventana: cae al chequeo de "sin telemetria real en absoluto"
            (mas abajo) -- una ruta recien vista no nacio muerta, todavia
            no tuvo su primera oportunidad.

        Los eventos FALLIDOS (`eventos.ok=0`) nunca contribuyen evidencia
        de muerte de forma directa aca, solo indirectamente via el chequeo
        de "sin telemetria": siguen siendo ambiguos (un 403 puede ser
        moderacion de contenido de un cliente puntual) de una forma en que
        una sonda -- pedido fijo, controlado por el propio gateway -- nunca
        lo es. Los eventos con `es_error_cliente=1` (Parte 1) tampoco
        cuentan como telemetria "real" para el chequeo de respaldo: una
        ruta que SOLO recibio pedidos malformados (400/413/422) todavia no
        tuvo su primera oportunidad de verdad, igual que una sin ningun
        trafico.

        `/v1/ranking` NO usa esta senal -- sigue con `confiabilidad`
        exactamente como antes (y tambien la usa `router.ordenar`, que
        ordena la cadena de intentos real -- no es SOLO el endpoint de
        diagnostico): una ruta que puntua mal solo pierde posicion y se
        autocorrige; una ruta que `/health` declara muerta REINICIA EL
        CONTENEDOR (Coolify usa `/health` como health check). La asimetria
        es el punto: el ranking puede permitirse ser sensible, la salud
        no.

        Round 9: UNA sonda fallida sola YA NO ALCANZA -- hacen falta DOS
        señales consecutivas fallidas (sin exito de por medio) antes de
        tratarlo como evidencia de muerte. Antes de round 9, "la señal mas
        reciente decide" bastaba porque el UNICO disparador de una sonda
        era el ciclo periodico (cada 5h, ritmo fijo, nunca influido por un
        cliente). Desde que el trafico real puede disparar sondas BAJO
        DEMANDA (hasta 60/h por ruta, ~300x mas seguido que el ciclo de 5h),
        un cliente controla CUANDO se muestrea una ruta -- y mas muestras
        significa mas chances de agarrar, por puro azar, un problema
        transitorio del proveedor (un blip de red, un timeout aislado) justo
        en UNA sonda, y que `/health` lo trate como un veredicto definitivo
        que sobrevive un reinicio de contenedor. Exigir DOS fallos
        consecutivos dejaria que un problema genuinamente transitorio se
        resuelva solo (la proxima sonda o el proximo pedido real, casi
        siempre exitosos) antes de que /health lo de por muerto, sin perder
        deteccion rapida de un apagon real (una ruta de verdad rota sigue
        fallando la SEGUNDA sonda igual que la primera). Un EXITO real
        sigue alcanzando con uno solo -- la asimetria "un exito prueba vida,
        un fallo aislado no prueba muerte" es la misma de siempre, solo que
        ahora tambien aplica DENTRO de la comparacion entre sondas."""
        corte = ahora - VENTANA_EVIDENCIA_VIDA_S
        senales = self._con.execute(
            """SELECT ok FROM (
                   SELECT momento, 1 AS ok FROM eventos
                       WHERE clave = ? AND ok = 1 AND momento >= ?
                   UNION ALL
                   SELECT momento, ok FROM sondas
                       WHERE clave = ? AND tipo = 'salud' AND momento >= ?
               )
               ORDER BY momento DESC
               LIMIT 2""",
            (clave, corte, clave, corte)).fetchall()
        if senales:
            if senales[0][0]:
                return True
            # La señal mas reciente es un fallo: hace falta que la ANTERIOR
            # tambien lo sea (dos consecutivos, sin exito de por medio) --
            # un solo fallo, o un fallo justo despues de un exito, todavia
            # no alcanza.
            return not (len(senales) >= 2 and not senales[1][0])
        # Round 10, MEDIUM del gate: este chequeo de respaldo (nada dentro
        # de la ventana, ni exito ni sonda) miraba CUALQUIER evento real
        # -- exito O FALLO -- para decidir "hay historia, declarar
        # muerta". Un fallo real, sin ninguna sonda que lo confirme, NUNCA
        # deberia ser autoritativo: es el mismo principio que motiva
        # sospecha+sonda desde round 8 (el trafico de un cliente solo
        # puede pedirle al gateway que vaya a mirar, nunca excluir el
        # mismo). Medido: UN pedido real fallido bastaba para tirar
        # /health a "caido" sin que NINGUNA sonda hubiera corrido --
        # menos de UMBRAL_SOSPECHA, asi que ninguna sonda bajo demanda
        # llega a dispararse para rescatarla. Contradice directamente el
        # principio del modulo: "mil fallos de un mismo cliente no
        # prueban que la ruta este rota". Ahora SOLO una SONDA (period
        # ica o bajo demanda, exitosa o no) cuenta como "hay historia" --
        # un evento real, exito o fallo, nunca alcanza por si solo.
        cualquier_sonda = self._con.execute(
            "SELECT 1 FROM sondas WHERE clave = ? LIMIT 1", (clave,)).fetchone()
        return cualquier_sonda is None

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
