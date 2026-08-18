import sqlite3

from llm_libre.models import (NEUTRAL_QUALITY, NEUTRAL_RELIABILITY, NEUTRAL_TTFT_MS,
                               Capabilities, Metrics, Route)

# The schema below is the CURRENT shape. A live database (the /datos volume, which
# deliberately survives redeploys) may still carry the previous Spanish-named
# shape, so `_migrate` renames tables, columns and stored values in place -- see
# `_SPANISH_TABLES`, `_SPANISH_COLUMNS` and `_SPANISH_VALUES` below.
SCHEMA = """
CREATE TABLE IF NOT EXISTS routes (
    key TEXT PRIMARY KEY, provider TEXT NOT NULL, model_id TEXT NOT NULL,
    tier TEXT NOT NULL, tools INTEGER NOT NULL, vision INTEGER NOT NULL,
    context INTEGER NOT NULL, max_output INTEGER NOT NULL,
    last_seen REAL NOT NULL, active INTEGER NOT NULL DEFAULT 1,
    priority INTEGER NOT NULL DEFAULT 100);

CREATE TABLE IF NOT EXISTS probes (
    key TEXT NOT NULL, kind TEXT NOT NULL, at REAL NOT NULL,
    ok INTEGER NOT NULL, latency_ms INTEGER, ttft_ms INTEGER, http_code INTEGER,
    cases_passed INTEGER, cases_total INTEGER);
CREATE INDEX IF NOT EXISTS ix_probes ON probes(key, kind, at DESC);

CREATE TABLE IF NOT EXISTS events (
    key TEXT NOT NULL, at REAL NOT NULL, ok INTEGER NOT NULL,
    ttft_ms INTEGER, http_code INTEGER, latency_ms INTEGER,
    is_client_error INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS ix_events ON events(key, at DESC);

CREATE TABLE IF NOT EXISTS paid_usage (
    api_key TEXT NOT NULL, day TEXT NOT NULL, requests INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (api_key, day));
"""

# The previous Spanish-named shape, kept only so `_migrate` can recognise it and
# rename it in place. A production database carries real telemetry (thousands of
# probe and event rows), so this is a RENAME, never a drop-and-recreate.
_SPANISH_TABLES = [("rutas", "routes"), ("sondas", "probes"),
                   ("eventos", "events"), ("uso_pago", "paid_usage")]

_SPANISH_COLUMNS = {
    "routes": [("clave", "key"), ("proveedor", "provider"), ("modelo_id", "model_id"),
               ("contexto", "context"), ("max_salida", "max_output"),
               ("visto_por_ultima_vez", "last_seen"), ("activa", "active"),
               ("prioridad", "priority")],
    "probes": [("clave", "key"), ("tipo", "kind"), ("momento", "at"),
               ("latencia_ms", "latency_ms"), ("codigo_http", "http_code"),
               ("casos_pasados", "cases_passed"), ("casos_totales", "cases_total")],
    "events": [("clave", "key"), ("momento", "at"), ("codigo_http", "http_code"),
               ("latencia_ms", "latency_ms"), ("es_error_cliente", "is_client_error")],
    "paid_usage": [("llave", "api_key"), ("dia", "day"), ("peticiones", "requests")],
}

# Stored VALUES, not just names: `tier` and `kind` carry Spanish words in every
# existing row. Renaming the columns without rewriting these would leave the
# queries looking for "free" against rows that still say "gratis" -- a silent,
# total loss of routing and of every quality measurement.
_SPANISH_VALUES = [("routes", "tier", "gratis", "free"), ("routes", "tier", "pago", "paid"),
                   ("probes", "kind", "salud", "health"),
                   ("probes", "kind", "calidad", "quality")]

WINDOW = 50  # how many recent observations weigh in reliability and latency

# How far back POSITIVE EVIDENCE that a route works is searched for, for /health
# (Task 13, round 6 review, Part 2). Larger than the default health-probe interval
# (HEALTH_PROBE_HOURS=5h) with enough margin that ONE probing pass failing or
# running late does not drop the route to "no evidence" -- and generous for a PAID
# route, which is never probed (design section 8) and can only prove it is alive
# through real traffic, which may be sporadic (it is the fallback tier, not the
# main traffic). 24h = ~5 default probing cycles, and it stays bounded by `prune()`
# (30 days of retention) from above.
LIVENESS_WINDOW_S = 24 * 3600.0

# WHAT EACH TIME COLUMN MEANS (decided in fix round 3, I5).
#
# `ttft_ms` means ONE thing only: milliseconds to the first useful token the
# caller could see. Only something reading a stream can measure it, so only the
# streaming path writes there. Both paths used to: the non-streaming one put the
# COMPLETE round-trip (7-27 s on a reasoning model, because nothing arrives until
# it finishes generating) and the streaming one the real time to the first chunk
# (~200 ms). Mixed into one p50, the number meant nothing and the `fast` profile
# was ordering by noise.
#
# `latency_ms` is the complete round-trip, which is what a non-streaming path
# (and the health probe) actually measures. It is stored -- design section 5
# already asked for it in `events` -- and exposed in /v1/ranking, but it does NOT
# feed the score's latency factor, which is calibrated on the ttft scale.
#
# Accepted consequence: in a deployment using only the non-streaming path, no ttft
# is ever measured and every route keeps the neutral value, i.e. latency stops
# discriminating. That is preferable to discriminating by an invented number: it
# is visible in /v1/ranking (ttft_p50_ms == the neutral for all) and fixes itself
# as soon as there is streaming traffic.


class Storage:
    def __init__(self, db_path: str):
        self._con = sqlite3.connect(db_path, check_same_thread=False)
        self._con.execute("PRAGMA journal_mode=WAL")

    def create_schema(self) -> None:
        # ORDER MATTERS: the Spanish shape is renamed BEFORE `CREATE TABLE IF NOT
        # EXISTS` runs. The other way round, the CREATE would find no `routes`,
        # build an EMPTY one next to the populated `rutas`, and the rename would
        # then fail against a name that is already taken -- leaving the gateway
        # reading an empty catalogue while every real row sat in the old table.
        self._rename_spanish_shape()
        self._con.executescript(SCHEMA)
        self._migrate()
        self._con.commit()

    def _table_exists(self, name: str) -> bool:
        return self._con.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (name,)).fetchone() is not None

    def _columns(self, table: str) -> set[str]:
        return {f[1] for f in self._con.execute(f"PRAGMA table_info({table})")}

    def _rename_spanish_shape(self) -> None:
        """Migrate a database written under the previous Spanish names to the
        current English ones, IN PLACE and PRESERVING EVERY ROW.

        This is not cosmetic bookkeeping: the production database lives on the
        /datos volume (which deliberately survives redeploys) and carries the only
        copy of the telemetry -- thousands of probe and event rows that are what
        the router scores routes with. Recreating the schema instead of renaming
        it would silently reset every quality measurement and every reliability
        window to neutral, and the gateway would look perfectly healthy while
        routing blind for days until the probing cycles measured it all again.

        Three levels, in order, each guarded so the whole thing is idempotent and
        safe on a database that is already migrated, half-migrated, or brand new:

          1. TABLES. Renamed only when the old name exists and the new one does
             not. Indexes follow their table automatically but KEEP THEIR OLD
             NAMES, so the legacy ones are dropped -- SCHEMA recreates them under
             the new names, and leaving both would mean two identical indexes on
             the same table.
          2. COLUMNS. Only those actually present are renamed: an old enough
             database may predate `latencia_ms` or `es_error_cliente` (the
             add-column migration below is what fills those in), so the map is a
             superset of what any given database holds.
          3. VALUES. `tier` and `kind` store Spanish WORDS, not just Spanish
             column names. Renaming the columns without rewriting the contents
             would leave every query asking for 'free' against rows that still say
             'gratis' -- no error, no log line: zero routes match, and the gateway
             answers 503 to everything.

        `ALTER TABLE ... RENAME COLUMN` needs SQLite >= 3.25 (2018); production
        runs 3.46."""
        for old, new in _SPANISH_TABLES:
            if self._table_exists(old) and not self._table_exists(new):
                self._con.execute(f"ALTER TABLE {old} RENAME TO {new}")
        for legacy_index in ("ix_sondas", "ix_eventos"):
            self._con.execute(f"DROP INDEX IF EXISTS {legacy_index}")
        for table, renames in _SPANISH_COLUMNS.items():
            if not self._table_exists(table):
                continue
            present = self._columns(table)
            for old, new in renames:
                if old in present and new not in present:
                    self._con.execute(
                        f"ALTER TABLE {table} RENAME COLUMN {old} TO {new}")
        for table, column, old, new in _SPANISH_VALUES:
            if self._table_exists(table) and column in self._columns(table):
                self._con.execute(
                    f"UPDATE {table} SET {column} = ? WHERE {column} = ?", (new, old))
        self._con.commit()

    def _migrate(self) -> None:
        """`CREATE TABLE IF NOT EXISTS` does not add columns to a table that
        already exists: a live database (the /datos volume, which deliberately
        survives redeploys) would be left without `events.latency_ms` -- or,
        since Task 13, without `routes.priority` or `events.is_client_error`.
        Same pattern for all three: detect the missing column and add it with a
        default that does not break existing rows. OLD `events` rows (written
        before the distinction existed) migrate to `is_client_error=0` -- they
        keep counting as route failures, as they did before: there is no way to
        retroactively reclassify something that was never distinguished when it
        was written."""
        event_columns = self._columns("events")
        if "latency_ms" not in event_columns:
            self._con.execute("ALTER TABLE events ADD COLUMN latency_ms INTEGER")
        if "is_client_error" not in event_columns:
            self._con.execute(
                "ALTER TABLE events ADD COLUMN is_client_error INTEGER NOT NULL DEFAULT 0")
        if "priority" not in self._columns("routes"):
            self._con.execute(
                "ALTER TABLE routes ADD COLUMN priority INTEGER NOT NULL DEFAULT 100")

    def upsert_routes(self, routes: list[Route], timestamp: float,
                      deactivate_missing: bool = True,
                      provider: str | None = None) -> None:
        for r in routes:
            c = r.capabilities
            self._con.execute(
                """INSERT INTO routes (key, provider, model_id, tier, tools, vision,
                       context, max_output, last_seen, active, priority)
                   VALUES (?,?,?,?,?,?,?,?,?,1,?)
                   ON CONFLICT(key) DO UPDATE SET
                       tools=excluded.tools, vision=excluded.vision,
                       context=excluded.context, max_output=excluded.max_output,
                       last_seen=excluded.last_seen, active=1,
                       priority=excluded.priority""",
                (r.key, r.provider, r.model_id, r.tier, int(c.tools), int(c.vision),
                 c.context, c.max_output, timestamp, r.priority))
        # What was not seen in this pass is deactivated, not deleted: the history
        # is what makes it possible to detect a model rename. This step can be
        # skipped when the caller only brings a subset (e.g. syncing a single
        # provider among several) and does not want to switch off the others'
        # routes.
        #
        # `provider`, when passed, SCOPES that deactivation to that provider's
        # routes: without it, an UPDATE not filtered by provider would also switch
        # off routes belonging to providers UNRELATED to this call (their
        # last_seen is always older than `timestamp`, so they would
        # fall too). It is what lets one provider's sync decide ITS removals
        # without waiting to learn what happened to the others in the same pass.
        # None (the default) preserves the historical behaviour: scoped to
        # nothing, i.e. to the whole table.
        if deactivate_missing:
            if provider is not None:
                self._con.execute(
                    "UPDATE routes SET active = 0 WHERE last_seen < ? AND provider = ?",
                    (timestamp, provider))
            else:
                self._con.execute(
                    "UPDATE routes SET active = 0 WHERE last_seen < ?", (timestamp,))
        self._con.commit()

    def deactivate_unregistered_providers(self, known_providers: set[str]) -> int:
        """Switch off (active=0, NEVER delete -- the history is used to detect
        model renames, see upsert_routes) every route whose `provider` is no
        longer in `known_providers`.

        It exists because `upsert_routes` (and therefore the removal of routes
        that no longer appear) is SCOPED to each call's provider
        (`sync_catalogue` invokes it once per provider, with its own
        `provider=p.id`): a provider REMOVED entirely from the registry
        (`providers.yaml`) is never synced again, so without this separate sweep
        its routes would stay `active=1` forever -- visible in `GET /v1/models`
        and `GET /v1/ranking`, and eligible as routing candidates that would
        always fail (a 401 from a key that no longer exists, for instance),
        with nothing to switch them off.

        It is called once per probing cycle, from `probing.sync_catalogue`, with
        the set of ids the PROCESS has loaded from `providers.yaml` at that
        moment -- not a fixed list inside this function. Note the precision of
        that sentence (gate review): this does NOT mean a YAML edit takes effect
        on its own, without a restart -- `providers.load()` is called ONCE, at
        process start (`main.build_state`), so that set stays fixed in
        memory until the next restart. What IS true: given that set (fixed until
        restart), the sweep runs on EVERY probing cycle, so removing (or
        re-adding) a provider takes effect without touching code -- only one
        process restart is needed to load the new list, after which the sweep
        needs no further manual intervention. Returns how many rows were switched
        off, so the caller can leave a `log.warning` (a provider disappearing from
        the registry is an operator decision, but it is worth making visible in
        the logs when it actually has an effect).

        An empty `known_providers` is a deliberate special case (LOW from a gate
        review): a syntactically valid `providers.yaml` with `proveedores: []`
        (a truncated or badly hand-edited file, more likely than a real decision
        of "no providers") does NOT trigger the sweep -- it is treated as "nothing
        is known yet", not as "everything is orphaned", because the latter would
        switch off the ENTIRE catalogue on the first probing cycle after a config
        typo, with no other prior symptom to warn anyone. Reachable only through a
        deliberate YAML edit, and fully recoverable (the routes reactivate
        themselves on the next cycle if the registry brings them back) -- but the
        guard is free."""
        if not known_providers:
            return 0
        rows = self._con.execute(
            "SELECT DISTINCT provider FROM routes WHERE active = 1").fetchall()
        orphans = [p for (p,) in rows if p not in known_providers]
        if not orphans:
            return 0
        placeholders = ",".join("?" * len(orphans))
        cur = self._con.execute(
            f"UPDATE routes SET active = 0 WHERE active = 1 AND provider IN ({placeholders})",
            orphans)
        self._con.commit()
        return cur.rowcount

    def active_routes(self) -> list[Route]:
        rows = self._con.execute(
            """SELECT provider, model_id, tier, tools, vision, context, max_output,
                      priority
               FROM routes WHERE active = 1 ORDER BY key""").fetchall()
        return [Route(p, m, t, Capabilities(bool(to), bool(vi), cx, ms), priority=pr)
                for p, m, t, to, vi, cx, ms, pr in rows]

    def record_probe(self, key: str, kind: str, ok: bool, latency_ms: int,
                     ttft_ms: int, http_code: int, cases_passed: int,
                     cases_total: int, timestamp: float) -> None:
        self._con.execute(
            "INSERT INTO probes VALUES (?,?,?,?,?,?,?,?,?)",
            (key, kind, timestamp, int(ok), latency_ms, ttft_ms, http_code,
             cases_passed, cases_total))
        self._con.commit()

    def record_event(self, key: str, ok: bool, ttft_ms: int,
                     http_code: int, timestamp: float,
                     latency_ms: int | None = None,
                     is_client_error: bool = False) -> None:
        """`ttft_ms` is only written by whoever could measure a real
        time-to-first-token (the streaming path); everyone else passes 0 and sends
        their round-trip in `latency_ms`. See the comment at the top of the file.

        `is_client_error` (Task 13 review, round 4): a 4xx that is not 429 is
        evidence about the REQUEST, not about the route -- proxy.py already stopped
        counting it toward the hard-failure cooldown (round 3), but it was STILL
        being written as an ordinary failed event, and that feeds `_reliability`,
        which `/health` uses to declare a route dead. Reproduced: 26 consecutive
        malformed requests from ONE key were enough to sink the reliability of
        EVERY route, with /health reporting "down" while a DIFFERENT key kept
        receiving 200s the whole time -- worse than the 503 round 3 already fixed,
        because on the persistent /datos volume a process restart does NOT clear
        the history: it keeps reporting "down" against the same database. The row
        is STORED anyway (it stays diagnosable: an operator looking at `events`
        can see the 4xx) but `_reliability` excludes it from the window entirely --
        it neither counts as a failure nor takes up a slot in the last WINDOW
        observations."""
        self._con.execute(
            """INSERT INTO events (key, at, ok, ttft_ms, http_code, latency_ms,
                   is_client_error)
               VALUES (?,?,?,?,?,?,?)""",
            (key, timestamp, int(ok), ttft_ms, http_code, latency_ms, int(is_client_error)))
        self._con.commit()

    def metrics(self) -> dict[str, Metrics]:
        out: dict[str, Metrics] = {}
        for (key,) in self._con.execute("SELECT key FROM routes WHERE active = 1"):
            quality, measured_at = self._quality(key)
            out[key] = Metrics(
                quality=quality,
                reliability=self._reliability(key),
                ttft_p50_ms=self._ttft_p50(key),
                cooldown_until=0.0,  # cooldown lives in the proxy's memory
                quality_measured_at=measured_at,
                last_probe_at=self._last_probe(key),
                latency_p50_ms=self._latency_p50(key),
            )
        return out

    def _quality(self, key: str) -> tuple[float, float | None]:
        """(quality, time of the measurement). The time is None if it was never
        measured: there the returned quality is the NEUTRAL value, an assumption
        -- and whoever consumes it has to be able to tell a measured 0.6 from an
        assumed one."""
        row = self._con.execute(
            """SELECT cases_passed, cases_total, at FROM probes
               WHERE key = ? AND kind = 'quality' AND cases_total > 0
               ORDER BY at DESC LIMIT 1""", (key,)).fetchone()
        if not row:
            return NEUTRAL_QUALITY, None
        return row[0] / row[1], row[2]

    def _last_probe(self, key: str) -> float | None:
        row = self._con.execute(
            "SELECT MAX(at) FROM probes WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _reliability(self, key: str) -> float:
        # `events.is_client_error = 0` excludes the client's 4xx (see the
        # docstring of record_event): they neither count as failures NOR take up a
        # slot in the window, as if that request had never reached this route.
        # `probes` has no such column -- probes are generated by the gateway
        # itself with a fixed payload, they are never "the client's fault" in this
        # sense -- so all of them are included, as before.
        rows = self._con.execute(
            """SELECT ok FROM (
                   SELECT at, ok FROM probes WHERE key = ?
                   UNION ALL
                   SELECT at, ok FROM events
                   WHERE key = ? AND is_client_error = 0
               ) ORDER BY at DESC LIMIT ?""", (key, key, WINDOW)).fetchall()
        if not rows:
            return NEUTRAL_RELIABILITY
        return sum(f[0] for f in rows) / len(rows)

    def has_liveness_evidence(self, key: str, now: float) -> bool:
        """For `/health` (Task 13, round 6 review, Part 2; corrected in round 7)
        -- "evidence of life", not "absence of death". `reliability` (above) is an
        AVERAGE of the last `WINDOW` observations, and an average can be dragged
        to 0 by any repeated traffic pattern from ONE client. Classifying codes
        (Part 1) is not enough to close this completely: `403` is GENUINELY
        ambiguous -- suspended account (evidence about the route) or moderated
        content (evidence about the REQUEST) -- and the gateway cannot tell them
        apart without parsing each provider's specific body, so classifying it as
        route evidence (correct for the first case) leaves it vulnerable to the
        second: 30 requests with moderated content from one client are enough to
        sink the average for EVERYONE.

        The right question is not "how many failures did it have" but "what is the
        MOST RECENT SIGNAL available about this route": within
        `LIVENESS_WINDOW_S`, the last real SUCCESS (`events.ok=1`) is compared
        with the last HEALTH PROBE (`probes.kind='health'`, any result) and the one
        with the newer `at` wins.

          - If the newest signal is a real success, or a SUCCESSFUL probe: alive.
          - If the newest signal is a FAILED probe: dead. The probe is the most
            reliable signal there is, because the GATEWAY controls its own payload
            -- a result against a probe is NEVER ambiguous, in any direction: if a
            4xx against a probe is ALWAYS evidence about the route (round 6), a
            probe that outright failed is just as much, and round 7 fixes the bug
            of looking at only half that argument (`ok=1`) and ignoring the other
            (`ok=0`). Reproduced: a real success 20h ago (inside a 24h window) but
            FOUR failed probes since (one every 5h, the probing default) -- the
            previous version found the old success, settled on it, and never
            looked at what happened AFTERWARDS. The 24h window is defensible for a
            PAID route that is never probed; it is not for a free route probed
            every 5h whose latest results are KNOWN and were being discarded.
          - If there is no signal at all (neither success nor probe) within the
            window: it falls through to the "no real telemetry whatsoever" check
            (below) -- a freshly seen route was not born dead, it simply has not
            had its first chance yet.

        FAILED events (`events.ok=0`) never contribute evidence of death directly
        here, only indirectly via the "no telemetry" check: they remain ambiguous
        (a 403 may be content moderation for one particular client) in a way that
        a probe -- a fixed request, controlled by the gateway itself -- never is.
        Events with `is_client_error=1` (Part 1) do not count as "real" telemetry
        for the fallback check either: a route that ONLY received malformed
        requests (400/413/422) has not had its first real chance yet, same as one
        with no traffic at all.

        `/v1/ranking` does NOT use this signal -- it stays on `reliability`
        exactly as before (and so does `router.order_routes`, which orders the
        real chain of attempts -- this is not ONLY the diagnostic endpoint): a
        route that scores badly merely loses position and self-corrects; a route
        `/health` declares dead RESTARTS THE CONTAINER (Coolify uses `/health` as
        its health check). The asymmetry is the point: the ranking can afford to
        be sensitive, health cannot.

        Round 9: ONE failed probe alone is NO LONGER ENOUGH -- TWO consecutive
        failed signals (with no success in between) are required before treating
        it as evidence of death. Before round 9, "the most recent signal decides"
        sufficed because the ONLY trigger for a probe was the periodic cycle
        (every 5h, fixed rhythm, never influenced by a client). Since real traffic
        can trigger ON-DEMAND probes (up to 60/h per route, ~300x more often than
        the 5h cycle), a client controls WHEN a route is sampled -- and more
        samples means more chances of catching, by pure luck, a transient provider
        problem (a network blip, an isolated timeout) in ONE probe, and having
        `/health` treat it as a definitive verdict that survives a container
        restart. Requiring TWO consecutive failures lets a genuinely transient
        problem resolve itself (the next probe or the next real request, almost
        always successful) before /health declares it dead, without losing fast
        detection of a real outage (a genuinely broken route keeps failing the
        SECOND probe just like the first). A real SUCCESS still suffices on its
        own -- the asymmetry "one success proves life, one isolated failure does
        not prove death" is the same as always, only now it applies WITHIN the
        comparison between probes too."""
        cutoff = now - LIVENESS_WINDOW_S
        signals = self._con.execute(
            """SELECT ok FROM (
                   SELECT at, 1 AS ok FROM events
                       WHERE key = ? AND ok = 1 AND at >= ?
                   UNION ALL
                   SELECT at, ok FROM probes
                       WHERE key = ? AND kind = 'health' AND at >= ?
               )
               ORDER BY at DESC
               LIMIT 2""",
            (key, cutoff, key, cutoff)).fetchall()
        if signals:
            if signals[0][0]:
                return True
            # The most recent signal is a failure: the PREVIOUS one must be a
            # failure too (two consecutive, with no success in between) -- a
            # single failure, or a failure right after a success, is not enough.
            return not (len(signals) >= 2 and not signals[1][0])
        # Round 10, MEDIUM from the gate: this fallback check (nothing within the
        # window, neither success nor probe) used to look at ANY real event --
        # success OR FAILURE -- to decide "there is history, declare it dead". A
        # real failure, with no probe confirming it, should NEVER be
        # authoritative: it is the same principle that motivates suspicion+probe
        # since round 8 (a client's traffic can only ask the gateway to go and
        # look, never exclude a route itself). Measured: ONE failed real request
        # was enough to drop /health to "down" without ANY probe having run --
        # fewer than SUSPICION_THRESHOLD, so no on-demand probe ever fires to rescue
        # it. It directly contradicts the module's principle: "a thousand failures
        # from one client do not prove the route is broken". Now ONLY a PROBE
        # (periodic or on-demand, successful or not) counts as "there is history"
        # -- a real event, success or failure, is never enough on its own.
        any_probe = self._con.execute(
            "SELECT 1 FROM probes WHERE key = ? LIMIT 1", (key,)).fetchone()
        return any_probe is None

    def _ttft_p50(self, key: str) -> float:
        """p50 of time-to-first-token. Only observations that genuinely measured a
        ttft (streaming) are included: the rest write 0 and fall outside the
        `ttft_ms > 0`. With none, the neutral value is returned."""
        rows = self._con.execute(
            """SELECT ttft_ms FROM (
                   SELECT at, ttft_ms FROM probes WHERE key = ? AND ok = 1
                   UNION ALL
                   SELECT at, ttft_ms FROM events WHERE key = ? AND ok = 1
               ) WHERE ttft_ms > 0 ORDER BY at DESC LIMIT ?""",
            (key, key, WINDOW)).fetchall()
        if not rows:
            return NEUTRAL_TTFT_MS
        values = sorted(f[0] for f in rows)
        return float(values[len(values) // 2])

    def _latency_p50(self, key: str) -> float | None:
        """p50 of the complete round-trip. It does not enter the score (the
        latency factor is calibrated on the ttft scale); it is exposed for
        diagnostics, which is exactly what was missing when the two measurements
        shared a column. None = never observed."""
        rows = self._con.execute(
            """SELECT latency_ms FROM (
                   SELECT at, latency_ms FROM probes WHERE key = ? AND ok = 1
                   UNION ALL
                   SELECT at, latency_ms FROM events WHERE key = ? AND ok = 1
               ) WHERE latency_ms > 0 ORDER BY at DESC LIMIT ?""",
            (key, key, WINDOW)).fetchall()
        if not rows:
            return None
        values = sorted(f[0] for f in rows)
        return float(values[len(values) // 2])

    def add_paid_usage(self, api_key: str, day: str) -> int:
        self._con.execute(
            """INSERT INTO paid_usage (api_key, day, requests) VALUES (?,?,1)
               ON CONFLICT(api_key, day) DO UPDATE SET requests = requests + 1""",
            (api_key, day))
        self._con.commit()
        return self.paid_usage(api_key, day)

    def paid_usage(self, api_key: str, day: str) -> int:
        row = self._con.execute(
            "SELECT requests FROM paid_usage WHERE api_key = ? AND day = ?",
            (api_key, day)).fetchone()
        return row[0] if row else 0

    def prune(self, before: float) -> None:
        self._con.execute("DELETE FROM probes WHERE at < ?", (before,))
        self._con.execute("DELETE FROM events WHERE at < ?", (before,))
        self._con.commit()
