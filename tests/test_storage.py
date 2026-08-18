import sqlite3

import pytest

from llm_libre.storage import Storage
from llm_libre.models import Capabilities, Route


def _route(modelo="a:free", provider="kilo", tools=True, priority=100):
    return Route(provider, modelo, "gratis",
                Capabilities(tools=tools, vision=False, context=1000, max_output=100),
                priority=priority)


@pytest.fixture
def store():
    a = Storage(":memory:")
    a.create_schema()
    return a


def test_it_stores_and_returns_routes(store):
    store.upsert_routes([_route()], timestamp=100.0)
    active = store.active_routes()
    assert len(active) == 1
    assert active[0].key == "kilo/a:free"
    assert active[0].capabilities.tools is True


def test_a_route_that_disappears_is_deactivated_not_deleted(store):
    store.upsert_routes([_route("vieja:free"), _route("nueva:free")], timestamp=100.0)
    store.upsert_routes([_route("nueva:free")], timestamp=200.0)
    active = [r.model_id for r in store.active_routes()]
    assert active == ["nueva:free"]
    # still in the table: the history is what detects renames
    row = store._con.execute(
        "SELECT activa FROM rutas WHERE modelo_id = 'vieja:free'").fetchone()
    assert row[0] == 0


def test_a_route_that_comes_back_is_reactivated(store):
    store.upsert_routes([_route("x:free")], timestamp=100.0)
    store.upsert_routes([], timestamp=200.0)
    store.upsert_routes([_route("x:free")], timestamp=300.0)
    assert len(store.active_routes()) == 1


def test_it_deactivates_nothing_when_asked_to_keep(store):
    store.upsert_routes([_route("vieja:free"), _route("nueva:free")], timestamp=100.0)
    store.upsert_routes([_route("nueva:free")], timestamp=200.0, deactivate_missing=False)
    active = sorted(r.model_id for r in store.active_routes())
    assert active == ["nueva:free", "vieja:free"]
    row = store._con.execute(
        "SELECT visto_por_ultima_vez FROM rutas WHERE modelo_id = 'nueva:free'").fetchone()
    assert row[0] == 200.0


def test_the_provider_scope_limits_removal_to_that_provider(store):
    # kilo and otro each have an old route. Re-syncing ONLY kilo (with
    # provider="kilo") switches off its old route, but otro's -- older still, and
    # not even mentioned in this call -- must survive: without the scope, an
    # UPDATE not filtered by provider would have switched it off too, because its
    # visto_por_ultima_vez is also older than the new `timestamp`.
    store.upsert_routes([_route("vieja:free", provider="kilo")], timestamp=50.0)
    store.upsert_routes([_route("vieja:free", provider="otro")], timestamp=50.0)
    store.upsert_routes([_route("nueva:free", provider="kilo")], timestamp=200.0, provider="kilo")
    active = {r.key for r in store.active_routes()}
    assert active == {"kilo/nueva:free", "otro/vieja:free"}


def test_the_provider_scope_does_not_change_the_default_behaviour(store):
    # provider=None (the default) preserves the historical behaviour: with no
    # scope, it covers the whole table -- exactly what
    # test_a_route_that_disappears_is_deactivated_not_deleted already covers. This
    # test only confirms that passing provider=None explicitly is the same.
    store.upsert_routes([_route("vieja:free", provider="kilo"),
                          _route("otra:free", provider="otro")], timestamp=100.0)
    store.upsert_routes([_route("otra:free", provider="otro")], timestamp=200.0, provider=None)
    active = {r.key for r in store.active_routes()}
    assert active == {"otro/otra:free"}   # kilo/vieja:free is switched off too, as before


# --- Removing a provider from proveedores.yaml (e.g. openrouter) must not
#     leave its routes at activa=1 forever: sync_catalogue can only remove, via
#     its scope, what is STILL in the registry -- a provider that disappears
#     entirely never passes through that loop again. This separate sweep covers
#     exactly that gap. ---

def test_deactivating_unregistered_providers_switches_off_the_departed_ones_routes(store):
    store.upsert_routes([_route("a:free", provider="kilo")], timestamp=50.0)
    store.upsert_routes([_route("b:free", provider="openrouter")], timestamp=50.0)
    switched_off = store.deactivate_unregistered_providers({"kilo"})
    assert switched_off == 1
    keys_ = {r.key for r in store.active_routes()}
    assert keys_ == {"kilo/a:free"}
    # Not deleted: still in the table, only inactive -- the same principle as
    # upsert_routes with routes that vanish from a provider's catalogue.
    row = store._con.execute(
        "SELECT activa FROM rutas WHERE clave = 'openrouter/b:free'").fetchone()
    assert row == (0,)


def test_deactivating_unregistered_providers_leaves_the_remaining_ones_alone(store):
    store.upsert_routes([_route("a:free", provider="kilo"),
                          _route("c:free", provider="chatgpt")], timestamp=50.0)
    store.upsert_routes([_route("b:free", provider="openrouter")], timestamp=50.0)
    store.deactivate_unregistered_providers({"kilo", "chatgpt"})
    keys_ = {r.key for r in store.active_routes()}
    assert keys_ == {"kilo/a:free", "chatgpt/c:free"}


def test_deactivating_unregistered_providers_does_nothing_when_all_remain(store):
    store.upsert_routes([_route("a:free", provider="kilo")], timestamp=50.0)
    switched_off = store.deactivate_unregistered_providers({"kilo", "chatgpt", "minimax"})
    assert switched_off == 0
    assert len(store.active_routes()) == 1


def test_deactivating_unregistered_providers_is_idempotent(store):
    # A route that is already inactive (for whatever reason) is neither counted
    # again nor re-touched on a second pass.
    store.upsert_routes([_route("b:free", provider="openrouter")], timestamp=50.0)
    first = store.deactivate_unregistered_providers({"kilo"})
    second = store.deactivate_unregistered_providers({"kilo"})
    assert first == 1
    assert second == 0


def test_deactivating_unregistered_providers_with_an_empty_set_switches_off_nothing(store):
    # Gate review: a syntactically valid `proveedores.yaml` with
    # `proveedores: []` (more likely a truncated or badly edited file than a real
    # decision of "no providers") must not switch off the ENTIRE catalogue on the
    # first cycle -- an empty set is treated as "nothing is known yet", not as
    # "everything is orphaned".
    store.upsert_routes([_route("a:free", provider="kilo"),
                          _route("c:free", provider="chatgpt")], timestamp=50.0)
    switched_off = store.deactivate_unregistered_providers(set())
    assert switched_off == 0
    assert {r.key for r in store.active_routes()} == {"kilo/a:free", "chatgpt/c:free"}


def test_quality_comes_from_the_last_quality_probe(store):
    store.upsert_routes([_route()], timestamp=100.0)
    store.record_probe("kilo/a:free", "calidad", True, 500, 200, 200, 2, 5, 100.0)
    store.record_probe("kilo/a:free", "calidad", True, 500, 200, 200, 4, 5, 200.0)
    assert store.metrics()["kilo/a:free"].quality == pytest.approx(0.8)


def test_reliability_mixes_probes_and_events(store):
    store.upsert_routes([_route()], timestamp=100.0)
    store.record_probe("kilo/a:free", "salud", True, 100, 50, 200, 0, 0, 100.0)
    store.record_event("kilo/a:free", False, 0, 500, 150.0)
    m = store.metrics()["kilo/a:free"]
    assert 0.0 < m.reliability < 1.0


# --- Task 13 re-review (round 4): a client 4xx no longer punishes the route
#     (round 3), but it was STILL being written as a failed event, and that feeds
#     reliability -- which /health uses to declare a route dead. Reproduced: 26
#     consecutive malformed requests from ONE key are enough to sink EVERY route's
#     reliability, with /health at "caido" while a DIFFERENT key keeps receiving
#     200s. `record_event` gains `is_client_error`, and _reliability excludes those
#     event rows ENTIRELY -- they neither count as failures nor take up a slot in
#     the window -- so a 4xx is evidence about the REQUEST, not about the route.
#     They stay written (not discarded) so they remain diagnosable. ---

def test_reliability_ignores_events_marked_as_client_errors(store):
    store.upsert_routes([_route()], timestamp=100.0)
    for i in range(30):
        store.record_event("kilo/a:free", False, 0, 400, 100.0 + i,
                                 is_client_error=True)
    m = store.metrics()["kilo/a:free"]
    # With no other observation, the window is EMPTY (not 30 rows counting as
    # failures): reliability falls back to the neutral value, not to 0.
    assert m.reliability == pytest.approx(0.8)   # NEUTRAL_RELIABILITY


def test_reliability_still_drops_on_failures_that_are_not_the_clients(store):
    # A direct regression: a 500 (is_client_error=False, the default) has to keep
    # counting as it did before.
    store.upsert_routes([_route()], timestamp=100.0)
    for i in range(30):
        store.record_event("kilo/a:free", False, 0, 500, 100.0 + i)
    m = store.metrics()["kilo/a:free"]
    assert m.reliability == pytest.approx(0.0)


def test_reliability_mixes_client_errors_and_ignores_only_those(store):
    store.upsert_routes([_route()], timestamp=100.0)
    store.record_event("kilo/a:free", True, 50, 200, 100.0)
    for i in range(10):
        store.record_event("kilo/a:free", False, 0, 400, 101.0 + i,
                                 is_client_error=True)
    m = store.metrics()["kilo/a:free"]
    # The only event that "counts" is the success: the 10 client errors stay
    # entirely outside the window.
    assert m.reliability == pytest.approx(1.0)


_OLD_SCHEMA_WITHOUT_CLIENT_ERROR_FLAG = """
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


def test_it_migrates_an_old_database_without_the_client_error_flag(tmp_path):
    db_path = str(tmp_path / "vieja_sin_flag.sqlite3")
    con = sqlite3.connect(db_path)
    con.executescript(_OLD_SCHEMA_WITHOUT_CLIENT_ERROR_FLAG)
    con.execute(
        """INSERT INTO rutas (clave, proveedor, modelo_id, tier, tools, vision,
               contexto, max_salida, visto_por_ultima_vez, activa, prioridad)
           VALUES ('kilo/vieja:free','kilo','vieja:free','gratis',1,0,1000,100,50.0,1,100)""")
    con.execute(
        """INSERT INTO eventos (clave, momento, ok, ttft_ms, codigo_http, latencia_ms)
           VALUES ('kilo/vieja:free', 60.0, 0, 0, 400, 20)""")
    con.commit()
    con.close()

    store = Storage(db_path)
    store.create_schema()   # no debe reventar (ALTER TABLE, no CREATE)

    # The old row, written BEFORE es_error_cliente existed, migrates to 0 (the
    # historical behaviour: it DOES count as a failure) -- an event that never
    # distinguished the cause cannot be reclassified retroactively.
    row = store._con.execute(
        "SELECT es_error_cliente FROM eventos WHERE clave = 'kilo/vieja:free'").fetchone()
    assert row[0] == 0

    # And the migrated database is still writable with the new flag.
    store.record_event("kilo/vieja:free", False, 0, 400, 70.0, is_client_error=True)
    rows = store._con.execute(
        "SELECT es_error_cliente FROM eventos WHERE clave = 'kilo/vieja:free' "
        "ORDER BY momento").fetchall()
    assert [f[0] for f in rows] == [0, 1]


def test_a_route_without_data_gets_neutral_metrics(store):
    store.upsert_routes([_route()], timestamp=100.0)
    m = store.metrics()["kilo/a:free"]
    assert m.quality == pytest.approx(0.6)
    assert m.cooldown_until == 0.0


def test_paid_usage_is_counted_per_key_and_day(store):
    assert store.paid_usage("k1", "2026-08-16") == 0
    assert store.add_paid_usage("k1", "2026-08-16") == 1
    assert store.add_paid_usage("k1", "2026-08-16") == 2
    assert store.paid_usage("k1", "2026-08-17") == 0
    assert store.paid_usage("k2", "2026-08-16") == 0


# --- Fix round 3, B2b/I3: tell "measured quality of 0.6" from "never measured". ---

def test_a_never_probed_route_declares_no_quality_timestamp(store):
    store.upsert_routes([_route()], timestamp=100.0)
    m = store.metrics()["kilo/a:free"]
    assert m.quality_measured_at is None
    assert m.last_probe_at is None
    assert m.quality == pytest.approx(0.6)   # the neutral still feeds the score


def test_a_probed_route_declares_the_time_of_its_last_quality_probe(store):
    store.upsert_routes([_route()], timestamp=100.0)
    store.record_probe("kilo/a:free", "calidad", True, 0, 0, 200, 2, 5, 300.0)
    store.record_probe("kilo/a:free", "calidad", True, 0, 0, 200, 4, 5, 900.0)
    m = store.metrics()["kilo/a:free"]
    assert m.quality_measured_at == 900.0
    assert m.quality == pytest.approx(0.8)


def test_the_last_probe_counts_health_probes_too(store):
    store.upsert_routes([_route()], timestamp=100.0)
    store.record_probe("kilo/a:free", "calidad", True, 0, 0, 200, 4, 5, 300.0)
    store.record_probe("kilo/a:free", "salud", True, 120, 0, 200, 0, 0, 800.0)
    m = store.metrics()["kilo/a:free"]
    assert m.last_probe_at == 800.0        # the most recent of any kind
    assert m.quality_measured_at == 300.0  # but the QUALITY one is still its own


# --- Fix round 3, I5: `ttft_ms` mixed two incompatible measurements into one
#     column. The non-streaming path stored the COMPLETE round-trip (7-27 s on a
#     reasoning model) and the streaming one the real time to the first chunk
#     (~200 ms). Mixed into one p50, the `rapido` profile was ordering by a number
#     that means nothing. ---

def test_the_ttft_p50_only_counts_genuine_ttft_measurements(store):
    store.upsert_routes([_route()], timestamp=100.0)
    # Streaming: a real ttft.
    store.record_event("kilo/a:free", True, 200, 200, 150.0)
    # Non-streaming: there is no ttft to measure, the round-trip goes to latencia_ms.
    store.record_event("kilo/a:free", True, 0, 200, 160.0, latency_ms=21000)
    store.record_event("kilo/a:free", True, 0, 200, 170.0, latency_ms=19000)
    assert store.metrics()["kilo/a:free"].ttft_p50_ms == 200.0


def test_total_latency_is_stored_even_though_it_does_not_feed_the_ttft(store):
    store.upsert_routes([_route()], timestamp=100.0)
    store.record_event("kilo/a:free", True, 0, 200, 160.0, latency_ms=21000)
    store.record_event("kilo/a:free", True, 0, 200, 170.0, latency_ms=19000)
    m = store.metrics()["kilo/a:free"]
    assert m.latency_p50_ms == 21000.0     # p50 of the two observations
    assert m.ttft_p50_ms == 1500.0         # the neutral: ttft was never measured


def test_with_no_ttft_observation_the_neutral_value_is_used(store):
    store.upsert_routes([_route()], timestamp=100.0)
    m = store.metrics()["kilo/a:free"]
    assert m.ttft_p50_ms == 1500.0
    assert m.latency_p50_ms is None


# --- Task 13: `priority` persists and an old database migrates without data loss. ---

def test_upsert_routes_persists_the_priority(store):
    store.upsert_routes([_route("chatgpt:free", provider="chatgpt", priority=0)],
                         timestamp=100.0)
    active = store.active_routes()
    assert len(active) == 1
    assert active[0].priority == 0


def test_upsert_routes_without_a_declared_priority_persists_the_default(store):
    store.upsert_routes([_route()], timestamp=100.0)
    assert store.active_routes()[0].priority == 100


def test_resyncing_updates_an_existing_routes_priority(store):
    # A `prioridad` change in the YAML (e.g. moving a provider up) has to
    # propagate on the next sync, not stay stuck to the value the route was first
    # seen with.
    store.upsert_routes([_route("a:free", priority=1)], timestamp=100.0)
    store.upsert_routes([_route("a:free", priority=0)], timestamp=200.0)
    assert store.active_routes()[0].priority == 0


# `_migrate()` already has a case (eventos.latencia_ms, see the header comment
# of storage.py) that adds a column to a table that ALREADY exists using
# `ALTER TABLE ... ADD COLUMN` -- because `CREATE TABLE IF NOT EXISTS` does not
# touch an existing table. This test reproduces the same risk for
# `rutas.prioridad`: production's `rutas` table (the /datos volume) exists from
# BEFORE this feature and already has rows. If the migration were not idempotent
# and compatible with existing data, a redeploy against that database would blow
# up at startup (or silently lose the column).
_OLD_SCHEMA_WITHOUT_PRIORITY = """
CREATE TABLE rutas (
    clave TEXT PRIMARY KEY, proveedor TEXT NOT NULL, modelo_id TEXT NOT NULL,
    tier TEXT NOT NULL, tools INTEGER NOT NULL, vision INTEGER NOT NULL,
    contexto INTEGER NOT NULL, max_salida INTEGER NOT NULL,
    visto_por_ultima_vez REAL NOT NULL, activa INTEGER NOT NULL DEFAULT 1);
"""


def test_it_migrates_an_old_database_with_rows_without_losing_data(tmp_path):
    db_path = str(tmp_path / "vieja.sqlite3")
    # Simulates the production database: the PRE-priority schema, with a real
    # row inside (visto_por_ultima_vez, activa -- everything the old version of
    # the code already wrote).
    con = sqlite3.connect(db_path)
    con.executescript(_OLD_SCHEMA_WITHOUT_PRIORITY)
    con.execute(
        """INSERT INTO rutas (clave, proveedor, modelo_id, tier, tools, vision,
               contexto, max_salida, visto_por_ultima_vez, activa)
           VALUES ('kilo/vieja:free','kilo','vieja:free','gratis',1,0,1000,100,50.0,1)""")
    con.commit()
    con.close()

    # Opening it with the NEW code must not blow up (ALTER TABLE, not CREATE).
    store = Storage(db_path)
    store.create_schema()

    active = store.active_routes()
    assert len(active) == 1
    assert active[0].key == "kilo/vieja:free"
    # The pre-existing row, with no priority at the time it was written,
    # migrates to the default (100), not to NULL nor to an invented value.
    assert active[0].priority == 100

    # And the migrated database is still writable: a new sync can declare a
    # priority for that same route or for a new one.
    store.upsert_routes([_route("vieja:free", provider="kilo", priority=0)], timestamp=200.0)
    store.upsert_routes([_route("nueva:free", provider="chatgpt", priority=0)],
                         timestamp=200.0, deactivate_missing=False)
    active = {r.key: r.priority for r in store.active_routes()}
    assert active == {"kilo/vieja:free": 0, "chatgpt/nueva:free": 0}


def test_migrating_an_old_database_is_idempotent(tmp_path):
    # Opening the migrated database a SECOND time (the next redeploy) must not
    # blow up with "duplicate column name".
    db_path = str(tmp_path / "vieja2.sqlite3")
    con = sqlite3.connect(db_path)
    con.executescript(_OLD_SCHEMA_WITHOUT_PRIORITY)
    con.close()

    Storage(db_path).create_schema()
    again = Storage(db_path)
    again.create_schema()   # must not blow up
    assert again.active_routes() == []


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

def test_liveness_evidence_with_no_telemetry_at_all(store):
    # Ruta recien vista: no nacio muerta, todavia no tuvo su primera
    # oportunidad ni para bien ni para mal.
    store.upsert_routes([_route()], timestamp=100.0)
    assert store.has_liveness_evidence("kilo/a:free", now=100.0) is True


def test_liveness_evidence_from_a_recent_success_despite_many_failures(store):
    store.upsert_routes([_route()], timestamp=100.0)
    store.record_event("kilo/a:free", True, 50, 200, 100.0)
    for i in range(30):
        store.record_event("kilo/a:free", False, 0, 403, 101.0 + i)
    assert store.has_liveness_evidence("kilo/a:free", now=200.0) is True


def test_liveness_evidence_from_a_recent_successful_health_probe(store):
    store.upsert_routes([_route()], timestamp=100.0)
    store.record_probe("kilo/a:free", "salud", True, 100, 50, 200, 0, 0, 150.0)
    assert store.has_liveness_evidence("kilo/a:free", now=200.0) is True


# --- Round 10, MEDIUM del gate: el chequeo de respaldo ("nada dentro de la
#     ventana") miraba CUALQUIER evento real -- exito o FALLO -- para
#     decidir "hay historia, declarar muerta". Medido: un solo pedido real
#     fallido (menos que SUSPICION_THRESHOLD, asi que ninguna sonda bajo
#     demanda llega a dispararse) bastaba para tirar /health a "caido" --
#     contradice el principio del modulo, "mil fallos de un mismo cliente
#     no prueban que la ruta este rota". Ahora SOLO una SONDA (nunca un
#     evento real, exito o fallo) cuenta como "hay historia". ---

def test_real_failures_alone_without_any_probe_do_not_declare_it_dead(store):
    # El camino que el trafico real SOLO puede activar (sin ninguna sonda
    # de por medio) ya no alcanza -- ni con 30 fallos.
    store.upsert_routes([_route()], timestamp=100.0)
    for i in range(30):
        store.record_event("kilo/a:free", False, 0, 500, 100.0 + i)
    assert store.has_liveness_evidence("kilo/a:free", now=200.0) is True


def test_real_failures_with_two_failed_probes_confirming_do_declare_it_dead(store):
    # El camino REAL para llegar a "muerta": trafico real dispara sospecha
    # (round 8), sospecha dispara sondas, y son DOS sondas fallidas
    # consecutivas (round 9) las que confirman -- nunca el trafico solo.
    store.upsert_routes([_route()], timestamp=100.0)
    for i in range(30):
        store.record_event("kilo/a:free", False, 0, 500, 100.0 + i)
    store.record_probe("kilo/a:free", "salud", False, 100, 0, 500, 0, 0, 131.0)
    store.record_probe("kilo/a:free", "salud", False, 100, 0, 500, 0, 0, 132.0)
    assert store.has_liveness_evidence("kilo/a:free", now=200.0) is False


def test_liveness_evidence_holds_when_every_failure_is_a_client_error(store):
    # Una ruta que SOLO recibio pedidos malformados (400/413/422, Parte 1)
    # todavia no tuvo su primera oportunidad de verdad -- se trata igual
    # que "sin telemetria", no como "muerta".
    store.upsert_routes([_route()], timestamp=100.0)
    for i in range(30):
        store.record_event("kilo/a:free", False, 0, 400, 100.0 + i,
                                 is_client_error=True)
    assert store.has_liveness_evidence("kilo/a:free", now=200.0) is True


def test_a_success_outside_the_window_with_only_recent_real_failures_is_not_dead(store):
    from llm_libre.storage import LIVENESS_WINDOW_S
    store.upsert_routes([_route()], timestamp=100.0)
    store.record_event("kilo/a:free", True, 50, 200, 100.0)
    now = 100.0 + LIVENESS_WINDOW_S + 1.0
    for i in range(30):
        store.record_event("kilo/a:free", False, 0, 500, now - 10.0 + i * 0.1)
    assert store.has_liveness_evidence("kilo/a:free", now=now) is True


def test_a_success_outside_the_window_with_two_recent_failed_probes_is_dead(store):
    from llm_libre.storage import LIVENESS_WINDOW_S
    store.upsert_routes([_route()], timestamp=100.0)
    store.record_event("kilo/a:free", True, 50, 200, 100.0)
    now = 100.0 + LIVENESS_WINDOW_S + 1.0
    store.record_probe("kilo/a:free", "salud", False, 100, 0, 500, 0, 0, now - 1)
    store.record_probe("kilo/a:free", "salud", False, 100, 0, 500, 0, 0, now)
    assert store.has_liveness_evidence("kilo/a:free", now=now) is False


# --- Round 9, hallazgo del gate ("el camino indirecto de /health"): desde
#     que el trafico real puede disparar sondas BAJO DEMANDA (hasta 60/h por
#     ruta, ~300x mas seguido que el ciclo periodico de 5h), un cliente
#     controla CUANDO se muestrea una ruta -- mas muestras, mas chances de
#     agarrar por azar un problema transitorio del proveedor en UNA sola
#     sonda, y que /health lo trate como veredicto definitivo (sobrevive un
#     reinicio de contenedor). Decision: UNA sonda fallida sola YA NO
#     alcanza -- hacen falta DOS consecutivas, sin exito de por medio. ---

def test_a_single_failed_probe_is_no_longer_enough_to_declare_it_dead(store):
    store.upsert_routes([_route()], timestamp=100.0)
    store.record_probe("kilo/a:free", "salud", False, 100, 0, 500, 0, 0, 150.0)
    assert store.has_liveness_evidence("kilo/a:free", now=200.0) is True


def test_two_consecutive_failed_probes_do_declare_it_dead(store):
    store.upsert_routes([_route()], timestamp=100.0)
    store.record_probe("kilo/a:free", "salud", False, 100, 0, 500, 0, 0, 140.0)
    store.record_probe("kilo/a:free", "salud", False, 100, 0, 500, 0, 0, 150.0)
    assert store.has_liveness_evidence("kilo/a:free", now=200.0) is False


def test_a_failed_probe_right_after_a_success_is_not_enough(store):
    # Un solo fallo precedido por un exito NO es "dos consecutivos": la
    # señal mas vieja de las dos ultimas es un exito, no otro fallo.
    store.upsert_routes([_route()], timestamp=100.0)
    store.record_event("kilo/a:free", True, 50, 200, 140.0)
    store.record_probe("kilo/a:free", "salud", False, 100, 0, 500, 0, 0, 150.0)
    assert store.has_liveness_evidence("kilo/a:free", now=200.0) is True


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

def test_a_failed_probe_newer_than_an_old_success_declares_the_route_dead(store):
    store.upsert_routes([_route()], timestamp=0.0)
    now = 100_000.0
    twenty_hours = 20 * 3600.0
    five_hours = 5 * 3600.0
    store.record_event("kilo/a:free", True, 50, 200, now - twenty_hours)
    # Cuatro sondas de salud FALLIDAS desde entonces, una cada 5h -- todas
    # mas recientes que el exito de arriba, la ULTIMA hace apenas 1h.
    for i in range(1, 5):
        store.record_probe("kilo/a:free", "salud", False, 100, 0, 500, 0, 0,
                                now - twenty_hours + i * five_hours)
    assert store.has_liveness_evidence("kilo/a:free", now=now) is False


def test_a_real_success_newer_than_a_failed_probe_declares_the_route_alive(store):
    # Simetrico al de arriba: si DESPUES de una sonda fallida hay un exito
    # real (un cliente de verdad recibio respuesta), esa es la senal mas
    # nueva y gana -- la ruta esta viva AHORA, sin importar el tropiezo
    # anterior de la sonda.
    store.upsert_routes([_route()], timestamp=0.0)
    store.record_probe("kilo/a:free", "salud", False, 100, 0, 500, 0, 0, 100.0)
    store.record_event("kilo/a:free", True, 50, 200, 150.0)
    assert store.has_liveness_evidence("kilo/a:free", now=200.0) is True


def test_a_successful_probe_newer_than_a_failed_one_declares_the_route_alive(store):
    store.upsert_routes([_route()], timestamp=0.0)
    store.record_probe("kilo/a:free", "salud", False, 100, 0, 500, 0, 0, 100.0)
    store.record_probe("kilo/a:free", "salud", True, 100, 50, 200, 0, 0, 150.0)
    assert store.has_liveness_evidence("kilo/a:free", now=200.0) is True
