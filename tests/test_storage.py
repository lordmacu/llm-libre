import sqlite3

import pytest

from llm_libre.storage import QUALITY_RUNS, Storage
from llm_libre.models import Capabilities, RateBudget, RequestTrace, Route


def _route(model="a:free", provider="kilo", tools=True, priority=100):
    return Route(provider, model, "free",
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
    store.upsert_routes([_route("old:free"), _route("new:free")], timestamp=100.0)
    store.upsert_routes([_route("new:free")], timestamp=200.0)
    active = [r.model_id for r in store.active_routes()]
    assert active == ["new:free"]
    # still in the table: the history is what detects renames
    row = store._con.execute(
        "SELECT active FROM routes WHERE model_id = 'old:free'").fetchone()
    assert row[0] == 0


def test_a_route_that_comes_back_is_reactivated(store):
    store.upsert_routes([_route("x:free")], timestamp=100.0)
    store.upsert_routes([], timestamp=200.0)
    store.upsert_routes([_route("x:free")], timestamp=300.0)
    assert len(store.active_routes()) == 1


def test_it_deactivates_nothing_when_asked_to_keep(store):
    store.upsert_routes([_route("old:free"), _route("new:free")], timestamp=100.0)
    store.upsert_routes([_route("new:free")], timestamp=200.0, deactivate_missing=False)
    active = sorted(r.model_id for r in store.active_routes())
    assert active == ["new:free", "old:free"]
    row = store._con.execute(
        "SELECT last_seen FROM routes WHERE model_id = 'new:free'").fetchone()
    assert row[0] == 200.0


def test_the_provider_scope_limits_removal_to_that_provider(store):
    # kilo and otro each have an old route. Re-syncing ONLY kilo (with
    # provider="kilo") switches off its old route, but otro's -- older still, and
    # not even mentioned in this call -- must survive: without the scope, an
    # UPDATE not filtered by provider would have switched it off too, because its
    # visto_por_ultima_vez is also older than the new `timestamp`.
    store.upsert_routes([_route("old:free", provider="kilo")], timestamp=50.0)
    store.upsert_routes([_route("old:free", provider="other")], timestamp=50.0)
    store.upsert_routes([_route("new:free", provider="kilo")], timestamp=200.0, provider="kilo")
    active = {r.key for r in store.active_routes()}
    assert active == {"kilo/new:free", "other/old:free"}


def test_the_provider_scope_does_not_change_the_default_behaviour(store):
    # provider=None (the default) preserves the historical behaviour: with no
    # scope, it covers the whole table -- exactly what
    # test_a_route_that_disappears_is_deactivated_not_deleted already covers. This
    # test only confirms that passing provider=None explicitly is the same.
    store.upsert_routes([_route("old:free", provider="kilo"),
                          _route("another:free", provider="other")], timestamp=100.0)
    store.upsert_routes([_route("another:free", provider="other")], timestamp=200.0, provider=None)
    active = {r.key for r in store.active_routes()}
    assert active == {"other/another:free"}   # kilo/old:free is switched off too, as before


# --- Removing a provider from providers.yaml (e.g. openrouter) must not
#     leave its routes at active=1 forever: sync_catalogue can only remove, via
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
        "SELECT active FROM routes WHERE key = 'openrouter/b:free'").fetchone()
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
    # Gate review: a syntactically valid `providers.yaml` with
    # `proveedores: []` (more likely a truncated or badly edited file than a real
    # decision of "no providers") must not switch off the ENTIRE catalogue on the
    # first cycle -- an empty set is treated as "nothing is known yet", not as
    # "everything is orphaned".
    store.upsert_routes([_route("a:free", provider="kilo"),
                          _route("c:free", provider="chatgpt")], timestamp=50.0)
    switched_off = store.deactivate_unregistered_providers(set())
    assert switched_off == 0
    assert {r.key for r in store.active_routes()} == {"kilo/a:free", "chatgpt/c:free"}


def test_quality_averages_the_last_few_quality_probes(store):
    """It used to be the LAST probe alone, which made `quality` as noisy as one
    battery run. Measured live 2026-08-18: the same route scored 6/6, then 0/6,
    then 6/6 on the same case, because a free tier sheds load on long generations.
    With a single sample that swings the score between 1.0 and 0.0 every cycle and
    reintroduces exactly the routing roulette this all started from."""
    store.upsert_routes([_route()], timestamp=100.0)
    store.record_probe("kilo/a:free", "quality", True, 500, 200, 200, 2, 5, 100.0)
    store.record_probe("kilo/a:free", "quality", True, 500, 200, 200, 4, 5, 200.0)
    # (2 + 4) / (5 + 5), not the last one's 4/5.
    assert store.metrics()["kilo/a:free"].quality == pytest.approx(0.6)


def test_one_bad_run_cannot_zero_a_consistently_good_route(store):
    """The whole point: a single shed request must dent the score, not erase it."""
    store.upsert_routes([_route()], timestamp=100.0)
    store.record_probe("kilo/a:free", "quality", True, 500, 200, 200, 5, 5, 100.0)
    store.record_probe("kilo/a:free", "quality", True, 500, 200, 200, 5, 5, 200.0)
    store.record_probe("kilo/a:free", "quality", True, 500, 200, 200, 0, 5, 300.0)
    q = store.metrics()["kilo/a:free"].quality
    assert q == pytest.approx(10 / 15)
    assert q > 0.5, "one shed run must not sink a route that passes twice as often"


def test_the_average_ignores_runs_older_than_the_window(store):
    """Bounded, so a route that genuinely got worse still converges instead of
    being propped up by ancient good runs forever."""
    store.upsert_routes([_route()], timestamp=100.0)
    for i in range(QUALITY_RUNS + 3):        # oldest ones must fall out
        store.record_probe("kilo/a:free", "quality", True, 500, 200, 200, 5, 5,
                           100.0 + i)
    store.record_probe("kilo/a:free", "quality", True, 500, 200, 200, 0, 5, 999.0)
    expected = (5 * (QUALITY_RUNS - 1)) / (5 * QUALITY_RUNS)
    assert store.metrics()["kilo/a:free"].quality == pytest.approx(expected)


def test_the_measurement_time_is_the_most_recent_run_not_an_average(store):
    """`quality_measured_at` answers "when did we last look", which is a real
    moment. Averaging it would invent a timestamp nothing happened at."""
    store.upsert_routes([_route()], timestamp=100.0)
    store.record_probe("kilo/a:free", "quality", True, 500, 200, 200, 5, 5, 100.0)
    store.record_probe("kilo/a:free", "quality", True, 500, 200, 200, 5, 5, 900.0)
    assert store.metrics()["kilo/a:free"].quality_measured_at == 900.0


def test_a_route_with_a_single_run_still_scores_from_it(store):
    store.upsert_routes([_route()], timestamp=100.0)
    store.record_probe("kilo/a:free", "quality", True, 500, 200, 200, 4, 5, 100.0)
    assert store.metrics()["kilo/a:free"].quality == pytest.approx(0.8)


def test_a_skipped_tools_case_does_not_distort_the_average(store):
    """Runs can have different denominators (the tools case is skipped for a route
    that does not declare tools). Summing both sides handles that; averaging the
    per-run ratios would silently weight a 4-case run like a 5-case one."""
    store.upsert_routes([_route()], timestamp=100.0)
    store.record_probe("kilo/a:free", "quality", True, 500, 200, 200, 4, 4, 100.0)
    store.record_probe("kilo/a:free", "quality", True, 500, 200, 200, 0, 5, 200.0)
    assert store.metrics()["kilo/a:free"].quality == pytest.approx(4 / 9)


def test_reliability_mixes_probes_and_events(store):
    store.upsert_routes([_route()], timestamp=100.0)
    store.record_probe("kilo/a:free", "health", True, 100, 50, 200, 0, 0, 100.0)
    store.record_event("kilo/a:free", False, 0, 500, 150.0)
    m = store.metrics()["kilo/a:free"]
    assert 0.0 < m.reliability < 1.0


# --- Task 13 re-review (round 4): a client 4xx no longer punishes the route
#     (round 3), but it was STILL being written as a failed event, and that feeds
#     reliability -- which /health uses to declare a route dead. Reproduced: 26
#     consecutive malformed requests from ONE key are enough to sink EVERY route's
#     reliability, with /health at "down" while a DIFFERENT key keeps receiving
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
    db_path = str(tmp_path / "old_without_flag.sqlite3")
    con = sqlite3.connect(db_path)
    con.executescript(_OLD_SCHEMA_WITHOUT_CLIENT_ERROR_FLAG)
    con.execute(
        """INSERT INTO rutas (clave, proveedor, modelo_id, tier, tools, vision,
               contexto, max_salida, visto_por_ultima_vez, activa, prioridad)
           VALUES ('kilo/old:free','kilo','old:free','gratis',1,0,1000,100,50.0,1,100)""")
    con.execute(
        """INSERT INTO eventos (clave, momento, ok, ttft_ms, codigo_http, latencia_ms)
           VALUES ('kilo/old:free', 60.0, 0, 0, 400, 20)""")
    con.commit()
    con.close()

    store = Storage(db_path)
    store.create_schema()   # must not blow up (ALTER TABLE, not CREATE)

    # The old row, written BEFORE es_error_cliente existed, migrates to 0 (the
    # historical behaviour: it DOES count as a failure) -- an event that never
    # distinguished the cause cannot be reclassified retroactively.
    row = store._con.execute(
        "SELECT is_client_error FROM events WHERE key = 'kilo/old:free'").fetchone()
    assert row[0] == 0

    # And the migrated database is still writable with the new flag.
    store.record_event("kilo/old:free", False, 0, 400, 70.0, is_client_error=True)
    rows = store._con.execute(
        "SELECT is_client_error FROM events WHERE key = 'kilo/old:free' "
        "ORDER BY at").fetchall()
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
    store.record_probe("kilo/a:free", "quality", True, 0, 0, 200, 2, 5, 300.0)
    store.record_probe("kilo/a:free", "quality", True, 0, 0, 200, 4, 5, 900.0)
    m = store.metrics()["kilo/a:free"]
    assert m.quality_measured_at == 900.0
    # The TIME is the last run's; the SCORE averages the window (see
    # QUALITY_RUNS): (2 + 4) / (5 + 5). The two answer different questions.
    assert m.quality == pytest.approx(0.6)


def test_the_last_probe_counts_health_probes_too(store):
    store.upsert_routes([_route()], timestamp=100.0)
    store.record_probe("kilo/a:free", "quality", True, 0, 0, 200, 4, 5, 300.0)
    store.record_probe("kilo/a:free", "health", True, 120, 0, 200, 0, 0, 800.0)
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
    # A `priority` change in the YAML (e.g. moving a provider up) has to
    # propagate on the next sync, not stay stuck to the value the route was first
    # seen with.
    store.upsert_routes([_route("a:free", priority=1)], timestamp=100.0)
    store.upsert_routes([_route("a:free", priority=0)], timestamp=200.0)
    assert store.active_routes()[0].priority == 0


# `_migrate()` already has a case (events.latency_ms, see the header comment
# of storage.py) that adds a column to a table that ALREADY exists using
# `ALTER TABLE ... ADD COLUMN` -- because `CREATE TABLE IF NOT EXISTS` does not
# touch an existing table. This test reproduces the same risk for
# `routes.priority`: production's routes table (the /datos volume) exists from
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
           VALUES ('kilo/old:free','kilo','old:free','gratis',1,0,1000,100,50.0,1)""")
    con.commit()
    con.close()

    # Opening it with the NEW code must not blow up (ALTER TABLE, not CREATE).
    store = Storage(db_path)
    store.create_schema()

    active = store.active_routes()
    assert len(active) == 1
    assert active[0].key == "kilo/old:free"
    # The pre-existing row, with no priority at the time it was written,
    # migrates to the default (100), not to NULL nor to an invented value.
    assert active[0].priority == 100

    # And the migrated database is still writable: a new sync can declare a
    # priority for that same route or for a new one.
    store.upsert_routes([_route("old:free", provider="kilo", priority=0)], timestamp=200.0)
    store.upsert_routes([_route("new:free", provider="chatgpt", priority=0)],
                         timestamp=200.0, deactivate_missing=False)
    active = {r.key: r.priority for r in store.active_routes()}
    assert active == {"kilo/old:free": 0, "chatgpt/new:free": 0}


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


# --- Round 6 review of Task 13, Part 2. Classifying the codes correctly (Part 1)
#     is not enough: `403` is GENUINELY ambiguous -- a suspended account (route
#     evidence) or moderated content (evidence about the REQUEST) -- and the
#     gateway cannot tell them apart without parsing each provider's specific
#     body. Classifying it as route evidence (correct for the first case) leaves it
#     vulnerable to the second: 30 requests with moderated content from ONE client
#     are enough to sink reliability (an AVERAGE of the last 50 observations) for
#     EVERYONE.
#
#     Redesign: /health stops using reliability. It becomes "evidence of life", not
#     "absence of death" -- ONE recent success proves the route serves; a thousand
#     failures from one client do not prove it cannot. `/v1/ranking` KEEPS using
#     reliability exactly as before (untouched): a badly scored route merely loses
#     position and self-corrects, whereas a misinformed /health RESTARTS THE
#     CONTAINER (Coolify) -- the asymmetry is the point. ---

def test_liveness_evidence_with_no_telemetry_at_all(store):
    # A freshly seen route: it was not born dead, it simply has not had its first
    # chance yet, for better or worse.
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
    store.record_probe("kilo/a:free", "health", True, 100, 50, 200, 0, 0, 150.0)
    assert store.has_liveness_evidence("kilo/a:free", now=200.0) is True


# --- Round 10, MEDIUM from the gate: the fallback check ("nothing within the
#     window") looked at ANY real event -- success OR FAILURE -- to decide "there
#     is history, declare it dead". Measured: a single failed real request (fewer
#     than SUSPICION_THRESHOLD, so no on-demand probe ever fires) was enough to
#     drop /health to "down" -- contradicting the module's principle, "a thousand
#     failures from one client do not prove the route is broken". Now ONLY a PROBE
#     (never a real event, success or failure) counts as "there is history". ---

def test_real_failures_alone_without_any_probe_do_not_declare_it_dead(store):
    # The path real traffic can trigger ON ITS OWN (with no probe in between) is
    # no longer enough -- not even with 30 failures.
    store.upsert_routes([_route()], timestamp=100.0)
    for i in range(30):
        store.record_event("kilo/a:free", False, 0, 500, 100.0 + i)
    assert store.has_liveness_evidence("kilo/a:free", now=200.0) is True


def test_real_failures_with_two_failed_probes_confirming_do_declare_it_dead(store):
    # The REAL path to "dead": real traffic triggers suspicion (round 8),
    # suspicion fires probes, and it is TWO consecutive failed probes (round 9)
    # that confirm it -- never the traffic alone.
    store.upsert_routes([_route()], timestamp=100.0)
    for i in range(30):
        store.record_event("kilo/a:free", False, 0, 500, 100.0 + i)
    store.record_probe("kilo/a:free", "health", False, 100, 0, 500, 0, 0, 131.0)
    store.record_probe("kilo/a:free", "health", False, 100, 0, 500, 0, 0, 132.0)
    assert store.has_liveness_evidence("kilo/a:free", now=200.0) is False


def test_liveness_evidence_holds_when_every_failure_is_a_client_error(store):
    # A route that ONLY received malformed requests (400/413/422, Part 1) has not
    # had its first real chance yet -- it is treated the same as "no telemetry",
    # not as "dead".
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
    store.record_probe("kilo/a:free", "health", False, 100, 0, 500, 0, 0, now - 1)
    store.record_probe("kilo/a:free", "health", False, 100, 0, 500, 0, 0, now)
    assert store.has_liveness_evidence("kilo/a:free", now=now) is False


# --- Round 9, a gate finding ("/health's indirect path"): since real traffic can
#     fire ON-DEMAND probes (up to 60/h per route, ~300x more often than the
#     periodic 5h cycle), a client controls WHEN a route is sampled -- more
#     samples, more chances of catching, by pure luck, a transient provider problem
#     in ONE probe, and having /health treat it as a definitive verdict (it
#     survives a container restart). Decision: ONE failed probe alone is NO LONGER
#     enough -- TWO consecutive ones are required, with no success in between. ---

def test_a_single_failed_probe_is_no_longer_enough_to_declare_it_dead(store):
    store.upsert_routes([_route()], timestamp=100.0)
    store.record_probe("kilo/a:free", "health", False, 100, 0, 500, 0, 0, 150.0)
    assert store.has_liveness_evidence("kilo/a:free", now=200.0) is True


def test_two_consecutive_failed_probes_do_declare_it_dead(store):
    store.upsert_routes([_route()], timestamp=100.0)
    store.record_probe("kilo/a:free", "health", False, 100, 0, 500, 0, 0, 140.0)
    store.record_probe("kilo/a:free", "health", False, 100, 0, 500, 0, 0, 150.0)
    assert store.has_liveness_evidence("kilo/a:free", now=200.0) is False


def test_a_failed_probe_right_after_a_success_is_not_enough(store):
    # A single failure preceded by a success is NOT "two consecutive": the older
    # of the last two signals is a success, not another failure.
    store.upsert_routes([_route()], timestamp=100.0)
    store.record_event("kilo/a:free", True, 50, 200, 140.0)
    store.record_probe("kilo/a:free", "health", False, 100, 0, 500, 0, 0, 150.0)
    assert store.has_liveness_evidence("kilo/a:free", now=200.0) is True


# --- Round 7, MEDIUM from the gate: `has_liveness_evidence` only looked at
#     SUCCESSFUL probes -- a probe with `ok=0` was invisible to the function.
#     Reproduced: all five routes dead, the last real success 20h ago (inside the
#     24h window), but FOUR failed health probes since then (one every 5h, the
#     probing default) -- the function found the old success, settled on it, and
#     never looked at what happened AFTERWARDS. The argument that already justifies
#     trusting a SUCCESSFUL probe (the gateway controls its own payload, so there
#     is no possible "this is about the request" ambiguity) applies EQUALLY to a
#     FAILED probe: it is unambiguous evidence that the route is broken, not only
#     that it is alive. The 24h window is defensible for a PAID route that is never
#     probed; it is not for a free route probed every 5h whose last four results
#     are KNOWN and were being discarded. ---

def test_a_failed_probe_newer_than_an_old_success_declares_the_route_dead(store):
    store.upsert_routes([_route()], timestamp=0.0)
    now = 100_000.0
    twenty_hours = 20 * 3600.0
    five_hours = 5 * 3600.0
    store.record_event("kilo/a:free", True, 50, 200, now - twenty_hours)
    # Four FAILED health probes since then, one every 5h -- all of them newer than
    # the success above, the LAST one barely 1h ago.
    for i in range(1, 5):
        store.record_probe("kilo/a:free", "health", False, 100, 0, 500, 0, 0,
                                now - twenty_hours + i * five_hours)
    assert store.has_liveness_evidence("kilo/a:free", now=now) is False


def test_a_real_success_newer_than_a_failed_probe_declares_the_route_alive(store):
    # Symmetric to the one above: if AFTER a failed probe there is a real success
    # (an actual client received a response), that is the newest signal and it
    # wins -- the route is alive NOW, regardless of the probe stumbling
    # earlier.
    store.upsert_routes([_route()], timestamp=0.0)
    store.record_probe("kilo/a:free", "health", False, 100, 0, 500, 0, 0, 100.0)
    store.record_event("kilo/a:free", True, 50, 200, 150.0)
    assert store.has_liveness_evidence("kilo/a:free", now=200.0) is True


def test_a_successful_probe_newer_than_a_failed_one_declares_the_route_alive(store):
    store.upsert_routes([_route()], timestamp=0.0)
    store.record_probe("kilo/a:free", "health", False, 100, 0, 500, 0, 0, 100.0)
    store.record_probe("kilo/a:free", "health", True, 100, 50, 200, 0, 0, 150.0)
    assert store.has_liveness_evidence("kilo/a:free", now=200.0) is True


def test_the_store_knows_when_the_battery_last_ran(store):
    """`cycle` needs this to pace itself on EVIDENCE rather than on an in-memory
    counter -- see probing.cycle for the 28-runs-in-a-day it fixes."""
    store.upsert_routes([_route()], timestamp=100.0)
    assert store.last_quality_probe_at() is None
    store.record_probe("kilo/a:free", "health", True, 10, 0, 200, 0, 0, 500.0)
    assert store.last_quality_probe_at() is None, "a health probe is not a battery run"
    store.record_probe("kilo/a:free", "quality", True, 0, 0, 200, 5, 5, 300.0)
    store.record_probe("kilo/b:free", "quality", True, 0, 0, 200, 5, 5, 900.0)
    assert store.last_quality_probe_at() == 900.0


# --- rate_budgets: what the gateway can infer about an allowance nobody publishes


def _spend(store, key, count, start, step=10.0, capability="chat"):
    """`count` successful requests, `step` seconds apart, from `start`."""
    for i in range(count):
        store.record_event(key, True, 50, 200, start + i * step,
                           capability=capability)
    return start + count * step


def test_an_allowance_is_only_inferred_from_a_refusal_that_stuck(tmp_path):
    """The estimate rests on exhaustions, and a 429 that clears at once is not one.

    Both routes below are refused after the same 5 successes. The difference is
    what happens NEXT: `spent` stays refused for a quarter of an hour, `busy`
    succeeds again three seconds later. Only the first is evidence about a budget
    -- see TRANSIENT_REFUSAL_S for the live measurement that forced the
    distinction.
    """
    store = Storage(str(tmp_path / "d.sqlite3"))
    store.create_schema()
    store.upsert_routes([_route("spent:free"), _route("busy:free")], timestamp=0.0)

    at = _spend(store, "kilo/spent:free", 5, 1000.0)
    store.record_event("kilo/spent:free", False, 0, 429, at)
    store.record_event("kilo/spent:free", True, 50, 200, at + 900.0)

    at = _spend(store, "kilo/busy:free", 5, 1000.0)
    store.record_event("kilo/busy:free", False, 0, 429, at)
    store.record_event("kilo/busy:free", True, 50, 200, at + 3.0)

    budgets = store.rate_budgets(now=2000.0)
    assert budgets["kilo/spent:free"].per_hour == 5
    assert budgets["kilo/spent:free"].measured is True
    assert budgets["kilo/busy:free"].per_hour is None
    assert budgets["kilo/busy:free"].measured is False


def test_an_hourly_estimate_under_the_floor_falls_through_to_the_daily_window(tmp_path):
    """An allowance smaller than a proven floor is usually the wrong WINDOW, not
    an unknowable quota.

    The route sustains 30 requests in one clean hour, so no real quota is under
    30. A later refusal after 3 requests would imply 3/h, which the same data
    refutes. Read over a day the two episodes are one budget of 33, and that IS
    coherent -- which is what the live data showed: grok's imagine agents look
    like 6/h nonsense hourly and like a stable ~60/day daily.
    """
    store = Storage(str(tmp_path / "d.sqlite3"))
    store.create_schema()
    store.upsert_routes([_route()], timestamp=0.0)

    _spend(store, "kilo/a:free", 30, 1000.0, step=60.0)     # a clean, busy hour
    later = 1000.0 + 10 * 3600.0
    at = _spend(store, "kilo/a:free", 3, later)
    store.record_event("kilo/a:free", False, 0, 429, at)
    store.record_event("kilo/a:free", True, 50, 200, at + 900.0)

    b = store.rate_budgets(now=later + 7200.0)["kilo/a:free"]
    assert b.floor >= 30
    assert b.window_s == 86400.0, "the hourly window cannot explain this refusal"
    assert b.allowance == 33
    assert b.measured is True


def test_when_no_window_can_explain_the_refusal_it_stays_unknown(tmp_path):
    """The backstop still has teeth: if EVERY candidate window yields an
    allowance the floor refutes, the honest answer remains that we do not know.

    Here the busy hour sits four days before the refusal, outside even the daily
    window, and the refusal itself follows only two requests.
    """
    store = Storage(str(tmp_path / "d.sqlite3"))
    store.create_schema()
    store.upsert_routes([_route()], timestamp=0.0)

    _spend(store, "kilo/a:free", 40, 1000.0, step=60.0)     # floor = 40/h
    much_later = 1000.0 + 4 * 86400.0
    at = _spend(store, "kilo/a:free", 2, much_later)
    store.record_event("kilo/a:free", False, 0, 429, at)
    store.record_event("kilo/a:free", True, 50, 200, at + 900.0)

    b = store.rate_budgets(now=much_later + 7200.0)["kilo/a:free"]
    assert b.floor >= 40
    assert b.allowance is None
    assert b.remaining is None


def test_an_image_refusal_does_not_shrink_the_chat_allowance(tmp_path):
    """Allowances belong to a resource, and chat is not image generation.

    This is the failure that was actually measured against production: grok's
    image agents refuse images while chat stays abundant, and counting the two
    together produced allowances below what the route was already sustaining.
    """
    store = Storage(str(tmp_path / "d.sqlite3"))
    store.create_schema()
    store.upsert_routes([_route("agent:free")], timestamp=0.0)
    key = "kilo/agent:free"

    _spend(store, key, 40, 1000.0, step=60.0)               # plenty of chat
    at = _spend(store, key, 2, 1000.0, step=5.0, capability="images")
    store.record_event(key, False, 0, 429, at, capability="images")
    store.record_event(key, True, 50, 200, at + 900.0, capability="images")

    chat = store.rate_budgets(now=5000.0, capability="chat")[key]
    images = store.rate_budgets(now=5000.0, capability="images")[key]
    assert chat.per_hour is None, "an image refusal says nothing about chat"
    assert chat.floor >= 40
    assert images.per_hour == 2


def test_remaining_and_exhaustion_are_silent_until_the_allowance_is_known(tmp_path):
    """Before the first measurement the honest answer is None, not a guess."""
    store = Storage(str(tmp_path / "d.sqlite3"))
    store.create_schema()
    store.upsert_routes([_route("a:free")], timestamp=0.0)
    _spend(store, "kilo/a:free", 4, 1000.0)

    b = store.rate_budgets(now=1100.0)["kilo/a:free"]
    assert b.per_hour is None
    assert b.remaining is None
    assert b.exhausts_in_s is None
    assert b.used == 4          # consumption is always known, even unmeasured


def test_remaining_counts_down_and_projects_when_the_allowance_is_known(tmp_path):
    """Once measured, `remaining` and `exhausts_in_s` answer the operator's
    question: how much is left, and how long that lasts at this rate."""
    store = Storage(str(tmp_path / "d.sqlite3"))
    store.create_schema()
    store.upsert_routes([_route("a:free")], timestamp=0.0)
    key = "kilo/a:free"

    at = _spend(store, key, 10, 1000.0)
    store.record_event(key, False, 0, 429, at)
    store.record_event(key, True, 50, 200, at + 900.0)

    # A fresh hour with 4 of the 10 spent.
    later = at + 10 * 3600.0
    _spend(store, key, 4, later)
    b = store.rate_budgets(now=later + 1800.0)[key]

    assert b.per_hour == 10
    assert b.used == 4
    assert b.remaining == 6
    # 4 requests in the trailing hour -> 6 left lasts an hour and a half.
    assert b.exhausts_in_s == pytest.approx(6 / (4 / 3600.0))


def test_consumption_past_the_estimate_reports_none_left_not_a_negative():
    """The estimate is a median over episodes, so consumption CAN pass it -- and
    "none left" is the truthful reading of that, never a negative budget.

    Constructed directly rather than through the estimator: a history where the
    route both serves more than its allowance and keeps a clean hour to prove it
    is self-contradictory, and the coherence backstop rightly refuses to report
    an allowance for it. The clamp being asserted here belongs to the value, not
    to the inference.
    """
    b = RateBudget(allowance=5, window_s=3600.0, used=9, floor=5, episodes=2)
    assert b.remaining == 0
    assert b.exhausts_in_s == 0


# --- traffic: what the gateway DID with client requests, not what it believes


def _attempt(store, rid, requested, key, ok, attempt, at):
    store.record_event(key, ok, 50 if ok else 0, 200 if ok else 503, at,
                       trace=RequestTrace(request_id=rid, requested=requested),
                       attempt=attempt)


def test_a_failover_chain_counts_as_one_request_not_as_three(tmp_path):
    """The whole reason `request_id` exists.

    Three rows land against three routes, but a client asked ONCE. Counting rows
    would report three requests and a 33% success rate for a request that
    succeeded. `attempt` orders the chain so the last one is the outcome.
    """
    store = Storage(str(tmp_path / "d.sqlite3"))
    store.create_schema()
    _attempt(store, "r1", "auto:strong", "chatgpt/gpt-5:free", False, 1, 100.0)
    _attempt(store, "r1", "auto:strong", "deepseek/deepseek-chat", False, 2, 101.0)
    _attempt(store, "r1", "auto:strong", "kilo/a:free", True, 3, 102.0)

    t = store.traffic(since=0.0)
    assert t["requests"] == 1
    assert t["attempts"] == 3
    assert t["needed_failover"] == 1
    # The chain is credited to the route that CLOSED it...
    assert t["served"] == {"kilo/a:free": {"ok": 1, "failed": 0}}
    # ...and the two it fell away from are named, which is the actionable half.
    assert t["fell_away_from"] == {"chatgpt/gpt-5:free": 1,
                                   "deepseek/deepseek-chat": 1}


def test_traffic_maps_what_was_asked_to_what_actually_served_it(tmp_path):
    """The question the whole feature exists for: is auto:strong landing where
    you think it is?"""
    store = Storage(str(tmp_path / "d.sqlite3"))
    store.create_schema()
    for i in range(3):
        _attempt(store, f"s{i}", "auto:strong", "chatgpt/gpt-5:free", True, 1, 100.0 + i)
    _attempt(store, "s9", "auto:strong", "kilo/a:free", True, 1, 110.0)
    _attempt(store, "f1", "auto:fast", "kilo/a:free", True, 1, 120.0)

    t = store.traffic(since=0.0)
    assert t["by_requested"]["auto:strong"] == {"chatgpt/gpt-5:free": 3,
                                                 "kilo/a:free": 1}
    assert t["by_requested"]["auto:fast"] == {"kilo/a:free": 1}
    assert t["needed_failover"] == 0


def test_probes_are_not_counted_as_client_traffic(tmp_path):
    """Probes carry no trace by design -- they are the gateway asking, not a
    client. Counting them would inflate exactly the ratios this measures."""
    store = Storage(str(tmp_path / "d.sqlite3"))
    store.create_schema()
    _attempt(store, "r1", "auto:strong", "kilo/a:free", True, 1, 100.0)
    for i in range(20):                      # a probing pass, no trace
        store.record_event("kilo/a:free", True, 50, 200, 101.0 + i)

    t = store.traffic(since=0.0)
    assert t["requests"] == 1
    assert t["attempts"] == 1


def test_traffic_honours_its_window(tmp_path):
    store = Storage(str(tmp_path / "d.sqlite3"))
    store.create_schema()
    _attempt(store, "old", "auto:strong", "kilo/a:free", True, 1, 100.0)
    _attempt(store, "new", "auto:strong", "kilo/a:free", True, 1, 900.0)

    assert store.traffic(since=0.0)["requests"] == 2
    assert store.traffic(since=500.0)["requests"] == 1


def test_the_store_knows_when_the_catalogue_was_last_swept(store):
    """`cycle` needs a marker for the SWEEP, not for any individual probe.

    The proxy fires its own on-demand health probes (proxy._probe_on_demand) and
    those land in the same table with the same `kind`. Reading MAX(at) would let
    one suspicion probe against one route pass for "the whole catalogue was just
    swept", and suppress the periodic sweep for a full interval.
    """
    store.upsert_routes([_route()], timestamp=100.0)
    assert store.last_health_sweep_at() is None
    store.record_probe("kilo/a:free", "health", True, 10, 0, 200, 0, 0, 500.0)
    assert store.last_health_sweep_at() is None, \
        "one route's probe is not a sweep of the catalogue"
    store.mark_health_sweep(900.0)
    assert store.last_health_sweep_at() == 900.0
    store.mark_health_sweep(1800.0)
    assert store.last_health_sweep_at() == 1800.0, "the marker moves forward, it does not pile up"


def test_the_new_capability_axes_survive_a_round_trip(store):
    caps = Capabilities(tools=False, vision=True, context=52815, max_output=8192,
                        images=True, audio_speech=True, audio_transcription=True,
                        translate=True, search=True)
    store.upsert_routes([Route("chatgpt", "gpt-5-6", "free", caps)], 100.0)
    assert store.active_routes()[0].capabilities == caps


def test_an_old_database_migrates_the_new_columns_to_false(store):
    # A row written before these columns existed predates anyone measuring the
    # capability, so it migrates to "cannot", never to a guess. The next sweep
    # overwrites it with what the provider actually reports.
    store._con.execute("ALTER TABLE routes DROP COLUMN audio_speech")
    store._con.execute("ALTER TABLE routes DROP COLUMN audio_transcription")
    store._con.execute("ALTER TABLE routes DROP COLUMN translate")
    store._con.execute("ALTER TABLE routes DROP COLUMN search")
    store._con.execute(
        "INSERT INTO routes (key, provider, model_id, tier, tools, vision, "
        "context, max_output, last_seen) VALUES "
        "('chatgpt/old', 'chatgpt', 'old', 'free', 0, 0, 1000, 100, 50.0)")
    store._con.commit()
    store.create_schema()
    route = [r for r in store.active_routes() if r.model_id == "old"][0]
    assert route.capabilities.audio_speech is False
    assert route.capabilities.search is False
