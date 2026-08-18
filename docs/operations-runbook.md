# Operations runbook

This is the part a future operator actually needs at 2am, not a restatement
of the API contract. If you are debugging routing decisions rather than an
outage, see [`routing-and-ranking.md`](routing-and-ranking.md) instead --
this doc assumes you already know what a cooldown or a score is and links
back there for the mechanics.

## The SQLite file: where it lives, and what losing it costs

- **Location**: `DB_PATH`, default `/datos/llm-libre.sqlite3`. In the
  documented Coolify deployment this path must be a **persistent volume**
  -- without one, a redeploy wipes the container's filesystem and the file
  goes with it.
- **What it holds**: the discovered route catalog (`routes`), every health
  and quality probe ever run (`sondas`), every real request's outcome
  (`events`), and the per-key daily paid-usage counters (`paid_usage`).
  Cooldowns and in-flight suspicion counters are **not** in here -- they
  live in `Proxy`'s process memory and reset on every restart regardless of
  the volume.
- **What losing it costs**: less than it sounds like, and faster to earn
  back than "up to `HEALTH_PROBE_HOURS`" would suggest -- **the planificador
  runs its first full probing cycle immediately on startup, before its first
  sleep**, not after `HEALTH_PROBE_HOURS`. Concretely: `planificador`'s
  loop counter starts at `0`, and `cycle(state, 0)` -- catalog sync, a
  health probe against every free route, AND (because `0 %
  QUALITY_PROBE_EVERY_N_CYCLES == 0` for any sane value of that setting)
  the FULL quality battery against every free route -- runs right away,
  concurrently with the process becoming ready to serve traffic, on every
  single restart. Verified directly: a fresh `Storage`/`Proxy` pair running
  `probing.cycle(state, 0)` once produces both `probes.kind='health'` AND
  `probes.kind='quality'` rows for every free route in that one pass. So
  after losing the file (or after ANY restart, empty database or not): the
  catalog and a first quality measurement for every free route are both
  back within roughly one probing pass -- seconds to low minutes, bounded by
  how long that many HTTP round-trips take, not by `HEALTH_PROBE_HOURS`.
  What genuinely takes longer to rebuild is the **history**: `reliability`
  and `ttft_p50_ms` are computed over a rolling window of past probes and
  real traffic (see the routing doc), so a fresh database starts every
  route at the neutral assumption for those two and only converges to a
  representative measurement after several cycles' worth of data
  accumulates -- that part is closer to days than minutes, even though the
  catalog and a first quality score are not. Today's paid-usage counters
  also reset, which is harmless (a fresh day of allowance) unless it
  happens to line up with a real attempt to stay under a budget across a
  redeploy.
- **The flip side of "runs the full battery on every restart"**: it is not
  free. See [below](#probes-spend-the-same-free-quota-real-traffic-uses)
  for what one full pass costs in free-tier quota -- a restart-happy
  deployment (or a Coolify crash-loop) burns through a meaningful chunk of
  a day's probing budget every time it comes back up, on top of whatever
  triggered the restart in the first place.
- **Practical implication**: treat the volume mount as load-bearing
  regardless -- losing it does not cost what an earlier version of this
  doc claimed (days before routing looks sane again), but it does erase
  the reliability/latency history that makes ranking decisions stable
  under real traffic patterns, and that part really does take a while to
  rebuild.

## `GET /health`: the three states, and what each one implies

`/health` is deliberately honest -- it checks for a route that could
*actually* serve a request right now, not just that the process is up (see
the design spec §6 for the incident that motivated this). It is meant to
be wired as the deployment's health check.

| `status` | HTTP | Meaning | What to do |
|---|---|---|---|
| `ok` | `200` | At least one **free** route is alive. | Normal operation. |
| `degraded` | `503` | No free route is alive; the paid route still is. | Every request is now either failing over onto a bill or, if `x_allow_paid: false` was set, failing outright. Go find out why the free tier is down (start with `GET /v1/ranking`'s `cooldown_until` column and the provider status of `chatgpt` / Kilo) before the daily paid allowance runs out for every key relying on this gateway. |
| `down` | `503` | Nothing is alive, free or paid. | `POST /v1/chat/completions` is returning `503` for every request. This is the real outage state. |

A route needs **two consecutive failed health probes** (with no success in
between, and counting only probes, never raw client failures -- see the
design spec §6 for why) before it is counted as dead here. This means
`/health` can lag a real recovery or a real outage by up to one probe
interval, but it also means a single transient blip -- made more likely
by the fact that a client's own failing traffic can indirectly trigger an
extra on-demand probe, sampling the provider more often than the fixed
periodic schedule would -- does not flip the whole service to `down` and
trigger a container restart over nothing.

⚠️ **`HEALTH_PROBE_HOURS` has a soft ceiling that `configuration.md` does
not mention: `VENTANA_EVIDENCIA_VIDA_S`, hardcoded to 24h.** `/health`
only looks for a signal (a real success, or any health probe result)
within the trailing 24h; outside that window it falls back to "has this
route ever been probed at all" -- and once a route HAS been probed at
least once, ever, that fallback stops treating it as "no evidence yet" and
starts treating a stale signal as **no current evidence**, i.e. dead.
Verified directly: a route whose only recorded signal is a *successful*
health probe from 26h ago (simulating `HEALTH_PROBE_HOURS` set at or
above roughly 24-26h, with no real traffic reaching that route in the
meantime to refresh the signal) is reported as having **no** evidence of
life -- despite that signal being a success, not a failure. Set
`HEALTH_PROBE_HOURS` at or anywhere near 24h and, for any route that is
not receiving a steady trickle of real traffic to independently refresh
its evidence between probes, expect `/health` to eventually flip that
route to dead purely from the probe cadence being too slow for the fixed
evidence window -- not from anything actually being wrong with the route.
Keep `HEALTH_PROBE_HOURS` well under 24h (the default, 5h, has roughly 4x
headroom); if the periodic cadence ever needs to go that high for
quota reasons, the on-demand probe mechanism (fired from real traffic
between periodic cycles, see the routing doc) is the intended way to keep
evidence fresher than the periodic schedule alone would.

Because Coolify uses this endpoint as its health check and restarts the
container on failure, whether that restart actually helps depends on
**why** `/health` is `down` -- and the two causes behave differently
enough that it is worth checking before assuming either way:

- **If every route that would otherwise be alive happens to be in an
  active cooldown right now**, a restart genuinely fixes it, immediately:
  cooldowns live only in `Proxy`'s process memory (see above), so a fresh
  process starts with none of them, and every route not independently
  marked dead (next bullet) is eligible again the instant the process is
  up. Verified directly: force a route's `cooldown_until` into the
  future against a database, then open a **second, independent** `Proxy`
  (simulating a restart) against that same file -- the new process's
  `cooldowns` dict is empty, because that state was never in the database
  to begin with.
- **If `Storage.tiene_evidencia_de_vida` genuinely has no positive evidence
  for a route** (two consecutive failed health probes, or nothing within
  the 24h evidence window at all -- see the design spec §6), a restart
  does **not** help: that evidence lives in the persisted `sondas` /
  `events` tables, on the volume, and a fresh process reading the same
  database reaches the same conclusion `/health` did before the restart.

In practice a restart is often at least a **partial**, temporary fix even
in the second case: it clears every cooldown outright, and (see
[above](#the-sqlite-file-where-it-lives-and-what-losing-it-costs)) a fresh
process immediately re-runs a full probe cycle against every free route on
startup, which gives every route a fresh, real chance to record a success.
It is not, however, a **guaranteed** fix for a genuine, sustained provider
outage -- if the outage is still happening when those fresh probes run,
they fail too, and `/health` reports `down` again.

## Everything is returning `503` from `/v1/chat/completions` -- where to look

`503` here specifically means "candidate routes exist, but none can serve
this request right now" (as opposed to `400`, which means no route could
*ever* satisfy the request -- an unwinnable capability combination, not an
outage). In rough order of what to check:

1. **`GET /health`.** `down` confirms a real, total outage; `degraded`
   narrows it to "the free tier specifically is down, the paid one is
   still up but may have its own cap." `ok` here while a specific request
   still gets `503` usually means that request's own filters (capability,
   `x_min_context`) leave it with a narrower candidate set than "any free
   route," and that narrower set happens to be exhausted -- check `GET
   /v1/ranking` for the `tools`/`vision`/`context` columns of the routes
   that would have qualified.
2. **`GET /v1/ranking`, look at `cooldown_until` across the board.** If
   most or all rows show a nonzero, still-future timestamp, most of the
   catalog is punished right now -- see
   [Cooldowns](routing-and-ranking.md#cooldowns-what-they-are-and-how-they-clear)
   for what set them and when they clear on their own. The response body
   of the `503` itself also carries `next_release`: the earliest
   moment any currently-cooling candidate frees up, `null` if nothing
   relevant is even in cooldown (i.e. the problem is something else, like
   the paid cap below).
3. **`GET /v1/usage`** for the calling key, and compare `paid_today` against
   `cap`. If the free tier is genuinely down and the key already spent its
   daily paid allowance, that specific key gets `503` with
   `tope_pago_alcanzado: true` in the error body -- deliberately, so an
   exhausted budget never turns into a silent, unexpected charge. A
   *different* key with allowance left would not see this.
4. **`chatgpt` specifically.** It has `priority: 0`, so it is the first
   route tried on essentially every free-tier request; it also runs
   against a service on a machine (`blog`) that is known to run saturated.
   If it is slow or down, expect it to show up in cooldown or with
   depressed `reliability` before other providers do. Check the
   `chatgpt-proxy` service directly if `GET /v1/ranking` points there.
5. **Kilo's anonymous tier.** It is explicitly a courtesy, not a contract
   (see the design spec's known risks) -- if it stops accepting anonymous
   requests, every `kilo/*` route starts failing until an operator sets
   `KILO_API_KEY` (see [`configuration.md`](configuration.md)).

## Probes spend the same free quota real traffic uses

This is easy to forget and worth stating plainly: **every probe the
gateway sends to a free provider is a real request against that
provider's free-tier quota**, indistinguishable from traffic a client
sent. This includes the periodic health probe (roughly every
`HEALTH_PROBE_HOURS`, default 5h, per free route), the periodic quality
battery (roughly once a day, up to 5 requests per free route, one per
battery case), and the on-demand probes the gateway fires itself when
client traffic raises suspicion about a route (rate-limited, but able to
add meaningfully more request volume against a route under sustained
failures -- see the routing doc's cooldowns section). None of this is
metered or capped against a separate "probing budget" distinct from the
gateway's normal free-tier usage; it draws from the exact same pool real
traffic does. On a catalog the size of the reference deployment, periodic
probing alone works out to on the order of 100 requests/day (see the
design spec's known risks) -- factor that in before turning the probing
cadence up, and remember that a provider-side rate limit hit by a probe
looks, and is handled, exactly like one hit by a client.
