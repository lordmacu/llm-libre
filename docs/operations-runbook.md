# Operations runbook

This is the part a future operator actually needs at 2am, not a restatement
of the API contract. If you are debugging routing decisions rather than an
outage, see [`routing-and-ranking.md`](routing-and-ranking.md) instead --
this doc assumes you already know what a cooldown or a score is and links
back there for the mechanics.

## The SQLite file: where it lives, and what losing it costs

- **Location**: `RUTA_DB`, default `/datos/llm-libre.sqlite3`. In the
  documented Coolify deployment this path must be a **persistent volume**
  -- without one, a redeploy wipes the container's filesystem and the file
  goes with it.
- **What it holds**: the discovered route catalog (`rutas`), every health
  and quality probe ever run (`sondas`), every real request's outcome
  (`eventos`), and the per-key daily paid-usage counters (`uso_pago`).
  Cooldowns and in-flight suspicion counters are **not** in here -- they
  live in `Proxy`'s process memory and reset on every restart regardless of
  the volume.
- **What losing it costs**: nothing catastrophic and nothing instant, but
  it is expensive to earn back. The route catalog rebuilds on the next sync
  cycle (up to `SONDEO_SALUD_HORAS`, default 5h, worst case). Every route's
  measured quality resets to the neutral assumption (`calidad_medida:
  false` everywhere in `GET /v1/ranking`) until the quality battery runs
  again on each one -- and that battery only runs roughly once a day
  (`SONDEO_CALIDAD_CADA_N_CICLOS`, default 5 health cycles), so a truly
  informed ranking takes **days** to rebuild, not minutes. In the meantime
  routing still works (unmeasured routes are still reachable, just
  deprioritized below measured ones -- see the routing doc), it is just
  flying without the accumulated data that normally breaks ties well.
  Today's paid-usage counters also reset, which is harmless (a fresh day of
  allowance) unless it happens to line up with a real attempt to stay under
  a budget across a redeploy.
- **Practical implication**: treat the volume mount as load-bearing.
  Losing it is not an outage, but it silently degrades routing quality for
  days, which is a much easier failure mode to miss than a hard error.

## `GET /health`: the three states, and what each one implies

`/health` is deliberately honest -- it checks for a route that could
*actually* serve a request right now, not just that the process is up (see
the design spec §6 for the incident that motivated this). It is meant to
be wired as the deployment's health check.

| `estado` | HTTP | Meaning | What to do |
|---|---|---|---|
| `ok` | `200` | At least one **free** route is alive. | Normal operation. |
| `degradado` | `503` | No free route is alive; the paid route still is. | Every request is now either failing over onto a bill or, if `x_permitir_pago: false` was set, failing outright. Go find out why the free tier is down (start with `GET /v1/ranking`'s `en_cooldown_hasta` column and the provider status of `chatgpt` / Kilo / OpenRouter) before the daily paid allowance runs out for every key relying on this gateway. |
| `caido` | `503` | Nothing is alive, free or paid. | `POST /v1/chat/completions` is returning `503` for every request. This is the real outage state. |

A route needs **two consecutive failed health probes** (with no success in
between, and counting only probes, never raw client failures -- see the
design spec §6 for why) before it is counted as dead here. This means
`/health` can lag a real recovery or a real outage by up to one probe
interval, but it also means a single transient blip -- made more likely
by the fact that a client's own failing traffic can indirectly trigger an
extra on-demand probe, sampling the provider more often than the fixed
periodic schedule would -- does not flip the whole service to `caido` and
trigger a container restart over nothing.

Because Coolify uses this endpoint as its health check and restarts the
container on failure, and the telemetry backing this decision lives in the
persistent volume (not process memory), **a container restart caused by a
real `caido` will not fix it** -- the new process reads the same evidence
from disk and reports the same thing. Restarting only helps if the failure
was actually in the process itself, which `/health` failing does not, by
itself, tell you.

## Everything is returning `503` from `/v1/chat/completions` -- where to look

`503` here specifically means "candidate routes exist, but none can serve
this request right now" (as opposed to `400`, which means no route could
*ever* satisfy the request -- an unwinnable capability combination, not an
outage). In rough order of what to check:

1. **`GET /health`.** `caido` confirms a real, total outage; `degradado`
   narrows it to "the free tier specifically is down, the paid one is
   still up but may have its own cap." `ok` here while a specific request
   still gets `503` usually means that request's own filters (capability,
   `x_min_contexto`) leave it with a narrower candidate set than "any free
   route," and that narrower set happens to be exhausted -- check `GET
   /v1/ranking` for the `tools`/`vision`/`contexto` columns of the routes
   that would have qualified.
2. **`GET /v1/ranking`, look at `en_cooldown_hasta` across the board.** If
   most or all rows show a nonzero, still-future timestamp, most of the
   catalog is punished right now -- see
   [Cooldowns](routing-and-ranking.md#cooldowns-what-they-are-and-how-they-clear)
   for what set them and when they clear on their own. The response body
   of the `503` itself also carries `proxima_liberacion`: the earliest
   moment any currently-cooling candidate frees up, `null` if nothing
   relevant is even in cooldown (i.e. the problem is something else, like
   the paid cap below).
3. **`GET /v1/uso`** for the calling key, and compare `pago_hoy` against
   `tope`. If the free tier is genuinely down and the key already spent its
   daily paid allowance, that specific key gets `503` with
   `tope_pago_alcanzado: true` in the error body -- deliberately, so an
   exhausted budget never turns into a silent, unexpected charge. A
   *different* key with allowance left would not see this.
4. **`chatgpt` specifically.** It has `prioridad: 0`, so it is the first
   route tried on essentially every free-tier request; it also runs
   against a service on a machine (`blog`) that is known to run saturated.
   If it is slow or down, expect it to show up in cooldown or with
   depressed `confiabilidad` before other providers do. Check the
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
`SONDEO_SALUD_HORAS`, default 5h, per free route), the periodic quality
battery (roughly once a day, up to 5 requests per free route, one per
battery case), and the on-demand probes the gateway fires itself when
client traffic raises suspicion about a route (rate-limited, but able to
add meaningfully more request volume against a route under sustained
failures -- see the routing doc's cooldowns section). None of this is
metered or capped against a separate "probing budget" distinct from the
gateway's normal free-tier usage; it draws from the exact same pool real
traffic does. On a catalog the size of the reference deployment, periodic
probing alone works out to on the order of 100 requests/day (see the
design spec's known risks) -- factor that in before turning the sondeo
cadence up, and remember that a provider-side rate limit hit by a probe
looks, and is handled, exactly like one hit by a client.
