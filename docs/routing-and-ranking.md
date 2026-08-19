# Routing and ranking

This is the part the code does not say out loud: **why** a given request
ends up at the route it ends up at, and how to read the one endpoint built
to answer that question, `GET /v1/ranking`.

## Three words that look interchangeable and are not

The codebase (and this doc) uses three words that are easy to conflate.
Mixing any two of them up will make you misread `GET /v1/ranking` sooner or
later, so get them straight first:

| Word | Values | What it answers | Lives on |
|---|---|---|---|
| **tier** | `free` \| `paid` | Does this route cost money? | `Route.tier` / `Provider.tier` |
| **perfil** | `fast` \| `balanced` \| `strong` | What does *this request* prefer (speed vs. measured quality)? | `Pedido.perfil` (set by the `model` alias you asked for) |
| **priority** | any integer, default `100` | Manual, operator-set ordering: which route gets tried first *before* anyone looks at a score | `Route.priority` / `Provider.priority` (from `providers.yaml`) |

They never override each other. In particular: **a paid route with
`priority: 0` still goes last.** `priority` only breaks ties *within* a
tier; it can never buy a paid route a place ahead of a free one. The
production config sets this up on purpose: `chatgpt` (a self-hosted
provider) has `priority: 0` so it is tried before third-party `kilo`
(`priority: 1`) -- but it is still `tier: free`, so this has nothing to
do with money, and `minimax` (`tier: paid`, `priority: 2`) is still tried
dead last regardless of that low-looking number.

## How the router orders candidates

`router.ordenar()` runs on **every** request, including one that names an
explicit real model id -- there is no shortcut that skips it. What an
explicit id changes is narrower than it sounds: `router.compatibles()`
adds one extra filter (`r.model_id == request.model`) on top of the same
capability/context checks every request goes through, and the request
still has to clear the same cooldown filter afterward. Concretely: an
explicit id combined with an impossible `x_min_context` still gets `400`
(step 1 below), and an explicit id whose only matching route happens to be
cooling down right now still gets `503` (step 2), exactly like `auto`
would. The steps, in order:

> **An explicit id buys you no failover, in practice none at all.** The
> filter keeps every route serving that id, and in principle several
> providers could serve the same one — but measured against this deployment
> on 2026-08-18, **not one of the 52 model ids is served by more than one
> provider**. So naming an id gives a chain of exactly ONE route: the first
> failure is the last, and it is returned as a bare `503` while `auto`
> would have walked 40+ alternatives. This bites hardest on a long prompt,
> because the failure is usually the provider *timeout* firing —
> `deepseek` declares `timeout_s: 60` and `grok` declares none and so takes
> the global `TIMEOUT_S` of 90s, which is exactly where 503s at 60s and 90s
> come from. The same ids answer a short prompt in ~2s. Check
> `error.routes_tried` in the 503 body to tell "one route, no failover"
> apart from "everything is down".

1. **Filter by capability.** A route is a candidate at all only if it
   satisfies what the request requires: `tools` / `vision` if requested
   (via `x_requires`, an `auto:tools` / `auto:vision` alias, or simply
   including a non-empty `tools` array in the request body -- see the
   OpenAPI docs at `/docs` for that last one, it is easy to miss), a
   context window at least as large as `x_min_context` if set, and (for an
   explicit id) that the route's `modelo_id` matches exactly. If nothing in
   the whole catalog can ever satisfy this, the request gets `400`
   immediately -- this filter runs before availability is even considered.
2. **Filter out anything currently in cooldown.** A route that is punished
   right now (see [cooldowns](#cooldowns-what-they-are-and-how-they-clear)
   below) is removed from the list entirely for this request. If that
   empties the list, the request gets `503` (routes exist, none are usable
   *right now*) -- a different failure from step 1's `400` (no route could
   *ever* work).
3. Sort what is left by, in this exact order:
   1. **`tier == "paid"` first.** Free before paid, always, full stop --
      this is the one invariant nothing below it can break.
   2. **`priority`, ascending**, within the same tier.
   3. **Whether the route has ever been measured by the quality battery.**
      A route with no measurement yet ranks *below* every measured route at
      the same priority -- but it still stays in the list, reachable, so it
      eventually gets a turn and gets measured. It is not penalized to the
      point of starvation, just deprioritized against known quantities.
   4. **The score**, `score = quality^wc * reliability^wr * f(latency)^wl`,
      highest first. The exponents (`wc`, `wr`, `wl`) come from the
      requested `profile` -- see `llm_libre/ranking.py`'s `WEIGHTS` table.
      `x_allow_paid: false` on the request removes paid routes from
      consideration even earlier, before this sort even runs.

The result is the full ordered attempt chain, not just a single winner --
`Proxy` walks down it on failure (see the runbook for what "failure" means
here). Only step 1's outcome is visible to a client as an HTTP status
(`400`); everything from step 2 onward is invisible unless you go looking,
which is what the rest of this document is for.

### The draw between tied routes, and why `strong` does not take part

Routes that score within `TIE_BAND` (5%) of the best in their category are
**shuffled at random** before the chain is built, so that traffic spreads
across genuinely interchangeable routes instead of hammering whichever one
sorts first and rate-limiting it. The draw can never cross a category, so
it cannot lift a paid route above a free one or override `priority` -- see
`router.shuffle_ties`.

`strong` is the exception: its band is **0.0** (`TIE_BAND_BY_PROFILE`), so
only an *exact* tie is drawn. The reason is that the wide band and the
meaning of `strong` contradict each other. `fast` and `balanced` are saying
"these routes are interchangeable, spread the load"; `strong` is saying
"give me the best route you can identify", and answering that with a
uniform random pick from everything within 5% is how the profile turned
into a coin flip. Measured on 2026-08-18 against this deployment, three
routes sat inside the band for `strong` -- one of them a 2.6B model -- and
consecutive identical requests landed on different ones, which is exactly
what a consumer sees as "the same prompt gives a great answer or a generic
one at random". An exact tie is still drawn, so identical routes do not
starve.

## Reading `GET /v1/ranking`

This is the operator's main debugging tool: "why did `X-Route-Used` say
what it said?" `GET /v1/ranking` returns one row per active route, **sorted
with the literal same key the router itself uses** (`router.sort_key`,
the same function `ordenar()` calls) -- so the row at the top of this table
is, right now, genuinely the route a fresh `auto` request would try first.
That was not always true: an earlier version sorted by score alone, and
could show a route at the top of the table while `X-Route-Used` reported a
completely different one, because the display ignored `priority` and
cooldown state. If you ever see that again, treat it as a regression.

### Worked example

Two real routing decisions, observed against the live deployment:

- `POST /v1/chat/completions` with `"model": "auto"` was served by
  **`chatgpt/gpt-5-3-mini`** (`X-Tier: free`, `X-Attempts: 1`, ~4.5s).
- The exact same prompt with `"model": "auto:tools"` fell through to
  **`kilo/cohere/north-mini-code:free`**, because every `chatgpt` route
  declares `tools: false` in `providers.yaml` (the anonymous backend does
  not support real function calling -- see the design spec for how that was
  verified) and so gets filtered out by step 1 above.

An illustrative `GET /v1/ranking` snapshot consistent with that (field
values below are for illustration -- your own deployment's exact scores and
timestamps will differ, but the shape and the reasoning are real):

```json
{
  "routes": [
    {
      "key": "chatgpt/gpt-5-3-mini", "tier": "free", "priority": 0,
      "score": 0.7912, "quality": 0.8, "quality_measured": true,
      "reliability": 0.98, "ttft_p50_ms": 900.0, "latency_p50_ms": 4500.0,
      "cooldown_until": 0.0, "tools": false, "vision": false, "context": 128000
    },
    {
      "key": "kilo/cohere/north-mini-code:free", "tier": "free", "priority": 1,
      "score": 0.8420, "quality": 0.8, "quality_measured": true,
      "reliability": 1.0, "ttft_p50_ms": 650.0, "latency_p50_ms": 2100.0,
      "cooldown_until": 0.0, "tools": true, "vision": false, "context": 128000
    }
  ]
}
```

Reading it top to bottom:

- **`chatgpt/gpt-5-3-mini` sits above `kilo/...` despite a *lower* `score`
  (0.79 vs. 0.84).** This is `priority` doing its job: `0 < 1`, and
  `priority` is compared before the score. This is exactly why `auto`
  picked `chatgpt` for the plain request above.
- **`auto:tools` skips `chatgpt` entirely** -- not because of ordering, but
  because `"tools": false` on that row fails the capability filter (step 1)
  before ordering is even considered. `kilo/...`, with `"tools": true`, is
  the next candidate in tier `free`, so it wins for that request.
- If `chatgpt/gpt-5-3-mini` were punished right now, its row would still
  appear in this table (it is diagnostic, not filtered by cooldown) but
  would sort to the **bottom** regardless of its score or `priority` --
  `cooldown_until` is the first element of the sort key, ahead of
  everything else. Check that field whenever a route you expected to win
  did not.

### Field notes worth knowing before you stare at this table

- **`quality` is `null`, and the number you'd expect is in `quality_assumed`
  instead, whenever `quality_measured` is `false`.** A route that has never
  been through the quality battery is not silently shown as if a real
  measurement of `0.6` existed -- that would make an assumption look like
  data. It still counts toward the score (as the neutral value), it is just
  labeled honestly.
- **`cooldown_until` is a raw Unix timestamp in seconds** (`0` = not in
  cooldown) -- unlike `last_probe` and `last_quality_probe`, which are
  ISO-8601 strings (or `null` if that probe never ran). This asymmetry is
  real, not a typo in this doc; convert it yourself if you need a human
  time. Production runs on Ubuntu (GNU coreutils): `date -d @<value>`.
  (`date -r <value>` is the macOS/BSD form -- handy for local development
  on a Mac, but it does not run as-is on the server; do not paste it into
  an SSH session against `blog` expecting it to work.)
- **`ttft_p50_ms` and `latency_p50_ms` measure different things and are
  not interchangeable.** `ttft_p50_ms` (time to first token) only gets
  populated by streaming traffic, and is what the score's latency factor is
  calibrated against. `latency_p50_ms` is the full round-trip of the
  non-streaming path (and of every probe). A deployment that only ever sees
  non-streaming traffic will show every route's `ttft_p50_ms` stuck at the
  neutral default (`1500.0`), because nothing ever measured a real one.
- **`latency_p50_ms` is no longer diagnostics-only: it BACKS the score
  whenever `ttft_p50_ms` was never really measured.** This changed on
  2026-08-18, and the reason is worth knowing, because the old behaviour
  disabled a feature silently. Measured against this deployment, 48 of 52
  routes sat at exactly `1500.0` — so the latency factor was the *same
  constant* for almost every route, and a constant common to all of them
  cannot reorder anything no matter what exponent it is raised to.
  `auto:fast`, `auto:balanced` and `auto:strong` returned **byte-identical
  orderings**: the profile knob did nothing at all. `latency_p50_ms` was
  populated on 51 of those same 52 routes, so the signal existed and the
  score simply never read it. See `ranking.latency_signal_ms`.
- **`quality` saturates, and that is a real limit of the battery, not a
  bug you can read around.** Measured 2026-08-18: of the 42 free routes
  carrying `tools`, **14 tied at exactly `1.0`** and 19 more at `0.8`. The
  battery is five short prompt-and-check cases (add 7+5, answer in one
  word, emit two JSON keys, call one tool, write a Spanish sentence), and a
  2026-vintage small model passes all five as reliably as a frontier
  reasoner does. Three harder candidate cases were written and run live
  against `liquid/lfm-2.5-2.6b` (2.6B params), `deepseek-reasoner`,
  `grok-3` and `nvidia/nemotron-3-super-120b` — the bat-and-ball trap, a
  four-step arithmetic chain, and a two-constraint generation — and **every
  route passed every case**. Grounded high-volume JSON did not separate them
  either: at that output length the reasoning models spend their token
  budget thinking and get truncated, so the case punishes the routes it is
  meant to reward. The tie the table reports is therefore *real* on this
  axis. Do not read `quality: 1.0` as "this route is as good as any other
  for a hard task" — read it as "this route passes every liveness check we
  currently know how to run cheaply". See `quality_suite.DISCRIMINATING_WEIGHT`.
- `score` is always computed with the `balanced` profile weights for
  this table, regardless of what any individual request asked for -- it is
  meant as a stable reference point, not a live prediction for every
  possible `profile`.

## Cooldowns: what they are and how they clear

A cooldown is a temporary exclusion from the attempt chain, tracked in
`Proxy.cooldowns` (in-memory, not persisted -- it resets on every restart).
Three different mechanisms can set one, with three different durations:

- **A `429` from the provider** punishes immediately, respecting the
  provider's own `Retry-After` header when present (falling back to a
  short default, always capped well under an hour).
- **A confirmed on-demand or periodic probe failure** punishes with an
  exponential backoff (starting at 60s, capped at 3600s) that grows if the
  route keeps failing its probes.
- **A paid route that keeps failing real traffic** gets a flat, capped
  60-second punishment of its own (paid routes are never probed, to avoid
  spending real money just to measure them -- see the runbook for why).

**A cooldown, once set, only ever ends by expiry -- there is no path that
ends one early.** This is stricter than it might look, and worth spelling
out because the three mechanisms above interact in a way that is easy to
get wrong:

- Real client traffic can never even reach a route that is currently
  cooling: `router.ordenar()` filters cooldown out of the candidate list
  *before* the attempt chain is built, so a request is never dispatched
  against a punished route in the first place -- there is nothing for it
  to "recover" by succeeding.
- The periodic health probe does **not** skip a cooling route (it bypasses
  `router.ordenar()` entirely and probes every free route on schedule,
  cooldown or not) -- but a *success* against one explicitly does **not**
  clear the punishment either: `Proxy.completar` only clears on success
  when the call is real traffic (`es_sonda=False`); a periodic probe
  passes `es_sonda=True` and that branch is skipped on purpose. Verified
  directly: force a cooldown, then run a periodic-style probe against that
  same route with a `200` response -- the cooldown timestamp is bit-for-bit
  unchanged afterward.
- An on-demand probe (see below) only ever fires *before* a cooldown
  exists, as the confirmation step that decides whether to set one in the
  first place -- it is not a recurring "check if this can end early" probe
  run against an already-cooling route.

What a subsequent success on a route with a *previous* punishment on
record actually clears is a different thing: the **escalation counter**
(`_castigos`) that decides how long the *next* cooldown would be if the
route gets punished again. A route that recovers on its own resets to a
fresh 60s base instead of continuing to escalate from wherever the
exponential backoff had left off -- but the cooldown that is active at any
given moment always runs its full course to `cooldown_until`.

There is currently no manual "un-cooldown this route" switch -- if a route
is stuck punished longer than its backoff should allow, that itself is
worth investigating (see the runbook).

### Client traffic can only make the gateway go check -- it cannot exclude a route by itself

A run of failures against a **free** route from real traffic only builds
up an internal *suspicion* counter; it never sets a cooldown directly.
Crossing a threshold (`UMBRAL_SOSPECHA`, 3 consecutive non-`429` failures)
makes the gateway fire its own confirmation probe (the same fixed `PING`
payload the periodic health probe uses) in the background, and only that
probe's result decides whether the route actually gets punished. This
exists specifically so that a client cannot cool down a healthy route just
by sending it requests that happen to fail for reasons that have nothing
to do with the route being broken (a huge prompt, a moderation flag, a
reasoning model that burns its whole token budget without answering...) --
see the design spec §7 for the long history of why this changed from "the
client's own failures count directly" to "the client can only make the
gateway go check."

This is the general rule, and it has exactly two exceptions, both listed
above and both deliberate: a `429` still punishes immediately (a rate
limit is unambiguous evidence from the provider itself, nothing a client
payload could fake), and a **paid** route's real-traffic failures punish
directly too, without any confirmation probe (probing a paid route would
spend real money with no key to charge it to -- see the runbook). Neither
exception applies to real traffic against a **free** route, which is the
case the rule above actually describes.

### The on-demand probe budget is fair-shared, not first-come-first-served

An on-demand probe is rate-limited two ways at once, so that no single
client -- hostile or just unlucky -- can make the gateway spend an
unbounded amount of the free quota investigating routes:

- **Per route**: at most one on-demand probe every `LIMITE_PROBE_BAJO_DEMANDA_S`
  (60s), no matter how many suspicious requests arrive against that route
  in the meantime.
- **Aggregate, across every route at once**: at most
  `LIMITE_PROBE_GLOBAL_POR_MINUTO` (5) on-demand probes start per minute,
  full stop -- independent of how many routes the catalog has. Without this
  second limit, a catalog with more routes just means more routes that can
  each independently burn their own 60s allowance at the same time.

When more routes are waiting for a probe than the aggregate limit allows
in one pass, `Proxy._admitir_sondas_pendientes` picks who goes next by
**staleness, not arrival order**: the candidate that has gone the longest
without an on-demand probe (never probed on demand = counts as infinitely
stale, so it always wins) gets the next free slot. Plain first-come-first-served
would starve whichever route sits last in the attempt chain forever, since
`completar()` walks candidates in the same `priority`/score order every
time and would always ask for a slot for the front of the chain first --
exactly the route most likely to already be healthy. Fair-share re-sorting
means every suspicious route eventually gets its probe within a bounded
number of admission rounds, not "never" for whichever one happens to sort
last.

`429` never goes through any of this -- it is not suspicion-based and does
not consume the on-demand probe budget at all (see above).
