# llm-libre

A gateway that exposes the **free** LLM models of several providers
(chatgpt-proxy and Kilo today; the registry is declarative and open to more,
see "How it decides" below) behind a single OpenAI-compatible contract, with
automatic selection of the best available model according to its own ranking
(measured quality, reliability and latency) and a paid tier (MiniMax) as a last
resort when everything free is down. Any client that speaks the OpenAI protocol
uses it with no special library, changing only `base_url` and `api_key`.

**Vocabulary, so the three do not get confused:** *tier* is `free` \| `paid`
(whether it costs money). *Profile* is `fast` \| `balanced` \| `strong` (what
the request prefers). **`priority` is a third concept, separate from both** —
the manual order in which the router tries providers before looking at any
score (see "How it decides" below). They never override each other: a paid
route with `priority: 0` still always goes last. Money outranks manual order.

## API reference, Postman collection and extended docs

- **Interactive API reference**: run the service and open `/docs`
  (Swagger UI) or `/openapi.json` — every endpoint, both auth schemes,
  every `x_*` extension, every status code with a real example. The
  committed [`openapi.json`](openapi.json) at the repo root is generated
  straight from the running app (`app.openapi()`), so it cannot drift from
  what `/docs` actually serves. Regenerate it after touching
  `src/llm_libre/openapi.py` or any route's docs:

  ```bash
  .venv/bin/python scripts/generate_openapi.py
  ```

- **Postman collection**: [`llm-libre.postman_collection.json`](llm-libre.postman_collection.json)
  at the repo root — import it, fill in the `base_url` and `api_key`
  collection variables, and every request works. Covers every `auto*`
  alias, an explicit model id, streaming, one request per `x_*`
  extension, both auth styles, and the error paths (missing key, unknown
  model, an impossible capability requirement).
- **[`docs/routing-and-ranking.md`](docs/routing-and-ranking.md)** — how
  the router actually decides, the `tier`/`profile`/`priority` vocabulary,
  how to read `GET /v1/ranking` to explain a routing decision (with a
  worked example), and what a cooldown is.
- **[`docs/operations-runbook.md`](docs/operations-runbook.md)** — the
  SQLite file and what losing it costs, `/health`'s three states, what to
  check when everything returns `503`, and why probes spend the same free
  quota real traffic does.
- **[`docs/configuration.md`](docs/configuration.md)** — every environment
  variable, including why `KILO_API_KEY` must stay genuinely unset and why
  `CHATGPT_PROXY_URL` needs its path suffix.
- **[`docs/providers.md`](docs/providers.md)** — how a provider is
  declared, the three registration patterns, and what
  `default_capabilities` is for, so adding a provider is a config
  change, not a code change.

## Quick start

With the OpenAI SDK (Python):

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<domain>/v1",
    api_key="<one-of-the-LLM_LIBRE_API_KEYS>",
)

resp = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "say hi"}],
)
print(resp.choices[0].message.content)
```

With `curl`:

```bash
curl -s -H "X-API-Key: <one-of-the-LLM_LIBRE_API_KEYS>" -H "Content-Type: application/json" -d '{"model":"auto","messages":[{"role":"user","content":"say hi"}]}' https://<domain>/v1/chat/completions
```

`base_url` always ends in `/v1`. The key is accepted through **either** of two
headers:

- `Authorization: Bearer <key>` — what any OpenAI SDK sends through its
  `api_key` parameter, with no extra configuration. It is what the snippet
  above uses.
- `X-API-Key: <key>` — the convention `arkiv-api`, the sibling gateway,
  already uses. It stays valid for whoever was already using it, and it
  **wins** if a request arrives carrying both headers at once.

## `auto*` aliases

The `model` field accepts a real id or one of these virtual aliases. The real
id is the **model's**, with no provider prefix — exactly as `GET /v1/models`
lists it (for example `nvidia/nemotron-3-super-120b-a12b:free`, where `nvidia/`
is part of the model's id, not the provider). That is deliberate: the same
model often exists at several providers, and asking for it by its id lets the
gateway fail over between them. Who served it is visible afterwards, in the
`X-Route-Used` header (`kilo/nvidia/nemotron-3-super-120b-a12b:free`), which
does carry the provider up front.

| Alias | What it selects |
|---|---|
| `auto` | **Balanced** profile: quality, reliability and latency weigh the same |
| `auto:fast` | **Fast** profile: prioritises low latency, gives up some quality |
| `auto:strong` | **Strong** profile: prioritises measured quality, gives up some latency |
| `auto:tools` | Balanced profile + requires the route to support function calling |
| `auto:vision` | Balanced profile + requires the route to support image input |

With any alias, if the chosen route fails the gateway retries with the next one
in the ordered list (automatic failover, including the jump to paid if it is
needed and allowed).

## `x_*` extensions

They go in the same JSON body as the rest of the request; a third-party SDK
that does not know them ignores them without breaking:

| Field | What it does |
|---|---|
| `x_requires` | List of required capabilities, e.g. `["tools", "vision"]` — equivalent to asking for them by alias |
| `x_min_context` | Minimum context window in tokens; discards routes below it |
| `x_allow_paid` | `false` disables the paid tier (MiniMax) for that one request |
| `x_raw` | `true` disables reasoning trimming (`<think>`, etc.) and returns `content` exactly as the provider sent it |

## `x_reasoning` in the response

Several models — free and paid alike — spill their chain of thought inside
`content`, between `<think>` / `<thinking>` / `<reasoning>` tags. The gateway
separates it: the `content` the client sees comes out clean and the trimmed
block comes back in a top-level field, `x_reasoning`, which any OpenAI SDK
ignores without breaking.

```json
{
  "choices": [{"message": {"role": "assistant", "content": "The answer is 4."}}],
  "x_reasoning": "2+2 is 4"
}
```

The field only appears if there genuinely was something to trim. **It is not
sent when streaming**: putting it there would require emitting a non-standard
SSE event, exactly what the contract avoids so as not to break the SDKs'
parsing. A client that streams and wants the reasoning asks for `x_raw: true`
and receives it inside `content`, exactly as the provider sent it.

## Response headers

Every **non-streaming** response from `/v1/chat/completions` carries:

- `X-Route-Used`: `<provider>/<model_id>` of the route that actually served it
- `X-Tier`: `free` or `paid`
- `X-Attempts`: how many routes were tried before answering (or giving up)

**When streaming (`stream: true`) these headers do NOT travel.** HTTP headers
are sent before the body, and at that moment the failover chain is not yet
resolved — there is no way to know which route will end up serving. Paid spend
is still recorded and visible another way: `GET /v1/usage`.

## Endpoints

| Endpoint | What it does |
|---|---|
| `POST /v1/chat/completions` | OpenAI's chat contract, with optional `stream: true` and the `x_*` extensions above. An explicit model the real provider no longer serves (a live 404, even though it is still in the local catalogue) returns `404` with suggestions instead of the generic `503` — only on the non-streaming path |
| `GET /v1/models` | The normalised catalogue (OpenAI shape) plus the `auto*` aliases |
| `GET /v1/ranking` | Each route's score (with its `priority`) and every component broken out, **sorted with the same key the router uses** — cooldown included: a punished route (`cooldown_until` in the row) goes last, even if it scores better than everything else — so you can audit why the router chose what it chose, without the top row contradicting `X-Route-Used` |
| `GET /v1/usage` | The calling key's paid consumption for the day, against its daily cap |
| `GET /health` | Honest: `ok` only if there is at least one live free route; `degraded` if only paid is left; `down` if nothing is serviceable. A route counts as alive by **positive evidence** (a recent success, or no telemetry yet) — not by an average reliability, which a single client can poison. One failed probe alone is not enough to declare it dead: two consecutive ones are needed, with no success in between. No key required |

## Configuration

Environment variables (see `.env.example`):

| Variable | Default | What it is |
|---|---|---|
| `LLM_LIBRE_API_KEYS` | *(no default — required)* | Keys the gateway accepts from clients, comma-separated. The process **does not start** if it is missing or empty: see below |
| `CHATGPT_PROXY_URL` | `http://127.0.0.1:8888/v1` (the YAML's default) | URL of `chatgpt-proxy` (an in-house service, deployed on `blog`). No credentials — just the address, which is not fixed yet, which is why it is configurable through the environment instead of being wired into `providers.yaml`. Ideally it includes the `/v1` (its real paths are `/v1/chat/completions` and `/v1/models`); if only the **host** is given (no path at all), the `/v1` is appended automatically, with a warning in the log. If instead **its own path** is given (e.g. a reverse-proxy mount, `.../v2`), that path is respected verbatim — nothing is overwritten, it only warns if it does not match `/v1`, in case it was accidental |
| `KILO_API_KEY` | *(unset)* | Optional. **Leave it UNSET**, not blank — see the note below |
| `MINIMAX_API_KEY` | *(unset)* | Key for the paid provider (the fallback tier) |
| `HEALTH_PROBE_HOURS` | `5` | How many hours between health probes of each route |
| `QUALITY_PROBE_EVERY_N_CYCLES` | `5` | How many probing cycles between runs of the quality battery |
| `DAILY_PAID_CAP` | `200` | Daily cap of requests to the paid tier, per key |
| `PER_MINUTE_LIMIT` | `60` | Requests per minute, per key |
| `DB_PATH` | `/datos/llm-libre.sqlite3` | Path to the SQLite file (catalogue + telemetry) |
| `PROVIDERS_YAML` | `providers.yaml` | Path to the provider registry |

These variables were renamed from Spanish on 2026-08-18. The code still reads
the OLD names as a fallback (the new one wins), so a deployment configured with
`RUTA_DB`, `TOPE_PAGO_DIARIO`, `LIMITE_POR_MINUTO`, `PROVEEDORES_YAML`,
`SONDEO_SALUD_HORAS`, `SONDEO_CALIDAD_CADA_N_CICLOS` or `ROTAR_EMPATES` keeps
working until it is updated. Once Coolify is updated, the fallback in
`main._env` can go.

**`KILO_API_KEY` must stay unset**, not empty-but-present: Kilo's anonymous
tier depends on the request travelling without any `Authorization` header. In
Coolify that simply means not creating that variable in the UI, not creating it
with an empty value.

**An unconfigured `LLM_LIBRE_API_KEYS` makes the process fail at startup**,
deliberately, with a message naming the variable: the alternative — letting it
start anyway — would produce a container that `/health` reports as `ok` while
rejecting 100% of requests with a 401 for every caller, with nothing in the
logs to distinguish "no keys are configured" from "wrong key". Better a
container that does not start, with a clear reason.

## Deployment

It is deployed with **Coolify, by git push to `main`** — Coolify redeploys on
every push on its own, building the `Dockerfile` at the root. **It is not by
rsync.** `docker-compose.yml` is only for bringing it up locally during
development.

It needs a **persistent volume mounted at `/datos`**: without it, the SQLite
file (route catalogue plus all the probing telemetry) is destroyed on every
redeploy. That does **not** mean the ranking starts from zero for days (fixed
in Task 14): `scheduler` runs its first full probing cycle — catalogue, health,
AND the WHOLE quality battery, not just health — as soon as the process starts,
before its first `sleep`: the cycle counter starts at `0`, and
`0 % QUALITY_PROBE_EVERY_N_CYCLES == 0` for any sane value of that variable.
Verified by running `probing.cycle(state, 0)` once against a freshly created
`Storage`: it leaves `probes.kind='health'` AND `probes.kind='quality'` rows
for every free route in that single pass. What does take days to rebuild is the
**history** feeding `reliability` and latency (a moving window of probes plus
real traffic) — that is where the roughly daily cadence of the quality battery
matters. Practical consequence: every process restart also spends a full pass
of free quota (see "How it decides" below, the probes section), not just the
catalogue sync.

## How it decides

- **Priority order among free providers:** `chatgpt` (`priority: 0`) is tried
  before `kilo` (`priority: 1`). It is an in-house service, so it gets
  preference over third parties — but it is still `tier: free`, it does not
  consume the daily paid cap. `minimax` (`paid`, `priority: 2`) still goes
  **always last**, regardless of its `priority`: that number orders routes
  within one `tier`, it never decides between `free` and `paid` (see the
  vocabulary note above).
- **`chatgpt` is served with `tools: false`, and that is still mandatory.** The
  anonymous backend stopped returning `HTTP 500` when sent `tools` (that
  changed), but it still does not support *function calling*: with
  `tool_choice: "required"` it returns `tool_calls: None` and plain prose. What
  it does support is "advanced tools" (bookings, shopping, widgets, canvas),
  not what this gateway's `tools` capability means. The new behaviour is
  **more** dangerous than the old 500 — a 500 failed honestly and triggered
  failover; silently returning prose would hand text to an agentic client
  expecting a structured call — so this declaration is the only barrier. A
  request with `tools` (or `auto:tools` / `x_requires: ["tools"]`)
  automatically discards `chatgpt`'s routes and falls to the next free provider
  that does support them (Kilo); only if that fails too does it reach the paid
  tier.
- `chatgpt-proxy` leaks ChatGPT's "canvas" mode into `content`, with markers of
  the form `:::word{...attributes...}` … `:::`. The gateway unwraps them (both
  in one piece and while streaming) keeping the text inside — unlike `<think>`,
  there it IS the answer, not something to discard. **It is a per-provider
  declaration** (`unwraps_canvas` in `providers.yaml`, off by default), not
  something universal: `:::nota{...}` / `:::tip{...}` is also standard
  Docusaurus/MDX syntax, and applying the unwrapping blindly would tear those
  markers out of a legitimate documentation answer from Kilo. Only `chatgpt`
  declares it.
- **The free providers' catalogue is always discovered from their own
  `/models`, never hardcoded** — Kilo **and `chatgpt` too**: that way a model
  that changes id, disappears or shows up is detected on its own, without
  touching code or `providers.yaml`. There are three patterns in the registry,
  depending on what each one's `/models` brings (the registry is declarative
  and open: `openrouter`, removed from the production registry on 2026-08-17
  because it never had a key configured and its 16 routes 401'd every time, is
  still a valid example of the first pattern — see `docs/providers.md`):
  - Kilo (and, as an example, OpenRouter): ids **and** capabilities, both discovered.
  - `minimax` (paid): neither ids nor capabilities — its real `/models` only
    brings `id`/`created`/`owned_by` — so both are declared by hand
    (`fixed_models`).
  - `chatgpt`: ids **discovered** (its `/v1/models` is genuinely dynamic, with
    a cache and TTL against ChatGPT's real backend), but **capabilities
    declared** (`default_capabilities`) — its catalogue never brings capability
    metadata, only `id`/`description`. It is a general mechanism, not something
    special to `chatgpt`: any future provider with an equally bare `/models`
    can use it without touching code.
  - Two kinds of entry are filtered out of `chatgpt`'s discovery, both because
    of what the response says about itself, never through a list of ids: the
    legacy aliases the proxy adds (`gpt-4o`, `gpt-4o-mini`, `gpt-4`,
    `gpt-3.5-turbo`) carry `description: "Alias → <target>"`; and `auto` (plus
    any compound alias `auto:fast` / `auto:strong` / `auto:tools` /
    `auto:vision`), reserved by llm-libre's own `parse_request` (it collides
    with its aliases), is discarded as a reserved id.
- Every route (provider + model) is probed for **health** every
  `HEALTH_PROBE_HOURS` (default 5h) and, among the live free routes, for
  **quality** every `QUALITY_PROBE_EVERY_N_CYCLES` cycles (default 5, i.e.
  roughly once a day) with a battery of code-verifiable cases (valid JSON, a
  correct tool call, a requested format, arithmetic, language) — no LLM judge.
  Beyond this cycle, the proxy fires its own health probes **on demand** when
  real traffic accumulates suspicion about a route (see below) — same payload,
  same destination, without waiting for the next scheduled cycle.
- The catalogue discards whatever the provider **describes** as something other
  than a general-purpose chat model — safety guardrails and classifiers,
  rerankers, embedding models — and *meta-routers*, which are not a model but a
  draw between other models. It is decided by reading the `name` and
  `description` fields of `/models` itself, never through a list of ids: an id
  blacklist would rot exactly like the hardcoded ids this gateway exists to
  replace.
- The ranking combines measured quality, recent reliability (successes over
  probes plus real traffic) and latency (p50 time-to-first-token), with weights
  that change according to the requested profile: `fast` weighs latency above
  quality, `strong` the other way round, `balanced` splits evenly. The context
  window (`x_min_context`) is a prior filter, it does not enter the score.
- **A route that has not been through the quality battery yet goes after the
  ones that have**, even if its neutral value scores higher: that 0.6 is an
  assumption, not a measurement, and `/v1/ranking` says so (`quality: null`,
  `quality_measured: false`). It stays in the chain of attempts so that it gets
  measured eventually.
- `ttft_p50_ms` is **time to first token** and only the streaming path measures
  it, because it is the only one that can. The non-streaming path (and the
  health probe) report their complete round-trip separately, in
  `latency_p50_ms`, which does not enter the score: they are two different
  magnitudes and averaging them together produced a meaningless number.
- Paid routes always go last in the chain of attempts, and are only used if the
  free ones are exhausted, the key has not exceeded its daily cap, and the
  request does not carry `x_allow_paid: false`.
- **A route that fails repeatedly in a row (with no success in between) enters
  cooldown, not just when the provider returns `429`** — and that same failure
  does not count toward measured reliability either, nor therefore toward
  `/health` or `/v1/ranking`, if it is evidence about the *request* rather than
  about the *route*. Before, a `500`, a timeout or a network error never left a
  cooldown: a broken or **hung** route kept being tried on every request, ahead
  of healthy ones if it had a better `priority`, indefinitely — with the default
  timeout (90s) that is up to 7.5 minutes per request on the longest chain, and
  `/health` stays `ok` as long as one route is alive. An isolated failure does
  not punish (it avoids pulling a healthy route for a hiccup); on the third
  consecutive failure it does, with the same exponential backoff `429` already
  uses. A provider can also declare its own `timeout_s` in `providers.yaml`
  (default: the global 90s) to bound the worst case of one known to be slow,
  without lowering the timeout for everyone — it applies equally to the
  synchronous and the streaming path.
- **The rule for a `4xx` is about ATTRIBUTION, not about retryability — and the
  default is INVERTED: a `4xx` counts as evidence about the *route* (just like
  a `500`) unless it is in a short, justified list of codes that are genuine
  evidence about the *request*.** It was not always this way: until a fifth
  review round the default was the opposite ("every `4xx` is the request's
  fault except these seven codes"), and that default silently hides any code
  nobody thought of — proven by adding `405` to the "does count" list (which
  the rule itself demanded) and watching the whole suite stay green, because
  nothing pinned the axis down, only the list. **Principle: when you cannot
  tell whose fault it is, count it** — a false alarm recovers on its own, a
  silent outage does not, and the costs are asymmetric. The short list, with
  its justification: `400` (Bad Request — the body could not even be
  interpreted), `413` (Payload Too Large — *this* request's size) and `422`
  (Unprocessable Entity — invalid for *this* particular request). None of the
  three counts toward cooldown or reliability — counting them would turn one
  client's mistake into an outage for everyone (verified: three malformed
  requests in a row are enough to leave all five routes in cooldown if they
  count). **Everything else counts as evidence about the route by default**,
  known today or not — including four codes an earlier version misclassified by
  reasoning "it is not retryable, so it does not count": `401` (the key
  expired), `402` (out of credit), `403` (suspended account or provider
  moderation — see below for why this alone is not enough) and **`404`**
  (`model_not_found` — literally the problem this project exists to detect).
  The cost of the previous classification was measured: with all 5 routes
  returning `401` (or, under the old default, any code nobody had enumerated —
  `405`, `409`, `415`, `418`, `431`, `451`…), the client received `503` on 100%
  of requests while `/health` stayed `ok` — an outage with a green light, with
  no backstop whatsoever for MiniMax (never probed). `408`, `425` and `429`
  also land on the route's side without needing any special mention in the
  code: under the inverted default they fall there on their own. The event is
  still always stored (it stays diagnosable — an operator looking at the
  `events` table sees every code alike), and reliability excludes from its
  window entirely those that are evidence about the request — verified: 26
  malformed requests in a row from a single key are enough to sink *every*
  route's reliability if they count.
- **`/health` no longer averages — it requires positive evidence, and that is
  why a single client can no longer poison it.** `403` is genuinely ambiguous:
  a suspended account (evidence about the route, correctly counted above) or
  content moderation for one particular client (evidence about the *request*) —
  and the gateway cannot tell them apart without parsing each provider's
  specific body. Even with `400`/`413`/`422` excluded, 30 requests with flagged
  content from a single key were enough to sink the reliability *average* of
  every route, with `/health` at `down` for all keys — and that is **worse**
  than a plain `503` in the real deployment: Coolify uses `/health` as its
  health check and restarts the container when it fails, but the `events` table
  lives on the persistent `/datos` volume, so a fresh process against the same
  database would keep seeing the same events and the restart would fix nothing.
  That is why `/health` stopped looking at the average: a route counts as alive
  according to the **most recent signal** available about it within a 24h window
  — a real success, or the result of the last health probe (`ok=1` counts as
  alive, `ok=0` counts as **dead**: a probe is never ambiguous, the gateway
  controls its own payload, and this holds for a failed result just as much as
  for a successful one) — or, if there is no signal at all yet, simply no real
  telemetry. *Real* failures (`events.ok=0`), on their own, are never enough to
  declare a route dead — they remain ambiguous in a way a probe never is.
  **`reliability` still feeds the ranking and the real route selection exactly
  as before** — not just the `/v1/ranking` diagnostic endpoint, but also
  `router.order_routes` (the real chain of attempts the proxy uses): a
  badly-scored route loses position and self-corrects as soon as someone looks
  or the router recalculates; a route `/health` declares dead restarts the
  container. The asymmetry is deliberate.
- **Only a probe written by the gateway itself can exclude a route. A real
  client's traffic never can — it can only ask the gateway to go and look.**
  Six rounds of this same mechanism (retryability, a list of codes, the list
  with the inverted default, per-response attribution, chain-level attribution)
  each fell to a new vector, and the last two were hatches in the previous
  design itself: a chain of ONE single route (the client forces it with an
  explicit `model` or with `x_min_context`, which `/v1/ranking` already
  publishes per route — with no internal knowledge at all) and the streaming
  branch that forces the stream closed once too many chunks with no useful
  content have been held back, which committed without comparing against
  anything. Measured: 15 identical requests were enough to cool five healthy
  routes, one by one, through either path — even with
  `{"model":"auto","stream":true}` and no extensions at all.

  The change is structural, not a seventh predicate: a real-traffic failure no
  longer counts toward any cooldown directly — it only increments a counter of
  CONSECUTIVE failures per route (*suspicion*, which excludes nothing and
  resets to 0 on any success, **with no time window** — an earlier version
  counted "3 failures in 10 minutes", and traffic slower than that window never
  reached the threshold: measured, 80 consecutive failures spaced just wider
  than the window fired no probe at all, leaving a dead route in a low-traffic
  deployment first in the queue forever). On reaching the threshold (3) it
  fires, in the background, its OWN probe with the same fixed payload (`PING`)
  the periodic health probing already uses — and it is that probe, never the
  client's request, that decides: if it fails, it punishes; if it passes, the
  suspicion is cleared. Why this closes all six vectors at once: a probe's
  payload is written by the gateway — there is nothing to attribute, so a
  failure against it is unambiguous evidence about the route, with no possible
  exception. A successful probe never cancels a cooldown NEWER than itself
  (e.g. a real 429 that arrived while the probe was still in flight) —
  otherwise a client could use a probe to cancel a provider's explicit "stop".

  On-demand probes are *rate-limited* to one per route every 60s **and,
  additionally, to an AGGREGATE cap** (5/min across all routes): the per-route
  limit does not bound the aggregate when the catalogue has N routes —
  measured, 11 routes with no global cap add up to 15,840 extra requests per
  day against the same quota real traffic needs, and N is not controlled by the
  operator (it comes from the live catalogue). With both limits, the extra cost
  a single hostile key can impose is bounded regardless of how many requests it
  sends or how many routes the catalogue has — and since the payload is the
  gateway's, that key can never make the probe fail. **Cost to the attacker: at
  most, bounded extra probing (≤60 requests/hour per route, also bounded in
  aggregate) — never a healthy route going down.**

  That aggregate cap is shared out by **fairness**, not first-come-first-served:
  a "first candidate to arrive, first served" admission starves the routes
  further back in the chain — and a genuinely broken route is exactly the one
  its collapsed `reliability` sends to the end of `router.order_routes`.
  Measured: with 6 equally suspected routes, the one at position 5 never got a
  probe in 60 simulated minutes; with 12, position 11 never did either (295
  probes run, zero against the victim). Now every candidate crossing the
  threshold enters a waiting queue that is reordered, on every admission
  attempt, by the age of its last probe (never probed = infinity, always wins) —
  the aggregate cap did not change, only who takes the next free slot. With
  that, no route is left out indefinitely: worst case bounded by
  `ceil(N / 5)` admission rounds — measured, the victim at position 11 of an
  11-route catalogue gets its probe (and its cooldown) at 139 simulated
  seconds, not "never".

  `429` still punishes immediately, without going through suspicion or a probe
  — unambiguous evidence on its own — but **it no longer reuses the exponential
  backoff of a confirmed probe** (which could escalate to an hour): the
  provider's `Retry-After` is honoured when it sends one, or a short default if
  not, always capped well below an hour — measured, 12 requests from one key
  were enough to cool 3 routes through a real 429 for far longer than the
  provider's own rate-limit window justifies.

  **Paid** routes are left out of suspicion+probe (they are never probed, and a
  probe would spend real money with no key owning that spend) — but they DO
  have their own direct punishment, with the same threshold, without a probe:
  leaving them entirely outside the mechanism, with no replacement, left a
  broken paid route billing every request forever with nothing to exclude it
  (measured: 40/40 billable calls, zero cooldowns). The duration of that
  punishment is **flat and capped (60s), not exponential** — the escalating
  backoff exists for CONFIRMED probes, and a paid punishment has no probe
  behind it by design: reusing that backoff measured 60→120→240→480→960→1920→3600s
  in just 24 requests from one key against the real API. "It only comes into
  play once everything free is exhausted" is an argument about blast radius,
  not a bound on duration — the duration comes solely from that flat 60s, same
  as the 429. Paid usage is also now counted by what is BILLABLE (any `200`
  from a paid route, whether it serves or not — an empty 200 also generates
  tokens the provider charges for), not only by what succeeded: before, an
  empty 200 was genuinely billed and stayed invisible to `/v1/usage` and the
  daily cap.

  A genuinely broken route with a healthy sibling in the same chain still falls
  in the same number of requests as before; a broken one with no sibling (or a
  whole genuinely down *pool*) cools down in the order of seconds/requests
  through its own probe, without waiting for the 5h cycle — and the cooldown
  that probe leaves is stamped with how long the probe took, not with when it
  started (otherwise a probe that takes the full timeout is born with that time
  already "eaten" from the backoff).

  A client now controls, indirectly, when a route is probed — up to 60/h
  through suspicion, against 1 every `HEALTH_PROBE_HOURS` in the periodic cycle,
  ~300× more sampling possible — which raises the chance of catching a
  transient provider blip on exactly one probe. That is why `/health` requires
  **two** consecutive failed probes (with no success in between), not just one,
  before treating it as evidence of death.

  That same logic's fallback check (nothing within the window, neither success
  nor probe) used to look at **any real event** — success or failure — to
  decide "there is history, declare it dead". A real failure with no probe
  confirming it should never be authoritative: it is the same principle as
  suspicion+probe, a client can only ask the gateway to go and look, never
  exclude by itself. Measured: one single failed real request was enough to
  drop `/health` to `down` without any probe having run. Now that check counts
  only **probes** as "there is history" — a real event, success or failure, is
  never enough on its own.

  Four more corrections in the same family (hostile data or a rate-limit are
  not evidence of death, and a probe's clock is its resolution, not its start):
  a negative or non-finite `Retry-After` (`-5`, `nan`) is no longer clamped to
  `0s` (an empty cooldown) but to the short default; a `429` against a probe is
  no longer recorded as a failed health probe (it has its own proportional
  punishment, and two 429s in a row should not be enough to declare a route
  dead); the quality battery (`probe_quality`) now also marks its calls as a
  probe (the same wiring `probe_health` already had), so a battery failure
  punishes directly instead of spending on-demand probe budget unnecessarily;
  and the row an on-demand probe leaves is stamped with the moment it resolved,
  not the moment it was scheduled, so as not to disorder the `ORDER BY at DESC`
  `/health` uses with a hung route.
