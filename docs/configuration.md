# Configuration

Every environment variable the service reads, what it defaults to, and the
two that have a real footgun attached. See `.env.example` for a copy-paste
template (used for local development only -- production, via Coolify,
sets these in its own UI, one variable at a time, precisely so there is no
`.env` file on disk that a careless `rsync --delete` could ever wipe).

| Variable | Default | What it is |
|---|---|---|
| `LLM_LIBRE_API_KEYS` | *(none -- required)* | Comma-separated keys clients authenticate with. **The process refuses to start if this is missing or empty** -- on purpose: the alternative is a container that starts fine, `/health` reports `ok` (it does not depend on keys), and every single `/v1/*` request silently returns `401` for every caller, with nothing in the logs to tell that apart from "someone sent the wrong key." A process that fails loudly at startup, with a message naming the exact variable, beats one that looks healthy and rejects everyone. |
| `CHATGPT_PROXY_URL` | `http://127.0.0.1:8888/v1` (the YAML's own default, which does not work in production) | Address of the self-hosted `chatgpt-proxy` service. **Include the `/v1` suffix if you can** -- its real routes are `/v1/chat/completions` and `/v1/models`, not the bare paths Kilo uses. If you set only a bare host (no path at all), the gateway adds `/v1` for you and logs a warning about it. If you set something with its *own* path that is not `/v1` (e.g. a reverse-proxy mount like `.../v2`), that is respected exactly as given -- never silently rewritten -- with only a warning in case it was a mistake. |
| `KILO_API_KEY` | *(unset)* | Optional -- see the callout below for what "optional" actually depends on. |
| `MINIMAX_API_KEY` | *(unset)* | Key for the paid fallback provider. Without it, the paid escape hatch simply cannot be used (every attempt against it fails); free-tier routing is unaffected. |
| `HEALTH_PROBE_HOURS` | `5` | Hours between health-probe cycles for every free route. **Keep this well under 24h** -- see the callout in [`operations-runbook.md`](operations-runbook.md#get-health-the-three-states-and-what-each-one-implies) for why a value at or near 24h eventually makes `/health` misreport low-traffic routes as dead, unrelated to whether they actually are. |
| `QUALITY_PROBE_EVERY_N_CYCLES` | `5` | How many health cycles pass between quality-battery runs (so, by default, roughly once a day). See the runbook for what this costs in free-tier quota. |
| `DAILY_PAID_CAP` | `200` | Daily request allowance against the paid fallback, per key. Counted in requests, not tokens or money -- see `GET /v1/uso`. |
| `PER_MINUTE_LIMIT` | `60` | Per-key rate limit, requests per minute. |
| `RUTA_DB` | `/datos/llm-libre.sqlite3` | Path to the SQLite file (catalog + all telemetry). See the runbook for why this needs to sit on a persistent volume. |
| `PROVIDERS_YAML` | `providers.yaml` | Path to the provider registry file. |

## `KILO_API_KEY`: what "optional" actually means

Verified directly against `proveedores.cargar` and `cliente.armar_peticion`:
**unset, set to `""`, and set to whitespace-only (`"   "`) are all exactly
equivalent** -- all three normalize to `Proveedor.clave == ""`
(`cargar` does `(entorno.get(clave_env, "") or "").strip()`), and
`armar_peticion` only adds an `Authorization` header when `p.clave.strip()`
is truthy. So, contrary to an earlier version of this doc: there is no
footgun in leaving `KILO_API_KEY` blank versus not creating the variable
at all in Coolify's UI -- both produce a request with no `Authorization`
header, which is exactly what Kilo's anonymous tier needs.

**The real footgun runs the other way.** Kilo's gateway accepts fully
anonymous requests -- no `Authorization` header at all -- as a courtesy
(not a documented contract; see the design spec's known risks for what
happens if that ever stops being true). If `KILO_API_KEY` is set to
anything **non-empty**, that value is sent as `Authorization: Bearer
<value>` on every request to Kilo, unconditionally -- there is no
fallback to anonymous mode if the key turns out to be stale, revoked, or
mistyped. A key that goes bad silently turns "every `kilo/*` route works
anonymously" into "every `kilo/*` route gets rejected by Kilo's own auth
check," which the gateway cannot tell apart from Kilo being down: it just
sees failing requests, raises suspicion, and eventually cools the routes
down (see the routing doc) -- exactly like a real outage would look. If
`kilo/*` routes are failing and `KILO_API_KEY` is set, checking whether
that specific key still works is worth doing before assuming Kilo itself
is the problem.

## Regenerating `openapi.json`

`openapi.json` (repo root) is not maintained by hand -- it is
`app.openapi()` from the real FastAPI app, dumped to disk, so it cannot
drift from what `/docs` and `/openapi.json` actually serve. Regenerate it
after changing anything under `src/llm_libre/openapi.py` or any route's
`summary`/`description`/`responses`/`openapi_extra`:

```bash
venv/bin/python scripts/generate_openapi.py
```

This builds the app with a minimal in-memory `Estado` (no `.env`, no
`/datos`, no network) -- exactly like the test suite does -- so it never
needs real credentials or a running deployment to produce an accurate
schema.
