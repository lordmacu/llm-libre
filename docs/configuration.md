# Configuration

Every environment variable the service reads, what it defaults to, and the
two that have a real footgun attached. See `.env.example` for a copy-paste
template (used for local development only -- production, via Coolify,
sets these in its own UI, one variable at a time, precisely so there is no
`.env` file on disk that a careless `rsync --delete` could ever wipe).

| Variable | Default | What it is |
|---|---|---|
| `LLM_LIBRE_API_KEYS` | *(none -- required)* | Comma-separated keys clients authenticate with. **The process refuses to start if this is missing or empty** -- on purpose: the alternative is a container that starts fine, `/health` reports `ok` (it does not depend on keys), and every single `/v1/*` request silently returns `401` for every caller, with nothing in the logs to tell that apart from "someone sent the wrong key." A process that fails loudly at startup, with a message naming the exact variable, beats one that looks healthy and rejects everyone. |
| `CHATGPT_PROXY_URL` | `http://127.0.0.1:8888/v1` (the YAML's own default, which does not work in production) | Address of the self-hosted `chatgpt-proxy` service. **Include the `/v1` suffix if you can** -- its real routes are `/v1/chat/completions` and `/v1/models`, not the bare paths Kilo/OpenRouter use. If you set only a bare host (no path at all), the gateway adds `/v1` for you and logs a warning about it. If you set something with its *own* path that is not `/v1` (e.g. a reverse-proxy mount like `.../v2`), that is respected exactly as given -- never silently rewritten -- with only a warning in case it was a mistake. |
| `KILO_API_KEY` | *(unset)* | Optional. **Must be left completely unset, not set to an empty string** -- see the callout below. |
| `OPENROUTER_API_KEY` | *(unset)* | OpenRouter's free tier requires a key for `/chat/completions` even though `/models` is public without one. Without this variable, every `openrouter/*` route fails. |
| `MINIMAX_API_KEY` | *(unset)* | Key for the paid fallback provider. Without it, the paid escape hatch simply cannot be used (every attempt against it fails); free-tier routing is unaffected. |
| `SONDEO_SALUD_HORAS` | `5` | Hours between health-probe cycles for every free route. |
| `SONDEO_CALIDAD_CADA_N_CICLOS` | `5` | How many health cycles pass between quality-battery runs (so, by default, roughly once a day). See the runbook for what this costs in free-tier quota. |
| `TOPE_PAGO_DIARIO` | `200` | Daily request allowance against the paid fallback, per key. Counted in requests, not tokens or money -- see `GET /v1/uso`. |
| `LIMITE_POR_MINUTO` | `60` | Per-key rate limit, requests per minute. |
| `RUTA_DB` | `/datos/llm-libre.sqlite3` | Path to the SQLite file (catalog + all telemetry). See the runbook for why this needs to sit on a persistent volume. |
| `PROVEEDORES_YAML` | `proveedores.yaml` | Path to the provider registry file. |

## `KILO_API_KEY` must be genuinely unset

This is the one variable in this table where "unset" and "set to an empty
string" are **not equivalent**, and getting it wrong quietly breaks
routing rather than erroring loudly.

Kilo's gateway accepts fully anonymous requests -- no `Authorization`
header at all -- as a courtesy (not a documented contract; see the design
spec's known risks for what happens if that ever stops being true). The
gateway's client code only omits the `Authorization` header when the
resolved key is empty *and never sent as an environment variable at all*.
Setting `KILO_API_KEY=` (present, but blank) in most deployment UIs still
counts as "the variable exists," and depending on how the platform handles
that, can end up sending an empty or malformed `Authorization` header
instead of none -- which is not the same thing to Kilo's backend. The safe
action in Coolify (or any similar UI) is: **do not create the
`KILO_API_KEY` variable at all** unless you actually have a Kilo key you
want to use. If you do have one, set it; it only raises your rate limits,
it is never required for the anonymous tier to work.

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
