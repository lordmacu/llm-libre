import asyncio
import contextlib
import os
import random
from contextlib import asynccontextmanager

import httpx

from llm_libre.api import State, create_app
from llm_libre.assets import AssetStore
from llm_libre.auth import RateLimiter
from llm_libre.probing import cycle
from llm_libre.providers import load
from llm_libre.notify import from_env as notify_from_env
from llm_libre.proxy import Proxy
from llm_libre.storage import Storage

def _env(new_name: str, legacy_name: str, default: str) -> str:
    """Read a deployment variable under its CURRENT name, falling back to the
    Spanish one it used to have.

    Environment variables are operator surface: they do not live in this repo,
    they live in Coolify's UI and in whatever .env a machine happens to have.
    Renaming one in the code alone means the next deploy silently ignores the
    configured value and falls back to the default -- with nothing to catch it,
    because a default is a perfectly valid value. Three of these ARE configured
    in Coolify right now (DB_PATH, PER_MINUTE_LIMIT, DAILY_PAID_CAP), and
    DB_PATH falling back to its default would point the gateway at a different
    file than the one holding the telemetry.

    So the new name wins and the old one is honoured as a fallback: renaming
    them in Coolify becomes something that can happen calmly, after the deploy,
    instead of a step that has to land in the same minute as the code. The
    legacy branch is a MIGRATION AID -- once the variables are renamed in the
    deployment, this function and its `legacy_name` argument can go.
    """
    return os.getenv(new_name) or os.getenv(legacy_name) or default


YAML = _env("PROVIDERS_YAML", "PROVEEDORES_YAML", "providers.yaml")
DB_PATH = _env("DB_PATH", "RUTA_DB", "/datos/llm-libre.sqlite3")
HEALTH_PROBE_HOURS = float(_env("HEALTH_PROBE_HOURS", "SONDEO_SALUD_HORAS", "5"))
# Draw between tied routes. On by default: with the quality battery giving 1.00
# to the entire catalogue (see quality_suite.py), a strict order ALWAYS sends
# traffic to the same route and burns one provider's quota while the others watch.
# Turning it off (ROTATE_TIES=false) restores the previous deterministic order,
# useful when debugging an odd response.
SHUFFLE_TIES = _env("ROTATE_TIES", "ROTAR_EMPATES", "true").strip().lower() \
    not in ("false", "0", "no")
# Where generated binaries are stored. Under the SAME persistent volume as the
# database on purpose: an asset URL a client stored has to survive a redeploy,
# exactly like the telemetry does.
ASSETS_DIR = os.getenv("ASSETS_DIR", "/datos/assets")
# The origin asset URLs carry. The gateway cannot discover its own public
# hostname -- behind a Cloudflare tunnel the request it sees says 127.0.0.1 --
# so it is declared. Left EMPTY, api.effective_public_base_url derives it per
# request from X-Forwarded-Proto/X-Forwarded-Host (what the tunnel sets) or the
# connection's own scheme and Host; only when nothing trustworthy can be
# derived does the images endpoint keep handing back the provider's own URL,
# which is the pre-asset behaviour and a safe default: a wrong origin here
# would produce URLs that resolve nowhere.
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip()


def build_state() -> State:
    """Build the process's real State: load the providers from the YAML plus the
    environment, open the SQLite database (creating the schema if missing) and
    share ONE httpx client between the proxy and the probing scheduler.

    It validates `LLM_LIBRE_API_KEYS` BEFORE touching disk or network, and
    deliberately fails hard (an uncaught exception, the process does not start) if
    no key is configured. Without this call, an operator who forgets that variable
    (say, in Coolify's UI) gets a container that starts normally, a `/health` that
    still says "ok" (it does not depend on the keys) and EVERY request to /v1/*
    returning 401 -- indistinguishable in the logs from "this client sent the
    wrong key". It is the same class of failure that honest /health (Task 9)
    already exists to prevent on the routes side; this closes it on the
    authentication side: better a process that does not start with a clear reason
    than one that looks healthy and silently rejects everybody.
    """
    api_keys = {k.strip() for k in os.getenv("LLM_LIBRE_API_KEYS", "").split(",") if k.strip()}
    if not api_keys:
        raise RuntimeError(
            "LLM_LIBRE_API_KEYS is not set (or is empty): without at least one "
            "key the service would start but reject 100% of requests to /v1/* "
            "with 401, for every caller, while /health kept reporting 'ok'. Set "
            "it to at least one key, comma-separated if there are several -- for "
            "example: LLM_LIBRE_API_KEYS=a-long-secret-key")
    providers = load(YAML, dict(os.environ))
    store = Storage(DB_PATH)
    store.create_schema()
    http = httpx.AsyncClient()
    proxy = Proxy({p.id: p for p in providers}, store, http,
                  notifier=notify_from_env())
    state = State(store=store, proxy=proxy, api_keys=api_keys,
                  daily_paid_cap=int(_env("DAILY_PAID_CAP", "TOPE_PAGO_DIARIO", "200")),
                  rate_limiter=RateLimiter(
                      int(_env("PER_MINUTE_LIMIT", "LIMITE_POR_MINUTO", "60"))),
                  # Bounds an UNAUTHENTICATED caller, which the per-key limiter
                  # cannot: it is keyed by a key, and a request without a valid
                  # one never reaches it. Generous by design -- a ceiling on
                  # abuse, not a quota for ordinary use.
                  ip_rate_limiter=RateLimiter(
                      int(os.getenv("PER_MINUTE_LIMIT_PER_IP", "120"))))
    state.providers = providers
    state.http = http
    state.assets = AssetStore(ASSETS_DIR, store._con)
    state.public_base_url = PUBLIC_BASE_URL
    state.rng = random.Random() if SHUFFLE_TIES else None
    return state


async def scheduler(state: State) -> None:
    """Background loop running `probing.cycle` forever, every HEALTH_PROBE_HOURS.

    A deviation from the original brief (Task 12): the brief had the entire body
    of the cycle -- sync the catalogue, probe health, probe quality every N
    passes, prune -- copied line by line right here. That logic already exists,
    written and tested, as `probing.cycle(state, counter)` since Task 11 (which
    includes the "every N cycles" via `QUALITY_EVERY_N_CYCLES` and the 30-day
    retention right there): two copies of the same loop only guarantee they drift
    apart over time. This scheduler limits itself to invoking it.

    `cycle` deliberately does NOT catch exceptions (see its docstring in
    probing.py): that responsibility belongs to whoever calls it in an infinite
    loop. That is why the try/except lives HERE: one particular cycle blowing up
    (a provider down, the database locked, whatever) is logged and the loop keeps
    sleeping until the next one, instead of taking this background task down
    forever -- which would leave the process serving traffic on ever-staler
    metrics without anyone noticing, and without the service going down.
    """
    counter = 0
    while True:
        try:
            await cycle(state, counter)
        except Exception as e:  # the scheduler must never kill the service
            print(f"[probing] cycle {counter} failed: {e}", flush=True)
        counter += 1
        await asyncio.sleep(HEALTH_PROBE_HOURS * 3600)


state = build_state()
app = create_app(state)


@asynccontextmanager
async def _lifespan(_app):
    """Replaces the brief's `@app.on_event("startup"/"shutdown")` -- deprecated in
    the FastAPI/Starlette version this project ships (it already warns in the
    suite) -- with the recommended `lifespan`. `create_app` (Task 9) exposes no
    parameter to pass it in the constructor, so it is hooked up by reassigning
    `app.router.lifespan_context` after creating the app: Starlette only reads
    that attribute when the ASGI lifespan message arrives (when uvicorn starts),
    not at assignment time, so replacing it here is equivalent to having passed it
    to the constructor.
    """
    task = asyncio.create_task(scheduler(state))
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


app.router.lifespan_context = _lifespan
