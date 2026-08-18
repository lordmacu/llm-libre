import asyncio
import contextlib
import os
import random
from contextlib import asynccontextmanager

import httpx

from llm_libre.api import State, create_app
from llm_libre.auth import PerKeyRateLimiter
from llm_libre.probing import cycle
from llm_libre.providers import load
from llm_libre.proxy import Proxy
from llm_libre.storage import Storage

# The environment variable NAMES stay as they are: they are operator surface,
# configured in Coolify and in .env, and renaming one silently changes behaviour
# on the next deploy with nothing to catch it.
YAML = os.getenv("PROVEEDORES_YAML", "proveedores.yaml")
DB_PATH = os.getenv("RUTA_DB", "/datos/llm-libre.sqlite3")
HEALTH_PROBE_HOURS = float(os.getenv("SONDEO_SALUD_HORAS", "5"))
# Draw between tied routes. On by default: with the quality battery giving 1.00
# to the entire catalogue (see quality_suite.py), a strict order ALWAYS sends
# traffic to the same route and burns one provider's quota while the others watch.
# Turning it off (ROTAR_EMPATES=false) restores the previous deterministic order,
# useful when debugging an odd response.
SHUFFLE_TIES = os.getenv("ROTAR_EMPATES", "true").strip().lower() not in ("false", "0", "no")


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
            "LLM_LIBRE_API_KEYS no esta definida (o esta vacia): sin al menos "
            "una llave el servicio arrancaria pero rechazaria el 100% de las "
            "peticiones a /v1/* con 401 para cualquier llamador, mientras "
            "/health seguiria informando 'ok'. Definila con al menos una "
            "llave, separadas por coma si son varias -- por ejemplo: "
            "LLM_LIBRE_API_KEYS=una-llave-larga-y-secreta")
    providers = load(YAML, dict(os.environ))
    store = Storage(DB_PATH)
    store.create_schema()
    http = httpx.AsyncClient()
    proxy = Proxy({p.id: p for p in providers}, store, http)
    state = State(store=store, proxy=proxy, api_keys=api_keys,
                  daily_paid_cap=int(os.getenv("TOPE_PAGO_DIARIO", "200")),
                  rate_limiter=PerKeyRateLimiter(int(os.getenv("LIMITE_POR_MINUTO", "60"))))
    state.providers = providers
    state.http = http
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
