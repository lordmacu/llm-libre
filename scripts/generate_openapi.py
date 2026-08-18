#!/usr/bin/env python3
"""Regenerate `openapi.json` (repo root) from the real FastAPI app -- Task 14.

It builds the SAME `create_app` that serves `/docs` in production, with a
minimal in-memory `State` (no network, no `.env`, no `/datos`) exactly like the
tests -- all `app.openapi()` cares about is the route tree and the
`openapi_extra`/`responses` each endpoint declares, not what data happens to be
loaded. That way `openapi.json` cannot drift from what the service actually
exposes: it is literally `app.openapi()`, not a hand-maintained copy.

Usage (see README.md):

    .venv/bin/python scripts/generate_openapi.py

CAREFUL: nothing runs this automatically, and nothing tests it. It went stale
once already -- it kept importing `llm_libre.almacen`/`Estado`/`crear_app` for
the whole of the internals rename, so it raised ImportError on every run and
`openapi.json` silently stayed frozen on an old contract. If you rename a module
or a symbol, this file is not covered by the suite: check it by hand.
"""

import json
from pathlib import Path

import httpx

from llm_libre.api import State, create_app
from llm_libre.providers import Provider
from llm_libre.proxy import Proxy
from llm_libre.storage import Storage

ROOT = Path(__file__).resolve().parents[1]


def _minimal_state() -> State:
    store = Storage(":memory:")
    store.create_schema()
    # A single filler provider: enough for create_app to build the app -- the
    # real catalogue and providers make no difference to the OpenAPI schema.
    providers = {"kilo": Provider("kilo", "free", "openai", "https://k.test",
                                  "", "/models", {}, [])}
    http = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={})))
    return State(store=store, proxy=Proxy(providers, store, http),
                 api_keys={"placeholder"}, daily_paid_cap=200)


def main() -> None:
    app = create_app(_minimal_state())
    schema = app.openapi()
    target = ROOT / "openapi.json"
    target.write_text(json.dumps(schema, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
                      encoding="utf-8")
    print(f"wrote {target} ({len(schema.get('paths', {}))} paths)")


if __name__ == "__main__":
    main()
