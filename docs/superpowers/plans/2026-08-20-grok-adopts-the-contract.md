# Grok Adopts the Capability Contract — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the two residuals the capability-contract branch left open, then make `grok-proxy` publish the contract truthfully — which means first making four of its booleans true by exposing capabilities its backend already has.

**Architecture:** Two llm-libre hardening commits, then grok-proxy gains the standard surface (`web_search` on chat, `/v1/files`, `/v1/conversations`, `/v1/audio/speech`, `/v1/audio/transcriptions`) over RPCs its backend already reaches, then `/health` and `/v1/models` publish the contract, then `providers.yaml` opts grok in and retires three hand-written exceptions.

**Tech Stack:** Python 3.12+, FastAPI, httpx, pytest, hand-built protobuf over gRPC-web (`grok_backend.py`'s `_str_field`/`_int_field`/`_bool_field`/`_bytes_field`/`_raw_unary`/`_raw_stream`/`_decode_proto`/`_first_str`).

**Spec:** [`docs/superpowers/specs/2026-08-20-proxy-capability-contract-design.md`](../specs/2026-08-20-proxy-capability-contract-design.md) — §3.4 is the endpoint surface each boolean promises; §7 makes grok rollout step 3.

## Global Constraints

- **All code, comments, identifiers, log messages and docstrings in English.** No Spanish anywhere. `grok-proxy` has pre-existing Spanish docstrings and comments — do not translate them wholesale, but write everything NEW in English.
- Git identity must be `lordmacu` / `10134930+lordmacu@users.noreply.github.com`. **Never** add a `Co-Authored-By:` trailer.
- Contract schema version is `1`. Required `capabilities` keys, exactly eleven: `chat`, `streaming`, `tools`, `vision`, `images`, `audio_speech`, `audio_transcription`, `translate`, `search`, `files`, `conversations`.
- `auth.mode` is one of `anonymous`, `account`, `unknown`. **grok has no plan tiers**: its whole entitlement story is whether `GROK_SESSION_TOKEN` is configured, so it reports `account` or `anonymous` and `plan: null`.
- A boolean tracks **entitlement, not the meter**. grok's image quota running out stays a 429 plus cooldown; it must not flip `images`.
- An endpoint whose capability is `false` answers **`501 Not Implemented`**, never `404`, never `503`.
- **grok's repo is nested inside perplexity's.** `/Users/cristian/per` (the `perplexity-proxy` repo) tracks files under `grok/`, and `/Users/cristian/per/grok` is its own `grok-proxy` repo. Always `git -C /Users/cristian/per/grok`, never `git add` from `/Users/cristian/per`.
- llm-libre tests: `.venv/bin/python -m pytest` from `/Users/cristian/llm-libre`. grok tests: see Task 3, which creates the harness.

---

## File Structure

**`/Users/cristian/llm-libre`** (Tasks 1, 2, 11)

| File | Responsibility |
|---|---|
| `src/llm_libre/contract.py` | Distinguish an absent/malformed `auth` from a proxy that said `"unknown"`. |
| `src/llm_libre/probing.py` | Act on that distinction; stop the carry-over path from deactivating discovered routes. |
| `providers.yaml` | Opt grok in; retire three exceptions; stop overclaiming its context window. |

**`/Users/cristian/per/grok/docker-api`** (Tasks 3–10)

| File | Responsibility |
|---|---|
| `capabilities.py` | **New.** The eleven booleans and the `auth` block. The only place that knows what grok's session state means. |
| `grok_backend.py` | New RPC wrappers: TTS, STT, file listing. |
| `main.py` | `/health` contract, `/v1/models` metadata, the standard endpoints, the 501 gate. |
| `tests/` | **New.** pytest harness plus one file per task. |
| `CAPABILITIES.md` | **New.** What was measured and how. |

---

## Part 1 — Close the two residuals in llm-libre

### Task 1: an absent `auth` is not a proxy saying "unknown"

**Repo:** `/Users/cristian/llm-libre`

**Files:**
- Modify: `src/llm_libre/contract.py` (`_auth`, `Auth`)
- Modify: `src/llm_libre/probing.py` (`_read_contract`)
- Test: `tests/test_contract.py`, `tests/test_probing.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Auth.resolved: bool` — `True` when the proxy reported a usable `auth` block, `False` when it was absent or malformed and this parser supplied `mode="unknown"` itself.

> **Why:** the final review found that `_auth` degrades an absent or malformed `auth` to `mode="unknown"` *silently*, and `probing` treats `"unknown"` as "the proxy could not resolve its account this cycle" and refuses the contract. So a proxy publishing a perfect `capabilities` block but no `auth` key is rejected on every sweep and — with nothing stored — has its whole catalogue frozen. That is the C1 failure again, by another route, and grok is exactly the proxy that would hit it: it has no plan tiers and a minimal `auth` story.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_contract.py`:

```python
def test_a_reported_unknown_is_distinguished_from_an_absent_auth_block():
    # Two very different facts that both used to arrive as mode="unknown":
    # "the proxy asked its vendor and could not tell" (act on it) versus
    # "this proxy has no account concept and said nothing" (do not).
    said = parse_health("p", _doc(auth={"mode": "unknown"}))
    assert said.auth.mode == "unknown"
    assert said.auth.resolved is True

    absent = _doc()
    del absent["auth"]
    silent = parse_health("p", absent)
    assert silent.auth.mode == "unknown"
    assert silent.auth.resolved is False


def test_a_malformed_auth_block_is_not_resolved():
    c = parse_health("p", _doc(auth="pending"))
    assert c.auth.mode == "unknown"
    assert c.auth.resolved is False


def test_a_well_formed_auth_block_is_resolved():
    c = parse_health("p", _doc())
    assert c.auth.resolved is True
```

Append to `tests/test_probing.py`:

```python
async def test_a_contract_with_no_auth_block_is_still_usable():
    # grok's shape: eleven honest booleans, nothing to say about an account.
    store = _store()
    doc = {k: v for k, v in _HEALTH.items() if k != "auth"}
    await sync_catalogue(_routed(health=doc), [_chatgpt()], store, now=100.0)
    assert [r.key for r in store.active_routes()] == ["chatgpt/gpt-5-6"]
    assert store.get_contract("chatgpt") is not None


async def test_a_proxy_reporting_unknown_is_still_refused(caplog):
    store = _store()
    doc = {**_HEALTH, "auth": {**_HEALTH["auth"], "mode": "unknown"}}
    with caplog.at_level(logging.WARNING):
        await sync_catalogue(_routed(health=doc), [_chatgpt()], store, now=100.0)
    assert store.get_contract("chatgpt") is None
    assert "could not resolve" in caplog.text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_contract.py tests/test_probing.py -v -k "resolved or auth_block or reporting_unknown"`
Expected: FAIL — `AttributeError: 'Auth' object has no attribute 'resolved'`

- [ ] **Step 3: Write the implementation**

In `src/llm_libre/contract.py`, add to `Auth`:

```python
    # Whether the proxy actually TOLD us this, or whether `_auth` supplied
    # `mode="unknown"` because the block was absent or malformed. Two different
    # facts used to arrive identically: "I asked my vendor and could not tell"
    # (a transient condition worth refusing a sweep over) and "I have no account
    # concept and said nothing" (perfectly normal — grok has no plan tiers).
    # Refusing the second freezes that provider's catalogue forever, which is the
    # failure this whole contract exists to prevent.
    resolved: bool = False
```

In `_auth`, return `Auth(mode="unknown")` unchanged for the absent/malformed paths — `resolved` defaults to `False` — and pass `resolved=True` on the path where a well-formed dict was read, including when its `mode` string was unrecognised and downgraded (the proxy did speak; it just said something this version does not know).

In `src/llm_libre/probing.py`, change the `"unknown"` refusal to also require that the proxy actually said it:

```python
        if contract.auth.mode == "unknown" and contract.auth.resolved:
```

and extend the surrounding comment to name the distinction.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS — 772 + 5 = 777 passed, 8 skipped, 1 warning.

- [ ] **Step 5: Commit**

```bash
cd /Users/cristian/llm-libre
git add src/llm_libre/contract.py src/llm_libre/probing.py tests/test_contract.py tests/test_probing.py
git commit -m "fix(contract): a proxy with no account concept is not a proxy that failed to resolve one"
```

---

### Task 2: a carried-over contract must not deactivate discovered routes

**Repo:** `/Users/cristian/llm-libre`

**Files:**
- Modify: `src/llm_libre/probing.py` (`sync_catalogue`'s fixed-models pass)
- Test: `tests/test_probing.py`

**Interfaces:**
- Consumes: the carry-over path from Task 10 of the previous plan.
- Produces: no new names.

> **Why:** the fix wave's carry-over path reaches `store.upsert_routes(fixed_routes(...), now, deactivate_missing=True, provider=p.id)`, which switches off every route of that provider whose `last_seen < now` — i.e. all discovered ones. They are normally restored moments later by the `/models` pass, but in the common failure mode (the whole proxy unreachable, so `/health` *and* `/models` both fail) `/models` `continue`s and they stay off until the next successful sweep — up to `HEALTH_PROBE_HOURS`. Before the wave a `/health` failure short-circuited ahead of the fixed pass, so a total outage left the catalogue intact, which is what spec §5.1 asks for.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_probing.py`:

```python
async def test_a_total_outage_does_not_deactivate_the_discovered_routes():
    # /health carried over from the stored contract AND /models failing is the
    # shape of a proxy that is simply down. Keeping what is known beats erasing
    # it -- the routes are unusable during the outage either way, but they must
    # come back the moment it ends, not a sweep interval later.
    store = _store()
    await sync_catalogue(_routed(), [_chatgpt()], store, now=100.0)
    before = [r.key for r in store.active_routes()]
    assert before, "the first sweep must have discovered something to lose"

    def dead(req):
        return httpx.Response(503)

    await sync_catalogue(httpx.AsyncClient(transport=httpx.MockTransport(dead)),
                         [_chatgpt()], store, now=200.0)
    assert [r.key for r in store.active_routes()] == before
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_probing.py -v -k total_outage`
Expected: FAIL — the discovered routes come back deactivated, so the lists differ.

- [ ] **Step 3: Write the implementation**

In `sync_catalogue`, thread whether the contract was carried over into the fixed-models pass and suppress only its deactivation:

```python
        if p.fixed_models:
            routes = fixed_routes(p, contract=contract)
            # `deactivate_missing` is what switches off routes this pass did not
            # see -- and on a carried-over contract this pass has not yet seen
            # ANY discovered route, because /models has not run. If /models then
            # fails too (the whole proxy is down, the common case), the sweep
            # would leave the catalogue emptied of everything except the fixed
            # entries, for a full interval. Deactivation is the /models pass's
            # job; the fixed pass only ever adds.
            store.upsert_routes(routes, now, deactivate_missing=not carried_over,
                                provider=p.id)
            total += len(routes)
```

Name the flag where the contract is resolved (Task 1's `_read_contract` returns fresh / carried / skip — return the provenance alongside, or set a local `carried_over` at the call site), and extend `sync_catalogue`'s docstring, which documents its failure discipline case by case.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS — 778 passed, 8 skipped, 1 warning.

- [ ] **Step 5: Commit**

```bash
cd /Users/cristian/llm-libre
git add src/llm_libre/probing.py tests/test_probing.py
git commit -m "fix(probing): a carried-over contract adds fixed routes, it does not retire discovered ones"
```

---

## Part 2 — grok gains the standard surface

### Task 3: the test harness and the capability module

**Repo:** `/Users/cristian/per/grok`

**Files:**
- Create: `docker-api/capabilities.py`, `docker-api/requirements-dev.txt`, `docker-api/pytest.ini`, `docker-api/tests/__init__.py`, `docker-api/tests/test_capabilities.py`

**Interfaces:**
- Consumes: `os.environ["GROK_SESSION_TOKEN"]`.
- Produces: `capabilities.REQUIRED_CAPABILITIES: tuple[str, ...]`, `capabilities.SessionState(mode: str)`, `capabilities.snapshot() -> SessionState`, `capabilities.effective(state) -> dict[str, bool]`, `capabilities.auth_block(state) -> dict`.

> **grok has no plan tiers.** `chatgpt-proxy` needed a cached vendor round trip to learn whether the account was on `go` or `free`; grok's entire entitlement story is whether a session token is configured. So `snapshot()` reads one environment variable and needs no cache, no lock, and no vendor call — and `auth_block` reports `plan: null`, `subscription_active: false`, `expires_at: null`, which is the truth rather than a placeholder.
>
> The four booleans Tasks 5–9 turn true start `False` here. Each of those tasks flips exactly one, in the same commit that makes it true. That ordering is the point: the contract must never claim something the endpoint cannot yet do.

- [ ] **Step 1: Write the failing tests**

Create `docker-api/requirements-dev.txt`:

```
-r requirements.txt
pytest==8.3.4
```

Create `docker-api/pytest.ini`:

```
[pytest]
```

Create empty `docker-api/tests/__init__.py`, then `docker-api/tests/test_capabilities.py`:

```python
import capabilities as cap


ACCOUNT = cap.SessionState(mode="account")
ANON = cap.SessionState(mode="anonymous")


def test_every_required_key_is_present_and_boolean():
    for state in (ACCOUNT, ANON):
        e = cap.effective(state)
        assert set(e) == set(cap.REQUIRED_CAPABILITIES)
        assert all(isinstance(v, bool) for v in e.values())


def test_chat_tools_and_vision_are_true_with_a_session():
    e = cap.effective(ACCOUNT)
    assert e["chat"] and e["streaming"] and e["tools"] and e["vision"] and e["images"]


def test_nothing_works_without_a_session():
    # Every grok RPC travels on the session token; without it the proxy has no
    # backend at all, so claiming any capability would be a lie.
    assert not any(cap.effective(ANON).values())


def test_the_capabilities_this_proxy_does_not_serve_yet_are_false():
    e = cap.effective(ACCOUNT)
    assert not e["search"]
    assert not e["files"]
    assert not e["conversations"]
    assert not e["audio_speech"]
    assert not e["audio_transcription"]


def test_translate_is_false_because_grok_has_no_translate_endpoint():
    assert cap.effective(ACCOUNT)["translate"] is False


def test_the_auth_block_reports_no_plan():
    # grok has no tiers. Reporting a plan name here would invent one.
    b = cap.auth_block(ACCOUNT)
    assert b == {"mode": "account", "plan": None,
                 "subscription_active": False, "expires_at": None}


def test_snapshot_follows_the_session_token(monkeypatch):
    monkeypatch.setenv("GROK_SESSION_TOKEN", "t")
    assert cap.snapshot().mode == "account"
    monkeypatch.delenv("GROK_SESSION_TOKEN")
    assert cap.snapshot().mode == "anonymous"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /Users/cristian/per/grok/docker-api
python3.12 -m venv .venv && .venv/bin/python -m pip install -q -r requirements-dev.txt
.venv/bin/python -m pytest tests/test_capabilities.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'capabilities'`

> If `python3.12` is unavailable, use whichever interpreter satisfies `requirements.txt`; record which one in your report so later tasks use the same.

- [ ] **Step 3: Write the implementation**

Create `docker-api/capabilities.py`:

```python
"""What this proxy can actually do right now.

Spec: the proxy capability contract, llm-libre
docs/superpowers/specs/2026-08-20-proxy-capability-contract-design.md

THE RULE: a boolean says what a request sent right now would ACHIEVE, not what
this codebase implements. Where the two differ, the endpoint is the liar and
this module is the correction.

Where the rule STOPS: a boolean tracks entitlement, not the meter. grok's image
quota running out is a 429 the gateway already handles with a cooldown and
recovers from on its own; it must never flip `images`. The dividing line is
durability -- if a fresh request tomorrow would still be refused for the same
reason, it belongs in the boolean.

Unlike chatgpt-proxy, there is no plan to resolve: grok has no tiers, and every
RPC travels on one session token. So the whole entitlement story is whether that
token is configured, `snapshot()` is a single environment read, and `auth_block`
reports `plan: null` rather than inventing a tier name.
"""
import os
from dataclasses import dataclass

# The eleven keys the contract requires, byte-for-byte the set the gateway
# validates against (llm_libre.contract.REQUIRED_CAPABILITIES). Duplicated
# rather than imported because the two live in different repos and deploy
# independently; tests/test_health_contract.py is what keeps them honest.
REQUIRED_CAPABILITIES = (
    "chat", "streaming", "tools", "vision", "images",
    "audio_speech", "audio_transcription", "translate",
    "search", "files", "conversations",
)


@dataclass(frozen=True)
class SessionState:
    mode: str          # "account" | "anonymous"


def snapshot() -> SessionState:
    """Whether this process has a session token. That is all grok's auth is."""
    token = (os.environ.get("GROK_SESSION_TOKEN") or "").strip()
    return SessionState(mode="account" if token else "anonymous")


def auth_block(state: SessionState) -> dict:
    """The contract's informational `auth` block.

    Every field except `mode` is null on purpose: grok sells no tiers, so there
    is no plan to name and no subscription to expire. Reporting a placeholder
    here would be the same class of lie the contract exists to end.
    """
    return {"mode": state.mode, "plan": None,
            "subscription_active": False, "expires_at": None}


def effective(state: SessionState) -> dict:
    """The eleven booleans. Every value below was measured, not assumed.

      - `tools` is TRUE and this is the unusual one: grok returns real
        tool_calls natively, measured 6/6 across three cases. Its own gateway
        entry records the measurement.
      - `vision` is TRUE, served inside /v1/chat/completions: image_url content
        parts are uploaded and the request is steered to a vision-capable model.
      - `images` is TRUE via the imagine-agent-mode family, the only grok models
        that generate.
      - `search`, `files`, `conversations`, `audio_speech` and
        `audio_transcription` are FALSE *for now*: the backend can do all five,
        but not yet at the paths §3.4 of the contract promises. Each flips in
        the same commit that makes its endpoint real.
      - `translate` is FALSE and stays false: grok has no translate endpoint,
        and routing it through a chat turn would be a different capability
        wearing this one's name.
    """
    live = state.mode == "account"
    return {
        "chat":                live,
        "streaming":           live,
        "tools":               live,
        "vision":              live,
        "images":              live,
        "audio_speech":        False,
        "audio_transcription": False,
        "translate":           False,
        "search":              False,
        "files":               False,
        "conversations":       False,
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
cd /Users/cristian/per/grok
git add docker-api/capabilities.py docker-api/requirements-dev.txt docker-api/pytest.ini docker-api/tests/
git commit -m "feat(capabilities): what this proxy can actually do, and what it cannot yet"
```

---

### Task 4: `/health` and `/v1/models` publish the contract

**Repo:** `/Users/cristian/per/grok`

**Files:**
- Modify: `docker-api/main.py` (the `/health` handler; the `/v1/models` handler)
- Test: `docker-api/tests/test_health_contract.py`, `docker-api/tests/test_models_metadata.py`

**Interfaces:**
- Consumes: `capabilities.snapshot`, `effective`, `auth_block`, `REQUIRED_CAPABILITIES`.
- Produces: a `/health` body carrying `contract: 1`, `provider: "grok"`, `auth`, `capabilities`; `/v1/models` entries carrying `context_window`, `max_output_tokens`, `capabilities: {tools, vision, images}`.

> **The per-model block is the point of this task.** `providers.yaml` currently hand-declares three `exceptions` — `imagine-agent-mode`, `imagine-agent-mode-dev`, `imagine-agent-mode-grok-4-5` — as `{tools: false, vision: false, images: true}`. Publishing that per model retires all three, which is what the final whole-branch review flagged as structurally identical to the `gpt-image-1` defect it caught.
>
> **grok publishes no context window anywhere** — verified: 26 advertised models, no `context_window`, no token field of any kind. So `/v1/models` must omit `context_window` rather than invent one, and the gateway falls back to `default_capabilities` (Task 11 corrects that comment rather than the number).

- [ ] **Step 1: Write the failing tests**

Create `docker-api/tests/test_health_contract.py`:

```python
from fastapi.testclient import TestClient

import capabilities as cap
import main

REQUIRED = set(cap.REQUIRED_CAPABILITIES)


def _health(monkeypatch, state):
    monkeypatch.setattr(cap, "snapshot", lambda: state)
    with TestClient(main.app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    return r.json()


def test_health_declares_the_contract(monkeypatch):
    body = _health(monkeypatch, cap.SessionState(mode="account"))
    assert body["contract"] == 1
    assert body["provider"] == "grok"


def test_capabilities_are_exactly_the_required_booleans(monkeypatch):
    body = _health(monkeypatch, cap.SessionState(mode="account"))
    assert set(body["capabilities"]) == REQUIRED
    assert all(isinstance(v, bool) for v in body["capabilities"].values())


def test_the_auth_block_names_no_plan(monkeypatch):
    body = _health(monkeypatch, cap.SessionState(mode="account"))
    assert body["auth"]["mode"] == "account"
    assert body["auth"]["plan"] is None


def test_health_needs_no_api_key(monkeypatch):
    # The gateway sweeps this on a schedule and it is the container health
    # check; requiring a key would make both depend on configuration they do
    # not carry.
    monkeypatch.setattr(main, "API_KEY", "a-secret")
    body = _health(monkeypatch, cap.SessionState(mode="account"))
    assert body["contract"] == 1


def test_the_legacy_fields_survive(monkeypatch):
    body = _health(monkeypatch, cap.SessionState(mode="account"))
    assert body["status"] == "ok"
    assert "version" in body
    assert "session_configured" in body
    assert "high_rate_pool_size" in body
```

Create `docker-api/tests/test_models_metadata.py`:

```python
from fastapi.testclient import TestClient

import capabilities as cap
import main

PER_MODEL = {"tools", "vision", "images"}
IMAGINE = {"imagine-agent-mode", "imagine-agent-mode-dev",
           "imagine-agent-mode-grok-4-5"}


def _models(monkeypatch):
    monkeypatch.setattr(cap, "snapshot",
                        lambda: cap.SessionState(mode="account"))
    monkeypatch.setattr(main, "API_KEY", "")
    with TestClient(main.app) as client:
        r = client.get("/v1/models")
    assert r.status_code == 200
    return r.json()["data"]


def test_every_model_carries_per_model_capabilities(monkeypatch):
    for m in _models(monkeypatch):
        assert set(m["capabilities"]) == PER_MODEL
        assert all(isinstance(v, bool) for v in m["capabilities"].values())


def test_only_the_imagine_family_claims_images(monkeypatch):
    models = _models(monkeypatch)
    drawing = {m["id"] for m in models if m["capabilities"]["images"]}
    assert drawing == IMAGINE & {m["id"] for m in models}


def test_the_imagine_family_claims_neither_tools_nor_vision(monkeypatch):
    # This is what retires the three hand-written exceptions in the gateway's
    # providers.yaml. It has to be exact, not approximately right.
    for m in _models(monkeypatch):
        if m["id"] in IMAGINE:
            assert m["capabilities"]["tools"] is False
            assert m["capabilities"]["vision"] is False


def test_the_chat_models_claim_tools_and_vision(monkeypatch):
    for m in _models(monkeypatch):
        if m["id"] not in IMAGINE:
            assert m["capabilities"]["tools"] is True
            assert m["capabilities"]["vision"] is True


def test_no_model_invents_a_context_window(monkeypatch):
    # grok publishes no context figure anywhere. Omitting the key lets the
    # gateway fall back to its declared floor; inventing one would be the
    # 128000-against-a-real-52815 mistake again.
    for m in _models(monkeypatch):
        assert "context_window" not in m or m["context_window"] is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_health_contract.py tests/test_models_metadata.py -v`
Expected: FAIL — `KeyError: 'contract'` and `KeyError: 'capabilities'`.

- [ ] **Step 3: Write the implementation**

Add `import capabilities` next to `main.py`'s other imports.

Replace the `/health` handler body, keeping every existing field:

```python
@app.get("/health")
def health():
    state = capabilities.snapshot()
    return {
        "status":  "ok",
        "version": APP_VERSION,
        # The capability contract (llm-libre spec 2026-08-20). `capabilities`
        # is EFFECTIVE: what a request would achieve right now, so the gateway
        # reads one boolean instead of learning what a grok session is.
        "contract": 1,
        "provider": "grok",
        "auth": capabilities.auth_block(state),
        "capabilities": capabilities.effective(state),
        # Kept: existing dashboards and the container health check read these.
        "session_configured":  state.mode == "account",
        "high_rate_pool_size": len(backend.HIGH_RATE_POOL),
    }
```

In the `/v1/models` handler, stamp each entry of the list it returns, immediately before the return:

```python
    # --- capability contract: per-model metadata -----------------------------
    # The provider-level block cannot say that the imagine family draws and
    # neither chats nor sees. A per-model value may only NARROW the
    # provider-level one -- `and provider_level[...]` is that rule applied at
    # the source, so an anonymous process reports False for everything.
    #
    # `context_window` is deliberately ABSENT: grok publishes no context figure
    # for any model, and inventing one here is exactly the mistake this contract
    # exists to end.
    provider_level = capabilities.effective(capabilities.snapshot())
    for entry in data:
        draws = entry["id"] in _IMAGINE_MODELS
        entry["max_output_tokens"] = 0 if draws else 8192
        entry["capabilities"] = {
            "tools":  (not draws) and provider_level["tools"],
            "vision": (not draws) and provider_level["vision"],
            "images": draws and provider_level["images"],
        }
```

Define `_IMAGINE_MODELS` as a module-level frozenset of the three ids beside the other model constants, with a comment that it is the only grok family that generates. If an equivalent constant already exists in `grok_backend.py`, import and reuse it rather than defining a second.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS, 17 tests.

- [ ] **Step 5: Commit**

```bash
cd /Users/cristian/per/grok
git add docker-api/main.py docker-api/tests/test_health_contract.py docker-api/tests/test_models_metadata.py
git commit -m "feat(health): publish the capability contract, and per-model capabilities on /v1/models"
```

---

### Task 5: `web_search`, and search on by default

**Repo:** `/Users/cristian/per/grok`

**Files:**
- Modify: `docker-api/main.py` (`ChatRequest`, the two chat handlers, `/grok/conversations/{id}/messages`)
- Modify: `docker-api/capabilities.py` (`search` → `live`)
- Test: `docker-api/tests/test_web_search.py`

**Interfaces:**
- Consumes: `backend.stream_chat(..., disable_search=...)`, unchanged.
- Produces: `ChatRequest.web_search: bool | None`.

> **This fixes a real quality problem, not just a naming one.** grok's native field is `disable_search`, and `main.py` defaults it to `True` — so every grok answer today is ungrounded unless a caller explicitly asks otherwise, and the gateway never sends the field at all. The contract's `search` promises `web_search: bool` with **search on by default** (§3.4), which is also what every other provider does.
>
> Keep `disable_search` working: it is this proxy's native parameter and something may already send it. `web_search` wins when both are present, because it is the standard one.

- [ ] **Step 1: Write the failing test**

Create `docker-api/tests/test_web_search.py`:

```python
import capabilities as cap
import main


def _resolve(**body):
    """What the handlers compute for `disable_search` from a request body."""
    return main.resolve_disable_search(main.ChatRequest(**body))


def test_search_is_on_by_default():
    # The behaviour change: grok used to answer ungrounded unless asked.
    assert _resolve(messages=[{"role": "user", "content": "hi"}]) is False


def test_web_search_false_turns_it_off():
    assert _resolve(messages=[{"role": "user", "content": "hi"}],
                    web_search=False) is True


def test_web_search_true_turns_it_on():
    assert _resolve(messages=[{"role": "user", "content": "hi"}],
                    web_search=True) is False


def test_the_native_disable_search_still_works():
    assert _resolve(messages=[{"role": "user", "content": "hi"}],
                    disable_search=True) is True


def test_web_search_wins_over_the_native_field():
    # Two ways to say the same thing; the standard one is authoritative.
    assert _resolve(messages=[{"role": "user", "content": "hi"}],
                    web_search=True, disable_search=True) is False


def test_the_contract_now_claims_search():
    assert cap.effective(cap.SessionState(mode="account"))["search"] is True
    assert cap.effective(cap.SessionState(mode="anonymous"))["search"] is False
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_web_search.py -v`
Expected: FAIL — `main` has no `resolve_disable_search`, and `ChatRequest` rejects `web_search`.

- [ ] **Step 3: Write the implementation**

In `main.py`, add the field to `ChatRequest` beside `disable_search`:

```python
    # The contract's standard name (llm-libre spec §3.4), and the default flips
    # with it: search ON unless a caller says otherwise, which is what every
    # other provider does and what a caller reasonably expects. `disable_search`
    # is grok's own inverted field and keeps working for whatever already sends
    # it; `web_search` wins when both arrive, because it is the standard one.
    web_search: Optional[bool] = None
```

Change `disable_search`'s default from `True` to `None`, and add the resolver next to it:

```python
def resolve_disable_search(req) -> bool:
    """The native `disable_search` value for a request, from either field.

    Returns a bool because that is what the backend takes. The precedence is
    `web_search` (standard, inverted) over `disable_search` (native) over the
    default, and the default is now "search on" -- see the field comment.
    """
    if req.web_search is not None:
        return not req.web_search
    if req.disable_search is not None:
        return req.disable_search
    return False
```

Replace every `disable_search=req.disable_search` call site with
`disable_search=resolve_disable_search(req)`. There are six, in the two chat
paths and the conversation-message paths — find them by searching, and change
all of them.

In `capabilities.py`, change `"search": False` to `"search": live` and rewrite
that bullet in the docstring to say it is served through `web_search` on chat
completions, with search on by default.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS, 23 tests.

- [ ] **Step 5: Commit**

```bash
cd /Users/cristian/per/grok
git add docker-api/main.py docker-api/capabilities.py docker-api/tests/test_web_search.py
git commit -m "feat(search): web_search as the standard name, and grounded answers by default"
```

---

### Task 6: `/v1/files`

**Repo:** `/Users/cristian/per/grok`

**Files:**
- Modify: `docker-api/grok_backend.py` (add `list_files`)
- Modify: `docker-api/main.py` (four handlers), `docker-api/capabilities.py`
- Test: `docker-api/tests/test_files.py`

**Interfaces:**
- Consumes: `backend.upload_file(filename, content, mime_type)` — already exists.
- Produces: `backend.list_files(limit: int = 100) -> list[dict]`; `POST /v1/files`, `GET /v1/files`, `GET /v1/files/{file_id}`, `DELETE /v1/files/{file_id}`.

> `/grok/files` already uploads. This task adds the OpenAI-shaped surface the contract's `files` boolean promises, and the listing RPC the backend does not wrap yet: `/grok_api_v2.FilesService/ListFiles`.

- [ ] **Step 1: Write the failing test**

Create `docker-api/tests/test_files.py`:

```python
import io

from fastapi.testclient import TestClient

import capabilities as cap
import main

UPLOADED = {"file_id": "file-abc", "mime_type": "text/plain",
            "storage_path": "u/1", "created_at": "2026-08-20T00:00:00Z"}


def _client(monkeypatch):
    monkeypatch.setattr(cap, "snapshot",
                        lambda: cap.SessionState(mode="account"))
    monkeypatch.setattr(main, "API_KEY", "")
    return TestClient(main.app)


def test_upload_returns_the_openai_shape(monkeypatch):
    monkeypatch.setattr(main.backend, "upload_file", lambda *a, **k: UPLOADED)
    with _client(monkeypatch) as c:
        r = c.post("/v1/files", files={"file": ("a.txt", io.BytesIO(b"hi"),
                                                "text/plain")})
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "file-abc"
    assert body["object"] == "file"
    assert body["purpose"] == "assistants"
    assert body["filename"] == "a.txt"
    assert isinstance(body["bytes"], int)


def test_listing_returns_a_list_object(monkeypatch):
    monkeypatch.setattr(main.backend, "list_files",
                        lambda limit=100: [dict(UPLOADED, filename="a.txt")])
    with _client(monkeypatch) as c:
        r = c.get("/v1/files")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "file-abc"


def test_a_single_file_is_addressable(monkeypatch):
    monkeypatch.setattr(main.backend, "list_files",
                        lambda limit=100: [dict(UPLOADED, filename="a.txt")])
    with _client(monkeypatch) as c:
        r = c.get("/v1/files/file-abc")
    assert r.status_code == 200
    assert r.json()["id"] == "file-abc"


def test_an_unknown_file_is_404(monkeypatch):
    monkeypatch.setattr(main.backend, "list_files", lambda limit=100: [])
    with _client(monkeypatch) as c:
        r = c.get("/v1/files/file-nope")
    assert r.status_code == 404


def test_delete_returns_the_openai_shape(monkeypatch):
    monkeypatch.setattr(main.backend, "delete_file", lambda fid: {"deleted": True})
    with _client(monkeypatch) as c:
        r = c.delete("/v1/files/file-abc")
    assert r.status_code == 200
    assert r.json() == {"id": "file-abc", "object": "file", "deleted": True}


def test_the_contract_now_claims_files():
    assert cap.effective(cap.SessionState(mode="account"))["files"] is True
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_files.py -v`
Expected: FAIL — 404 on every route; `main.backend` has no `list_files`.

- [ ] **Step 3: Write the implementation**

In `grok_backend.py`, next to `upload_file`, add the listing wrapper, following
the file's established idiom (`_raw_unary`, `_decode_proto`, `_first_str`):

```python
def list_files(limit: int = 100) -> list[dict]:
    """Calls grok_api_v2.FilesService/ListFiles.

    ListFilesRequest: f1 = limit. The response repeats file entries on f1, each
    shaped like UploadFile's reply (f1=file_id, f2=mime_type, f4=storage_path,
    f6=created_at timestamp) -- see upload_file, which parses the same shape.
    """
    raw = _raw_unary("/grok_api_v2.FilesService/ListFiles",
                     _int_field(1, limit), timeout=15)
    f = _decode_proto(raw)
    files = []
    for kind, val in f.get(1, []):
        data = val.encode("latin-1") if kind == "str" else (val if kind == "raw" else b"")
        inner = _decode_proto(data)
        created_at = None
        for k2, v2 in inner.get(6, []):
            blob = v2 if k2 == "raw" else (v2.encode("latin-1") if k2 == "str" else b"")
            created_at = _ts_to_iso(blob)
        files.append({
            "file_id":      _first_str(inner, 1),
            "mime_type":    _first_str(inner, 2),
            "storage_path": _first_str(inner, 4),
            "created_at":   created_at,
        })
    return files


def delete_file(file_id: str) -> dict:
    """Calls grok_api_v2.AssetRepository/DeleteAsset. f1 = the file id."""
    _raw_unary("/grok_api_v2.AssetRepository/DeleteAsset",
               _str_field(1, file_id), timeout=10)
    return {"deleted": True}
```

> **Verify both RPC paths against the decompiled APK before trusting them**:
> `grep -rho "/grok_api_v2\.\(FilesService\|AssetRepository\)/[A-Za-z]*" /Users/cristian/per/grok/jadx_out | sort -u`.
> If `DeleteAsset` is not the right call for a chat-uploaded file, report it
> rather than guessing — `files` can ship without delete, with that one endpoint
> answering 501 and a note in `CAPABILITIES.md`.

In `main.py`, add the four handlers, each translating to the OpenAI file shape
(`id`, `object: "file"`, `bytes`, `created_at`, `filename`, `purpose`). Keep
`/grok/files` exactly as it is — it is this proxy's native surface and something
may already use it.

In `capabilities.py`, change `"files": False` to `"files": live` and update the
docstring bullet.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS, 29 tests.

- [ ] **Step 5: Commit**

```bash
cd /Users/cristian/per/grok
git add docker-api/main.py docker-api/grok_backend.py docker-api/capabilities.py docker-api/tests/test_files.py
git commit -m "feat(files): the OpenAI-shaped file surface the contract promises"
```

---

### Task 7: `/v1/conversations`

**Repo:** `/Users/cristian/per/grok`

**Files:**
- Modify: `docker-api/main.py` (two handlers), `docker-api/capabilities.py`
- Test: `docker-api/tests/test_conversations.py`

**Interfaces:**
- Consumes: `backend.list_conversations(limit, cursor)`, `backend.get_conversation(conv_id)` — both already exist.
- Produces: `GET /v1/conversations`, `GET /v1/conversations/{conversation_id}`.

> Pure aliasing: `/grok/conversations` already does this. The contract's §3.4 promises listing and detail at the standard paths, and this is what lets `conversations` be true.

- [ ] **Step 1: Write the failing test**

Create `docker-api/tests/test_conversations.py`:

```python
from fastapi.testclient import TestClient

import capabilities as cap
import main

CONV = {"conversation_id": "c-1", "title": "hello",
        "create_time": "2026-08-20T00:00:00Z"}


def _client(monkeypatch):
    monkeypatch.setattr(cap, "snapshot",
                        lambda: cap.SessionState(mode="account"))
    monkeypatch.setattr(main, "API_KEY", "")
    return TestClient(main.app)


def test_listing_returns_a_list_object(monkeypatch):
    monkeypatch.setattr(main.backend, "list_conversations",
                        lambda **k: {"conversations": [CONV], "next_cursor": None})
    with _client(monkeypatch) as c:
        r = c.get("/v1/conversations")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "c-1"


def test_detail_is_addressable(monkeypatch):
    monkeypatch.setattr(main.backend, "get_conversation", lambda cid: CONV)
    with _client(monkeypatch) as c:
        r = c.get("/v1/conversations/c-1")
    assert r.status_code == 200
    assert r.json()["id"] == "c-1"


def test_an_unknown_conversation_is_404(monkeypatch):
    monkeypatch.setattr(main.backend, "get_conversation", lambda cid: {})
    with _client(monkeypatch) as c:
        r = c.get("/v1/conversations/nope")
    assert r.status_code == 404


def test_the_native_surface_still_works(monkeypatch):
    monkeypatch.setattr(main.backend, "list_conversations",
                        lambda **k: {"conversations": [CONV], "next_cursor": None})
    with _client(monkeypatch) as c:
        r = c.get("/grok/conversations")
    assert r.status_code == 200


def test_the_contract_now_claims_conversations():
    assert cap.effective(cap.SessionState(mode="account"))["conversations"] is True
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_conversations.py -v`
Expected: FAIL — 404 on both `/v1/conversations` routes.

- [ ] **Step 3: Write the implementation**

Add the two handlers in `main.py`, mapping `conversation_id` → `id` and wrapping
the list as `{"object": "list", "data": [...]}`. Return 404 when the backend
gives an empty detail. Keep the `/grok/conversations*` family untouched.

In `capabilities.py`, change `"conversations": False` to `"conversations": live`
and update the docstring bullet.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS, 34 tests.

- [ ] **Step 5: Commit**

```bash
cd /Users/cristian/per/grok
git add docker-api/main.py docker-api/capabilities.py docker-api/tests/test_conversations.py
git commit -m "feat(conversations): listing and detail at the standard paths"
```

---

### Task 8: `/v1/audio/speech` — text to speech

**Repo:** `/Users/cristian/per/grok`

**Files:**
- Modify: `docker-api/grok_backend.py` (add `text_to_speech`), `docker-api/main.py`, `docker-api/capabilities.py`
- Test: `docker-api/tests/test_audio_speech.py`

**Interfaces:**
- Consumes: `_str_field`, `_int_field`, `_bool_field`, `_raw_stream`, `_decode_proto` — all existing.
- Produces: `backend.text_to_speech(text, voice_id, language, codec) -> tuple[bytes, str]`; `POST /v1/audio/speech`.

> **Recovered from the decompiled APK, not guessed.** `/grok_api.Chat/TextToSpeech` is a **streaming** call (`GrpcChatClient.java:361` — `newStreamingCall`), taking `TextToSpeechRequest` and returning a stream of `AudioChunk`.
>
> `TextToSpeechRequest` tags: `1 text` (string), `2 voice_id` (string), `3 language` (string), `4 codec` (string), `5 sample_rate` (int32), `6 bit_rate` (int32), `7 optimize_streaming_latency` (int32), `8 text_normalization` (bool), `9 speed` (float).
>
> `AudioChunk` tags: `1 data` (bytes), `2 content_type` (string).
>
> **Send only the fields you need.** `speed` is a float and `grok_backend.py` has no float-field helper; it is optional, so omit it rather than adding one. Concatenate every chunk's field 1 in order; take the content type from the first chunk that carries field 2, defaulting to `audio/mpeg`.
>
> Voice ids come from the existing `/grok/voices` (`Voice/ListVoices`). Default to the first top voice rather than hardcoding an id that may not exist on this account.

- [ ] **Step 1: Write the failing test**

Create `docker-api/tests/test_audio_speech.py`:

```python
from fastapi.testclient import TestClient

import capabilities as cap
import main

MP3 = b"ID3\x03fake-audio-bytes"


def _client(monkeypatch):
    monkeypatch.setattr(cap, "snapshot",
                        lambda: cap.SessionState(mode="account"))
    monkeypatch.setattr(main, "API_KEY", "")
    monkeypatch.setattr(main.backend, "text_to_speech",
                        lambda *a, **k: (MP3, "audio/mpeg"))
    return TestClient(main.app)


def test_speech_returns_raw_audio_bytes(monkeypatch):
    # The contract promises bytes, not a JSON envelope: every OpenAI client
    # writes this response body straight to a file.
    with _client(monkeypatch) as c:
        r = c.post("/v1/audio/speech", json={"input": "hola"})
    assert r.status_code == 200
    assert r.content == MP3
    assert r.headers["content-type"].startswith("audio/mpeg")


def test_an_empty_input_is_400(monkeypatch):
    with _client(monkeypatch) as c:
        r = c.post("/v1/audio/speech", json={"input": "   "})
    assert r.status_code == 400


def test_the_voice_is_passed_through(monkeypatch):
    seen = {}

    def fake(text, voice_id=None, **k):
        seen["voice"] = voice_id
        return MP3, "audio/mpeg"

    with _client(monkeypatch) as c:
        monkeypatch.setattr(main.backend, "text_to_speech", fake)
        c.post("/v1/audio/speech", json={"input": "hola", "voice": "ara"})
    assert seen["voice"] == "ara"


def test_the_contract_now_claims_audio_speech():
    assert cap.effective(cap.SessionState(mode="account"))["audio_speech"] is True
```

And a unit test for the frame builder, in the same file:

```python
def test_the_request_frame_carries_the_recovered_tags():
    import grok_backend as gb
    frame = gb.build_tts_request("hola", voice_id="ara", language="es")
    fields = gb._decode_proto(frame)
    assert gb._first_str(fields, 1) == "hola"     # text
    assert gb._first_str(fields, 2) == "ara"      # voice_id
    assert gb._first_str(fields, 3) == "es"       # language
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_audio_speech.py -v`
Expected: FAIL — 404 on the route, and `grok_backend` has no `build_tts_request`.

- [ ] **Step 3: Write the implementation**

In `grok_backend.py`:

```python
# TextToSpeechRequest, recovered from the decompiled APK
# (jadx_out/sources/grok_api/TextToSpeechRequest.java). Only the fields this
# proxy sends are listed; `speed` (f9) is a float and there is no float helper
# here, so it is deliberately omitted rather than approximated.
_TTS_TEXT, _TTS_VOICE, _TTS_LANGUAGE, _TTS_CODEC = 1, 2, 3, 4


def build_tts_request(text: str, voice_id: str = "", language: str = "en",
                      codec: str = "mp3") -> bytes:
    """The TextToSpeechRequest frame. Separated from the call so the wire shape
    can be tested without a network."""
    return (_str_field(_TTS_TEXT, text)
            + _str_field(_TTS_VOICE, voice_id)
            + _str_field(_TTS_LANGUAGE, language)
            + _str_field(_TTS_CODEC, codec))


def text_to_speech(text: str, voice_id: str = "", language: str = "en",
                   codec: str = "mp3") -> tuple[bytes, str]:
    """Calls grok_api.Chat/TextToSpeech. Returns (audio bytes, content type).

    STREAMING, not unary -- the APK declares it with newStreamingCall and the
    reply is a sequence of AudioChunk (f1 = data bytes, f2 = content_type). The
    audio is the concatenation of every chunk's f1, in order.
    """
    audio, content_type = bytearray(), ""
    for raw in _raw_stream("/grok_api.Chat/TextToSpeech",
                           build_tts_request(text, voice_id, language, codec),
                           timeout=120):
        f = _decode_proto(raw)
        for kind, val in f.get(1, []):
            audio += val if kind == "raw" else (
                val.encode("latin-1") if kind == "str" else b"")
        content_type = content_type or _first_str(f, 2)
    return bytes(audio), content_type or "audio/mpeg"
```

> Check `_raw_stream`'s actual yield shape before writing the loop — it is used
> by the chat path and may yield decoded frames rather than raw bytes. Match it.

In `main.py`, add the handler returning `Response(content=audio, media_type=...)`,
with a `SpeechRequest` model taking `input`, `voice`, `response_format` and
`model` (the last two accepted and ignored, as OpenAI clients send them).

In `capabilities.py`, flip `audio_speech` to `live` and update the docstring.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS, 39 tests.

- [ ] **Step 5: Commit**

```bash
cd /Users/cristian/per/grok
git add docker-api/grok_backend.py docker-api/main.py docker-api/capabilities.py docker-api/tests/test_audio_speech.py
git commit -m "feat(tts): text to speech over Chat/TextToSpeech"
```

---

### Task 9: `/v1/audio/transcriptions` — speech to text

**Repo:** `/Users/cristian/per/grok`

**Files:**
- Modify: `docker-api/grok_backend.py` (add `speech_to_text`), `docker-api/main.py`, `docker-api/capabilities.py`
- Test: `docker-api/tests/test_audio_transcriptions.py`

**Interfaces:**
- Produces: `backend.speech_to_text(audio: bytes, audio_format: str) -> str`; `POST /v1/audio/transcriptions`.

> **grok has two speech-to-text RPCs and this task picks the right one.**
> `Voice/Transcribe` returns `TranscribeResponse { 1: segments }` — a repeated
> structure that would have to be concatenated. `Voice/SpeechToText` is unary
> (`GrpcVoiceClient.java:145`, `newCall`) and returns
> `SpeechToTextGenerateResponse { 1: text, 2: sampling_time, 3: words,
> 4: language_code }` — a top-level `text` that maps straight onto OpenAI's
> `{"text": ...}`. Use `SpeechToText`; mention `Transcribe` in `CAPABILITIES.md`
> as the native alternative that returns segments.
>
> `SpeechToTextGenerateRequest` tags: `1 audio_base64` (string), `2 audio_format`
> (string), `3 max_tokens` (int32), `4 enhance` (bool), `5 history`, `6
> refinement_level`. Send 1 and 2; leave the rest at their defaults.

- [ ] **Step 1: Write the failing test**

Create `docker-api/tests/test_audio_transcriptions.py`:

```python
import base64
import io

from fastapi.testclient import TestClient

import capabilities as cap
import main


def _client(monkeypatch):
    monkeypatch.setattr(cap, "snapshot",
                        lambda: cap.SessionState(mode="account"))
    monkeypatch.setattr(main, "API_KEY", "")
    monkeypatch.setattr(main.backend, "speech_to_text",
                        lambda *a, **k: "hola que tal")
    return TestClient(main.app)


def test_transcription_returns_the_openai_shape(monkeypatch):
    with _client(monkeypatch) as c:
        r = c.post("/v1/audio/transcriptions",
                   files={"file": ("a.mp3", io.BytesIO(b"\x00\x01"), "audio/mpeg")})
    assert r.status_code == 200
    assert r.json() == {"text": "hola que tal"}


def test_response_format_text_returns_plain_text(monkeypatch):
    with _client(monkeypatch) as c:
        r = c.post("/v1/audio/transcriptions",
                   files={"file": ("a.mp3", io.BytesIO(b"\x00\x01"), "audio/mpeg")},
                   data={"response_format": "text"})
    assert r.status_code == 200
    assert r.text.strip() == "hola que tal"


def test_the_format_comes_from_the_filename(monkeypatch):
    seen = {}

    def fake(audio, audio_format="mp3", **k):
        seen["format"] = audio_format
        return "ok"

    with _client(monkeypatch) as c:
        monkeypatch.setattr(main.backend, "speech_to_text", fake)
        c.post("/v1/audio/transcriptions",
               files={"file": ("clip.wav", io.BytesIO(b"\x00"), "audio/wav")})
    assert seen["format"] == "wav"


def test_the_request_frame_carries_the_recovered_tags():
    import grok_backend as gb
    frame = gb.build_stt_request(b"\x00\x01", "mp3")
    fields = gb._decode_proto(frame)
    assert gb._first_str(fields, 1) == base64.b64encode(b"\x00\x01").decode()
    assert gb._first_str(fields, 2) == "mp3"


def test_the_contract_now_claims_audio_transcription():
    assert cap.effective(cap.SessionState(mode="account"))["audio_transcription"] is True
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_audio_transcriptions.py -v`
Expected: FAIL — 404 on the route, and `grok_backend` has no `build_stt_request`.

- [ ] **Step 3: Write the implementation**

In `grok_backend.py`:

```python
# SpeechToTextGenerateRequest, recovered from the decompiled APK
# (jadx_out/sources/grok_api/SpeechToTextGenerateRequest.java). The audio
# travels BASE64 IN A STRING FIELD, not as bytes -- the field is named
# audio_base64 and typed STRING in the APK, and sending raw bytes here would
# produce a frame the backend cannot read.
_STT_AUDIO_B64, _STT_FORMAT = 1, 2


def build_stt_request(audio: bytes, audio_format: str = "mp3") -> bytes:
    """The SpeechToTextGenerateRequest frame, separated so the wire shape can be
    tested without a network."""
    import base64
    return (_str_field(_STT_AUDIO_B64, base64.b64encode(audio).decode())
            + _str_field(_STT_FORMAT, audio_format))


def speech_to_text(audio: bytes, audio_format: str = "mp3") -> str:
    """Calls grok_api.Voice/SpeechToText. Returns the transcript.

    UNARY (GrpcVoiceClient declares it with newCall), and chosen over
    Voice/Transcribe because its reply carries a top-level `text` (f1) that maps
    straight onto OpenAI's {"text": ...}, where Transcribe returns repeated
    segments that would have to be stitched.
    """
    raw = _raw_unary("/grok_api.Voice/SpeechToText",
                     build_stt_request(audio, audio_format), timeout=120)
    return _first_str(_decode_proto(raw), 1)
```

In `main.py`, add the multipart handler: `file: UploadFile`, plus `model`,
`language` and `response_format` as form fields (accepted; `response_format:
"text"` returns plain text, anything else returns `{"text": ...}`). Derive the
format from the filename extension, defaulting to `mp3`.

In `capabilities.py`, flip `audio_transcription` to `live` and update the docstring.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS, 44 tests.

- [ ] **Step 5: Commit**

```bash
cd /Users/cristian/per/grok
git add docker-api/grok_backend.py docker-api/main.py docker-api/capabilities.py docker-api/tests/test_audio_transcriptions.py
git commit -m "feat(stt): speech to text over Voice/SpeechToText"
```

---

### Task 10: the 501 gate, and the docs

**Repo:** `/Users/cristian/per/grok`

**Files:**
- Modify: `docker-api/main.py`
- Create: `CAPABILITIES.md` (repo root)
- Test: `docker-api/tests/test_capability_gate.py`

**Interfaces:**
- Produces: `main.require_capability(name: str) -> None`.

> An endpoint whose capability is `false` answers **`501`**, never `404` (indistinguishable from a routing mistake) and never `503` (which makes the gateway retry and accumulate suspicion against a route that was never going to work). For grok that means: with no session token, every gated endpoint refuses immediately instead of failing deep inside a gRPC call.
>
> grok's handlers are **synchronous** (`def health()`, not `async def`). `capabilities.snapshot()` is a single environment read with no lock and no network, so it is safe to call directly — do NOT add `asyncio.to_thread` here; that pattern existed in `chatgpt-proxy` because its snapshot made a vendor round trip.

- [ ] **Step 1: Write the failing test**

Create `docker-api/tests/test_capability_gate.py`:

```python
import io

from fastapi.testclient import TestClient

import capabilities as cap
import main


def _anon(monkeypatch):
    monkeypatch.setattr(cap, "snapshot",
                        lambda: cap.SessionState(mode="anonymous"))
    monkeypatch.setattr(main, "API_KEY", "")
    return TestClient(main.app)


def test_image_generation_without_a_session_is_501(monkeypatch):
    with _anon(monkeypatch) as c:
        r = c.post("/v1/images/generations", json={"prompt": "a cat"})
    assert r.status_code == 501


def test_speech_without_a_session_is_501(monkeypatch):
    with _anon(monkeypatch) as c:
        r = c.post("/v1/audio/speech", json={"input": "hola"})
    assert r.status_code == 501


def test_transcription_without_a_session_is_501(monkeypatch):
    with _anon(monkeypatch) as c:
        r = c.post("/v1/audio/transcriptions",
                   files={"file": ("a.mp3", io.BytesIO(b"\x00"), "audio/mpeg")})
    assert r.status_code == 501


def test_files_without_a_session_is_501(monkeypatch):
    with _anon(monkeypatch) as c:
        r = c.get("/v1/files")
    assert r.status_code == 501


def test_conversations_without_a_session_is_501(monkeypatch):
    with _anon(monkeypatch) as c:
        r = c.get("/v1/conversations")
    assert r.status_code == 501


def test_the_501_body_names_the_capability(monkeypatch):
    with _anon(monkeypatch) as c:
        r = c.post("/v1/audio/speech", json={"input": "hola"})
    assert "audio_speech" in r.text


def test_models_and_health_are_never_gated(monkeypatch):
    with _anon(monkeypatch) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/v1/models").status_code == 200
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_capability_gate.py -v`
Expected: FAIL — the endpoints answer 200 or 500, not 501.

- [ ] **Step 3: Write the implementation**

In `main.py`:

```python
def require_capability(name: str) -> None:
    """Refuse with 501 when this proxy cannot serve `name` right now.

    501, not 404 and not 503, and the distinction is load-bearing for the
    gateway in front. A 404 is indistinguishable from a routing mistake. A 503
    says "it broke" -- so the gateway retries, accumulates suspicion against the
    route and fails over, spending attempts on something that was never going to
    work in this configuration. 501 says: this proxy, deliberately, does not do
    this right now.

    Synchronous on purpose: capabilities.snapshot() reads one environment
    variable, with no lock and no network.
    """
    if not capabilities.effective(capabilities.snapshot())[name]:
        raise HTTPException(
            501,
            f"This proxy cannot serve '{name}' in its current configuration "
            f"(see GET /health, capabilities.{name}).")
```

Call it as the first statement of: `/v1/images/generations` (`images`),
`/v1/audio/speech` (`audio_speech`), `/v1/audio/transcriptions`
(`audio_transcription`), the four `/v1/files*` handlers (`files`), and the two
`/v1/conversations*` handlers (`conversations`). Leave `/health`, `/v1/models`
and `/v1/chat/completions` ungated.

Create `CAPABILITIES.md` at the repo root recording, per capability: what it is,
which RPC serves it, and how it was verified. Note explicitly that
`Voice/Transcribe` exists as a segment-returning alternative to the
`Voice/SpeechToText` this proxy uses, and that `translate` is false because grok
has no translate endpoint.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS, 51 tests.

- [ ] **Step 5: Commit**

```bash
cd /Users/cristian/per/grok
git add docker-api/main.py CAPABILITIES.md docker-api/tests/test_capability_gate.py
git commit -m "feat(gate): a capability this configuration lacks answers 501"
```

---

## Part 3 — the gateway opts grok in

### Task 11: `providers.yaml`

**Repo:** `/Users/cristian/llm-libre`

**Files:**
- Modify: `providers.yaml` (the `grok` entry)
- Test: `tests/test_providers.py`

**Interfaces:**
- Consumes: `Provider.reads_capabilities` (previous plan, Task 7).

> Three changes, and the second is the one the final whole-branch review asked for.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_providers.py`:

```python
def test_grok_reads_the_contract():
    providers = load("providers.yaml", {})
    assert sorted(p.id for p in providers if p.reads_capabilities) == ["chatgpt", "grok"]


def test_grok_no_longer_pins_images_through_exceptions():
    # The gpt-image-1 defect, in its other home: `exceptions` applies LAST, so
    # a hand-declared images:true would survive a contract that says otherwise.
    # The per-model block on grok's /v1/models supplies these now.
    grok = [p for p in load("providers.yaml", {}) if p.id == "grok"][0]
    for override in grok.exceptions.values():
        assert "images" not in override
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_providers.py -v -k grok`
Expected: FAIL — grok does not read the contract, and its three exceptions still carry `images`.

- [ ] **Step 3: Write the implementation**

In the `grok` entry of `providers.yaml`:

1. Add `reads_capabilities: true`, with a comment saying its `/health` publishes the contract and that the declarations below become the fallback for a sweep that could not read it.
2. **Delete the three `exceptions` entries entirely.** grok's `/v1/models` now publishes `{tools, vision, images}` per model, and the imagine family reports exactly what those three hand-declared. Replace them with a comment recording that they were retired because the provider publishes the truth, and pointing at the final-review finding that `exceptions` applies last and would have defeated the contract.
3. **Correct the `context: 128000` comment.** Do not change the number — there is nothing to change it to. Say plainly that grok publishes no context figure for any model (verified: 26 advertised models, no `context_window`, no token field), so this is a declared floor that nothing has measured, and that the contract cannot correct it because the provider has nothing to report.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/cristian/llm-libre
git add providers.yaml tests/test_providers.py
git commit -m "feat(grok): read its contract, and retire the exceptions it made unnecessary"
```

---

## Deployment

Both repos deploy by push to `main`; Coolify builds on the push.

**grok-proxy first**, so the gateway never reads a contract from a proxy that cannot yet honour it. Its Dockerfile is at `/Users/cristian/per/grok/docker-api/Dockerfile` — **check whether it copies modules by name**, as `chatgpt-proxy`'s does. `capabilities.py` is new, and a `COPY` manifest that does not list it produces a container that crash-loops on `ModuleNotFoundError` with the port dark. That has happened twice in this codebase's history.

Then llm-libre. `providers.yaml` is read once at startup, so Task 11 needs a restart, not a sweep.

Verify:

```bash
ssh blog 'curl -s http://127.0.0.1:8893/health' | python3 -m json.tool
ssh blog 'curl -s -H "X-API-Key: <key>" http://127.0.0.1:8102/health' | python3 -m json.tool
```

Expected: grok's `/health` carries `contract: 1` with `search`, `files`,
`conversations`, `audio_speech` and `audio_transcription` all true; the
gateway's `providers` block lists both `chatgpt` and `grok`.

## What this plan does not do

Perplexity, deepseek and mistral adopting the contract — the same shape, once
each. `translate` on grok, which has no endpoint to expose. The gateway
endpoints that would route by the new axes (`/v1/audio/speech`,
`/v1/audio/transcriptions`, `/v1/translate`) — with grok live, two providers
will report `audio_speech` and `audio_transcription`, which is the point at
which those become worth building. And `audio/mpeg`/`audio/wav` in
`assets._SAFE_TYPES`, without which a generated mp3 is served as
`application/octet-stream`.
