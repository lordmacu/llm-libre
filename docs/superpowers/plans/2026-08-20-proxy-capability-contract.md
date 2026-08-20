# Proxy Capability Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make each in-house proxy publish what it can actually do right now, and make the gateway read that instead of trusting a hand-written snapshot in `providers.yaml`.

**Architecture:** A proxy publishes a versioned `capabilities` block on `GET /health`, already resolved against its account and plan, plus per-model `context_window`/`max_output_tokens`/`capabilities` on `GET /v1/models`. The gateway parses it in a new `contract.py`, applies it in `catalog.normalize` under a fixed precedence (`exceptions` > per-model > provider-level > `default_capabilities`), persists the last-seen document so it survives restarts, and alerts when a capability disappears. A proxy that has not adopted the contract behaves exactly as it does today.

**Tech Stack:** Python 3.12+, FastAPI, httpx, pytest (+ pytest-asyncio, `asyncio_mode = "auto"`), SQLite.

**Spec:** [`docs/superpowers/specs/2026-08-20-proxy-capability-contract-design.md`](../specs/2026-08-20-proxy-capability-contract-design.md)

## Global Constraints

- **All code, comments, identifiers, log messages and docstrings in English.** No Spanish anywhere in code output. (Existing Spanish test names in `llm-libre` are left alone; do not add new ones.)
- **Git identity for every commit in every repo:** `user.name = lordmacu`, `user.email = 10134930+lordmacu@users.noreply.github.com`. **Never** add a `Co-Authored-By:` trailer.
- **`contract` version is `1`** for this whole plan. A document declaring any other integer is ignored with a warning.
- **Required `capabilities` keys, exactly these eleven:** `chat`, `streaming`, `tools`, `vision`, `images`, `audio_speech`, `audio_transcription`, `translate`, `search`, `files`, `conversations`. All booleans. Missing any key means the document is not a contract document.
- **`auth.mode` is one of** `anonymous`, `account`, `unknown`. `auth.plan` is a free-form vendor string or `null`. `auth.expires_at` is ISO 8601 UTC or `null`.
- **The gateway never branches on `auth.plan`.** Everything it acts on is in `capabilities`.
- **A capability boolean tracks entitlement, not the meter.** Spent daily quota stays a `429` + cooldown; it must not flip a boolean.
- **Two repos.** Tasks 1 and 6–12 are in `/Users/cristian/llm-libre`. Tasks 2–5 are in `/Users/cristian/chatgpt-proxy`. Each task's `Files:` block states which.
- **Run llm-libre tests with** `.venv/bin/python -m pytest` from `/Users/cristian/llm-libre`.
- **Run chatgpt-proxy tests with** `python3 -m pytest` from `/Users/cristian/chatgpt-proxy`.

---

## File Structure

**`/Users/cristian/llm-libre`**

| File | Responsibility |
|---|---|
| `src/llm_libre/contract.py` | **New.** Parse and validate a `/health` document into `ProviderContract`. Knows the wire format; knows nothing about routes. |
| `src/llm_libre/models.py` | `Capabilities` gains four provider-level axes. |
| `src/llm_libre/providers.py` | `Provider.reads_capabilities`; `fixed_routes` masking. |
| `src/llm_libre/catalog.py` | Capability resolution: contract → per-model narrowing → `exceptions`. |
| `src/llm_libre/probing.py` | Fetch `/health`, persist it, degrade on failure, alert on transitions. |
| `src/llm_libre/storage.py` | Four new `routes` columns; new `provider_contracts` table. |
| `src/llm_libre/api.py` | Surface contract state on the gateway's own `/health`. |
| `providers.yaml` | `reads_capabilities: true` on `chatgpt`. |
| `tests/test_contract.py` | **New.** Parsing and degradation. |
| `tests/test_catalog.py`, `test_probing.py`, `test_providers.py`, `test_storage.py`, `test_wire_contract.py`, `test_notify.py` | Extended. |

**`/Users/cristian/chatgpt-proxy`**

| File | Responsibility |
|---|---|
| `capabilities.py` | **New.** Cached account/plan state + the effective capability booleans. The only place vendor plan rules live. |
| `main.py` | `/health` and `/v1/models` emit the contract; capability-gated endpoints answer `501`. |
| `tests/test_capabilities.py` | **New.** The mapping from account state to booleans. |
| `tests/test_health_contract.py` | **New.** `/health` satisfies the contract. |
| `requirements-dev.txt` | **New.** pytest, pytest-asyncio. |
| `CAPABILITIES.md` | Updated with the contract. |

---

## Part 1 — The gateway defines the contract

### Task 1: `contract.py` — parse a `/health` document

**Repo:** `/Users/cristian/llm-libre`

**Files:**
- Create: `src/llm_libre/contract.py`
- Test: `tests/test_contract.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `contract.VERSION: int`, `contract.REQUIRED_CAPABILITIES: frozenset[str]`, `contract.Auth(mode: str, plan: str | None, subscription_active: bool, expires_at: str | None)`, `contract.ProviderContract(version: int, provider: str, auth: Auth, capabilities: dict[str, bool])`, `contract.parse_health(provider: str, doc: object) -> ProviderContract | None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_contract.py`:

```python
import logging

from llm_libre.contract import (REQUIRED_CAPABILITIES, VERSION, Auth,
                                ProviderContract, parse_health)


def _caps(**overrides):
    caps = {k: False for k in REQUIRED_CAPABILITIES}
    caps.update(overrides)
    return caps


def _doc(**overrides):
    doc = {
        "status": "ok",
        "provider": "chatgpt",
        "version": "2.5.0",
        "contract": VERSION,
        "auth": {"mode": "account", "plan": "go",
                 "subscription_active": True,
                 "expires_at": "2026-09-06T00:28:46Z"},
        "capabilities": _caps(chat=True, streaming=True, vision=True, images=True),
    }
    doc.update(overrides)
    return doc


def test_a_compliant_document_is_parsed():
    c = parse_health("chatgpt", _doc())
    assert isinstance(c, ProviderContract)
    assert c.version == VERSION
    assert c.provider == "chatgpt"
    assert c.auth == Auth(mode="account", plan="go", subscription_active=True,
                          expires_at="2026-09-06T00:28:46Z")
    assert c.capabilities["images"] is True
    assert c.capabilities["translate"] is False


def test_a_document_without_the_contract_key_is_not_a_contract(caplog):
    # The pre-contract proxies. This is the NORMAL case during rollout, so it
    # must be silent: a warning per provider per sweep would train the operator
    # to ignore the log.
    doc = _doc()
    del doc["contract"]
    with caplog.at_level(logging.WARNING):
        assert parse_health("chatgpt", doc) is None
    assert caplog.records == []


def test_a_different_contract_version_is_refused_and_logged(caplog):
    with caplog.at_level(logging.WARNING):
        assert parse_health("chatgpt", _doc(contract=VERSION + 1)) is None
    assert "version" in caplog.text


def test_a_missing_capability_key_refuses_the_whole_document(caplog):
    caps = _caps(chat=True)
    del caps["translate"]
    with caplog.at_level(logging.WARNING):
        assert parse_health("chatgpt", _doc(capabilities=caps)) is None
    assert "translate" in caplog.text


def test_a_non_boolean_capability_refuses_the_whole_document(caplog):
    # chatgpt-proxy's pre-contract block carried English prose in these fields
    # ("automatic (override with web_search: true/false)"). A truthy string
    # would silently read as True, which is exactly the wrong direction.
    with caplog.at_level(logging.WARNING):
        assert parse_health("chatgpt", _doc(capabilities=_caps(search="automatic"))) is None
    assert "search" in caplog.text


def test_unknown_capability_keys_are_ignored():
    # How a new capability ships before the gateway learns to use it.
    c = parse_health("chatgpt", _doc(capabilities=_caps(chat=True, video=True)))
    assert set(c.capabilities) == set(REQUIRED_CAPABILITIES)


def test_a_malformed_auth_block_degrades_to_unknown_without_losing_capabilities():
    c = parse_health("chatgpt", _doc(auth="account"))
    assert c.auth.mode == "unknown"
    assert c.capabilities["chat"] is True


def test_an_unrecognised_auth_mode_degrades_to_unknown(caplog):
    with caplog.at_level(logging.WARNING):
        c = parse_health("chatgpt", _doc(auth={"mode": "subscriber"}))
    assert c.auth.mode == "unknown"
    assert "subscriber" in caplog.text


def test_a_body_that_is_not_an_object_is_not_a_contract():
    assert parse_health("chatgpt", ["ok"]) is None
    assert parse_health("chatgpt", None) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_contract.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_libre.contract'`

- [ ] **Step 3: Write the implementation**

Create `src/llm_libre/contract.py`:

```python
"""The proxy capability contract: what an in-house proxy says it can do NOW.

Spec: docs/superpowers/specs/2026-08-20-proxy-capability-contract-design.md

The gateway used to declare every provider's capabilities by hand in
providers.yaml, at one moment in time, and nothing detected when that snapshot
stopped matching reality: `context: 128000` against a real 52815, `images: true`
against an account whose paid plan expires on a date the gateway cannot see.
This module reads the replacement -- a versioned block the proxy itself
publishes on GET /health -- and refuses anything it cannot fully trust.

Refusing means returning None, and None is a NORMAL, supported answer, not an
error: it means "this proxy has not adopted the contract", and the caller falls
back to exactly the behaviour that exists today. That is what makes the rollout
incremental, one proxy at a time, with no flag day.

This module knows the WIRE FORMAT and nothing else. It does not know what a
Route is, it does not decide precedence, and it never talks to the network --
catalog.py and probing.py own those. Keeping it that narrow is what lets the
contract be tested against fixtures alone.
"""
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

# The schema version THIS gateway speaks. A proxy declaring anything else is
# refused rather than parsed optimistically: a contract whose meaning we are
# guessing at is worse than no contract, because the fallback path is known-good.
VERSION = 1

# Every key a compliant `capabilities` block must carry. A proxy that cannot do
# something says `false`; it never omits the key. That distinction is the whole
# point of requiring the full set: an omission is indistinguishable from a proxy
# that forgot to report a capability it HAS, and guessing either way is wrong.
REQUIRED_CAPABILITIES = frozenset({
    "chat", "streaming", "tools", "vision", "images",
    "audio_speech", "audio_transcription", "translate",
    "search", "files", "conversations",
})

_AUTH_MODES = frozenset({"anonymous", "account", "unknown"})


@dataclass(frozen=True)
class Auth:
    """Who the proxy is talking to the vendor as. INFORMATIONAL ONLY.

    The gateway must never branch on `plan`: plan names are vendor-specific
    ("go", "free", "plus", "pro", ...) and teaching the gateway to read them
    rebuilds exactly the coupling this contract removes. It exists for the
    operator's /health view and for the subscription-expiry alert; everything
    the gateway ACTS on lives in `ProviderContract.capabilities`.
    """
    mode: str                            # "anonymous" | "account" | "unknown"
    plan: str | None = None
    subscription_active: bool = False
    expires_at: str | None = None        # ISO 8601 UTC, or None


@dataclass(frozen=True)
class ProviderContract:
    version: int
    provider: str
    auth: Auth
    capabilities: dict


def parse_health(provider: str, doc: object) -> ProviderContract | None:
    """A parsed contract, or None when this is not a contract document.

    `provider` is the id from providers.yaml, used for log messages and as the
    fallback when the document does not name itself.

    Every refusal below is all-or-nothing on purpose. A half-read contract would
    mix discovered values with fallback ones, and the resulting capability set
    would belong to no single source -- impossible to reason about when a route
    starts failing. Refuse, log why, and let the caller use the known-good path.
    """
    if not isinstance(doc, dict):
        return None
    version = doc.get("contract")
    # A missing key is the pre-contract proxy: SILENT. During the rollout that is
    # the majority case, and warning once per provider per sweep would train the
    # operator to ignore this log.
    if not isinstance(version, int) or isinstance(version, bool):
        return None
    if version != VERSION:
        log.warning(
            "contract %s: the proxy speaks version %r, this gateway speaks %d. "
            "Ignored -- falling back to providers.yaml.", provider, version, VERSION)
        return None
    caps = doc.get("capabilities")
    if not isinstance(caps, dict):
        log.warning(
            "contract %s: declares version %d but carries no 'capabilities' "
            "object. Ignored -- falling back to providers.yaml.", provider, version)
        return None
    missing = sorted(REQUIRED_CAPABILITIES - caps.keys())
    if missing:
        log.warning(
            "contract %s: 'capabilities' is missing %s. Ignored -- a partial "
            "block cannot be told apart from a proxy that forgot to report a "
            "capability it has.", provider, missing)
        return None
    # `isinstance(True, int)` is True in Python, so this check is written the
    # strict way round. It is not pedantry: chatgpt-proxy's pre-contract block
    # carried English prose in these fields, and a non-empty string is truthy --
    # a loose check would read "automatic (override with web_search...)" as
    # "yes, this provider does search", which is the wrong direction.
    not_boolean = sorted(k for k in REQUIRED_CAPABILITIES
                         if not isinstance(caps[k], bool))
    if not_boolean:
        log.warning(
            "contract %s: these capabilities are not booleans: %s. Ignored.",
            provider, not_boolean)
        return None
    named = doc.get("provider")
    return ProviderContract(
        version=version,
        provider=str(named) if isinstance(named, str) and named else provider,
        auth=_auth(provider, doc.get("auth")),
        # Only the required keys are kept: an unknown key is a capability this
        # gateway does not understand yet, and carrying it further would let it
        # reach code that assumes the fixed set.
        capabilities={k: bool(caps[k]) for k in REQUIRED_CAPABILITIES},
    )


def _auth(provider: str, data: object) -> Auth:
    """The `auth` block, degrading to "unknown" rather than refusing.

    A malformed `auth` does NOT invalidate the document, unlike a malformed
    `capabilities`: nothing routes on `auth`. Losing it costs an operator a line
    in /health and one alert; losing the capabilities would cost correct routing.
    The two are treated differently because they are worth different amounts.
    """
    if not isinstance(data, dict):
        return Auth(mode="unknown")
    mode = data.get("mode")
    if mode not in _AUTH_MODES:
        log.warning(
            "contract %s: auth.mode=%r is not one of %s; reported as 'unknown'.",
            provider, mode, sorted(_AUTH_MODES))
        mode = "unknown"
    plan = data.get("plan")
    expires_at = data.get("expires_at")
    return Auth(
        mode=mode,
        plan=plan if isinstance(plan, str) and plan else None,
        subscription_active=bool(data.get("subscription_active", False)),
        expires_at=expires_at if isinstance(expires_at, str) and expires_at else None,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_contract.py -v`
Expected: PASS, 10 tests.

- [ ] **Step 5: Commit**

```bash
cd /Users/cristian/llm-libre
git add src/llm_libre/contract.py tests/test_contract.py
git commit -m "feat(contract): parse the capability document a proxy publishes about itself"
```

---

## Part 2 — chatgpt-proxy publishes the contract

### Task 2: cached account state and the effective capability booleans

**Repo:** `/Users/cristian/chatgpt-proxy`

**Files:**
- Create: `capabilities.py`
- Create: `requirements-dev.txt`
- Create: `tests/__init__.py` (empty), `tests/test_capabilities.py`
- Test: `tests/test_capabilities.py`

**Interfaces:**
- Consumes: `auth.is_authenticated()` from the existing `auth.py`.
- Produces: `capabilities.AccountState(mode: str, plan: str | None, subscription_active: bool, expires_at: str | None)`, `capabilities.effective(state: AccountState) -> dict[str, bool]`, `capabilities.snapshot() -> AccountState` (cached), `capabilities.REFRESH_INTERVAL_S: float`, `capabilities.reset()` (tests).

- [ ] **Step 1: Write the failing tests**

Create `requirements-dev.txt`:

```
-r requirements.txt
pytest==8.3.4
pytest-asyncio==0.25.0
```

Create empty `tests/__init__.py`, then `tests/test_capabilities.py`:

```python
import capabilities as cap


GO = cap.AccountState(mode="account", plan="go", subscription_active=True,
                      expires_at="2026-09-06T00:28:46Z")
FREE = cap.AccountState(mode="account", plan="free", subscription_active=False,
                        expires_at=None)
ANON = cap.AccountState(mode="anonymous", plan=None, subscription_active=False,
                        expires_at=None)


def test_every_required_key_is_present_in_all_three_modes():
    for state in (GO, FREE, ANON):
        assert set(cap.effective(state)) == set(cap.REQUIRED_CAPABILITIES)
        assert all(isinstance(v, bool) for v in cap.effective(state).values())


def test_anonymous_gets_only_chat_translate_and_search():
    e = cap.effective(ANON)
    assert e["chat"] and e["streaming"] and e["translate"] and e["search"]
    assert not e["vision"]
    assert not e["images"]
    assert not e["audio_speech"]
    assert not e["audio_transcription"]
    assert not e["files"]
    assert not e["conversations"]


def test_a_free_account_gets_everything_except_images():
    # Measured, and recorded in CAPABILITIES.md: on free the model DOES invoke
    # the image tool, the generation returns empty, and the proxy answers
    # "no image was generated". That is a plan block, not a transient failure.
    e = cap.effective(FREE)
    assert e["vision"] and e["audio_speech"] and e["audio_transcription"]
    assert e["files"] and e["conversations"]
    assert not e["images"]


def test_a_paid_plan_gets_images():
    assert cap.effective(GO)["images"] is True


def test_an_expired_paid_plan_loses_images():
    # The event this whole contract exists for: the subscription lapses and the
    # boolean turns itself off, with nobody editing YAML.
    lapsed = cap.AccountState(mode="account", plan="go",
                              subscription_active=False, expires_at="2026-09-06T00:28:46Z")
    assert cap.effective(lapsed)["images"] is False


def test_tools_is_false_on_every_plan():
    # No function calling on any backend: measured 0/3, twice, with
    # tool_choice:"required" returning tool_calls:None and prose.
    for state in (GO, FREE, ANON):
        assert cap.effective(state)["tools"] is False


def test_snapshot_is_cached_and_does_not_refetch_within_the_interval():
    calls = []

    def fake_resolve():
        calls.append(1)
        return GO

    cap.reset()
    assert cap.snapshot(_resolve=fake_resolve, _now=1000.0) == GO
    assert cap.snapshot(_resolve=fake_resolve, _now=1000.0 + cap.REFRESH_INTERVAL_S - 1) == GO
    assert len(calls) == 1


def test_snapshot_refetches_once_the_interval_has_passed():
    calls = []

    def fake_resolve():
        calls.append(1)
        return GO

    cap.reset()
    cap.snapshot(_resolve=fake_resolve, _now=1000.0)
    cap.snapshot(_resolve=fake_resolve, _now=1000.0 + cap.REFRESH_INTERVAL_S + 1)
    assert len(calls) == 2


def test_a_failing_resolve_keeps_the_last_known_state():
    # /health must not start lying because the vendor had a bad minute.
    def boom():
        raise RuntimeError("upstream down")

    cap.reset()
    cap.snapshot(_resolve=lambda: GO, _now=1000.0)
    assert cap.snapshot(_resolve=boom, _now=1000.0 + cap.REFRESH_INTERVAL_S + 1) == GO


def test_a_failing_resolve_with_no_previous_state_reports_unknown():
    def boom():
        raise RuntimeError("upstream down")

    cap.reset()
    state = cap.snapshot(_resolve=boom, _now=1000.0)
    assert state.mode == "unknown"
    assert cap.effective(state)["images"] is False
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /Users/cristian/chatgpt-proxy
python3 -m pip install -r requirements-dev.txt
python3 -m pytest tests/test_capabilities.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'capabilities'`

- [ ] **Step 3: Write the implementation**

Create `capabilities.py`:

```python
"""What this proxy can actually do right now, resolved against the account.

Spec: the proxy capability contract, llm-libre
docs/superpowers/specs/2026-08-20-proxy-capability-contract-design.md

THE RULE THIS MODULE EXISTS TO ENFORCE: a capability boolean says what a request
sent right now would ACHIEVE, not what this codebase implements. On a free plan
/v1/images/generations exists, accepts the request, invokes the image tool, and
returns nothing -- so `images` is False. To the gateway "the endpoint exists" and
"a request to it produces an image" are the same question, and only the second
one is worth answering.

Where the rule STOPS: a boolean tracks entitlement, not the meter. `images` is
False for anonymous, free, expired or revoked -- never for "today's 106
generations are spent". Exhaustion is a 429 the gateway already handles with a
cooldown and the vendor's own Retry-After, and it recovers by itself; flipping
a boolean for it would flap between sweeps and replace self-healing with a wait
for the next one. The dividing line is durability: if a fresh request TOMORROW
would still be refused for the same reason, it belongs in the boolean.

This is also the ONLY place that knows what "go" or "free" mean. The gateway
never sees a plan name -- it reads booleans, which is what keeps it ignorant of
every vendor's billing.
"""
import logging
import os
import threading
import time
from dataclasses import dataclass

import auth

log = logging.getLogger(__name__)

# The eleven keys the contract requires, byte-for-byte the same set the gateway
# validates against (llm_libre.contract.REQUIRED_CAPABILITIES). Duplicated here
# rather than imported because the two live in different repos and deploy
# independently; tests/test_health_contract.py is what keeps them honest.
REQUIRED_CAPABILITIES = (
    "chat", "streaming", "tools", "vision", "images",
    "audio_speech", "audio_transcription", "translate",
    "search", "files", "conversations",
)

# /health is called by the gateway on every catalogue sweep and by Coolify as a
# container health check, so it must NOT reach the vendor per hit: that would
# make both depend on OpenAI being up, which is the opposite of what a health
# check is for. The account state is resolved at most this often and served from
# cache in between. An hour is far shorter than anything that can change here
# (a plan lapses on a date, a token is revoked once) and far longer than the
# sweep interval.
REFRESH_INTERVAL_S = float(os.environ.get("CAPABILITY_REFRESH_S", "3600"))

_lock = threading.Lock()
_state: "AccountState | None" = None
_resolved_at: float = 0.0


@dataclass(frozen=True)
class AccountState:
    """The vendor-side facts the capability booleans are derived from."""
    mode: str                          # "anonymous" | "account" | "unknown"
    plan: str | None = None            # vendor string: "go", "free", ...
    subscription_active: bool = False
    expires_at: str | None = None      # ISO 8601 UTC


UNKNOWN = AccountState(mode="unknown")


def _paid(state: AccountState) -> bool:
    """Whether this account holds a live PAID plan.

    Both halves are needed. `plan` alone is stale the moment a subscription
    lapses -- the vendor keeps reporting plan_type "go" with
    subscription.active false -- and `subscription_active` alone cannot tell a
    paid plan from a free tier that reports itself as an active subscription.
    """
    return state.mode == "account" and state.subscription_active \
        and (state.plan or "free") != "free"


def effective(state: AccountState) -> dict:
    """The eleven booleans, for this account state. Measured, not guessed.

    Every False below was observed, and CAPABILITIES.md records how:
      - anonymous reaches 5 of 14 endpoints; `synthesize`, `library`, `gizmos`
        and the file APIs have no /backend-anon variant at all, so everything
        that needs one is False.
      - `tools` is False on EVERY plan: with tool_choice:"required" the backend
        returns tool_calls:None and prose. Measured 0/3, twice. This is the one
        boolean that does not vary, and claiming it would make the gateway route
        agentic traffic here and receive prose -- a silent failure, where a
        refusal would at least fail over.
      - `images` needs a paid plan. On free the tool IS invoked and returns
        empty; that is a plan block, not a transient failure.
      - `translate` and `search` work anonymously: /v1/translate does not even
        spend a chat message, and search is a flag on the chat request.
    """
    account = state.mode == "account"
    return {
        "chat":                True,
        "streaming":           True,
        "tools":               False,
        "vision":              account,
        "images":              _paid(state),
        "audio_speech":        account,
        "audio_transcription": account,
        "translate":           True,
        "search":              True,
        "files":               account,
        "conversations":       account,
    }


def snapshot(_resolve=None, _now=None) -> AccountState:
    """The cached account state, refreshed at most every REFRESH_INTERVAL_S.

    A failed refresh KEEPS the previous value rather than degrading to unknown:
    the last known state is far better evidence than "we could not ask just now",
    and turning capabilities off because the vendor blinked would take routes out
    of the gateway's rotation for no reason. Only a failure with nothing cached
    yet -- a cold start while the vendor is down -- reports unknown, where every
    account-gated capability is False. That direction is the safe one: claiming
    a capability we cannot confirm sends real traffic into a wall.

    `_resolve` and `_now` are injection points for tests; production passes
    neither.
    """
    global _state, _resolved_at
    resolve = _resolve or _resolve_from_vendor
    now = time.time() if _now is None else _now
    with _lock:
        if _state is not None and (now - _resolved_at) < REFRESH_INTERVAL_S:
            return _state
        try:
            _state = resolve()
        except Exception as e:                       # noqa: BLE001 -- see docstring
            log.warning("capabilities: could not resolve the account state "
                        "(%s: %s); keeping %s.", type(e).__name__, e,
                        "the previous value" if _state else "unknown")
            if _state is None:
                _state = UNKNOWN
        _resolved_at = now
        return _state


def reset() -> None:
    """Drop the cache. For tests, and for a token change at runtime."""
    global _state, _resolved_at
    with _lock:
        _state, _resolved_at = None, 0.0


def _resolve_from_vendor() -> AccountState:
    """Ask ChatGPT what this token's account is. Wired in Task 3.

    Kept separate from `snapshot` so the caching rules can be tested without a
    network, and so the vendor call has exactly one home.
    """
    if not auth.is_authenticated():
        return AccountState(mode="anonymous")
    return AccountState(mode="account")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_capabilities.py -v`
Expected: PASS, 10 tests.

- [ ] **Step 5: Commit**

```bash
cd /Users/cristian/chatgpt-proxy
git add capabilities.py requirements-dev.txt tests/
git commit -m "feat(capabilities): resolve what this account can actually do, cached"
```

---

### Task 3: `/health` emits the contract

**Repo:** `/Users/cristian/chatgpt-proxy`

**Files:**
- Modify: `capabilities.py` (`_resolve_from_vendor` reaches the vendor)
- Modify: `main.py:1747-1781` (the `/health` handler)
- Test: `tests/test_health_contract.py`

**Interfaces:**
- Consumes: `capabilities.snapshot()`, `capabilities.effective()`, `capabilities.AccountState` from Task 2.
- Produces: a `GET /health` body carrying `contract`, `provider`, `auth`, `capabilities`, parseable by `llm_libre.contract.parse_health`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_health_contract.py`:

```python
from fastapi.testclient import TestClient

import capabilities as cap
import main


REQUIRED = set(cap.REQUIRED_CAPABILITIES)
AUTH_MODES = {"anonymous", "account", "unknown"}


def _health(monkeypatch, state):
    cap.reset()
    monkeypatch.setattr(cap, "snapshot", lambda **kw: state)
    with TestClient(main.app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    return r.json()


def test_health_declares_the_contract_version(monkeypatch):
    body = _health(monkeypatch, cap.AccountState(mode="anonymous"))
    assert body["contract"] == 1
    assert body["provider"] == "chatgpt"


def test_health_capabilities_are_exactly_the_required_booleans(monkeypatch):
    body = _health(monkeypatch, cap.AccountState(mode="anonymous"))
    assert set(body["capabilities"]) == REQUIRED
    assert all(isinstance(v, bool) for v in body["capabilities"].values())


def test_health_auth_block_reports_the_account(monkeypatch):
    state = cap.AccountState(mode="account", plan="go", subscription_active=True,
                             expires_at="2026-09-06T00:28:46Z")
    body = _health(monkeypatch, state)
    assert body["auth"]["mode"] in AUTH_MODES
    assert body["auth"] == {"mode": "account", "plan": "go",
                            "subscription_active": True,
                            "expires_at": "2026-09-06T00:28:46Z"}


def test_images_is_false_without_a_paid_plan(monkeypatch):
    body = _health(monkeypatch, cap.AccountState(mode="account", plan="free"))
    assert body["capabilities"]["images"] is False


def test_the_legacy_status_and_version_fields_survive(monkeypatch):
    # Coolify's container health check and every existing dashboard read these.
    body = _health(monkeypatch, cap.AccountState(mode="anonymous"))
    assert body["status"] == "ok"
    assert isinstance(body["version"], str)
    assert body["auth_mode"] in AUTH_MODES
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_health_contract.py -v`
Expected: FAIL — `KeyError: 'contract'`

- [ ] **Step 3: Write the implementation**

In `capabilities.py`, replace `_resolve_from_vendor` with the real call:

```python
def _resolve_from_vendor() -> AccountState:
    """Ask ChatGPT what this token's account is.

    Synchronous and blocking on purpose: it runs at most once an hour, behind
    `snapshot`'s lock, and making it async would push an event loop requirement
    into every caller of a function that is meant to be trivially callable.
    """
    if not auth.is_authenticated():
        return AccountState(mode="anonymous")
    import httpx
    r = httpx.get(
        "https://chatgpt.com/backend-api/accounts/check/v4-2023-04-27",
        params={"timezone_offset_min": "0"},
        headers={"Authorization": "Bearer " + auth.access_token(),
                 "User-Agent": "chatgpt-proxy/capabilities",
                 "Accept": "application/json"},
        timeout=15.0,
    )
    r.raise_for_status()
    accounts = (r.json().get("accounts") or {})
    account = accounts.get("default") or next(iter(accounts.values()), {}) or {}
    entitlement = account.get("entitlement") or {}
    inner = account.get("account") or {}
    return AccountState(
        mode="account",
        plan=inner.get("plan_type"),
        subscription_active=bool(entitlement.get("has_active_subscription")),
        expires_at=entitlement.get("expires_at"),
    )
```

In `main.py`, replace the `/health` handler body (currently lines 1747-1781) with:

```python
@app.get("/health")
async def health():
    total_sessions = sum(len(p._pool) for p in _pools.values())
    total_files    = sum(len(uf) for uf in _files.values())
    state = capabilities.snapshot()
    return {
        "status":  "ok",
        "version": "2.5.0",
        # The capability contract (llm-libre spec 2026-08-20). Everything under
        # `capabilities` is EFFECTIVE: already resolved against this account and
        # its plan, so the gateway reads one boolean instead of learning what a
        # ChatGPT plan is. See capabilities.effective.
        "contract": 1,
        "provider": "chatgpt",
        "auth": {
            "mode":                state.mode,
            "plan":                state.plan,
            "subscription_active": state.subscription_active,
            "expires_at":          state.expires_at,
        },
        "capabilities": capabilities.effective(state),
        # Kept for compatibility: Coolify's container health check and the
        # existing dashboards read these, and the contract is additive.
        # `auth_mode` was the only machine-readable field the old block had.
        "auth_mode":             state.mode,
        "active_users":          len(_pools),
        "total_sessions":        total_sessions,
        "total_files_in_memory": total_files,
    }
```

Add `import capabilities` to `main.py`'s import block, next to `import auth`.

> Note: the old free-prose `capabilities` block is REMOVED, not kept alongside.
> It reported `image_input: false` after vision shipped in `774d019` and
> `voice: false` with both audio endpoints live -- a hand-written block that
> drifts is worse than none, and it is the reason this contract exists.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/ -v`
Expected: PASS, 15 tests.

- [ ] **Step 5: Commit**

```bash
cd /Users/cristian/chatgpt-proxy
git add capabilities.py main.py tests/test_health_contract.py
git commit -m "feat(health): publish the capability contract instead of hand-written prose"
```

---

### Task 4: `/v1/models` publishes per-model metadata

**Repo:** `/Users/cristian/chatgpt-proxy`

**Files:**
- Modify: `main.py:1172` (the `/v1/models` handler)
- Test: `tests/test_models_metadata.py`

**Interfaces:**
- Consumes: `capabilities.snapshot()`, `capabilities.effective()`.
- Produces: each `/v1/models` entry carries `context_window: int`, `max_output_tokens: int`, and `capabilities: {"tools": bool, "vision": bool, "images": bool}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_models_metadata.py`:

```python
import time

from fastapi.testclient import TestClient

import capabilities as cap
import main

PER_MODEL = {"tools", "vision", "images"}

# Seeded straight into the handler's cache so no test reaches the network. The
# shape is the vendor's, as fetch_anon_models returns it: `max_tokens` is what
# becomes `context_window`, and it is absent on the aliases.
UPSTREAM = [
    {"slug": "gpt-5-6", "title": "GPT-5.6 Luna", "max_tokens": 52815,
     "reasoning_type": "auto", "enabled_tools": ["tools", "search"]},
    {"slug": "gpt-5-6-t-mini", "title": "GPT-5.6 T Mini", "max_tokens": 262144,
     "reasoning_type": "reasoning", "enabled_tools": ["tools", "search"]},
]


def _models(monkeypatch, state):
    cap.reset()
    monkeypatch.setattr(cap, "snapshot", lambda **kw: state)
    monkeypatch.setattr(main, "_models_cache", (time.time(), UPSTREAM))
    with TestClient(main.app) as client:
        r = client.get("/v1/models")
    assert r.status_code == 200
    return r.json()["data"]


def test_every_model_carries_a_context_window(monkeypatch):
    # The gateway declared 128000 by hand for every id while the real value is
    # 52815 (and 262144 for the two -t-mini). Publishing it is what ends that.
    for m in _models(monkeypatch, cap.AccountState(mode="account", plan="go")):
        assert isinstance(m["context_window"], int)
        assert m["context_window"] > 0


def test_every_model_carries_per_model_capabilities(monkeypatch):
    for m in _models(monkeypatch, cap.AccountState(mode="account", plan="go")):
        assert set(m["capabilities"]) == PER_MODEL
        assert all(isinstance(v, bool) for v in m["capabilities"].values())


def test_a_per_model_capability_is_never_wider_than_the_provider_level_one(monkeypatch):
    # The contract's narrowing rule: on a free plan nothing may claim images.
    state = cap.AccountState(mode="account", plan="free")
    provider_level = cap.effective(state)
    for m in _models(monkeypatch, state):
        for name in PER_MODEL:
            assert not (m["capabilities"][name] and not provider_level[name]), \
                f"{m['id']} claims {name} while the provider reports it false"


def test_only_the_image_models_claim_images(monkeypatch):
    models = _models(monkeypatch, cap.AccountState(mode="account", plan="go"))
    drawing = {m["id"] for m in models if m["capabilities"]["images"]}
    assert drawing == set(main._IMAGE_MODELS)


def test_an_image_model_reports_no_output_ceiling(monkeypatch):
    # It returns a picture, not tokens. 0 is what fixed_models already declares
    # for dall-e-3 in the gateway's providers.yaml.
    models = _models(monkeypatch, cap.AccountState(mode="account", plan="go"))
    for m in models:
        if m["id"] in main._IMAGE_MODELS:
            assert m["max_output_tokens"] == 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_models_metadata.py -v`
Expected: FAIL — `KeyError: 'context_window'` on the entries that lack it.

- [ ] **Step 3: Write the implementation**

In `main.py`, in the `list_models` handler, immediately before the final
`return {"object": "list", "data": data}` (currently line 1246), stamp every
entry of the `data` list the handler has just assembled:

```python
    # --- capability contract: per-model metadata -----------------------------
    # The provider-level block on /health cannot say that gpt-image-1 draws and
    # does not chat, or that the two -t-mini models carry 5x the context of the
    # rest. Those are the three things that genuinely vary per model, plus the
    # sizes. A per-model value may only NARROW the provider-level one -- claiming
    # a capability the account does not have would be a lie with a smaller blast
    # radius, not a smaller lie.
    #
    # `context_window` is already set above from the vendor's `max_tokens`, and
    # is None for the aliases and the image models (which get _BLANK_CAPS). The
    # default below is a floor for exactly those, not a replacement for a real
    # value.
    _DEFAULT_CONTEXT = 52815          # what the vendor reports for this family
    _DEFAULT_MAX_OUTPUT = 8192

    provider_level = capabilities.effective(capabilities.snapshot())
    for entry in data:
        draws = entry["id"] in _IMAGE_MODELS
        entry["context_window"] = int(entry.get("context_window") or _DEFAULT_CONTEXT)
        entry["max_output_tokens"] = 0 if draws else _DEFAULT_MAX_OUTPUT
        entry["capabilities"] = {
            # `and provider_level[...]` IS the narrowing rule, applied at the
            # source: on an anonymous session or a free plan these come back
            # False no matter what the model would be capable of.
            "tools":  False,
            "vision": (not draws) and provider_level["vision"],
            "images": draws and provider_level["images"],
        }
```

> `_IMAGE_MODELS` already exists in `main.py` (the handler uses it a few lines
> above to always expose the image models); do not redefine it.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/ -v`
Expected: PASS, 20 tests.

- [ ] **Step 5: Commit**

```bash
cd /Users/cristian/chatgpt-proxy
git add main.py tests/test_models_metadata.py
git commit -m "feat(models): publish context window and per-model capabilities"
```

---

### Task 5: the endpoints behave the way the booleans promise

**Repo:** `/Users/cristian/chatgpt-proxy`

**Files:**
- Modify: `main.py` (the `/v1/images/generations`, `/v1/audio/speech`, `/v1/audio/from-message`, `/v1/audio/transcriptions`, `/v1/files`, `/v1/conversations` handlers)
- Modify: `CAPABILITIES.md`
- Test: `tests/test_capability_gate.py`, `tests/test_audio_speech_bytes.py`

**Interfaces:**
- Consumes: `capabilities.snapshot()`, `capabilities.effective()`.
- Produces: `main.require_capability(name: str) -> None`, raising `HTTPException(501, ...)`; `POST /v1/audio/speech` returning raw audio bytes.

> **Two deliverables, one concern.** A boolean is only worth reading if the
> endpoint behind it behaves as the contract says. Today `audio_speech: true`
> would be a lie by the contract's own §3.4 — the endpoint returns JSON carrying
> an mp3 URL, where the contract promises bytes — and a capability that is off
> answers `503`, which the gateway reads as "it broke" and retries.

- [ ] **Step 1: Write the failing test**

Create `tests/test_capability_gate.py`:

```python
from fastapi.testclient import TestClient

import capabilities as cap
import main


def _client(monkeypatch, state):
    cap.reset()
    monkeypatch.setattr(cap, "snapshot", lambda **kw: state)
    return TestClient(main.app)


def test_image_generation_on_a_free_plan_is_501_not_503(monkeypatch):
    # 501 says "this proxy, deliberately, does not do this". A 503 said "it
    # broke", so the gateway retried, accumulated suspicion and failed over --
    # for a capability that was never going to work on this plan.
    with _client(monkeypatch, cap.AccountState(mode="account", plan="free")) as c:
        r = c.post("/v1/images/generations", json={"prompt": "a cat"})
    assert r.status_code == 501


def test_audio_speech_on_an_anonymous_session_is_501(monkeypatch):
    with _client(monkeypatch, cap.AccountState(mode="anonymous")) as c:
        r = c.post("/v1/audio/speech", json={"input": "hello"})
    assert r.status_code == 501


def test_the_501_body_names_the_capability(monkeypatch):
    with _client(monkeypatch, cap.AccountState(mode="anonymous")) as c:
        r = c.post("/v1/audio/transcriptions", files={"file": ("a.mp3", b"x", "audio/mpeg")})
    assert r.status_code == 501
    assert "audio_transcription" in r.text


def test_translate_is_never_gated(monkeypatch):
    # It works anonymously and spends no chat message; gating it would be wrong.
    with _client(monkeypatch, cap.AccountState(mode="anonymous")) as c:
        r = c.post("/v1/translate", json={"text": "hi", "target": "es"})
    assert r.status_code != 501
```

Create `tests/test_audio_speech_bytes.py`:

```python
import httpx
import pytest
from fastapi.testclient import TestClient

import capabilities as cap
import main

MP3 = b"ID3\x03\x00\x00\x00fake-mp3-bytes"


@pytest.fixture
def synthesized(monkeypatch):
    """Stand in for the chat turn + /backend-api/synthesize round trip."""
    cap.reset()
    monkeypatch.setattr(cap, "snapshot",
                        lambda **kw: cap.AccountState(mode="account", plan="go"))

    async def fake_synthesize(text, voice, fmt, model):
        return main.Synthesized(audio=MP3, media_type="audio/mpeg", text=text,
                                exact_match=True, voice=voice, format=fmt,
                                conversation_id="conv-1", message_id="msg-1")

    monkeypatch.setattr(main, "_synthesize", fake_synthesize)
    return MP3


def test_audio_speech_returns_raw_bytes(synthesized):
    with TestClient(main.app) as c:
        r = c.post("/v1/audio/speech", json={"input": "hola", "voice": "juniper"})
    assert r.status_code == 200
    assert r.content == MP3
    assert r.headers["content-type"].startswith("audio/mpeg")


def test_the_metadata_travels_in_headers_not_in_the_body(synthesized):
    # `exact_match` is a real signal -- this flow makes the model echo the input,
    # and it sometimes alters it -- so it is kept, just out of the body, where
    # the OpenAI contract says only audio goes.
    with TestClient(main.app) as c:
        r = c.post("/v1/audio/speech", json={"input": "hola"})
    assert r.headers["X-Exact-Match"] == "true"
    assert r.headers["X-Conversation-Id"] == "conv-1"
    assert r.headers["X-Message-Id"] == "msg-1"
    assert r.headers["X-Audio-Url"].endswith(".mp3")


def test_the_native_endpoint_still_returns_the_json_form(synthesized):
    with TestClient(main.app) as c:
        r = c.post("/chatgpt/audio/speech", json={"input": "hola"})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "hola"
    assert body["exact_match"] is True
    assert body["url"].endswith(".mp3")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_capability_gate.py tests/test_audio_speech_bytes.py -v`
Expected: FAIL — the image endpoint answers `401`/`503` rather than `501`, and
`main` has no `Synthesized` / `_synthesize`.

- [ ] **Step 3: Write the implementation**

In `main.py`, next to the existing `_needs_account()` helper:

```python
def require_capability(name: str) -> None:
    """Refuse with 501 when this account cannot do `name`.

    501, not 404 and not 503, and the distinction is load-bearing for the
    gateway. A 404 is indistinguishable from a routing mistake. A 503 says "it
    broke" -- so the gateway retries, accumulates suspicion against the route and
    fails over, spending attempts on something that was never going to work on
    this plan. 501 says: this proxy, deliberately, does not do this right now.
    """
    if not capabilities.effective(capabilities.snapshot())[name]:
        raise HTTPException(
            501,
            f"This proxy cannot serve '{name}' with its current account "
            f"(see GET /health, capabilities.{name}).")
```

Add `require_capability("...")` as the first statement of each gated handler:

| Handler | Call |
|---|---|
| `POST /v1/images/generations` | `require_capability("images")` |
| `POST /v1/audio/speech` | `require_capability("audio_speech")` |
| `GET /v1/audio/from-message` | `require_capability("audio_speech")` |
| `POST /v1/audio/transcriptions` | `require_capability("audio_transcription")` |
| `POST·GET /v1/files`, `GET·DELETE /v1/files/{file_id}` | `require_capability("files")` |
| `GET /v1/conversations`, `GET /v1/conversations/{conversation_id}` | `require_capability("conversations")` |

Leave `/v1/chat/completions`, `/v1/models`, `/v1/translate`, `/v1/limits` and `/health` ungated: their capabilities are `True` in every mode.

In `audio_speech` (currently `main.py:2616`), the existing
`if not auth.is_authenticated(): return _needs_account()` at lines 2634-2635 is
**replaced** by `require_capability("audio_speech")` — same guard, same
fail-fast position before the chat turn is spent, but the status the contract
asks for.

Then split that handler in two. Extract everything from the chat turn to the
`synthesize` response into `_synthesize`, returning a small record:

```python
@dataclass(frozen=True)
class Synthesized:
    """One synthesis, before it is shaped for a particular endpoint."""
    audio: bytes
    media_type: str
    text: str            # what was ACTUALLY synthesized
    exact_match: bool    # False when the model altered the text on the way
    voice: str
    format: str
    conversation_id: str
    message_id: str


async def _synthesize(text: str, voice: str, fmt: str, model: str) -> Synthesized:
    """The chat-echo + /backend-api/synthesize round trip.

    Extracted so the OpenAI-shaped endpoint and the native one are two thin
    shells over one implementation, instead of one endpoint with a `raw=` flag
    threading a branch through 60 lines.
    """
    # ... the body of the current audio_speech, lines 2637-2675, unchanged
    # except that it returns Synthesized(...) instead of a dict.
```

The OpenAI-shaped endpoint returns bytes, because that is what the contract's
`audio_speech: true` promises:

```python
@app.post("/v1/audio/speech")
async def audio_speech(req: SpeechRequest, request: Request):
    """OpenAI-compatible TTS: raw audio bytes, with the correct Content-Type.

    It used to return JSON carrying an mp3 URL. Every OpenAI client writes the
    response body straight to a file, so that shape needed special-casing this
    one provider -- which is the thing the gateway in front of it exists to
    avoid. The extra facts this flow produces (`exact_match` in particular: it
    makes the model echo the input, and the model sometimes edits it) are real
    and are kept, as response headers, where a client that does not care never
    sees them and one that does can still read them. The JSON form lives on at
    /chatgpt/audio/speech.
    """
    require_capability("audio_speech")
    text = (req.input or "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": {
            "message": "'input' is required", "type": "invalid_request_error"}})
    s = await _synthesize(text, req.voice, req.format, req.model)
    stored = _store_audio(s.audio, s.message_id[:36], s.voice, s.format, s.media_type)
    return Response(content=s.audio, media_type=s.media_type, headers={
        "X-Audio-Url":       stored["url"],
        "X-Exact-Match":     "true" if s.exact_match else "false",
        "X-Conversation-Id": s.conversation_id,
        "X-Message-Id":      s.message_id,
    })


@app.post("/chatgpt/audio/speech")
async def chatgpt_audio_speech(req: SpeechRequest, request: Request):
    """The pre-contract JSON shape, under this provider's own prefix.

    Anything a provider offers beyond the standard surface lives here; the
    standard path stays the one every OpenAI client already knows.
    """
    require_capability("audio_speech")
    text = (req.input or "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": {
            "message": "'input' is required", "type": "invalid_request_error"}})
    s = await _synthesize(text, req.voice, req.format, req.model)
    stored = _store_audio(s.audio, s.message_id[:36], s.voice, s.format, s.media_type)
    return {**stored, "text": s.text, "exact_match": s.exact_match,
            "voice": s.voice, "format": s.format,
            "conversation_id": s.conversation_id, "message_id": s.message_id}
```

Add `from fastapi import Response` and `from dataclasses import dataclass` to
`main.py`'s imports if they are not already there.

Then update `CAPABILITIES.md`, appending after the endpoint matrix:

```markdown
## El contrato de capacidades

Desde la versión 2.5.0 este proxy publica en `GET /health` un bloque
`capabilities` con once booleanos y una clave `contract: 1`. Los valores son
**efectivos**: ya resueltos contra la cuenta y el plan de este despliegue. Si la
suscripción vence, `images` pasa a `false` solo, y llm-libre deja de rutear
generación de imágenes acá sin que nadie edite un YAML.

Un endpoint cuya capacidad está en `false` responde **`501 Not Implemented`**,
no `404` ni `503`: `404` no se distingue de un error de ruteo, y `503` hace que
el gateway reintente y acumule sospecha contra una ruta que en este plan nunca
iba a funcionar.

La matriz de arriba es la referencia humana; `GET /health` es la que leen las
máquinas, y es la que no se desactualiza.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/ -v`
Expected: PASS, 27 tests.

- [ ] **Step 5: Commit**

```bash
cd /Users/cristian/chatgpt-proxy
git add main.py CAPABILITIES.md tests/test_capability_gate.py tests/test_audio_speech_bytes.py
git commit -m "feat(gate): endpoints behave as the contract promises -- 501 when off, bytes from TTS"
```

---

## Part 3 — The gateway consumes the contract

### Task 6: `Capabilities` gains four axes, and the routes table with it

**Repo:** `/Users/cristian/llm-libre`

**Files:**
- Modify: `src/llm_libre/models.py:24-35`
- Modify: `src/llm_libre/storage.py:13-19` (SCHEMA), `:306-319` (migration), `:320-355` (`upsert_routes`), `:415-422` (`active_routes`)
- Test: `tests/test_models.py`, `tests/test_storage.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Capabilities(tools, vision, context, max_output, images=False, audio_speech=False, audio_transcription=False, translate=False, search=False)`; those four persisted as `routes.audio_speech`, `routes.audio_transcription`, `routes.translate`, `routes.search`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_models.py`:

```python
def test_the_new_capability_axes_default_to_false():
    # Same shape `images` took: every existing construction site keeps working
    # unchanged, and a provider has to CLAIM a capability to get it.
    c = Capabilities(tools=True, vision=False, context=1000, max_output=100)
    assert c.audio_speech is False
    assert c.audio_transcription is False
    assert c.translate is False
    assert c.search is False
```

Append to `tests/test_storage.py`:

```python
def test_the_new_capability_axes_survive_a_round_trip():
    store = _store()
    caps = Capabilities(tools=False, vision=True, context=52815, max_output=8192,
                        images=True, audio_speech=True, audio_transcription=True,
                        translate=True, search=True)
    store.upsert_routes([Route("chatgpt", "gpt-5-6", "free", caps)], 100.0)
    assert store.active_routes()[0].capabilities == caps


def test_an_old_database_migrates_the_new_columns_to_false():
    # A row written before these columns existed predates anyone measuring the
    # capability, so it migrates to "cannot", never to a guess. The next sweep
    # overwrites it with what the provider actually reports.
    store = _store()
    store._con.execute("ALTER TABLE routes DROP COLUMN audio_speech")
    store._con.execute("ALTER TABLE routes DROP COLUMN audio_transcription")
    store._con.execute("ALTER TABLE routes DROP COLUMN translate")
    store._con.execute("ALTER TABLE routes DROP COLUMN search")
    store._con.execute(
        "INSERT INTO routes (key, provider, model_id, tier, tools, vision, "
        "context, max_output, last_seen) VALUES "
        "('chatgpt/old', 'chatgpt', 'old', 'free', 0, 0, 1000, 100, 50.0)")
    store._con.commit()
    store.create_schema()
    route = [r for r in store.active_routes() if r.model_id == "old"][0]
    assert route.capabilities.audio_speech is False
    assert route.capabilities.search is False
```

> `tests/test_storage.py` already defines `_store()`; reuse it. Add
> `Capabilities` and `Route` to that file's imports from `llm_libre.models` if
> they are not there yet.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_models.py tests/test_storage.py -v`
Expected: FAIL — `AttributeError: 'Capabilities' object has no attribute 'audio_speech'`

- [ ] **Step 3: Write the implementation**

In `src/llm_libre/models.py`, extend the dataclass after `images`:

```python
    # PROVIDER-LEVEL capabilities, stamped identically onto every route of a
    # provider: unlike tools/vision/images they do not vary per model, because
    # the endpoints behind them take no model. They live on Capabilities anyway
    # so a future /v1/audio/speech can filter routes with the same
    # `compatible_routes` machinery as everything else, instead of growing a
    # second, parallel notion of what a provider can do.
    #
    # All four default to False, exactly as `images` did: every existing
    # construction site keeps working unchanged, and a provider gains a
    # capability by claiming it, never by omission.
    audio_speech: bool = False           # POST /v1/audio/speech      (TTS)
    audio_transcription: bool = False    # POST /v1/audio/transcriptions (STT)
    translate: bool = False              # POST /v1/translate
    search: bool = False                 # web_search on chat completions
```

In `src/llm_libre/storage.py`, extend `SCHEMA`'s `routes` table:

```sql
    images INTEGER NOT NULL DEFAULT 0,
    audio_speech INTEGER NOT NULL DEFAULT 0,
    audio_transcription INTEGER NOT NULL DEFAULT 0,
    translate INTEGER NOT NULL DEFAULT 0,
    search INTEGER NOT NULL DEFAULT 0);
```

In the migration block, after the `images` migration:

```python
        # Same DEFAULT 0 reasoning as `images` above: a row that predates these
        # columns predates anyone measuring the capability, so it migrates to
        # "cannot" rather than to a guess. The next catalogue sync overwrites it
        # with what the provider reports.
        for column in ("audio_speech", "audio_transcription", "translate", "search"):
            if column not in route_columns:
                self._con.execute(
                    f"ALTER TABLE routes ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0")
```

In `upsert_routes`, extend the INSERT:

```python
                """INSERT INTO routes (key, provider, model_id, tier, tools, vision,
                       context, max_output, last_seen, active, priority, images,
                       audio_speech, audio_transcription, translate, search)
                   VALUES (?,?,?,?,?,?,?,?,?,1,?,?,?,?,?,?)
                   ON CONFLICT(key) DO UPDATE SET
                       tools=excluded.tools, vision=excluded.vision,
                       context=excluded.context, max_output=excluded.max_output,
                       last_seen=excluded.last_seen, active=1,
                       priority=excluded.priority, images=excluded.images,
                       audio_speech=excluded.audio_speech,
                       audio_transcription=excluded.audio_transcription,
                       translate=excluded.translate, search=excluded.search""",
                (r.key, r.provider, r.model_id, r.tier, int(c.tools), int(c.vision),
                 c.context, c.max_output, timestamp, r.priority, int(c.images),
                 int(c.audio_speech), int(c.audio_transcription),
                 int(c.translate), int(c.search)))
```

In `active_routes`:

```python
        rows = self._con.execute(
            """SELECT provider, model_id, tier, tools, vision, context, max_output,
                      priority, images, audio_speech, audio_transcription,
                      translate, search
               FROM routes WHERE active = 1 ORDER BY key""").fetchall()
        return [Route(p, m, t,
                      Capabilities(bool(to), bool(vi), cx, ms, images=bool(im),
                                   audio_speech=bool(sp), audio_transcription=bool(tr),
                                   translate=bool(tl), search=bool(se)),
                      priority=pr)
                for p, m, t, to, vi, cx, ms, pr, im, sp, tr, tl, se in rows]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, whole suite green.

- [ ] **Step 5: Commit**

```bash
cd /Users/cristian/llm-libre
git add src/llm_libre/models.py src/llm_libre/storage.py tests/test_models.py tests/test_storage.py
git commit -m "feat(models): four provider-level capability axes, persisted with the routes"
```

---

### Task 7: `reads_capabilities` on a provider

**Repo:** `/Users/cristian/llm-libre`

**Files:**
- Modify: `src/llm_libre/providers.py:71` (the `Provider` dataclass), `:171-190` (`load`)
- Modify: `providers.yaml` (the `chatgpt` entry)
- Test: `tests/test_providers.py`, `tests/test_wire_contract.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Provider.reads_capabilities: bool = False`, read from the YAML key `reads_capabilities`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_providers.py`:

```python
def test_reads_capabilities_defaults_to_false(tmp_path):
    yaml_path = tmp_path / "p.yaml"
    yaml_path.write_text(
        "providers:\n"
        "  - id: kilo\n    tier: free\n    dialect: openai\n"
        "    base_url: https://k.test\n    models_path: /models\n")
    assert load(str(yaml_path), {})[0].reads_capabilities is False


def test_reads_capabilities_is_read_from_the_yaml(tmp_path):
    yaml_path = tmp_path / "p.yaml"
    yaml_path.write_text(
        "providers:\n"
        "  - id: chatgpt\n    tier: free\n    dialect: openai\n"
        "    base_url: https://c.test/v1\n    models_path: /models\n"
        "    reads_capabilities: true\n")
    assert load(str(yaml_path), {})[0].reads_capabilities is True


def test_the_real_registry_only_lets_chatgpt_read_the_contract():
    # The rollout is per proxy. Any other provider turning this on before its
    # proxy publishes /health would cost it a sweep of skipped syncs.
    providers = load("providers.yaml", {})
    assert [p.id for p in providers if p.reads_capabilities] == ["chatgpt"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_providers.py -v -k reads_capabilities`
Expected: FAIL — `TypeError: Provider.__init__() got an unexpected keyword argument` / `AttributeError`.

- [ ] **Step 3: Write the implementation**

In `src/llm_libre/providers.py`, add to `Provider` after `emulates_tools`:

```python
    # Whether this provider's /health publishes the capability contract (spec
    # 2026-08-20). False -- the default -- is today's behaviour exactly:
    # capabilities come from default_capabilities/fixed_models and nothing extra
    # is requested. It is opt-in per provider precisely so the five in-house
    # proxies can adopt the contract one at a time, in any order, with no flag
    # day: a provider that has not adopted it must not have its sync skipped.
    reads_capabilities: bool = False
```

In `load()`, add to the constructor call:

```python
        reads_capabilities=bool(p.get("reads_capabilities", False)),
```

In `providers.yaml`, in the `chatgpt` entry, immediately after `timeout_s: 150`:

```yaml
    # Its /health publishes the capability contract (spec 2026-08-20): eleven
    # booleans already resolved against the account's plan, plus context_window
    # per model in /v1/models. That replaces guessing here -- `context: 128000`
    # below was 2.4x the real 52815 -- and it is what makes `images` turn itself
    # off when the Go subscription lapses (2026-09-06) instead of sending every
    # image request to a route that answers 503.
    #
    # The declarations below STAY, and they are not dead weight: they are the
    # fallback for a sweep where /health could not be read, and `exceptions`
    # remains the strongest voice of all -- see the precedence note in
    # catalog.normalize. `tools: false` in particular is a MEASURED correction
    # to what this proxy says about itself, and no discovery may overrule it.
    reads_capabilities: true
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_providers.py tests/test_wire_contract.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/cristian/llm-libre
git add src/llm_libre/providers.py providers.yaml tests/test_providers.py
git commit -m "feat(providers): opt a provider into reading the capability contract"
```

---

### Task 8: capability precedence in `catalog`

**Repo:** `/Users/cristian/llm-libre`

**Files:**
- Modify: `src/llm_libre/catalog.py:150-196` (`normalize`)
- Test: `tests/test_catalog.py`

**Interfaces:**
- Consumes: `contract.ProviderContract` (Task 1), `Capabilities` (Task 6).
- Produces: `catalog.capabilities_from_contract(c: ProviderContract, fallback: Capabilities | None) -> Capabilities`, `catalog.apply_model_metadata(base: Capabilities, m: dict, provider: str) -> Capabilities`, and `normalize(..., contract: ProviderContract | None = None)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_catalog.py`:

```python
from llm_libre.catalog import apply_model_metadata, capabilities_from_contract
from llm_libre.contract import REQUIRED_CAPABILITIES, Auth, ProviderContract


def _contract(**overrides):
    caps = {k: False for k in REQUIRED_CAPABILITIES}
    caps.update(chat=True, streaming=True, vision=True, images=True,
                translate=True, search=True)
    caps.update(overrides)
    return ProviderContract(version=1, provider="chatgpt",
                            auth=Auth(mode="account", plan="go",
                                      subscription_active=True),
                            capabilities=caps)


def test_the_contract_supplies_the_provider_level_capabilities():
    routes = normalize("chatgpt", {"data": [{"id": "gpt-5-6"}]},
                       default_capabilities=_CHATGPT_DEFAULTS,
                       contract=_contract())
    c = routes[0].capabilities
    assert c.vision is True
    assert c.images is True
    assert c.translate is True
    assert c.search is True
    assert c.tools is False


def test_per_model_metadata_supplies_the_real_context_window():
    # The whole reason this exists: 128000 declared, 52815 real.
    routes = normalize("chatgpt",
                       {"data": [{"id": "gpt-5-6", "context_window": 52815,
                                  "max_output_tokens": 8192}]},
                       default_capabilities=_CHATGPT_DEFAULTS,
                       contract=_contract())
    assert routes[0].capabilities.context == 52815
    assert routes[0].capabilities.max_output == 8192


def test_a_model_without_a_context_window_falls_back_to_the_yaml():
    routes = normalize("chatgpt", {"data": [{"id": "gpt-5-6"}]},
                       default_capabilities=_CHATGPT_DEFAULTS,
                       contract=_contract())
    assert routes[0].capabilities.context == _CHATGPT_DEFAULTS.context


def test_a_per_model_capability_may_narrow_the_provider_level_one():
    routes = normalize("chatgpt",
                       {"data": [{"id": "gpt-image-1",
                                  "capabilities": {"vision": False, "images": True}}]},
                       default_capabilities=_CHATGPT_DEFAULTS,
                       contract=_contract())
    assert routes[0].capabilities.vision is False
    assert routes[0].capabilities.images is True


def test_a_per_model_capability_may_not_widen_the_provider_level_one(caplog):
    with caplog.at_level(logging.WARNING):
        routes = normalize("chatgpt",
                           {"data": [{"id": "gpt-5-6",
                                      "capabilities": {"images": True}}]},
                           default_capabilities=_CHATGPT_DEFAULTS,
                           contract=_contract(images=False))
    assert routes[0].capabilities.images is False
    assert "gpt-5-6" in caplog.text


def test_exceptions_beat_the_contract():
    # The strongest voice, and it must stay that way: `tools: false` for chatgpt
    # is a MEASURED correction (0/3 tool_calls, twice) to what the proxy claims
    # about itself. A discovery source that could overrule it would re-open the
    # exact failure the declaration exists to prevent.
    routes = normalize("chatgpt", {"data": [{"id": "gpt-5-6"}]},
                       default_capabilities=_CHATGPT_DEFAULTS,
                       exceptions={"gpt-5-6": {"vision": False}},
                       contract=_contract(vision=True))
    assert routes[0].capabilities.vision is False


def test_without_a_contract_nothing_changes():
    routes = normalize("chatgpt", {"data": [{"id": "gpt-5-6"}]},
                       default_capabilities=_CHATGPT_DEFAULTS)
    assert routes[0].capabilities == _CHATGPT_DEFAULTS
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_catalog.py -v -k contract`
Expected: FAIL — `ImportError: cannot import name 'capabilities_from_contract'`

- [ ] **Step 3: Write the implementation**

In `src/llm_libre/catalog.py`, add the import and the two helpers above `normalize`:

```python
from llm_libre.contract import ProviderContract

# The three capabilities that genuinely vary between models of the SAME
# provider: grok publishes 31 ids of which the three imagine-agent-mode ones
# draw but neither chat nor see, while the other 28 are the opposite. Everything
# else in the contract is a property of the provider's account, not of a model.
_PER_MODEL = ("tools", "vision", "images")


def capabilities_from_contract(c: ProviderContract,
                               fallback: Capabilities | None) -> Capabilities:
    """The provider-level capabilities a contract asserts.

    The contract carries no sizes: `context` and `max_output` vary per model and
    arrive from /v1/models (see `apply_model_metadata`). Until one does, the YAML
    declaration is what there is -- and 0 when there is not even that, which is
    the honest value: `x_min_context` filtering on 0 excludes the route from
    requests that need a big window, rather than promising one it may not have.
    """
    caps = c.capabilities
    return Capabilities(
        tools=caps["tools"],
        vision=caps["vision"],
        context=fallback.context if fallback else 0,
        max_output=fallback.max_output if fallback else 0,
        images=caps["images"],
        audio_speech=caps["audio_speech"],
        audio_transcription=caps["audio_transcription"],
        translate=caps["translate"],
        search=caps["search"],
    )


def apply_model_metadata(base: Capabilities, m: dict, provider: str) -> Capabilities:
    """Narrow `base` with what /v1/models says about THIS model.

    A per-model value may only be NARROWER than the provider-level one. Widening
    is a contradiction, not extra information: the provider-level block is
    resolved against the ACCOUNT, so "this account cannot generate images" is a
    fact about the whole proxy, and a model claiming otherwise is describing what
    it could do on some other plan. It is logged and dropped, rather than
    refusing the whole document, because one confused entry should not cost the
    other twelve their real context windows.
    """
    fields = {}
    context = m.get("context_window")
    if isinstance(context, int) and not isinstance(context, bool) and context > 0:
        fields["context"] = context
    max_output = m.get("max_output_tokens")
    if isinstance(max_output, int) and not isinstance(max_output, bool) and max_output >= 0:
        fields["max_output"] = max_output
    per_model = m.get("capabilities")
    if isinstance(per_model, dict):
        for name in _PER_MODEL:
            value = per_model.get(name)
            if not isinstance(value, bool):
                continue
            if value and not getattr(base, name):
                log.warning(
                    "catalog %s: model %s claims %s=true while the provider "
                    "reports it false; the provider-level value wins.",
                    provider, m.get("id"), name)
                continue
            fields[name] = value
    return replace(base, **fields) if fields else base
```

Change the `normalize` signature and its capability branch:

```python
def normalize(provider: str, data: dict | list, priority: int = 100,
              default_capabilities: Capabilities | None = None,
              exceptions: dict | None = None,
              emulates_tools: bool = False,
              measured_rates: dict[str, float] | None = None,
              contract: ProviderContract | None = None) -> list[Route]:
```

Extend the docstring with the precedence, then replace the branch:

```python
        # PRECEDENCE, strongest first -- and the order is the whole design:
        #   1. `exceptions` (providers.yaml)  -- where a MEASURED lie is recorded
        #   2. /v1/models per-model values    -- narrowing only
        #   3. /health provider-level values  -- the contract
        #   4. `default_capabilities` (YAML)  -- the fallback for a proxy that
        #                                        has not adopted the contract
        if contract is not None:
            capabilities = apply_model_metadata(
                capabilities_from_contract(contract, default_capabilities),
                m, provider)
        elif default_capabilities is not None:
            capabilities = default_capabilities
        else:
            if not _is_free(m):
                continue
            arch = m.get("architecture") or {}
            outputs = set(arch.get("output_modalities") or ["text"])
            if outputs != {"text"}:
                continue
            supported = m.get("supported_parameters") or []
            top = m.get("top_provider") or {}
            capabilities = Capabilities(
                tools="tools" in supported,
                vision="image" in (arch.get("input_modalities") or []),
                context=int(m.get("context_length") or top.get("context_length") or 0),
                max_output=int(top.get("max_completion_tokens") or 0),
            )
        # `exceptions` applies to BOTH declared paths, and applies LAST: it is
        # the hand-written override, and the only thing that may contradict a
        # provider's own report of itself.
        if contract is not None or default_capabilities is not None:
            override = (exceptions or {}).get(m["id"])
            if override:
                capabilities = replace(capabilities, **override)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_catalog.py -v`
Expected: PASS, including every pre-existing test.

- [ ] **Step 5: Commit**

```bash
cd /Users/cristian/llm-libre
git add src/llm_libre/catalog.py tests/test_catalog.py
git commit -m "feat(catalog): resolve capabilities from the contract, exceptions still strongest"
```

---

### Task 9: a fixed route the contract contradicts is dropped

**Repo:** `/Users/cristian/llm-libre`

**Files:**
- Modify: `src/llm_libre/providers.py:194-202` (`fixed_routes`)
- Test: `tests/test_providers.py`

**Interfaces:**
- Consumes: `contract.ProviderContract`.
- Produces: `fixed_routes(p: Provider, contract: ProviderContract | None = None) -> list[Route]`.

> **Why this task exists:** the spec's precedence covers `exceptions` and
> discovery but says nothing about `fixed_models`, and `chatgpt` declares
> `dall-e-3` there with `images: true`. Without this, the one route the whole
> design is meant to retire when the Go plan lapses would survive.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_providers.py`:

```python
from llm_libre.contract import REQUIRED_CAPABILITIES, Auth, ProviderContract
from llm_libre.providers import Provider, fixed_routes


def _contract(**overrides):
    caps = {k: False for k in REQUIRED_CAPABILITIES}
    caps.update(chat=True, streaming=True, images=True)
    caps.update(overrides)
    return ProviderContract(version=1, provider="chatgpt",
                            auth=Auth(mode="account"), capabilities=caps)


def _chatgpt():
    return Provider("chatgpt", "free", "openai", "https://c.test/v1", "", "/models",
                    {}, [{"id": "dall-e-3", "tools": False, "vision": False,
                          "images": True, "context": 128000, "max_output": 0}])


def test_a_fixed_route_survives_when_the_contract_confirms_it():
    routes = fixed_routes(_chatgpt(), contract=_contract(images=True))
    assert [r.model_id for r in routes] == ["dall-e-3"]
    assert routes[0].capabilities.images is True


def test_a_fixed_route_is_dropped_when_the_contract_contradicts_it():
    # The event the design exists for: the Go plan lapses, `images` goes false,
    # and dall-e-3 -- which can do nothing else -- leaves the catalogue instead
    # of staying on as a chat route that answers nothing.
    assert fixed_routes(_chatgpt(), contract=_contract(images=False)) == []


def test_without_a_contract_a_fixed_route_is_untouched():
    routes = fixed_routes(_chatgpt())
    assert [r.model_id for r in routes] == ["dall-e-3"]
    assert routes[0].capabilities.images is True


def test_a_fixed_route_gains_the_provider_level_axes():
    routes = fixed_routes(_chatgpt(), contract=_contract(images=True, translate=True))
    assert routes[0].capabilities.translate is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_providers.py -v -k fixed_route`
Expected: FAIL — `TypeError: fixed_routes() got an unexpected keyword argument 'contract'`

- [ ] **Step 3: Write the implementation**

Replace `fixed_routes` in `src/llm_libre/providers.py`:

```python
def fixed_routes(p: Provider, contract=None) -> list[Route]:
    """The routes declared by hand in `fixed_models`.

    When the provider publishes the capability contract, a declaration here is
    checked against it and the route is DROPPED -- not silently downgraded --
    when the contract contradicts it. `chatgpt`'s `dall-e-3` is the case: it
    declares `images: true` and can do nothing else, so on a lapsed plan
    downgrading it to `images: false` would leave a route that looks like an
    ordinary chat model, gets picked for chat, and answers nothing. Dropping it
    is the honest outcome: the declaration asserted a shape the provider no
    longer has.

    `fixed_models` is not `exceptions`, and only `exceptions` outranks discovery.
    This block exists to name ids a catalogue does not publish, not to overrule
    live account state -- which is precisely what "the Go subscription expired"
    is.
    """
    routes = []
    for m in p.fixed_models:
        capabilities = Capabilities(
            tools=True if p.emulates_tools else bool(m["tools"]),
            vision=bool(m["vision"]),
            context=int(m["context"]), max_output=int(m["max_output"]),
            images=bool(m.get("images", False)))
        if contract is not None:
            caps = contract.capabilities
            contradicted = [name for name in ("tools", "vision", "images")
                            if getattr(capabilities, name) and not caps[name]]
            if contradicted:
                log.warning(
                    "%s: the fixed model %s declares %s, which this provider's "
                    "/health reports it cannot do right now. The route is left "
                    "out of the catalogue for this sweep.",
                    p.id, m["id"], ", ".join(contradicted))
                continue
            capabilities = replace(
                capabilities,
                audio_speech=caps["audio_speech"],
                audio_transcription=caps["audio_transcription"],
                translate=caps["translate"],
                search=caps["search"])
        routes.append(Route(p.id, m["id"], p.tier, capabilities, priority=p.priority))
    return routes
```

Add `from dataclasses import dataclass, field, replace` to the imports at the top of `providers.py` (it currently imports only `dataclass, field`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_providers.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/cristian/llm-libre
git add src/llm_libre/providers.py tests/test_providers.py
git commit -m "feat(providers): a fixed route the contract contradicts leaves the catalogue"
```

---

### Task 10: `probing` fetches, persists and degrades

**Repo:** `/Users/cristian/llm-libre`

**Files:**
- Modify: `src/llm_libre/storage.py` (SCHEMA, plus two methods)
- Modify: `src/llm_libre/probing.py:78-150` (`sync_catalogue`), `:296-320` (`cycle`)
- Test: `tests/test_probing.py`, `tests/test_storage.py`

**Interfaces:**
- Consumes: `contract.parse_health` (Task 1), `Provider.reads_capabilities` (Task 7), `normalize(..., contract=)` (Task 8), `fixed_routes(p, contract=)` (Task 9).
- Produces: `Storage.put_contract(provider: str, doc: dict, timestamp: float) -> None`, `Storage.get_contract(provider: str) -> dict | None`, `Storage.all_contracts() -> dict[str, dict]`; `sync_catalogue(http, providers, store, now, notifier=None)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_storage.py`:

```python
def test_a_contract_document_round_trips():
    store = _store()
    doc = {"contract": 1, "capabilities": {"images": True}}
    store.put_contract("chatgpt", doc, 100.0)
    assert store.get_contract("chatgpt") == doc


def test_putting_a_contract_twice_keeps_the_latest():
    store = _store()
    store.put_contract("chatgpt", {"contract": 1, "n": 1}, 100.0)
    store.put_contract("chatgpt", {"contract": 1, "n": 2}, 200.0)
    assert store.get_contract("chatgpt")["n"] == 2


def test_an_unknown_provider_has_no_contract():
    assert _store().get_contract("nobody") is None
```

Append to `tests/test_probing.py`:

```python
from llm_libre.contract import REQUIRED_CAPABILITIES

_CAPS = {k: False for k in REQUIRED_CAPABILITIES}
_HEALTH = {"status": "ok", "contract": 1, "provider": "chatgpt",
           "auth": {"mode": "account", "plan": "go", "subscription_active": True},
           "capabilities": {**_CAPS, "chat": True, "streaming": True,
                            "vision": True, "images": True}}
_MODELS = {"data": [{"id": "gpt-5-6", "context_window": 52815,
                     "max_output_tokens": 8192,
                     "capabilities": {"tools": False, "vision": True,
                                      "images": False}}]}


def _chatgpt(**kw):
    return Provider("chatgpt", "free", "openai", "https://c.test/v1", "",
                    "/models", {}, [], reads_capabilities=True, **kw)


def _routed(health=None, models=None, health_status=200):
    def handler(req):
        if req.url.path.endswith("/health"):
            return httpx.Response(health_status, json=health or _HEALTH)
        return httpx.Response(200, json=models or _MODELS)
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_sync_applies_the_contract_to_the_discovered_routes():
    store = _store()
    await sync_catalogue(_routed(), [_chatgpt()], store, now=100.0)
    caps = store.active_routes()[0].capabilities
    assert caps.vision is True
    assert caps.context == 52815          # not the 128000 anyone declared
    assert caps.images is False           # narrowed per model


async def test_sync_persists_the_contract_document():
    store = _store()
    await sync_catalogue(_routed(), [_chatgpt()], store, now=100.0)
    assert store.get_contract("chatgpt")["auth"]["plan"] == "go"


async def test_a_failing_health_keeps_the_previous_catalogue(caplog):
    store = _store()
    await sync_catalogue(_routed(), [_chatgpt()], store, now=100.0)
    before = store.active_routes()
    with caplog.at_level(logging.WARNING):
        await sync_catalogue(_routed(health_status=500), [_chatgpt()], store, now=200.0)
    assert store.active_routes() == before
    assert "chatgpt" in caplog.text


async def test_a_provider_that_does_not_read_capabilities_never_requests_health():
    seen = []

    def handler(req):
        seen.append(req.url.path)
        return httpx.Response(200, json=CATALOGUE)

    store = _store()
    prov = [Provider("kilo", "free", "openai", "https://k.test", "", "/models", {}, [])]
    await sync_catalogue(httpx.AsyncClient(transport=httpx.MockTransport(handler)),
                         prov, store, now=100.0)
    assert not any(p.endswith("/health") for p in seen)


async def test_a_health_without_the_contract_falls_back_to_the_yaml():
    store = _store()
    defaults = Capabilities(tools=False, vision=False, context=128000, max_output=8192)
    provider = Provider("chatgpt", "free", "openai", "https://c.test/v1", "",
                        "/models", {}, [], reads_capabilities=True,
                        default_capabilities=defaults)
    await sync_catalogue(_routed(health={"status": "ok"}), [provider], store, now=100.0)
    assert store.active_routes()[0].capabilities.context == 128000
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_probing.py tests/test_storage.py -v -k contract`
Expected: FAIL — `AttributeError: 'Storage' object has no attribute 'put_contract'`

- [ ] **Step 3: Write the implementation**

In `src/llm_libre/storage.py`, add to `SCHEMA`:

```sql
CREATE TABLE IF NOT EXISTS provider_contracts (
    provider TEXT PRIMARY KEY, doc TEXT NOT NULL, seen_at REAL NOT NULL);
```

and the three methods:

```python
    def put_contract(self, provider: str, doc: dict, timestamp: float) -> None:
        """Record the last /health document a provider published.

        Persisted rather than kept in memory for two reasons, both real: the
        gateway's own /health has to be able to report the contract without
        waiting for a sweep after a restart, and the capability-off alert needs
        the PREVIOUS document to detect a transition -- an in-memory copy would
        make every restart look like a fresh start and either re-alert or go
        silent.
        """
        self._con.execute(
            """INSERT INTO provider_contracts (provider, doc, seen_at)
               VALUES (?,?,?)
               ON CONFLICT(provider) DO UPDATE SET
                   doc=excluded.doc, seen_at=excluded.seen_at""",
            (provider, json.dumps(doc), timestamp))
        self._con.commit()

    def get_contract(self, provider: str) -> dict | None:
        row = self._con.execute(
            "SELECT doc FROM provider_contracts WHERE provider = ?",
            (provider,)).fetchone()
        return json.loads(row[0]) if row else None

    def all_contracts(self) -> dict:
        return {p: json.loads(d) for p, d in self._con.execute(
            "SELECT provider, doc FROM provider_contracts")}
```

Ensure `import json` is present at the top of `storage.py`.

In `src/llm_libre/probing.py`, add the import:

```python
from llm_libre.contract import parse_health
```

and restructure the top of the per-provider loop in `sync_catalogue`:

```python
    for p in providers:
        headers = dict(p.extra_headers)
        if p.api_key.strip():
            headers["Authorization"] = "Bearer " + p.api_key
        # The capability contract, BEFORE anything is written for this provider.
        # A failure here skips the provider entirely -- exactly what a failing
        # /models already does -- so the previous catalogue survives instead of
        # being rewritten with fallback values that would look like a real
        # measurement. Keeping what is known beats erasing it.
        contract = None
        if p.reads_capabilities:
            try:
                r = await http.get(join_path(p.base_url, "/health"),
                                   headers=headers, timeout=15.0)
                r.raise_for_status()
                doc = r.json()
            except (httpx.HTTPError, ValueError) as e:
                log.warning(
                    "capabilities of %s: could not read /health (%s: %s). "
                    "Keeping the previous catalogue for this provider.",
                    p.id, type(e).__name__, e)
                continue
            contract = parse_health(p.id, doc)
            if contract is None:
                log.warning(
                    "capabilities of %s: /health does not implement the "
                    "capability contract; falling back to providers.yaml.", p.id)
            else:
                # `_announce_changes` compares against the STORED document, so
                # it must run BEFORE the new one replaces it. What it returns is
                # its own bookkeeping (see Task 12), merged in so a single write
                # persists both the contract and the alert state.
                extra = _announce_changes(store, p.id, contract, notifier)
                store.put_contract(p.id, {**doc, **extra}, now)
        if p.fixed_models:
            routes = fixed_routes(p, contract=contract)
            store.upsert_routes(routes, now, deactivate_missing=True, provider=p.id)
            total += len(routes)
        if not p.models_path:
            continue
        try:
            r = await http.get(join_path(p.base_url, p.models_path),
                               headers=headers, timeout=30.0)
```

> Delete the now-duplicated `headers = ...` lines that used to sit between the
> `models_path` guard and the `/models` request.

Pass the contract into `normalize` at the call site further down that function:

```python
        routes = normalize(p.id, data, priority=p.priority,
                           default_capabilities=p.default_capabilities,
                           exceptions=p.exceptions,
                           emulates_tools=p.emulates_tools,
                           measured_rates=measured_rates,
                           contract=contract)
```

Change the signature and have `cycle` pass the notifier:

```python
async def sync_catalogue(http: httpx.AsyncClient, providers: list[Provider],
                         store, now: float, notifier=None) -> int:
```

```python
    await sync_catalogue(state.http, state.providers, state.store, now,
                         notifier=state.proxy.notifier)
```

`_announce_changes` is written in Task 12; for now add the stub so this task's
tests pass on their own:

```python
def _announce_changes(store, provider: str, contract, notifier) -> dict:
    """Alert on capability transitions, and return any bookkeeping to persist
    alongside the contract document. Filled in by Task 12."""
    return {}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, whole suite green.

- [ ] **Step 5: Commit**

```bash
cd /Users/cristian/llm-libre
git add src/llm_libre/probing.py src/llm_libre/storage.py tests/test_probing.py tests/test_storage.py
git commit -m "feat(probing): read each proxy's capability contract, keep the catalogue when it fails"
```

---

### Task 11: the gateway surfaces what each proxy says

**Repo:** `/Users/cristian/llm-libre`

**Files:**
- Modify: `src/llm_libre/api.py:631` (the `/health` handler), `:595-621` (the `/v1/ranking` row builder)
- Test: `tests/test_api.py`, `tests/test_wire_contract.py`

**Interfaces:**
- Consumes: `Storage.all_contracts()` (Task 10), `Capabilities` (Task 6).
- Produces: a `providers` object in the gateway's `/health` body, keyed by provider id, each carrying `contract`, `auth_mode`, `plan`, `expires_at`, `capabilities`; four new columns on every `/v1/ranking` row.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_api.py`:

```python
def test_health_reports_each_providers_contract(client, state):
    state.store.put_contract("chatgpt", {
        "contract": 1,
        "auth": {"mode": "account", "plan": "go", "subscription_active": True,
                 "expires_at": "2026-09-06T00:28:46Z"},
        "capabilities": {"images": True, "vision": True},
    }, 100.0)
    body = client.get("/health").json()
    entry = body["providers"]["chatgpt"]
    assert entry["contract"] == 1
    assert entry["auth_mode"] == "account"
    assert entry["plan"] == "go"
    assert entry["expires_at"] == "2026-09-06T00:28:46Z"
    assert entry["capabilities"]["images"] is True


def test_health_reports_a_provider_without_a_contract_as_null(client, state):
    body = client.get("/health").json()
    assert body["providers"] == {}
```

```python
def test_ranking_rows_carry_the_new_capability_axes(client, state):
    caps = Capabilities(tools=False, vision=True, context=52815, max_output=8192,
                        audio_speech=True, translate=True, search=True)
    state.store.upsert_routes([Route("chatgpt", "gpt-5-6", "free", caps)], time.time())
    row = client.get("/v1/ranking", headers=_KEY).json()["routes"][0]
    assert row["audio_speech"] is True
    assert row["audio_transcription"] is False
    assert row["translate"] is True
    assert row["search"] is True
```

> `tests/test_api.py` already builds a `client`/`state` pair and already has a
> header constant for the API key; reuse the existing fixtures and constant
> rather than adding new ones. Add `Capabilities`/`Route` to its imports from
> `llm_libre.models` if they are not there yet.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_api.py -v -k contract`
Expected: FAIL — `KeyError: 'providers'`

- [ ] **Step 3: Write the implementation**

In `src/llm_libre/api.py`, inside the `/health` handler, build the block and add
it to the returned body:

```python
        # What each in-house proxy says about itself, so an operator can see
        # "chatgpt: account/go, images on, expires 2026-09-06" without opening a
        # shell. Read from the database, not from memory: after a restart this
        # has to answer before the first sweep has run.
        contracts = {}
        for provider, doc in state.store.all_contracts().items():
            auth = doc.get("auth") or {}
            contracts[provider] = {
                "contract":     doc.get("contract"),
                "auth_mode":    auth.get("mode"),
                "plan":         auth.get("plan"),
                "expires_at":   auth.get("expires_at"),
                "capabilities": doc.get("capabilities") or {},
            }
```

and include `"providers": contracts` in the response dict.

In the `/v1/ranking` row builder, next to the existing `"images"` and
`"context"` entries, add the four axes:

```python
                         # The provider-level axes. They do not vary between a
                         # provider's routes, and they are here anyway for the
                         # same reason `images` is: the question this table
                         # answers is "which routes could even be considered for
                         # X", and an operator should not have to cross-reference
                         # /health to answer it for audio or translation.
                         "audio_speech": r.capabilities.audio_speech,
                         "audio_transcription": r.capabilities.audio_transcription,
                         "translate": r.capabilities.translate,
                         "search": r.capabilities.search,
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_api.py tests/test_wire_contract.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/cristian/llm-libre
git add src/llm_libre/api.py tests/test_api.py
git commit -m "feat(health): surface each proxy's contract, plan and capability axes"
```

---

### Task 12: alert when a capability disappears

**Repo:** `/Users/cristian/llm-libre`

**Files:**
- Modify: `src/llm_libre/probing.py` (`_announce_changes`, stubbed in Task 10)
- Test: `tests/test_probing.py`

**Interfaces:**
- Consumes: `Storage.get_contract` (Task 10), `notify.Notifier.notify`.
- Produces: `probing.EXPIRY_WARNING_S: float`; `_announce_changes(store, provider, contract, notifier) -> dict` fully implemented, returning the keys `sync_catalogue` merges into the stored document.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_probing.py`:

```python
class _Spy:
    def __init__(self):
        self.sent = []

    def notify(self, text):
        self.sent.append(text)


async def test_a_capability_turning_off_is_alerted():
    store, spy = _store(), _Spy()
    await sync_catalogue(_routed(), [_chatgpt()], store, now=100.0, notifier=spy)
    off = {**_HEALTH, "capabilities": {**_HEALTH["capabilities"], "images": False}}
    await sync_catalogue(_routed(health=off), [_chatgpt()], store, now=200.0,
                         notifier=spy)
    assert len(spy.sent) == 1
    assert "images" in spy.sent[0]
    assert "chatgpt" in spy.sent[0]


async def test_a_capability_turning_on_is_not_alerted():
    store, spy = _store(), _Spy()
    off = {**_HEALTH, "capabilities": {**_HEALTH["capabilities"], "images": False}}
    await sync_catalogue(_routed(health=off), [_chatgpt()], store, now=100.0,
                         notifier=spy)
    await sync_catalogue(_routed(), [_chatgpt()], store, now=200.0, notifier=spy)
    assert spy.sent == []


async def test_an_unchanged_capability_set_is_not_re_alerted():
    store, spy = _store(), _Spy()
    off = {**_HEALTH, "capabilities": {**_HEALTH["capabilities"], "images": False}}
    for at in (100.0, 200.0, 300.0):
        await sync_catalogue(_routed(health=off), [_chatgpt()], store, now=at,
                             notifier=spy)
    assert spy.sent == []


async def test_the_first_sweep_ever_does_not_alert():
    # Nothing to compare against is not the same as "it just turned off".
    store, spy = _store(), _Spy()
    off = {**_HEALTH, "capabilities": {**_HEALTH["capabilities"], "images": False}}
    await sync_catalogue(_routed(health=off), [_chatgpt()], store, now=100.0,
                         notifier=spy)
    assert spy.sent == []


async def test_a_subscription_expiring_soon_is_alerted_once_a_day():
    store, spy = _store(), _Spy()
    soon = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                         time.gmtime(time.time() + 3 * 86400))
    doc = {**_HEALTH, "auth": {**_HEALTH["auth"], "expires_at": soon}}
    await sync_catalogue(_routed(health=doc), [_chatgpt()], store,
                         now=time.time(), notifier=spy)
    await sync_catalogue(_routed(health=doc), [_chatgpt()], store,
                         now=time.time() + 60, notifier=spy)
    assert len([s for s in spy.sent if "expires" in s]) == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_probing.py -v -k alert`
Expected: FAIL — `assert [] == [...]`, nothing is sent.

- [ ] **Step 3: Write the implementation**

Replace the `_announce_changes` stub in `src/llm_libre/probing.py`:

Add `import calendar` to `probing.py`'s imports, then:

```python
# How close to `expires_at` the operator is told. A week is enough to renew a
# subscription deliberately, and short enough that the warning still means
# something when it arrives.
EXPIRY_WARNING_S = 7 * 86400


def _announce_changes(store, provider: str, contract, notifier) -> dict:
    """Alert on the two transitions worth waking someone for.

    Returns the bookkeeping `sync_catalogue` merges into the document it is
    about to store. Nothing is written here: one writer per document is what
    keeps "the alert fired" and "the contract we saw" from racing each other
    within a single sweep.

    A capability turning OFF is the event this whole design exists for: a plan
    lapsed, a token was revoked, an account was downgraded. Left undetected it
    shows up as a route that fails every request until somebody notices.

    A capability turning ON is logged, not alerted. It is good news, and good
    news does not need a notification.

    The comparison is against the PREVIOUS stored document, so it detects a
    transition rather than a state -- without that, a provider sitting at
    `images: false` would alert on every sweep, twice a day, until it was muted
    and stopped being read at all. The first sweep for a provider alerts on
    nothing: having nothing to compare against is not the same as something
    having turned off.
    """
    if notifier is None:
        return {}
    previous = store.get_contract(provider)
    if previous is not None:
        was = previous.get("capabilities") or {}
        lost = sorted(name for name, value in contract.capabilities.items()
                      if was.get(name) is True and value is False)
        gained = sorted(name for name, value in contract.capabilities.items()
                        if was.get(name) is False and value is True)
        if gained:
            log.info("capabilities of %s: gained %s", provider, ", ".join(gained))
        if lost:
            plan = f" on plan {contract.auth.plan}" if contract.auth.plan else ""
            notifier.notify(
                f"⚠️ {provider}: lost {', '.join(lost)}. The account is "
                f"{contract.auth.mode}{plan}. Routes that needed it have left "
                f"the catalogue.")
    return _expiry_warning(previous, provider, contract, notifier)


def _expiry_warning(previous, provider: str, contract, notifier) -> dict:
    """One warning per provider per day while the subscription is nearly up.

    Per day, not per sweep: the sweep interval is configurable and a burst of
    redeploys runs several of them within minutes, so "once per sweep" is not a
    rate anybody chose. The timestamp rides inside the stored contract document
    rather than in a table of its own -- it is one float per provider whose only
    reader is this function, and `contract.parse_health` ignores unknown
    top-level keys, so it never reaches the contract itself.
    """
    expires_at = contract.auth.expires_at
    if not expires_at:
        return {}
    try:
        # timegm, not mktime: `expires_at` is UTC by the contract, and mktime
        # would read it as local time -- five hours of silent drift here.
        deadline = calendar.timegm(time.strptime(expires_at[:19], "%Y-%m-%dT%H:%M:%S"))
    except ValueError:
        log.warning("capabilities of %s: auth.expires_at=%r is not ISO 8601; "
                    "no expiry warning will be sent.", provider, expires_at)
        return {}
    remaining = deadline - time.time()
    if not 0 < remaining <= EXPIRY_WARNING_S:
        return {}
    now = time.time()
    warned_at = (previous or {}).get("_expiry_warned_at")
    if warned_at is not None and (now - warned_at) < 86400:
        # Carry the old timestamp forward: dropping it would restart the
        # once-a-day clock on the very next sweep.
        return {"_expiry_warned_at": warned_at}
    notifier.notify(
        f"⏳ {provider}: the {contract.auth.plan or 'current'} subscription "
        f"expires in {int(remaining // 86400)} day(s), on {expires_at}. When it "
        f"does, its plan-gated capabilities turn themselves off.")
    return {"_expiry_warned_at": now}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, whole suite green.

- [ ] **Step 5: Commit**

```bash
cd /Users/cristian/llm-libre
git add src/llm_libre/probing.py tests/test_probing.py
git commit -m "feat(alerts): say on Telegram when a provider loses a capability or is about to"
```

---

## Deployment

After Task 5, deploy `chatgpt-proxy` to `blog:8890` (it is currently 2 commits
behind: it lacks `/v1/projects` and the vision commit `774d019`). After Task 12,
redeploy `llm-libre` to `blog:8102`. **`providers.yaml` is read once, at startup
(`main.build_state`)** — the `reads_capabilities: true` added in Task 7 needs a
restart, not a sweep.

Verify against the live deployment:

```bash
ssh blog 'curl -s http://127.0.0.1:8890/health' | python3 -m json.tool
ssh blog 'curl -s http://127.0.0.1:8102/health' | python3 -m json.tool
```

Expected: the first carries `contract: 1` and eleven booleans; the second
carries `providers.chatgpt` with `plan: "go"` and `capabilities.images: true`.

## Deviations from the spec

Two, both deliberate, recorded here so the spec can be amended rather than
quietly diverged from.

**No `contract/health.schema.json`.** §6 called for a JSON Schema authored here
and vendored into each proxy's tests. `contract.parse_health` (Task 1) is the
executable contract instead, and `chatgpt-proxy`'s `tests/test_health_contract.py`
asserts conformance from the other side. A schema file would need a `jsonschema`
dependency in six repos to validate anything, and a schema that nothing runs is
a second source of truth that drifts from the first. The eleven required keys are
duplicated once, in `capabilities.REQUIRED_CAPABILITIES`, and the cross-repo test
is what keeps the two honest.

**`fixed_models` is masked by the contract.** The spec's precedence (§4.2) names
`exceptions` and the discovery sources but is silent on `fixed_models` — and
`chatgpt` declares `dall-e-3` there with `images: true`. Task 9 resolves it: a
fixed route the contract contradicts is dropped for that sweep. Without this the
one route the whole design exists to retire when the Go plan lapses would be the
one route that survived it.

## What this plan does not do

Named so the boundary is explicit, per §8 of the spec: exposing grok's TTS/STT
and document attachments, perplexity's Soniox STT, deepseek's file upload and
history; the four remaining proxies adopting the contract (the same shape as
Tasks 2-5, once each, and worth its own plan); the gateway endpoints that would
route by the new axes (`POST /v1/audio/speech`, `POST /v1/audio/transcriptions`,
`POST /v1/translate`, `web_search` passthrough); and `audio/mpeg`/`audio/wav` in
`assets._SAFE_TYPES`, without which a generated mp3 is served as
`application/octet-stream`.
