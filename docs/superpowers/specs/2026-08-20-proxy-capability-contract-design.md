# The proxy capability contract

**Date:** 2026-08-20
**Status:** design, awaiting review
**Repos touched:** `llm-libre`, `chatgpt-proxy`, `perplexity-proxy`, `deepseek`, `grok-proxy`, `mistral-proxy`

A single, machine-readable way for an in-house proxy to state what it can
actually do right now, and for the gateway to read it instead of guessing.

---

## 1. The problem, stated from evidence

The gateway declares each provider's capabilities **by hand**, in
`providers.yaml`, at a moment in time. Five separate observations on
2026-08-20 show what that costs.

**A declared capability can be wrong from the day it is written.**
`chatgpt` declares `context: 128000` for every discovered id. Its
`/v1/models` now publishes the real number: `52815` for most models and
`262144` for `gpt-5-4-t-mini` and `gpt-5-6-t-mini`. The declaration
overstates the common case by 2.4x and understates the two large ones by
5x. `x_min_context` filters on this value, so both errors route real
requests wrong.

**A capability can be conditional on something the gateway cannot see.**
`chatgpt` declares `dall-e-3` and `gpt-image-1` with `images: true`. That
is true only while the account behind `chatgpt-proxy` holds a paid plan.
Measured today: `plan_type: "go"`, `chatgptgoplan` active, **expiring
2026-09-06**. On a free plan the backend accepts the image request,
invokes the tool, and returns nothing — the proxy answers
`503 "No image was generated"`. To the gateway that is an ordinary failed
attempt: `chatgpt` has `priority: 0`, so every image request would try it
first, fail, accumulate suspicion, and fail over — for a capability that
no longer exists.

**A provider's own capability report already drifts.**
`chatgpt-proxy`'s `/health` reports `image_input: false` and
`voice: false`. Both are stale as of commit `774d019`: it has image input,
and it has both `/v1/audio/speech` and `/v1/audio/transcriptions`. The
block is hand-written English prose mixed with booleans, so nothing can
check it.

**Capabilities exist that nobody knows about.** The static audit found
`grok_api.Chat/TextToSpeech`, `grok_api.Voice/Transcribe` and
`grok_api.Voice/SpeechToText` in grok's backend, unexposed; Soniox STT in
perplexity's, unexposed and self-documented as pending; and
`/api/v0/file/upload_file` plus a RAG index in deepseek's, unexposed.

**And the reverse: a capability the gateway has and does not use.**
`grok` supports web search through `disable_search`, which the proxy
defaults to `True` — search off. The gateway never sends the field, so
every grok answer today is un-grounded by default.

The common shape: **the gateway trusts a hand-written snapshot, and
nothing detects when it stops matching reality.** This is the same
failure the project already solved for model *ids* — discovered from
`/models`, never hardcoded — applied one level up, to capabilities.

## 2. Scope

**In scope.** A contract two endpoints of every in-house proxy must
satisfy (`GET /health`, `GET /v1/models`), the rule for resolving what a
capability *effectively* is, how `llm-libre` consumes it, and how an
operator is told when a capability disappears.

**Out of scope, deliberately.** Exposing capabilities that are not
exposed yet (grok TTS/STT, perplexity STT, deepseek files), and aligning
the URLs themselves onto one shape. Both are real and both are next —
see §8. They are excluded here because they are large, they are
per-repo, and they are *verifiable* only once this contract exists: with
it, the gateway can report which proxy still does not comply, instead of
someone re-auditing five repos by hand.

**Not a goal:** changing how routing, scoring, cooldown or failover work.
This spec changes where capability values come from, nothing downstream
of that.

## 3. The contract

### 3.1 `GET /health`

Unauthenticated, and **served from cache**: it must answer in under a
second without making a request to the upstream vendor. Resolving
`auth` does need upstream data (`chatgpt-proxy` reads it from
ChatGPT's own `/account` and `/limits`), so the proxy refreshes that
state on its own schedule — at startup, and no more often than hourly —
and `/health` reports the last known value. A health endpoint that
called the vendor on every hit would make the gateway's sweep depend on
the vendor being up, which is the opposite of what the sweep is for.

```json
{
  "status": "ok",
  "provider": "chatgpt",
  "version": "2.5.0",
  "contract": 1,
  "auth": {
    "mode": "account",
    "plan": "go",
    "subscription_active": true,
    "expires_at": "2026-09-06T00:28:46Z"
  },
  "capabilities": {
    "chat": true,
    "streaming": true,
    "tools": false,
    "vision": true,
    "images": true,
    "audio_speech": true,
    "audio_transcription": true,
    "translate": true,
    "search": true,
    "files": true,
    "conversations": true
  }
}
```

`contract` is an integer version for this document's schema. A proxy that
does not carry it is treated as non-compliant (§5.3), not as broken.

`auth.mode` is one of `anonymous`, `account`, or `unknown` (the proxy
could not resolve it this cycle). `plan` is a free-form vendor string or
`null`; `expires_at` is ISO 8601 in UTC, or `null` when there is no
subscription to expire.

`auth` is **informational**, for operators and alerts. The gateway must
never branch on `plan`: plan names are provider-specific
(`go`, `free`, `plus`, `pro`, …) and encoding them in the gateway
recreates exactly the coupling this contract removes. Everything the
gateway acts on is in `capabilities`.

`capabilities` values are booleans. Every key is required; a proxy that
cannot do something says `false`, never omits the key. Unknown extra keys
are ignored by the gateway, which is how a new capability gets introduced
before the gateway learns to use it.

### 3.2 The effective-capability rule

**`capabilities` reports what the proxy can do *right now*, already
resolved against the account, the plan and the quota.** Not what the
software implements.

This is the single load-bearing rule of the contract. `chatgpt-proxy`
must report `images: false` when the account is anonymous or on a free
plan — even though `/v1/images/generations` exists and would return a
`503` — because to the gateway "the endpoint exists" and "a request to it
will produce an image" are the same question, and only the second one
matters. When the Go subscription lapses on 2026-09-06, the boolean flips
on its own and the gateway stops routing image requests there. Nobody
edits YAML.

The rule is what lets the gateway stay ignorant of ChatGPT plans, Grok
rate tiers and Mistral Pro billing: each proxy resolves its own vendor's
rules and publishes one boolean.

**Where the rule stops: a boolean tracks entitlement, not the meter.**
`images: false` means *this account may not generate images* — anonymous,
free plan, expired subscription, revoked token. It does **not** mean
"today's 106 image generations are spent". Transient exhaustion is
already handled, better, by machinery the gateway owns: a `429` sends the
route to cooldown with the vendor's own `Retry-After`, and it comes back
by itself. Flipping the boolean for that would make it flap between
sweeps and would replace a mechanism that recovers automatically with one
that waits up to twelve hours for the next sweep. The dividing line is
durability: **if a fresh request tomorrow would still be refused for the
same reason, it belongs in the boolean.**

### 3.3 `GET /v1/models`

Per-model metadata, for the capabilities that genuinely vary by model.
The provider-level block cannot express that grok's `imagine-agent-mode*`
generate images but do not do tools or vision, while its other 28 routes
are the opposite.

```json
{
  "object": "list",
  "data": [
    {
      "id": "gpt-5-6",
      "object": "model",
      "context_window": 52815,
      "max_output_tokens": 8192,
      "capabilities": {"tools": false, "vision": true, "images": false}
    }
  ]
}
```

`context_window` and `max_output_tokens` are required when the proxy can
know them; omitted otherwise, and the gateway falls back (§5.2).
`capabilities` here is a **subset** of the provider block — only
`tools`, `vision`, `images` vary per model — and is optional: omitted
means "same as the provider-level value".

A per-model value may only be **narrower** than the provider-level one.
`vision: true` on a model whose provider reports `vision: false` is a
contradiction; the gateway logs it and takes the provider-level value.

### 3.4 Standard endpoint surface

The contract states *what a capability boolean promises*. Each boolean
maps to exactly one endpoint and one wire shape, so `true` is a testable
claim rather than a description.

| Capability | Endpoint | Contract |
|---|---|---|
| `chat` | `POST /v1/chat/completions` | OpenAI chat completions |
| `streaming` | same, `stream: true` | SSE, `data: [DONE]` terminated |
| `tools` | same, `tools` in body | returns real `tool_calls`, never prose |
| `vision` | same, `image_url` content parts | data: URIs and https URLs both accepted |
| `search` | same, `web_search: bool` in body | `true` grounds the answer; **default on** |
| `images` | `POST /v1/images/generations` | OpenAI images |
| `audio_speech` | `POST /v1/audio/speech` | **raw audio bytes** + correct `Content-Type` |
| `audio_transcription` | `POST /v1/audio/transcriptions` | multipart `file` → `{"text": ...}` |
| `translate` | `POST /v1/translate` | `{text, target, source?}` → `{text, target, source}` |
| `files` | `POST·GET /v1/files`, `GET·DELETE /v1/files/{id}` | OpenAI files |
| `conversations` | `GET /v1/conversations`, `GET /v1/conversations/{id}` | listing + detail |

An endpoint whose capability is `false` answers **`501 Not Implemented`**,
not `404`. `404` is indistinguishable from a routing mistake; `501` says
"this proxy, deliberately, does not do this".

Anything a provider offers beyond this lives under its own prefix —
`/grok/*`, `/perplexity/*`, `/mistral/*` — which is already the pattern in
grok-proxy and perplexity-proxy. The standard surface is a floor, not a
ceiling.

Three known normalisations this forces, all inside the proxies:

- **`chatgpt`**: `/v1/audio/speech` returns JSON carrying an mp3 URL. It
  must return the bytes. Its current JSON (with `text` and `exact_match`,
  a real signal since the flow makes the model echo the input) moves to
  response headers or to a native `/chatgpt/audio/speech`.
- **`grok`**: gains `web_search` as the inverted alias of
  `disable_search`, and the default flips to search **on**, matching every
  other provider.
- **`mistral`**: keeps `/v1/vision` as a native extra; the standard path
  for vision becomes `image_url` parts in chat, which its `chat.py`
  already handles.

## 4. What the gateway does with it

### 4.1 Reading

`probing.sync_catalogue` already runs once per health sweep and already
walks the provider list. It gains one `GET /health` per provider, before
the existing `models_path` request, for any provider declaring the new
`reads_capabilities: true` in `providers.yaml`.

The result is applied in `catalog.normalize`, which is where
`default_capabilities` and `exceptions` already meet. No new module.

### 4.2 Precedence

From strongest to weakest:

1. **`exceptions` in `providers.yaml`** — the hand-written override, and
   it must stay strongest. This is where a *measured lie* is recorded:
   `chatgpt` declares `tools: false` because it was measured 0/3 twice
   while the proxy reports the feature as available. A discovery contract
   that could overrule that would re-open the exact failure the
   declaration exists to prevent.
2. **`/v1/models` per-model values.**
3. **`/health.capabilities` provider-level values.**
4. **`default_capabilities` in `providers.yaml`** — now a *fallback*
   for a proxy that has not adopted the contract, not the primary source.

### 4.3 New capability axes

`models.Capabilities` gains `audio_speech`, `audio_transcription`,
`translate` and `search`, all `bool = False` — the same shape `images`
took: default false, so every existing construction site keeps working and
a provider must claim a capability to get it. `context` and `max_output`
start being populated from `/v1/models` rather than declared.

These axes are recorded and exposed in this phase; the gateway endpoints
that route by them are §8.

### 4.4 Surfacing

- `GET /health` on the gateway reports, per provider: contract version or
  `null`, `auth.mode`, `auth.plan`, `expires_at`, and the effective
  capability set. An operator sees "chatgpt: account/go, images on, expires
  2026-09-06" without opening a shell.
- `GET /v1/ranking` gains the capability columns it does not have.

### 4.5 Alerting

`notify.Notifier` already exists and already carries rate-limit events to
Telegram. Two new events:

- **A capability turned off.** `images` was `true` last sweep and is
  `false` now — the plan lapsed, the quota ran out, the token expired.
  This is the event the whole design is for.
- **A subscription is within 7 days of `expires_at`.** Sent once per
  provider per day, so a lapse is a decision rather than a surprise.

A capability turning **on** is logged, not alerted: it is good news and it
does not need waking anyone.

## 5. Failure and degradation

The contract must never make things worse than the hand-declared status
quo. Three paths, matching how `sync_catalogue` already handles a failing
`/models`:

**5.1 `/health` unreachable or malformed.** Keep the previous sweep's
capability values, log a warning naming the provider and the reason.
Identical to how a failed `/models` keeps the previous catalogue: freezing
what is known beats erasing it.

**5.2 A field is missing.** Fall back one level down the precedence
chain. A proxy with no `context_window` per model keeps using
`default_capabilities.context`.

**5.3 A proxy has not adopted the contract.** No `contract` key means
today's behaviour, exactly: `default_capabilities` and `exceptions` decide
everything. This is what makes the rollout incremental — one proxy at a
time, in any order, with no flag day.

**5.4 A proxy reports a capability it does not have.** The failure is a
routed request that fails, then normal failover and suspicion — the
behaviour that exists today for every wrong declaration. The contract does
not add a new failure mode here; it adds a place to correct it
(`exceptions`) that is already the strongest voice.

## 6. Testing

**Contract-level.** One JSON Schema, `contract/health.schema.json`,
authored in `llm-libre` and vendored into each proxy's test suite. Each
proxy asserts its own `/health` validates. The gateway asserts it can
parse and apply a compliant document, and that it degrades correctly on
each of the four failure paths in §5.

**Gateway-level**, extending the existing suites:

- `test_catalog.py` — precedence: `exceptions` beats per-model beats
  provider-level beats `default_capabilities`; a per-model value wider than
  the provider-level one is rejected and logged.
- `test_probing.py` — a failing `/health` keeps the previous values; a
  provider without `reads_capabilities` is untouched.
- `test_wire_contract.py` — the new keys in the gateway's own `/health`
  and `/v1/ranking`.
- `test_notify.py` — the capability-off and expiry-warning events fire
  once, on transition, not every sweep.

**No live-quota tests.** Every case above runs against fixtures. What
must be checked against the real deployments is the one thing fixtures
cannot check — that each proxy's reported booleans match what its backend
actually does — and that is `smoke_test.py`'s job, which already exists in
`chatgpt-proxy` and gets ported to the other four.

## 7. Rollout

Per proxy, in this order, each independently deployable:

1. **`chatgpt-proxy`** — first, because it already has `/v1/account`,
   `/v1/limits` and `CAPABILITIES.md`, so it has the most of the contract
   already and the least to invent. It is also the one whose plan expires
   on 2026-09-06, which is the deadline that makes this worth doing now.
2. **`llm-libre`** — read the contract, with `chatgpt` the only provider
   declaring `reads_capabilities`. At this point the gateway is strictly
   better off and the other four are untouched.
3. **`mistral-proxy`**, **`grok-proxy`**, **`perplexity-proxy`**,
   **`deepseek`** — in any order.

Each proxy gets a `CAPABILITIES.md` recording what was measured and how,
following the one `chatgpt-proxy` already has.

## 8. What comes after

Named here so the boundary is explicit, not because it is committed:

- **Expose what exists but is not exposed.** grok TTS (`Chat/TextToSpeech`)
  and STT (`Voice/Transcribe`, `Voice/SpeechToText`) — the protobuf field
  tags are already recovered from the decompiled APK, so this is wiring on
  top of `grok_backend.py`'s existing hand-built frames. perplexity STT
  over the Soniox WebSocket. deepseek file upload and history.
- **Gateway endpoints for the new axes**: `POST /v1/audio/speech`,
  `POST /v1/audio/transcriptions`, `POST /v1/translate`, and `web_search`
  passthrough routed by capability. `assets.py` needs `audio/mpeg` and
  `audio/wav` in `_SAFE_TYPES` before any of it is useful — today a
  generated mp3 would be stored and served as
  `application/octet-stream`.
- **Retiring `default_capabilities`** for any provider that has adopted
  the contract, once its numbers have been confirmed against the live
  deployment.
