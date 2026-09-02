# Android chat app for llm-libre — design

Date: 2026-09-01
Status: approved, pending implementation plan

## What this is

A Flutter Android app that talks to the llm-libre gateway at
`https://llm.comparadorinternet.co`. Personal use, one user, one API key. It is
a chat client shaped like the ChatGPT app, but its job is to make the gateway's
own advantage visible and usable: 48 live routes across six providers, each with
a different set of capabilities, and a router that already knows how to pick
between them.

The app lives in its own repository at `~/llm-libre-chat`, following the
one-folder-per-app convention of the other Flutter projects. This design
document lives in the llm-libre repo because part of the work is gateway-side
(see "Changes to llm-libre").

## Decisions taken

| Question | Answer |
|---|---|
| Audience | One user. No accounts, no per-user quotas, no login. |
| Platform | Android, Flutter (matches seven existing projects). |
| Scope for v1 | Chat + vision + image generation + audio (speech and dictation) + a routes/usage panel. |
| History | On-device only. No backend, nothing to deploy. |
| Model selection | Intent-first: the app sends `auto` plus derived `x_requires`. A rich model picker exists as an override. |
| UI reference | The ChatGPT app. |

## What the gateway actually offers

Measured against the live deployment on 2026-09-01, not read off the docs.

Endpoints: `POST /v1/chat/completions`, `POST /v1/images/generations`,
`POST /v1/audio/speech`, `POST /v1/audio/transcriptions`, `POST /v1/translate`,
`GET /v1/assets/{id}`, `GET /v1/models`, `GET /v1/ranking`, `GET /v1/traffic`,
`GET /v1/usage`, `GET /health`.

Capability coverage across the 48 live routes:

| Capability | Routes | Providers |
|---|---|---|
| `tools` | 42 | grok 14, kilo 16, chatgpt 8, deepseek 2, mistral 1, minimax 1 |
| `audio_transcription` | 31 | chatgpt 10, grok 17, deepseek 2, perplexity 1, mistral 1 |
| `vision` | 30 | grok 14, chatgpt 8, kilo 6, mistral 1, minimax 1 |
| `search` | 29 | grok 17, chatgpt 10, perplexity 1, mistral 1 |
| `audio_speech` | 29 | grok 17, chatgpt 10, perplexity 1, mistral 1 |
| `translate` | 10 | chatgpt 10 |
| `images` | 6 | grok 3, chatgpt 2, mistral 1 |

47 routes free, 1 paid. Context windows from 32k to 1,048k. Every capability in
the v1 scope has live routes behind it.

### Three findings that shaped the design

**`/v1/models` does not carry capabilities.** It returns `{id, object,
owned_by}` and nothing else. Everything the app needs to say what a model can do
— `tools`, `vision`, `images`, `search`, `audio_speech`,
`audio_transcription`, `translate`, `context`, `tier`, `quality`,
`latency_p50_ms`, `cooldown_until`, and the whole `rate_*` family — lives in
`GET /v1/ranking`, which requires the API key.

**Streaming carries the model but not the provider.** The README documents that
`X-Route-Used`, `X-Tier` and `X-Attempts` do not travel when `stream: true`,
because headers are sent before the failover chain resolves. But the SSE chunks
themselves do carry `"model": "turbo"`. So attribution while streaming is
recoverable by crossing that id against the cached catalogue — except when the
same model id exists at two providers, where it stays ambiguous. This is what
motivates gateway change 2 below.

**Provider metadata leaks into the stream.** Perplexity's final chunk carries a
`"_pplx": {"display_model": ...}` object inside an OpenAI-shaped payload. The
client must ignore unknown fields regardless; the leak is cosmetic, and gateway
change 2 cleans it.

## Architecture

```
lib/
  api/                  Pure Dart. No Flutter imports.
    llm_client.dart       chat (streaming and not), images, audio, translate
    sse.dart              Server-Sent Events parser
    catalog.dart          /v1/ranking -> per-route capabilities, cached
    types.dart            Route, Capability, ChatDelta, LlmError
  data/
    db.dart               drift: conversations, messages, attachments
  features/
    chat/                 the main screen
    catalog/              routes panel: override picker and telemetry
    settings/             key, base URL, preferences
  app.dart                go_router
```

Stack: `flutter_riverpod`, `go_router`, `drift` + `drift_flutter`,
`flutter_secure_storage`, `http`, `image_picker`, `file_picker`, a markdown
renderer (`gpt_markdown`, which handles streaming partial markdown better than
`flutter_markdown`), `record` for microphone capture and `just_audio` for
playback. This matches `~/testenglish`, the most recent Android project, rather
than introducing a new set of habits.

Note that `flutter_tts` — present in `testenglish` — is **not** used here.
Speech comes from the gateway's `/v1/audio/speech` over 29 real routes, not from
the device's own synthesiser, which is the whole point of routing it through
llm-libre.

**`api/` imports no Flutter on purpose.** All the fragile logic lives there —
parsing SSE split at arbitrary byte boundaries, deriving `x_requires`,
distinguishing a 404-with-suggestions from a 503, separating reasoning from
content. With no Flutter dependency those tests run under `dart test` in a
second, with no emulator. It is the code most likely to break and the cheapest
place to prove it works.

**No repository layer.** `api/` is already the boundary. A second indirection on
top of it, in a single-user app, buys nothing and costs a file to read.

### The API key

Stored in `flutter_secure_storage` (Android Keystore), entered on first launch.
Never compiled into the binary: an APK is trivially decompiled, and a key held
in settings can be rotated without a rebuild. The base URL is configurable too,
defaulting to `https://llm.comparadorinternet.co`.

## The client (`api/`)

### Streaming

`http.Client().send()` returns a `StreamedResponse`. The byte stream passes
through `utf8.decoder` and then a buffer that splits on `\n\n`. Both must be
stateful transformers, and that is not a detail: chunks arrive split at
arbitrary byte offsets, including the middle of a multi-byte character. A naive
`utf8.decode` per chunk corrupts any accented character or emoji that lands on a
boundary. This is the first test written.

The parser terminates on `data: [DONE]` and ignores any field it does not
recognise, `_pplx` included.

### Deriving intent

The core of the chosen approach: the app does not ask which model to use, it
describes what is needed and lets the router choose.

| User action | Request |
|---|---|
| Plain text | `model: "auto"` (or the active override) |
| Attaches an image | `x_requires: ["vision"]` |
| Enables web search | `x_requires: ["search"]` |
| Pastes a long document | `x_min_context: <estimate>` |
| Picks fast / strong profile | `auto:fast` / `auto:strong` |
| Taps "generate image" | `POST /v1/images/generations` (separate endpoint) |
| Dictates | `POST /v1/audio/transcriptions`, then a normal chat turn |
| Taps "read aloud" | `POST /v1/audio/speech` over the received answer |

Image generation cannot be inferred from prose, and the design does not pretend
otherwise: it needs an explicit affordance in the composer.

### Reasoning

Without streaming the gateway returns trimmed reasoning in a top-level
`x_reasoning` field. With streaming it does not travel at all, and the only
route to it is `x_raw: true`, which returns the provider's `<think>` block
inside `content`. Therefore: the "show reasoning" toggle sends `x_raw: true` and
the app parses the tags itself into a collapsible block. One code path, so
behaviour does not change between streaming and not.

### Error handling

Each failure gets its own treatment, because each means something different:

- **404 with `suggestions`** — the requested model is not in the catalogue.
  `{"detail": {"message": "the model 'X' no longer exists", "suggestions": [...]}}`.
  Render the suggestions as tappable chips that retry. **Verified 2026-09-01 that
  this reaches streaming requests too** (`stream: true` returned 404 with three
  suggestions), because the check runs before the SSE headers go out. Only the
  rarer *live* 404 — still in the catalogue, but the provider 404s mid-attempt —
  cannot be surfaced on a stream.
- **400 `no route satisfies the request`** — a different failure from 503, and
  the app must not merge them. It means no route can EVER serve this: the
  requirements are unsatisfiable, not the routes unavailable. The body carries
  `request` (the parsed requirements) and `active_routes`. Shown as "no model can
  do this", naming the offending requirement.
- **503** — routes exist that could serve, but all are down or in cooldown. The
  body carries `compatible_routes`, `paid_cap_reached`, and `next_release`, a
  timestamp for when the first punished route comes back. That last field is a
  concrete countdown to show, not a generic failure message.
- **429** — the gateway's per-minute cap. Back off and retry; not a red error.
  Its `detail` is a bare string (`"too many requests for this key"`), unlike
  400/404/503 whose `detail` is an object. So is 401's (`"invalid api key"`). The
  client must handle both shapes.
- **Network loss mid-stream** — persist the partial answer marked incomplete,
  offer "continue".
- **Stop button** — cancels the stream and keeps the partial text, as ChatGPT
  does.

### Catalogue

`/v1/ranking` is read at launch and refreshed at most every 10 minutes, cached
in the database. It feeds the capability chips, context sizes, tier badges and
cooldown states. **If it cannot be read, the app still works**: `auto` needs no
local knowledge. The catalogue powers the override panel; it never gates the
chat.

The 10-minute floor matters on the gateway side, not just the client's: probe
volume there is already dominated by restarts, and a chat app polling a
telemetry endpoint every few seconds would be a second avoidable source of load
for no benefit — cooldowns are measured in minutes, not seconds.

## Local data (drift)

```
conversations   id, title, created_at, updated_at, pinned,
                model_override, system_prompt
messages        id, conversation_id, role, content, reasoning,
                model_used, provider_used, tier, attempts,
                status (ok|streaming|partial|error), error_kind, created_at
attachments     id, message_id, kind (image_in|image_out|audio_in|audio_out),
                local_path, remote_url, mime, bytes
routes_cache    key, capabilities json, context, tier, quality,
                cooldown_until, fetched_at
```

Indexed on `(conversation_id, created_at)`.

**Assets are downloaded to disk.** `GET /v1/assets/{id}` needs no API key, so a
generated image could be rendered straight from the URL — but if the gateway
container is recycled the image vanishes from the history. The app downloads and
stores it locally, keeping the remote URL alongside.

**Automatic titles**, as ChatGPT does: after the first exchange, one `auto:fast`
call with a small `max_tokens` asks for a three-or-four-word title. Cheap, and it
makes the drawer navigable.

## UI

Modelled on the ChatGPT app.

**Chat screen.** User messages in a grey bubble, right-aligned; answers
full-width with no bubble. Markdown with copy-buttons and highlighting on code
blocks. A blinking cursor while streaming, with the send button becoming
**stop**. Under each answer: copy, regenerate, and a quiet chip naming who
answered (`perplexity · turbo · free`). Long-press on your own message to edit
and resend, truncating the conversation from that point.

**Drawer.** Conversations grouped into Today / Yesterday / Last 7 days / older,
with a search field, "new conversation", and long-press to rename, pin or
delete.

**Composer.** Grows to about six lines. A `+` button for photo, camera and file.
A microphone for dictation. Two visible toggles: web search, and reasoning.

**Model picker.** A bottom sheet opened from the title, where ChatGPT puts it.
The five `auto*` aliases on top with one line each explaining what they select;
below them the models grouped by provider, each with capability chips
(`tools`, `vision`, `images`, `search`), its context size, and — greyed out with
a countdown — the ones currently in cooldown. A search field, because there are
48.

**Routes panel.** Its own screen: `/v1/ranking` as a sortable table,
`/v1/traffic` showing how much failover happened and what `auto` actually chose,
`/v1/usage` with paid spend against the cap of 200, and `/health`.

**Settings.** Key, base URL, default model, system prompt, light/dark theme,
export history to JSON.

**Deliberately excluded:** `translate` gets no screen. It has chatgpt-only
routes and does what asking a chat to translate already does. It appears instead
as a message action ("translate this answer") backed by the endpoint.

## Changes to llm-libre

Authorised by the user as part of this project. Ordered by value against cost.

**1. `x_requires` reached only two of its capabilities — FIXED, commit
`08cbc46`.** Found while writing the implementation plan, and it invalidated the
chosen approach until fixed. `parse_request` consumed the parsed `x_requires` set
for `tools` and `vision` and dropped every other name silently; `search` had no
`RouteRequest` flag and no `_satisfies` branch at all, despite being declared on
`Capabilities` as "web_search on chat completions" and published on
`/v1/ranking` for 29 routes. Measured live before the fix: `["translate"]` was
served by `grok/grok-plugins-4p6-powerpoint` (translate=false), and
`["images","translate","search"]` by `kilo/nvidia/nemotron-3.5-lightning:free`,
false on all three. Without this, "enable web search" in the composer would have
sent a requirement the gateway ignored.

The fix makes `search` requirable from the body and returns 400 for any other
name; the four endpoint capabilities stay URL-decided. 936 tests pass.
**Deploying it is a separate decision** — the new 400 is a behaviour change for
any existing caller sending a name outside `{tools, vision, search}`.

**2. `x_route` in the final streamed chunk.** The stream carries
`"model": "turbo"` but not the provider, leaving attribution ambiguous when a
model id exists at two providers. Adding a gateway-owned field to the final chunk
is the same pattern the codebase already uses for `x_*`: SDKs ignore what they do
not know. Small, non-breaking, and it is what makes the per-answer attribution
chip honest.

**3. Strip the `_pplx` leak.** Perplexity's own metadata inside an OpenAI-shaped
payload. The client ignores unknown fields anyway, so this is hygiene rather than
need.

**Explicitly not doing:** adding capability fields to `/v1/models`. The app
needs `/v1/ranking` for the panel regardless, so this would duplicate the truth
across two endpoints for no gain.

## Testing

`dart test` over `api/`, no emulator required:

- SSE split mid multi-byte character, `data: [DONE]`, unknown fields such as
  `_pplx`.
- `x_requires` derivation for each user action in the table above.
- Error mapping: 404-with-suggestions, 503, 429, mid-stream disconnect,
  cancellation.
- Catalogue parsing against a captured `/v1/ranking` payload.

Widget tests for markdown rendering and the streaming / partial / error states.
One integration test against the live gateway, tagged so it does not run by
default.

## Build order

This is a large v1 for a single plan, so it is built in vertical slices, each one
usable on its own rather than a horizontal layer that does nothing until the next
one lands:

1. **Talking at all** — settings screen with the key, `api/` with non-streaming
   chat, one hardcoded conversation on screen. Proves the transport and the auth.
2. **A real chat** — SSE streaming, stop button, markdown, drift persistence,
   the drawer, automatic titles.
3. **Knowing who answered** — catalogue from `/v1/ranking`, the attribution chip,
   the model picker bottom sheet with capability chips. Gateway change 2 lands
   here, because that is where its absence starts to show.
4. **Images** — vision attachments in, `/v1/images/generations` out, asset
   download and local storage.
5. **Audio** — dictation into the composer, read-aloud on answers.
6. **The panel** — `/v1/ranking`, `/v1/traffic`, `/v1/usage`, `/health`.

Slices 1 and 2 are the app. Everything after them is addition, and each can be
cut or reordered without stranding the ones before it.

## Out of scope for v1

Sync across devices, multi-user accounts, iOS, tool calling from the app itself
(the `tools` capability is about what routes support, not about the app defining
functions), and any backend.
