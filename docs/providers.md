# The provider model

How a provider is declared, and what each field in `providers.yaml`
buys you. The goal of this design is that **adding a provider is a config
change, not a code change** -- every field here exists to make that true
for one specific real-world constraint a provider might have.

## The three registration patterns

Every provider in `providers.yaml` fits one of three shapes, depending on
what its own `/models` endpoint can tell the gateway:

| Pattern | Model ids come from | Capabilities come from | Who uses it |
|---|---|---|---|
| **Fully discovered** | its `/models` | its `/models` | Kilo (OpenRouter too, as an example -- see the note below) |
| **Fully declared** | `modelos_fijos` in the YAML | `modelos_fijos` in the YAML | MiniMax |
| **Discovered ids, declared capabilities** | its `/models` | `capacidades_por_defecto` in the YAML | `chatgpt` |

> **A note on OpenRouter, mentioned throughout this doc as an example of
> the "fully discovered" pattern:** it is not currently in the live
> `providers.yaml` (removed 2026-08-17 -- `OPENROUTER_API_KEY` was never
> configured in production, so all of its routes 401ed on every request
> and just sat in cooldown, eating almost half the catalog's probe budget
> and ranking-table space to repeatedly prove they were dead). It stays
> useful here purely as a worked example: unlike Kilo, its free tier
> requires an API key for `/chat/completions` even though `/models` is
> public without one, which is a good illustration of `clave_env` being
> *declared* on a provider without that provider's tier being optional in
> practice. Re-adding it is a YAML entry plus `OPENROUTER_API_KEY`, not a
> code change -- see "Adding a new provider" below.
>
> **Removing OpenRouter's YAML entry did not, by itself, deactivate its
> already-discovered routes.** `sincronizar_catalogo` only deactivates
> stale routes for the provider it is currently syncing (scoped by
> `Almacen.upsert_rutas(..., proveedor=p.id)`) -- a provider dropped from
> the registry entirely never gets synced again, so without something
> else, its routes would stay `activa = 1` forever: still listed in
> `GET /v1/models`, still shown in `GET /v1/ranking`, still eligible as
> routing candidates that would fail every time. `Almacen.desactivar_proveedores_no_registrados`
> closes that gap: it runs once per sondeo cycle, before the per-provider
> loop, and deactivates (never deletes -- same "history detects renames"
> principle as everything else) any route whose provider is absent from
> the process's current registry. This is why OpenRouter's 16 rows
> actually disappeared from `/v1/models` instead of lingering as
> permanently-broken candidates.

Adding or removing a provider both need a **restart** to take effect --
see "Adding a new provider" below for why (`proveedores.cargar()` only
runs once, at startup).

All three are still "discovery" in the sense the project cares about
(§1 of the design spec: *the catalog is discovered from `/models`,
always*): even the third pattern never hardcodes a model **id** anywhere.
What can vary is only where the gateway learns a route's *capabilities*
from, because not every provider's `/models` is equally informative.

### Fully discovered (Kilo; OpenRouter as an example)

The provider's `/models` response includes everything the gateway needs
per model: whether it is free (`pricing.prompt == "0"`, more reliable than
trusting an `:free` suffix in the id), whether it supports tool calls
(`"tools" in supported_parameters`), whether it accepts image input
(`"image" in architecture.input_modalities`), its context window, and its
max output tokens. `catalogo.normalizar()` reads all of this straight from
the response; nothing about the provider needs declaring beyond its
`base_url` and (if it needs one) an API key.

Two independent filters run on top of this regardless of pattern, both
based on what the provider's own response *says* about a model, never on
a list of ids:

- A model whose `name`/`description` reads as a guardrail, a
  classifier, a reranker, an embeddings model, or a speech model gets
  discarded -- it is not a general-purpose chat model, no matter how cheap
  or fast it looks. (This exists because, in a fresh deployment where
  every route starts at the same neutral quality score, the fastest thing
  in the free pool used to be a content-safety classifier that just
  replies `"User Safety: safe"` to anything -- and it was winning `auto`.)
- A model that describes itself as a *meta-router* ("rotates through
  available free models", "selects a model at random") gets discarded too:
  it is not a model, it is a lottery, and measuring or ranking it measures
  whoever happened to answer that one time.

### Fully declared (MiniMax)

MiniMax's real `/models` response is bare -- `id`/`created`/`owned_by`,
nothing about pricing or capability -- so there is nothing useful to
discover. Both the id and the capabilities are written by hand under
`modelos_fijos`:

```yaml
modelos_fijos:
  - id: MiniMax-M3
    tools: true
    vision: false
    contexto: 128000
    max_salida: 32768
```

This is also, not coincidentally, the only pattern used for a `tier: pago`
provider today: paid routes are never probed (spending real money just to
measure them would defeat the point of a free-first gateway), so there is
no periodic re-verification of these numbers the way a discovered route
gets. Keep them conservative -- `contexto` here is documented in the YAML
as "a conservative floor, used only to filter by `x_min_contexto`"; raise
it only once you have confirmed a larger one against the real API.

### Discovered ids, declared capabilities (`chatgpt`)

`chatgpt-proxy`'s `/v1/models` is dynamic (it proxies ChatGPT's own
catalog, with a TTL cache and a fallback to the last-good response) but,
unlike Kilo/OpenRouter, never reports pricing or modality metadata --
every entry is just `id`/`object`/`created`/`owned_by`/`description`. This
is exactly the gap `capacidades_por_defecto` closes: declare the
capabilities **once**, for the provider as a whole, and every id that
provider's `/models` reports gets stamped with them:

```yaml
capacidades_por_defecto:
  tools: false
  vision: false
  contexto: 128000
  max_salida: 8192
```

When `capacidades_por_defecto` is set for a provider, `catalogo.normalizar()`
applies it to every discovered id **and skips the price/modality checks
entirely** -- a provider that declares defaults is asserting the things
its own catalog cannot tell the gateway, so those checks would have
nothing to check against anyway. The model ids themselves are still 100%
discovered: rename a model upstream and the new id shows up on its own,
with the same declared capabilities, no YAML edit required. This is the
general mechanism, not something special-cased to `chatgpt` -- **any**
future provider whose `/models` is similarly bare (ids only, no metadata)
can use it exactly the same way, by adding this one block to its entry.

Two more filters apply here too, but they are **not** something declared
in `chatgpt`'s YAML entry -- they live in `catalogo.normalizar()` /
`catalogo.es_id_reservado()` and run against **every** provider's
discovery pass, the same as the guardrail/meta-router filters two
sections up. They just happen to matter for `chatgpt` specifically,
because its catalog is the one that actually contains what they filter:

- Legacy aliases the proxy adds for compatibility (`gpt-4o`,
  `gpt-4o-mini`, ...), self-identified by `description` starting with
  `"Alias → "` -- not by a hardcoded id list.
- The id `auto` (and any `auto:<suffix>`), reserved by the gateway itself
  regardless of which provider tries to publish it: a real route with
  that literal id would be permanently unreachable, because
  `interpretar_pedido` always resolves `"auto"` and `"auto:*"` to the
  virtual alias first, never to a literal model id.

Any other provider whose discovered catalog happened to contain an id
matching either of these two shapes would have it filtered out exactly
the same way -- it is a property of `catalogo.normalizar()`, not of any
one provider's config.

## Other per-provider fields worth knowing about

- **`prioridad`** (default `100`): manual ordering within a tier, see
  [`routing-and-ranking.md`](routing-and-ranking.md). Does not affect
  discovery, only the order routes get tried in.
- **`base_url_env`**: names an environment variable whose value overrides
  `base_url` at load time, for a provider whose real address is not fixed
  yet (today: `chatgpt`, self-hosted, deployed alongside this gateway).
  See [`configuration.md`](configuration.md) for the path-suffix behavior
  this implies.
- **`desenvuelve_canvas`** (default `false`): whether the gateway should
  unwrap `:::word{...}` ... `:::` fences from this provider's responses.
  This is **not** a generic Markdown cleanup -- `:::note{...}` is also
  legitimate Docusaurus/MDX syntax, and applying this blindly to every
  provider once corrupted real documentation content coming back from
  Kilo. Only set this for a provider that is verified to actually emit
  these fences as a UI-mode artifact (today: only `chatgpt-proxy`, whose
  "canvas" mode does this).
- **`timeout_s`** (default: unset, meaning "use the gateway's global
  `TIMEOUT_S`", currently 90s): passed to httpx as a single scalar for
  both the non-streaming and the streaming path, which httpx applies
  uniformly to its `connect`/`write`/`read`/`pool` timeouts -- verified
  directly: a scalar `timeout=45.0` shows up as
  `{"connect": 45.0, "write": 45.0, "read": 45.0, "pool": 45.0}` in the
  request's extensions. **The dimension that matters for "does this bound
  a hang" is `read`, and `read` is a per-chunk timeout, not a wall-clock
  deadline on the whole attempt.** It resets every time a chunk of data
  arrives: a response that goes completely silent for `timeout_s` trips
  it cleanly (the case this setting is meant to catch), but a *streaming*
  response that keeps the connection alive by sending some chunk every
  `timeout_s - ε` seconds, forever, would never trip it at all -- that
  attempt could stay open far longer than `timeout_s` in total. Set this
  for a provider you know can hang rather than fail cleanly -- see
  `providers.yaml`'s own comment on the `chatgpt` entry for a worked
  example of picking a number from measured latency data, not a guess --
  but do not read it as a hard cap on how long any single attempt can
  possibly run.

## Adding a new provider

In the common case (a provider whose `/models` is as informative as
Kilo/OpenRouter's), this really is config-only:

1. Add an entry to `providers.yaml` with `id`, `tier`, `dialecto:
   openai`, `base_url`, and `modelos_path` (plus `clave_env` if it needs an
   API key).
2. If its `/models` does not report capabilities, add
   `capacidades_por_defecto` instead of expecting discovery to work.
3. If its `/models` reports **neither** ids nor capabilities usefully
   (MiniMax's shape), use `modelos_fijos` instead -- this is about what
   the catalog endpoint can tell the gateway, not about `tier`. A `tier:
   pago` provider with a `/models` as informative as Kilo's could use full
   discovery exactly the same way a free one does; MiniMax happens to use
   `modelos_fijos` because its `/models` is bare, and happens to also be
   the paid one, but those are two independent facts about it. The one
   thing that IS tied to `tier: pago`, unconditionally and regardless of
   which registration pattern a provider uses: it is never probed (see
   `sondeo.sondear_salud` / `sondear_calidad`, both filter to
   `tier == "gratis"`) -- spending real money just to measure a route
   would defeat the point of a free-first gateway.
4. **Restart the process.** `proveedores.cargar()` only runs once, at
   startup (`principal.crear_estado`) -- there is no mechanism that
   re-reads `providers.yaml` on its own, so a YAML edit needs a restart
   before anything downstream sees it. ("Wait for the next sync cycle"
   alone does *not* work, despite `sincronizar_catalogo` running
   periodically: it always operates on the same in-memory provider list
   loaded at startup, never a fresh read of the file.) Once restarted, the
   new routes show up in `GET /v1/models` and start accumulating
   measurements in `GET /v1/ranking` on their own (skipped for a `tier:
   pago` provider, per the point above -- it only ever gets measured by
   real traffic). The same applies in reverse when removing a provider --
   see the callout above on `desactivar_proveedores_no_registrados` for
   what happens to its already-discovered routes once the process comes
   back up with a shorter registry.

No code in `src/llm_libre/` needs to change for any of this -- if you find
yourself editing a `.py` file to add a provider, something about that
provider does not fit one of the three patterns above, and that gap is
worth raising rather than working around inline.
