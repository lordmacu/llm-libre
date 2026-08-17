# The provider model

How a provider is declared, and what each field in `proveedores.yaml`
buys you. The goal of this design is that **adding a provider is a config
change, not a code change** -- every field here exists to make that true
for one specific real-world constraint a provider might have.

## The three registration patterns

Every provider in `proveedores.yaml` fits one of three shapes, depending on
what its own `/models` endpoint can tell the gateway:

| Pattern | Model ids come from | Capabilities come from | Who uses it |
|---|---|---|---|
| **Fully discovered** | its `/models` | its `/models` | Kilo, OpenRouter |
| **Fully declared** | `modelos_fijos` in the YAML | `modelos_fijos` in the YAML | MiniMax |
| **Discovered ids, declared capabilities** | its `/models` | `capacidades_por_defecto` in the YAML | `chatgpt` |

All three are still "discovery" in the sense the project cares about
(§1 of the design spec: *the catalog is discovered from `/models`,
always*): even the third pattern never hardcodes a model **id** anywhere.
What can vary is only where the gateway learns a route's *capabilities*
from, because not every provider's `/models` is equally informative.

### Fully discovered (Kilo, OpenRouter)

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

`chatgpt`'s own YAML entry filters out two more things at discovery time,
both self-identified by the response itself, never by a hardcoded id list:
legacy aliases the proxy adds for compatibility (their `description`
starts with `"Alias → "`), and the id `auto` (and any `auto:<suffix>`) --
reserved by the gateway itself, since a real route with that literal id
would be permanently unreachable (`interpretar_pedido` always resolves
`"auto"` and `"auto:*"` to the virtual alias, never to a literal model id).

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
  `TIMEOUT_S`", currently 90s): caps how long the gateway waits on a
  single attempt against this specific provider, for both the
  non-streaming and the streaming path. Set this for a provider you know
  can hang rather than fail cleanly -- see `proveedores.yaml`'s own
  comment on the `chatgpt` entry for a worked example of picking a number
  from measured latency data, not a guess.

## Adding a new provider

In the common case (a provider whose `/models` is as informative as
Kilo/OpenRouter's), this really is config-only:

1. Add an entry to `proveedores.yaml` with `id`, `tier`, `dialecto:
   openai`, `base_url`, and `modelos_path` (plus `clave_env` if it needs an
   API key).
2. If its `/models` does not report capabilities, add
   `capacidades_por_defecto` instead of expecting discovery to work.
3. If it is `tier: pago`, use `modelos_fijos` instead -- paid routes are
   never probed, so there is no discovery loop to lean on for them.
4. Restart (or wait for the next sync cycle): the new routes show up in
   `GET /v1/models` and start accumulating measurements in
   `GET /v1/ranking` on their own.

No code in `src/llm_libre/` needs to change for any of this -- if you find
yourself editing a `.py` file to add a provider, something about that
provider does not fit one of the three patterns above, and that gap is
worth raising rather than working around inline.
