"""OpenAPI/Swagger documentation content for llm-libre (Task 14).

This module changes NO runtime behaviour: it only assembles text, examples and
JSON Schema fragments that `api.create_app` wires in through FastAPI's native
parameters (`summary`, `description`, `responses`, `openapi_extra`) and through
`customise_openapi`, which replaces `app.openapi` -- a function that runs only
when someone asks for `/docs` or `/openapi.json`, never on the path of a real
request.

In particular, `POST /v1/chat/completions` does NOT gain a Pydantic model: the
`requestBody` below is purely descriptive (`openapi_extra`), and the real
endpoint keeps reading `await request.json()` (see api.completions) so that an
unknown field still reaches the provider verbatim. See
`tests/test_api.py::test_an_unknown_field_reaches_the_provider_verbatim` and
`tests/test_client.py` for the proof that this is still intact.

The example bodies below are WIRE FORMAT: they are what a client reads in
/docs and copies into its own code. When a message string changes in api.py,
it has to change here too -- there is no test tying them together.
"""

from fastapi.openapi.utils import get_openapi

TITLE = "llm-libre"
VERSION = "0.1.0"  # keep in sync with pyproject.toml [project].version
SUMMARY = "One OpenAI-compatible endpoint for several providers' free LLM models."

DESCRIPTION = """
llm-libre is a gateway that puts several LLM providers' **free** tiers
behind a single [OpenAI-compatible](https://platform.openai.com/docs/api-reference/chat)
contract. Any OpenAI SDK works against it unmodified: point `base_url` at
this service and pass one of the configured API keys, nothing else changes.

## Why "auto" instead of picking a model yourself

Free model catalogs churn: ids get renamed or removed, and no single
provider's free tier is reliably the best one. llm-libre measures its own
providers (a small health probe plus a code-verifiable quality battery,
see `GET /v1/ranking`) and, when you request the `auto` model or one of
its variants, routes you to the best currently-healthy candidate --
retrying the next one automatically on failure.

## Authentication

Every endpoint except `GET /health` requires an API key, accepted through
**either** header (if both are present, `X-API-Key` wins):

- `X-API-Key: <key>`
- `Authorization: Bearer <key>` -- what any OpenAI SDK sends automatically
  when you set its `api_key` parameter, so the "just change `base_url`"
  promise holds without extra configuration.

## Rate limiting

Each key is limited to a fixed number of requests per minute (see the
`PER_MINUTE_LIMIT` deployment variable). Exceeding it returns `429`.

## Status codes, and why four of them are easy to confuse

- **`400`** -- no route could *ever* satisfy this request (an impossible
  capability/context combination). Not retryable as-is; fix the request.
- **`404`** -- the specific model id you asked for no longer exists, either
  in the local catalog or, for a model that is still in the local catalog,
  because the upstream provider itself just returned a live `404` for it.
  Comes with `suggestions` (nearest known ids).
- **`503`** -- routes that *could* serve this request exist, but right now
  they are all down or cooling down, or (for a request that allows the paid
  fallback) the calling key already spent its daily paid allowance. Safe to
  retry later; see `GET /v1/ranking` and `GET /health` to find out why.
- **`401`** -- missing or invalid API key.
- **`429`** -- the calling key is over its per-minute rate limit.
"""

# --- Security schemes: purely documentary, see customise_openapi() below. There
#     is no new fastapi.Security() in the code: the real authentication is still
#     the manual `require_api_key` api.py already had, untouched. This only makes
#     /docs show the padlock and lets you try the API from there. ------------

_SECURITY_SCHEMES = {
    "ApiKeyHeader": {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
        "description": "One of the configured LLM_LIBRE_API_KEYS. Wins over "
                       "Authorization if both headers are present.",
    },
    "BearerAuth": {
        "type": "http",
        "scheme": "bearer",
        "description": "Same keys as ApiKeyHeader, via the header any OpenAI "
                       "SDK sends automatically for its `api_key` parameter.",
    },
}
_OPERATION_SECURITY = [{"ApiKeyHeader": []}, {"BearerAuth": []}]

# Endpoints that do NOT require a key (see api.require_api_key -- /health is the only one).
_UNAUTHENTICATED_PATHS = {"/health"}


def customise_openapi(app) -> None:
    """Replace `app.openapi` with a version that adds the security schemes and the
    per-operation `security` requirement -- the only thing FastAPI cannot infer on
    its own, because this project's real authentication is a manual header check
    (`require_api_key`), not a `fastapi.Security(...)` declared in each endpoint's
    signature.

    It runs LAZILY (only the first time someone asks for `/docs` or
    `/openapi.json`, cached afterwards in `app.openapi_schema` -- the same pattern
    FastAPI uses internally) and never on the path of a real request to `/v1/*` or
    `/health`: zero risk of changing runtime behaviour.
    """
    def _generate():
        if app.openapi_schema:
            return app.openapi_schema
        schema_ = get_openapi(title=app.title, version=app.version,
                              summary=app.summary, description=app.description,
                              routes=app.routes)
        schema_.setdefault("components", {})["securitySchemes"] = _SECURITY_SCHEMES
        for path, operations in schema_.get("paths", {}).items():
            if path in _UNAUTHENTICATED_PATHS:
                continue
            for method, operation in operations.items():
                if method in {"get", "post", "put", "patch", "delete"}:
                    operation["security"] = _OPERATION_SECURITY
        _strip_spurious_422(schema_)
        app.openapi_schema = schema_
        return app.openapi_schema
    app.openapi = _generate


def _strip_spurious_422(schema_: dict) -> None:
    """Post-Task-14 review (gate), a Minor finding: FastAPI adds an automatic
    `422 Validation Error` to ANY operation with parameters declared via
    `Header`/`Query`/`Path`/body, without looking at whether they can actually
    fail their own validation. This gateway's five endpoints only declare
    `str | None = Header(None)` -- always valid (any string, or absent) -- so none
    of them can return that 422 in practice: documenting it would promise a
    response the service never produces. It is stripped from every operation;
    `HTTPValidationError`/`ValidationError` in `components.schemas` are ONLY
    referenced from that 422 (it is the only thing `get_openapi` generates them
    for), so with no 422 in any operation they are left orphaned and stripped
    too."""
    for operations in schema_.get("paths", {}).values():
        for operation in operations.values():
            if isinstance(operation, dict):
                operation.get("responses", {}).pop("422", None)
    schemas = schema_.get("components", {}).get("schemas", {})
    schemas.pop("HTTPValidationError", None)
    schemas.pop("ValidationError", None)


# --- POST /v1/chat/completions --------------------------------------------

_CHAT_BODY_SCHEMA = {
    "type": "object",
    "required": ["messages"],
    "description": (
        "This endpoint is a pure passthrough (see llm_libre.client.build_request): "
        "the request is read as raw JSON, never validated against a strict schema, "
        "and forwarded to the chosen provider almost verbatim -- only the gateway "
        "extensions below (`x_*`) are stripped, and `model` is rewritten to the "
        "provider's real id. Any other field, including standard OpenAI parameters "
        "not listed here (`temperature`, `top_p`, `max_tokens`, `response_format`, "
        "`seed`, `logprobs`, or any field a given upstream provider defines on its "
        "own) still reaches the provider unchanged. Note `model` is NOT required: "
        "a request with no `model` field "
        "at all defaults cleanly to the `auto` alias and returns `200`, exactly "
        "like sending `\"model\": \"auto\"` explicitly."
    ),
    "properties": {
        "model": {
            "type": "string",
            "example": "auto",
            "description": (
                "Either a real route id -- as listed by `GET /v1/models`, e.g. "
                "`nvidia/nemotron-3-super-120b-a12b:free` -- which fails over "
                "between every provider that serves that exact model; or one of "
                "the virtual aliases: `auto` (balanced profile), `auto:fast` "
                "(fast profile: latency weighs more), `auto:strong` (powerful "
                "profile: measured quality weighs more), `auto:tools` (balanced "
                "+ requires function-calling support), `auto:vision` (balanced + "
                "requires image-input support). `auto:balanced` also works "
                "(redundant with plain `auto`) but is not listed separately in "
                "`GET /v1/models`. An alias fails over across the gateway's "
                "whole ranked candidate list, paid fallback included if "
                "allowed. An `auto:` prefix with anything else after it (e.g. "
                "a typo like `auto:tolls`) returns `400` naming the unrecognized "
                "alias, instead of silently falling back to plain `auto`. "
                "Omitting `model` entirely (or sending `null`) defaults to "
                "`auto`; sending it as anything other than a string (a "
                "number, a boolean, an array, an object) returns `400` "
                "naming this field."
            ),
        },
        "messages": {
            "type": "array",
            "items": {"type": "object"},
            "description": "Standard OpenAI chat messages array, forwarded as-is.",
        },
        "stream": {
            "type": "boolean",
            "default": False,
            "description": (
                "If true, the response is Server-Sent Events "
                "(`text/event-stream`), OpenAI-shaped chunks ending in "
                "`data: [DONE]`. IMPORTANT: a streaming response never carries "
                "the `X-Route-Used` / `X-Tier` / `X-Attempts` headers -- see the "
                "endpoint description."
            ),
        },
        "tools": {
            "type": "array",
            "items": {"type": "object"},
            "description": (
                "Standard OpenAI tools array. Sending a non-empty `tools` array "
                "makes function-calling support REQUIRED for this request, "
                "exactly like `x_requires: [\"tools\"]` or the `auto:tools` "
                "alias -- you do not need to also set those; the gateway checks "
                "for the field's presence regardless of which model alias you used."
            ),
        },
        "tool_choice": {
            "description": "Standard OpenAI tool_choice, forwarded as-is.",
        },
        "temperature": {"type": "number", "description": "Forwarded as-is."},
        "max_tokens": {"type": "integer", "description": "Forwarded as-is."},
        "x_requires": {
            "oneOf": [
                {"type": "array", "items": {"type": "string", "enum": ["tools", "vision"]}},
                {"type": "string", "enum": ["tools", "vision"]},
            ],
            "example": ["tools"],
            "description": (
                "Gateway extension, interpreted here and never forwarded "
                "upstream. Capabilities the chosen route MUST advertise. Only "
                "`\"tools\"` and `\"vision\"` currently have any effect; other "
                "*string* values are accepted but ignored. Equivalent to (and "
                "combinable with) the `auto:tools` / `auto:vision` aliases. A "
                "single bare string (`\"tools\"`, not `[\"tools\"]`) is also "
                "accepted, as a convenience, and treated the same as a "
                "one-element list. Anything that is neither a string nor a "
                "list of strings (a number, a boolean, a list containing a "
                "non-string) returns `400` naming this field, instead of a "
                "generic server error."
            ),
        },
        "x_min_context": {
            "type": "integer",
            "example": 100000,
            "description": (
                "Gateway extension, interpreted here and never forwarded "
                "upstream. Minimum context window, in tokens, a route must "
                "advertise to be considered. `0` or omitted means no minimum. "
                "A value that cannot be read as an integer (e.g. a non-numeric "
                "string) returns `400` naming this field, instead of a generic "
                "server error."
            ),
        },
        "x_allow_paid": {
            "type": "boolean",
            "default": True,
            "description": (
                "Gateway extension, interpreted here and never forwarded "
                "upstream. Defaults to **true**: if every free route is down "
                "or exhausted, the gateway is allowed to fall back to the paid "
                "route (MiniMax) for this request, counted against the "
                "calling key's daily allowance (see `GET /v1/usage`) and always "
                "reported via `X-Tier: paid`. Set to `false` to forbid that "
                "fallback for this one request -- you then get `503` instead "
                "of an unexpected charge."
            ),
        },
        "x_raw": {
            "type": "boolean",
            "default": False,
            "description": (
                "Gateway extension, interpreted here and never forwarded "
                "upstream. Disables reasoning-tag stripping "
                "(`<think>`/`<thinking>`/`<reasoning>`): `content` comes back "
                "exactly as the provider sent it, and `x_reasoning` is never "
                "added. Useful for a streaming client, since `x_reasoning` "
                "is never available in streaming responses regardless of this flag."
            ),
        },
    },
    "additionalProperties": True,
}

_CHAT_BODY_EXAMPLES = {
    "auto": {
        "summary": "Let the gateway pick the best free route (balanced profile)",
        "value": {
            "model": "auto",
            "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
        },
    },
    "explicit_model": {
        "summary": "Ask for one specific model id (still fails over between "
                   "providers that serve it)",
        "value": {
            # The MODEL id exactly as GET /v1/models lists it -- WITHOUT the
            # provider prefix that X-Route-Used does carry
            # ("kilo/cohere/north-mini-code:free"). Sending the id WITH that
            # prefix matches no real modelo_id and returns a 404.
            "model": "cohere/north-mini-code:free",
            "messages": [{"role": "user", "content": "Write a haiku about databases."}],
        },
    },
    "streaming": {
        "summary": "Streaming (Server-Sent Events)",
        "value": {
            "model": "auto:fast",
            "stream": True,
            "messages": [{"role": "user", "content": "Count from 1 to 5."}],
        },
    },
    "tools_required": {
        "summary": "Function calling required (chatgpt's routes are skipped: "
                   "they declare tools=false)",
        "value": {
            "model": "auto:tools",
            "messages": [{"role": "user", "content": "What's the weather in Bogota?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Weather for a city",
                    "parameters": {"type": "object",
                                   "properties": {"city": {"type": "string"}},
                                   "required": ["city"]},
                },
            }],
        },
    },
    "no_paid_fallback": {
        "summary": "Never fall back to the paid route for this request",
        "value": {
            "model": "auto",
            "x_allow_paid": False,
            "messages": [{"role": "user", "content": "Summarize this in one line."}],
        },
    },
}

_CHAT_RESPONSE_EXAMPLE = {
    "id": "chatcmpl-example",
    "object": "chat.completion",
    "created": 1755400000,
    "model": "gpt-5-3-mini",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "Hello! How can I help you today?"},
        "finish_reason": "stop",
    }],
    "usage": {"prompt_tokens": 12, "completion_tokens": 9, "total_tokens": 21},
}

_CHAT_RESPONSE_EXAMPLE_WITH_REASONING = {
    **_CHAT_RESPONSE_EXAMPLE,
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "4"},
        "finish_reason": "stop",
    }],
    "x_reasoning": "The user asked for 2+2, which is 4.",
}

_STREAM_EXAMPLE = (
    'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n'
    'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
    'data: {"choices":[{"delta":{"content":"!"}}]}\n\n'
    'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
    'data: [DONE]\n\n'
)

CHAT_COMPLETIONS_DOCS = {
    "tags": ["Chat"],
    "summary": "Create a chat completion (OpenAI-compatible, with automatic routing)",
    "description": DESCRIPTION.split("## Status codes")[0].strip() + """

**Response headers** (non-streaming only -- see below):
- `X-Route-Used`: `<provider>/<model_id>` of the route that actually served the request
- `X-Tier`: `free` or `paid`
- `X-Attempts`: how many routes were tried before responding (or before giving up)

**Streaming responses never carry these three headers.** HTTP headers are
sent before the response body, and at that point the failover chain has
not been resolved yet -- the gateway does not yet know which route will
end up serving the request. Paid usage is still recorded even while
streaming (check `GET /v1/usage`), and the `model` field inside the
stream's own chunks says which model answered.

**Reasoning stripping.** Several free (and paid) models leak their
chain-of-thought inside `content`, wrapped in `<think>`, `<thinking>` or
`<reasoning>` tags. By default the gateway strips that from `content` and
returns it separately as `x_reasoning`, a top-level field any OpenAI
SDK ignores harmlessly -- only present in non-streaming responses, and
only when something was actually stripped. Set `x_raw: true` to disable
stripping entirely.
""",
    "openapi_extra": {
        "requestBody": {
            "required": True,
            "content": {"application/json": {
                "schema": _CHAT_BODY_SCHEMA,
                "examples": _CHAT_BODY_EXAMPLES,
            }},
        },
    },
    "responses": {
        200: {
            "description": (
                "Success. Non-streaming: the upstream provider's chat "
                "completion, passed through (plus `x_reasoning` if "
                "something was stripped). Streaming: Server-Sent Events."
            ),
            "headers": {
                "X-Route-Used": {"schema": {"type": "string"}, "example": "chatgpt/gpt-5-3-mini",
                                 "description": "Non-streaming only."},
                "X-Tier": {"schema": {"type": "string", "enum": ["free", "paid"]},
                          "example": "free", "description": "Non-streaming only."},
                "X-Attempts": {"schema": {"type": "integer"}, "example": 1,
                              "description": "Non-streaming only."},
            },
            "content": {
                "application/json": {
                    "examples": {
                        "plain": {"summary": "Plain answer (real example: auto -> "
                                             "chatgpt/gpt-5-3-mini)",
                                  "value": _CHAT_RESPONSE_EXAMPLE},
                        "with_reasoning_stripped": {
                            "summary": "A model that leaked <think> got it stripped",
                            "value": _CHAT_RESPONSE_EXAMPLE_WITH_REASONING},
                    },
                },
                "text/event-stream": {
                    "schema": {"type": "string"},
                    "example": _STREAM_EXAMPLE,
                },
            },
        },
        400: {"description": (
                 "The request cannot be satisfied -- either no route could ever "
                 "match what was asked (an impossible capability/context "
                 "combination), or the request itself is malformed in a way "
                 "this gateway checks for (an unrecognized `auto:<suffix>` "
                 "alias, or `model` / `x_requires` / `x_min_context` sent "
                 "with the wrong type -- these three always answer with the "
                 "same `{message, field, received_value}` shape)."),
             "content": {"application/json": {"examples": {
                 "impossible_capability": {
                     "summary": "No route can ever satisfy this combination",
                     "value": {"detail": {
                         "message": "no route satisfies the request",
                         "request": {"model": None, "needs_tools": False,
                                   "needs_vision": True, "min_context": 0,
                                   "profile": "balanced", "allow_paid": True},
                         "active_routes": 18}}},
                 "unknown_alias_suffix": {
                     "summary": "'auto:<typo>' -- not silently treated as plain 'auto'",
                     "value": {"detail": {
                         "message": "unknown model alias: 'auto:tolls'",
                         "suggestions": ["auto", "auto:fast", "auto:strong",
                                        "auto:tools", "auto:vision"]}}},
                 "invalid_x_min_context": {
                     "summary": "x_min_context is not a number",
                     "value": {"detail": {
                         "message": "x_min_context must be an integer",
                         "field": "x_min_context", "received_value": "a hundred thousand"}}},
                 "invalid_x_requires": {
                     "summary": "x_requires is neither a string nor a list of strings",
                     "value": {"detail": {
                         "message": "x_requires must be a string or a list of strings",
                         "field": "x_requires", "received_value": 5}}},
                 "invalid_model_type": {
                     "summary": "model is not a string",
                     "value": {"detail": {
                         "message": "model must be a string",
                         "field": "model", "received_value": 5}}},
             }}}},
        401: {"description": "Missing or invalid API key.",
             "content": {"application/json": {"example": {"detail": "invalid api key"}}}},
        404: {"description": "The requested model id no longer exists -- either "
                            "in the local catalog, or (for a model that is still "
                            "in the local catalog) because the provider itself "
                            "just returned a live 404 for it.",
             "content": {"application/json": {"example": {"detail": {
                 "message": "the model 'poolside/laguna-m.1:free' no longer exists",
                 "suggestions": ["poolside/laguna-s-2.1:free", "poolside/laguna-xs-2.1:free"],
             }}}}},
        429: {"description": "Per-key rate limit exceeded.",
             "content": {"application/json": {"example": {
                 "detail": "too many requests for this key"}}}},
        503: {"description": (
                 "Candidate routes exist but none can serve this request right "
                 "now: they are all down or cooling down (see `GET /v1/ranking` "
                 "for `cooldown_until` per route and `GET /health`), or the "
                 "calling key already spent its daily paid allowance."),
             "content": {"application/json": {"examples": {
                 "all_down_or_cooling": {
                     "summary": "Every candidate is down or in cooldown "
                                "(raised before any attempt)",
                     "value": {"detail": {
                         "message": "every route that could serve is down "
                                   "or in cooldown",
                         "request": {"model": None, "needs_tools": False,
                                   "needs_vision": False, "min_context": 0,
                                   "profile": "balanced", "allow_paid": True},
                         "compatible_routes": 18,
                         "next_release": 1755400300.0,
                         "paid_cap_reached": False}}},
                 "paid_allowance_spent": {
                     "summary": "Free routes exhausted and the key's daily "
                                "paid allowance is spent",
                     "value": {"detail": {
                         "message": "every route that could serve is down "
                                   "or in cooldown",
                         "request": {"model": None, "needs_tools": False,
                                   "needs_vision": False, "min_context": 0,
                                   "profile": "balanced", "allow_paid": True},
                         "compatible_routes": 18, "next_release": None,
                         "paid_cap_reached": True}}},
                 "chain_exhausted_after_attempts": {
                     "summary": "Every candidate was actually attempted and "
                                "each one failed just now (not pre-filtered by "
                                "cooldown) -- this shape has no `detail` key, "
                                "and DOES carry an X-Attempts header (see below)",
                     "value": {"error": {
                         "message": "no routes available",
                         "detail": "HTTP 500",
                         "next_release": 1755400060.0}}},
             }}},
             "headers": {
                 "X-Attempts": {
                     "schema": {"type": "integer"},
                     "example": 3,
                     "description": (
                         "Present ONLY on the 'chain exhausted after real "
                         "attempts' shape (`chain_exhausted_after_attempts` "
                         "above) -- absent on the 'pre-filtered, nothing was "
                         "attempted' shape (`all_down_or_cooling` / "
                         "`paid_allowance_spent`), which is raised before any "
                         "route is tried. Never accompanied by `X-Route-Used` "
                         "or `X-Tier` on a 503: those two only appear once a "
                         "route actually succeeds."
                     ),
                 },
             },
        },
    },
}


# --- GET /v1/models ---------------------------------------------------------

MODELS_DOCS = {
    "tags": ["Chat"],
    "summary": "List the normalized model catalog",
    "description": (
        "The OpenAI-shaped catalog (`GET /models` format, so OpenAI SDK "
        "tooling that lists models works unmodified) plus the gateway's own "
        "`auto*` aliases. `owned_by` is the provider id for a real route, or "
        "`\"llm-libre\"` for an alias. The number of REAL entries here always "
        "equals `GET /health`'s `active_routes` for that same moment -- the "
        "total returned here is that count plus the 5 fixed `auto*` aliases; "
        "cross-checking the two is a cheap sanity check that nothing is being "
        "double-counted or silently dropped."
    ),
    "responses": {
        200: {"description": "The catalog.", "content": {"application/json": {"example": {
            "object": "list",
            "data": [
                {"id": "gpt-5-3-mini", "object": "model", "owned_by": "chatgpt"},
                {"id": "cohere/north-mini-code:free", "object": "model", "owned_by": "kilo"},
                {"id": "poolside/laguna-s-2.1:free", "object": "model", "owned_by": "kilo"},
                {"id": "MiniMax-M3", "object": "model", "owned_by": "minimax"},
                {"id": "auto", "object": "model", "owned_by": "llm-libre"},
                {"id": "auto:fast", "object": "model", "owned_by": "llm-libre"},
                {"id": "auto:strong", "object": "model", "owned_by": "llm-libre"},
                {"id": "auto:tools", "object": "model", "owned_by": "llm-libre"},
                {"id": "auto:vision", "object": "model", "owned_by": "llm-libre"},
            ],
        }}}},
        401: {"description": "Missing or invalid API key."},
    },
}


# --- GET /v1/ranking ---------------------------------------------------------

RANKING_DOCS = {
    "tags": ["Diagnostics"],
    "summary": "Per-route ranking (not part of the OpenAI contract) -- the "
              "operator's main tool to explain a routing decision",
    "description": (
        "One row per active route, with every component that feeds its score "
        "broken out, **sorted with the exact same key the router uses** "
        "(`(in_cooldown, tier == \"paid\", priority, unmeasured, -score)`), "
        "not by score alone -- so the top row here is genuinely the route the "
        "router would try first right now, matching `X-Route-Used` on the next "
        "request. A route currently in cooldown sorts to the bottom of the "
        "table even with a perfect score, because the router would not pick it "
        "at this instant either.\n\n"
        "`quality` is `null` and `quality_assumed` holds the neutral value "
        "instead whenever `quality_measured` is `false`: a route that has never "
        "been through the quality battery is not silently shown as if a 0.6 "
        "had been measured.\n\n"
        "`cooldown_until` is a **raw Unix timestamp in seconds** (`0` means "
        "not in cooldown) -- unlike `last_probe` / `last_quality_probe`, "
        "which are ISO-8601 strings (or `null`).\n\n"
        "`ttft_p50_ms` (time to first token) is only ever measured by "
        "streaming traffic and probes; `latency_p50_ms` is the full "
        "round-trip and does not feed the score, only diagnostics -- the two "
        "are different magnitudes, do not average them together.\n\n"
        "The example rows below are illustrative (scores, timestamps and "
        "the exact set of routes drift as the live catalog and its "
        "measurements change) -- they use real route ids to stay concrete, "
        "not a captured live snapshot."
    ),
    "responses": {
        200: {"description": "The ranking table.", "content": {"application/json": {"example": {
            "routes": [
                {"key": "chatgpt/gpt-5-3-mini", "tier": "free", "priority": 0,
                 "score": 0.7912, "quality": 0.8, "quality_measured": True,
                 "quality_assumed": None, "last_quality_probe": "2026-08-17T03:12:04Z",
                 "last_probe": "2026-08-17T08:00:11Z", "reliability": 0.98,
                 "ttft_p50_ms": 900.0, "latency_p50_ms": 4500.0,
                 "cooldown_until": 0.0, "tools": False, "vision": False,
                 "context": 128000},
                {"key": "kilo/cohere/north-mini-code:free", "tier": "free",
                 "priority": 1, "score": 0.8420, "quality": 0.8,
                 "quality_measured": True, "quality_assumed": None,
                 "last_quality_probe": "2026-08-17T03:14:51Z",
                 "last_probe": "2026-08-17T08:01:02Z", "reliability": 1.0,
                 "ttft_p50_ms": 650.0, "latency_p50_ms": 2100.0,
                 "cooldown_until": 0.0, "tools": True, "vision": False,
                 "context": 128000},
                {"key": "kilo/poolside/laguna-s-2.1:free", "tier": "free",
                 "priority": 1, "score": 0.5015, "quality": 0.6,
                 "quality_measured": False, "quality_assumed": 0.6,
                 "last_quality_probe": None, "last_probe": "2026-08-17T08:02:00Z",
                 "reliability": 0.8, "ttft_p50_ms": 1500.0, "latency_p50_ms": None,
                 "cooldown_until": 1755400930.0, "tools": False, "vision": False,
                 "context": 65536},
                {"key": "minimax/MiniMax-M3", "tier": "paid", "priority": 2,
                 "score": 0.95, "quality": 0.9, "quality_measured": True,
                 "quality_assumed": None, "last_quality_probe": None,
                 "last_probe": None, "reliability": 0.8, "ttft_p50_ms": 1500.0,
                 "latency_p50_ms": None, "cooldown_until": 0.0, "tools": True,
                 "vision": False, "context": 128000},
            ],
        }}}},
        401: {"description": "Missing or invalid API key."},
    },
}


# --- GET /v1/usage ---------------------------------------------------------

ASSETS_DOCS = {
    "tags": ["OpenAI-compatible"],
    "summary": "Fetch a binary this gateway generated",
    "description": (
        "Serves an asset produced by `POST /v1/images/generations`. **No API key**, "
        "on purpose: the URL has to work in an `<img>` tag, a markdown preview or a "
        "browser address bar, none of which can attach a header.\n\n"
        "What guards it instead is the id -- the SHA-256 of the content, so it cannot "
        "be guessed, enumerated or derived from the prompt. Nothing lists assets, and "
        "nothing here accepts a URL from a caller.\n\n"
        "The response is `immutable` and that is literal rather than optimistic: the "
        "id IS the hash of the bytes, so a given id can never return different "
        "content. Assets are pruned by age, after which the id returns `404` -- never "
        "something else.\n\n"
        "Why the gateway hosts these at all: a provider's own URL expires (Mistral "
        "hands out Azure Blob SAS links), may need credentials the client does not "
        "have, and names who served the request -- which is the one thing this "
        "gateway exists to keep clients from having to care about."
    ),
    "responses": {
        200: {"description": "The asset bytes, with its stored content type."},
        404: {"description": "Unknown id, or the asset was pruned.",
              "content": {"application/json": {"example": {"detail": "asset not found"}}}},
    },
}

IMAGES_DOCS = {
    "tags": ["OpenAI-compatible"],
    "summary": "Generate images, routed only to models that actually can",
    "description": (
        "OpenAI's image-generation contract. As with chat, a client changes only "
        "`base_url`.\n\n"
        "What this endpoint adds over calling a provider directly is the **filter**: "
        "the gateway tracks image generation as a routing capability of its own "
        "(`images` in `GET /v1/ranking`), separate from `vision` -- `vision` is image "
        "INPUT, this is image OUTPUT, and a route can have either, both or neither. "
        "Routes that cannot generate are dropped BEFORE any request is attempted, so a "
        "prompt never spends a full timeout on a chat-only model just to be told it "
        "cannot draw.\n\n"
        "Failover, cooldowns, per-route telemetry and the daily paid cap all work "
        "exactly as they do for chat, and the same `X-Route-Used` / `X-Tier` / "
        "`X-Attempts` headers come back.\n\n"
        "A `200` whose `data` array is EMPTY is not treated as a success: the gateway "
        "counts it as a failed attempt for that route and moves to the next one, "
        "rather than handing back `{\"data\": []}`.\n\n"
        "`503` means every image-capable route is down, in cooldown, or out of quota "
        "-- both upstreams rate-limit generation far more aggressively than chat."
    ),
    "responses": {
        200: {"description": "One or more generated images.",
              "content": {"application/json": {"example": {
                  "created": 1755400000,
                  "data": [{"url": "https://.../generated.png"}]}}}},
        401: {"description": "Missing or invalid API key.",
              "content": {"application/json": {"example": {"detail": "invalid api key"}}}},
        400: {"description": "No route can generate images at all -- the pool has none, "
                             "which is a different situation from them all being busy.",
              "content": {"application/json": {"example": {"detail": {
                  "message": "no route satisfies the request",
                  "request": {"model": None, "needs_tools": False, "needs_vision": False,
                              "needs_images": True, "min_context": 0,
                              "profile": "balanced", "allow_paid": True},
                  "active_routes": 52}}}}},
        503: {"description": "Image-capable routes exist but none can serve right now.",
              "content": {"application/json": {"example": {"error": {
                  "message": "no routes available",
                  "detail": "200 with no generated image",
                  "next_release": None}}}}},
    },
}

USAGE_DOCS = {
    "tags": ["Diagnostics"],
    "summary": "Today's paid-usage counter for the calling key",
    "description": (
        "How many requests the calling key spent against the paid fallback "
        "(MiniMax) today (UTC), against its daily allowance "
        "(`DAILY_PAID_CAP`). Counts every **billable** attempt, not only "
        "successful ones -- a `200` from the paid provider is billable "
        "whether or not it contained a usable answer, and streaming counts "
        "exactly once per request, not once per chunk. This is the only place "
        "a streaming client can see whether the paid fallback served (or was "
        "attempted for) its last request, since streaming responses do not "
        "carry `X-Tier`."
    ),
    "responses": {
        200: {"description": "Today's paid usage for this key.",
             "content": {"application/json": {"example": {
                 "day": "2026-08-17", "paid_today": 0, "cap": 200}}}},
        401: {"description": "Missing or invalid API key."},
    },
}


# --- GET /health ---------------------------------------------------------

HEALTH_DOCS = {
    "tags": ["Health"],
    "summary": "Service health -- no API key required",
    "description": (
        "Honest by design (see the design spec, §6): this does not just check "
        "that the process is up, it checks whether there is a route that "
        "could actually serve a request right now. Meant to be wired as the "
        "deployment's health check (Coolify's, in production).\n\n"
        "- **`ok`** (HTTP `200`) -- at least one **free** route is alive: not "
        "in cooldown, and with positive evidence that it serves (a recent "
        "real success, a recent successful health probe, or no telemetry at "
        "all yet -- a freshly-discovered route is not born dead). Normal.\n"
        "- **`degraded`** (HTTP `503`) -- no free route is alive, but the "
        "paid route is. Every request is now either failing or being billed; "
        "find out why the free tier is down (`GET /v1/ranking`, provider "
        "status) before the daily paid allowance runs out for every key.\n"
        "- **`down`** (HTTP `503`) -- nothing is alive. `POST "
        "/v1/chat/completions` will return `503` for every request.\n\n"
        "A route needs **two consecutive failed health probes** (no success "
        "in between) to be counted as dead here -- a single failed probe is "
        "not enough, to avoid one transient blip (made more likely by the "
        "fact that a client's own traffic can indirectly trigger on-demand "
        "probes) flipping this to `down` and restarting the container over "
        "nothing. A real outage still fails its second probe just as fast as "
        "its first, so genuine detection is not slowed down.\n\n"
        "**With a valid API key** the body also carries `providers`: the last "
        "capability contract each in-house proxy published on its own "
        "`/health` -- contract version, `auth_mode`, `plan`, `expires_at`, the "
        "reported capability booleans and when they were last confirmed "
        "(`seen_at`). It is behind a key because it names the operator's "
        "account tier and renewal date; a keyless caller gets exactly the four "
        "fields above, which is what a health check reads. The booleans there "
        "are what the PROXY claims -- the effective per-route values, after "
        "`exceptions` and `emulates_tools`, are in `GET /v1/ranking`."
    ),
    "responses": {
        200: {"description": "At least one free route is alive.",
             "content": {"application/json": {"example": {
                 "status": "ok", "active_routes": 18, "available_routes": 17,
                 "free_available": 17}}}},
        503: {"description": "`degraded` (only the paid route is alive) or "
                            "`down` (nothing is alive).",
             "content": {"application/json": {"examples": {
                 "degraded": {"summary": "Only the paid route is alive",
                              "value": {"status": "degraded", "active_routes": 18,
                                       "available_routes": 1, "free_available": 0}},
                 "down": {"summary": "Nothing is alive",
                          "value": {"status": "down", "active_routes": 18,
                                   "available_routes": 0, "free_available": 0}},
             }}}},
    },
}
