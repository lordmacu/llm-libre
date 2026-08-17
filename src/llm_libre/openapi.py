"""Contenido de documentacion OpenAPI/Swagger para llm-libre (Task 14).

En INGLES a proposito -- el resto del codigo, sus comentarios y el spec
quedan en espanol; solo el texto que termina sirviendose en `/docs` y
`/openapi.json` (el entregable de Task 14) esta en ingles aca.

Este modulo NO cambia comportamiento de runtime: solo arma texto,
ejemplos y fragmentos de JSON Schema que `api.crear_app` conecta via los
parametros nativos de FastAPI (`summary`, `description`, `responses`,
`openapi_extra`) y via `personalizar_openapi`, que reemplaza `app.openapi`
-- una funcion que solo corre cuando alguien pide `/docs` u
`/openapi.json`, nunca en el camino de una peticion real. En particular,
`POST /v1/chat/completions` NO gana un modelo Pydantic: el `requestBody`
de aca abajo es puramente descriptivo (`openapi_extra`), y el endpoint
real sigue leyendo `await request.json()` (ver api.completions) para que
un campo desconocido siga llegando al proveedor tal cual -- ver
`tests/test_api.py::test_un_campo_desconocido_llega_al_proveedor_tal_cual`
y `tests/test_cliente.py` para la prueba de que esto sigue intacto.
"""

from fastapi.openapi.utils import get_openapi

TITULO = "llm-libre"
VERSION = "0.1.0"  # mantener sincronizado con pyproject.toml [project].version
RESUMEN = "One OpenAI-compatible endpoint for several providers' free LLM models."

DESCRIPCION = """
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
`LIMITE_POR_MINUTO` deployment variable). Exceeding it returns `429`.

## Status codes, and why four of them are easy to confuse

- **`400`** -- no route could *ever* satisfy this request (an impossible
  capability/context combination). Not retryable as-is; fix the request.
- **`404`** -- the specific model id you asked for no longer exists, either
  in the local catalog or, for a model that is still in the local catalog,
  because the upstream provider itself just returned a live `404` for it.
  Comes with `sugerencias` (nearest known ids).
- **`503`** -- routes that *could* serve this request exist, but right now
  they are all down or cooling down, or (for a request that allows the paid
  fallback) the calling key already spent its daily paid allowance. Safe to
  retry later; see `GET /v1/ranking` and `GET /health` to find out why.
- **`401`** -- missing or invalid API key.
- **`429`** -- the calling key is over its per-minute rate limit.
"""

# --- Security schemes: puramente documental, ver personalizar_openapi() mas
#     abajo. No hay ningun fastapi.Security() nuevo en el codigo: la
#     autenticacion real sigue siendo el `exigir_llave` manual que api.py ya
#     tenia, sin tocar. Esto solo hace que /docs muestre el candado y deje
#     probar la API desde ahi. --------------------------------------------

_ESQUEMAS_SEGURIDAD = {
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
_SEGURIDAD_OPERACION = [{"ApiKeyHeader": []}, {"BearerAuth": []}]

# Endpoints que NO exigen llave (ver api.exigir_llave -- /health es el unico).
_SIN_AUTENTICACION = {"/health"}


def personalizar_openapi(app) -> None:
    """Reemplaza `app.openapi` por una version que agrega los esquemas de
    seguridad y el requisito `security` por operacion -- lo unico que
    FastAPI no puede inferir solo, porque la autenticacion real de este
    proyecto es un chequeo manual de cabeceras (`exigir_llave`), no un
    `fastapi.Security(...)` declarado en la firma de cada endpoint.

    Corre PEREZOSO (solo la primera vez que alguien pide `/docs` u
    `/openapi.json`, cacheado despues en `app.openapi_schema` -- el mismo
    patron que usa FastAPI internamente) y nunca en el camino de una
    peticion real a `/v1/*` o `/health`: cero riesgo de cambiar
    comportamiento de runtime.
    """
    def _generar():
        if app.openapi_schema:
            return app.openapi_schema
        esquema = get_openapi(title=app.title, version=app.version,
                              summary=app.summary, description=app.description,
                              routes=app.routes)
        esquema.setdefault("components", {})["securitySchemes"] = _ESQUEMAS_SEGURIDAD
        for ruta, operaciones in esquema.get("paths", {}).items():
            if ruta in _SIN_AUTENTICACION:
                continue
            for metodo, operacion in operaciones.items():
                if metodo in {"get", "post", "put", "patch", "delete"}:
                    operacion["security"] = _SEGURIDAD_OPERACION
        _quitar_422_espureo(esquema)
        app.openapi_schema = esquema
        return app.openapi_schema
    app.openapi = _generar


def _quitar_422_espureo(esquema: dict) -> None:
    """Revision post-Task-14 (gate), hallazgo Minor: FastAPI agrega un `422
    Validation Error` automatico a CUALQUIER operacion con parametros
    declarados via `Header`/`Query`/`Path`/body, sin mirar si de verdad
    pueden fallar su propia validacion. Los cinco endpoints de este
    gateway solo declaran `str | None = Header(None)` -- siempre validos
    (cualquier string, o ausente) -- asi que ninguno puede devolver este
    422 en la practica: documentarlo prometeria una respuesta que el
    servicio nunca produce. Se saca de cada operacion; `HTTPValidationError`/
    `ValidationError` en `components.schemas` SOLO se referencian desde ese
    422 (es lo unico que `get_openapi` los genera para), asi que sin
    ningun 422 en ninguna operacion quedan huerfanos y se sacan tambien."""
    for operaciones in esquema.get("paths", {}).values():
        for operacion in operaciones.values():
            if isinstance(operacion, dict):
                operacion.get("responses", {}).pop("422", None)
    schemas = esquema.get("components", {}).get("schemas", {})
    schemas.pop("HTTPValidationError", None)
    schemas.pop("ValidationError", None)


# --- POST /v1/chat/completions --------------------------------------------

_ESQUEMA_CUERPO_CHAT = {
    "type": "object",
    "required": ["messages"],
    "description": (
        "This endpoint is a pure passthrough (see llm_libre.cliente.armar_peticion): "
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
                "the virtual aliases: `auto` (balanced profile), `auto:rapido` "
                "(fast profile: latency weighs more), `auto:potente` (powerful "
                "profile: measured quality weighs more), `auto:tools` (balanced "
                "+ requires function-calling support), `auto:vision` (balanced + "
                "requires image-input support). `auto:balanceado` also works "
                "(redundant with plain `auto`) but is not listed separately in "
                "`GET /v1/models`. An alias fails over across the gateway's "
                "whole ranked candidate list, paid fallback included if "
                "allowed. An `auto:` prefix with anything else after it (e.g. "
                "a typo like `auto:tolls`) returns `400` naming the unrecognized "
                "alias, instead of silently falling back to plain `auto`."
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
                "the `X-Ruta-Usada` / `X-Tier` / `X-Intentos` headers -- see the "
                "endpoint description."
            ),
        },
        "tools": {
            "type": "array",
            "items": {"type": "object"},
            "description": (
                "Standard OpenAI tools array. Sending a non-empty `tools` array "
                "makes function-calling support REQUIRED for this request, "
                "exactly like `x_requiere: [\"tools\"]` or the `auto:tools` "
                "alias -- you do not need to also set those; the gateway checks "
                "for the field's presence regardless of which model alias you used."
            ),
        },
        "tool_choice": {
            "description": "Standard OpenAI tool_choice, forwarded as-is.",
        },
        "temperature": {"type": "number", "description": "Forwarded as-is."},
        "max_tokens": {"type": "integer", "description": "Forwarded as-is."},
        "x_requiere": {
            "oneOf": [
                {"type": "array", "items": {"type": "string", "enum": ["tools", "vision"]}},
                {"type": "string", "enum": ["tools", "vision"]},
            ],
            "example": ["tools"],
            "description": (
                "Gateway extension, interpreted here and never forwarded "
                "upstream. Capabilities the chosen route MUST advertise. Only "
                "`\"tools\"` and `\"vision\"` currently have any effect; other "
                "values are accepted but ignored. Equivalent to (and "
                "combinable with) the `auto:tools` / `auto:vision` aliases. A "
                "single bare string (`\"tools\"`, not `[\"tools\"]`) is also "
                "accepted, as a convenience, and treated the same as a "
                "one-element list."
            ),
        },
        "x_min_contexto": {
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
        "x_permitir_pago": {
            "type": "boolean",
            "default": True,
            "description": (
                "Gateway extension, interpreted here and never forwarded "
                "upstream. Defaults to **true**: if every free route is down "
                "or exhausted, the gateway is allowed to fall back to the paid "
                "route (MiniMax) for this request, counted against the "
                "calling key's daily allowance (see `GET /v1/uso`) and always "
                "reported via `X-Tier: pago`. Set to `false` to forbid that "
                "fallback for this one request -- you then get `503` instead "
                "of an unexpected charge."
            ),
        },
        "x_crudo": {
            "type": "boolean",
            "default": False,
            "description": (
                "Gateway extension, interpreted here and never forwarded "
                "upstream. Disables reasoning-tag stripping "
                "(`<think>`/`<thinking>`/`<reasoning>`): `content` comes back "
                "exactly as the provider sent it, and `x_razonamiento` is never "
                "added. Useful for a streaming client, since `x_razonamiento` "
                "is never available in streaming responses regardless of this flag."
            ),
        },
    },
    "additionalProperties": True,
}

_EJEMPLOS_CUERPO_CHAT = {
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
            # El id de MODELO tal cual lo lista GET /v1/models -- SIN el
            # prefijo de proveedor que si lleva X-Ruta-Usada
            # ("kilo/cohere/north-mini-code:free"). Mandar el id CON ese
            # prefijo no matchea ningun modelo_id real y da 404.
            "model": "cohere/north-mini-code:free",
            "messages": [{"role": "user", "content": "Write a haiku about databases."}],
        },
    },
    "streaming": {
        "summary": "Streaming (Server-Sent Events)",
        "value": {
            "model": "auto:rapido",
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
            "x_permitir_pago": False,
            "messages": [{"role": "user", "content": "Summarize this in one line."}],
        },
    },
}

_EJEMPLO_RESPUESTA_CHAT = {
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

_EJEMPLO_RESPUESTA_CHAT_CON_RAZONAMIENTO = {
    **_EJEMPLO_RESPUESTA_CHAT,
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "4"},
        "finish_reason": "stop",
    }],
    "x_razonamiento": "The user asked for 2+2, which is 4.",
}

_EJEMPLO_STREAM = (
    'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n'
    'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
    'data: {"choices":[{"delta":{"content":"!"}}]}\n\n'
    'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
    'data: [DONE]\n\n'
)

CHAT_COMPLETIONS_DOCS = {
    "tags": ["Chat"],
    "summary": "Create a chat completion (OpenAI-compatible, with automatic routing)",
    "description": DESCRIPCION.split("## Status codes")[0].strip() + """

**Response headers** (non-streaming only -- see below):
- `X-Ruta-Usada`: `<provider>/<model_id>` of the route that actually served the request
- `X-Tier`: `gratis` or `pago`
- `X-Intentos`: how many routes were tried before responding (or before giving up)

**Streaming responses never carry these three headers.** HTTP headers are
sent before the response body, and at that point the failover chain has
not been resolved yet -- the gateway does not yet know which route will
end up serving the request. Paid usage is still recorded even while
streaming (check `GET /v1/uso`), and the `model` field inside the
stream's own chunks says which model answered.

**Reasoning stripping.** Several free (and paid) models leak their
chain-of-thought inside `content`, wrapped in `<think>`, `<thinking>` or
`<reasoning>` tags. By default the gateway strips that from `content` and
returns it separately as `x_razonamiento`, a top-level field any OpenAI
SDK ignores harmlessly -- only present in non-streaming responses, and
only when something was actually stripped. Set `x_crudo: true` to disable
stripping entirely.
""",
    "openapi_extra": {
        "requestBody": {
            "required": True,
            "content": {"application/json": {
                "schema": _ESQUEMA_CUERPO_CHAT,
                "examples": _EJEMPLOS_CUERPO_CHAT,
            }},
        },
    },
    "responses": {
        200: {
            "description": (
                "Success. Non-streaming: the upstream provider's chat "
                "completion, passed through (plus `x_razonamiento` if "
                "something was stripped). Streaming: Server-Sent Events."
            ),
            "headers": {
                "X-Ruta-Usada": {"schema": {"type": "string"}, "example": "chatgpt/gpt-5-3-mini",
                                 "description": "Non-streaming only."},
                "X-Tier": {"schema": {"type": "string", "enum": ["gratis", "pago"]},
                          "example": "gratis", "description": "Non-streaming only."},
                "X-Intentos": {"schema": {"type": "integer"}, "example": 1,
                              "description": "Non-streaming only."},
            },
            "content": {
                "application/json": {
                    "examples": {
                        "plain": {"summary": "Plain answer (real example: auto -> "
                                             "chatgpt/gpt-5-3-mini)",
                                  "value": _EJEMPLO_RESPUESTA_CHAT},
                        "with_reasoning_stripped": {
                            "summary": "A model that leaked <think> got it stripped",
                            "value": _EJEMPLO_RESPUESTA_CHAT_CON_RAZONAMIENTO},
                    },
                },
                "text/event-stream": {
                    "schema": {"type": "string"},
                    "example": _EJEMPLO_STREAM,
                },
            },
        },
        400: {"description": (
                 "The request cannot be satisfied -- either no route could ever "
                 "match what was asked (an impossible capability/context "
                 "combination), or the request itself is malformed in a way "
                 "this gateway checks for (an unrecognized `auto:<suffix>` "
                 "alias, a non-numeric `x_min_contexto`)."),
             "content": {"application/json": {"examples": {
                 "impossible_capability": {
                     "summary": "No route can ever satisfy this combination",
                     "value": {"detail": {
                         "message": "ninguna ruta cumple lo pedido",
                         "pedido": {"modelo": None, "requiere_tools": False,
                                   "requiere_vision": True, "min_contexto": 0,
                                   "perfil": "balanceado", "permitir_pago": True},
                         "rutas_activas": 18}}},
                 "unknown_alias_suffix": {
                     "summary": "'auto:<typo>' -- not silently treated as plain 'auto'",
                     "value": {"detail": {
                         "message": "alias de modelo desconocido: 'auto:tolls'",
                         "sugerencias": ["auto", "auto:rapido", "auto:potente",
                                        "auto:tools", "auto:vision"]}}},
                 "invalid_x_min_contexto": {
                     "summary": "x_min_contexto is not a number",
                     "value": {"detail": {
                         "message": "x_min_contexto debe ser un numero entero",
                         "campo": "x_min_contexto", "valor_recibido": "cien mil"}}},
             }}}},
        401: {"description": "Missing or invalid API key.",
             "content": {"application/json": {"example": {"detail": "llave invalida"}}}},
        404: {"description": "The requested model id no longer exists -- either "
                            "in the local catalog, or (for a model that is still "
                            "in the local catalog) because the provider itself "
                            "just returned a live 404 for it.",
             "content": {"application/json": {"example": {"detail": {
                 "message": "el modelo 'poolside/laguna-m.1:free' ya no existe",
                 "sugerencias": ["poolside/laguna-s-2.1:free", "poolside/laguna-xs-2.1:free"],
             }}}}},
        429: {"description": "Per-key rate limit exceeded.",
             "content": {"application/json": {"example": {
                 "detail": "demasiadas peticiones para esta llave"}}}},
        503: {"description": (
                 "Candidate routes exist but none can serve this request right "
                 "now: they are all down or cooling down (see `GET /v1/ranking` "
                 "for `en_cooldown_hasta` per route and `GET /health`), or the "
                 "calling key already spent its daily paid allowance."),
             "content": {"application/json": {"examples": {
                 "all_down_or_cooling": {
                     "summary": "Every candidate is down or in cooldown "
                                "(raised before any attempt)",
                     "value": {"detail": {
                         "message": "todas las rutas que podrian servir estan "
                                   "caidas o en cooldown",
                         "pedido": {"modelo": None, "requiere_tools": False,
                                   "requiere_vision": False, "min_contexto": 0,
                                   "perfil": "balanceado", "permitir_pago": True},
                         "rutas_compatibles": 18,
                         "proxima_liberacion": 1755400300.0,
                         "tope_pago_alcanzado": False}}},
                 "paid_allowance_spent": {
                     "summary": "Free routes exhausted and the key's daily "
                                "paid allowance is spent",
                     "value": {"detail": {
                         "message": "todas las rutas que podrian servir estan "
                                   "caidas o en cooldown",
                         "pedido": {"modelo": None, "requiere_tools": False,
                                   "requiere_vision": False, "min_contexto": 0,
                                   "perfil": "balanceado", "permitir_pago": True},
                         "rutas_compatibles": 18, "proxima_liberacion": None,
                         "tope_pago_alcanzado": True}}},
                 "chain_exhausted_after_attempts": {
                     "summary": "Every candidate was actually attempted and "
                                "each one failed just now (not pre-filtered by "
                                "cooldown) -- this shape has no `detail` key, "
                                "and DOES carry an X-Intentos header (see below)",
                     "value": {"error": {
                         "message": "sin rutas disponibles",
                         "detalle": "HTTP 500",
                         "proxima_liberacion": 1755400060.0}}},
             }}},
             "headers": {
                 "X-Intentos": {
                     "schema": {"type": "integer"},
                     "example": 3,
                     "description": (
                         "Present ONLY on the 'chain exhausted after real "
                         "attempts' shape (`chain_exhausted_after_attempts` "
                         "above) -- absent on the 'pre-filtered, nothing was "
                         "attempted' shape (`all_down_or_cooling` / "
                         "`paid_allowance_spent`), which is raised before any "
                         "route is tried. Never accompanied by `X-Ruta-Usada` "
                         "or `X-Tier` on a 503: those two only appear once a "
                         "route actually succeeds."
                     ),
                 },
             },
        },
    },
}


# --- GET /v1/models ---------------------------------------------------------

MODELOS_DOCS = {
    "tags": ["Chat"],
    "summary": "List the normalized model catalog",
    "description": (
        "The OpenAI-shaped catalog (`GET /models` format, so OpenAI SDK "
        "tooling that lists models works unmodified) plus the gateway's own "
        "`auto*` aliases. `owned_by` is the provider id for a real route, or "
        "`\"llm-libre\"` for an alias. The number of REAL entries here always "
        "equals `GET /health`'s `rutas_activas` for that same moment -- the "
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
                {"id": "auto:rapido", "object": "model", "owned_by": "llm-libre"},
                {"id": "auto:potente", "object": "model", "owned_by": "llm-libre"},
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
        "(`(en_cooldown, tier == \"pago\", prioridad, unmeasured, -puntaje)`), "
        "not by score alone -- so the top row here is genuinely the route the "
        "router would try first right now, matching `X-Ruta-Usada` on the next "
        "request. A route currently in cooldown sorts to the bottom of the "
        "table even with a perfect score, because the router would not pick it "
        "at this instant either.\n\n"
        "`calidad` is `null` and `calidad_asumida` holds the neutral value "
        "instead whenever `calidad_medida` is `false`: a route that has never "
        "been through the quality battery is not silently shown as if a 0.6 "
        "had been measured.\n\n"
        "`en_cooldown_hasta` is a **raw Unix timestamp in seconds** (`0` means "
        "not in cooldown) -- unlike `ultima_sonda` / `ultima_sonda_calidad`, "
        "which are ISO-8601 strings (or `null`).\n\n"
        "`ttft_p50_ms` (time to first token) is only ever measured by "
        "streaming traffic and probes; `latencia_p50_ms` is the full "
        "round-trip and does not feed the score, only diagnostics -- the two "
        "are different magnitudes, do not average them together.\n\n"
        "The example rows below are illustrative (scores, timestamps and "
        "the exact set of routes drift as the live catalog and its "
        "measurements change) -- they use real route ids to stay concrete, "
        "not a captured live snapshot."
    ),
    "responses": {
        200: {"description": "The ranking table.", "content": {"application/json": {"example": {
            "rutas": [
                {"clave": "chatgpt/gpt-5-3-mini", "tier": "gratis", "prioridad": 0,
                 "puntaje": 0.7912, "calidad": 0.8, "calidad_medida": True,
                 "calidad_asumida": None, "ultima_sonda_calidad": "2026-08-17T03:12:04Z",
                 "ultima_sonda": "2026-08-17T08:00:11Z", "confiabilidad": 0.98,
                 "ttft_p50_ms": 900.0, "latencia_p50_ms": 4500.0,
                 "en_cooldown_hasta": 0.0, "tools": False, "vision": False,
                 "contexto": 128000},
                {"clave": "kilo/cohere/north-mini-code:free", "tier": "gratis",
                 "prioridad": 1, "puntaje": 0.8420, "calidad": 0.8,
                 "calidad_medida": True, "calidad_asumida": None,
                 "ultima_sonda_calidad": "2026-08-17T03:14:51Z",
                 "ultima_sonda": "2026-08-17T08:01:02Z", "confiabilidad": 1.0,
                 "ttft_p50_ms": 650.0, "latencia_p50_ms": 2100.0,
                 "en_cooldown_hasta": 0.0, "tools": True, "vision": False,
                 "contexto": 128000},
                {"clave": "kilo/poolside/laguna-s-2.1:free", "tier": "gratis",
                 "prioridad": 1, "puntaje": 0.5015, "calidad": 0.6,
                 "calidad_medida": False, "calidad_asumida": 0.6,
                 "ultima_sonda_calidad": None, "ultima_sonda": "2026-08-17T08:02:00Z",
                 "confiabilidad": 0.8, "ttft_p50_ms": 1500.0, "latencia_p50_ms": None,
                 "en_cooldown_hasta": 1755400930.0, "tools": False, "vision": False,
                 "contexto": 65536},
                {"clave": "minimax/MiniMax-M3", "tier": "pago", "prioridad": 2,
                 "puntaje": 0.95, "calidad": 0.9, "calidad_medida": True,
                 "calidad_asumida": None, "ultima_sonda_calidad": None,
                 "ultima_sonda": None, "confiabilidad": 0.8, "ttft_p50_ms": 1500.0,
                 "latencia_p50_ms": None, "en_cooldown_hasta": 0.0, "tools": True,
                 "vision": False, "contexto": 128000},
            ],
        }}}},
        401: {"description": "Missing or invalid API key."},
    },
}


# --- GET /v1/uso ---------------------------------------------------------

USO_DOCS = {
    "tags": ["Diagnostics"],
    "summary": "Today's paid-usage counter for the calling key",
    "description": (
        "How many requests the calling key spent against the paid fallback "
        "(MiniMax) today (UTC), against its daily allowance "
        "(`TOPE_PAGO_DIARIO`). Counts every **billable** attempt, not only "
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
                 "dia": "2026-08-17", "pago_hoy": 0, "tope": 200}}}},
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
        "- **`degradado`** (HTTP `503`) -- no free route is alive, but the "
        "paid route is. Every request is now either failing or being billed; "
        "find out why the free tier is down (`GET /v1/ranking`, provider "
        "status) before the daily paid allowance runs out for every key.\n"
        "- **`caido`** (HTTP `503`) -- nothing is alive. `POST "
        "/v1/chat/completions` will return `503` for every request.\n\n"
        "A route needs **two consecutive failed health probes** (no success "
        "in between) to be counted as dead here -- a single failed probe is "
        "not enough, to avoid one transient blip (made more likely by the "
        "fact that a client's own traffic can indirectly trigger on-demand "
        "probes) flipping this to `caido` and restarting the container over "
        "nothing. A real outage still fails its second probe just as fast as "
        "its first, so genuine detection is not slowed down."
    ),
    "responses": {
        200: {"description": "At least one free route is alive.",
             "content": {"application/json": {"example": {
                 "estado": "ok", "rutas_activas": 18, "rutas_libres": 17,
                 "gratis_libres": 17}}}},
        503: {"description": "`degradado` (only the paid route is alive) or "
                            "`caido` (nothing is alive).",
             "content": {"application/json": {"examples": {
                 "degradado": {"summary": "Only the paid route is alive",
                              "value": {"estado": "degradado", "rutas_activas": 18,
                                       "rutas_libres": 1, "gratis_libres": 0}},
                 "caido": {"summary": "Nothing is alive",
                          "value": {"estado": "caido", "rutas_activas": 18,
                                   "rutas_libres": 0, "gratis_libres": 0}},
             }}}},
    },
}
