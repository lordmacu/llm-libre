# Local access

How to reach the running gateway on the server, and from your own machine. This
is the operational how-to; for what each endpoint means see the interactive docs
at `/docs`, and for the routing behaviour behind them see
[`routing-and-ranking.md`](routing-and-ranking.md).

The service runs on the **`blog`** host as a Coolify Docker container, listening
on port **8102** (it maps `8102 -> 8101` inside the container). Every `/v1/*`
route that returns data requires an API key; `/health` and `/v1/assets/{id}` do
not.

## Getting the API key

The key is not stored in the repo. Read it from the running container, so it
never has to be pasted anywhere:

```bash
ssh blog 'C=$(sudo -n docker ps -q --filter name=nhh7 | head -1); sudo -n docker exec "$C" printenv LLM_LIBRE_API_KEYS'
```

It begins with `llmlibre_`. `LLM_LIBRE_API_KEYS` may hold several keys separated
by commas; any one of them works. In the examples below, substitute it for
`YOUR_KEY`.

## From the server (over SSH)

Port 8102 listens on the server's loopback only, so these run **on `blog`**:

```bash
# health -- no key
ssh blog 'curl -s http://127.0.0.1:8102/health'
```

```bash
# chat -- with key
ssh blog 'curl -s -H "Authorization: Bearer YOUR_KEY" -H "Content-Type: application/json" -d "{\"model\":\"auto\",\"messages\":[{\"role\":\"user\",\"content\":\"hola\"}]}" http://127.0.0.1:8102/v1/chat/completions'
```

A successful chat response carries the routing headers `X-Route-Used`,
`X-Tier` and `X-Attempts` -- add `-D -` to `curl` to see them.

## From your own machine (SSH tunnel)

Because 8102 is loopback-only on the server, forward it to your laptop:

```bash
ssh -L 8102:127.0.0.1:8102 blog
```

Leave that running. In another terminal, `http://127.0.0.1:8102` now behaves as
if the gateway were local:

```bash
curl -s -H "Authorization: Bearer YOUR_KEY" -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"hola"}]}' \
  http://127.0.0.1:8102/v1/chat/completions
```

## From anywhere (public URL)

The same gateway is published at **`https://llm.comparadorinternet.co`** with the
same key and the same endpoints -- no SSH, no tunnel. Use this from an app or an
SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="https://llm.comparadorinternet.co/v1", api_key="YOUR_KEY")
print(client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "hola"}]).choices[0].message.content)
```

## Interactive docs, OpenAPI and Postman

Swagger UI and ReDoc are served live by the gateway -- try every endpoint from
the browser, no import needed:

| | Public URL | Over the SSH tunnel |
|---|---|---|
| Swagger UI | https://llm.comparadorinternet.co/docs | http://127.0.0.1:8102/docs |
| ReDoc | https://llm.comparadorinternet.co/redoc | http://127.0.0.1:8102/redoc |
| OpenAPI JSON | https://llm.comparadorinternet.co/openapi.json | http://127.0.0.1:8102/openapi.json |

The Swagger page has an **Authorize** button; paste `YOUR_KEY` there once and
every request on the page carries it.

**Postman.** Two ways in:

- **Import the OpenAPI URL** directly -- in Postman, *Import -> Link* and paste
  `https://llm.comparadorinternet.co/openapi.json`. This always matches what the
  service actually serves, since that document is generated from the running app.
- **Import the curated collection** shipped in the repo,
  [`llm-libre.postman_collection.json`](../llm-libre.postman_collection.json)
  (repo root). It has worked examples grouped into folders -- chat, images,
  vision, diagnostics, error paths. The repo is private, so there is no public
  download link: use the file from your checkout. After importing, set the
  collection variables `base_url` (e.g. `https://llm.comparadorinternet.co`, no
  trailing slash, no `/v1`) and `api_key`.

## Endpoints

| Call | Key | What it does |
|---|---|---|
| `GET /health` | no | Status: `ok` / `degraded` / `down`, plus route counts |
| `GET /v1/models` | yes | The normalised catalogue plus the `auto*` aliases |
| `GET /v1/ranking` | yes | Every route scored, in the router's real order |
| `GET /v1/usage` | yes | The calling key's paid consumption for the day |
| `POST /v1/chat/completions` | yes | Chat; send `image_url` parts for vision |
| `POST /v1/images/generations` | yes | Generate an image (routed only to image-capable routes) |
| `GET /v1/assets/{id}` | no | A re-hosted generated image; works in an `<img>` tag |

### Model aliases

`model` takes a real id or one of these:

```
auto            balanced
auto:fast       prioritise latency
auto:strong     prioritise measured quality
auto:tools      require function calling
auto:vision     require image input
```

Sending an `image_url` in a chat message sets `needs_vision` automatically -- you
do not have to use `auto:vision`. The gateway then routes only to a
vision-capable route and never tries the others.

## Notes

- Both header styles are accepted: `Authorization: Bearer <key>` (what an OpenAI
  SDK sends) and `X-API-Key: <key>`. If both are present, `X-API-Key` wins.
- The container name changes on each redeploy; the `--filter name=nhh7` above
  matches on the stable Coolify app id prefix, so the key command keeps working.
- No key on a data route returns `401`; a request no route can satisfy returns
  `400`; every capable route being down or in cooldown returns `503`.
