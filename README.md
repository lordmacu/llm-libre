# llm-libre

Un gateway que expone los modelos de LLM **gratis** de varios proveedores
(hoy Kilo y OpenRouter) detrás de un único contrato compatible con OpenAI, con
selección automática del mejor modelo disponible según un ranking propio
(calidad medida, confiabilidad y latencia) y un escalón de pago (MiniMax)
como último recurso cuando todo lo gratis está caído. Cualquier cliente que
hable el protocolo de OpenAI lo usa sin librería propia, cambiando solo
`base_url` y `api_key`.

## Uso rápido

Con el SDK de OpenAI (Python):

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<dominio>/v1",
    api_key="<una-de-las-LLM_LIBRE_API_KEYS>",
)

resp = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "di hola"}],
)
print(resp.choices[0].message.content)
```

Con `curl`:

```bash
curl -s -H "X-API-Key: <una-de-las-LLM_LIBRE_API_KEYS>" \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"di hola"}]}' \
  https://<dominio>/v1/chat/completions
```

`base_url` siempre termina en `/v1`. La llave se acepta por **cualquiera**
de dos cabeceras:

- `Authorization: Bearer <llave>` — la que manda, sin configuración extra,
  el parámetro `api_key` de cualquier SDK de OpenAI. Es la que usa el
  snippet de arriba.
- `X-API-Key: <llave>` — la convención que ya usa `arkiv-api`, el gateway
  hermano. Sigue siendo válida para quien ya la usaba, y **gana** si una
  petición llega con las dos cabeceras a la vez.

## Alias `auto*`

El campo `model` acepta un id real o uno de estos alias virtuales. El id real
es el del **modelo**, sin prefijo de proveedor — tal cual lo lista
`GET /v1/models` (por ejemplo `nvidia/nemotron-3-super-120b-a12b:free`, donde
`nvidia/` es parte del id del modelo, no el proveedor). Es a propósito: el
mismo modelo suele existir en varios proveedores, y pedirlo por su id deja que
el gateway haga failover entre ellos. Quién lo sirvió se ve después, en la
cabecera `X-Ruta-Usada` (`kilo/nvidia/nemotron-3-super-120b-a12b:free`), que
sí lleva el proveedor adelante.

| Alias | Qué selecciona |
|---|---|
| `auto` | Perfil **balanceado**: calidad, confiabilidad y latencia pesan igual |
| `auto:rapido` | Perfil **rápido**: prioriza latencia baja, resigna algo de calidad |
| `auto:potente` | Perfil **potente**: prioriza calidad medida, resigna algo de latencia |
| `auto:tools` | Perfil balanceado + exige que la ruta soporte function calling |
| `auto:vision` | Perfil balanceado + exige que la ruta soporte entrada de imágenes |

Con cualquier alias, si la ruta elegida falla el gateway reintenta con la
siguiente de la lista ordenada (failover automático, incluido el salto a
pago si hace falta y está permitido).

## Extensiones `x_*`

Van en el mismo cuerpo JSON que el resto de la petición; un SDK ajeno que no
las conoce las ignora sin romperse:

| Campo | Qué hace |
|---|---|
| `x_requiere` | Lista de capacidades obligatorias, p.ej. `["tools", "vision"]` — equivalente a pedirlas por alias |
| `x_min_contexto` | Ventana de contexto mínima en tokens; descarta rutas por debajo |
| `x_permitir_pago` | `false` desactiva el escalón de pago (MiniMax) para esa petición puntual |
| `x_crudo` | `true` desactiva el recorte de razonamiento (`<think>`, etc.) y devuelve el `content` tal cual lo mandó el proveedor |

## `x_razonamiento` en la respuesta

Varios modelos —gratis y de pago— escupen su cadena de pensamiento dentro de
`content`, entre etiquetas `<think>` / `<thinking>` / `<reasoning>`. El gateway
la separa: el `content` que ve el cliente queda limpio y el bloque recortado
vuelve en un campo de nivel superior, `x_razonamiento`, que cualquier SDK de
OpenAI ignora sin romperse.

```json
{
  "choices": [{"message": {"role": "assistant", "content": "La respuesta es 4."}}],
  "x_razonamiento": "2+2 son 4"
}
```

El campo solo aparece si de verdad hubo algo que recortar. **En streaming no
va**: meterlo ahí obligaría a emitir un evento SSE no estándar, justo lo que
el contrato evita para no romper el parseo de los SDK. Un cliente que streamea
y quiere el razonamiento pide `x_crudo: true` y lo recibe dentro del `content`,
tal cual lo mandó el proveedor.

## Cabeceras de respuesta

Toda respuesta **no-streaming** de `/v1/chat/completions` lleva:

- `X-Ruta-Usada`: `<proveedor>/<modelo_id>` de la ruta que efectivamente sirvió
- `X-Tier`: `gratis` o `pago`
- `X-Intentos`: cuántas rutas se intentaron antes de responder (o de agotarse)

**En streaming (`stream: true`) estas cabeceras NO viajan.** Las cabeceras
HTTP se mandan antes del cuerpo, y en ese momento la cadena de failover
todavía no se resolvió — no se sabe qué ruta va a terminar sirviendo. El
gasto de pago igual queda registrado y visible por otra vía: consultar
`GET /v1/uso`.

## Endpoints

| Endpoint | Qué hace |
|---|---|
| `POST /v1/chat/completions` | El contrato de chat de OpenAI, con `stream: true` opcional y las extensiones `x_*` de arriba |
| `GET /v1/models` | El catálogo normalizado (formato OpenAI) más los alias `auto*` |
| `GET /v1/ranking` | Puntaje de cada ruta con sus componentes desglosados y la fecha de su última sonda — para auditar por qué el router eligió lo que eligió |
| `GET /v1/uso` | Consumo de pago del día para la llave que llama, contra su tope diario |
| `GET /health` | Honesto: `ok` solo si hay al menos una ruta gratis viva; `degradado` si solo queda pago; `caido` si no hay nada servible. No requiere llave |

## Configuración

Variables de entorno (ver `.env.example`):

| Variable | Default | Qué es |
|---|---|---|
| `LLM_LIBRE_API_KEYS` | *(sin default — obligatoria)* | Llaves que aceptan los clientes, separadas por coma. El proceso **no arranca** si falta o queda vacía: ver más abajo |
| `KILO_API_KEY` | *(sin definir)* | Opcional. **Dejar SIN DEFINIR**, no en blanco — ver nota abajo |
| `OPENROUTER_API_KEY` | *(sin definir)* | Llave de OpenRouter (su tier gratis sí exige llave para completions, aunque `/models` sea público) |
| `MINIMAX_API_KEY` | *(sin definir)* | Llave del proveedor de pago (escalón de fallback) |
| `SONDEO_SALUD_HORAS` | `5` | Cada cuántas horas se sondea la salud de cada ruta |
| `SONDEO_CALIDAD_CADA_N_CICLOS` | `5` | Cada cuántos ciclos de sondeo se corre además la batería de calidad |
| `TOPE_PAGO_DIARIO` | `200` | Tope diario de peticiones al escalón de pago, por llave |
| `LIMITE_POR_MINUTO` | `60` | Límite de peticiones por minuto, por llave |
| `RUTA_DB` | `/datos/llm-libre.sqlite3` | Ruta del archivo SQLite (catálogo + telemetría) |
| `PROVEEDORES_YAML` | `proveedores.yaml` | Ruta del registro de proveedores |

**`KILO_API_KEY` debe quedar sin definir**, no vacía-pero-presente: el tier
anónimo de Kilo depende de que la petición viaje sin ninguna cabecera
`Authorization`. En Coolify eso significa simplemente no crear esa variable
en la UI, no crearla con valor vacío.

**`LLM_LIBRE_API_KEYS` sin configurar hace que el proceso falle al arrancar**,
a propósito, con un mensaje que nombra la variable: la alternativa —dejarlo
arrancar igual— produciría un contenedor que `/health` reporta `ok` mientras
rechaza el 100% de las peticiones con 401 para cualquier llamador, sin nada
en los logs que distinga "no hay llaves configuradas" de "llave incorrecta".
Mejor un contenedor que no arranca con una razón clara.

## Despliegue

Se despliega con **Coolify, por git push a `main`** — Coolify redespliega
solo en cada push, construyendo el `Dockerfile` de la raíz. **No es por
rsync.** `docker-compose.yml` solo sirve para levantarlo en local durante
desarrollo.

Necesita un **volumen persistente montado en `/datos`**: sin él, el archivo
SQLite (catálogo de rutas + toda la telemetría de sondeo) se destruye en cada
redeploy y el ranking —que tarda días en construirse porque la calidad se
sondea aproximadamente una vez al día— vuelve a cero.

## Cómo decide

- **El catálogo de los proveedores gratis (Kilo, OpenRouter) se descubre
  siempre desde su propio `/models`, nunca se hardcodea**: así un modelo que
  cambia de id o desaparece se detecta solo, sin tocar código. La excepción
  es el proveedor de pago (MiniMax): su `/models` real solo devuelve
  `id`/`created`/`owned_by`, sin metadatos de capacidades, así que su único
  modelo se declara a mano en `proveedores.yaml` (`modelos_fijos`).
- Cada ruta (proveedor + modelo) se sondea por **salud** cada
  `SONDEO_SALUD_HORAS` (default 5 h) y, de las rutas gratis vivas, por
  **calidad** cada `SONDEO_CALIDAD_CADA_N_CICLOS` ciclos (default 5, o sea
  aproximadamente una vez al día) con una batería de casos verificables por
  código (JSON válido, tool call correcto, formato pedido, aritmética,
  idioma) — sin juez-LLM.
- El catálogo descarta lo que el proveedor **describe** como algo que no es un
  modelo de chat de propósito general —guardrails y clasificadores de
  seguridad, rerankers, modelos de embeddings— y los *meta-routers*, que no son
  un modelo sino un sorteo entre otros modelos. Se decide leyendo los campos
  `name` y `description` del propio `/models`, nunca por una lista de ids: una
  lista negra de ids se pudriría igual que los ids cableados que este gateway
  existe para reemplazar.
- El ranking combina calidad medida, confiabilidad reciente (éxitos sobre
  sondas + tráfico real) y latencia (p50 de time-to-first-token), con pesos
  que cambian según el perfil pedido: `rapido` pondera la latencia por
  encima de la calidad, `potente` al revés, `balanceado` reparte parejo. La
  ventana de contexto (`x_min_contexto`) es un filtro previo, no entra en el
  puntaje.
- **Una ruta que todavía no pasó por la batería de calidad va después de las
  que sí**, aunque su valor neutro puntúe más alto: ese 0.6 es un supuesto, no
  una medición, y `/v1/ranking` lo dice (`calidad: null`, `calidad_medida:
  false`). Sigue en la cadena de intentos para que llegue a medirse alguna vez.
- `ttft_p50_ms` es **tiempo hasta el primer token** y solo lo mide el camino de
  streaming, que es el único que puede. El camino no-streaming (y la sonda de
  salud) reportan su round-trip completo aparte, en `latencia_p50_ms`, que no
  entra en el puntaje: son dos magnitudes distintas y promediarlas juntas daba
  un número sin significado.
- Las rutas de pago siempre van al final de la cadena de intentos, y solo se
  usan si se agotaron las gratis, la llave no superó su tope diario y la
  petición no trae `x_permitir_pago: false`.
