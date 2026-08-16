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
    api_key="no-se-usa",  # el SDK exige un valor no vacio; llm-libre no lo lee
    default_headers={"X-API-Key": "<una-de-las-LLM_LIBRE_API_KEYS>"},
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

`base_url` siempre termina en `/v1`. La llave va en la cabecera `X-API-Key`,
**no** en `Authorization`: `llm-libre` no lee `Authorization` en absoluto, así
que el parámetro `api_key` del SDK de OpenAI no autentica nada por sí solo
— por eso el snippet la manda aparte, vía `default_headers`.

## Alias `auto*`

El campo `model` acepta un id real (`kilo/nvidia/nemotron-3-super-120b-a12b:free`,
por ejemplo) o uno de estos alias virtuales:

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
| `GET /v1/ranking` | Puntaje de cada ruta con sus componentes desglosados (calidad, confiabilidad, latencia, cooldown) — para auditar por qué el router eligió lo que eligió |
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
- El ranking combina calidad medida, confiabilidad reciente (éxitos sobre
  sondas + tráfico real) y latencia (p50 de time-to-first-token), con pesos
  que cambian según el perfil pedido: `rapido` pondera la latencia por
  encima de la calidad, `potente` al revés, `balanceado` reparte parejo. La
  ventana de contexto (`x_min_contexto`) es un filtro previo, no entra en el
  puntaje.
- Las rutas de pago siempre van al final de la cadena de intentos, y solo se
  usan si se agotaron las gratis, la llave no superó su tope diario y la
  petición no trae `x_permitir_pago: false`.
