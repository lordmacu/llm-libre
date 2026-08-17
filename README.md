# llm-libre

Un gateway que expone los modelos de LLM **gratis** de varios proveedores
(chatgpt-proxy, Kilo y OpenRouter) detrás de un único contrato compatible con
OpenAI, con selección automática del mejor modelo disponible según un ranking
propio (calidad medida, confiabilidad y latencia) y un escalón de pago
(MiniMax) como último recurso cuando todo lo gratis está caído. Cualquier
cliente que hable el protocolo de OpenAI lo usa sin librería propia,
cambiando solo `base_url` y `api_key`.

**Vocabulario, para no confundirlo:** *tier* es `gratis` \| `pago` (si cuesta
plata). *Perfil* es `rapido` \| `balanceado` \| `potente` (qué prefiere la
petición). **`prioridad` es un tercer concepto, aparte de los dos** — el
orden manual en que el router prueba los proveedores antes de mirar puntaje
(ver "Cómo decide" más abajo). Nunca se pisan entre sí: una ruta de pago con
`prioridad: 0` sigue yendo siempre al final, plata manda sobre orden manual.

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
| `GET /v1/ranking` | Puntaje de cada ruta (con `prioridad`) y sus componentes desglosados, **ordenado con la misma clave que usa el router** — cooldown incluido: una ruta castigada (`en_cooldown_hasta` en la fila) va al final, aunque puntúe mejor que todas — para auditar por qué el router eligió lo que eligió, sin que la fila de arriba contradiga a `X-Ruta-Usada` |
| `GET /v1/uso` | Consumo de pago del día para la llave que llama, contra su tope diario |
| `GET /health` | Honesto: `ok` solo si hay al menos una ruta gratis viva; `degradado` si solo queda pago; `caido` si no hay nada servible. No requiere llave |

## Configuración

Variables de entorno (ver `.env.example`):

| Variable | Default | Qué es |
|---|---|---|
| `LLM_LIBRE_API_KEYS` | *(sin default — obligatoria)* | Llaves que aceptan los clientes, separadas por coma. El proceso **no arranca** si falta o queda vacía: ver más abajo |
| `CHATGPT_PROXY_URL` | `http://127.0.0.1:8888/v1` (el default del YAML) | URL de `chatgpt-proxy` (servicio propio, se despliega en `blog`). Sin credenciales — solo la dirección, que todavía no está fija, por eso es configurable por entorno en vez de estar cableada en `proveedores.yaml`. Idealmente incluye el `/v1` (sus rutas reales son `/v1/chat/completions` y `/v1/models`); si se pone **solo el host** (sin ninguna ruta), el `/v1` se agrega solo, con un aviso en el log. Si en cambio se pone **una ruta propia** (p.ej. un mount de reverse proxy, `.../v2`), esa ruta se respeta tal cual — no se le pisa nada, solo se avisa si no coincide con `/v1` por si fue sin querer |
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

- **Orden de prioridad entre proveedores gratis:** `chatgpt` (`prioridad: 0`)
  se prueba antes que `kilo`/`openrouter` (`prioridad: 1`). Es un servicio
  propio, así que se le da preferencia sobre terceros — pero sigue siendo
  `tier: gratis`, no consume el tope de pago diario. `minimax` (`pago`,
  `prioridad: 2`) sigue yendo **siempre al final**, sin importar su
  `prioridad`: ese número ordena dentro de un mismo `tier`, nunca decide
  entre `gratis` y `pago` (ver la nota de vocabulario más arriba).
- **`chatgpt` se sirve con `tools: false`, y sigue siendo obligatorio.** El
  backend anónimo dejó de devolver `HTTP 500` al mandarle `tools` (eso
  cambió), pero sigue sin soportar *function calling*: con
  `tool_choice: "required"` devuelve `tool_calls: None` y prosa en texto
  plano. Es "tools avanzados" (reservas, shopping, widgets, canvas) lo que
  sí soporta, no lo que la capacidad `tools` de este gateway significa. El
  comportamiento nuevo es **más** peligroso que el 500 viejo — un 500 fallaba
  honesto y disparaba failover; devolver prosa en silencio le entregaría
  texto a un cliente agentic que espera una llamada estructurada — así que
  esta declaración es la única barrera. Una petición con `tools` (o
  `auto:tools` / `x_requiere: ["tools"]`) descarta automáticamente las rutas
  de `chatgpt` y cae al siguiente proveedor gratis que sí las soporte (Kilo u
  OpenRouter); recién si esos también fallan, al escalón de pago.
- `chatgpt-proxy` filtra el modo "canvas" de ChatGPT al `content`, con marcas
  de la forma `:::palabra{...atributos...}` … `:::`. El gateway las
  desenvuelve (en bloque y en streaming) conservando el texto de adentro —
  a diferencia de `<think>`, ahí ES la respuesta, no algo para descartar.
  **Es una declaración por proveedor** (`desenvuelve_canvas` en
  `proveedores.yaml`, apagada por defecto), no algo universal:
  `:::nota{...}` / `:::tip{...}` es también sintaxis Docusaurus/MDX estándar,
  y aplicar el desenvuelto a ciegas le arrancaría esas marcas a una respuesta
  de documentación legítima de Kilo u OpenRouter. Solo `chatgpt` lo declara.
- **El catálogo de los proveedores gratis se descubre siempre desde su propio
  `/models`, nunca se hardcodea** — Kilo, OpenRouter **y también `chatgpt`**:
  así un modelo que cambia de id, desaparece o aparece se detecta solo, sin
  tocar código ni `proveedores.yaml`. Hay tres patrones en el registro, según
  qué trae el `/models` de cada uno:
  - Kilo / OpenRouter: ids **y** capacidades, los dos descubiertos.
  - `minimax` (pago): ni ids ni capacidades — su `/models` real solo trae
    `id`/`created`/`owned_by` — así que los dos se declaran a mano
    (`modelos_fijos`).
  - `chatgpt`: ids **descubiertos** (su `/v1/models` sí es dinámico, con
    caché y TTL contra el backend real de ChatGPT), pero **capacidades
    declaradas** (`capacidades_por_defecto`) — su catálogo nunca trae
    metadatos de capacidad, solo `id`/`description`. Es un mecanismo
    general, no algo especial de `chatgpt`: cualquier proveedor futuro con
    un `/models` igual de desnudo lo puede usar sin tocar código.
  - Dos entradas se filtran del descubrimiento de `chatgpt`, las dos por lo
    que la propia respuesta dice de sí misma, nunca por una lista de ids:
    los alias legacy que el proxy agrega (`gpt-4o`, `gpt-4o-mini`, `gpt-4`,
    `gpt-3.5-turbo`) traen `description: "Alias → <target>"`; y `auto`,
    reservado por el propio `interpretar_pedido` de llm-libre (colisiona con
    su alias `auto`), se descarta como id reservado.
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
- **Una ruta que falla repetidas veces seguidas (sin ningún éxito en el
  medio) entra en cooldown, no solo cuando el proveedor devuelve `429`.**
  Antes, un `500`, un timeout o un error de red no dejaban nunca un
  cooldown: una ruta rota o **colgada** se seguía probando en cada pedido,
  adelante de las sanas si tenía mejor `prioridad`, indefinidamente — con
  el timeout por defecto (90 s) eso son hasta 7,5 minutos por pedido en la
  cadena más larga, y `/health` sigue en `ok` mientras quede una ruta viva.
  Un fallo aislado no castiga (evita sacar una ruta sana por un hiccup); al
  tercer fallo seguido, sí, con el mismo backoff exponencial que ya usa el
  `429`. **Un `4xx` (que no sea `429`, `408` ni `425`) nunca cuenta para
  esto**: es un error determinista del *cliente* — payload inválido, un
  parámetro que el proveedor no soporta — que el proveedor le devolvería a
  cualquiera que mande ese mismo pedido. Contarlo convertiría el error de un
  cliente en un apagón para todos los demás (verificado: tres pedidos
  malformados seguidos bastan para dejar las cinco rutas en cooldown si se
  cuenta el `4xx`); un `400` solo debe perjudicar a quien lo mandó, como
  siempre. (`408`/`425` quedan afuera de esta excepción a propósito: por
  espíritu son más "el upstream se colgó" o una nuance de protocolo que un
  payload inválido, así que sí cuentan como fallo real — un proveedor que
  usa `408` cuando se cuelga tiene que poder entrar en cooldown igual.) Un
  proveedor puede además declarar su propio `timeout_s` en
  `proveedores.yaml` (default: el global, 90 s) para acotar el peor caso de
  uno que se sepa lento, sin bajarle el timeout a todos — aplica igual al
  camino síncrono y al de streaming.
- **Ese mismo `4xx` tampoco cuenta para la confiabilidad medida, ni por lo
  tanto para `/health` ni para `/v1/ranking`.** El evento se sigue
  guardando (queda diagnosticable — un operador viendo la tabla `eventos`
  ve los `4xx` igual), pero se marca `es_error_cliente` y la confiabilidad
  lo excluye por completo de su ventana: ni cuenta como fallo ni ocupa un
  lugar entre las últimas observaciones. Antes solo se sacó del *contador
  de cooldown*, pero seguía escribiéndose como fallo común, y eso alimenta
  la confiabilidad que `/health` usa para su piso — verificado: 26 pedidos
  malformados seguidos de una sola llave bastan para tirar la confiabilidad
  de *todas* las rutas por el piso, con `/health` en `caido` mientras una
  llave distinta con un pedido válido sigue recibiendo `200`. Esto es
  **peor** que el 503 anterior en el despliegue real: Coolify usa `/health`
  como *health check* y reinicia el contenedor cuando falla, pero la tabla
  `eventos` vive en el volumen persistente `/datos` — un proceso nuevo
  contra la misma base sigue viendo los mismos 26 fallos, así que el
  reinicio no lo arregla. Un `500` sigue tirando `/health` y bajando la
  posición en `/v1/ranking` exactamente como antes.
