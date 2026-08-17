# llm-libre — router de modelos gratis con ranking propio

**Fecha:** 2026-08-16
**Estado:** diseño aprobado, sin implementar

## 1. Propósito

Un servicio en `blog` que expone los modelos de LLM **gratis** de varios proveedores
detrás de un solo contrato compatible con OpenAI, para que cualquier app interna,
contenedor o experimento los use cambiando únicamente `base_url`.

Resuelve tres problemas concretos:

1. **Los nombres de modelo se pudren.** Descubierto el 2026-08-16: dos de los tres ids
   que trae SlimePet cableados (`poolside/laguna-m.1:free`, `poolside/laguna-xs.2:free`)
   ya devuelven `404 model_not_found`, y el default de OpenRouter
   (`minimax/minimax-m2.5:free`) tampoco existe. Cablear ids garantiza romperse.
   El catálogo se descubre desde `/models`, siempre.
2. **"El mejor modelo gratis" no se sabe sin medir.** El servicio lo mide y mantiene
   un ranking propio.
3. **Cada consumidor repite la misma plomería** (claves, failover, reintentos).

## 2. Decisiones tomadas

Estas ya se discutieron. No re-litigar sin motivo nuevo.

| Decisión | Elección | Razón |
|---|---|---|
| Proveedores | chatgpt-proxy + Kilo + OpenRouter, con registro declarativo abierto | Sumar Groq/Cerebras después debe ser una entrada de config, no código |
| Auto-selección | Capacidades requeridas + salud medida, con perfil opcional y prioridad manual | Filtrar por datos reales del catálogo, no por listas curadas a mano |

**Nota de vocabulario, para que el código no lo confunda:** *tier* es siempre el tipo de
proveedor (`gratis` \| `pago`). *Perfil* es siempre lo que prefiere la petición
(`rapido` \| `balanceado` \| `potente`). **`prioridad` (Task 13) es un TERCER concepto,
separado de los dos**: el orden manual en que el router prueba los proveedores antes de
mirar puntaje (p.ej. un proveedor propio antes que terceros). Nunca usar una palabra por
otra, y nunca dejar que `prioridad` decida lo que decide `tier`: una ruta de pago con
`prioridad: 0` sigue yendo siempre al final (§7).
| Despliegue | `blog`, docker, público por cloudflared con `X-API-Key` | Mismo patrón que `arkiv-api`; hay consumidores fuera de la LAN |
| Escape a pago | **Sí**, MiniMax como último escalón, con tope diario por llave | Decisión explícita del usuario (2026-08-16), revirtiendo "solo gratis" |
| Ranking | Sonda de salud + batería de calidad verificable por código | Puntaje auditable, sin juez-LLM ruidoso ni doble gasto de cuota |
| Cadencia de sondeo | Salud cada 5 h; calidad cada 5º ciclo (≈ diaria) | ~100 peticiones/día de cuota gastadas en medir. Configurable |
| Stack | Python + FastAPI + httpx + SQLite | Mismo stack y camino de deploy que `arkiv-api` |

## 3. Hallazgos verificados (2026-08-16)

Medidos contra las APIs reales, no deducidos de documentación.

**Kilo Gateway acepta peticiones anónimas.** `POST https://api.kilo.ai/api/gateway/chat/completions`
sin cabecera `Authorization` devuelve `200`. Verificado en cuatro modelos, con **chat,
function calling y streaming SSE** funcionando. Una clave de Kilo es opcional y solo sube
los límites.

**`/models` trae los metadatos necesarios.** Ambos proveedores exponen el mismo esquema:

- `supported_parameters` incluye `tools` / `tool_choice` → detección de function calling
- `architecture.input_modalities` contiene `image` → detección de visión
- `context_length` y `top_provider.max_completion_tokens`
- `pricing.prompt == "0"` → **detección de gratis más confiable que el sufijo `:free`**
- Kilo agrega `isFree` y `preferredIndex` (su propio orden de preferencia)

**Tamaño del pozo:** Kilo 361 modelos totales, 14 gratis (11 con tools, 6 multimodales).
OpenRouter 413 totales, 19 gratis (16 con tools, 9 multimodales).

**Filtrar por precio no alcanza.** `pricing.prompt == "0"` también captura
`google/lyria-3-pro-preview` y `google/lyria-3-clip-preview`, que son modelos de **música**.
Hay que exigir además modalidad de salida texto.

**Los catálogos se solapan.** `poolside/laguna-s-2.1:free`, los `nvidia/nemotron-*` y
`dots-studio/dots-3-note-preview:free` están en los dos proveedores. Esto es lo que
justifica modelar **rutas** en vez de modelos (§5).

**MiniMax expone tres dialectos.** Sondeado sin clave, distinguiendo 401 (existe) de 404:

| Endpoint | Dialecto | Auth que pide |
|---|---|---|
| `/v1/chat/completions` | OpenAI | `Authorization` |
| `/anthropic/v1/messages` | Anthropic Messages | `X-Api-Key` |
| `/v1/text/chatcompletion_v2` | nativo MiniMax | `Authorization` |

Los 7 contenedores de Coolify en `blog` (bots de WhatsApp) usan hoy el **dialecto Anthropic**
(`LLM_BASE_URL=https://api.minimax.io/anthropic`, `LLM_MODEL=MiniMax-M3`), cada uno con su
propia copia de la clave. **No existe un MiniMax local**: llaman a la nube.

**El endpoint OpenAI de MiniMax sirve para todo lo que necesitamos** (verificado con la clave
real): `chat`, `function calling` (devuelve `tool_calls` en formato OpenAI) y `stream` SSE.
Por lo tanto **no hace falta ningún adaptador Anthropic**: MiniMax entra como un proveedor
más de dialecto `openai`.

**Varios modelos filtran su razonamiento dentro de `content`.** MiniMax-M3 devuelve
`<think>...</think>` en línea; `nvidia/nemotron-3.5-lightning:free` responde con
`"Here's a thinking process: ..."`. Un consumidor que pide "responde solo: hola" recibe el
monólogo entero. Esto es transversal a proveedores y tiers, así que se resuelve en el
gateway (§6.1), no en cada app.

Las 7 copias de la clave de MiniMax en los contenedores de Coolify son **idénticas**
(mismo SHA-256, 125 caracteres), así que una sola sirve para todo.

**chatgpt-proxy (Task 13, 2026-08-16), verificado contra el proxy real.** Servicio
propio que expone ChatGPT por su flujo anónimo (sin credenciales — `Authorization`
solo separa sesiones, no autentica), contrato compatible con OpenAI
(`/v1/chat/completions`, `/v1/models`, `/health`). Chat sin streaming y streaming
(forma real de OpenAI: `finish_reason` hermano de `delta`, no adentro) **funcionan**.
`temperature`/`max_tokens`/etc. se aceptan y se ignoran. **Filtra el modo "canvas" de
ChatGPT al `content`**, con marcas `:::palabra{...}` … `:::` que envuelven la respuesta
real (no algo para descartar, a diferencia de `<think>`) — el gateway las desenvuelve,
en bloque y en streaming (§6.1).

**`tools` (revisado de nuevo en la revisión de la review): ya NO devuelve HTTP 500,
pero sigue sin soportar function calling.** El usuario reportó "ya tenemos habilitados
los tools" en el proxy; verificado ejecutándolo: con `tool_choice: "required"` sigue
devolviendo `tool_calls: None` y prosa. El propio docstring del proxy distingue dos
cosas — *"Tools avanzados: reservas, shopping, widgets, canvas"* (SÍ soportado) vs.
*"Function calling / tool_calls en respuesta: no soportado por el backend anónimo"*
(NO) — y lo que la capacidad `tools` de llm-libre significa es la segunda. `tools:
false` sigue OBLIGATORIO, y el comportamiento nuevo es **más** peligroso que el 500
viejo: un 500 fallaba honesto y disparaba failover; devolver prosa en silencio le
entregaría texto a un cliente agentic que espera una llamada estructurada, sin que nada
avise del error. Esta declaración es hoy la única barrera.

**Follow-up (mismo día): `/v1/models` pasó a ser dinámico.** El usuario actualizó
chatgpt-proxy para que consulte el catálogo real de ChatGPT (con caché y TTL,
fallback a la caché vieja si el backend falla, `502` solo si no tiene nada) en vez de
una lista escrita a mano. Verificado en vivo: 10 entradas, solo con
`id`/`object`/`created`/`owned_by`/`description` — **sigue sin metadatos de
capacidad**. 5 son modelos reales (`gpt-5-5`, `gpt-5-6`, `gpt-5-3-mini`,
`gpt-5-5-mini`, `gpt-5-6-mini`), 4 son alias legacy que el proxy agrega para
compatibilidad (`description` empieza con `"Alias → <target>"` — p.ej.
`gpt-4o` → `"Alias → auto"`), y 1 es `auto` (`description: "Auto"`, sin el prefijo de
alias). Esto cambió el patrón de registro de `chatgpt`: pasa de "ids y capacidades
declarados a mano" (como MiniMax) a "ids descubiertos, capacidades declaradas" — un
tercer patrón, ver §4.

## 4. Arquitectura

Un contenedor. **Un solo worker de uvicorn**: mantiene coherente el estado en memoria
(cooldowns, límite por llave) sin necesitar Redis, y es lo apropiado para una máquina en swap.

| Módulo | Responsabilidad | Depende de |
|---|---|---|
| `proveedores` | Carga `proveedores.yaml`, resuelve claves desde el entorno | — |
| `catalogo` | Trae `/models`, normaliza, filtra, persiste | `proveedores` |
| `sondeo` | Mide salud y calidad de cada ruta | `catalogo`, `proxy` |
| `ranking` | Calcula puntajes desde el histórico. **Función pura, sin red** | datos |
| `router` | Pedido → lista ordenada de intentos. **Función pura, sin red** | `ranking` |
| `proxy` | Ejecuta la lista, maneja cooldowns, SSE en passthrough | `proveedores` |
| `api` | Endpoints FastAPI, auth, límites | todos |

`ranking` y `router` son puros a propósito: es la lógica que más va a costar afinar, y así
se prueba entera sin tocar internet.

### Registro de proveedores

Tres patrones, todos declarativos — ninguno cablea ids en código:

| Patrón | Ids | Capacidades | Quién |
|---|---|---|---|
| Todo descubierto | `/models` | `/models` | Kilo, OpenRouter |
| Todo declarado | `modelos_fijos` | `modelos_fijos` | MiniMax |
| Ids descubiertos, capacidades declaradas | `/models` | `capacidades_por_defecto` | chatgpt |

`capacidades_por_defecto` (follow-up de Task 13) es el mecanismo general para el
tercer patrón: cuando un proveedor lo declara, `catalogo.normalizar` aplica esas
capacidades a CADA id que descubra y saltea los chequeos de precio/modalidad de
salida — un proveedor que declara defaults está afirmando lo que su catálogo no
puede decir. Sigue siendo descubrimiento en el sentido que importa (§1): los IDS se
leen de `/models`, nunca se cablean. Cualquier proveedor futuro con un catálogo
igual de desnudo lo usa sin tocar código.

```yaml
proveedores:
  - id: chatgpt
    tier: gratis
    prioridad: 0                 # antes que todo lo demas gratis
    dialecto: openai
    base_url_env: CHATGPT_PROXY_URL   # la direccion real viene del entorno
    base_url: http://127.0.0.1:8888/v1   # default; OJO con el /v1 (ver nota)
    modelos_path: /models
    desenvuelve_canvas: true   # SOLO chatgpt -- ver la nota debajo
    # timeout_s: 20             # opcional; default None = el TIMEOUT_S global
    capacidades_por_defecto:
      tools: false   # OBLIGATORIO -- function calling no soportado (ver §3)
      vision: false
      contexto: 128000
      max_salida: 8192
  - id: kilo
    tier: gratis
    prioridad: 1
    dialecto: openai
    base_url: https://api.kilo.ai/api/gateway
    clave_env: KILO_API_KEY      # opcional: si falta, va anónimo
    modelos_path: /models
  - id: openrouter
    tier: gratis
    prioridad: 1
    dialecto: openai
    base_url: https://openrouter.ai/api/v1
    clave_env: OPENROUTER_API_KEY
    modelos_path: /models
    cabeceras_extra:
      HTTP-Referer: https://github.com/lordmacu/llm-libre
      X-Title: llm-libre
  - id: minimax
    tier: pago
    prioridad: 2
    dialecto: openai
    base_url: https://api.minimax.io/v1
    clave_env: MINIMAX_API_KEY
    modelos_fijos: [MiniMax-M3]   # no se descubre: es de pago y no se sondea
```

`prioridad` (Task 13, default `100` si no se declara) es un entero manual —
ver la nota de vocabulario en §2 — que el router usa ANTES del puntaje, pero
DESPUÉS del filtro `tier == "pago"` (§7): nunca puede comprarle a una ruta de
pago un lugar antes que lo gratis. `base_url_env`, si está declarada, permite
que la URL real venga del entorno (con el `base_url` del YAML como default) —
existe porque `chatgpt-proxy` se despliega en `blog` y su dirección todavía
no está fija.

**El `/v1` de `base_url` importa — y desde la revisión, se auto-corrige si falta (pero
solo si falta).** Las rutas reales de chatgpt-proxy son `/v1/chat/completions` y
`/v1/models` (verificado contra el código del proxy), no `/chat/completions` a secas
como Kilo/OpenRouter. `cliente.armar_peticion` siempre agrega `/chat/completions` sobre
`base_url` sin agregar `/v1` por su cuenta — mismo patrón que MiniMax (`base_url:
https://api.minimax.io/v1`) — así que `base_url` tiene que incluir el `/v1` ella
misma. Esto ya se arregló una vez del lado del YAML (Task 13 original tenía
`base_url` sin `/v1`); el mismo footgun sobrevivía del lado de `CHATGPT_PROXY_URL`,
que reemplaza TODO `base_url` y el operador tiene que acordarse de poner el sufijo él
mismo. `proveedores._resolver_base_url` normaliza, con una regla ajustada en una
segunda revisión (la primera versión era demasiado ansiosa: agregaba el sufijo sin
condición, así que `.../v2` — una ruta PROPIA del operador, p.ej. un mount de reverse
proxy — terminaba en `.../v2/v1/chat/completions`, sin escape). La regla final: si la
variable de entorno NO trae ninguna ruta propia (vacía o `/`, o sea el operador puso
nada más que el host), se le agrega el sufijo que el YAML declara como default, con un
`log.warning`. Si SÍ trae una ruta propia que no coincide con ese sufijo, se usa TAL
CUAL, sin modificar — solo se avisa, por si fue sin querer. Elegido sobre fallar al
arrancar porque hay una interpretación por default razonable (el sufijo que el propio
YAML declara) para el caso común (solo host), y para el caso con ruta propia no hay
ninguna interpretación segura salvo respetar lo que el operador escribió.

**LOW, corregido en la tercera revisión — pero solo a medias.** `_resolver_base_url`
(donde se GUARDA `Proveedor.base_url`) pasó a parsear la URL (`urlsplit`/`urlunsplit`)
en vez de concatenar texto: la primera versión hacía `desde_entorno + sufijo`, así que
una URL con path vacío pero CON query string (`.../8888?token=abc`) terminaba con el
sufijo pegado DENTRO del valor de la query (`.../8888?token=abc/v1`) en vez de en la
posición correcta (`.../8888/v1?token=abc`).

**La cuarta revisión encontró que la astilla se había movido una capa más abajo.**
`_resolver_base_url` guardaba `base_url` bien, pero `cliente.armar_peticion`
(`p.base_url.rstrip("/") + "/chat/completions"`) y `sondeo.sincronizar_catalogo`
(`p.base_url.rstrip("/") + p.modelos_path`) seguían construyendo la URL FINAL — la
que de verdad se manda por la red — con la misma concatenación cruda. El test de la
revisión anterior afirmaba sobre `Proveedor.base_url`, no sobre la URL pedida, así que
quedó en verde sin cubrir el bug real. Los tres puntos ahora comparten
`proveedores.unir_ruta(base_url, sufijo)`, que parsea y reconstruye una sola vez; los
tests nuevos afirman sobre la URL que el mock transport recibió de verdad
(`armar_peticion`'s `url` de retorno, y `req.url` en `sincronizar_catalogo`), no sobre
un campo intermedio.

**`desenvuelve_canvas` y `timeout_s` son declaraciones por proveedor, mismo shape que
`prioridad`/`capacidades_por_defecto` (revisión de Task 13).** `desenvuelve_canvas`
(default `False`) decide si `razonamiento`/`proxy` desenvuelven las cercas de canvas de
esa ruta (§6.1) — apagado por defecto porque `:::nota{...}` es también sintaxis
Docusaurus/MDX estándar, y aplicarlo a ciegas (como hacía la Task 13 original)
corrompe una respuesta de documentación legítima de cualquier otro proveedor
(reproducido en vivo contra Kilo). Solo `chatgpt` lo declara en `true`. `timeout_s`
(default `None` = usa el `TIMEOUT_S` global de `proxy.py`, hoy 90 s) permite acotar el
peor caso de un proveedor puntual sin bajarle el timeout a todos — se agregó junto con
el cooldown de fallos duros (§7) por la misma razón: una ruta colgada en una máquina
saturada (`blog`) es el modo de falla realista.

**Dos filtros en el descubrimiento de `chatgpt`, los dos derivados de la respuesta,
nunca de una lista de ids cableada:**

1. Los alias legacy (`gpt-4o`, `gpt-4o-mini`, `gpt-4`, `gpt-3.5-turbo`) se
   autoidentifican: su `description` empieza con `"Alias → <target>"`. Quedárselos
   crearía una ruta duplicada apuntando al mismo modelo, y el ranking terminaría
   midiendo — y compitiendo — el mismo modelo consigo mismo bajo dos nombres.
2. `auto` no se autoidentifica (su `description` es simplemente `"Auto"`), así que
   se filtra como **id reservado por llm-libre mismo** (`catalogo.es_id_reservado`):
   colisiona con el alias `auto` propio de `interpretar_pedido` — una ruta real con
   ese `modelo_id` literal quedaría inalcanzable, porque pedir `"auto"` siempre
   resuelve al alias, nunca a una ruta. Es un contrato de nuestra propia
   nomenclatura, no algo específico de chatgpt: cualquier proveedor futuro que
   exponga un id `"auto"` chocaría igual. **Desde la sexta ronda de revisión, la
   reserva cubre también cualquier alias compuesto `"auto:<sufijo>"`**
   (`"auto:rapido"`, `"auto:potente"`, `"auto:tools"`, `"auto:vision"` — el mismo
   `ALIAS` que resuelve `interpretar_pedido` en `api.py`, ver §6): el literal
   `"auto"` en `IDS_RESERVADOS` no bastaba, porque `interpretar_pedido` resuelve
   CUALQUIER id que empiece con `"auto:"` como alias antes de compararlo contra un
   id literal, así que un proveedor que publicara un modelo real con uno de esos
   ids creaba una ruta permanentemente inalcanzable por el mismo mecanismo.

## 5. Modelo de datos

**La unidad es la ruta (proveedor + modelo), no el modelo.** Un modelo lógico presente en
dos proveedores da failover sin cambiar de calidad: si Kilo rate-limitea
`poolside/laguna-s-2.1:free`, OpenRouter sirve el mismo modelo.

SQLite, un archivo en volumen:

- **`rutas`** — `proveedor`, `modelo_id`, `tier`, capacidades normalizadas
  (`soporta_tools`, `soporta_vision`, `contexto`, `max_salida`), `visto_por_ultima_vez`,
  `activa`, `prioridad` (Task 13, `INTEGER NOT NULL DEFAULT 100`, migración idempotente
  con `ALTER TABLE`, mismo patrón que `latencia_ms`). Las rutas que desaparecen del
  catálogo se marcan inactivas, **no se borran**: el histórico sirve para detectar
  renombres como el de `laguna`.
- **`sondas`** — una fila por sonda: `ruta`, `tipo` (salud|calidad), `momento`, `ok`,
  `latencia_ms`, `ttft_ms`, `codigo_http`, `casos_pasados`, `casos_totales`.
- **`eventos`** — telemetría del tráfico real: `ruta`, `momento`, `ok`, `latencia_ms`,
  `ttft_ms`, `codigo_http`, `llave`, `es_error_cliente` (tercera revisión de Task 13,
  `INTEGER NOT NULL DEFAULT 0`, misma migración idempotente). Alimenta el ranking entre
  sondas — salvo las filas con `es_error_cliente=1`, que `_confiabilidad` excluye de su
  ventana por completo (§7): quedan igual en la tabla, para diagnóstico, pero no pesan.
- **`uso_pago`** — `llave`, `dia`, `peticiones`. Para el tope diario de fallback.

Podado de `sondas` y `eventos` a los 30 días.

## 6. Contrato de la API

Drop-in de OpenAI: cualquier SDK funciona cambiando `base_url`, sin librería propia.

### `POST /v1/chat/completions`

Idéntico a OpenAI, con `stream: true`. El campo `model` acepta:

- **un id real** (`poolside/laguna-s-2.1:free`) → esa ruta, con failover automático a otro
  proveedor que sirva el mismo modelo
- **un alias virtual**: `auto` (perfil balanceado), `auto:rapido`, `auto:potente` —
  y los de capacidad `auto:tools`, `auto:vision`, que son atajos equivalentes a pedir
  `auto` con `x_requiere`

Extensiones opcionales, que un SDK ajeno ignora sin romperse:

- `x_requiere: ["tools", "vision"]` — capacidades obligatorias
- `x_min_contexto: 100000` — ventana mínima
- `x_permitir_pago: false` — desactiva el escape a pago para esta petición

Toda respuesta lleva `X-Ruta-Usada: <proveedor>/<modelo>`, `X-Tier: gratis|pago` y
`X-Intentos: <n>`. **El fallback de pago nunca es invisible.**

**Excepción, decidida el 2026-08-16:** en streaming esas cabeceras **no van**. Las cabeceras
HTTP se mandan antes del cuerpo, así que en ese momento todavía no se sabe qué ruta va a
servir — la cadena de intentos aún no se resolvió. La alternativa sería emitir un evento SSE
no estándar al principio del flujo, pero eso arriesga romper el parseo de los SDK que este
contrato existe para complacer, que es peor. El requisito de que el gasto pago no sea
invisible se cumple igual por dos vías: el consumo **sí** se contabiliza en streaming, y
`/v1/uso` lo refleja. Un cliente que streamea no sabe por cabecera quién lo atendió; lo sabe
por `/v1/uso` y por el `model` que vienen en los propios chunks.

### 6.1 Normalización del razonamiento filtrado

Varios modelos —gratis y de pago— escupen su cadena de pensamiento dentro de `content`
(§3). El gateway la separa antes de responder:

- Se recorta lo delimitado por etiquetas conocidas (`<think>`, `<thinking>`, `<reasoning>`)
  y se devuelve en un campo aparte, `x_razonamiento`, para quien lo quiera.
- El `content` que ve el cliente queda limpio.
- **En streaming hay que recortar sobre el flujo**, no al final: las etiquetas llegan
  partidas entre chunks, así que el normalizador mantiene un buffer pequeño y no emite
  texto hasta poder decidir si está dentro o fuera de un bloque de razonamiento.
- Los preámbulos en prosa sin etiqueta (`"Here's a thinking process:"`) **no se tocan**:
  recortarlos requeriría heurísticas frágiles que romperían respuestas legítimas. En su
  lugar, el caso "respeta el formato pedido" de la batería de calidad (§8) los penaliza,
  y esos modelos caen solos en el ranking.

Se puede desactivar por petición con `x_crudo: true`.

**Cercas de canvas (Task 13, alcance corregido en la revisión).** `chatgpt-proxy`
filtra el modo "canvas" de ChatGPT al `content` con marcas
`:::palabra{...atributos...}` … `:::` envolviendo la respuesta. **Al revés que
`<think>`: el contenido de ADENTRO es la respuesta**, no algo para descartar — solo se
quitan las dos líneas de marca (apertura y cierre), conservando todo lo demás carácter
por carácter. Mismo requisito que el recorte de razonamiento: la marca puede llegar
partida entre chunks en streaming. Una cerca que nunca cierra no pierde contenido
(solo se descuenta la marca de apertura, ya confirmada); un `:::` que no está al
inicio de línea, o al que no le sigue una palabra, nunca se confunde con una marca
real.

**El desenvuelto es POR PROVEEDOR (`Proveedor.desenvuelve_canvas`, §4), no global.**
La primera versión de Task 13 lo aplicaba a toda respuesta, de cualquier proveedor —
`recortar()` y `RecortadorStreamCompuesto` lo hacían incondicionalmente. Encontrado en
la revisión: `:::note` / `:::tip` / `:::warning` es sintaxis Docusaurus/MDX estándar
para admoniciones, y se reprodujo en vivo contra una ruta de **Kilo**: pedir
documentación devolvía `":::note\nGuarda el token en el .env.\n:::"`, y el cliente
recibía la marca arrancada — corrompiendo también un bloque de código que estuviera
*demostrando* la sintaxis. `recortar(texto, desenvolver_canvas=False)` y
`RecortadorStreamCompuesto(desenvolver_canvas=False)` ahora saltean el paso de canvas
por completo si el llamador no lo pide; `proxy.py` lo decide por ruta, leyendo
`Proveedor.desenvuelve_canvas` de la que está sirviendo el intento. Solo `chatgpt` lo
declara en `true`.

### `GET /v1/models`

El catálogo normalizado en formato OpenAI (para que los SDK lo listen), más los alias `auto*`.

### `GET /v1/ranking`

Propio, no OpenAI. Puntaje de cada ruta con **sus componentes desglosados** (calidad,
confiabilidad, latencia, cooldown, `prioridad`) y la fecha de su última sonda. Ordenado
con `router.clave_de_orden` — la MISMA clave que usa el router, cooldown incluido: una
ruta castigada va al final de la tabla aunque puntúe mejor que todas, porque el router
jamás la elegiría ahora mismo (revisión de Task 13; antes el orden era solo por
puntaje). Existe para poder auditar por qué el router eligió lo que eligió, sin que la
fila de arriba contradiga a `X-Ruta-Usada`.

### `GET /v1/uso`

Consumo por llave, con el gasto de pago del día contra su tope.

### `GET /v1/health`

**Honesto**: responde `ok` solo si hay al menos una ruta viva capaz de servir. Si todo lo
gratis está caído, dice `degradado`; si no hay nada, `caido`.

> Esto es lección directa del incidente del gateway de arkiv (2026-08-15): `/v1/health`
> decía `ok` mientras todos los endpoints autenticados daban 503 durante ~3 horas, porque
> solo comprobaba que el proceso estuviera arriba. Un health que no puede fallar no sirve.

**Una ruta cuenta como sana si no está en cooldown Y hay evidencia POSITIVA de que
sirve** (`Almacen.tiene_evidencia_de_vida`, revisión de Task 13, sexta ronda, Parte 2).
Las dos condiciones, no una; mirar solo el cooldown reproduce el mismo incidente que
motivó el endpoint (los cooldowns se llenan únicamente con 429 y con fallos "duros"
seguidos — ver §7 —, así que una ruta que se cuelga sin encajar en ninguno de los dos
nunca entra en cooldown y el health la seguiría contando como viva).

> **"Evidencia de vida, no ausencia de muerte."** Un éxito prueba que una ruta puede
> servir. Mil fallos de UN cliente no prueban que no puede. Antes de esta ronda, `/health`
> exigía que `confiabilidad` (un promedio de tráfico reciente, §7) superara un piso — y un
> promedio se arrastra a 0 con cualquier patrón repetido de un solo cliente. El caso que lo
> probó es `403`: genuinamente ambiguo (cuenta suspendida = evidencia de la ruta, vs.
> moderación de contenido = evidencia del PEDIDO), y el gateway no puede desambiguarlo sin
> parsear el cuerpo específico de cada proveedor. Clasificarlo del lado de la ruta por
> default (correcto, §7) no alcanza: 30 pedidos con contenido flageado de UNA llave bastan
> para tirar la confiabilidad de TODAS por el piso, con `/health` en `caido` — y Coolify usa
> `/health` como *health check*, así que reinicia el contenedor sin que el reinicio resuelva
> nada (la telemetría vive en el volumen persistente `/datos`).

Dentro de una ventana de 24 h (`VENTANA_EVIDENCIA_VIDA_S`, generosa contra el sondeo
default de 5 h y contra una ruta de pago que nunca se sondea y solo puede probar que vive
con tráfico real, esporádico por naturaleza), se junta el último **éxito real**
(`eventos.ok=1`) con la última **sonda de salud** (`sondas.tipo='salud'`, cualquier
resultado) y gana la que tenga el `momento` más nuevo:

- si la señal más nueva es un éxito real, o una sonda **exitosa** — la sonda es la señal
  más confiable que existe porque el gateway controla su propio payload: un `4xx` contra
  una sonda es *siempre* evidencia de la ruta, nunca hay ambigüedad de "contenido del
  cliente" posible — la ruta cuenta como viva;
- si la señal más nueva es una sonda **fallida** — la ruta cuenta como muerta. **Corregido
  en una séptima revisión**: la versión anterior solo miraba sondas exitosas, así que un
  éxito real dentro de la ventana (p.ej. hace 20 h) seguía contando como viva aunque
  hubiera sondas fallidas *después* de ese éxito (cuatro, una cada 5 h, la última hace 1 h)
  — el mismo argumento que ya justificaba confiar en una sonda exitosa (payload controlado
  por el gateway, nunca ambiguo) vale igual para una fallida: es evidencia inequívoca de
  que la ruta está rota, no solo de que está viva, y mirar solo la mitad del argumento
  dejaba `/health` en `ok` durante toda la ventana mientras cada cliente real recibía `503`;
- si no hay ninguna señal (ni éxito ni sonda) dentro de la ventana, se cae al chequeo de
  **sin telemetría real en absoluto** — una ruta recién sincronizada no nace muerta,
  todavía no tuvo su primera oportunidad (el mismo principio que ya regía con el valor
  neutro de `confiabilidad`, ahora expresado sin depender de esa métrica). Los eventos con
  `es_error_cliente=1` (400/413/422, ver §7) tampoco cuentan como telemetría real para este
  chequeo: una ruta que solo recibió pedidos malformados está en el mismo lugar que una sin
  tráfico.

**Los fallos reales (`eventos.ok=0`), solos, nunca son suficientes para declarar una ruta
muerta aquí** — siguen siendo ambiguos (un `403` puede ser moderación de contenido de un
cliente puntual) de una forma en que una sonda —pedido fijo, controlado por el propio
gateway— nunca lo es.

**`confiabilidad` sigue alimentando el ranking y la ordenación real de la cadena de
intentos exactamente como antes** (ver §7) — no solo el endpoint de diagnóstico
`GET /v1/ranking`: `router.ordenar` también la consume, vía `clave_de_orden` →
`ranking.puntuar`, para decidir el orden real en que el proxy prueba las rutas. La
asimetría es intencional: una ruta mal puntuada pierde posición en esa cadena y se
autocorrige sola en cuanto alguien mira la tabla o el router recalcula; una ruta que
`/health` declara muerta reinicia el contenedor. El ranking (y la selección real) pueden
permitirse ser sensibles a patrones de tráfico de un solo cliente — la salud no.

## 7. Ranking y selección

Por ruta:

- **`calidad`** (0–1) — fracción de casos de la batería que pasó en la última evaluación.
  Las rutas nunca evaluadas arrancan en un valor neutro configurable, no en 0 (si no,
  jamás las elegiría y nunca se evaluarían).
- **`confiabilidad`** (0–1) — EWMA de éxitos, mezclando sondas y tráfico real. **Un `4xx`
  del cliente (ver más abajo) NUNCA entra en esta cuenta** (corrección de una tercera
  revisión, severidad HIGH): el evento se sigue guardando en `eventos` (queda
  diagnosticable — `es_error_cliente=1` en la fila), pero `Almacen._confiabilidad`
  excluye esas filas de la ventana por completo, ni como fallo ni como observación.
  Antes solo se sacó del *contador de cooldown* (revisión anterior), pero seguía
  contando como fallo común hacia confiabilidad. Reproducido: 26 pedidos malformados
  SEGUIDOS de una sola llave (un cliente reintentando el mismo error — algo que los SDK
  de OpenAI hacen solos) bastan para tirar la confiabilidad de TODAS las rutas por el
  piso. **`confiabilidad` sigue alimentando `/v1/ranking` exactamente como siempre —
  pero, desde la sexta ronda de revisión, `/health` ya NO la usa** (ver la nota de
  `GET /v1/health` en §6): incluso con la exclusión de `es_error_cliente`, un código
  genuinamente ambiguo como `403` (evidencia de la ruta por default, pero puede ser
  moderación de contenido de un solo cliente) puede seguir arrastrando el promedio, y
  Coolify usa `/health` como *health check* — un promedio envenenado ahí REINICIA el
  contenedor, contra un servicio que responde bien, y `eventos` vive en el volumen
  persistente `/datos` así que el reinicio no lo cura. `/health` usa en cambio
  `Almacen.tiene_evidencia_de_vida` — evidencia positiva, no ausencia de fallos —,
  precisamente porque el ranking puede permitirse ser sensible (se autocorrige) y la
  salud no.
- **`latencia`** — p50 de time-to-first-token de las últimas N observaciones.
- **`cooldown`** — una ruta que devolvió 429 queda excluida hasta que expire su castigo
  (backoff exponencial con tope). **Desde la revisión de Task 13, también una ruta que
  falla `TOPE_FALLOS_SEGUIDOS` (3) veces SEGUIDAS por un fallo "duro"** (5xx, timeout,
  error de red, 200 sin contenido usable) — con el MISMO backoff. Antes solo el 429
  castigaba: una ruta rota o colgada se seguía probando en cada pedido, adelante de las
  sanas si tenía mejor `prioridad`, para siempre — con `TIMEOUT_S=90` eso son hasta
  5×90s=450s por pedido en la cadena más larga, y `/health` sigue en `ok` mientras quede
  una ruta viva. `blog` es una máquina saturada: colgado-no-rechazado es el modo de
  falla realista. Un fallo aislado no castiga (un hiccup no debe sacar una ruta sana de
  rotación); un éxito limpia el contador. Un proveedor puede declarar su propio
  `timeout_s` (§4) para acotar el peor caso sin bajarle el timeout a todos — aplica al
  camino síncrono y al de streaming por igual.

  **Un `4xx` cuenta para este cooldown (y para confiabilidad) según de quién es la
  CULPA — el pedido, o la ruta — no según si es reintentable.** El eje siempre fue
  atribución, pero hasta una cuarta revisión la clasificación arrancaba de retryability
  (un falso-verde: el reviewer probó agregar un código "no reintentable" al lado
  equivocado y la suite entera seguía en verde). Hasta una quinta ronda, encima, el
  **default estaba invertido**: "todo `4xx` cuenta como pedido SALVO estos siete
  códigos" — un default que esconde en silencio cualquier código en el que nadie pensó
  todavía. Probado con el mutante obvio: agregar `405` al conjunto de "sí cuenta" (lo
  que la propia regla exigía) dejaba la suite entera en verde, porque ningún test fijaba
  el EJE, todos enumeraban códigos.

  > **El default está INVERTIDO desde la sexta ronda: un `4xx` es evidencia de LA RUTA
  > salvo que esté en una lista CORTA y justificada de códigos que son evidencia
  > GENUINA del pedido — nunca de la ruta. Todo lo demás, conocido hoy o no, cuenta
  > igual que un `500`.**
  >
  > **Principio: cuando no se puede saber de quién es la culpa, hay que CONTARLO.** Una
  > falsa alarma se recupera sola — alguien mira el ranking o `/health`, ve que la ruta
  > está bien, sigue. Una salida silenciosa no — nadie la mira nunca. Los costos son
  > asimétricos, y el default tiene que inclinarse hacia notar, no hacia callar.

  La lista corta (`proxy._CODIGOS_EVIDENCIA_DE_PEDIDO`), con la justificación de cada
  entrada como evidencia genuina del PAYLOAD de este pedido puntual — nunca de si la
  ruta sirve:

  - `400` — Bad Request: el cuerpo ni se pudo interpretar. Es sobre la sintaxis de ESTE
    pedido, nunca sobre si la ruta sirve.
  - `413` — Payload Too Large: el TAMAÑO de este pedido excede un límite; un pedido más
    chico andaría bien contra la MISMA ruta.
  - `422` — Unprocessable Entity: sintácticamente válido pero inválido para este pedido
    puntual (un parámetro fuera de rango, un tipo equivocado) — no dice nada sobre si la
    ruta está sana.

  `armar_peticion` reenvía el cuerpo del cliente tal cual, así que contar estos tres
  convertiría el error de un cliente en un apagón para todos — verificado contra el
  registro real de 5 rutas, tres pedidos malformados (`400`) seguidos bastaban para
  dejar las cinco en cooldown, con una llave DISTINTA recibiendo `503` mientras tanto.

  **Todo lo demás cuenta como evidencia de la ruta por default — incluidos códigos que
  una versión anterior de esta regla excluía explícitamente:**

  - `401` — la clave del proveedor venció o se rotó. Otra clave andaría bien con el
    mismo pedido; el pedido no tiene la culpa.
  - `402` — sin crédito (`insufficient credits` de OpenRouter). Recargar arregla
    cualquier pedido contra esta ruta.
  - `403` — cuenta suspendida o moderación del lado del proveedor, no un problema del
    payload. **Genuinamente ambiguo** (ver la nota de `GET /v1/health` en §6): la
    moderación de contenido SÍ es evidencia del pedido, pero el gateway no puede
    distinguirla de una cuenta suspendida sin parsear el cuerpo específico de cada
    proveedor — se resuelve contando el código del lado de la ruta (el default) y
    sacando a `/health` del promedio de confiabilidad, no intentando adivinar el
    sub-caso.
  - `404` — `model_not_found`: literalmente el problema central que este proyecto
    existe para detectar (§1). Reachable hasta 5 h entre sincronizaciones de catálogo,
    y las rutas de pago (nunca sondeadas) no tienen ningún otro backstop. Cuando el
    modelo pedido explícitamente SIGUE en el catálogo local pero el proveedor ya
    devuelve `404` en vivo, `/v1/chat/completions` lo refleja como `404` con
    sugerencias — no como el `503` genérico de "sin rutas disponibles" — en el camino
    síncrono (§6).
  - `405`, `409`, `415`, `418`, `431`, `451` — y cualquier otro `4xx` en el que nadie
    pensó todavía: caen del lado de la ruta por el simple hecho de no estar en la lista
    corta. Esa es la prueba de que el eje quedó bien plantado, no una enumeración más.
  - `408` — el upstream se colgó esperando: el mismo síntoma que motivó todo este
    mecanismo, no una nuance del pedido.
  - `425` — nuance de protocolo/TLS (replay contra esta conexión), no un payload
    inválido.
  - `429` — rate-limit: tiene su propio castigo INMEDIATO (`_castigar`, sin pasar por
    el cooldown de fallos duros), pero conceptualmente también es evidencia de la
    ruta (su capacidad ahora mismo), no del pedido.

  `408`/`425`/`429` ya no necesitan mención especial en el código — bajo el default
  invertido caen solos del lado de la ruta, sin tener que aparecer en ninguna lista.

  Medido el costo de la clasificación anterior (default invertido): cinco rutas
  devolviendo `405` (o `409`/`415`/`418`/`431`/`451`, cualquier `4xx` que nadie hubiera
  anticipado) dejaban al cliente con `503` en el 100 % de los pedidos, CERO cooldowns, y
  `/health` en `200 ok` — la misma salida silenciosa de una revisión anterior, con un
  código distinto cada vez. `proxy._es_error_del_cliente` (y el conjunto
  `_CODIGOS_EVIDENCIA_DE_PEDIDO` que lo respalda, con la justificación de cada entrada)
  es la única fuente de verdad para "de quién es la culpa" — la usan tanto el gate del
  cooldown como el flag `es_error_cliente` que se guarda en `eventos`, para que las dos
  exclusiones no puedan desincronizarse.

  **El test que pin­ea esto recorre el rango `4xx` completo (`range(400, 500)`), no una
  muestra de códigos conocidos**, contra una copia INDEPENDIENTE de la lista corta
  esperada (no importa la constante real): si alguien agrega un código a la lista real
  sin actualizar el test, se pone rojo — pin del EJE, no de la lista. Es la forma
  práctica de responder "¿esto discrimina la regla de la lista?" que una lista de
  ejemplos, por larga que sea, no puede responder por construcción.

  **Séptima revisión: la atribución de arriba se aplicaba solo a nivel de RESPUESTA — el
  cooldown seguía roto.** `_es_error_del_cliente` decide bien qué código es evidencia de
  QUÉ, pero `completar`/`completar_stream` juzgaban cada ruta de la cadena SOLA, sin mirar
  qué pasó con sus hermanas en el MISMO pedido. Eso deja un vector que ni siquiera necesita
  un código mal clasificado: un `403` de moderación de contenido, un `451`
  legal/geográfico, un timeout por un prompt gigante, o un `200` sin contenido de un
  modelo de razonamiento son, los cuatro, **correctamente** evidencia de la ruta por la
  regla de arriba — pero también son deterministas: CUALQUIER ruta le devuelve lo mismo a
  ESTE pedido puntual. Medido: tres pedidos IDÉNTICOS de una sola llave contra un gateway
  de 5 rutas SANAS bastaban para poner las cinco en cooldown — `/health` `caido`, otra
  llave con un pedido válido recibiendo `503`.

  > **El principio de arriba ("cuando no se sabe, contar") es por-CONSUMIDOR, no global —
  > y los dos consumidores de esta clasificación necesitan el default OPUESTO.**
  > `confiabilidad` es una MEDICIÓN: una falsa alarma ahí se autocorrige sola (alguien mira
  > el ranking, ve que la ruta está bien, sigue), así que ante la duda, CONTAR. `cooldown`
  > es una EXCLUSIÓN: una falsa alarma ahí ES la interrupción — una ruta sana excluida por
  > error es el apagón con luz verde que el mecanismo entero existe para prevenir — así que
  > ante la misma duda, **NO excluir**.

  El eje que resuelve esto no vive en el código HTTP sino en la CADENA: `completar` recorre
  todas las rutas candidatas de un pedido, así que sabe algo que ninguna respuesta
  individual puede decir — si el MISMO pedido falla en TODAS las rutas de la cadena, el
  factor común es el pedido, no que todas las rutas hayan fallado a la vez. Un fallo solo
  se computa hacia el contador de fallos duros de SU ruta si, en esa MISMA vuelta de la
  cadena, alguna OTRA ruta tuvo éxito — evidencia real de que esa ruta en particular falló,
  no el pedido. Si la cadena se agota entera sin ningún éxito y tiene más de una ruta, se
  descartan TODOS los fallos de esa vuelta.

  `429` no pasa por este mecanismo — sigue castigando de inmediato, sin importar el resto
  de la cadena: es una señal inequívoca de la capacidad de ESA ruta ahora mismo, no algo
  que un payload pueda inducir idéntico en varios proveedores independientes a la vez.

  **Octava revisión: la atribución a nivel de cadena seguía rota — las dos fugas que
  quedaban estaban en las EXCEPCIONES que la propia séptima revisión escribió a
  propósito.** La cadena de UNA sola ruta ("sin hermana con quién compararse, se computa
  de inmediato") parecía un caso borde, pero el CLIENTE la puede forzar él mismo — con un
  `model` explícito, o con `x_min_contexto` (que `GET /v1/ranking` ya publica por ruta, sin
  hacer falta ningún conocimiento interno del gateway): 15 pedidos idénticos contra una
  cadena de una sola ruta bastaban para enfriarla, una por una, las cinco rutas del
  registro. Peor: la rama `if emitido:` de `completar_stream` (el force-flush de
  `TOPE_PENDIENTES` cuando un stream retiene demasiados chunks sin contenido útil)
  commiteaba de inmediato SIN chequear ni hermana ni longitud de cadena — ni siquiera hacía
  falta narrows la cadena: `{"model":"auto","stream":true}`, sin extensiones, con un prompt
  que agota el presupuesto de razonamiento sin emitir texto nunca, apagaba las cinco rutas
  en 15 pedidos.

  > **Cuando las fugas están en las excepciones que uno mismo escribió a propósito, el eje
  > está mal, no sub-enumerado.** Seis rondas del mismo mecanismo (retryability, una lista
  > de códigos, la lista con el default invertido, atribución a nivel de respuesta,
  > atribución a nivel de cadena) cayeron cada una por un vector nuevo sobre la MISMA señal
  > compartida: qué le pasó al PEDIDO del cliente.

  **El cambio, esta vez estructural: el tráfico de un cliente real deja de ser una señal
  que EXCLUYE una ruta, y pasa a ser una señal que puede, como mucho, pedirle al gateway
  que vaya a mirar.**

  > **Solo una sonda escrita por el propio gateway puede excluir una ruta. El tráfico de
  > un cliente nunca puede — solo puede pedirle al gateway que vaya a mirar.**

  La razón por la que esto cierra los seis vectores de una vez: el payload de una sonda
  (`PING`, ahora en `proxy.py` — antes en `sondeo.py`, ver más abajo) lo escribe EL
  GATEWAY. No hay nada que atribuir. Si la sonda falla, la ruta está rota, punto — ninguno
  de los vectores de las seis rondas anteriores (contenido flageado, prompt gigante, un
  modelo de razonamiento que se gasta el presupuesto, una cadena corta, un early-return de
  streaming) puede tocar el payload de una sonda, porque el cliente nunca lo escribe.

  Mecánica (`Proxy._sospechar`, `Proxy._probar_bajo_demanda`):

  - Un fallo de tráfico real (no-429, no evidencia-de-pedido) ya NO se computa hacia
    ningún cooldown — solo se agrega a una lista de marcas de tiempo por ruta
    (`_sospechas`), que se poda a `VENTANA_SOSPECHA_S` (10 min) en cada llamada. La
    sospecha DECAE a propósito: mucho más corta que "horas", así que fallos de un
    incidente real disperso en el tiempo nunca se suman para disparar una sonda espuria.
  - Al cruzar `UMBRAL_SOSPECHA` (3, el mismo número que el viejo `TOPE_FALLOS_SEGUIDOS`)
    marcas dentro de la ventana, se programa — en SEGUNDO PLANO (`asyncio.create_task`,
    sin bloquear la respuesta al cliente que disparó el umbral: ese cliente no tiene por
    qué pagar la latencia de investigar una ruta ajena, y bloquear ahí arriesgaría colgar
    el pedido si la ruta está de verdad trabada) — una sonda BAJO DEMANDA: el mismo
    `completar()` de siempre, con el mismo payload `PING` que la sonda periódica, sobre una
    cadena de una sola ruta, con una bandera (`es_sonda=True`) que hace que UN fallo (sin
    necesitar sospecha, sin poder programar otra sonda anidada) castigue de inmediato.
  - Si la sonda falla, `_castigar` corre exactamente igual que con un `429` (mismo backoff
    exponencial, mismo tope). Si pasa, la sospecha se limpia y la ruta sigue en rotación.
    La sonda deja además su propia fila en `sondas` — la misma tabla que ya lee
    `Almacen.tiene_evidencia_de_vida` para `/health` — así que, para el resto del sistema,
    una sonda bajo demanda y una periódica son indistinguibles: solo cambia quién la
    disparó.
  - **Rate limit por ruta: como máximo una sonda bajo demanda cada
    `LIMITE_PROBE_BAJO_DEMANDA_S` (60 s).** Una sonda gasta la MISMA cuota gratis que el
    tráfico real (el mismo endpoint, el mismo límite por minuto del proveedor), así que el
    costo extra que UN cliente hostil puede imponerle a UNA ruta queda topeado en, como
    mucho, 60 pedidos extra por hora — sin importar cuántos pedidos mande el cliente
    (podría mandar miles y el tope sigue siendo el mismo). Y como el payload de esa sonda
    es del gateway, el cliente jamás puede hacer que ese pedido extra FALLE: una ruta sana
    pasa la sonda siempre, así que el peor resultado posible para el atacante es gastar ese
    cupo fijo sin lograr nunca que una ruta sana caiga. **Costo para el atacante: con UNA
    sola llave, lo peor que puede lograr es forzar sondeo extra acotado (como mucho 60
    pedidos/hora por ruta) — nunca la caída de una ruta sana**, porque solo el gateway
    escribe el payload que de verdad decide.
  - `429` sigue exactamente igual que siempre — no pasa por sospecha, no dispara ninguna
    sonda: es evidencia inequívoca de ESA ruta sin necesitar confirmación.
  - **Las rutas de PAGO quedan afuera del mecanismo entero, deliberadamente.** Nunca se
    sondean (§8), así que una sonda bajo demanda gastaría PLATA REAL contra un proveedor de
    pago sin un dueño natural de esa cuenta (`TOPE_PAGO_DIARIO` es por LLAVE, no por ruta).
    Sin sonda posible, tampoco tiene sentido acumular sospecha que nunca se va a resolver.
    Van últimas en la cadena de todos modos, así que solo importan cuando TODO lo gratis ya
    falló o está en cooldown — el mismo escenario en el que `/health` ya reporta el apagón.
    Solo el `429` las sigue pudiendo castigar. (La alternativa considerada —una sonda bajo
    demanda que cuente contra el tope diario— quedó descartada: no hay una llave "dueña"
    natural de ese gasto para una sonda que dispara el propio gateway.)

  **Promptness, no solo corrección**: una ruta genuinamente rota, con una hermana sana en
  su misma cadena, sigue cayendo en el mismo número de pedidos que antes (round 7) —
  verificado con un test de contraste directo, en el camino síncrono y en el de streaming.
  Una cadena de una sola ruta que está rota de verdad también se enfría en
  `UMBRAL_SOSPECHA` pedidos más una sonda — segundos u órdenes de pedidos, nunca el ciclo
  de 5 h del sondeo periódico. Un *pool* genuinamente caído (todas las rutas rotas a la
  vez) se excluye ruta por ruta con la misma rapidez, sin esperar horas.

`puntaje = calidad^wc · confiabilidad^wr · f(latencia)^wl`, con los pesos según el **perfil**
pedido: `rapido` pondera latencia, `potente` pondera calidad y contexto, `balanceado` (el
default) reparte parejo.

El router devuelve una **lista ordenada**, no un ganador único: el proxy baja por ella ante
fallos. La clave de orden completa (Task 13, con el criterio de cooldown agregado en la
segunda revisión) es `(en-cooldown, tier == "pago", prioridad, no-medida, -puntaje)`:

0. **`en_cooldown_hasta > ahora` decide antes que nada.** En `router.ordenar` esto es un
   no-op (las rutas en cooldown ya se filtraron de la lista antes de llegar al sort); es
   en `GET /v1/ranking` (§6) — que muestra TODAS las rutas activas por diagnóstico, sin
   filtrar cooldown — donde este criterio hace el trabajo: una ruta castigada no puede
   encabezar la tabla solo por tener buena `prioridad`/puntaje, porque el router jamás la
   elegiría ahora mismo.
1. **`tier == "pago"` decide después, siempre.** Las rutas de pago van siempre al final, y
   solo entran si (a) se agotaron las gratis, (b) la llave no superó su tope diario y (c) la
   petición no trae `x_permitir_pago: false`. **Ninguna `prioridad` puede comprar ese
   lugar**: una ruta de pago con `prioridad: 0` sigue yendo última. La plata es la razón.
2. **`prioridad`** (entero manual, default `100`, ver §2 y §4) decide dentro de un mismo
   `tier` — p.ej. `chatgpt` (`prioridad: 0`) se prueba antes que `kilo`/`openrouter`
   (`prioridad: 1`), aunque puntúe peor.
3. A igual prioridad, una ruta nunca sondeada por la batería de calidad va después de una
   con medición real (el criterio que ya existía).
4. Y recién ahí decide el puntaje.

Un `model` explícito (un id real, no un alias `auto*`) sigue evitando todo este orden: filtra
directo a esa ruta, sin mirar prioridad ni puntaje.

Esta clave vive en `router.clave_de_orden` (toma `ahora` como argumento explícito, sin
default — un default fijo como `0.0` haría que cualquier `en_cooldown_hasta` ya vencido
se leyera como "todavía castigado" para siempre), factorizada (revisión de Task 13) para
que `GET /v1/ranking` (§6) la reuse en vez de ordenar solo por puntaje: antes podía mostrar
una ruta arriba de todo mientras `X-Ruta-Usada` decía otra distinta, porque no miraba
`prioridad`. Ahora cada fila trae `prioridad` y el orden de la respuesta es el orden
real que usaría el router.

## 8. Sondeo

**Salud, cada 5 h, todas las rutas gratis:** un chat mínimo con `max_tokens` bajo. Registra
si existe, si responde, cuánto tarda hasta el primer token, y el código de error si falla.
Detecta los renombres de modelo el mismo día en que pasan.

**Calidad, cada 5º ciclo (≈ diaria), rutas gratis vivas:** una batería chica de casos con
respuesta **verificable por código**, sin juez:

- JSON válido contra un schema dado
- tool call con el nombre y los argumentos correctos
- respetar un formato pedido (p. ej. "responde solo con una palabra")
- aritmética simple de resultado único
- responder en español cuando se lo pide

Cada caso es un assert: el puntaje es auditable y reproducible. Las rutas de pago **no se
sondean** — no tiene sentido gastar dinero puntuando el escape de emergencia.

**Sonda bajo demanda (octava revisión, ver §7):** ademas de este ciclo de 5 h, `Proxy`
dispara sondas de salud por su cuenta cuando el trafico real acumula suficiente sospecha
sobre una ruta — mismo payload `PING` (ahora vive en `proxy.py`, no en `sondeo.py`, para
que las dos sondas sean literalmente el mismo pedido), mismo destino en `sondas`, rate-
limitada a una por ruta cada `LIMITE_PROBE_BAJO_DEMANDA_S`. Es el mecanismo que le da a
una ruta genuinamente rota una forma de excluirse en el orden de segundos/pedidos, sin
esperar el proximo ciclo programado — el ciclo de 5 h sigue como esta, esto le agrega un
disparador, no lo reemplaza. Las rutas de pago (nunca sondeadas, arriba) tampoco reciben
sondas bajo demanda: ver §7 para la justificacion completa.

## 9. Errores

| Situación | Respuesta |
|---|---|
| Ninguna ruta cumple las capacidades pedidas | `400` con qué se pidió y qué hay disponible |
| Todas las candidatas caídas o en cooldown, sin pago disponible | `503` con cuándo se libera la primera |
| Proveedor devuelve 429 | Transparente: cooldown y siguiente ruta |
| Llave superó su tope de pago diario | `503`, nunca un cobro silencioso |
| Modelo pedido explícitamente que ya no existe en el catálogo local | `404` con los ids vigentes más parecidos |
| Modelo pedido explícitamente que SIGUE en el catálogo local pero el proveedor real devuelve `404` en vivo (sexta ronda, "la razón de ser del proyecto") | `404` con los ids vigentes más parecidos — camino síncrono únicamente; en streaming el `200`/cabeceras SSE ya salieron antes de saber si la ruta sirvió, así que se queda en el `503` genérico |

## 10. Seguridad

- `X-API-Key`, llaves en `.env` (`LLM_LIBRE_API_KEYS=k1,k2,...`), igual que `arkiv-api`.
- Límite de peticiones por minuto por llave, en memoria. No es burocracia: es lo que evita
  que un bucle de una app interna queme la cuota gratis de todos los consumidores.
- Las claves de proveedor nunca salen hacia el cliente.
- Público por cloudflared.

## 11. Despliegue

**Por Coolify** (ya corre en `blog`, versión 4.1.2, con Traefik de proxy), no por rsync.
Repo privado `lordmacu/llm-libre` en GitHub; Coolify construye desde el `Dockerfile` de la
raíz y redespliega en cada push a `main`.

Esto **elimina el problema del `.env`**, que es la razón principal de elegirlo: las
credenciales viven en la UI de Coolify, no en un archivo del disco. Un `rsync --delete` sin
`--exclude .env` ya borró las credenciales de producción de `arkiv-api` el 2026-08-11; acá
ese fallo no puede ocurrir porque no hay archivo que borrar. Es además el patrón que ya usan
los 7 bots de WhatsApp de la misma máquina.

Piezas que hay que configurar en la UI:

- **Port** `8101`, **Build Pack** Dockerfile, **Health Check Path** `/health`
- **Volumen persistente** montado en `/datos` — sin él el SQLite se borra en cada redeploy
  y el ranking, que tarda días en construirse, vuelve a cero
- Variables de entorno, con **`KILO_API_KEY` deliberadamente sin definir**: el tier anónimo
  necesita que no viaje ninguna cabecera `Authorization`

⚠️ El build corre en `blog`. Es aceptable porque `pip install` de estas cuatro dependencias
baja wheels y no compila nada — la regla de "no compilar en blog" apunta a Rust, no a esto.

Queda un `docker-compose.yml` en el repo, pero solo para levantarlo en local durante el
desarrollo.

## 12. Pruebas

- **Mayoría sin red:** `ranking` y `router` son funciones puras; se prueban con datos
  fabricados. Incluye los casos que importan: empate, todo en cooldown, ninguna ruta con
  tools, ruta nueva sin evaluar, tope de pago alcanzado, y (Task 13) el invariante de que
  una ruta de pago con `prioridad: 0` y puntaje perfecto sigue yendo al final contra una
  ruta gratis mediocre, y que un `model` explícito evita el orden por completo.
- **Normalización de catálogo:** contra JSON reales grabados de Kilo y OpenRouter, incluido
  el caso `lyria` (precio 0 pero modelo de música → debe descartarse). Y (follow-up de
  Task 13) contra un fixture grabado de `/v1/models` de chatgpt-proxy
  (`tests/fixtures/chatgpt_models.json`, 10 entradas reales): los 5 ids reales se
  descubren, los 4 alias legacy y `auto` se descartan, y las `capacidades_por_defecto`
  se aplican a cada uno. Un segundo fixture con un id que hoy no existe
  (`chatgpt_models_con_modelo_nuevo.json`) prueba que un modelo nuevo del proxy
  aparece solo, sin tocar `proveedores.yaml` ni el código. Un proveedor SIN
  `capacidades_por_defecto` (Kilo/OpenRouter) se prueba con el comportamiento exacto
  de antes — pricing y modalidad de salida siguen filtrando.
- **Recorte de razonamiento:** con la etiqueta partida en todas las posiciones posibles
  entre chunks, bloques anidados, y una etiqueta que nunca cierra (no debe tragarse la
  respuesta entera ni colgar el stream).
- **Cercas de canvas (Task 13):** mismo estándar que el recorte de razonamiento — la marca
  partida en cada posición posible entre chunks, una cerca que nunca cierra (no debe
  perder contenido), texto con `:::` que no es una cerca real (no debe tocarse), y
  contenido normal intacto. Además (revisión de Task 13): una ruta SIN
  `desenvuelve_canvas` (Kilo) deja `:::note` intacto, en bloque y en streaming; cuatro
  tests dedicados a mutaciones puntuales del autómata (la marca de cierre se descarta,
  no se emite; sin el seguimiento de inicio-de-línea se pierde texto de respuesta real;
  el patrón de apertura exige la línea entera; el de cierre exige la línea exacta), cada
  uno verificado ejecutando la mutación descrita y confirmando que el test se pone rojo.
- **Migración de esquema (Task 13):** `rutas.prioridad` verificada contra una base con el
  esquema viejo (sin esa columna) y filas ya adentro — no debe reventar al abrir, la fila
  preexistente migra al default, y la base sigue siendo escribible después.
- **Cooldown por fallos duros (revisión de Task 13):** N fallos NO-429 seguidos ponen una
  ruta en cooldown incluso con `prioridad: 0`; un éxito limpia el contador; el camino del
  429 (backoff exponencial inmediato) se prueba sin tocar, para confirmar que sigue
  intacto. **Corrección HIGH de una segunda revisión:** tres `400` seguidos NO castigan
  (y un pedido válido posterior sigue sirviéndose); una mezcla de `4xx`/`5xx` cuenta
  solo los `5xx`; probado en `completar()` y en los tres caminos de falla de
  `completar_stream()`. Y `timeout_s` por proveedor, que el reporte anterior decía
  aplicado a "las dos llamadas HTTP reales" pero solo cubría la no-streaming — corregido
  y pineado con un test que inspecciona `req.extensions["timeout"]` en el camino de
  streaming también.
- **`/v1/ranking` modela el cooldown (segunda revisión):** una ruta castigada, aunque
  tenga la mejor `prioridad` y el mejor puntaje, va al final de la tabla — no solo
  `prioridad` (primera revisión, hallazgo 3).
- **Un `4xx` de evidencia-del-pedido no cuenta para confiabilidad/health/ranking
  (tercera revisión, HIGH):** 30 `400` seguidos (y 30 `422`) dejan `/health` en `ok` y
  todas las rutas sobre el piso, y `/v1/ranking` no se mueve; 30 `500` seguidos siguen
  tirando `/health` y el ranking exactamente como antes. Probado también contra un
  proceso NUEVO abierto sobre la MISMA base de archivo en las dos direcciones (el caso
  de loop de reinicios de Coolify): un cliente-evidencia (`400`) sigue leyendo `ok`, un
  ruta-evidencia (`401`) sigue leyendo `caido` — sin nada en memoria del proceso
  original. **Pin del EJE (cuarta revisión):** `401`/`402`/`403`/`404` — no
  reintentables, pero SÍ evidencia de la ruta — cuentan igual que un `500` hacia
  cooldown, confiabilidad y `/v1/ranking`, probado tanto a nivel `proxy`
  (`_es_error_del_cliente`) como end-to-end (30 seguidos por cada código, más el mismo
  test de proceso-nuevo-misma-base para `401`). Un test que reclasificara estos cuatro
  de vuelta al lado del cliente (razonando "no son reintentables") se pondría rojo —
  verificado revirtiendo el conjunto a solo `{408, 425, 429}` y confirmando que los
  seis tests nuevos fallan con el mensaje exacto que motivó la revisión.
- **Normalización de `base_url_env` (segunda revisión):** una ruta propia en la variable
  de entorno que no coincide con el sufijo default se usa tal cual, sin pisarla — solo
  se loguea el aviso. **Corrección LOW, en dos capas:** la tercera revisión parseó la
  URL (`urlsplit`/`urlunsplit`) en `_resolver_base_url`, donde se guarda
  `Proveedor.base_url`; la cuarta encontró que `cliente.armar_peticion` y
  `sondeo.sincronizar_catalogo` seguían concatenando texto para construir la URL FINAL
  — la que de verdad se pide — así que la astilla sobrevivía una capa más abajo. Los
  tests de la tercera revisión afirmaban sobre `Proveedor.base_url` y por eso quedaron
  en verde sin cubrirlo; los nuevos afirman sobre la URL real (`armar_peticion`'s `url`
  de retorno, `req.url` capturado por el mock transport en `sincronizar_catalogo`), a
  través del helper compartido `proveedores.unir_ruta`.
- **Rungs sin pinnear, cerrados en la revisión:** `rutas_fijas` estampando la `prioridad`
  real del proveedor (no una constante — probado con un YAML sintético y un valor
  distintivo), y el orden `(prioridad, no-medida)` en `router.clave_de_orden` (no
  `(no-medida, prioridad)` — el caso que decide el rollout real: el día de un deploy,
  todo `chatgpt` arranca sin medir mientras Kilo ya carga mediciones de producción).
  Ambos verificados ejecutando la mutación descrita (intercambiar el orden de la tupla /
  hardcodear la prioridad) y confirmando rojo.
- **Integración real:** un puñado marcado para no correr en CI, contra los proveedores vivos
  — incluye `chatgpt-proxy` (Task 13), que se salta limpio si `CHATGPT_PROXY_URL` no está
  configurada, y que además (revisión) confirma que el catálogo descubierto no viene
  vacío ni con alias colados, y (segunda revisión) que mandarle `tools`
  +`tool_choice:"required"` sigue sin producir `tool_calls` de verdad — el hecho que
  sostiene `tools: false` en el YAML, hasta ahora solo verificado a mano. Este archivo
  (`tests/test_vivo.py`) todavía no había corrido contra una instancia real; se resuelve
  la URL con `proveedores.cargar` (el mismo camino de producción, `/v1` incluido) en vez
  de reconstruirla a mano, para que corra igual sin importar cómo el operador configuró
  `CHATGPT_PROXY_URL`. **Corregido en la tercera revisión:** el test de chat mandaba un
  id CABLEADO (`"gpt-5-3-mini"`) — justo lo que esta Task existe para prohibir — y la
  revisión probó el costo: cuando ese id dejó de existir, el test de `tools` falló con
  el diagnóstico EQUIVOCADO ("¿volvió el 500 viejo?") en vez de decir que el id ya no
  existía. Ahora los dos tests descubren el catálogo real primero (`/v1/models`) y usan
  esos ids. Además, `tools: false` es una afirmación sobre TODAS las rutas descubiertas
  de `chatgpt`, no sobre un modelo puntual — un backend que ganara function calling en
  un modelo distinto del probado dejaría el test en verde con el YAML ya mintiendo, así
  que el test de `tools` ahora recorre TODOS los ids que el catálogo real trajo, no uno
  solo.

## 13. Fuera de alcance

Deliberadamente afuera, para no inflar la primera versión:

- Caché de respuestas
- Embeddings, imágenes, audio
- UI web (el `/v1/ranking` en JSON alcanza)
- Presupuestos en dinero (el tope de pago se cuenta en peticiones, no en tokens)
- Migrar los 7 bots de WhatsApp a este gateway — **posible seguimiento** interesante, porque
  hoy cada uno carga su copia de la clave de MiniMax y esto la centralizaría en un lugar.
  Pero es otro proyecto.

## 14. Riesgos conocidos

- **El tier anónimo de Kilo puede cerrarse en cualquier momento.** No hay contrato: es cortesía.
  Mitigación: OpenRouter con clave gratis como segundo proveedor, y el fallback de pago.
- **El recorte de razonamiento en streaming es la pieza más delicada** (§6.1): un buffer mal
  llevado corta texto legítimo o retiene la respuesta. Necesita pruebas con etiquetas
  partidas entre chunks.
- **Sondear consume la cuota que el servicio necesita.** A 5 h son ~100 peticiones/día, pero
  si se sube la cadencia hay que recalcular.
- **`blog` está saturado** (load ~4x, 2 GB en swap). El servicio es I/O-bound y liviano, pero
  es un proceso más en una máquina que ya sufre.
