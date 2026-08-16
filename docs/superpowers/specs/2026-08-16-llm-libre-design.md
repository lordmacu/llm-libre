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
**`tools` NO soportado: devuelve HTTP 500**, no un fallo elegante — por eso sus
modelos se declaran `tools: false` a mano en `proveedores.yaml`, igual que MiniMax
(su `/v1/models` tampoco trae metadatos de capacidad). `temperature`/`max_tokens`/etc.
se aceptan y se ignoran. **Filtra el modo "canvas" de ChatGPT al `content`**, con
marcas `:::palabra{...}` … `:::` que envuelven la respuesta real (no algo para
descartar, a diferencia de `<think>`) — el gateway las desenvuelve, en bloque y en
streaming (§6.1).

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

```yaml
proveedores:
  - id: chatgpt
    tier: gratis
    prioridad: 0                 # antes que todo lo demas gratis
    dialecto: openai
    base_url_env: CHATGPT_PROXY_URL   # la direccion real viene del entorno
    base_url: http://127.0.0.1:8888   # default si la variable no esta
    modelos_fijos: [gpt-5-6, gpt-5-5, gpt-5-6-mini, gpt-5-5-mini, gpt-5-3-mini]
    # tools:false en los cinco -- mandarle tools devuelve HTTP 500, no un
    # fallo elegante (verificado 2026-08-16). No se declara "auto" (colisiona
    # con el alias de llm-libre) ni los alias legacy gpt-4o/gpt-4/gpt-3.5.
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
  `ttft_ms`, `codigo_http`, `llave`. Alimenta el ranking entre sondas.
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

**Cercas de canvas (Task 13).** `chatgpt-proxy` filtra el modo "canvas" de ChatGPT al
`content` con marcas `:::palabra{...atributos...}` … `:::` envolviendo la respuesta.
**Al revés que `<think>`: el contenido de ADENTRO es la respuesta**, no algo para
descartar — solo se quitan las dos líneas de marca (apertura y cierre), conservando
todo lo demás carácter por carácter. Mismo requisito que el recorte de razonamiento: la
marca puede llegar partida entre chunks en streaming. Una cerca que nunca cierra no
pierde contenido (solo se descuenta la marca de apertura, ya confirmada); un `:::` que
no está al inicio de línea, o al que no le sigue una palabra, nunca se confunde con una
marca real.

### `GET /v1/models`

El catálogo normalizado en formato OpenAI (para que los SDK lo listen), más los alias `auto*`.

### `GET /v1/ranking`

Propio, no OpenAI. Puntaje de cada ruta con **sus componentes desglosados** (calidad,
confiabilidad, latencia, cooldown) y la fecha de su última sonda. Existe para poder auditar
por qué el router eligió lo que eligió.

### `GET /v1/uso`

Consumo por llave, con el gasto de pago del día contra su tope.

### `GET /v1/health`

**Honesto**: responde `ok` solo si hay al menos una ruta viva capaz de servir. Si todo lo
gratis está caído, dice `degradado`; si no hay nada, `caido`.

> Esto es lección directa del incidente del gateway de arkiv (2026-08-15): `/v1/health`
> decía `ok` mientras todos los endpoints autenticados daban 503 durante ~3 horas, porque
> solo comprobaba que el proceso estuviera arriba. Un health que no puede fallar no sirve.

**Una ruta cuenta como sana si no está en cooldown Y su confiabilidad reciente supera un
piso.** Las dos condiciones, no una. Mirar solo el cooldown reproduce el mismo incidente
que motivó el endpoint: los cooldowns se llenan únicamente con 429, así que una ruta que
devuelve 500 para siempre nunca entra en cooldown y el health la sigue contando como viva.
Se detectó exactamente así durante la implementación (2026-08-16), con 10 peticiones reales
fallando y cayendo a pago mientras `/health` decía `ok`. Una ruta sin telemetría todavía
cuenta como sana: arranca con el valor neutro, y tratar lo desconocido como roto la sacaría
de rotación antes de haber tenido su primera oportunidad.

## 7. Ranking y selección

Por ruta:

- **`calidad`** (0–1) — fracción de casos de la batería que pasó en la última evaluación.
  Las rutas nunca evaluadas arrancan en un valor neutro configurable, no en 0 (si no,
  jamás las elegiría y nunca se evaluarían).
- **`confiabilidad`** (0–1) — EWMA de éxitos, mezclando sondas y tráfico real.
- **`latencia`** — p50 de time-to-first-token de las últimas N observaciones.
- **`cooldown`** — una ruta que devolvió 429 queda excluida hasta que expire su castigo
  (backoff exponencial con tope).

`puntaje = calidad^wc · confiabilidad^wr · f(latencia)^wl`, con los pesos según el **perfil**
pedido: `rapido` pondera latencia, `potente` pondera calidad y contexto, `balanceado` (el
default) reparte parejo.

El router devuelve una **lista ordenada**, no un ganador único: el proxy baja por ella ante
fallos. La clave de orden completa (Task 13) es
`(tier == "pago", prioridad, no-medida, -puntaje)`:

1. **`tier == "pago"` decide primero, siempre.** Las rutas de pago van siempre al final, y
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

## 9. Errores

| Situación | Respuesta |
|---|---|
| Ninguna ruta cumple las capacidades pedidas | `400` con qué se pidió y qué hay disponible |
| Todas las candidatas caídas o en cooldown, sin pago disponible | `503` con cuándo se libera la primera |
| Proveedor devuelve 429 | Transparente: cooldown y siguiente ruta |
| Llave superó su tope de pago diario | `503`, nunca un cobro silencioso |
| Modelo pedido explícitamente que ya no existe | `404` con los ids vigentes más parecidos |

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
  el caso `lyria` (precio 0 pero modelo de música → debe descartarse).
- **Recorte de razonamiento:** con la etiqueta partida en todas las posiciones posibles
  entre chunks, bloques anidados, y una etiqueta que nunca cierra (no debe tragarse la
  respuesta entera ni colgar el stream).
- **Cercas de canvas (Task 13):** mismo estándar que el recorte de razonamiento — la marca
  partida en cada posición posible entre chunks, una cerca que nunca cierra (no debe
  perder contenido), texto con `:::` que no es una cerca real (no debe tocarse), y
  contenido normal intacto.
- **Migración de esquema (Task 13):** `rutas.prioridad` verificada contra una base con el
  esquema viejo (sin esa columna) y filas ya adentro — no debe reventar al abrir, la fila
  preexistente migra al default, y la base sigue siendo escribible después.
- **Integración real:** un puñado marcado para no correr en CI, contra los proveedores vivos
  — incluye `chatgpt-proxy` (Task 13), que se salta limpio si `CHATGPT_PROXY_URL` no está
  configurada.

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
