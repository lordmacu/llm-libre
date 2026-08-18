import asyncio
import contextlib
import os
import random
from contextlib import asynccontextmanager

import httpx

from llm_libre.storage import Storage
from llm_libre.api import Estado, crear_app
from llm_libre.auth import PerKeyRateLimiter
from llm_libre.providers import load
from llm_libre.proxy import Proxy
from llm_libre.probing import cycle

YAML = os.getenv("PROVEEDORES_YAML", "proveedores.yaml")
RUTA_DB = os.getenv("RUTA_DB", "/datos/llm-libre.sqlite3")
HORAS_SALUD = float(os.getenv("SONDEO_SALUD_HORAS", "5"))
# Sorteo entre rutas empatadas. Encendido por defecto: con la bateria de
# calidad dandole 1.00 a todo el catalogo (ver quality_suite.py), el orden estricto
# manda SIEMPRE la misma ruta y quema la cuota de un solo proveedor mientras
# los demas miran. Apagarlo (ROTAR_EMPATES=false) devuelve el orden
# determinista de antes, util para depurar una respuesta rara.
ROTAR = os.getenv("ROTAR_EMPATES", "true").strip().lower() not in ("false", "0", "no")


def crear_estado() -> Estado:
    """Arma el Estado real del proceso: carga los proveedores desde el YAML +
    entorno, abre la DB SQLite (creando el esquema si falta) y comparte UN
    solo cliente httpx entre el proxy y el planificador de sondeo.

    Valida `LLM_LIBRE_API_KEYS` ANTES de tocar disco o red, y a proposito
    revienta fuerte (excepcion sin atrapar, el proceso no arranca) si no hay
    ninguna llave configurada. Sin esta llamada, un operador que se olvida
    esa variable (p.ej. en la UI de Coolify) obtiene un contenedor que
    arranca normal, `/health` que sigue diciendo "ok" (no depende de las
    llaves) y CADA peticion a /v1/* devolviendo 401 -- indistinguible en los
    logs de "este cliente mando una llave equivocada". Es la misma clase de
    fallo que /health honesto (Task 9) ya existe para evitar del lado de las
    rutas; esto la cierra del lado de la autenticacion: mejor un proceso que
    no arranca con una razon clara que uno que parece sano y rechaza a todo
    el mundo en silencio.
    """
    llaves = {k.strip() for k in os.getenv("LLM_LIBRE_API_KEYS", "").split(",") if k.strip()}
    if not llaves:
        raise RuntimeError(
            "LLM_LIBRE_API_KEYS no esta definida (o esta vacia): sin al menos "
            "una llave el servicio arrancaria pero rechazaria el 100% de las "
            "peticiones a /v1/* con 401 para cualquier llamador, mientras "
            "/health seguiria informando 'ok'. Definila con al menos una "
            "llave, separadas por coma si son varias -- por ejemplo: "
            "LLM_LIBRE_API_KEYS=una-llave-larga-y-secreta")
    proveedores = load(YAML, dict(os.environ))
    almacen = Storage(RUTA_DB)
    almacen.create_schema()
    http = httpx.AsyncClient()
    proxy = Proxy({p.id: p for p in proveedores}, almacen, http)
    estado = Estado(almacen=almacen, proxy=proxy, llaves=llaves,
                    tope_pago_diario=int(os.getenv("TOPE_PAGO_DIARIO", "200")),
                    limitador=PerKeyRateLimiter(int(os.getenv("LIMITE_POR_MINUTO", "60"))))
    estado.proveedores = proveedores
    estado.http = http
    estado.aleatorio = random.Random() if ROTAR else None
    return estado


async def planificador(estado: Estado) -> None:
    """Loop de fondo que corre `probing.cycle` sin parar, cada HORAS_SALUD.

    Desviacion respecto del brief original (Task 12): el brief traia el
    cuerpo entero del ciclo -- sincronizar catalogo, sondear salud, sondear
    calidad cada N pasadas, podar -- copiado linea por linea aca mismo. Esa
    logica ya existe, escrita y probada, como `probing.cycle(estado,
    contador)` desde la Task 11 (incluye alli mismo el "cada N ciclos" via
    `QUALITY_EVERY_N_CYCLES` y la retencion de 30 dias): dos copias de un
    mismo loop solo garantizan que se desincronicen con el tiempo. Este
    planificador se limita a invocarla.

    `ciclo` a proposito NO atrapa excepciones (ver su docstring en
    sondeo.py): esa responsabilidad es de quien lo llama en un loop
    infinito. Por eso el try/except vive ACA: un ciclo puntual que revienta
    (proveedor caido, DB bloqueada, lo que sea) se loguea y se sigue
    durmiendo hasta el proximo, en vez de tumbar esta tarea de fondo para
    siempre -- lo que dejaria al proceso sirviendo trafico con metricas cada
    vez mas viejas sin que nadie se entere, y sin tumbar el servicio.

    El contador local se llama `contador`, no `ciclo` (como lo nombraba el
    brief): ese nombre ya lo usa la funcion importada de `sondeo`, y
    reusarlo la taparia dentro de este loop.
    """
    contador = 0
    while True:
        try:
            await cycle(estado, contador)
        except Exception as e:  # el planificador nunca debe matar al servicio
            print(f"[sondeo] ciclo {contador} fallo: {e}", flush=True)
        contador += 1
        await asyncio.sleep(HORAS_SALUD * 3600)


estado = crear_estado()
app = crear_app(estado)


@asynccontextmanager
async def _ciclo_de_vida(_app):
    """Reemplaza el `@app.on_event("startup"/"shutdown")` del brief -- deprecado
    en la version de FastAPI/Starlette que trae este proyecto (ya emite
    warning en la suite) -- por el `lifespan` recomendado. `crear_app` (Task
    9) no expone un parametro para pasarlo en el constructor, asi que se
    engancha reasignando `app.router.lifespan_context` despues de crear la
    app: Starlette solo lee ese atributo cuando llega el mensaje ASGI de
    lifespan (al arrancar uvicorn), no en el momento de la asignacion, asi
    que reemplazarlo aca es equivalente a haberlo pasado en el constructor.
    """
    tarea = asyncio.create_task(planificador(estado))
    try:
        yield
    finally:
        tarea.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await tarea


app.router.lifespan_context = _ciclo_de_vida
