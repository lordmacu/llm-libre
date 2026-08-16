import os
import time

import httpx

from llm_libre.bateria import CASOS, evaluar
from llm_libre.catalogo import normalizar
from llm_libre.modelos import Ruta
from llm_libre.proveedores import Proveedor, rutas_fijas

PING = {"messages": [{"role": "user", "content": "ping"}], "max_tokens": 8,
        "temperature": 0}

# Cada cuanto ciclos se corre la bateria de calidad (gasta cuota gratis, por eso
# no en cada pasada) y cuanto se retiene la telemetria vieja antes de podarla.
CALIDAD_CADA_N_CICLOS = int(os.getenv("SONDEO_CALIDAD_CADA_N_CICLOS", "5"))
RETENCION_DIAS = 30


async def sincronizar_catalogo(http: httpx.AsyncClient, proveedores: list[Proveedor],
                               almacen, ahora: float) -> int:
    """Refresca el catalogo. Si un proveedor falla se conserva lo que ya habia:
    mejor un catalogo viejo que uno vacio."""
    descubiertas: list[Ruta] = []
    fallo = False
    for p in proveedores:
        if p.modelos_fijos:
            descubiertas.extend(rutas_fijas(p))
            continue
        if not p.modelos_path:
            continue
        cabeceras = dict(p.cabeceras_extra)
        if p.clave.strip():
            cabeceras["Authorization"] = "Bearer " + p.clave
        try:
            r = await http.get(p.base_url.rstrip("/") + p.modelos_path,
                               headers=cabeceras, timeout=30.0)
        except httpx.HTTPError:
            fallo = True
            continue
        if r.status_code != 200:
            fallo = True
            continue
        try:
            nuevas = normalizar(p.id, r.json())
        except (ValueError, TypeError, AttributeError, KeyError):
            # Cuerpo no-JSON (ValueError/JSONDecodeError) o JSON de una forma
            # inesperada -- p.ej. un error de auth disfrazado de 200, que deja
            # a normalizar() iterando algo que no son dicts de modelo
            # (AttributeError/KeyError/TypeError). Un proveedor roto no debe
            # tirar abajo la sincronizacion de los demas.
            fallo = True
            continue
        if not nuevas:
            # Un 200 con cero modelos utilizables es mucho mas probablemente un
            # proveedor roto (o momentaneamente sin catalogo) que la verdad: un
            # proveedor que normalmente sirve modelos y de golpe no reporta
            # ninguno no merece borrar lo que ya se sabia de el. Se trata igual
            # que cualquier otro fallo de ESTE proveedor -- no afecta a los
            # demas, que se siguen procesando y persistiendo normalmente.
            fallo = True
            continue
        descubiertas.extend(nuevas)
    if fallo and not descubiertas:
        return 0
    # Si un proveedor fallo no se desactiva nada (desactivar_faltantes=False):
    # las rutas descubiertas por los proveedores que SI respondieron son reales
    # y deben quedar con el `ahora` verdadero en visto_por_ultima_vez -- pisarlo
    # con 0.0 (como haria pasar `momento=0.0` a upsert_rutas) corromperia esa
    # marca de tiempo tambien para ellas, no solo para las que faltaron, y esa
    # marca es justo lo que permite notar despues que un modelo fue renombrado.
    almacen.upsert_rutas(descubiertas, ahora, desactivar_faltantes=not fallo)
    return len(descubiertas)


async def sondear_salud(proxy, almacen, rutas: list[Ruta], ahora: float) -> None:
    for ruta in rutas:
        t0 = time.monotonic()
        r = await proxy.completar([ruta], dict(PING), ahora)
        ms = int((time.monotonic() - t0) * 1000)
        almacen.registrar_sonda(ruta.clave, "salud", r.estado == 200, ms, ms,
                                r.estado, 0, 0, ahora)


async def sondear_calidad(proxy, almacen, rutas: list[Ruta], ahora: float) -> None:
    # Las rutas de pago no se sondean: no tiene sentido gastar plata puntuando
    # el escape de emergencia.
    for ruta in (r for r in rutas if r.tier == "gratis"):
        resultados = []
        for caso in CASOS:
            if caso.nombre == "tools" and not ruta.capacidades.tools:
                # Esta ruta no declara soporte de tools (ver Capacidades, viene
                # de /models o del YAML de fijos): pedirselo igual solo gastaria
                # cuota gratis para un fallo garantizado, y contarlo como caso
                # fallido mezclaria "no promete esta capacidad" con "la prometio
                # y la hizo mal" -- dos cosas distintas que no deben verse
                # igual en el puntaje de calidad. Se omite el caso entero: no
                # se llama al proxy y no cuenta ni para pasados ni para totales.
                continue
            cuerpo = dict(caso.cuerpo)
            r = await proxy.completar([ruta], cuerpo, ahora)
            resultados.append(r.estado == 200 and caso.verificar(r.json))
        pasados, totales = evaluar(resultados)
        almacen.registrar_sonda(ruta.clave, "calidad", pasados > 0, 0, 0, 200,
                                pasados, totales, ahora)


async def ciclo(estado, contador: int) -> None:
    """Una pasada completa del sondeo, para que el planificador (Task 12) la
    invoque en su loop: sincroniza el catalogo, sondea salud siempre, sondea
    calidad solo cada CALIDAD_CADA_N_CICLOS pasadas (misma cuota gratis que el
    trafico real) y poda telemetria vieja al final.

    No atrapa excepciones: esa responsabilidad es del planificador que llama
    esto en un loop infinito y no debe morirse porque un ciclo puntual falle.

    `estado` es cualquier objeto con `.http`, `.proveedores`, `.almacen` y
    `.proxy` (p.ej. `llm_libre.api.Estado`); no se importa ese tipo aqui para
    no crear una dependencia de sondeo hacia api.
    """
    ahora = time.time()
    await sincronizar_catalogo(estado.http, estado.proveedores, estado.almacen, ahora)
    rutas = estado.almacen.rutas_activas()
    await sondear_salud(estado.proxy, estado.almacen, rutas, ahora)
    if contador % CALIDAD_CADA_N_CICLOS == 0:
        await sondear_calidad(estado.proxy, estado.almacen, rutas, ahora)
    estado.almacen.podar(ahora - RETENCION_DIAS * 86400)
