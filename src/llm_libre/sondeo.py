import logging
import os
import time

import httpx

from llm_libre.bateria import CASOS, evaluar
from llm_libre.catalogo import normalizar
from llm_libre.modelos import Ruta
from llm_libre.proveedores import Proveedor, rutas_fijas, unir_ruta
from llm_libre.proxy import PING

log = logging.getLogger(__name__)

# `PING` vive en proxy.py desde round 8: proxy tambien dispara sondas por su
# cuenta (bajo demanda, ver UMBRAL_SOSPECHA ahi) y necesita el MISMO payload
# fijo que esta sonda periodica para que las dos sean, literalmente, el mismo
# pedido -- se importa de ahi en vez de duplicarlo. Se re-expone aca (el
# import de arriba) para no romper a quien ya hacia
# `from llm_libre.sondeo import PING`.

# Cada cuanto ciclos se corre la bateria de calidad (gasta cuota gratis, por eso
# no en cada pasada) y cuanto se retiene la telemetria vieja antes de podarla.
CALIDAD_CADA_N_CICLOS = int(os.getenv("SONDEO_CALIDAD_CADA_N_CICLOS", "5"))
RETENCION_DIAS = 30


async def sincronizar_catalogo(http: httpx.AsyncClient, proveedores: list[Proveedor],
                               almacen, ahora: float) -> int:
    """Refresca el catalogo, proveedor por proveedor.

    La baja de rutas que ya no aparecen se decide POR PROVEEDOR, no para toda
    la sincronizacion: cada proveedor que responde bien se persiste (y sus
    rutas que desaparecieron de SU catalogo se desactivan) en el momento,
    usando el scope `proveedor=` de Almacen.upsert_rutas -- sin ese scope, un
    UPDATE de "lo que no se vio se desactiva" no filtrado por proveedor
    tambien apagaria las rutas de proveedores ajenos a esta pasada (su
    visto_por_ultima_vez siempre es mas vieja que `ahora`). Que OTRO
    proveedor este roto o vacio en la misma pasada no debe frenar esa baja: el
    caso que este proyecto existe para detectar es justo un modelo que
    desaparecio de su proveedor, y no puede quedar dependiendo de que todos
    los demas proveedores tengan un dia perfecto.

    Un proveedor que falla (red, status distinto de 200, cuerpo no
    parseable/con forma inesperada, o un 200 con cero modelos utilizables --
    mas probable un proveedor roto que un catalogo genuinamente vacio) no se
    toca en absoluto: mejor conservar lo que ya se sabia de EL que borrarlo.

    Cada uno de esos cuatro caminos DEJA UN WARNING con el proveedor y la
    razon. Sin eso, conservar el catalogo viejo -- que es lo correcto -- se
    vuelve indistinguible de que todo funcione: el catalogo de ese proveedor
    se congela para siempre y nadie se entera. Esta es justo la capa que
    existe para que un catalogo no se pudra sin aviso (§1 del diseno).

    Antes del loop por proveedor, un barrido aparte
    (`Almacen.desactivar_proveedores_no_registrados`) apaga las rutas de
    cualquier proveedor que YA NO este en `proveedores` -- necesario porque
    el loop de abajo solo puede dar de baja, via su scope, lo que SIGUE en
    el registro; un proveedor sacado del YAML (p.ej. `openrouter` sin
    `OPENROUTER_API_KEY`, cuyas rutas terminaban todas en cooldown por 401)
    nunca vuelve a pasar por ese loop, asi que sin el barrido sus rutas
    quedarian `activa=1` para siempre: visibles en `GET /v1/models` y
    `GET /v1/ranking`, y elegibles como candidatas que fallarian siempre.
    """
    conocidos = {p.id for p in proveedores}
    desactivadas = almacen.desactivar_proveedores_no_registrados(conocidos)
    if desactivadas:
        log.warning(
            "catalogo: %d ruta(s) desactivadas porque su proveedor ya no esta "
            "en proveedores.yaml (no se borran -- el historico sirve para "
            "detectar renombres, ver Almacen.upsert_rutas)", desactivadas)
    total = 0
    for p in proveedores:
        if p.modelos_fijos:
            rutas = rutas_fijas(p)
            almacen.upsert_rutas(rutas, ahora, desactivar_faltantes=True, proveedor=p.id)
            total += len(rutas)
            continue
        if not p.modelos_path:
            continue
        cabeceras = dict(p.cabeceras_extra)
        if p.clave.strip():
            cabeceras["Authorization"] = "Bearer " + p.clave
        try:
            # unir_ruta parsea y reconstruye, no concatena texto crudo: ver
            # su docstring en proveedores.py para el bug que evita.
            r = await http.get(unir_ruta(p.base_url, p.modelos_path),
                               headers=cabeceras, timeout=30.0)
        except httpx.HTTPError as e:
            log.warning("catalogo de %s: no se pudo consultar %s (%s: %s). "
                        "Se conserva el catalogo anterior.",
                        p.id, p.modelos_path, type(e).__name__, e)
            continue
        if r.status_code != 200:
            log.warning("catalogo de %s: %s respondio HTTP %s. "
                        "Se conserva el catalogo anterior.",
                        p.id, p.modelos_path, r.status_code)
            continue
        try:
            nuevas = normalizar(p.id, r.json(), p.prioridad, p.capacidades_por_defecto,
                                p.excepciones)
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            # Cuerpo no-JSON (ValueError/JSONDecodeError) o JSON de una forma
            # inesperada -- p.ej. un error de auth disfrazado de 200, que deja
            # a normalizar() iterando algo que no son dicts de modelo
            # (AttributeError/KeyError/TypeError). Un proveedor roto no debe
            # tirar abajo la sincronizacion de los demas.
            log.warning("catalogo de %s: no se pudo interpretar la respuesta de %s "
                        "(%s: %s). Se conserva el catalogo anterior.",
                        p.id, p.modelos_path, type(e).__name__, e)
            continue
        if not nuevas:
            # Un 200 con cero modelos utilizables no autoriza a apagar lo que
            # ya se sabia de este proveedor: se trata igual que cualquier otro
            # fallo de ESTE proveedor puntual, sin afectar a los demas.
            log.warning("catalogo de %s: HTTP 200 con cero modelos utilizables. "
                        "Mas probable un proveedor roto que un catalogo vacio de "
                        "verdad: se conserva el catalogo anterior.", p.id)
            continue
        # Este proveedor respondio bien: se persiste YA, con su propio scope,
        # sin esperar a saber que paso con el resto de la lista.
        almacen.upsert_rutas(nuevas, ahora, desactivar_faltantes=True, proveedor=p.id)
        total += len(nuevas)
    return total


async def sondear_salud(proxy, almacen, rutas: list[Ruta], ahora: float) -> None:
    # Igual que sondear_calidad: las rutas de pago NO se sondean (§8 del
    # diseno). Esta funcion recibe rutas_activas(), que incluye
    # minimax/MiniMax-M3, y cada pasada le pegaba: ~5 llamadas facturables por
    # dia que ademas no pasan por sumar_uso_pago, asi que no aparecen ni en
    # /v1/uso ni contra TOPE_PAGO_DIARIO. Plata real, invisible.
    for ruta in (r for r in rutas if r.tier == "gratis"):
        t0 = time.monotonic()
        # HIGH 1 (round 9): sin `es_sonda=True` un fallo aca solo acumulaba
        # SOSPECHA (round 8) -- esta sonda periodica corre UNA vez cada 5h
        # por ruta, muy por debajo de cualquier umbral de sospecha, asi que
        # nunca alcanzaba a castigar nada: 20 sondas periodicas contra una
        # ruta muerta, 20 filas `sondas ok=0`, cero cooldowns. Una sonda
        # (periodica o bajo demanda) es la unica fuente que puede castigar
        # sin ambiguedad -- ver el comentario de cabecera de
        # UMBRAL_SOSPECHA en proxy.py.
        r = await proxy.completar([ruta], dict(PING), ahora, es_sonda=True)
        ms = int((time.monotonic() - t0) * 1000)
        # `ttft_ms=0`, no `ms`: esta sonda es no-streaming, asi que lo que midio
        # es un round-trip completo y no un time-to-first-token. Escribirlo en
        # la columna de ttft mezclaba dos magnitudes distintas en un mismo p50
        # (ver el comentario de cabecera de almacen.py).
        #
        # `codigo_http` guarda el status del PROVEEDOR, no el `estado` que
        # sintetiza el gateway. Con el 503 sintetico, la tabla `sondas` no podia
        # distinguir "el proveedor esta caido" de "el proveedor dijo 200 y vino
        # vacio" -- justo la diferencia que hacia falta para diagnosticar por
        # que el ping de 8 tokens mataba rutas sanas.
        #
        # Round 10, fix chico: un 429 contra la sonda NO se graba aca -- ya
        # tiene su propio castigo proporcional (Proxy._castigar_429, adentro
        # de completar()), y es evidencia de que la ruta esta rate-limitada
        # AHORA, no de que este rota. Grabarlo tambien como sonda de salud
        # fallida la confundiria con una ruta caida: dos 429 seguidos
        # bastarian para que tiene_evidencia_de_vida (round 9) la de por
        # muerta -- una senal de capacidad momentanea no es evidencia de
        # muerte.
        if r.codigo_upstream != 429:
            almacen.registrar_sonda(ruta.clave, "salud", r.estado == 200, ms, 0,
                                    r.codigo_upstream, 0, 0, ahora)


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
            # Round 10, fix chico: mismo hueco de wiring que HIGH 1 (round
            # 9), una funcion mas alla. `caso.cuerpo` es tan gateway-autor
            # como `PING` -- son CASOS fijos de bateria.py, nunca algo que
            # un cliente real escriba -- asi que un fallo aca es evidencia
            # de la ruta igual de inequivoca. Sin `es_sonda=True`, un fallo
            # de bateria alimentaba `_sospechar` (pensada para trafico de
            # CLIENTE) y gastaba cupo del presupuesto de sondas bajo
            # demanda, escaso y compartido con el trafico real.
            r = await proxy.completar([ruta], cuerpo, ahora, es_sonda=True)
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
