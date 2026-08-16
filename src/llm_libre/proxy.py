import json
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from llm_libre.cliente import armar_peticion
from llm_libre.modelos import Ruta
from llm_libre.razonamiento import RecortadorStream, recortar

COOLDOWN_BASE_S = 60.0
COOLDOWN_TOPE_S = 3600.0
TIMEOUT_S = 90.0


@dataclass
class Respuesta:
    estado: int
    json: dict
    ruta: Ruta | None
    intentos: int
    razonamiento: str = ""


class Proxy:
    def __init__(self, proveedores: dict, almacen, cliente_http: httpx.AsyncClient):
        self.proveedores = proveedores
        self.almacen = almacen
        self.http = cliente_http
        self.cooldowns: dict[str, float] = {}
        self._castigos: dict[str, int] = {}

    async def completar(self, rutas: list[Ruta], cuerpo: dict, ahora: float,
                        crudo: bool = False) -> Respuesta:
        intentos = 0
        ultimo_error = None
        claves_del_pedido = {ruta.clave for ruta in rutas}
        for ruta in rutas:
            proveedor = self.proveedores[ruta.proveedor]
            url, cabeceras, payload = armar_peticion(proveedor, cuerpo, ruta.modelo_id)
            intentos += 1
            t0 = time.monotonic()
            try:
                resp = await self.http.post(url, headers=cabeceras, json=payload,
                                            timeout=TIMEOUT_S)
                codigo = resp.status_code
            except httpx.HTTPError as e:
                codigo, resp, ultimo_error = 0, None, str(e)
            ttft = int((time.monotonic() - t0) * 1000)

            # Un 200 con cuerpo no parseable (p.ej. una pagina de mantenimiento
            # HTML servida con status 200) no es un exito: se trata como intento
            # fallido, sin excepcion escapando y sin castigo (no es rate-limit,
            # esta rota).
            datos = None
            if codigo == 200:
                try:
                    datos = resp.json()
                except ValueError:
                    datos = None
                    ultimo_error = "200 con cuerpo no-JSON"

            exito = codigo == 200 and datos is not None
            self.almacen.registrar_evento(ruta.clave, exito, ttft, codigo, ahora)

            if exito:
                self._castigos.pop(ruta.clave, None)
                self.cooldowns.pop(ruta.clave, None)
                razon = "" if crudo else self._limpiar(datos)
                return Respuesta(200, datos, ruta, intentos, razon)

            if codigo == 429:
                self._castigar(ruta.clave, ahora)
            ultimo_error = ultimo_error or f"HTTP {codigo}"

        # Solo cuentan los cooldowns de las rutas de ESTE pedido: el proxy vive
        # mas alla de una sola llamada y puede tener castigadas rutas ajenas a
        # esta cadena, cuyo vencimiento no le sirve de nada al que esta pidiendo.
        cooldowns_del_pedido = {c: v for c, v in self.cooldowns.items()
                                if c in claves_del_pedido}
        return Respuesta(503, {"error": {
            "message": "sin rutas disponibles",
            "detalle": ultimo_error,
            "proxima_liberacion": (min(cooldowns_del_pedido.values())
                                   if cooldowns_del_pedido else None),
        }}, None, intentos)

    async def completar_stream(self, rutas: list[Ruta], cuerpo: dict, ahora: float,
                               crudo: bool = False,
                               en_ruta_comprometida: Callable[[Ruta], None] | None = None):
        """Emite lineas SSE ya recortadas, terminando siempre en `data: [DONE]`.

        Hace failover solo ANTES del primer byte util: una vez que al cliente le
        llego contenido de una ruta, cambiar de modelo mezclaria dos respuestas
        distintas en un mismo stream. Por eso una falla de red DESPUES de emitir
        no reintenta la siguiente ruta: cierra el stream ahi mismo.

        `en_ruta_comprometida`, si se pasa, se llama COMO MUCHO una vez por
        llamada a este generador: exactamente cuando (y si) una ruta queda
        confirmada como la que de verdad sirvio la peticion. Se dispara desde
        el mismo lugar que ya decide "esto fue un exito real" para la
        telemetria (`_registrar_exito_una_vez`, mas abajo) -- no desde el
        status 200 crudo, porque un 200 que muere sin emitir nada antes del
        primer byte util TODAVIA hace failover a la siguiente ruta (ver mas
        arriba) y ahi no hubo servicio real. Sirve para que el llamador pueda
        contar uso de pago (u otra cosa) atado a "esta ruta sirvio", sin
        arriesgarse a contar de mas ni de menos.
        """
        for ruta in rutas:
            proveedor = self.proveedores[ruta.proveedor]
            url, cabeceras, payload = armar_peticion(proveedor, cuerpo, ruta.modelo_id)
            payload["stream"] = True
            t0 = time.monotonic()
            emitido = False          # ya salio algun chunk util hacia el cliente
            evento_registrado = False  # ya se conto la telemetria de este intento

            def _registrar_exito_una_vez() -> None:
                # Exactamente un evento por intento, nunca cero ni dos: se
                # dispara la primera vez que hay algo util que mandar (asi el
                # ttft mide el primer token real, no el cierre del stream) o,
                # si nunca hubo nada util, al terminar el intento igual (mas
                # abajo) para no desaparecer de la historia de la ruta.
                nonlocal evento_registrado
                if not evento_registrado:
                    self.almacen.registrar_evento(
                        ruta.clave, True, int((time.monotonic() - t0) * 1000),
                        200, ahora)
                    evento_registrado = True
                    if en_ruta_comprometida is not None:
                        en_ruta_comprometida(ruta)

            try:
                async with self.http.stream("POST", url, headers=cabeceras, json=payload,
                                            timeout=TIMEOUT_S) as resp:
                    if resp.status_code != 200:
                        if resp.status_code == 429:
                            self._castigar(ruta.clave, ahora)
                        self.almacen.registrar_evento(ruta.clave, False, 0,
                                                      resp.status_code, ahora)
                        continue
                    self._castigos.pop(ruta.clave, None)
                    self.cooldowns.pop(ruta.clave, None)
                    rec = RecortadorStream()
                    async for linea in resp.aiter_lines():
                        if not linea.startswith("data:"):
                            continue
                        carga = linea[5:].strip()
                        if carga == "[DONE]":
                            break
                        try:
                            obj = json.loads(carga)
                        except json.JSONDecodeError:
                            continue
                        delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                        if not crudo and isinstance(delta.get("content"), str):
                            limpio = rec.alimentar(delta["content"])
                            # Un chunk de tool_calls (o el de role inicial) suele
                            # viajar con content="": miramos la PRESENCIA de otras
                            # claves, no su valor, porque algo como "tool_calls": []
                            # (valor falsy pero presente) igual es util para el
                            # cliente y no se puede tirar junto con el contenido.
                            otras = {k: v for k, v in delta.items() if k != "content"}
                            if not limpio and not otras:
                                continue    # nada util que mandar en este chunk
                            delta["content"] = limpio
                        _registrar_exito_una_vez()
                        emitido = True
                        yield f"data: {json.dumps(obj)}\n\n"
                    resto = rec.cerrar()
                    if resto:
                        _registrar_exito_una_vez()
                        emitido = True
                        yield ('data: {"choices":[{"delta":{"content":%s}}]}\n\n'
                               % json.dumps(resto))
                    # La conexion funciono (200, stream leido sin excepcion) aunque
                    # nunca hubiera nada util que mandar (p.ej. todo era
                    # razonamiento filtrado): sigue siendo un intento exitoso a
                    # nivel HTTP, y sin este registro la ruta desaparece por
                    # completo de su propia historia de confiabilidad.
                    _registrar_exito_una_vez()
                    yield "data: [DONE]\n\n"
                    return
            except httpx.HTTPError:
                if not evento_registrado:
                    self.almacen.registrar_evento(ruta.clave, False, 0, 0, ahora)
                if emitido:
                    yield "data: [DONE]\n\n"
                    return
                continue

        yield 'data: {"error":{"message":"sin rutas disponibles"}}\n\n'
        yield "data: [DONE]\n\n"

    def _castigar(self, clave: str, ahora: float) -> None:
        n = self._castigos.get(clave, 0) + 1
        self._castigos[clave] = n
        self.cooldowns[clave] = ahora + min(COOLDOWN_BASE_S * (2 ** (n - 1)), COOLDOWN_TOPE_S)

    @staticmethod
    def _limpiar(datos: dict) -> str:
        razon_total = ""
        for eleccion in datos.get("choices", []):
            msg = eleccion.get("message") or {}
            contenido = msg.get("content")
            if isinstance(contenido, str):
                limpio, razon = recortar(contenido)
                msg["content"] = limpio
                razon_total += razon
        return razon_total
