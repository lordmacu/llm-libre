import time
from dataclasses import dataclass

import httpx

from llm_libre.cliente import armar_peticion
from llm_libre.modelos import Ruta
from llm_libre.razonamiento import recortar

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
