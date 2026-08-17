from llm_libre.modelos import EXTENSIONES_GATEWAY
from llm_libre.proveedores import Proveedor, unir_ruta


def armar_peticion(p: Proveedor, cuerpo: dict,
                   modelo_real: str | None = None) -> tuple[str, dict, dict]:
    """Devuelve (url, cabeceras, cuerpo) listos para POST /chat/completions.

    El cuerpo devuelto es una copia somera (shallow copy) del cuerpo original,
    SIN las extensiones propias del gateway (`x_*`, ver EXTENSIONES_GATEWAY).
    Solo las claves de nivel superior pueden ser reasignadas por el llamador sin
    afectar el original; las estructuras anidadas (como `messages`) se comparten
    con el original y deben tratarse como de solo lectura durante reintentos.

    Se sacan solo las extensiones de ESTE gateway, no todo lo que empiece por
    `x_`: el contrato es passthrough, y un parametro nuevo del proveedor de
    turno tiene que poder viajar.
    """
    # unir_ruta parsea y reconstruye la URL, no concatena texto crudo: ver su
    # docstring para el bug que evita cuando base_url trae un query string
    # (via CHATGPT_PROXY_URL sin ruta propia).
    url = unir_ruta(p.base_url, "/chat/completions")
    cabeceras = {"Content-Type": "application/json", **p.cabeceras_extra}
    # Clave vacia => NO mandar Authorization. El tier anonimo de Kilo depende de esto.
    if p.clave.strip():
        cabeceras["Authorization"] = "Bearer " + p.clave
    nuevo = {k: v for k, v in cuerpo.items() if k not in EXTENSIONES_GATEWAY}
    if modelo_real is not None:
        nuevo["model"] = modelo_real
    return url, cabeceras, nuevo
