from llm_libre.proveedores import Proveedor


def armar_peticion(p: Proveedor, cuerpo: dict,
                   modelo_real: str | None = None) -> tuple[str, dict, dict]:
    """Devuelve (url, cabeceras, cuerpo) listos para POST /chat/completions.

    El cuerpo devuelto es una copia somera (shallow copy) del cuerpo original.
    Solo las claves de nivel superior pueden ser reasignadas por el llamador sin
    afectar el original; las estructuras anidadas (como `messages`) se comparten
    con el original y deben tratarse como de solo lectura durante reintentos.
    """
    url = p.base_url.rstrip("/") + "/chat/completions"
    cabeceras = {"Content-Type": "application/json", **p.cabeceras_extra}
    # Clave vacia => NO mandar Authorization. El tier anonimo de Kilo depende de esto.
    if p.clave.strip():
        cabeceras["Authorization"] = "Bearer " + p.clave
    nuevo = dict(cuerpo)
    if modelo_real is not None:
        nuevo["model"] = modelo_real
    return url, cabeceras, nuevo
