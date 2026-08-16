from llm_libre.modelos import Capacidades, Ruta


def normalizar(proveedor: str, datos: dict | list) -> list[Ruta]:
    """Convierte la respuesta de /models en rutas gratis utilizables para chat."""
    items = datos.get("data", datos) if isinstance(datos, dict) else datos
    rutas: list[Ruta] = []
    for m in items:
        if not _es_gratis(m):
            continue
        arch = m.get("architecture") or {}
        # Exigir salida SOLO texto: los modelos de musica (lyria) tambien traen
        # "text" entre sus salidas, asi que "contiene texto" los dejaria pasar.
        salidas = set(arch.get("output_modalities") or ["text"])
        if salidas != {"text"}:
            continue
        soportados = m.get("supported_parameters") or []
        top = m.get("top_provider") or {}
        rutas.append(Ruta(
            proveedor=proveedor,
            modelo_id=m["id"],
            tier="gratis",
            capacidades=Capacidades(
                tools="tools" in soportados,
                vision="image" in (arch.get("input_modalities") or []),
                contexto=int(m.get("context_length") or top.get("context_length") or 0),
                max_salida=int(top.get("max_completion_tokens") or 0),
            ),
        ))
    return rutas


def _es_gratis(m: dict) -> bool:
    precio = (m.get("pricing") or {}).get("prompt")
    try:
        return float(precio) == 0.0
    except (TypeError, ValueError):
        return False
