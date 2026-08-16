from dataclasses import dataclass, field

import yaml

from llm_libre.modelos import Capacidades, Ruta


@dataclass(frozen=True)
class Proveedor:
    id: str
    tier: str            # "gratis" | "pago"
    dialecto: str        # "openai"
    base_url: str
    clave: str           # "" = anonimo
    modelos_path: str = ""
    cabeceras_extra: dict = field(default_factory=dict)
    modelos_fijos: list = field(default_factory=list)


def cargar(ruta_yaml: str, entorno: dict) -> list[Proveedor]:
    with open(ruta_yaml, encoding="utf-8") as f:
        datos = yaml.safe_load(f)
    return [Proveedor(
        id=p["id"], tier=p["tier"], dialecto=p["dialecto"], base_url=p["base_url"],
        clave=(entorno.get(p.get("clave_env", ""), "") or "").strip(),
        modelos_path=p.get("modelos_path", ""),
        cabeceras_extra=p.get("cabeceras_extra", {}) or {},
        modelos_fijos=p.get("modelos_fijos", []) or [],
    ) for p in datos["proveedores"]]


def rutas_fijas(p: Proveedor) -> list[Ruta]:
    return [Ruta(p.id, m["id"], p.tier,
                 Capacidades(tools=bool(m["tools"]), vision=bool(m["vision"]),
                             contexto=int(m["contexto"]), max_salida=int(m["max_salida"])))
            for m in p.modelos_fijos]
