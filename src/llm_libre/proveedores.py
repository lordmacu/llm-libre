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
    # Concepto DISTINTO de tier (gratis|pago) y de perfil (rapido|balanceado|
    # potente): el orden manual en que el router prueba los proveedores antes
    # de mirar puntaje. Default 100 para que un proveedor que no lo declara
    # quede ultimo entre sus pares. Ver el comentario largo en modelos.Ruta.
    prioridad: int = 100


def _resolver_base_url(p: dict, entorno: dict) -> str:
    """`base_url` en el YAML es siempre el default; `base_url_env`, si esta
    declarada, nombra una variable de entorno que la pisa cuando trae algo
    (blank o ausente cae al default). Existe para que un proveedor propio
    (chatgpt-proxy) pueda apuntar a una direccion que todavia no esta fija
    sin tener que editar el YAML en cada despliegue."""
    env_var = p.get("base_url_env")
    if env_var:
        desde_entorno = (entorno.get(env_var, "") or "").strip()
        if desde_entorno:
            return desde_entorno
    return p["base_url"]


def cargar(ruta_yaml: str, entorno: dict) -> list[Proveedor]:
    with open(ruta_yaml, encoding="utf-8") as f:
        datos = yaml.safe_load(f)
    return [Proveedor(
        id=p["id"], tier=p["tier"], dialecto=p["dialecto"],
        base_url=_resolver_base_url(p, entorno),
        clave=(entorno.get(p.get("clave_env", ""), "") or "").strip(),
        modelos_path=p.get("modelos_path", ""),
        cabeceras_extra=p.get("cabeceras_extra", {}) or {},
        modelos_fijos=p.get("modelos_fijos", []) or [],
        prioridad=int(p.get("prioridad", 100)),
    ) for p in datos["proveedores"]]


def rutas_fijas(p: Proveedor) -> list[Ruta]:
    return [Ruta(p.id, m["id"], p.tier,
                 Capacidades(tools=bool(m["tools"]), vision=bool(m["vision"]),
                             contexto=int(m["contexto"]), max_salida=int(m["max_salida"])),
                 prioridad=p.prioridad)
            for m in p.modelos_fijos]
