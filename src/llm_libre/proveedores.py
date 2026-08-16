import logging
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import yaml

from llm_libre.modelos import Capacidades, Ruta

log = logging.getLogger(__name__)


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
    # Un TERCER patron de registro, junto a "todo descubierto" (Kilo/
    # OpenRouter, via modelos_path) y "todo declarado" (MiniMax, via
    # modelos_fijos): un proveedor cuyo /models trae IDS pero NINGUN metadato
    # de capacidad (chatgpt-proxy). None => modo normal (capacidades se
    # descubren desde /models, como siempre). Si esta declarado, catalogo.
    # normalizar aplica estas capacidades a CADA id que descubra y saltea los
    # chequeos de precio/modalidad -- pero los IDS siguen viniendo de
    # /models, nunca de aca: sigue siendo descubrimiento, no una lista
    # cableada con otro nombre.
    capacidades_por_defecto: Capacidades | None = None
    # Revision de Task 13, hallazgo 1: el desenvuelto de cercas de canvas
    # (':::palabra{...}' ... ':::') era GLOBAL, pero ':::nota{...}' es
    # tambien sintaxis Docusaurus/MDX estandar -- aplicarlo a ciegas le
    # arranca las marcas a CUALQUIER proveedor (verificado en vivo contra
    # Kilo). Default False: solo chatgpt-proxy, que de verdad filtra el modo
    # canvas al content, lo declara en true.
    desenvuelve_canvas: bool = False
    # Revision de Task 13, hallazgo 2 (de paso): None = usar el TIMEOUT_S
    # global de proxy.py (el comportamiento de siempre). Declarado, permite
    # acotar el peor caso de UN proveedor puntual sin bajarle el timeout a
    # todos -- util para uno que puede colgarse (ver el cooldown de fallos
    # duros en Proxy, la razon por la que esto se agrego).
    timeout_s: float | None = None


def _resolver_base_url(p: dict, entorno: dict) -> str:
    """`base_url` en el YAML es siempre el default; `base_url_env`, si esta
    declarada, nombra una variable de entorno que la pisa cuando trae algo
    (blank o ausente cae al default). Existe para que un proveedor propio
    (chatgpt-proxy) pueda apuntar a una direccion que todavia no esta fija
    sin tener que editar el YAML en cada despliegue.

    Si el `base_url` por default trae una RUTA (p.ej. el "/v1" de chatgpt) y
    la variable de entorno no la trae, se la agregamos: el operador solo
    tiene que acordarse del host, no de la ruta interna del proveedor. Sin
    esto, "CHATGPT_PROXY_URL=https://blog.example" (sin /v1) deja cada
    peticion de chat pegando a una URL que no existe -- el MISMO footgun que
    ya se arreglo del lado del YAML (base_url original sin /v1, Task 13),
    sobreviviendo del lado del entorno. Se normaliza en vez de reventar al
    arrancar porque hay una unica interpretacion correcta (el sufijo que el
    propio YAML ya declara); se loguea igual, para que quede visible."""
    env_var = p.get("base_url_env")
    if not env_var:
        return p["base_url"]
    desde_entorno = (entorno.get(env_var, "") or "").strip()
    if not desde_entorno:
        return p["base_url"]
    desde_entorno = desde_entorno.rstrip("/")
    sufijo = urlsplit(p["base_url"]).path.rstrip("/")
    if sufijo and not desde_entorno.endswith(sufijo):
        log.warning(
            "%s: %s='%s' no trae el sufijo '%s' que proveedores.yaml declara "
            "para este proveedor; se agrega automaticamente (%s%s). Definir "
            "la variable con el sufijo ya incluido evita este aviso.",
            p["id"], env_var, desde_entorno, sufijo, desde_entorno, sufijo)
        desde_entorno += sufijo
    return desde_entorno


def _capacidades_por_defecto(p: dict) -> Capacidades | None:
    datos = p.get("capacidades_por_defecto")
    if not datos:
        return None
    return Capacidades(tools=bool(datos["tools"]), vision=bool(datos["vision"]),
                       contexto=int(datos["contexto"]), max_salida=int(datos["max_salida"]))


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
        capacidades_por_defecto=_capacidades_por_defecto(p),
        desenvuelve_canvas=bool(p.get("desenvuelve_canvas", False)),
        timeout_s=(float(p["timeout_s"]) if p.get("timeout_s") is not None else None),
    ) for p in datos["proveedores"]]


def rutas_fijas(p: Proveedor) -> list[Ruta]:
    return [Ruta(p.id, m["id"], p.tier,
                 Capacidades(tools=bool(m["tools"]), vision=bool(m["vision"]),
                             contexto=int(m["contexto"]), max_salida=int(m["max_salida"])),
                 prioridad=p.prioridad)
            for m in p.modelos_fijos]
