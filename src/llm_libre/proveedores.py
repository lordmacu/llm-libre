import logging
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

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
    # Excepciones por id de modelo sobre `capacidades_por_defecto`: solo los
    # campos que difieren. Ver `_excepciones` para el porque.
    excepciones: dict = field(default_factory=dict)
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
    # Tool-calling emulation via prompt injection: True = this provider does not
    # support native tools, but inject_into_body converts `tools` to system-prompt
    # text and proxy.py detects JSON tool-call responses and converts them to the
    # OpenAI tool_calls format. See tool_emulator.py.
    emula_tools: bool = False


def unir_ruta(base_url: str, sufijo: str) -> str:
    """Une un sufijo de ruta (p.ej. "/chat/completions", "/models") a
    `base_url` PARSEANDO la URL, no concatenando texto crudo.

    `base_url` puede traer un query string -- via `CHATGPT_PROXY_URL` sin
    ruta propia, ver `_resolver_base_url` -- y una concatenacion cruda
    (`base_url.rstrip("/") + sufijo`) pega el sufijo DENTRO del valor de la
    query en vez de agregarlo al path: `"...:8888?token=abc" +
    "/chat/completions"` da `"...:8888?token=abc/chat/completions"` en vez
    de `"...:8888/chat/completions?token=abc"`. Round 5: este bug se
    arreglo primero solo en `_resolver_base_url` (donde se GUARDA
    `Proveedor.base_url`), pero `client.build_request` y
    `sondeo.sincronizar_catalogo` seguian concatenando texto al construir
    la URL FINAL que de verdad se manda por la red -- la astilla se habia
    movido una capa mas abajo. Los dos ahora pasan por aca."""
    partes = urlsplit(base_url)
    return urlunsplit(partes._replace(path=partes.path.rstrip("/") + sufijo))


def _resolver_base_url(p: dict, entorno: dict) -> str:
    """`base_url` en el YAML es siempre el default; `base_url_env`, si esta
    declarada, nombra una variable de entorno que la pisa cuando trae algo
    (blank o ausente cae al default). Existe para que un proveedor propio
    (chatgpt-proxy) pueda apuntar a una direccion que todavia no esta fija
    sin tener que editar el YAML en cada despliegue.

    Si el `base_url` por default trae una RUTA (p.ej. el "/v1" de chatgpt) y
    la variable de entorno NO TRAE NINGUNA RUTA PROPIA (esta vacia o es solo
    "/", o sea el operador puso nada mas que el host), se la agregamos: el
    operador solo tiene que acordarse del host, no de la ruta interna del
    proveedor. Sin esto, "CHATGPT_PROXY_URL=https://blog.example" (sin /v1)
    deja cada peticion de chat pegando a una URL que no existe -- el MISMO
    footgun que ya se arreglo del lado del YAML (base_url original sin /v1,
    Task 13), sobreviviendo del lado del entorno.

    Si la variable de entorno SI trae una ruta propia (p.ej. un mount de
    reverse proxy que sirve el chat bajo su propio prefijo) y esa ruta no
    coincide con el sufijo esperado, NO se toca: agregarla igual convertiria
    ".../v2" (una ruta que el operador eligio a proposito) en
    ".../v2/v1/chat/completions", sin ninguna forma de decir "no, dejala
    como esta". El operador dijo lo que quiso decir; solo se avisa, por si
    fue sin querer -- ver el aviso mas abajo."""
    env_var = p.get("base_url_env")
    if not env_var:
        return p["base_url"]
    desde_entorno = (entorno.get(env_var, "") or "").strip().rstrip("/")
    if not desde_entorno:
        return p["base_url"]
    sufijo = urlsplit(p["base_url"]).path.rstrip("/")
    if not sufijo:
        return desde_entorno
    partes = urlsplit(desde_entorno)
    ruta_entorno = partes.path.rstrip("/")
    if not ruta_entorno:
        log.warning(
            "%s: %s='%s' no trae ninguna ruta; se agrega el sufijo '%s' que "
            "proveedores.yaml declara para este proveedor. Definir la "
            "variable con el sufijo ya incluido evita este aviso.",
            p["id"], env_var, desde_entorno, sufijo)
        # unir_ruta parsea y reconstruye (urlsplit/urlunsplit), no concatena
        # texto crudo -- ver su docstring para el bug que evita.
        return unir_ruta(desde_entorno, sufijo)
    if ruta_entorno != sufijo:
        log.warning(
            "%s: %s='%s' trae una ruta ('%s') distinta del sufijo '%s' que "
            "proveedores.yaml declara por default para este proveedor; se usa "
            "TAL CUAL, sin modificar -- si fue sin querer, corregi la variable.",
            p["id"], env_var, desde_entorno, ruta_entorno, sufijo)
    return desde_entorno


def _capacidades_por_defecto(p: dict) -> Capacidades | None:
    datos = p.get("capacidades_por_defecto")
    if not datos:
        return None
    return Capacidades(tools=bool(datos["tools"]), vision=bool(datos["vision"]),
                       contexto=int(datos["contexto"]), max_salida=int(datos["max_salida"]))


def _excepciones(p: dict) -> dict:
    """Overrides por id de modelo sobre `capacidades_por_defecto`.

    `capacidades_por_defecto` declara UNA capacidad para todos los ids que se
    descubren de /models, y eso alcanza cuando el catalogo es homogeneo. El de
    grok no lo es: de sus 31 modelos, 25 devuelven un tool_call real y 6 no
    (medido el 2026-08-18 contra el proxy, no supuesto). Sin excepciones habria
    que elegir entre mentir sobre 6 rutas o renunciar al descubrimiento
    dinamico y declararlas todas a mano.

    Solo se declaran los campos que difieren; el resto se hereda del default.
    Un id que no aparece aca usa el default entero, que es el caso normal.
    """
    return {str(k): dict(v or {}) for k, v in (p.get("excepciones") or {}).items()}


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
        excepciones=_excepciones(p),
        desenvuelve_canvas=bool(p.get("desenvuelve_canvas", False)),
        timeout_s=(float(p["timeout_s"]) if p.get("timeout_s") is not None else None),
        emula_tools=bool(p.get("emula_tools", False)),
    ) for p in datos["proveedores"]]


def rutas_fijas(p: Proveedor) -> list[Ruta]:
    return [Ruta(p.id, m["id"], p.tier,
                 Capacidades(tools=True if p.emula_tools else bool(m["tools"]),
                             vision=bool(m["vision"]),
                             contexto=int(m["contexto"]), max_salida=int(m["max_salida"])),
                 prioridad=p.prioridad)
            for m in p.modelos_fijos]
