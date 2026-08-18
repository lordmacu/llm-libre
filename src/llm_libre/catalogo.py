import logging
import re

from dataclasses import replace

from llm_libre.modelos import Capacidades, Ruta

log = logging.getLogger(__name__)

# Senales de que un modelo, aunque sea gratis y devuelva texto, NO es un modelo
# de chat de proposito general y no tiene nada que hacer en la rotacion de
# `auto`. Se leen de los campos `name` y `description` que el PROPIO proveedor
# publica en /models -- campos que este normalizador venia descartando.
#
# Esto sigue siendo DESCUBRIMIENTO, no ids cableados, y la distincion es la
# premisa entera del proyecto (§1 del diseno: "el catalogo se descubre desde
# /models, siempre"): una lista negra de ids se pudriria exactamente igual que
# la lista de ids cableados que este gateway existe para reemplazar, mientras
# que un guardrail nuevo que aparezca manana con otro nombre se va a seguir
# describiendo a si mismo como guardrail.
#
# Por que importa: en una instalacion nueva TODAS las rutas arrancan con la
# calidad neutra, asi que hasta la primera bateria el unico discriminador es la
# latencia -- y lo mas rapido del pozo gratis es
# `nvidia/nemotron-3.5-content-safety:free`, un clasificador que contesta
# "User Safety: safe" a cualquier cosa. Quedaba #1 en `auto`.
_ESPECIALIDAD = (
    r"guardrail",
    r"content safety",
    r"\bmoderation\b",
    r"\bmoderates\b",
    r"\bclassifier\b",
    r"\breranker\b", r"\bre-ranker\b", r"\breranking\b",
    r"embeddings? model", r"text embeddings?\b",
    r"speech[- ]to[- ]text", r"text[- ]to[- ]speech",
)

# Meta-routers (`kilo-auto/free`, `openrouter/free`): no son un modelo, son un
# sorteo entre otros modelos. Puntuarlos mide una ruleta -- su calidad y su
# latencia son las de quien haya salido esa vez -- y ademas tapan que ruta
# sirvio de verdad, que es justo lo que `X-Ruta-Usada` promete. Se descartan
# por el mismo medio que todo lo demas: lo que ELLOS dicen de si mismos
# ("rotates through available free models", "a router that selects free models
# at random"), no por su id.
_META_ROUTER = (
    r"\bmodels? router\b",
    r"\bis a router\b",
    r"rotates through",
    r"\brouter\b.{0,60}\bselects\b",
    r"selects .{0,40}\bmodels\b.{0,20}at random",
)

_DESCARTAR = re.compile("|".join(_ESPECIALIDAD + _META_ROUTER), re.IGNORECASE)

# Ids reservados por llm-libre MISMO, no por ningun proveedor: "auto" colisiona
# con el alias propio de interpretar_pedido (api.py). Pedir el modelo "auto"
# siempre resuelve al alias, nunca a una ruta real -- asi que una ruta con ese
# modelo_id literal quedaria inalcanzable para siempre. Se filtra aca, en el
# descubrimiento, para que NINGUN proveedor (presente o futuro) pueda colar
# sin querer una ruta invalida por coincidencia de nombres.
#
# INFO de la revision round 6: el literal "auto" no alcanza. `ALIAS` en
# api.py (y `interpretar_pedido`) tambien trata como alias reservado
# CUALQUIER id de la forma "auto:<sufijo>" -- "auto:rapido", "auto:potente",
# "auto:tools", "auto:vision" -- resolviendolo SIEMPRE antes de comparar
# contra `pedido.modelo`. Un proveedor que publicara un modelo real con uno
# de esos ids (o cualquier otro "auto:*" que api.py agregue manana) creaba
# una ruta permanentemente inalcanzable, exactamente por el mismo mecanismo
# que ya justificaba reservar "auto". Por eso `es_id_reservado` cubre el
# PATRON completo, no una lista de ids conocidos hoy.
IDS_RESERVADOS = frozenset({"auto"})


def es_id_reservado(id_: str) -> bool:
    """True si `id_` colisiona con "auto" o con cualquier alias compuesto
    "auto:<sufijo>" que interpretar_pedido (api.py) resuelve antes de
    comparar contra un id literal -- ver el comentario de arriba de
    IDS_RESERVADOS."""
    return id_ in IDS_RESERVADOS or id_.startswith("auto:")

# Los proveedores que agregan alias legacy a su propio catalogo (chatgpt-proxy
# expone "gpt-4o" como alias de "auto", p.ej.) se autoidentifican con este
# prefijo en su `description`. Quedarselos crearia una ruta DUPLICADA
# apuntando al mismo modelo bajo otro nombre -- el ranking terminaria
# midiendo, y compitiendo, el mismo modelo consigo mismo.
_PREFIJO_ALIAS = "Alias →"


def es_de_especialidad(m: dict) -> bool:
    """True si el proveedor describe este modelo como algo que no es chat general."""
    return _DESCARTAR.search(f"{m.get('name') or ''} {m.get('description') or ''}") is not None


def _es_alias(m: dict) -> bool:
    return str(m.get("description") or "").startswith(_PREFIJO_ALIAS)


def normalizar(proveedor: str, datos: dict | list, prioridad: int = 100,
              capacidades_por_defecto: Capacidades | None = None,
              excepciones: dict | None = None,
              emula_tools: bool = False) -> list[Ruta]:
    """Convierte la respuesta de /models en rutas gratis utilizables para chat.

    `prioridad` es la del PROVEEDOR (ver Proveedor.prioridad), no algo que
    /models pueda traer: se estampa igual en cada ruta descubierta para que
    el router las ordene sin tener que volver a consultar el registro.

    `capacidades_por_defecto` (Proveedor.capacidades_por_defecto) marca un
    proveedor cuyo /models trae IDS pero ningun metadato de capacidad --
    chatgpt-proxy es el caso que motivo esto: su catalogo es dinamico (se
    descubre, no se cablea) pero solo trae
    id/object/created/owned_by/description, nunca pricing ni modalidades.
    Cuando esta declarado, esas capacidades se aplican IGUAL a cada id
    descubierto y se SALTEAN los chequeos de precio y de modalidad de salida
    -- un proveedor que declara defaults esta afirmando lo que su catalogo no
    puede decir. El filtro de especialidad (guardrails, meta-routers) sigue
    activo en los dos modos: no tiene nada que ver con lo que el catalogo
    puede o no reportar. Sigue siendo DESCUBRIMIENTO en el sentido que
    importa (§1 del diseno): los IDS se leen de /models, nunca se cablean;
    solo las capacidades vienen del registro.
    """
    items = datos.get("data", datos) if isinstance(datos, dict) else datos
    rutas: list[Ruta] = []
    for m in items:
        # Una entrada sin `id` (o que ni siquiera es un dict) reventaba con
        # KeyError/AttributeError; sondeo.py se lo tragaba y el catalogo ENTERO
        # de ese proveedor quedaba congelado para siempre, sin una linea de log
        # que lo dijera. Se omite la entrada rota y se deja constancia.
        if not isinstance(m, dict):
            log.warning("catalogo %s: entrada que no es un objeto, se omite: %.120r",
                        proveedor, m)
            continue
        if not m.get("id"):
            log.warning("catalogo %s: entrada sin 'id', se omite: %.120r", proveedor, m)
            continue
        if es_id_reservado(m["id"]):
            continue
        if _es_alias(m):
            continue
        if es_de_especialidad(m):
            log.info("catalogo %s: %s se descarta, el proveedor no lo describe como "
                     "un modelo de chat general", proveedor, m["id"])
            continue
        if capacidades_por_defecto is not None:
            capacidades = capacidades_por_defecto
            # Un catalogo descubierto NO tiene por que ser homogeneo: grok
            # publica 31 ids de los que 25 hacen tool calls y 6 no. La
            # excepcion pisa SOLO los campos declarados; lo que no se nombra
            # se hereda, asi que corregir una capacidad de un modelo no
            # obliga a repetir las otras tres.
            faltante = (excepciones or {}).get(m["id"])
            if faltante:
                capacidades = replace(capacidades, **faltante)
        else:
            if not _es_gratis(m):
                continue
            arch = m.get("architecture") or {}
            # Exigir salida SOLO texto: los modelos de musica (lyria) tambien
            # traen "text" entre sus salidas, asi que "contiene texto" los
            # dejaria pasar.
            salidas = set(arch.get("output_modalities") or ["text"])
            if salidas != {"text"}:
                continue
            soportados = m.get("supported_parameters") or []
            top = m.get("top_provider") or {}
            capacidades = Capacidades(
                tools="tools" in soportados,
                vision="image" in (arch.get("input_modalities") or []),
                contexto=int(m.get("context_length") or top.get("context_length") or 0),
                max_salida=int(top.get("max_completion_tokens") or 0),
            )
        if emula_tools:
            capacidades = replace(capacidades, tools=True)
        rutas.append(Ruta(
            proveedor=proveedor,
            modelo_id=m["id"],
            tier="gratis",
            capacidades=capacidades,
            prioridad=prioridad,
        ))
    return rutas


def _es_gratis(m: dict) -> bool:
    precio = (m.get("pricing") or {}).get("prompt")
    try:
        return float(precio) == 0.0
    except (TypeError, ValueError):
        return False
