import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable

HERRAMIENTA = {"type": "function", "function": {
    "name": "get_weather", "description": "Clima de una ciudad",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}


@dataclass(frozen=True)
class Caso:
    nombre: str
    cuerpo: dict
    verificar: Callable[[dict], bool]


def _texto(r: dict) -> str:
    msg = (r.get("choices") or [{}])[0].get("message") or {}
    return (msg.get("content") or "").strip()


def _tool_calls(r: dict) -> list:
    msg = (r.get("choices") or [{}])[0].get("message") or {}
    return msg.get("tool_calls") or []


def _ok_aritmetica(r: dict) -> bool:
    return _texto(r).rstrip(".") == "12"


def _ok_formato(r: dict) -> bool:
    # Una sola palabra: castiga los preambulos de razonamiento en prosa.
    return len(_texto(r).split()) == 1


def _ok_json(r: dict) -> bool:
    try:
        obj = json.loads(_texto(r))
    except (json.JSONDecodeError, ValueError):
        return False
    return isinstance(obj, dict) and {"ciudad", "pais"} <= set(obj)


def _sin_acentos(s: str) -> str:
    # NFKD separa cada letra acentuada en (letra base, marca combinante);
    # descartar las marcas deja "Bogota" y "Bogotá" iguales para comparar.
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _ok_tools(r: dict) -> bool:
    llamadas = _tool_calls(r)
    if not llamadas:
        return False
    f = llamadas[0].get("function") or {}
    if f.get("name") != "get_weather":
        return False
    try:
        ciudad = json.loads(f.get("arguments") or "{}").get("city", "")
    except (json.JSONDecodeError, ValueError):
        return False
    return "bogota" in _sin_acentos(ciudad).lower()


# Palabras funcionales del espanol, exigidas como palabra COMPLETA (\b...\b):
# un chequeo por subcadena (el que habia antes) deja pasar ingles como
# "The buses arrive at noon." porque "buses " contiene la subcadena "es ".
_PALABRAS_ESPANOL = re.compile(r"\b(el|la|es|un|una)\b", re.IGNORECASE)


def _ok_espanol(r: dict) -> bool:
    return _PALABRAS_ESPANOL.search(_texto(r)) is not None


def _usuario(texto: str) -> list:
    return [{"role": "user", "content": texto}]


# Presupuesto de tokens por caso.
#
# Los valores anteriores (32/32/128/256/128) median terquedad, no calidad: casi
# todos estos modelos son de razonamiento y los tokens de pensamiento salen del
# MISMO presupuesto de la completion. Verificado en vivo contra Kilo el
# 2026-08-16 con nvidia/nemotron-3.5-lightning:free y el caso de aritmetica:
#
#   max_tokens=32   -> finish_reason "length", content = "Here's a thinking
#                      process:\n\n1. **Analyze User Input:** ..."   -> FALLA
#   max_tokens=512  -> finish_reason "stop",   content = "12"        -> PASA
#
# El mismo modelo, la misma pregunta, la misma respuesta correcta: lo unico que
# se estaba midiendo era si le entraba el pensamiento en 32 tokens. El §14 del
# diseno cuenta el presupuesto de sondeo en PETICIONES, no en tokens, asi que
# subir estos topes no cuesta nada de lo que ahi se presupuesto.
#
# NO se manda `reasoning: {"enabled": false}` (que los 26 modelos gratis
# declaran soportar), por dos razones. Primera: la bateria tiene que medir lo
# que el trafico real recibe, y el trafico real no desactiva el razonamiento --
# medirlo apagado puntuaria un modo que ningun cliente va a ver. Segunda: es
# exactamente el riesgo que el fix I6 acaba de sacar del camino de peticion
# (mandarle al proveedor campos que quizas no acepte); un proveedor estricto
# que devuelva 400 haria puntuar 0/5 a una ruta sana. Los topes de arriba ya
# resuelven el problema por si solos, como muestra la verificacion en vivo.
TOPE_CORTO = 512      # respuesta de una o dos palabras + su razonamiento
TOPE_LARGO = 1024     # json, tool call o una frase + su razonamiento

CASOS: list[Caso] = [
    Caso("aritmetica",
         {"messages": _usuario("Cuanto es 7 mas 5? Responde solo el numero."),
          "max_tokens": TOPE_CORTO, "temperature": 0}, _ok_aritmetica),
    Caso("formato",
         {"messages": _usuario("Saluda con UNA sola palabra, sin explicaciones."),
          "max_tokens": TOPE_CORTO, "temperature": 0}, _ok_formato),
    Caso("json",
         {"messages": _usuario('Devuelve SOLO json con las claves "ciudad" y "pais" '
                               'para Bogota. Sin texto extra.'),
          "max_tokens": TOPE_LARGO, "temperature": 0}, _ok_json),
    Caso("tools",
         {"messages": _usuario("Que clima hace en Bogota? Usa la herramienta."),
          "tools": [HERRAMIENTA], "tool_choice": "auto",
          "max_tokens": TOPE_LARGO, "temperature": 0}, _ok_tools),
    Caso("espanol",
         {"messages": _usuario("Describe el mar en una frase corta, en espanol."),
          "max_tokens": TOPE_LARGO, "temperature": 0}, _ok_espanol),
]


def evaluar(resultados: list[bool]) -> tuple[int, int]:
    return sum(1 for r in resultados if r), len(resultados)
