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


CASOS: list[Caso] = [
    Caso("aritmetica",
         {"messages": _usuario("Cuanto es 7 mas 5? Responde solo el numero."),
          "max_tokens": 32, "temperature": 0}, _ok_aritmetica),
    Caso("formato",
         {"messages": _usuario("Saluda con UNA sola palabra, sin explicaciones."),
          "max_tokens": 32, "temperature": 0}, _ok_formato),
    Caso("json",
         {"messages": _usuario('Devuelve SOLO json con las claves "ciudad" y "pais" '
                               'para Bogota. Sin texto extra.'),
          "max_tokens": 128, "temperature": 0}, _ok_json),
    Caso("tools",
         {"messages": _usuario("Que clima hace en Bogota? Usa la herramienta."),
          "tools": [HERRAMIENTA], "tool_choice": "auto",
          "max_tokens": 256, "temperature": 0}, _ok_tools),
    Caso("espanol",
         {"messages": _usuario("Describe el mar en una frase corta, en espanol."),
          "max_tokens": 128, "temperature": 0}, _ok_espanol),
]


def evaluar(resultados: list[bool]) -> tuple[int, int]:
    return sum(1 for r in resultados if r), len(resultados)
