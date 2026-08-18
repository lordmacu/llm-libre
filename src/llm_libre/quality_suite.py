"""The quality battery: a fixed set of cases every free route is scored against.

The prompts below stay in Spanish ON PURPOSE. They are test DATA, not code: the
`spanish` case exists to measure whether a model can answer in Spanish at all,
and the others were calibrated against real providers with this exact wording.
Translating them would silently change what is being measured.
"""
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable

WEATHER_TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Clima de una ciudad",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}


@dataclass(frozen=True)
class Case:
    name: str
    body: dict
    check: Callable[[dict], bool]


def _text(r: dict) -> str:
    msg = (r.get("choices") or [{}])[0].get("message") or {}
    return (msg.get("content") or "").strip()


def _tool_calls(r: dict) -> list:
    msg = (r.get("choices") or [{}])[0].get("message") or {}
    return msg.get("tool_calls") or []


def _ok_arithmetic(r: dict) -> bool:
    return _text(r).rstrip(".") == "12"


def _ok_format(r: dict) -> bool:
    # A single word: penalises prose reasoning preambles.
    return len(_text(r).split()) == 1


def _ok_json(r: dict) -> bool:
    try:
        obj = json.loads(_text(r))
    except (json.JSONDecodeError, ValueError):
        return False
    return isinstance(obj, dict) and {"ciudad", "pais"} <= set(obj)


def _strip_accents(s: str) -> str:
    # NFKD splits each accented letter into (base letter, combining mark);
    # dropping the marks makes "Bogota" and "Bogotá" compare equal.
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _ok_tools(r: dict) -> bool:
    calls = _tool_calls(r)
    if not calls:
        return False
    f = calls[0].get("function") or {}
    if f.get("name") != "get_weather":
        return False
    try:
        city = json.loads(f.get("arguments") or "{}").get("city", "")
    except (json.JSONDecodeError, ValueError):
        return False
    return "bogota" in _strip_accents(city).lower()


# Spanish function words, required as a COMPLETE word (\b...\b): a substring
# check (which is what this used to be) lets English through, because
# "The buses arrive at noon." contains the substring "es ".
_SPANISH_WORDS = re.compile(r"\b(el|la|es|un|una)\b", re.IGNORECASE)


def _ok_spanish(r: dict) -> bool:
    return _SPANISH_WORDS.search(_text(r)) is not None


def _user(text: str) -> list:
    return [{"role": "user", "content": text}]


# Token budget per case.
#
# The previous values (32/32/128/256/128) measured stubbornness, not quality:
# almost all of these models are reasoning models, and thinking tokens come out
# of the SAME completion budget. Verified live against Kilo on 2026-08-16 with
# nvidia/nemotron-3.5-lightning:free and the arithmetic case:
#
#   max_tokens=32   -> finish_reason "length", content = "Here's a thinking
#                      process:\n\n1. **Analyze User Input:** ..."   -> FAILS
#   max_tokens=512  -> finish_reason "stop",   content = "12"        -> PASSES
#
# Same model, same question, same correct answer: the only thing being measured
# was whether its thinking fit in 32 tokens. Section 14 of the design counts the
# probe budget in REQUESTS, not tokens, so raising these caps costs nothing that
# was budgeted there.
#
# `reasoning: {"enabled": false}` is deliberately NOT sent (all 26 free models
# declare support for it), for two reasons. First: the battery has to measure
# what real traffic receives, and real traffic does not disable reasoning --
# measuring it switched off would score a mode no client will ever see. Second:
# it is exactly the risk that fix I6 just removed from the request path (sending
# providers fields they may not accept); a strict provider returning 400 would
# score a healthy route 0/5. The caps above already solve the problem on their
# own, as the live verification shows.
SHORT_TOKEN_BUDGET = 512   # a one or two word answer + its reasoning
LONG_TOKEN_BUDGET = 1024   # json, a tool call or a sentence + its reasoning

CASES: list[Case] = [
    Case("arithmetic",
         {"messages": _user("Cuanto es 7 mas 5? Responde solo el numero."),
          "max_tokens": SHORT_TOKEN_BUDGET, "temperature": 0}, _ok_arithmetic),
    Case("format",
         {"messages": _user("Saluda con UNA sola palabra, sin explicaciones."),
          "max_tokens": SHORT_TOKEN_BUDGET, "temperature": 0}, _ok_format),
    Case("json",
         {"messages": _user('Devuelve SOLO json con las claves "ciudad" y "pais" '
                            'para Bogota. Sin texto extra.'),
          "max_tokens": LONG_TOKEN_BUDGET, "temperature": 0}, _ok_json),
    Case("tools",
         {"messages": _user("Que clima hace en Bogota? Usa la herramienta."),
          "tools": [WEATHER_TOOL], "tool_choice": "auto",
          "max_tokens": LONG_TOKEN_BUDGET, "temperature": 0}, _ok_tools),
    Case("spanish",
         {"messages": _user("Describe el mar en una frase corta, en espanol."),
          "max_tokens": LONG_TOKEN_BUDGET, "temperature": 0}, _ok_spanish),
]


def evaluate(results: list[bool]) -> tuple[int, int]:
    return sum(1 for r in results if r), len(results)
