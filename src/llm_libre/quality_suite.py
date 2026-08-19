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
    # How many points this case is worth. 1 = a liveness check (can this route
    # add 7+5, emit one word, call one tool); >1 = a DISCRIMINATING case, one a
    # small model gets wrong and a real reasoner gets right.
    #
    # The weight exists because an unweighted pass-count saturated. Measured
    # 2026-08-18 against the live deployment: 14 of the 42 free routes carrying
    # tools scored exactly 1.0, among them a 2.6B model sitting above
    # deepseek-reasoner and grok-3. Every one of the five original cases is
    # something a 2.6B model does perfectly, so `quality` had no headroom left to
    # express "this one is actually better" -- and `auto:strong`, whose entire
    # job is to weigh quality (its exponent is 2.0), had nothing to weigh.
    weight: int = 1


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

# What a discriminating case would be worth against a liveness case worth 1.
#
# Nothing carries it yet, and that is a measured result, not an oversight. Three
# candidate case families were written and run against the live deployment on
# 2026-08-18 -- the bat-and-ball trap, a four-step arithmetic chain, and a
# two-constraint generation ("exactly five words, all starting with P") -- against
# liquid/lfm-2.5-2.6b (2.6B parameters) and against deepseek-reasoner,
# grok-3 and nvidia/nemotron-3-super-120b. EVERY route passed EVERY case. Two
# further families (grounded high-volume JSON over a supplied catalogue, and
# faithfulness to a context that omits the answer) also failed to separate them:
# at that output volume the reasoning models blow their token budget on thinking
# and get truncated, so the case penalises exactly the routes it is meant to
# reward.
#
# The conclusion is about the AXIS, not the difficulty: short, well-specified
# prompt-and-check tasks do not separate these routes, because a 2026-vintage 2.6B
# model genuinely is as good at them as a frontier reasoner. The tie the battery
# reports is real. Whatever `auto:strong` should be weighing, it is not this.
DISCRIMINATING_WEIGHT = 2

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


def evaluate(scored: list[tuple[Case, bool]]) -> tuple[int, int]:
    """(points earned, points possible) over the cases that actually ran.

    Takes (case, passed) pairs rather than bare booleans because the caller skips
    cases a route cannot be fairly asked (see probing.sample_quality and the
    `tools` case): a skipped case has to leave BOTH sides of the fraction alone,
    and with bare booleans there was no way to tell which weight to drop.
    """
    return (sum(c.weight for c, ok in scored if ok),
            sum(c.weight for c, _ in scored))
