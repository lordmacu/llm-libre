from llm_libre.quality_suite import CASES, Case, evaluate


def _response(text):
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def test_there_are_at_least_five_cases():
    assert len(CASES) >= 5


def test_the_budgets_leave_room_for_reasoning_tokens():
    # Fix round 3, I1. This test used to assert the opposite -- `max_tokens <= 256`,
    # "probing consumes quota" -- and so enshrined the bug: almost all of these
    # models are reasoning models and thinking tokens come out of the SAME
    # completion budget, so the battery was measuring whether the thinking fit in
    # 32 tokens, not the quality of the answer.
    #
    # Verified live (Kilo, 2026-08-16, nvidia/nemotron-3.5-lightning:free,
    # arithmetic case): with max_tokens=32 it returns finish_reason "length" and
    # the monologue ("Here's a thinking process: ..."), FAILS; with max_tokens=512
    # it returns finish_reason "stop" and "12", PASSES.
    for c in CASES:
        assert c.body["max_tokens"] >= 512, c.name
        assert c.body["max_tokens"] <= 2048, c.name   # still bounded


def test_the_battery_stays_small_in_number_of_requests():
    # Section 14 budgets probing in REQUESTS, not tokens: what has to be watched
    # is how many cases there are, not how much each one may write.
    #
    # Still 6: the three candidate discriminating cases were measured against the
    # live deployment and removed again, because they separated nothing (see
    # DISCRIMINATING_WEIGHT). Adding cases costs free quota per CYCLE across every
    # free route, so a case has to earn its place with evidence that it moves the
    # ranking, not with the plausibility of its prompt.
    assert len(CASES) <= 6


def test_the_arithmetic_case_accepts_the_correct_answer():
    case = next(c for c in CASES if c.name == "arithmetic")
    assert case.check(_response("12")) is True
    assert case.check(_response("13")) is False


def test_the_format_case_penalises_a_reasoning_preamble():
    # nvidia/nemotron answers "Here's a thinking process:..." to a one-word request.
    case = next(c for c in CASES if c.name == "format")
    assert case.check(_response("hi")) is True
    assert case.check(_response("Here's a thinking process: ... hi")) is False


def test_the_json_case_validates_against_the_schema():
    case = next(c for c in CASES if c.name == "json")
    assert case.check(_response('{"ciudad":"Bogota","pais":"Colombia"}')) is True
    assert case.check(_response('{"ciudad":"Bogota"}')) is False
    assert case.check(_response("no es json")) is False


def test_the_tools_case_requires_the_correct_call():
    case = next(c for c in CASES if c.name == "tools")
    good = {"choices": [{"message": {"tool_calls": [
        {"function": {"name": "get_weather", "arguments": '{"city":"Bogota"}'}}]}}]}
    assert case.check(good) is True
    assert case.check(_response("el clima esta lindo")) is False


def test_the_tools_case_accepts_bogota_with_an_accent():
    # "Bogotá" with the accent is the correct Spanish spelling; a model that uses
    # it must not be penalised against one that answers "Bogota" without it.
    case = next(c for c in CASES if c.name == "tools")
    accented = {"choices": [{"message": {"tool_calls": [
        {"function": {"name": "get_weather", "arguments": '{"city":"Bogotá"}'}}]}}]}
    assert case.check(accented) is True


def test_the_spanish_case_requires_whole_words_not_substrings():
    case = next(c for c in CASES if c.name == "spanish")
    # Real Spanish: must pass.
    assert case.check(_response("El mar es azul y tranquilo.")) is True
    assert case.check(_response("Una casa grande junto a la playa.")) is True
    # Real English: must fail. These two sentences contain, as a SUBSTRING (not
    # as a whole word), one of the Spanish function words being searched for --
    # "buses " contains "es ", "fun today" contains "un " -- which is exactly the
    # naive check this case used to have. If the check ever goes back to
    # substring matching, these two assertions fail.
    assert case.check(_response("The buses arrive at noon.")) is False
    assert case.check(_response("We had so much fun today.")) is False
    assert case.check(_response("The weather is nice today.")) is False


def test_evaluate_adds_up_the_weights_of_the_passing_cases():
    a, b = CASES[0], CASES[1]
    assert evaluate([(a, True), (b, False)]) == (a.weight, a.weight + b.weight)


# --- Weighting, added 2026-08-18 ---------------------------------------------
#
# `evaluate` returns POINTS rather than a count of passing cases, so that a case
# can be worth more than another. Nothing carries a weight above 1 yet -- see
# DISCRIMINATING_WEIGHT in quality_suite.py for the live measurement that says why
# -- so these tests pin the mechanism, not a particular battery.


def test_every_case_declares_a_weight():
    for c in CASES:
        assert isinstance(c.weight, int) and c.weight >= 1, c.name


def test_passing_everything_scores_exactly_one():
    earned, possible = evaluate([(c, True) for c in CASES])
    assert earned == possible


def test_failing_everything_scores_zero():
    earned, possible = evaluate([(c, False) for c in CASES])
    assert earned == 0 and possible > 0


def test_a_skipped_case_leaves_both_sides_of_the_fraction_alone():
    """The `tools` case is skipped for a route that does not declare tools. It
    must not count as failed -- that would conflate "does not promise this" with
    "promised it and got it wrong"."""
    tools = next(c for c in CASES if c.name == "tools")
    ran = [(c, True) for c in CASES if c is not tools]
    assert evaluate(ran) == (sum(c.weight for c in CASES) - tools.weight,) * 2


def test_a_heavier_case_moves_the_score_more_than_a_light_one():
    """The mechanism the battery will need the day a case earns weight > 1."""
    light = Case("light", {}, lambda r: True, weight=1)
    heavy = Case("heavy", {}, lambda r: True, weight=3)
    assert evaluate([(light, True), (heavy, False)])[0] < evaluate(
        [(light, False), (heavy, True)])[0]
