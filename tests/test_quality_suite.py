from llm_libre.quality_suite import CASES, evaluate


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
    assert len(CASES) <= 6


def test_the_arithmetic_case_accepts_the_correct_answer():
    case = next(c for c in CASES if c.name == "arithmetic")
    assert case.check(_response("12")) is True
    assert case.check(_response("13")) is False


def test_the_format_case_penalises_a_reasoning_preamble():
    # nvidia/nemotron answers "Here's a thinking process:..." to a one-word request.
    case = next(c for c in CASES if c.name == "format")
    assert case.check(_response("hola")) is True
    assert case.check(_response("Here's a thinking process: ... hola")) is False


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


def test_evaluate_counts_the_passing_cases():
    assert evaluate([True, False, True]) == (2, 3)
