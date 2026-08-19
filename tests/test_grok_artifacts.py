"""grok-proxy's namespaced cards, stripped at the gateway.

Same posture as `unwraps_canvas` for chatgpt-proxy: the front leaks its own UI
artifacts into `content`, and the gateway does not trust the proxy to be the only
thing standing between them and the client.

WHAT THIS DOES NOT COVER, on purpose: grok also interleaves plain-text status
labels ("Compilando las 20 recomendaciones") with the answer. Those carry NO
marker by the time they reach the gateway -- they are ordinary prose spliced at a
token boundary -- so nothing here can tell them from an answer, and a phrase list
would strip legitimate text. They are dropped at the source instead, by kind
(field 18 of the stream), in grok-proxy's `is_status_header`.
"""

from llm_libre.reasoning import (CompositeStreamTrimmer, XaiCardTrimmer,
                                 strip_xai_cards, trim)


def test_a_complete_card_is_removed():
    assert strip_xai_cards("before<xai:tool_usage_card>noise</xai:tool_usage_card>after") == "beforeafter"


def test_the_grok_namespace_too():
    assert strip_xai_cards("a<grok:render>x</grok:render>b") == "ab"


def test_text_without_cards_passes_through_untouched():
    text = "Los tags <b>en negrilla</b> y a < b siguen igual."
    assert strip_xai_cards(text) == text


def test_a_self_closing_card_is_removed():
    assert strip_xai_cards("a<xai:card id='1'/>b") == "ab"


def test_nested_cards_are_removed_whole():
    raw = "a<xai:outer>1<xai:inner>2</xai:inner>3</xai:outer>b"
    assert strip_xai_cards(raw) == "ab"


def test_a_card_that_never_closes_swallows_the_rest():
    # Deliberate, and the SAME choice as an unclosed <think>: what follows an
    # opened card is card innards, not answer. Emitting them would put raw XML
    # on screen, which is the failure being prevented.
    assert strip_xai_cards("visible<xai:card>innards and no close") == "visible"


def test_streaming_with_the_marker_split_at_every_possible_position():
    text = "hi<xai:card>gone</xai:card>bye"
    for cut in range(1, len(text)):
        trimmer = XaiCardTrimmer()
        got = trimmer.feed(text[:cut]) + trimmer.feed(text[cut:]) + trimmer.close()
        assert got == "hibye", f"split at {cut} gave {got!r}"


def test_streaming_character_by_character():
    text = "a<grok:render>x</grok:render>b"
    trimmer = XaiCardTrimmer()
    got = "".join(trimmer.feed(ch) for ch in text) + trimmer.close()
    assert got == "ab"


def test_a_lone_less_than_is_not_held_back_forever():
    # A '<' that cannot become a card must be released, or an answer ending in
    # "a < b" would hang on the last character.
    trimmer = XaiCardTrimmer()
    assert trimmer.feed("2 < 3") + trimmer.close() == "2 < 3"


def test_composite_trimmer_strips_cards_when_asked():
    trimmer = CompositeStreamTrimmer(strip_cards=True)
    got = trimmer.feed("<think>r</think>a<xai:card>x</xai:card>b") + trimmer.close()
    assert got == "ab"
    assert trimmer.reasoning == "r"


def test_composite_trimmer_leaves_cards_alone_by_default():
    trimmer = CompositeStreamTrimmer()
    raw = "a<xai:card>x</xai:card>b"
    assert trimmer.feed(raw) + trimmer.close() == raw


def test_trim_strips_cards_only_when_asked():
    raw = "a<xai:card>x</xai:card>b"
    assert trim(raw)[0] == raw
    assert trim(raw, strip_cards=True)[0] == "ab"
