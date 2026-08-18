from llm_libre.reasoning import (CanvasFenceTrimmer, CompositeStreamTrimmer,
                                 ReasoningTrimmer, strip_canvas_fences, trim)


def test_trims_a_complete_block():
    clean, reasoning = trim("<think>pienso mucho</think>hola")
    assert clean == "hola"
    assert reasoning == "pienso mucho"


def test_text_without_tags_passes_through_untouched():
    assert trim("hola mundo") == ("hola mundo", "")


def test_accepts_all_three_tag_spellings():
    for open_tag, close_tag in (("<think>", "</think>"),
                                ("<thinking>", "</thinking>"),
                                ("<reasoning>", "</reasoning>")):
        assert trim(f"{open_tag}x{close_tag}listo")[0] == "listo"


def test_a_tag_that_never_closes_swallows_no_useful_text():
    # Everything after it is reasoning; the client gets what came before,
    # neither empty nor hanging.
    clean, reasoning = trim("antes<think>y nunca cierro")
    assert clean == "antes"
    assert reasoning == "y nunca cierro"


def test_several_blocks_in_one_response():
    clean, reasoning = trim("a<think>1</think>b<think>2</think>c")
    assert clean == "abc"
    assert reasoning == "12"


def test_streaming_with_the_tag_split_at_every_possible_position():
    text = "hola<think>secreto</think>chau"
    expected = "holachau"
    for cut in range(1, len(text)):
        trimmer = ReasoningTrimmer()
        out = trimmer.feed(text[:cut]) + trimmer.feed(text[cut:]) + trimmer.close()
        assert out == expected, f"failed cutting at {cut}"
        assert trimmer.reasoning == "secreto", f"failed cutting at {cut}"


def test_streaming_character_by_character():
    text = "ab<thinking>zzz</thinking>cd"
    trimmer = ReasoningTrimmer()
    out = "".join(trimmer.feed(c) for c in text) + trimmer.close()
    assert out == "abcd"
    assert trimmer.reasoning == "zzz"


def test_streaming_holds_back_nothing_that_cannot_be_a_tag():
    # Emit as early as possible: only a possible tag prefix is held back.
    trimmer = ReasoningTrimmer()
    assert trimmer.feed("hola mundo") == "hola mundo"


def test_streaming_holds_an_ambiguous_prefix_until_resolved():
    trimmer = ReasoningTrimmer()
    assert trimmer.feed("hola<thi") == "hola"
    assert trimmer.feed("nk>oculto</think>fin") == "fin"
    assert trimmer.reasoning == "oculto"


def test_a_lone_less_than_sign_is_not_lost():
    trimmer = ReasoningTrimmer()
    out = trimmer.feed("2 < 3") + trimmer.close()
    assert out == "2 < 3"


# --- Task 13: canvas fences (':::word{...}' ... ':::'). UNLIKE <think>, the
#     content INSIDE is the answer: it is not discarded, only the two marker
#     lines are removed. ---

_FENCE = (':::writing{variant="document" id="58321" title="Bogota"}\n'
          'Bogota, ciudad de niebla y montana,\n'
          'jamas se cansa de sonar.\n'
          ':::')
_CONTENT = 'Bogota, ciudad de niebla y montana,\njamas se cansa de sonar.\n'


def test_strips_the_canvas_fence_keeping_the_content():
    assert strip_canvas_fences(_FENCE) == _CONTENT


def test_trim_unwraps_the_fence_only_when_the_caller_asks():
    # trim() (the entry point of the non-streaming path) must do both things
    # when unwrap_canvas=True: keep trimming <think> AND unwrap the canvas fence.
    clean, reasoning = trim("<think>pienso</think>" + _FENCE, unwrap_canvas=True)
    assert clean == _CONTENT
    assert reasoning == "pienso"


def test_a_bare_canvas_fence_ends_up_free_of_markers():
    clean, reasoning = trim(_FENCE, unwrap_canvas=True)
    assert clean == _CONTENT
    assert reasoning == ""


def test_trim_leaves_fences_alone_by_default():
    # The default (unwrap_canvas=False) is deliberate: ':::nota{...}' is also
    # standard Docusaurus/MDX syntax. A provider that does not declare
    # desenvuelve_canvas (Kilo, OpenRouter) must not lose those markers --
    # <think> IS still always trimmed, which is orthogonal.
    clean, reasoning = trim("<think>pienso</think>" + _FENCE)
    assert clean == _FENCE
    assert reasoning == "pienso"


def test_ordinary_content_without_a_fence_is_untouched():
    assert strip_canvas_fences("hola mundo, todo normal.") == "hola mundo, todo normal."


def test_triple_colon_that_is_not_a_fence_is_untouched():
    # ':::' appears but NOT at the start of a line as a real marker: not a
    # single character may be touched.
    text = "el separador de esta plantilla es ::: en este formato, nada mas."
    assert strip_canvas_fences(text) == text


def test_triple_colon_at_line_start_without_a_word_is_not_an_opening():
    # ':::' followed by a space (not by a word) does not satisfy the opening
    # syntax: it is preserved verbatim.
    text = "::: esto no es una cerca\nsegunda linea"
    assert strip_canvas_fences(text) == text


def test_a_fence_that_never_closes_loses_no_content():
    # Unlike an unclosed <think> (which swallows everything as reasoning), an
    # unclosed canvas fence discards nothing: only the opening marker, already
    # identified with certainty, is removed.
    text = ':::writing{title="x"}\nhola\nmundo sin cerrar'
    assert strip_canvas_fences(text) == "hola\nmundo sin cerrar"


def test_streaming_with_the_fence_split_at_every_possible_position():
    for cut in range(1, len(_FENCE)):
        trimmer = CanvasFenceTrimmer()
        out = trimmer.feed(_FENCE[:cut]) + trimmer.feed(_FENCE[cut:]) + trimmer.close()
        assert out == _CONTENT, f"failed cutting at {cut}"


def test_streaming_the_fence_character_by_character():
    trimmer = CanvasFenceTrimmer()
    out = "".join(trimmer.feed(c) for c in _FENCE) + trimmer.close()
    assert out == _CONTENT


def test_streaming_embedded_triple_colon_is_untouched():
    text = "el separador es ::: y sigue el texto normal despues de eso."
    for cut in range(1, len(text)):
        trimmer = CanvasFenceTrimmer()
        out = trimmer.feed(text[:cut]) + trimmer.feed(text[cut:]) + trimmer.close()
        assert out == text, f"failed cutting at {cut}"


def test_streaming_a_fence_that_never_closes_loses_no_content():
    text = ':::writing{title="x"}\nhola\nmundo sin cerrar'
    expected = "hola\nmundo sin cerrar"
    for cut in range(1, len(text)):
        trimmer = CanvasFenceTrimmer()
        out = trimmer.feed(text[:cut]) + trimmer.feed(text[cut:]) + trimmer.close()
        assert out == expected, f"failed cutting at {cut}"


# --- Finding 4 of the Task 13 review: four mutations of the canvas automaton
#     survived the entire suite. Each is a real behaviour change, verified by
#     running the mutation by hand against these exact tests (not assumed). ---

def test_close_discards_the_closing_marker_it_does_not_emit_it():
    # Every fence in this file ends in ':::' WITHOUT a trailing newline, so the
    # real closing marker is only ever consumed by close() (never by the main
    # loop of feed(), which only acts on complete lines ending in '\n'). If
    # close() emitted `rest` instead of "" on confirming the close, the fence
    # would be left with a ':::' dangling at the end.
    out = strip_canvas_fences(_FENCE)
    assert out == _CONTENT
    assert not out.endswith(":::")


def test_without_line_start_tracking_answer_text_would_be_lost():
    # ':::' embedded mid-line (not at the start) is NEVER a marker. Without the
    # "this line was already ruled out" tracking (_at_line_start), an isolated
    # ':' arriving right after a flush is re-evaluated as if it began a new
    # line, and "a:::b" would swallow its own ending as if it were an opening --
    # the silent text loss this area exists to prevent.
    text = "el rango a:::b\nsigue\n"
    trimmer = CanvasFenceTrimmer()
    out = "".join(trimmer.feed(c) for c in text) + trimmer.close()
    assert out == text


def test_the_opening_pattern_requires_the_whole_line_to_be_the_marker():
    # ":::nota" is a valid prefix, but "esto no es cerca" ruins the rest of the
    # line: the opening marker requires the ENTIRE LINE, not a prefix. If the
    # pattern were loosened (e.g. stopped anchoring the end), this line would be
    # swallowed whole instead of preserved.
    text = ":::nota esto no es cerca\nsegunda linea\n"
    assert strip_canvas_fences(text) == text


def test_the_closing_pattern_requires_the_exact_line_not_any_triple_colon():
    # A ':::otro' line INSIDE an open fence is not the close -- the close is
    # EXACTLY ':::' alone. If the closing pattern were loosened (e.g. accepted
    # any line starting with ':::'), this fence would be cut short and "resto
    # real" would end up OUTSIDE, treated as if it had never been inside (and
    # here it would also be lost, left as an invalid opening line).
    text = (':::writing{title="x"}\n'
            'primera linea\n'
            ':::otro\n'
            'resto real\n'
            ':::')
    expected = 'primera linea\n:::otro\nresto real\n'
    assert strip_canvas_fences(text) == expected


def test_composite_trimmer_chains_think_and_canvas_when_asked():
    # The real streaming path (proxy.py) uses a single object: reasoning is
    # discarded first, then the fence is unwrapped over what is ALREADY visible
    # content -- but only if the route's provider declares it (unwrap_canvas=True).
    text = "<think>mmm</think>" + _FENCE
    trimmer = CompositeStreamTrimmer(unwrap_canvas=True)
    out = ""
    for cut in range(0, len(text), 3):
        out += trimmer.feed(text[cut:cut + 3])
    out += trimmer.close()
    assert out == _CONTENT
    assert trimmer.reasoning == "mmm"


def test_composite_trimmer_leaves_fences_alone_by_default():
    text = "<think>mmm</think>" + _FENCE
    trimmer = CompositeStreamTrimmer()
    out = ""
    for cut in range(0, len(text), 3):
        out += trimmer.feed(text[cut:cut + 3])
    out += trimmer.close()
    assert out == _FENCE
    assert trimmer.reasoning == "mmm"
