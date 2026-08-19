import re

OPEN_TAGS = ("<think>", "<thinking>", "<reasoning>")
CLOSE_TAGS = ("</think>", "</thinking>", "</reasoning>")


class ReasoningTrimmer:
    """Splits reasoning from useful text across a stream of deltas.

    The hard part is that tags arrive split across chunks: the tail of the buffer
    that COULD still be the prefix of a tag has to be held back, while everything
    else is emitted immediately.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._inside = False
        self.reasoning = ""

    def feed(self, delta: str) -> str:
        self._buf += delta
        out = ""
        while True:
            tags = CLOSE_TAGS if self._inside else OPEN_TAGS
            i, tag = _first_of(self._buf, tags)
            if i == -1:
                cut = len(self._buf) - _ambiguous_tail(self._buf, tags)
                chunk, self._buf = self._buf[:cut], self._buf[cut:]
                if self._inside:
                    self.reasoning += chunk
                else:
                    out += chunk
                return out
            chunk = self._buf[:i]
            if self._inside:
                self.reasoning += chunk
            else:
                out += chunk
            self._buf = self._buf[i + len(tag):]
            self._inside = not self._inside

    def close(self) -> str:
        """Close the stream. An unclosed block counts entirely as reasoning."""
        rest, self._buf = self._buf, ""
        if self._inside:
            self.reasoning += rest
            return ""
        return rest


def trim(text: str, unwrap_canvas: bool = False,
         strip_cards: bool = False) -> tuple[str, str]:
    """`unwrap_canvas` (default False) is a PROVIDER DECISION
    (Provider.unwraps_canvas), not a universal one: ':::note{...}' is also
    standard Docusaurus/MDX syntax (admonitions), and applying it blindly strips
    the markers from a Kilo/OpenRouter response that is quoting or generating
    that syntax on purpose -- which is what happened before this fix. Only a
    provider that genuinely leaks canvas mode (chatgpt-proxy) should pass True.

    `strip_cards` (default False) is the same kind of decision for grok-proxy
    (Provider.strips_xai_cards). See XaiCardTrimmer.
    """
    trimmer = ReasoningTrimmer()
    clean = trimmer.feed(text) + trimmer.close()
    if unwrap_canvas:
        clean = strip_canvas_fences(clean)
    if strip_cards:
        clean = strip_xai_cards(clean)
    return clean, trimmer.reasoning


def _first_of(s: str, tags: tuple[str, ...]) -> tuple[int, str]:
    best, which = -1, ""
    for t in tags:
        i = s.find(t)
        if i != -1 and (best == -1 or i < best):
            best, which = i, t
    return best, which


def _ambiguous_tail(s: str, tags: tuple[str, ...]) -> int:
    """Length of the suffix of s that could still complete one of the tags."""
    longest = max(len(t) for t in tags) - 1
    for length in range(min(longest, len(s)), 0, -1):
        tail = s[-length:]
        if any(t.startswith(tail) for t in tags):
            return length
    return 0


# --- Canvas fences: ':::word{...attributes...}' ... ':::' -----------------
#
# chatgpt-proxy (Task 13) leaks ChatGPT's "canvas" mode into `content`, markers
# and all (verified against the real proxy, 2026-08-16):
#
#   :::writing{variant="document" id="58321" title="Bogota"}
#   Bogota, ciudad de niebla y montana,
#   ...
#   :::
#
# KEY DIFFERENCE from <think>: what sits inside the fence is the ANSWER, not
# something to discard. Only the two marker lines are removed (the opening one
# with its attributes, and the closing ':::'); everything else -- including the
# content "inside" -- is kept verbatim, character for character, exactly like
# text that was never inside a fence at all.
#
# That is why this is a simpler automaton than ReasoningTrimmer: there is nothing
# to accumulate on a second channel (no ".reasoning" here), only a decision that
# TWO complete lines are not emitted.
_RE_CANVAS_OPEN = re.compile(r"^:::[A-Za-z][\w-]*(\{[^\n]*\})?$")
_RE_CANVAS_CLOSE = re.compile(r"^:::$")


def _could_be_canvas_open(s: str) -> bool:
    """True if `s` (a still-incomplete line, with no '\\n') is a valid prefix of
    an opening ':::word{...}' line. Used to decide, BEFORE the rest of the line
    arrives, whether it must keep being held back or can already be released as
    ordinary text."""
    marker = ":::"
    n = min(len(s), len(marker))
    if s[:n] != marker[:n]:
        return False
    rest = s[len(marker):]
    if rest == "":
        return True
    if not rest[0].isalpha():
        return False
    i = 1
    while i < len(rest) and (rest[i].isalnum() or rest[i] in "_-"):
        i += 1
    if i == len(rest):
        return True   # everything seen so far is the "word"; what follows is unknown
    return rest[i] == "{"   # attributes have started: everything after is ambiguous


def _could_be_canvas_close(s: str) -> bool:
    """True if `s` could still complete into the closing line ':::'."""
    return ":::".startswith(s)


class CanvasFenceTrimmer:
    """Unwraps canvas fences over a stream of deltas, KEEPING the inner content
    -- the opposite of ReasoningTrimmer, which discards it.

    As in ReasoningTrimmer the marker may arrive split across chunks; unlike
    ReasoningTrimmer, the unit of detection here is the complete LINE (up to
    '\\n'), because the opening marker has a variable shape (its attributes). So
    as not to delay the streaming of ordinary text, the moment a line stops being
    a possible marker it is released immediately (`_at_line_start` flips to
    False) and the rest of THAT line is never re-examined -- which avoids the
    false positive of a ':::' embedded mid-sentence, never at the start of a real
    line.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._inside = False
        self._at_line_start = True

    def feed(self, delta: str) -> str:
        self._buf += delta
        out = ""
        while True:
            if not self._at_line_start:
                # This line was already ruled out as a candidate: it passes
                # through verbatim up to the next '\n' (inclusive), unexamined.
                i = self._buf.find("\n")
                if i == -1:
                    out += self._buf
                    self._buf = ""
                    return out
                out += self._buf[:i + 1]
                self._buf = self._buf[i + 1:]
                self._at_line_start = True
                continue

            i = self._buf.find("\n")
            if i == -1:
                candidate = self._buf
                possible = (_could_be_canvas_close(candidate) if self._inside
                            else _could_be_canvas_open(candidate))
                if possible:
                    return out   # held back, waiting for the rest of the line
                out += self._buf
                self._buf = ""
                self._at_line_start = False
                return out

            line, self._buf = self._buf[:i], self._buf[i + 1:]
            if self._inside:
                if _RE_CANVAS_CLOSE.fullmatch(line):
                    self._inside = False
                else:
                    out += line + "\n"
                continue
            if _RE_CANVAS_OPEN.fullmatch(line):
                self._inside = True
            else:
                out += line + "\n"
            continue

    def close(self) -> str:
        """Close the stream. An unclosed fence does NOT swallow its content
        (unlike <think>): only the already-confirmed opening marker is lost --
        everything else has already gone out through `feed`."""
        rest, self._buf = self._buf, ""
        if self._at_line_start:
            if self._inside and _RE_CANVAS_CLOSE.fullmatch(rest):
                self._inside = False
                return ""
            if not self._inside and _RE_CANVAS_OPEN.fullmatch(rest):
                self._inside = True
                return ""
        return rest


def strip_canvas_fences(text: str) -> str:
    trimmer = CanvasFenceTrimmer()
    return trimmer.feed(text) + trimmer.close()


# --- xAI cards: '<xai:...>' / '<grok:...>' ------------------------------
#
# grok-proxy (2026-08-18) writes the web UI's own cards into the token stream,
# markers and all:
#
#   <xai:tool_usage_card>
#     <xai:tool_name>web_search</xai:tool_name>
#   </xai:tool_usage_card>
#
# Today grok-proxy removes them itself, with a hand-rolled state machine over a
# stream whose tokens split tags at arbitrary points. This is the same posture
# the gateway already takes with chatgpt-proxy's canvas: the front is not the
# only thing between its artifacts and the client.
#
# LIKE <think>, and unlike a canvas fence: what sits inside a card is discarded.
# A card is UI chrome describing what the model did, never the answer.
#
# The namespaces are the whole safety argument: '<xai:' and '<grok:' are not
# valid HTML and are not syntax any model emits by accident, so this cannot eat
# ordinary prose the way a phrase list would. It is still PROVIDER-SCOPED
# (Provider.strips_xai_cards) rather than global, because a model that is asked
# to explain grok's wire format would legitimately quote these tags.
_RE_XAI_TAG = re.compile(r"<(/?)(?:xai|grok):[^>]*>")
_XAI_PREFIXES = ("<xai:", "<grok:", "</xai:", "</grok:")


def _could_be_xai_tag(s: str) -> bool:
    """True if `s`, an incomplete tag with no '>' yet, could still become one."""
    return any(p.startswith(s) or s.startswith(p) for p in _XAI_PREFIXES)


class XaiCardTrimmer:
    """Removes '<xai:...>'/'<grok:...>' cards, innards included, over a stream.

    Nesting is counted rather than matched by name: a card that opens another
    card is removed whole, and only the OUTERMOST close ends the discard. A
    self-closing tag ('<xai:card/>') opens nothing.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._depth = 0

    def feed(self, delta: str) -> str:
        self._buf += delta
        out = ""
        while True:
            start = self._buf.find("<")
            if start == -1:
                # No '<' at all: inside a card everything is discarded, outside
                # it everything is answer.
                if self._depth == 0:
                    out += self._buf
                self._buf = ""
                return out
            if self._depth == 0:
                out += self._buf[:start]
                self._buf = self._buf[start:]
            elif start > 0:
                self._buf = self._buf[start:]

            end = self._buf.find(">")
            if end == -1:
                if not _could_be_xai_tag(self._buf):
                    # Cannot become a card marker: release it (outside a card)
                    # and go on, so "2 < 3" is not held back forever.
                    if self._depth == 0:
                        out += self._buf[0]
                        self._buf = self._buf[1:]
                        continue
                    self._buf = self._buf[1:]
                    continue
                return out   # still ambiguous: wait for the rest of the tag

            tag = self._buf[:end + 1]
            match = _RE_XAI_TAG.fullmatch(tag)
            if match is None:
                # An ordinary '<...>' -- '<b>' and friends are answer text.
                if self._depth == 0:
                    out += tag
                self._buf = self._buf[end + 1:]
                continue
            self._buf = self._buf[end + 1:]
            if match.group(1):                      # '</xai:...>'
                self._depth = max(0, self._depth - 1)
            elif not tag.endswith("/>"):
                self._depth += 1

    def close(self) -> str:
        """Close the stream. An unclosed card keeps swallowing, exactly like an
        unclosed <think>: what follows an opened card is card innards."""
        rest, self._buf = self._buf, ""
        if self._depth > 0:
            return ""
        return "" if _could_be_xai_tag(rest) else rest


def strip_xai_cards(text: str) -> str:
    trimmer = XaiCardTrimmer()
    return trimmer.feed(text) + trimmer.close()


class CompositeStreamTrimmer:
    """Chains ReasoningTrimmer (reasoning is discarded) with CanvasFenceTrimmer
    (canvas fences, inner content kept) so the streaming producer (proxy.py)
    talks to a single object. Order matters: reasoning is trimmed first, then the
    fence is unwrapped over what is ALREADY visible content.

    `unwrap_canvas` (default False, same argument as in `trim()`) is a DECISION
    OF THE PROVIDER serving this stream -- proxy.py passes it from
    Provider.unwraps_canvas. With False the canvas step is skipped entirely:
    CanvasFenceTrimmer is not even instantiated, so it is obvious by reading that
    this branch does nothing to ':::' -- legitimate Docusaurus/MDX syntax in any
    provider that is not chatgpt-proxy."""

    def __init__(self, unwrap_canvas: bool = False,
                 strip_cards: bool = False) -> None:
        self._thinking = ReasoningTrimmer()
        self._canvas = CanvasFenceTrimmer() if unwrap_canvas else None
        self._cards = XaiCardTrimmer() if strip_cards else None

    @property
    def reasoning(self) -> str:
        return self._thinking.reasoning

    def feed(self, delta: str) -> str:
        out = self._thinking.feed(delta)
        if self._canvas is not None:
            out = self._canvas.feed(out)
        return self._cards.feed(out) if self._cards is not None else out

    def close(self) -> str:
        rest = self._thinking.close()
        if self._canvas is not None:
            rest = self._canvas.feed(rest) + self._canvas.close()
        if self._cards is None:
            return rest
        return self._cards.feed(rest) + self._cards.close()
