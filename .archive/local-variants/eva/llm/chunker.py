"""Turn a token stream into TTS-sized chunks with an early first chunk.

The chunker is fed raw LLM deltas and returns pieces of text that are safe to hand to
the TTS as soon as they are complete:

* The **first** chunk may be cut at a comma / semicolon / dash once at least
  ``first_chunk_min_chars`` characters are buffered, so audio starts early
  ("Honestly," ... "that sounds really rough.").
* Every later chunk is cut at a sentence ender ``. ! ? …`` (or a run of them such as
  ``?!`` / ``...``) or at a newline.
* It does not split inside numbers (``3.5``), after common abbreviations (``Dr.``,
  ``e.g.``), right after a single capital letter (``J. K.``), inside an unclosed
  ``[audio tag]``, or inside an unclosed ``{``: gpt-oss leaks tool calls as JSON text
  with commas and newlines in it, and kept whole the pipeline recovers the call instead
  of speaking half of it.  A ``{`` left open for more than ``MAX_BRACE_HOLD`` characters
  is ignored so a stray brace can never hold a reply back.
* Chunks shorter than ``min_chunk_chars`` are merged into the next one ("Hi." +
  " How are you?" -> "Hi. How are you?"), except at :meth:`flush`.

Nothing is decided until the character *after* an ender is known (whitespace means a
real boundary; a digit or letter means ``3.5`` / ``e.g.``), so an ender at the very end
of the buffer waits for the next delta or for :meth:`flush`.
"""
from __future__ import annotations

_ENDERS = ".!?…"
MAX_BRACE_HOLD = 400  # characters an unclosed "{" may hold a cut back
_CLOSERS = "\"'”’)]»"
_EARLY_CUTS = ",;—–"

# Words after which a period is (almost always) not a sentence end.
_ABBREVIATIONS = frozenset(
    {
        "dr", "mr", "mrs", "ms", "mx", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
        "inc", "ltd", "corp", "mt", "approx", "dept", "u.s", "u.k", "a.m", "p.m", "ph.d",
        "jan", "feb", "apr", "aug", "sep", "sept", "oct", "nov", "dec",
        "tue", "tues", "thu", "thur", "thurs", "fri",
    }
)


class SentenceChunker:
    """Incremental sentence splitter tuned for spoken replies.

    Usage::

        ch = SentenceChunker()
        async for ev in llm.stream(...):
            if isinstance(ev, LLMDelta):
                for chunk in ch.feed(ev.text):
                    await tts_queue.put(chunk)
        for chunk in ch.flush():
            await tts_queue.put(chunk)
    """

    def __init__(self, first_chunk_min_chars: int = 14, min_chunk_chars: int = 6) -> None:
        self.first_chunk_min_chars = first_chunk_min_chars
        self.min_chunk_chars = min_chunk_chars
        self._buf = ""
        self._scan = 0  # first index in _buf that has not been examined as a cut point
        self._emitted_first = False
        self.chunks_emitted = 0

    # ------------------------------------------------------------------ public
    def reset(self) -> None:
        """Forget buffered text and start a new response (called by :meth:`flush`)."""
        self._buf = ""
        self._scan = 0
        self._emitted_first = False
        self.chunks_emitted = 0

    @property
    def pending(self) -> str:
        """Text buffered but not yet emitted."""
        return self._buf

    def feed(self, delta: str) -> list[str]:
        """Add streamed text and return any chunks that became ready."""
        if not delta:
            return []
        self._buf += delta
        out: list[str] = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            chunk = self._buf[:cut].strip()
            self._buf = self._buf[cut:]
            self._scan = 0
            if chunk:
                out.append(chunk)
                self._emitted_first = True
                self.chunks_emitted += 1
        return out

    def flush(self) -> list[str]:
        """Emit whatever is left (end of the LLM turn) and reset for the next turn."""
        rest = self._buf.strip()
        self.reset()
        return [rest] if rest else []

    # ----------------------------------------------------------------- private
    def _balanced_brackets(self, upto: int) -> bool:
        """No unclosed ``[`` before ``upto``, and no unclosed ``{`` within MAX_BRACE_HOLD."""
        seg = self._buf[:upto]
        if seg.count("[") > seg.count("]"):
            return False
        if seg.count("{") > seg.count("}"):
            return upto - seg.rfind("{") > MAX_BRACE_HOLD
        return True

    def _is_abbreviation(self, end: int) -> bool:
        """True if the '.' at ``end`` follows an abbreviation or a lone capital letter."""
        before = self._buf[:end]
        # last whitespace-separated token (strip leading quotes/parens)
        j = len(before)
        while j > 0 and not before[j - 1].isspace():
            j -= 1
        token = before[j:].lstrip("\"'“‘([")
        if not token:
            return False
        low = token.lower()
        if low in _ABBREVIATIONS:
            return True
        if len(token) == 1 and token.isalpha() and token.isupper():
            return True
        return False

    def _find_cut(self) -> int | None:
        """Return the index (exclusive) where the next chunk ends, or None to wait."""
        buf = self._buf
        n = len(buf)
        i = self._scan
        while i < n:
            c = buf[i]
            if c == "\n":
                candidate = i + 1
                if self._balanced_brackets(i) and self._accept(candidate, hard=True):
                    return candidate
                i += 1
                continue
            if c in _ENDERS:
                j = i
                while j + 1 < n and buf[j + 1] in _ENDERS:
                    j += 1
                k = j + 1
                while k < n and buf[k] in _CLOSERS:
                    k += 1
                if k >= n:
                    self._scan = i  # need to see what follows the ender
                    return None
                if not buf[k].isspace():
                    # "3.5", "e.g", "U.S.A" ... but "done.Anything else?" (gpt-oss glues
                    # its messages together without a space) is a real boundary: a
                    # lowercase letter, the ender run, then an uppercase letter.
                    glued = (
                        k == j + 1
                        and buf[k].isalpha()
                        and buf[k].isupper()
                        and i > 0
                        and buf[i - 1].isalpha()
                        and buf[i - 1].islower()
                        and not (c == "." and j == i and self._is_abbreviation(i))
                    )
                    if not glued:
                        i = j + 1
                        continue
                    if self._accept(k, hard=False):
                        return k
                    i = j + 1
                    continue
                if c == "." and j == i and self._is_abbreviation(i):
                    i = j + 1
                    continue
                if not self._balanced_brackets(i):
                    i = j + 1
                    continue
                if self._accept(k, hard=False):
                    return k
                i = j + 1
                continue
            if not self._emitted_first and c in _EARLY_CUTS:
                head = buf[:i].strip()
                if len(head) >= self.first_chunk_min_chars and self._balanced_brackets(i):
                    if c in ",;":
                        if i + 1 >= n:
                            self._scan = i
                            return None
                        if not buf[i + 1].isspace():
                            i += 1  # "1,000"
                            continue
                    return i + 1
                i += 1
                continue
            if not self._emitted_first and c == "-" and i > 0 and buf[i - 1] == " ":
                # spaced hyphen used as a dash: "well - I mean"
                if len(buf[:i].strip()) >= self.first_chunk_min_chars and self._balanced_brackets(i):
                    if i + 1 >= n:
                        self._scan = i
                        return None
                    if buf[i + 1] == " ":
                        return i + 1
            i += 1
        self._scan = n
        return None

    def _accept(self, cut: int, hard: bool) -> bool:
        """Apply the min-length merge rule to a candidate cut."""
        chunk = self._buf[:cut].strip()
        if not chunk:
            return hard  # blank line: drop it (feed skips empty chunks)
        return len(chunk) >= self.min_chunk_chars
