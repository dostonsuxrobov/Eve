"""Delivery layer: *how* a sentence is said, not what it says.

Three small, dependency-free helpers shared by the pipeline and the TTS providers:

* **Delivery cues.** The brain may start a sentence with one bracketed cue such as
  ``[warm]`` or ``[teasing]``. :func:`extract_cue` splits it off. A TTS that renders
  tags natively (ElevenLabs v3) receives the cue inline; a TTS that does not (Flash,
  Turbo) gets it mapped onto voice settings by :func:`cue_settings`.
* **Language detection.** :func:`detect_lang` tells Russian from English by script so
  the right voice and the right fillers are used per sentence.
* **Hallucination gate.** :func:`looks_hallucinated` rejects the classic noise
  transcripts that Whisper-class models invent on silence ("Thank you.", "you", ...).
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

# Cues the persona prompt offers. Keys are what the brain writes; values are the
# ElevenLabs voice-setting deltas applied for models without native tags.
#   stability: lower = more expressive / variable, higher = steadier
#   style:     higher = more exaggerated delivery (0-1)
#   speed:     0.7-1.2, 1.0 = normal
CUE_SETTINGS: dict[str, dict[str, float]] = {
    "warm": {"stability": 0.50, "style": 0.15, "speed": 0.97},
    "soft": {"stability": 0.60, "style": 0.05, "speed": 0.92},
    "gentle": {"stability": 0.60, "style": 0.05, "speed": 0.92},
    "quiet": {"stability": 0.65, "style": 0.00, "speed": 0.90},
    "sad": {"stability": 0.65, "style": 0.05, "speed": 0.90},
    "thoughtful": {"stability": 0.55, "style": 0.10, "speed": 0.90},
    "slow": {"stability": 0.55, "style": 0.10, "speed": 0.88},
    "bright": {"stability": 0.35, "style": 0.35, "speed": 1.05},
    "playful": {"stability": 0.35, "style": 0.35, "speed": 1.04},
    "teasing": {"stability": 0.35, "style": 0.40, "speed": 1.00},
    "excited": {"stability": 0.30, "style": 0.45, "speed": 1.08},
    "curious": {"stability": 0.45, "style": 0.25, "speed": 1.02},
    "amused": {"stability": 0.40, "style": 0.30, "speed": 1.02},
    "serious": {"stability": 0.60, "style": 0.05, "speed": 0.95},
    "flat": {"stability": 0.55, "style": 0.00, "speed": 1.00},
}

# Non-verbal tags that ElevenLabs v3 renders as sounds; they never map to settings.
NONVERBAL_TAGS: frozenset[str] = frozenset(
    {
        "laughs", "laughing", "chuckles", "giggles", "sighs", "exhales", "gasps", "whispers",
        "pause", "short pause", "long pause", "clears throat", "sniffs", "snorts", "hums",
        "mischievously", "sarcastic", "crying", "yawns",
        "groans", "coughs", "sniffles",  # Orpheus renders these too (eva/tts/voice_server.py)
    }
)

# Everything a v3-capable TTS may receive inline.
V3_TAG_WHITELIST: frozenset[str] = frozenset(CUE_SETTINGS) | NONVERBAL_TAGS

_LEADING_CUE_RE = re.compile(r"^\s*\[\s*([A-Za-z][A-Za-z \-]{1,24})\s*\]\s*")
_ANY_TAG_RE = re.compile(r"\[[^\[\]]{1,30}\]")
_WS_RE = re.compile(r"\s+")


def normalise_tag(inner: str) -> str:
    return _WS_RE.sub(" ", inner.strip().lower().replace("-", " "))


def extract_cue(text: str) -> tuple[str | None, str]:
    """Split a leading ``[cue]`` off ``text``.

    Returns ``(cue, rest)``; ``cue`` is ``None`` when the text does not start with a
    known cue. Non-verbal tags (``[laughs]``) are *not* cues: they stay in the text so a
    tag-capable TTS can render them.
    """
    m = _LEADING_CUE_RE.match(text)
    if not m:
        return None, text
    cue = normalise_tag(m.group(1))
    if cue in CUE_SETTINGS:
        return cue, text[m.end() :]
    return None, text


def strip_tags(text: str) -> str:
    """Remove every ``[bracketed]`` tag (for a TTS request that cannot render them)."""
    return _WS_RE.sub(" ", _ANY_TAG_RE.sub("", text)).strip()


def cue_settings(cue: str | None, base: dict[str, Any]) -> dict[str, Any]:
    """Voice settings for ``cue`` on a tag-less model: ``base`` merged with the cue deltas."""
    if not cue or cue not in CUE_SETTINGS:
        return dict(base)
    out = dict(base)
    out.update(CUE_SETTINGS[cue])
    return out


# ------------------------------------------------------------------- language
_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def detect_lang(text: str, default: str = "en") -> str:
    """``"ru"`` if Cyrillic letters dominate, ``"en"`` if Latin letters do, else ``default``."""
    cyr = len(_CYRILLIC_RE.findall(text))
    lat = len(_LATIN_RE.findall(text))
    if cyr == 0 and lat == 0:
        return default
    return "ru" if cyr >= lat else "en"


# --------------------------------------------------------- hallucination gate
_ALPHA_RE = re.compile(r"[^\W\d_]", re.UNICODE)
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

# Subtitle-corpus junk that no STT should ever turn into a reply.
SUBTITLE_PHANTOMS: frozenset[str] = frozenset(
    {
        "thank you for watching", "thanks for watching", "subtitles by the amazing team",
        "please subscribe", "like and subscribe", "the end", "продолжение следует", "субтитры",
        "спасибо за просмотр",
    }
)
# What Whisper-family models invent for silence, breaths and fan noise. Only applied to
# a Whisper-class transcript: on Scribe / Parakeet these are real short answers
# ("yeah", "no", "okay", "bye") and dropping them looked like Eva ignoring the user.
WHISPER_PHANTOMS: frozenset[str] = SUBTITLE_PHANTOMS | frozenset(
    {
        "you", "thank you", "thanks", "bye", "goodbye", "okay", "ok", "so", "the", "um", "uh",
        "hmm", "mm", "oh", "yeah", "yes", "no", "i'm sorry", "sorry", "спасибо", "да", "нет",
        "так", "ну", "хм",
    }
)
# A bare hesitation: the user is thinking, not done. Waited on, never answered.
HESITATIONS: frozenset[str] = frozenset(
    {"um", "uh", "uhm", "hmm", "hm", "mm", "mhm", "er", "erm", "эм", "ээ", "мм", "хм", "ну", "э"}
)


def is_hesitation(text: str) -> bool:
    """True if ``text`` is nothing but hesitation sounds ("Uh.", "Hmm, uh")."""
    bare = _WS_RE.sub(" ", _PUNCT_RE.sub("", text).strip().lower())
    words = bare.split()
    return bool(words) and all(w in HESITATIONS for w in words)


def looks_hallucinated(text: str, meta: dict[str, Any] | None = None, *, whisper_class: bool | None = None) -> str | None:
    """Return a reason string if ``text`` should be dropped as a phantom transcript, else ``None``.

    Rules, cheapest first: no letters at all; subtitle junk on its own; for a
    Whisper-class transcript (``whisper_class``, else inferred from the presence of
    ``segment_scores`` in ``meta``) also the silence phantoms and segment scores that
    all say "no speech" or "very low confidence".
    """
    stripped = text.strip()
    if not _ALPHA_RE.search(stripped):
        return "no letters"
    bare = _PUNCT_RE.sub("", stripped).strip().lower()
    bare = _WS_RE.sub(" ", bare)
    if bare in SUBTITLE_PHANTOMS:
        return f"phantom phrase {bare!r}"
    scores = (meta or {}).get("segment_scores") or []
    if whisper_class is None:
        whisper_class = bool(scores)
    if not whisper_class:
        return None
    if bare in WHISPER_PHANTOMS:
        return f"phantom phrase {bare!r}"
    if scores:
        nsp = [float(s.get("no_speech_prob", 0.0)) for s in scores]
        lp = [float(s.get("avg_logprob", 0.0)) for s in scores]
        if nsp and min(nsp) > 0.5:
            return f"no_speech_prob {min(nsp):.2f}"
        if lp and max(lp) < -1.0:
            return f"avg_logprob {max(lp):.2f}"
        # A single short word with weak evidence is almost always noise.
        if len(bare.split()) <= 2 and nsp and min(nsp) > 0.3 and lp and max(lp) < -0.6:
            return "short + weak scores"
    return None


# ------------------------------------------------------ turn completeness
_TERMINAL_RE = re.compile(r"[.!?…]\s*[\"'»)]*\s*$")
_TRAILING_OPEN_RE = re.compile(r"[,;:\-–—]\s*$")
CONTINUATION_WORDS: frozenset[str] = frozenset(
    {
        # en: conjunctions, prepositions, determiners and fillers nobody ends a sentence on
        "and", "but", "so", "because", "or", "then", "like", "um", "uh", "the", "a", "an", "to",
        "of", "with", "that", "if", "when", "which", "while", "although", "though", "as", "for",
        "in", "on", "at", "by", "from", "about", "into", "than", "whether", "unless", "until",
        # ru
        "и", "а", "но", "что", "чтобы", "потому", "если", "когда", "или", "как", "типа", "в",
        "на", "с", "у", "за", "по", "про", "для", "от", "до", "из", "чем", "бы", "хотя", "пока",
        "чтоб", "будто", "либо",
    }
)


def looks_incomplete(text: str) -> bool:
    """Heuristic: does this transcript look like the user was cut off mid-thought?

    Unfinished if it is nothing but hesitation sounds ("Uh.", "Hmm"), trails off with a
    comma/dash, ends on a continuation word ("and", "because", "и", "потому что"), or
    is three or more words with no final punctuation. Short punctuation-less replies
    ("yeah", "not really") count as done.
    """
    t = text.strip()
    if not t:
        return False
    if is_hesitation(t):
        return True
    if _TERMINAL_RE.search(t):
        return False
    if _TRAILING_OPEN_RE.search(t):
        return True
    words = re.findall(r"[\w']+", t.lower(), re.UNICODE)
    if not words:
        return False
    if words[-1] in CONTINUATION_WORDS:
        return True
    return len(words) >= 3


# ------------------------------------------------------------- self echo
_WORDS_RE = re.compile(r"[\w']+", re.UNICODE)


def echo_similarity(text: str, spoken: str) -> float:
    """0..1: how much ``text`` resembles some stretch of ``spoken`` (character-level).

    The STT garbles Eva's own voice coming back through the speakers ("когда придёшь к
    решению" came back as "А когда придёшь к ней, шей"), so exact word runs miss it.
    This slides a window of about the transcript's length over what she said and takes
    the best :class:`difflib.SequenceMatcher` ratio.
    """
    words = _WORDS_RE.findall(text.lower())
    said = _WORDS_RE.findall(spoken.lower())
    if not words or not said:
        return 0.0
    target = " ".join(words)
    best = 0.0
    n = len(words)
    for size in range(max(1, n - 2), n + 3):
        for j in range(0, max(1, len(said) - size + 1)):
            window = " ".join(said[j : j + size])
            r = SequenceMatcher(None, target, window).ratio()
            if r > best:
                best = r
                if best >= 0.99:
                    return best
    return best


def looks_like_echo(
    text: str, spoken: str, *, min_words: int = 3, max_words: int = 12, min_overlap: float = 0.8, fuzzy: float | None = None
) -> bool:
    """True if a short transcript is (a piece of) what the agent itself just said.

    Through speakers the microphone hears the reply, and a mic's echo canceller takes
    the first seconds of a session to converge, so early on the STT can return
    Eva's own greeting ("What's on your mind today?") or a fragment of it. A
    transcript of ``min_words``..``max_words`` words that is, for at least
    ``min_overlap`` of its length, a contiguous run of ``spoken`` is echo, not a turn;
    with ``fuzzy`` set, an :func:`echo_similarity` at or above it counts too (for
    utterances that began while she was audible, where echo is the likely story).
    One- and two-word transcripts are never judged by the exact rule: "not much" after
    "not much, you?" is an answer.
    """
    words = _WORDS_RE.findall(text.lower())
    if len(words) < min_words or len(words) > max_words:
        return False
    said = _WORDS_RE.findall(spoken.lower())
    if len(said) < min_words:
        return False
    if fuzzy is not None and echo_similarity(text, spoken) >= fuzzy:
        return True
    # longest contiguous run of consecutive transcript words found consecutively in spoken
    best = 0
    for i in range(len(words)):
        for j in range(len(said)):
            k = 0
            while i + k < len(words) and j + k < len(said) and words[i + k] == said[j + k]:
                k += 1
            best = max(best, k)
    return best / len(words) >= min_overlap


__all__ = [
    "CUE_SETTINGS",
    "CONTINUATION_WORDS",
    "HESITATIONS",
    "SUBTITLE_PHANTOMS",
    "echo_similarity",
    "is_hesitation",
    "looks_incomplete",
    "looks_like_echo",
    "NONVERBAL_TAGS",
    "V3_TAG_WHITELIST",
    "WHISPER_PHANTOMS",
    "cue_settings",
    "detect_lang",
    "extract_cue",
    "looks_hallucinated",
    "normalise_tag",
    "strip_tags",
]
