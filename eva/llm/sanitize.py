"""Make LLM text safe and natural for a TTS engine.

:func:`clean_for_tts` removes everything a voice should not read out loud (markdown,
emoji, URLs, stage directions, stray tags) and normalises punctuation.  It is applied
per chunk, right before synthesis, so it must be cheap and must never raise.

:func:`strip_think` removes ``<think>...</think>`` blocks (closed or unclosed).
"""
from __future__ import annotations

import re

# Audio tags ElevenLabs v3 renders and we allow through when the TTS supports them.
# Tags a tag-capable TTS (ElevenLabs v3) may receive inline: the non-verbal sounds plus
# the delivery cues from eva.delivery ([warm], [teasing], ...). Everything else is dropped.
from ..delivery import V3_TAG_WHITELIST as AUDIO_TAG_WHITELIST  # noqa: E402

# Words that, alone inside *asterisks* or (parentheses), are stage directions, not emphasis.
_STAGE_VERBS = frozenset(
    {
        "laughs", "laugh", "sighs", "sigh", "smiles", "smile", "nods", "nod", "pauses", "pause",
        "chuckles", "chuckle", "giggles", "giggle", "grins", "grin", "winks", "wink", "shrugs",
        "shrug", "gasps", "gasp", "exhales", "exhale", "inhales", "inhale", "whispers", "whisper",
        "sniffs", "sniff", "yawns", "yawn", "coughs", "cough", "blushes", "frowns", "hums", "hum",
        "beat", "thinking", "thinks", "sobs", "cries", "snorts", "groans", "groan", "moans", "clears throat",
        "leans in", "leans back", "looks up", "looks down", "smiling", "laughing", "sighing", "whispering",
        "softly", "gently", "warmly", "quietly",
    }
)

_THINK_BLOCK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>\s*", re.DOTALL | re.IGNORECASE)
_THINK_UNCLOSED_RE = re.compile(r"<think(?:ing)?>.*\Z", re.DOTALL | re.IGNORECASE)
_THINK_STRAY_CLOSE_RE = re.compile(r"</?think(?:ing)?>", re.IGNORECASE)

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n?")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")
_URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"')\]]+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_HEADER_RE = re.compile(r"^[ \t]*#{1,6}[ \t]+", re.MULTILINE)
_BULLET_RE = re.compile(r"^[ \t]*(?:[-*•▪◦+]|\d{1,3}[.)])[ \t]+", re.MULTILINE)
_BLOCKQUOTE_RE = re.compile(r"^[ \t]*>[ \t]?", re.MULTILINE)
_HRULE_RE = re.compile(r"^[ \t]*(?:[-*_][ \t]*){3,}$", re.MULTILINE)
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__", re.DOTALL)
_STAR_SPAN_RE = re.compile(r"\*([^*\n]{1,80}?)\*")
_UNDERSCORE_EM_RE = re.compile(r"(?<![\w])_([^_\n]{1,80}?)_(?![\w])")
_PAREN_STAGE_RE = re.compile(r"\(([^()\n]{1,40})\)")
_BRACKET_TAG_RE = re.compile(r"\[([^\[\]]*)\]")
_ANGLE_TAG_RE = re.compile(r"</?[a-zA-Z][^<>]{0,40}>")
_MULTI_SPACE_RE = re.compile(r"[ \t\r\f\v]+")
# Control characters (a NUL byte was received from gpt-oss-120b) except \n and \t.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f﻿]")
# Model-emitted transcript markers ("--- End of conversation ---", "(End)", "END OF TRANSCRIPT").
_MARKER_LINE_RE = re.compile(
    r"^[ \t]*[-*_=#]*[ \t]*(?:end of (?:conversation|transcript|response|message|turn|reply)|"
    r"end of dialog(?:ue)?|\(?end\)?)[ \t]*[-*_=#]*[ \t.]*$",
    re.IGNORECASE | re.MULTILINE,
)
# The same marker glued to the end of a line ("... see you. --- End of conversation ---").
_MARKER_INLINE_RE = re.compile(
    r"[ \t]*[-*_=#]{2,}[ \t]*(?:end of (?:conversation|transcript|response|message|turn|reply)|"
    r"end of dialog(?:ue)?)[ \t]*[-*_=#]*[ \t.]*$",
    re.IGNORECASE | re.MULTILINE,
)
# A bare JSON object in spoken text (gpt-oss leaks tool calls as content after a tool
# result: {"text":"Call mom later tonight"}). Nothing a voice should ever read.
_JSON_OBJ_RE = re.compile(r"\{\s*\"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}")
_DASH_RE = re.compile(r"[ \t]*[—―]+[ \t]*")  # em dash / horizontal bar -> comma pause
_SPACED_HYPHEN_RE = re.compile(r"(?<=\S)[ \t]+-{1,2}[ \t]+(?=\S)")  # "well - I mean" / "well -- I mean"
_DUP_SHORT_RE = re.compile(r"\b(\w{1,5}[.!?])(?:\s*\1)+")  # "mm.mm." / "Okay. Okay." -> once
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.!?;:…])")
_PUNCT_RUN_RE = re.compile(r"([!?]){3,}")

# Emoji / pictographs / symbols that no TTS should read.  Ranges, not exhaustive lists.
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"  # emoticons, symbols, pictographs, flags, supplemental
    "\U0001FB00-\U0001FFFF"
    "\u2600-\u27BF"  # misc symbols, dingbats
    "\u2B00-\u2BFF"  # arrows / stars
    "\u2300-\u23FF"  # misc technical (watch, alarm clock)
    "\u2190-\u21FF"  # arrows
    "\u25A0-\u25FF"  # geometric shapes
    "\u2900-\u297F"
    "\u3030\u303D\u3297\u3299\u00A9\u00AE\u2122\u2139"
    "\u200D\uFE0F\u20E3"  # ZWJ, variation selector, keycap
    "\U000E0020-\U000E007F"  # tag characters (flags)
    "]+",
)

_QUOTE_MAP = str.maketrans(
    {
        "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
        "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"', "\u2033": '"',
        "\u00ab": '"', "\u00bb": '"',
        # gpt-oss likes typographic spaces / hyphens: "7:17<nnbsp>PM", "two<nb-hyphen>minute"
        "\u00a0": " ", "\u2007": " ", "\u2009": " ", "\u202f": " ", "\u200b": "", "\u2060": "",
        "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2043": "-",
    }
)


def strip_think(text: str) -> str:
    """Remove ``<think>...</think>`` (or ``<thinking>``) blocks, closed or unclosed."""
    if not text:
        return ""
    text = _THINK_BLOCK_RE.sub("", text)
    text = _THINK_UNCLOSED_RE.sub("", text)
    text = _THINK_STRAY_CLOSE_RE.sub("", text)
    return text.lstrip("\n")


def _normalise_tag(inner: str) -> str:
    return re.sub(r"\s+", " ", inner.strip().lower().strip(".!,"))


def _star_span(m: re.Match[str]) -> str:
    inner = m.group(1).strip()
    if not inner:
        return ""
    low = inner.lower().rstrip(".!")
    words = low.split()
    # Stage direction: a known verb, or a multi-word gesture description.
    if low in _STAGE_VERBS or (len(words) >= 2 and words[0] in _STAGE_VERBS) or (
        len(words) >= 2 and words[-1] in _STAGE_VERBS
    ):
        return ""
    if len(words) >= 3 and not any(ch in inner for ch in ",;:?!"):
        # "*leans forward with a grin*" style narration
        if words[0].endswith("s") or words[0].endswith("ing"):
            return ""
    return inner  # plain emphasis: unwrap


def _paren_span(m: re.Match[str]) -> str:
    inner = m.group(1).strip()
    if _normalise_tag(inner) in _STAGE_VERBS:
        return ""
    return m.group(0)


def clean_for_tts(text: str, keep_audio_tags: bool = False) -> str:
    """Return ``text`` stripped of everything a voice should not read.

    Parameters
    ----------
    text:
        A chunk (or a full reply) of LLM output.
    keep_audio_tags:
        If True, whitelisted bracket tags such as ``[laughs]`` are kept (normalised to
        lower case) for TTS engines that render them; every other ``[tag]`` is dropped.
        If False all bracketed tags are removed.
    """
    if not text:
        return ""
    t = strip_think(text)
    t = _CONTROL_RE.sub("", t)
    t = t.translate(_QUOTE_MAP)
    t = _MARKER_LINE_RE.sub("", t)
    t = _MARKER_INLINE_RE.sub("", t)
    t = _JSON_OBJ_RE.sub(" ", t)
    t = _FENCE_RE.sub("", t)
    t = t.replace("`", "")
    t = _MD_LINK_RE.sub(r"\1", t)
    t = _URL_RE.sub("a link", t)
    t = _EMAIL_RE.sub("an email address", t)
    t = _HRULE_RE.sub("", t)
    t = _HEADER_RE.sub("", t)
    t = _BLOCKQUOTE_RE.sub("", t)
    t = _BULLET_RE.sub("", t)
    t = t.replace("|", " ")
    t = _BOLD_RE.sub(lambda m: m.group(1) or m.group(2) or "", t)
    t = _STAR_SPAN_RE.sub(_star_span, t)
    t = _UNDERSCORE_EM_RE.sub(r"\1", t)
    t = t.replace("*", "")
    t = _PAREN_STAGE_RE.sub(_paren_span, t)
    t = _ANGLE_TAG_RE.sub("", t)

    def _tag(m: re.Match[str]) -> str:
        inner = _normalise_tag(m.group(1))
        if keep_audio_tags and inner in AUDIO_TAG_WHITELIST:
            return f"[{inner}]"
        return ""

    t = _BRACKET_TAG_RE.sub(_tag, t)
    t = _EMOJI_RE.sub("", t)
    # Dashes used as pauses ("That's rough — really rough") read more naturally as a
    # comma pause in every TTS tested; a dash at the very end of a chunk is dropped.
    t = _DASH_RE.sub(", ", t)
    t = _SPACED_HYPHEN_RE.sub(", ", t)
    t = re.sub(r",\s*,", ",", t)
    t = re.sub(r"([.!?;:,])\s*,", r"\1", t)
    t = re.sub(r"(?:^|\n)\s*,\s*", lambda m: m.group(0)[0] if m.group(0)[0] == "\n" else "", t)
    t = re.sub(r",\s*$", "", t)
    t = _PUNCT_RUN_RE.sub(r"\1\1", t)
    t = _DUP_SHORT_RE.sub(r"\1", t)
    t = t.replace("\n", " ")
    t = _MULTI_SPACE_RE.sub(" ", t)
    t = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", t)
    t = re.sub(r"\(\s*\)", "", t)
    t = _MULTI_SPACE_RE.sub(" ", t).strip()
    return t


__all__ = ["clean_for_tts", "strip_think", "AUDIO_TAG_WHITELIST"]
