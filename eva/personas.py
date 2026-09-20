"""Persona loading and system-prompt rendering.

Persona files live in ``eva/assets/personas/<lang>/<name>.md``: one file per language,
the English one being the reference and the fallback.  Each starts with a small front
matter block between two ``---`` lines (``name``, ``description``, ``suggested_voice``
and, optionally, ``fillers`` / ``tool_hints`` / ``backchannels`` that override the
language's defaults from ``eva/assets/lang``) followed by the system prompt template.
The template contains literal slots that :func:`render` fills:

    {user_name}        who Eva is talking to
    {now}              a human readable local timestamp
    {memory}           durable facts about the user (see eva.memory)
    {audio_tags_rule}  allow / forbid bracketed audio tags depending on the TTS
    {tool_notes}       what tools are connected right now, or a note that none are

Only those five slots are substituted, so stray braces elsewhere in a prompt are
left untouched (no ``str.format`` surprises).

Usage::

    from eva.personas import load_persona, render, now_string
    p = load_persona("eva", lang="ru")          # falls back to en/eva.md if there is no ru/
    system_prompt = render(p, supports_audio_tags=False, memory_text=mem.as_prompt_text(),
                           now=now_string(), user_name="Sam", tool_notes=registry.notes(),
                           locked_language="Russian")
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

PERSONAS_DIR = Path(__file__).resolve().parent / "assets" / "personas"
DEFAULT_PERSONA = "eva"
DEFAULT_LANG = "en"

TEMPLATE_SLOTS = ("user_name", "now", "memory", "audio_tags_rule", "tool_notes", "language_rule")
_SLOT_RE = re.compile(r"\{(" + "|".join(TEMPLATE_SLOTS) + r")\}")

AUDIO_TAGS_ALLOWED = (
    "Delivery. Your voice renders bracketed tags, so you shape HOW each sentence sounds. "
    "You may start a sentence with exactly one delivery cue from this list and no other: "
    "[warm] [soft] [gentle] [quiet] [sad] [thoughtful] [slow] [bright] [playful] [teasing] "
    "[excited] [curious] [amused] [serious]. Use a cue on roughly one sentence in two, "
    "chosen for the feeling behind the words, and vary them; a reply is not a list of tags. "
    "Sounds are allowed too, only where a real person would make them: [laughs], [chuckles], "
    "[sighs], [exhales], [whispers], [pause], at most one sound per reply, never as a "
    "substitute for words. Never write any other bracketed text."
)
DELIVERY_CUES_ONLY = (
    "Delivery. You can shape HOW a sentence sounds by starting it with exactly one cue from "
    "this list and no other: [warm] [soft] [gentle] [quiet] [sad] [thoughtful] [slow] "
    "[bright] [playful] [teasing] [excited] [curious] [amused] [serious]. The cue is never "
    "spoken; it only changes the voice. Use one on roughly one sentence in two, chosen for the "
    "feeling behind the words, and vary them. Never write sound tags like [laughs] or [sighs] "
    "or any other bracketed text; show those through words and rhythm."
)
LANGUAGE_RULE = (
    "Language. Answer in the language {user_name} just used. In Russian, talk the way a close "
    "friend talks: informal, natural spoken Russian, short sentences, no anglicisms and no "
    "translated-sounding phrasing; if they switch languages, switch with them without comment."
)
LOCKED_LANGUAGE_RULE = (
    "Language. This conversation is in {language}: always answer in {language}, even if a word "
    "or a sentence from {user_name} comes through in another language."
)
AUDIO_TAGS_FORBIDDEN = (
    "Audio tags. Never write bracketed stage directions or sound tags like [laughs] or "
    "[sighs]; they'd be read aloud or dropped. Show feeling through word choice and "
    "rhythm instead."
)

NO_MEMORY_TEXT = (
    "Nothing yet. This may be your first conversation, so pick up their name and the "
    "small details naturally as they come. You do not know their name yet: never guess "
    "or invent one, just talk to them without a name until they tell you."
)
NO_TOOLS_TEXT = "No tools are connected right now."


@dataclass(frozen=True)
class Persona:
    """One loaded persona file: front matter fields plus the raw prompt template.

    ``lang`` is the directory the file came from; ``fillers`` / ``tool_hints`` /
    ``backchannels`` are optional per-persona overrides of that language's defaults
    (empty = use ``eva.lang``'s).
    """

    name: str
    description: str
    suggested_voice: str
    lang: str = DEFAULT_LANG
    fillers: list[str] = field(default_factory=list)
    tool_hints: list[str] = field(default_factory=list)
    backchannels: list[str] = field(default_factory=list)
    template: str = ""

    def slots_present(self) -> set[str]:
        """Which of the template slots this template actually uses."""
        return set(_SLOT_RE.findall(self.template))


# --------------------------------------------------------------------- loading
def list_personas(lang: str = DEFAULT_LANG) -> list[str]:
    """Names of every persona file for ``lang`` (English by default), default first."""
    names = sorted(p.stem for p in (PERSONAS_DIR / lang).glob("*.md"))
    if DEFAULT_PERSONA in names:
        names.remove(DEFAULT_PERSONA)
        names.insert(0, DEFAULT_PERSONA)
    return names


def persona_path(name: str, lang: str = DEFAULT_LANG) -> Path:
    """``assets/personas/<lang>/<name>.md`` if it exists, else the English file."""
    for code in (lang, DEFAULT_LANG):
        path = PERSONAS_DIR / code / f"{name}.md"
        if path.exists():
            return path
    raise FileNotFoundError(f"persona {name!r} not found; available: {', '.join(list_personas())}")


def load_persona(name: str = DEFAULT_PERSONA, lang: str = DEFAULT_LANG) -> Persona:
    """Load and parse the persona ``name`` for ``lang`` (falling back to English).

    ``name`` may also be a path to a ``.md`` file outside the package.
    """
    path = Path(name)
    if path.suffix == ".md" and path.exists():
        code = path.parent.name if (PERSONAS_DIR / path.parent.name).is_dir() else lang
    else:
        path = persona_path(name, lang)
        code = path.parent.name
    text = path.read_text(encoding="utf-8")
    meta, body = _split_front_matter(text)
    return Persona(
        name=str(meta.get("name") or path.stem),
        description=str(meta.get("description") or ""),
        suggested_voice=str(meta.get("suggested_voice") or "sarah"),
        lang=code,
        fillers=_as_list(meta.get("fillers")),
        tool_hints=_as_list(meta.get("tool_hints")),
        backchannels=_as_list(meta.get("backchannels")),
        template=body.strip() + "\n",
    )


def _split_front_matter(text: str) -> tuple[dict[str, object], str]:
    """Split ``---`` delimited front matter from the body. Tolerates no front matter."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return _parse_front_matter(lines[1:i]), "\n".join(lines[i + 1 :])
    return {}, text  # unterminated block: treat everything as body


def _parse_front_matter(lines: list[str]) -> dict[str, object]:
    """Tiny YAML subset: ``key: value``, ``key: [a, b]`` and ``key:`` + ``- item`` lines."""
    meta: dict[str, object] = {}
    current_list: list[str] | None = None
    for raw in lines:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        stripped = raw.strip()
        if stripped.startswith("- ") and current_list is not None:
            current_list.append(_unquote(stripped[2:]))
            continue
        if ":" in stripped:
            key, _, value = stripped.partition(":")
            key, value = key.strip(), value.strip()
            if not value:
                current_list = []
                meta[key] = current_list
            elif value.startswith("[") and value.endswith("]"):
                meta[key] = [_unquote(v) for v in value[1:-1].split(",") if v.strip()]
                current_list = None
            else:
                meta[key] = _unquote(value)
                current_list = None
    return meta


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    return s


def _as_list(v: object) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    return [str(x) for x in v]  # type: ignore[union-attr]


# -------------------------------------------------------------------- rendering
def now_string(dt: datetime | None = None) -> str:
    """A prompt-friendly local timestamp, e.g. ``Thursday 17 September 2026, 7:05 pm``."""
    dt = dt or datetime.now().astimezone()
    hour12 = dt.hour % 12 or 12
    ampm = "am" if dt.hour < 12 else "pm"
    return f"{dt:%A} {dt.day} {dt:%B %Y}, {hour12}:{dt:%M} {ampm}"


def render(
    persona: Persona,
    *,
    supports_audio_tags: bool,
    memory_text: str,
    now: str | None = None,
    user_name: str | None = None,
    tool_notes: str | None = None,
    delivery_cues: bool = False,
    locked_language: str | None = None,
) -> str:
    """Fill the persona template and return the final system prompt.

    ``supports_audio_tags`` decides whether ``{audio_tags_rule}`` becomes permission
    to use a few ElevenLabs-v3 style tags or a rule to never write bracketed tags.
    ``locked_language`` (a language name, e.g. "Russian") turns ``{language_rule}`` into
    "always answer in <language>"; without it the bilingual follow-the-user rule is used.
    Empty or ``None`` ``memory_text`` / ``tool_notes`` / ``user_name`` / ``now`` get
    sensible fallbacks so the prompt never contains a dangling empty section.
    """
    if supports_audio_tags:
        delivery = AUDIO_TAGS_ALLOWED  # v3: cues AND sounds inline
    elif delivery_cues:
        delivery = DELIVERY_CUES_ONLY  # flash/turbo: cues mapped to voice settings
    else:
        delivery = AUDIO_TAGS_FORBIDDEN
    name = (user_name or "").strip() or "your friend"
    values = {
        "user_name": name,
        "now": (now or "").strip() or now_string(),
        "memory": (memory_text or "").strip() or NO_MEMORY_TEXT,
        "audio_tags_rule": delivery,
        "tool_notes": (tool_notes or "").strip() or NO_TOOLS_TEXT,
        "language_rule": (
            LOCKED_LANGUAGE_RULE.replace("{language}", locked_language).replace("{user_name}", name)
            if locked_language
            else LANGUAGE_RULE.replace("{user_name}", name)
        ),
    }
    return _SLOT_RE.sub(lambda m: values[m.group(1)], persona.template)


__all__ = [
    "Persona",
    "PERSONAS_DIR",
    "DEFAULT_PERSONA",
    "TEMPLATE_SLOTS",
    "AUDIO_TAGS_ALLOWED",
    "AUDIO_TAGS_FORBIDDEN",
    "list_personas",
    "load_persona",
    "persona_path",
    "render",
    "now_string",
]
