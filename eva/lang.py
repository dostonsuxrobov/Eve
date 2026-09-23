"""Language assets and the ``--lang`` modes.

Everything that changes with the language lives in ``eva/assets/lang/<code>.toml``
(voice, STT hint, fillers, tool hints, backchannels) and the persona prompts in
``eva/assets/personas/<code>/``. The code is script-agnostic; this module only decides
which assets are active for a session.

Modes (:func:`plan`):

``auto`` (default)
    Every language is active. The STT is boxed into them (``LangPlan.stt_box``: the
    primary as ``language_code``, the rest as ``secondary_languages``) and reports the
    language it heard; the pipeline never switches on one outside the box. The voice is picked per sentence
    by script (``voices_by_lang``), fillers follow the language of the user's last
    turn, the persona is the English prompt with the bilingual language rule.
``en`` / ``ru`` (locked)
    One language. The STT gets a language hint (Scribe is more reliable on one-word
    answers with it), the voice is pinned, fillers are that language's, the persona is
    ``assets/personas/<code>/<name>.md`` when it exists (a prompt written in the
    language beats an English prompt with a "speak natural Russian" rule), else the
    English one with a "answer only in <language>" rule.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LANG_DIR = ASSETS_DIR / "lang"
DEFAULT_LANG = "en"
AUTO = "auto"


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    stt_language_code: str | None
    voice: str
    fillers: list[str] = field(default_factory=list)
    tool_hints: list[str] = field(default_factory=list)
    backchannels: list[str] = field(default_factory=list)


def load_languages(directory: Path = LANG_DIR) -> dict[str, Language]:
    """Every ``<code>.toml`` under ``directory``, keyed by code; the default first."""
    langs: dict[str, Language] = {}
    for path in sorted(directory.glob("*.toml")):
        with path.open("rb") as f:
            raw = tomllib.load(f)
        code = str(raw.get("code") or path.stem).lower()
        langs[code] = Language(
            code=code,
            name=str(raw.get("name") or code),
            stt_language_code=raw.get("stt_language_code") or None,
            voice=str(raw.get("voice") or ""),
            fillers=[str(x) for x in raw.get("fillers", [])],
            tool_hints=[str(x) for x in raw.get("tool_hints", [])],
            backchannels=[str(x) for x in raw.get("backchannels", [])],
        )
    if DEFAULT_LANG in langs:  # the default language leads: it is the fallback for everything
        langs = {DEFAULT_LANG: langs[DEFAULT_LANG], **{k: v for k, v in langs.items() if k != DEFAULT_LANG}}
    return langs


@dataclass(frozen=True)
class LangPlan:
    """What a session uses, resolved from a mode."""

    mode: str  # "auto" or a language code
    primary: Language  # greeting language, initial fillers
    active: list[Language]  # languages whose assets are loaded
    stt_language_code: str | None  # sent to the STT only when locked
    voice: str  # the TTS default voice
    voices_by_lang: dict[str, str]  # per-sentence switching (auto) or empty (locked)
    persona_lang: str  # which personas/<code>/ directory to prefer

    @property
    def locked(self) -> bool:
        return self.mode != AUTO

    @property
    def codes(self) -> list[str]:
        """The session's language codes, primary first (``["en", "ru"]``)."""
        return [lang.code for lang in self.active]

    @property
    def stt_box(self) -> tuple[str | None, list[str]]:
        """``(language_code, secondary_languages)`` for the STT: the hint when locked,
        otherwise every active language (primary first) so the recogniser does not
        wander into the ~90 others (Scribe heard Russian / English as Dutch, ``ja``, ``mk``)."""
        if self.locked:
            return self.stt_language_code, []
        codes = [lang.stt_language_code for lang in self.active if lang.stt_language_code]
        return (codes[0], codes[1:]) if len(codes) > 1 else (None, [])

    def by_lang(self, attr: str) -> dict[str, list[str]]:
        """``{"en": [...], "ru": [...]}`` for ``fillers`` / ``tool_hints`` / ``backchannels``."""
        return {lang.code: list(getattr(lang, attr)) for lang in self.active if getattr(lang, attr)}


def plan(mode: str, languages: dict[str, Language] | None = None) -> LangPlan:
    """Resolve ``mode`` (``auto`` or a code) into a :class:`LangPlan`."""
    langs = languages if languages is not None else load_languages()
    if not langs:
        raise RuntimeError(f"no language files in {LANG_DIR}")
    mode = (mode or AUTO).lower()
    if mode == AUTO:
        primary = langs.get(DEFAULT_LANG) or next(iter(langs.values()))
        return LangPlan(
            mode=AUTO,
            primary=primary,
            active=list(langs.values()),
            stt_language_code=None,
            voice=primary.voice,
            voices_by_lang={lang.code: lang.voice for lang in langs.values() if lang.voice},
            persona_lang=DEFAULT_LANG,
        )
    if mode not in langs:
        raise ValueError(f"unknown language {mode!r}; available: auto, {', '.join(langs)}")
    lang = langs[mode]
    return LangPlan(
        mode=mode,
        primary=lang,
        active=[lang],
        stt_language_code=lang.stt_language_code,
        voice=lang.voice,
        voices_by_lang={},
        persona_lang=mode,
    )


def modes(languages: dict[str, Language] | None = None) -> list[str]:
    """``["auto", "en", "ru"]`` for the CLI."""
    return [AUTO, *(languages if languages is not None else load_languages())]


__all__ = ["ASSETS_DIR", "LANG_DIR", "DEFAULT_LANG", "AUTO", "Language", "LangPlan", "load_languages", "plan", "modes"]
