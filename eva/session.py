"""Assemble everything a conversation needs from a preset and a language mode.

``run.py`` (the CLI) and ``bench/e2e_sim.py`` (the simulator) both call
:func:`build_session`, so the providers, the language plan, the persona prompt, the
memory and the filler lists are composed in exactly one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from .config import MEMORY_FILE, Keys, Preset
from .factory import Stack, build_stack
from .interfaces import Tool
from .lang import DEFAULT_MODE, LangPlan, plan as lang_plan
from .memory import Memory
from .personas import Persona, load_persona, render
from .tools import get_tools, tool_notes

EventHandler = Callable[[str, dict[str, Any]], None]


@dataclass
class Session:
    preset: Preset
    plan: LangPlan
    stack: Stack
    persona: Persona
    memory: Memory
    tools: list[Tool]
    system_prompt: str
    fillers: dict[str, list[str]] = field(default_factory=dict)
    tool_hints: dict[str, list[str]] = field(default_factory=dict)
    backchannels: dict[str, list[str]] = field(default_factory=dict)
    dropped_facts: list[str] = field(default_factory=list)

    @property
    def stt(self) -> Any:
        return self.stack.stt

    @property
    def llm(self) -> Any:
        return self.stack.llm

    @property
    def tts(self) -> Any:
        return self.stack.tts

    def greeting_event(self, user_name: str) -> dict[str, Any]:
        return {"type": "session_start", "user_name": user_name, "lang": self.plan.primary.code, "language": self.plan.primary.name}


def stt_language_settings(plan: LangPlan, stt_kind: str) -> dict[str, Any]:
    """What the STT is told about the session's languages.

    Locked (the English default): the one language as a hint, nothing else. Auto: the
    realtime socket is boxed into every active language (the batch endpoint's
    multi-language support is unverified, so only realtime is boxed).
    """
    out: dict[str, Any] = {}
    if plan.stt_language_code and stt_kind.startswith("elevenlabs"):
        out["language"] = plan.stt_language_code
    if stt_kind == "elevenlabs-realtime":
        primary, secondary = plan.stt_box
        if primary and secondary:
            out.update(language=primary, secondary_languages=secondary)
        out["language_detection"] = True
    return out


def build_session(
    preset: Preset,
    keys: Keys,
    *,
    lang: str = DEFAULT_MODE,
    brain: str | None = None,
    persona: str | None = None,
    user_name: str = "",
    voice: str | None = None,
    fallbacks: bool = True,
    mute_fillers: bool = False,
    memory_path: Any = MEMORY_FILE,
    on_event: EventHandler | None = None,
) -> Session:
    """Build providers, persona prompt and memory for one session.

    The language plan (``eva.lang``) decides the STT language hint, the voice(s) and
    which persona directory is preferred; the persona file may override the language's
    fillers / hints / backchannels. Stale name facts are dropped from memory when
    ``user_name`` is known (see ``Memory.drop_name_facts``).
    """
    plan = lang_plan(lang)
    stt_overrides = stt_language_settings(plan, preset.stt["kind"])
    tts_overrides: dict[str, Any] = {}
    if preset.tts["kind"] == "elevenlabs":
        tts_overrides = {"voice": plan.voice, "voices_by_lang": plan.voices_by_lang}
    if voice:
        tts_overrides["voice"] = voice
    stack = build_stack(
        preset, keys, brain=brain, stt_overrides=stt_overrides, tts_overrides=tts_overrides,
        fallbacks=fallbacks, on_event=on_event,
    )

    persona_obj = load_persona(persona or preset.persona, lang=plan.persona_lang)
    memory = Memory(memory_path)
    memory.load()
    dropped = memory.drop_name_facts(user_name) if user_name else []
    if dropped:
        memory.save()
    tools = get_tools()
    system_prompt = render(
        persona_obj,
        supports_audio_tags=stack.tts.supports_audio_tags,
        memory_text=memory.as_prompt_text(),
        now=datetime.now().strftime("%A %d %B %Y, %H:%M"),
        user_name=user_name,
        tool_notes=tool_notes(tools),
        delivery_cues=bool(getattr(stack.tts, "supports_cues", False)),
        locked_language=plan.primary.name if plan.locked else None,
        languages=[lang.name for lang in plan.active],
    )
    fillers = plan.by_lang("fillers")
    hints = plan.by_lang("tool_hints")
    backchannels = plan.by_lang("backchannels")
    if persona_obj.fillers:
        fillers[persona_obj.lang] = list(persona_obj.fillers)
    if persona_obj.tool_hints:
        hints[persona_obj.lang] = list(persona_obj.tool_hints)
    if persona_obj.backchannels:
        backchannels[persona_obj.lang] = list(persona_obj.backchannels)
    if mute_fillers:
        fillers = {}
    if not preset.settings.backchannels:
        backchannels = {}
    return Session(
        preset=preset, plan=plan, stack=stack, persona=persona_obj, memory=memory, tools=tools,
        system_prompt=system_prompt, fillers=fillers, tool_hints=hints, backchannels=backchannels,
        dropped_facts=dropped,
    )


__all__ = ["Session", "build_session"]
