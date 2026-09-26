"""Assemble everything a conversation needs from a preset and a language mode.

``run.py`` (the CLI) calls :func:`build_session`, so the providers, the language plan,
the persona prompt, the memory and the filler lists are composed in exactly one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from .config import MEMORY_FILE, Preset
from .factory import Stack, build_stack
from .interfaces import Tool
from .lang import DEFAULT_MODE, LangPlan, plan as lang_plan
from .memory import Memory
from .personas import Persona, load_persona, render
from .guard import SpeechGuard
from .toolgate import ToolGate, home_city
from .tools import get_tools, tool_notes

# A small brain behind the tool gate: no tool list and no "MUST call" in the prompt (with no
# tool offered for a line, MiniCPM5 1B wrote the call out as markup and it was spoken,
# 2026-09-26). A tool appears only when the user's words ask for it.
GATED_TOOL_NOTES = (
    "When they ask for the weather, the time, a timer, a reminder or a note, a tool is offered to you "
    "for that moment: call it, then say what it returned in a sentence or two. At any other moment you "
    "can't look anything up or do anything outside this conversation, so don't offer to."
)

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
    tool_filter: Callable[[str, list[Tool]], list[Tool]] | None = None
    speech_guard: SpeechGuard | None = None
    tool_args: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None
    tool_router: Callable[[str], list[tuple[str, dict[str, Any]]]] | None = None

    def agent_kwargs(self) -> dict[str, Any]:
        """The scaffolding every VoiceAgent of this session gets (run.py, the phone client, the benches)."""
        return {"tool_filter": self.tool_filter, "speech_guard": self.speech_guard, "tool_args": self.tool_args,
                "tool_router": self.tool_router}

    @property
    def stt(self) -> Any:
        return self.stack.stt

    @property
    def llm(self) -> Any:
        return self.stack.llm

    @property
    def tts(self) -> Any:
        return self.stack.tts

    @property
    def ollama_models(self) -> set[str]:
        """The Ollama models this variant uses: the brain, and Orpheus when it is the voice."""
        models = {self.preset.llm["model"]} if self.preset.llm.get("kind") == "ollama" else set()
        extra = getattr(self.stack.tts, "ollama_model", None)
        if extra:
            models.add(extra)
        return models

    def greeting_event(self, user_name: str) -> dict[str, Any]:
        return {"type": "session_start", "user_name": user_name, "lang": self.plan.primary.code, "language": self.plan.primary.name}


def build_session(
    preset: Preset,
    *,
    lang: str = DEFAULT_MODE,
    brain: str | None = None,
    persona: str | None = None,
    user_name: str = "",
    voice: str | None = None,
    mute_fillers: bool = False,
    memory_path: Any = MEMORY_FILE,
    on_event: EventHandler | None = None,
) -> Session:
    """Build providers, persona prompt and memory for one session.

    The language plan (``eva.lang``) decides which persona directory is preferred; the
    persona file may override the language's fillers / hints / backchannels. A brain
    whose Ollama template has no tools (``"tools": False``) gets none and a prompt that
    says so. Stale name facts are dropped from memory when ``user_name`` is known (see
    ``Memory.drop_name_facts``).
    """
    plan = lang_plan(lang)
    tts_overrides: dict[str, Any] = {"voice": voice} if voice else {}
    stack = build_stack(preset, brain=brain, tts_overrides=tts_overrides)

    persona_obj = load_persona(persona or preset.persona, lang=plan.persona_lang)
    memory = Memory(memory_path)
    memory.load()
    dropped = memory.drop_name_facts(user_name) if user_name else []
    if dropped:
        memory.save()
    tools = get_tools() if preset.llm.get("tools", True) else []
    # small brains: the tool gate, the speech guard, the short tool note (eva/toolgate.py, eva/guard.py)
    gated = bool(preset.llm.get("tool_gate"))
    gate = ToolGate(home_city=home_city(memory.facts)) if gated else None
    system_prompt = render(
        persona_obj,
        supports_audio_tags=stack.tts.supports_audio_tags,
        memory_text=memory.as_prompt_text(subject=user_name or "They"),
        now=datetime.now().strftime("%A %d %B %Y, %H:%M"),
        user_name=user_name,
        tool_notes=GATED_TOOL_NOTES if (gated and tools) else tool_notes(tools),
        delivery_cues=bool(getattr(stack.tts, "supports_cues", False)),
        locked_language=plan.primary.name if plan.locked else None,
        languages=[lang.name for lang in plan.active],
        sounds=getattr(stack.tts, "sound_tags", None),
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
        dropped_facts=dropped, tool_filter=gate.filter if gate else None,
        speech_guard=SpeechGuard(user_name=user_name, tool_names=tuple(t.name for t in get_tools()),
                                 known=memory.as_prompt_text().lower()) if gated else None,
        tool_args=gate.fix_args if gate else None,
        tool_router=gate.route if gate else None,
    )


__all__ = ["Session", "build_session"]
