"""Speech-to-speech backends: one model hears the caller and speaks back (OpenAI Realtime, Gemini
Live), with the same tools and persona as Eva's own loop.

Both clients speak one small event language to ``eva.s2s.driver.Call``:

    {"type": "audio", "pcm": bytes}             her voice, int16 mono at ``client.out_rate``
    {"type": "text_out", "delta": str}          the transcript of what she says, as it streams
    {"type": "text_in", "text": str}            what the model heard (its own transcription)
    {"type": "speech_started"}                  the caller started talking (barge-in)
    {"type": "interrupted"}                     the model stopped itself because the caller cut in
    {"type": "tool_calls", "calls": [(id, name, args)]}
    {"type": "turn_done", "usage": dict, "cost": float, "status": str}
    {"type": "error", "message": str}

Prices are per 1M tokens, from the providers' pricing pages on 2026-09-27 (OpenAI: audio in 1
token per 100 ms, out 1 per 50 ms; Gemini: ~25 tokens a second both ways).
"""
from __future__ import annotations

from typing import Any

PRICES: dict[str, dict[str, float]] = {
    # audio_in, cached_audio_in, audio_out, text_in, cached_text_in, text_out
    "gpt-realtime-2.1": {"audio_in": 32, "audio_in_cached": 0.40, "audio_out": 64, "text_in": 4, "text_in_cached": 0.40, "text_out": 24},
    "gpt-realtime-2.1-mini": {"audio_in": 10, "audio_in_cached": 0.30, "audio_out": 20, "text_in": 0.60, "text_in_cached": 0.06, "text_out": 2.40},
    "gpt-realtime-2": {"audio_in": 32, "audio_in_cached": 0.40, "audio_out": 64, "text_in": 4, "text_in_cached": 0.40, "text_out": 24},
    "gpt-realtime": {"audio_in": 32, "audio_in_cached": 0.40, "audio_out": 64, "text_in": 4, "text_in_cached": 0.40, "text_out": 16},
    "gpt-realtime-mini": {"audio_in": 10, "audio_in_cached": 0.30, "audio_out": 20, "text_in": 0.60, "text_in_cached": 0.06, "text_out": 2.40},
    "gemini-3.8-live": {"audio_in": 3.0, "audio_out": 12.0, "text_in": 0.75, "text_out": 4.50},
    "gemini-3.8-live-extended-thinking": {"audio_in": 3.0, "audio_out": 12.0, "text_in": 0.75, "text_out": 4.50},
    "gemini-3.1-flash-live-preview": {"audio_in": 3.0, "audio_out": 12.0, "text_in": 0.75, "text_out": 4.50},
    "gemini-2.5-flash-native-audio-latest": {"audio_in": 3.0, "audio_out": 12.0, "text_in": 0.50, "text_out": 2.0},
    "gpt-realtime-1.5": {"audio_in": 32, "audio_in_cached": 0.40, "audio_out": 64, "text_in": 4, "text_in_cached": 0.40, "text_out": 16},
}

MODELS = {
    "openai": ["gpt-realtime-2.1", "gpt-realtime-2.1-mini", "gpt-realtime-2", "gpt-realtime-1.5", "gpt-realtime", "gpt-realtime-mini"],
    "openai-live": ["gpt-live-1", "gpt-live-1+gpt-5.6-sol"],  # full duplex; "+backend" picks the Responses model
    "gemini": ["gemini-3.8-live", "gemini-3.8-live-extended-thinking", "gemini-3.1-flash-live-preview",
               "gemini-2.5-flash-native-audio-latest"],
}


def make_client(provider: str, model: str, instructions: str, tools: list[Any], **kw: Any) -> Any:
    if provider == "openai":
        from .openai_rt import OpenAIRealtime

        return OpenAIRealtime(model, instructions, tools, **kw)
    if provider == "openai-live":
        from .openai_live import OpenAILive

        live, _, backend = model.partition("+")
        return OpenAILive(live, instructions, tools, **({"backend": backend} if backend else {}), **kw)
    if provider == "gemini":
        from .gemini_live import GeminiLive

        return GeminiLive(model, instructions, tools, **kw)
    raise ValueError(f"unknown provider {provider!r}")


def live_instructions(job: str, user_name: str = "Doston") -> tuple[str, str]:
    """GPT-Live's two prompts for a job, in OpenAI's recommended structure: the voice model's (role,
    backchannel, interruption and delegation policies) and the backend's (procedures, rules, how to
    return a result). Files: eva/assets/personas/en/<persona>_live.md and <persona>_backend.md."""
    from ..jobs import job_persona
    from ..personas import load_persona, now_string, render

    out = []
    for part in ("live", "backend"):
        out.append(render(load_persona(f"{job_persona(job)}_{part}"), supports_audio_tags=False, memory_text="",
                          now=now_string(), user_name=user_name, tool_notes=""))
    return out[0], out[1]


def job_client(provider: str, model: str, job: str = "dispatch", user_name: str = "Doston", **kw: Any) -> tuple[Any, list[Any]]:
    """A speech-to-speech client with the job's persona and tools (GPT-Live gets its two prompts)."""
    prompt, tools = job_instructions(job, user_name)
    if provider == "openai-live":
        prompt, backend = live_instructions(job, user_name)
        kw["backend_instructions"] = backend
    return make_client(provider, model, prompt, tools, **kw), tools


def job_instructions(job: str, user_name: str = "Doston") -> tuple[str, list[Any]]:
    """The job's persona rendered for a voice that takes no bracketed tags, and its tools."""
    from ..jobs import job_persona, job_tools
    from ..lang import plan
    from ..personas import load_persona, now_string, render
    from ..tools import tool_notes

    tools = job_tools(job)
    p = plan("en")
    prompt = render(load_persona(job_persona(job)), supports_audio_tags=False, memory_text="", now=now_string(),
                    user_name=user_name, tool_notes=tool_notes(tools), locked_language=p.primary.name,
                    languages=[x.name for x in p.active])
    return prompt, tools
