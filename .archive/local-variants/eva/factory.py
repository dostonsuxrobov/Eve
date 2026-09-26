"""Build STT / LLM / TTS instances from a preset's config dicts.

Imports are lazy so that optional heavy dependencies (sherpa-onnx, kokoro-onnx) are
only loaded for the stack that needs them. Everything is local: Parakeet on the CPU,
brains in Ollama, Kokoro in-process or an expressive voice behind voice/server.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import BRAINS, CEREBRAS_BASE_URL, OLLAMA_BASE_URL, VOICE_SERVER_URL, Preset, cerebras_key
from .interfaces import LLM, STT, TTS


def build_stt(cfg: dict[str, Any]) -> STT:
    kind = cfg["kind"]
    if kind == "parakeet":
        from .stt.sherpa_parakeet import SherpaParakeetSTT

        return SherpaParakeetSTT(
            model_dir=cfg.get("model_dir"),
            num_threads=cfg.get("num_threads", 4),
            min_audio_s=cfg.get("min_audio_s", 1.5),
        )
    raise ValueError(f"unknown stt kind {kind!r}")


def build_llm(cfg: dict[str, Any]) -> LLM:
    kind = cfg["kind"]
    if kind == "ollama":
        # Native /api/chat: the only endpoint where `think` reaches hybrid Qwen3 models
        # (the OpenAI-compatible one ignores it and reasons for seconds per turn).
        from .llm.ollama_native import OllamaNativeLLM

        model = cfg["model"]
        return OllamaNativeLLM(
            name=f"ollama/{model}",
            base_url=cfg.get("base_url", OLLAMA_BASE_URL),
            model=model,
            think=cfg.get("think", False),
            max_tokens=cfg.get("max_tokens", 300),
            temperature=cfg.get("temperature", 0.8),
            options=cfg.get("options"),
        )
    if kind == "cerebras":
        # the cloud era's client (eva/llm/openai_compat.py), for the owner's comparison
        from .llm.openai_compat import OpenAICompatLLM

        key = cerebras_key()
        if not key:
            raise RuntimeError("no Cerebras key: put it in cerebras_api_key.txt or CEREBRAS_API_KEY")
        model = cfg["model"]
        reasoning = cfg.get("reasoning")
        # qwen with reasoning off truncated 25-40 % of very short replies mid-word; "low" didn't (archived eval)
        extra: dict[str, Any] = {"disable_reasoning": True} if reasoning in (None, "none", "off", False) else {"reasoning_effort": reasoning}
        return OpenAICompatLLM(
            name=f"cerebras/{model}", base_url=cfg.get("base_url", CEREBRAS_BASE_URL), api_key=key, model=model,
            extra_body=extra, max_tokens=cfg.get("max_tokens", 800), temperature=cfg.get("temperature", 0.8),
        )
    raise ValueError(f"unknown llm kind {kind!r}")


def build_tts(cfg: dict[str, Any]) -> TTS:
    kind = cfg["kind"]
    if kind == "kokoro":
        from .tts.kokoro_local import KokoroTTS

        return KokoroTTS(
            voice=cfg.get("voice", "af_heart"),
            speed=cfg.get("speed", 1.0),
            model=cfg.get("model", "fp32"),
            intra_threads=cfg.get("intra_threads"),
            lang=cfg.get("lang", "en-us"),
        )
    if kind == "voice-server":
        from .tts.voice_server import VoiceServerTTS

        return VoiceServerTTS(engine=cfg["engine"], voice=cfg["voice"], base_url=cfg.get("base_url", VOICE_SERVER_URL))
    raise ValueError(f"unknown tts kind {kind!r}")


@dataclass
class Stack:
    stt: STT
    llm: LLM
    tts: TTS


def build_stack(
    preset: Preset,
    *,
    brain: str | None = None,
    stt_overrides: dict[str, Any] | None = None,
    tts_overrides: dict[str, Any] | None = None,
) -> Stack:
    """Build a preset's three providers. ``brain`` picks an entry of ``config.BRAINS``
    instead of the preset's LLM; the overrides are merged into the provider configs."""
    stt_cfg = {**preset.stt, **(stt_overrides or {})}
    llm_cfg = {k: v for k, v in BRAINS[brain].items() if k not in ("label", "persona")} if brain else dict(preset.llm)
    llm_cfg.pop("tools", None)
    llm_cfg.pop("tool_gate", None)
    tts_cfg = {**preset.tts, **(tts_overrides or {})}
    return Stack(stt=build_stt(stt_cfg), llm=build_llm(llm_cfg), tts=build_tts(tts_cfg))
