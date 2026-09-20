"""Build STT / LLM / TTS instances from a preset's config dicts.

Imports are lazy so that optional heavy dependencies (sherpa-onnx, kokoro-onnx) are
only loaded for the stack that needs them.

Constructor signatures below are the contract that each provider module must honor.
:func:`build_stack` assembles a whole preset, wrapping each provider in its
``eva.failover`` counterpart when the preset names a fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .config import (
    BRAINS,
    CEREBRAS_BASE_URL,
    EL_VOICES,
    OLLAMA_BASE_URL,
    OPENAI_BASE_URL,
    Keys,
    Preset,
)
from .interfaces import LLM, STT, TTS

EventHandler = Callable[[str, dict[str, Any]], None]


def build_stt(cfg: dict[str, Any], keys: Keys) -> STT:
    kind = cfg["kind"]
    if kind == "elevenlabs":
        from .stt.elevenlabs_scribe import ElevenLabsScribeSTT

        assert keys.elevenlabs, "ElevenLabs key missing"
        return ElevenLabsScribeSTT(
            api_key=keys.elevenlabs,
            model_id=cfg.get("model_id", "scribe_v1"),
            language=cfg.get("language"),
        )
    if kind == "elevenlabs-realtime":
        from .stt.elevenlabs_realtime import ElevenLabsRealtimeSTT

        assert keys.elevenlabs, "ElevenLabs key missing"
        return ElevenLabsRealtimeSTT(
            api_key=keys.elevenlabs,
            model_id=cfg.get("model_id", "scribe_v2_realtime"),
            language=cfg.get("language"),
        )
    if kind == "parakeet":
        from .stt.sherpa_parakeet import SherpaParakeetSTT

        return SherpaParakeetSTT(
            model_dir=cfg.get("model_dir"),
            num_threads=cfg.get("num_threads", 4),
            min_audio_s=cfg.get("min_audio_s", 1.5),
        )
    raise ValueError(f"unknown stt kind {kind!r}")


def build_llm(cfg: dict[str, Any], keys: Keys) -> LLM:
    from .llm.openai_compat import OpenAICompatLLM

    kind = cfg["kind"]
    if kind == "cerebras":
        assert keys.cerebras, "Cerebras key missing"
        model = cfg["model"]
        extra: dict[str, Any] = {}
        reasoning = cfg.get("reasoning")
        if model.startswith("gpt-oss"):
            # gpt-oss always reasons; low is the fastest setting (not a brain any more, kept for --brain overrides via config)
            extra["reasoning_effort"] = reasoning if reasoning in ("low", "medium", "high") else "low"
        elif reasoning in (None, "none", "off", False):
            extra["disable_reasoning"] = True
        else:
            extra["reasoning_effort"] = reasoning
        return OpenAICompatLLM(
            name=f"cerebras/{model}",
            base_url=cfg.get("base_url", CEREBRAS_BASE_URL),
            api_key=keys.cerebras,
            model=model,
            extra_body=extra,
            max_tokens=cfg.get("max_tokens", 400),
            temperature=cfg.get("temperature", 0.8),
        )
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
    if kind == "openai":
        assert keys.openai, "OpenAI key missing (openAI_api.txt or OPENAI_API_KEY)"
        model = cfg["model"]
        reasoning_model = model.startswith(("gpt-5", "o1", "o3", "o4")) and "chat-latest" not in model
        extra: dict[str, Any] = {}
        if reasoning_model:
            # "none" (5.1+) / "minimal" (5.0): the least thinking; a voice turn cannot wait for more
            extra["reasoning_effort"] = cfg.get("reasoning", "minimal")
        return OpenAICompatLLM(
            name=f"openai/{model}",
            base_url=cfg.get("base_url", OPENAI_BASE_URL),
            api_key=keys.openai,
            model=model,
            extra_body=extra,
            max_tokens=cfg.get("max_tokens", 400),
            temperature=None if reasoning_model else cfg.get("temperature", 0.8),
            token_param="max_completion_tokens",
        )
    if kind == "openai-compat":
        return OpenAICompatLLM(
            name=cfg.get("name", cfg["model"]),
            base_url=cfg["base_url"],
            api_key=cfg.get("api_key", "none"),
            model=cfg["model"],
            extra_body=cfg.get("extra_body", {}),
            max_tokens=cfg.get("max_tokens", 400),
            temperature=cfg.get("temperature", 0.8),
        )
    raise ValueError(f"unknown llm kind {kind!r}")


def build_tts(cfg: dict[str, Any], keys: Keys) -> TTS:
    kind = cfg["kind"]
    if kind == "elevenlabs":
        from .tts.elevenlabs import ElevenLabsTTS

        assert keys.elevenlabs, "ElevenLabs key missing"
        voice = cfg.get("voice", "sarah")
        voice_id = EL_VOICES.get(voice, voice)  # allow a raw voice id
        by_lang = {k: EL_VOICES.get(v, v) for k, v in (cfg.get("voices_by_lang") or {}).items()}
        return ElevenLabsTTS(
            api_key=keys.elevenlabs,
            voice_id=voice_id,
            model_id=cfg.get("model_id", "eleven_flash_v2_5"),
            stability=cfg.get("stability"),
            similarity_boost=cfg.get("similarity_boost"),
            style=cfg.get("style"),
            speed=cfg.get("speed"),
            voices_by_lang=by_lang or None,
            first_chunk_model=cfg.get("first_chunk_model"),
            continuity=cfg.get("continuity", True),
            **({"level_dbfs": cfg["level_dbfs"]} if "level_dbfs" in cfg else {}),  # None turns leveling off
        )
    if kind == "kokoro":
        from .tts.kokoro_local import KokoroTTS

        return KokoroTTS(
            voice=cfg.get("voice", "af_heart"),
            speed=cfg.get("speed", 1.0),
            model=cfg.get("model", "fp32"),
            intra_threads=cfg.get("intra_threads"),
            lang=cfg.get("lang", "en-us"),
        )
    raise ValueError(f"unknown tts kind {kind!r}")


@dataclass
class Stack:
    stt: STT
    llm: LLM
    tts: TTS


def build_stack(
    preset: Preset,
    keys: Keys,
    *,
    brain: str | None = None,
    stt_overrides: dict[str, Any] | None = None,
    tts_overrides: dict[str, Any] | None = None,
    fallbacks: bool = True,
    on_event: EventHandler | None = None,
) -> Stack:
    """Build a preset's three providers.

    ``brain`` picks an entry of ``config.BRAINS`` instead of the preset's LLM;
    ``stt_overrides`` / ``tts_overrides`` are merged into the provider configs (the
    language layer uses them for the STT hint and the voices). With ``fallbacks`` each
    provider that has a ``*_fallback`` in the preset is wrapped in its ``eva.failover``
    counterpart, which reports through ``on_event``.
    """
    stt_cfg = {**preset.stt, **(stt_overrides or {})}
    llm_cfg = dict(BRAINS[brain]) if brain else dict(preset.llm)
    tts_cfg = {**preset.tts, **(tts_overrides or {})}
    stt = build_stt(stt_cfg, keys)
    llm = build_llm(llm_cfg, keys)
    tts = build_tts(tts_cfg, keys)
    if fallbacks:
        from .failover import FailoverLLM, FailoverSTT, FailoverTTS

        if preset.stt_fallback and preset.stt_fallback["kind"] != stt_cfg["kind"]:
            stt = FailoverSTT(stt, build_stt(preset.stt_fallback, keys), on_event=on_event)
        if preset.llm_fallback and preset.llm_fallback != llm_cfg:
            llm = FailoverLLM(llm, build_llm(preset.llm_fallback, keys), on_event=on_event)
        if preset.tts_fallback and preset.tts_fallback["kind"] != tts_cfg["kind"]:
            tts = FailoverTTS(tts, build_tts(preset.tts_fallback, keys), on_event=on_event)
    return Stack(stt=stt, llm=llm, tts=tts)
