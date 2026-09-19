"""Build STT / LLM / TTS instances from a preset's config dicts.

Imports are lazy so that optional heavy dependencies (faster-whisper, sherpa-onnx,
kokoro-onnx) are only loaded for the preset that needs them.

Constructor signatures below are the contract that each provider module must honor.
"""
from __future__ import annotations

from typing import Any

from .config import (
    CEREBRAS_BASE_URL,
    EL_VOICES,
    OLLAMA_BASE_URL,
    Keys,
)
from .interfaces import LLM, STT, TTS


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
    if kind == "faster-whisper":
        from .stt.faster_whisper_local import FasterWhisperSTT

        return FasterWhisperSTT(
            model=cfg.get("model", "base.en"),
            device=cfg.get("device", "auto"),
            compute_type=cfg.get("compute_type", "int8"),
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
            # gpt-oss always reasons; low is the fastest setting.
            extra["reasoning_effort"] = reasoning if reasoning in ("low", "medium", "high") else "low"
        elif reasoning in (None, "none", "off", False):
            extra["disable_reasoning"] = True
        else:
            extra["reasoning_effort"] = reasoning
        return OpenAICompatLLM(
            name=f"cerebras/{model}",
            base_url=CEREBRAS_BASE_URL,
            api_key=keys.cerebras,
            model=model,
            extra_body=extra,
            max_tokens=cfg.get("max_tokens", 400),
            temperature=cfg.get("temperature", 0.8),
        )
    if kind == "ollama":
        model = cfg["model"]
        return OpenAICompatLLM(
            name=f"ollama/{model}",
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",
            model=model,
            extra_body={},
            max_tokens=cfg.get("max_tokens", 300),
            temperature=cfg.get("temperature", 0.8),
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
            mode=cfg.get("mode", "ws"),  # "ws" (stream-input websocket) or "http" (per-chunk stream)
            stability=cfg.get("stability"),
            similarity_boost=cfg.get("similarity_boost"),
            style=cfg.get("style"),
            speed=cfg.get("speed"),
            voices_by_lang=by_lang or None,
            first_chunk_model=cfg.get("first_chunk_model"),
            continuity=cfg.get("continuity", True),
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
