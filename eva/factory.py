"""Build STT / LLM / TTS instances from a preset's config dicts.

Imports are lazy so that optional heavy dependencies (sherpa-onnx, kokoro-onnx) are only
loaded for the stack that needs them. The ears and the brain are local (Parakeet, Ollama); the
voice is ElevenLabs, wrapped in ``eva.failover.FailoverTTS`` with Kokoro behind it when the
preset names a fallback, so no internet or no credits means a local voice, not silence. On the
RunPod pod (deploy/runpod/) the brain is vLLM ("openai") and the voice is voice/server.py
("voice-server"), with the same Kokoro fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .config import BRAINS, OLLAMA_BASE_URL, VOICE_SERVER_URL, Preset, elevenlabs_key, key_file
from .interfaces import LLM, STT, TTS

EventHandler = Callable[[str, dict[str, Any]], None]


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
    if kind == "openai":
        # any OpenAI-compatible server; on the pod, vLLM next to the loop (no key)
        from .llm.openai_compat import OpenAICompatLLM

        model = cfg["model"]
        key = key_file(*cfg["key"]) if cfg.get("key") else "none"
        if not key:
            raise RuntimeError(f"no key for {model}: put it in {cfg['key'][0]} or {cfg['key'][1]}")
        return OpenAICompatLLM(
            name=f"openai/{model}",
            base_url=cfg["base_url"],
            api_key=key,
            model=model,
            extra_body=cfg.get("extra_body"),
            max_tokens=cfg.get("max_tokens", 400),
            temperature=cfg.get("temperature", 0.8),  # None: left out (OpenAI's reasoning models reject it)
            token_param=cfg.get("token_param", "max_tokens"),
        )
    raise ValueError(f"unknown llm kind {kind!r}")


def build_tts(cfg: dict[str, Any]) -> TTS:
    kind = cfg["kind"]
    if kind == "elevenlabs":
        from .tts.elevenlabs import ElevenLabsTTS

        key = elevenlabs_key()
        if not key:
            raise RuntimeError("no ElevenLabs key: put it in elevenlabs_key.txt or ELEVENLABS_API_KEY")
        return ElevenLabsTTS(api_key=key, voice_id=cfg["voice"], model_id=cfg["model_id"],
                             **({"level_dbfs": cfg["level_dbfs"]} if "level_dbfs" in cfg else {}))
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

        return VoiceServerTTS(cfg["engine"], cfg["voice"], cfg.get("base_url", VOICE_SERVER_URL),
                              **({"level_dbfs": cfg["level_dbfs"]} if "level_dbfs" in cfg else {}))
    raise ValueError(f"unknown tts kind {kind!r}")


@dataclass
class Stack:
    stt: STT
    llm: LLM
    tts: TTS
    voice: Any = None  # the primary (cloud) voice under any failover wrapper, for the credit meter


def build_stack(
    preset: Preset,
    *,
    brain: str | None = None,
    stt_overrides: dict[str, Any] | None = None,
    tts_overrides: dict[str, Any] | None = None,
    on_event: EventHandler | None = None,
) -> Stack:
    """Build a preset's three providers; the voice gets its local fallback when the preset names one."""
    stt_cfg = {**preset.stt, **(stt_overrides or {})}
    llm_cfg = {k: v for k, v in BRAINS[brain].items() if k not in ("label", "persona")} if brain else dict(preset.llm)
    for flag in ("tools", "tool_gate"):
        llm_cfg.pop(flag, None)
    tts_cfg = {**preset.tts, **(tts_overrides or {})}
    voice = build_tts(tts_cfg)
    tts: TTS = voice
    if preset.tts_fallback and preset.tts_fallback["kind"] != tts_cfg["kind"]:
        from .failover import FailoverTTS

        tts = FailoverTTS(voice, build_tts(preset.tts_fallback), on_event=on_event)
    return Stack(stt=build_stt(stt_cfg), llm=build_llm(llm_cfg), tts=tts, voice=voice)
