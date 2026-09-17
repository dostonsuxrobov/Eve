"""Keys, paths and the preset table ("paths") that define each agent variant."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
SAMPLES_DIR = ROOT / "samples"
MEMORY_FILE = ROOT / "memory.json"

# Cerebras sits behind Cloudflare and returns 403 (error 1010) for the default
# python-urllib / python-httpx user agents. Always send this.
USER_AGENT = "eva-voice-agent/0.1"

# On this Windows box `localhost` resolves to ::1 first and costs ~2 s per request
# before falling back. Always use the IPv4 literal.
OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"

# ElevenLabs premade voice IDs (the key lacks voices_read, so we hardcode).
EL_VOICES: dict[str, str] = {
    "rachel": "21m00Tcm4TlvDq8ikWAM",
    "sarah": "EXAVITQu4vr4xnSDxMaL",
    "laura": "FGY2WhTYpPnrIDTdsKH5",
    "charlotte": "XB0fDUnXU5powFXDhCwa",
    "alice": "Xb7hH8MSUJpSbSDYk0k2",
    "matilda": "XrExE9yKIg1WjnnlVkGX",
    "jessica": "cgSgspJ2msm6clMXDiKS",
    "lily": "pFZP5JQG7iQjIQuC4Bku",
}


def _read_key(filename: str, env: str) -> str | None:
    v = os.environ.get(env)
    if v:
        return v.strip()
    p = ROOT / filename
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return None


@dataclass
class Keys:
    cerebras: str | None
    elevenlabs: str | None


def load_keys() -> Keys:
    return Keys(
        cerebras=_read_key("cerebras_api_key.txt", "CEREBRAS_API_KEY"),
        elevenlabs=_read_key("elevenlabs_key.txt", "ELEVENLABS_API_KEY"),
    )


@dataclass
class PipelineSettings:
    """Turn-taking and smoothness knobs. Defaults tuned for a natural feel."""

    vad_threshold: float = 0.5
    endpoint_silence_ms: int = 550  # silence after speech that ends the user's turn
    min_speech_ms: int = 200  # ignore blips shorter than this
    prespeech_buffer_ms: int = 300  # audio kept before VAD fires so we don't clip onsets
    max_utterance_s: float = 30.0
    barge_in: bool = True
    barge_in_min_speech_ms: int = 300  # how long the user must talk to interrupt
    filler_after_ms: int = 900  # play a filler if no agent audio by then (0 = off)
    first_chunk_min_chars: int = 14  # send first TTS chunk early (after a comma) for low TTFA
    tts_parallelism: int = 2  # sentences synthesized ahead of playback
    input_device: int | None = None
    output_device: int | None = None
    echo_guard: bool = True  # raise VAD threshold while speaking when not using headphones


@dataclass
class Preset:
    name: str
    description: str
    stt: dict[str, Any]
    llm: dict[str, Any]
    tts: dict[str, Any]
    persona: str = "eva"
    settings: PipelineSettings = field(default_factory=PipelineSettings)


PRESETS: dict[str, Preset] = {
    "cloud-fast": Preset(
        name="cloud-fast",
        description="ElevenLabs Scribe STT + Cerebras qwen-3.8-27b (no reasoning) + ElevenLabs Flash. Lowest cloud latency.",
        stt={"kind": "elevenlabs", "model_id": "scribe_v1"},
        llm={"kind": "cerebras", "model": "qwen-3.8-27b", "reasoning": "none"},
        tts={"kind": "elevenlabs", "voice": "sarah", "model_id": "eleven_flash_v2_5", "mode": "ws"},
    ),
    "cloud-smart": Preset(
        name="cloud-smart",
        description="Same audio stack, Cerebras gpt-oss-120b with low reasoning. Smarter, a bit slower.",
        stt={"kind": "elevenlabs", "model_id": "scribe_v1"},
        llm={"kind": "cerebras", "model": "gpt-oss-120b", "reasoning": "low"},
        tts={"kind": "elevenlabs", "voice": "sarah", "model_id": "eleven_flash_v2_5", "mode": "ws"},
    ),
    "expressive": Preset(
        name="expressive",
        description="Cerebras qwen + ElevenLabs v3 with audio tags ([laughs], [sighs]). Most emotional, highest TTS latency.",
        stt={"kind": "elevenlabs", "model_id": "scribe_v1"},
        llm={"kind": "cerebras", "model": "qwen-3.8-27b", "reasoning": "none"},
        tts={"kind": "elevenlabs", "voice": "sarah", "model_id": "eleven_v3", "mode": "http"},
    ),
    "local-brain": Preset(
        name="local-brain",
        description="Cloud audio, local Ollama qwen3:4b brain. Tests how far a small local model gets.",
        stt={"kind": "elevenlabs", "model_id": "scribe_v1"},
        llm={"kind": "ollama", "model": "qwen3:4b-instruct-2507-q4_K_M"},
        tts={"kind": "elevenlabs", "voice": "sarah", "model_id": "eleven_flash_v2_5", "mode": "ws"},
    ),
    "local-stt": Preset(
        name="local-stt",
        description="Local faster-whisper STT + Cerebras + ElevenLabs. Removes one network hop.",
        stt={"kind": "faster-whisper", "model": "base.en"},
        llm={"kind": "cerebras", "model": "qwen-3.8-27b", "reasoning": "none"},
        tts={"kind": "elevenlabs", "voice": "sarah", "model_id": "eleven_flash_v2_5", "mode": "ws"},
    ),
    "fully-local": Preset(
        name="fully-local",
        description="Everything on this laptop: faster-whisper + Ollama + Kokoro. Free, private, less natural voice.",
        stt={"kind": "faster-whisper", "model": "base.en"},
        llm={"kind": "ollama", "model": "qwen3:4b-instruct-2507-q4_K_M"},
        tts={"kind": "kokoro", "voice": "af_heart"},
    ),
    "parakeet-local": Preset(
        name="parakeet-local",
        description="Like fully-local but NVIDIA Parakeet (sherpa-onnx) for STT, which is faster and more accurate than whisper base.",
        stt={"kind": "parakeet"},
        llm={"kind": "ollama", "model": "qwen3:4b-instruct-2507-q4_K_M"},
        tts={"kind": "kokoro", "voice": "af_heart"},
    ),
}
