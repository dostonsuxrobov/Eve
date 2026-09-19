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
NOTES_FILE = ROOT / "notes.json"  # eva.tools remember_note / recall_notes

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
    "lily": "pFZP5JQG7iQjIQuC4Bku",
    # "jessica" (cgSgspJ2msm6clMXDiKS) returns 404 voice_not_found on this key; removed.
    # Voices the user picked for the Maya-like preset (verified on this key, flash + v3):
    "eva_en": "QLAlOeRuLwKX0skeTR7R",
    "eva_ru": "yMBZR4SLoc24wOJLWAB2",
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

    # 0.5 / 200 ms let laptop fan and keyboard noise through as "speech", which Whisper
    # then turned into phantom sentences; 0.6 / 250 ms is quiet in a normal room.
    vad_threshold: float = 0.6
    endpoint_silence_ms: int = 550  # silence after speech that ends the user's turn
    min_speech_ms: int = 250  # ignore blips shorter than this
    prespeech_buffer_ms: int = 300  # audio kept before VAD fires so we don't clip onsets
    max_utterance_s: float = 30.0
    barge_in: bool = True
    barge_in_min_speech_ms: int = 300  # how long the user must talk to interrupt
    filler_after_ms: int = 900  # play a filler if no agent audio by then (0 = off)
    first_chunk_min_chars: int = 14  # send first TTS chunk early (after a comma) for low TTFA
    min_chunk_chars: int = 6  # sentences shorter than this are merged into the next chunk
    # Envelope of a spoken turn (eva.audio.envelope): TTS clips are trimmed hard at both
    # ends, so without this a reply starts at full volume the instant the endpoint fires
    # and stops dead on the last sample. Milliseconds; 0 disables a stage.
    lead_in_ms: int = 80  # silence before the first audio of a reply (skipped after a filler)
    fade_in_ms: int = 50
    fade_out_ms: int = 100
    tail_ms: int = 350  # silence after the last word before she is "listening" again
    sentence_gap_ms: int = 0  # extra silence between sentence chunks (pace is tuned; leave 0)
    tts_parallelism: int = 2  # sentences synthesized ahead of playback
    input_device: int | None = None
    output_device: int | None = None
    echo_guard: bool = True  # raise VAD threshold while speaking when not using headphones
    # Backchannels ("mm-hm") at natural dips inside a long user utterance. Off by default:
    # through speakers the sound reaches the mic; with headphones it feels alive.
    backchannels: bool = False
    backchannel_after_ms: int = 4000  # the user must have been talking this long
    backchannel_dip_ms: int = 240  # VAD below the end threshold for this long = a breath/pause
    backchannel_min_gap_s: float = 8.0
    # If the transcript looks unfinished ("...and then," / no final punctuation), wait this
    # long for the user to go on before answering; if they do, the pieces are merged into
    # one turn and no LLM call is wasted. 0 = off.
    incomplete_grace_ms: int = 600


@dataclass
class Preset:
    name: str
    description: str
    stt: dict[str, Any]
    llm: dict[str, Any]
    tts: dict[str, Any]
    persona: str = "eva"
    settings: PipelineSettings = field(default_factory=PipelineSettings)


# LLM "reasoning" values for Cerebras (mapped by eva.factory.build_llm):
#   "low" / "medium" / "high" -> reasoning_effort (the default for every qwen preset here).
#   "none" (or None / "off")  -> disable_reasoning: true. Fastest content TTFT (0.29 s vs
#       0.43 s median) but qwen-3.8-27b then ends 25-40 % of very short replies mid-word
#       ("That stings a") and the broken text cascades through the history
#       (docs/EVAL_REPORT.md section 6). Set it only if you accept that.
# max_tokens is raised to 800 on the reasoning presets so that a long think can never
# leave the reply empty (finish=length with 400 was measured 2/82 turns).
PRESETS: dict[str, Preset] = {
    # ---- the Maya-like family: user-picked voices, EN/RU switching, delivery cues ----
    "maya": Preset(
        name="maya",
        description=(
            "Best effort at Maya: Scribe realtime STT, Cerebras qwen-3.8-27b, ElevenLabs v3 with "
            "delivery tags on the eva_en / eva_ru voices, Flash for the first chunk so the reply "
            "starts fast, prosodic continuity between sentences, backchannels (headphones)."
        ),
        stt={"kind": "elevenlabs-realtime", "model_id": "scribe_v2_realtime"},
        llm={"kind": "cerebras", "model": "qwen-3.8-27b", "reasoning": "low", "max_tokens": 800},
        tts={
            "kind": "elevenlabs",
            "voice": "eva_en",
            "voices_by_lang": {"ru": "eva_ru"},
            "model_id": "eleven_v3",
            "first_chunk_model": "eleven_flash_v2_5",
            "mode": "http",
        },
        settings=PipelineSettings(
            endpoint_silence_ms=500, filler_after_ms=800, backchannels=True,
            first_chunk_min_chars=18, min_chunk_chars=10,
        ),
    ),
    "maya-v3": Preset(
        name="maya-v3",
        description="maya with every chunk on ElevenLabs v3 (most expressive, ~0.5 s slower to start).",
        stt={"kind": "elevenlabs-realtime", "model_id": "scribe_v2_realtime"},
        llm={"kind": "cerebras", "model": "qwen-3.8-27b", "reasoning": "low", "max_tokens": 800},
        tts={
            "kind": "elevenlabs",
            "voice": "eva_en",
            "voices_by_lang": {"ru": "eva_ru"},
            "model_id": "eleven_v3",
            "mode": "http",
        },
        settings=PipelineSettings(
            endpoint_silence_ms=500, filler_after_ms=800, backchannels=True,
            first_chunk_min_chars=18, min_chunk_chars=10,
        ),
    ),
    "maya-fast": Preset(
        name="maya-fast",
        description="maya on ElevenLabs Flash only: fastest; delivery cues become voice settings per sentence.",
        stt={"kind": "elevenlabs-realtime", "model_id": "scribe_v2_realtime"},
        llm={"kind": "cerebras", "model": "qwen-3.8-27b", "reasoning": "low", "max_tokens": 800},
        tts={
            "kind": "elevenlabs",
            "voice": "eva_en",
            "voices_by_lang": {"ru": "eva_ru"},
            "model_id": "eleven_flash_v2_5",
            "mode": "http",
        },
        settings=PipelineSettings(
            endpoint_silence_ms=500, filler_after_ms=800, backchannels=True,
            first_chunk_min_chars=18, min_chunk_chars=10,
        ),
    ),
    "cloud-fast": Preset(
        name="cloud-fast",
        description="ElevenLabs Scribe realtime STT + Cerebras qwen-3.8-27b (low reasoning) + ElevenLabs Flash. Lowest cloud latency.",
        stt={"kind": "elevenlabs-realtime", "model_id": "scribe_v2_realtime"},
        llm={"kind": "cerebras", "model": "qwen-3.8-27b", "reasoning": "low", "max_tokens": 800},
        tts={"kind": "elevenlabs", "voice": "sarah", "model_id": "eleven_flash_v2_5", "mode": "http"},
    ),
    "cloud-smart": Preset(
        name="cloud-smart",
        description="Same audio stack, Cerebras gpt-oss-120b with low reasoning. Smarter, a bit slower.",
        stt={"kind": "elevenlabs-realtime", "model_id": "scribe_v2_realtime"},
        llm={"kind": "cerebras", "model": "gpt-oss-120b", "reasoning": "low"},
        tts={"kind": "elevenlabs", "voice": "sarah", "model_id": "eleven_flash_v2_5", "mode": "http"},
    ),
    "expressive": Preset(
        name="expressive",
        description="Cerebras qwen + ElevenLabs v3 with audio tags ([laughs], [sighs]). Most emotional, highest TTS latency.",
        stt={"kind": "elevenlabs-realtime", "model_id": "scribe_v2_realtime"},
        llm={"kind": "cerebras", "model": "qwen-3.8-27b", "reasoning": "low", "max_tokens": 800},
        tts={"kind": "elevenlabs", "voice": "sarah", "model_id": "eleven_v3", "mode": "http"},
    ),
    "local-brain": Preset(
        name="local-brain",
        description="Cloud audio (batch Scribe v2), local Ollama qwen3:4b brain. Tests how far a small local model gets.",
        stt={"kind": "elevenlabs", "model_id": "scribe_v2"},
        llm={"kind": "ollama", "model": "qwen3:4b-instruct-2507-q4_K_M"},
        tts={"kind": "elevenlabs", "voice": "sarah", "model_id": "eleven_flash_v2_5", "mode": "http"},
    ),
    "local-stt": Preset(
        name="local-stt",
        description="Local faster-whisper STT + Cerebras + ElevenLabs. Removes one network hop.",
        stt={"kind": "faster-whisper", "model": "base.en", "device": "cpu"},
        llm={"kind": "cerebras", "model": "qwen-3.8-27b", "reasoning": "low", "max_tokens": 800},
        tts={"kind": "elevenlabs", "voice": "sarah", "model_id": "eleven_flash_v2_5", "mode": "http"},
    ),
    "fully-local": Preset(
        name="fully-local",
        description="Everything on this laptop: faster-whisper + Ollama + Kokoro. Free, private, less natural voice.",
        stt={"kind": "faster-whisper", "model": "base.en", "device": "cpu"},
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
