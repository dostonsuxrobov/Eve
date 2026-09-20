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
    # While she is audible, a VAD onset alone must not interrupt her: through speakers her
    # own voice reaches the mic, and a session on 2026-09-19 looped twelve turns on that
    # (she cut herself off, transcribed her own garbled echo, answered it). "words": the
    # streaming STT's partial transcript must carry real words that are not a fuzzy match
    # of what she is saying, and the echo detector must not hear the speakers; with no
    # partial at all after barge_in_words_wait_ms of speech the onset counts anyway (the
    # STT is lagging). "vad": the old 300 ms-of-speech rule. While she is silent (thinking)
    # the VAD rule applies in both modes: there is nothing to echo.
    barge_in_confirm: str = "words"
    barge_in_words_wait_ms: int = 1500
    echo_detector: bool = True  # eva.audio.echo: correlate the mic with what the player just played
    filler_after_ms: int = 900  # play a filler if no agent audio by then (0 = off)
    first_chunk_min_chars: int = 14  # send first TTS chunk early (after a comma) for low TTFA
    min_chunk_chars: int = 6  # sentences shorter than this are merged into the next chunk
    # Self-echo gate on transcripts: drop an utterance that began while she was audible and
    # reads like a garbled copy of what she was saying (the laptop's mic leaks the speakers).
    # Off for the phone client: the browser's echo canceller does the job and the gate only
    # produced false positives (people repeat the other person's words).
    self_echo_gate: bool = True
    # Envelope of a spoken turn (eva.audio.envelope): TTS clips are trimmed hard at both
    # ends, so without this a reply starts at full volume the instant the endpoint fires
    # and stops dead on the last sample. Milliseconds; 0 disables a stage.
    # Measured on v3 clips (2026-09-20): they are trimmed hot, the first 10 ms at -39 dBFS and
    # the last 10 ms at -31 dBFS, so short fades left an audible start and an abrupt stop.
    lead_in_ms: int = 100  # room tone before the first audio of a reply (skipped after a filler)
    fade_in_ms: int = 120
    fade_out_ms: int = 280
    tail_ms: int = 450  # room tone after the last word before she is "listening" again
    sentence_gap_ms: int = 0  # extra room tone between sentence chunks (pace is tuned; leave 0)
    chunk_edge_ms: int = 40  # short fades at every chunk boundary: each v3 clip has hot edges
    # Optional faint noise bed in the gaps and while idle (None = digital silence). Tried at
    # -62 dBFS on 2026-09-20: audible hiss on an iPhone speaker, and unnecessary once every
    # sentence is v3 (in-speech floor -65..-84 dBFS) with 120/280 ms fades. Off.
    room_tone_dbfs: float | None = None
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
    """One runnable stack. ``*_fallback`` are provider configs used only when the primary
    stops responding (see ``eva.failover``); ``None`` means no fallback."""

    name: str
    description: str
    stt: dict[str, Any]
    llm: dict[str, Any]
    tts: dict[str, Any]
    persona: str = "eva"
    settings: PipelineSettings = field(default_factory=PipelineSettings)
    stt_fallback: dict[str, Any] | None = None
    llm_fallback: dict[str, Any] | None = None
    tts_fallback: dict[str, Any] | None = None


# The brain is Cerebras qwen-3.8-27b (decided 2026-09-19 after the live comparison: "much
# more natural by much larger margins"). Every other candidate measured, cloud and local, is in
# docs/EVAL_REPORT.md sections 3 and 10 and in git history. `local` is not a choice of brain: it
# is the on-device fallback when Cerebras is unreachable, and the `local` offline preset.
#   Cerebras "reasoning": "low" / "medium" / "high" -> reasoning_effort. "none" ->
#       disable_reasoning (faster first token, but qwen then truncates 25-40 % of very short
#       replies mid-word, section 6). max_tokens 800 so a long think never empties the reply.
#   Ollama brains run on the native API with thinking off (eva/llm/ollama_native.py).
BRAINS: dict[str, dict[str, Any]] = {
    # 6.2/10, 6/6 tools, 0.30 s TTFT, about $0.003 per exchange ($0.99 / $1.49 per M tokens)
    "qwen": {"kind": "cerebras", "model": "qwen-3.8-27b", "reasoning": "low", "max_tokens": 800},
    # qwen3:8b: 3.7/10 as a companion but 6/6 on tool calls; the fallback brain and the offline preset
    "local": {"kind": "ollama", "model": "qwen3:8b"},
}
DEFAULT_BRAIN = "qwen"

# Local providers: the `local` preset and the fallbacks of the cloud preset.
LOCAL_STT: dict[str, Any] = {"kind": "parakeet"}
LOCAL_TTS: dict[str, Any] = {"kind": "kokoro", "voice": "af_heart"}

_MAYA_STT: dict[str, Any] = {"kind": "elevenlabs-realtime", "model_id": "scribe_v2_realtime"}
_MAYA_TTS: dict[str, Any] = {
    "kind": "elevenlabs",
    "voice": "eva_en",  # per-language voices come from eva/assets/lang/*.toml
    "model_id": "eleven_v3",
    # v3 for every chunk. The Flash first chunk (0.2 s to first audio against v3's 0.5-0.9) put
    # a different timbre, pace and noise floor on the first sentence of every reply: "rushed
    # start, then it settles". Set "first_chunk_model": "eleven_flash_v2_5" to trade back.
    "first_chunk_model": None,
}
_MAYA_SETTINGS = PipelineSettings(
    endpoint_silence_ms=500, filler_after_ms=800, backchannels=True,
    # a first chunk that is a whole clause and no tiny clips: short clips sound rushed and clipped
    first_chunk_min_chars=40, min_chunk_chars=20,
)
_MAYA_EARS_AND_VOICE = (
    "Scribe realtime STT, ElevenLabs v3 on the eva_en / eva_ru voices with one delivery cue per reply, "
    "backchannels (headphones). Falls back to Parakeet / Ollama qwen3:8b / Kokoro when a cloud service "
    "stops answering."
)


def _maya(name: str, brain: str, blurb: str) -> Preset:
    """The Maya stack with one of the three brains; same ears and voice, so what differs is the brain."""
    return Preset(
        name=name,
        description=f"{blurb} {_MAYA_EARS_AND_VOICE}",
        stt=dict(_MAYA_STT),
        llm=dict(BRAINS[brain]),
        tts=dict(_MAYA_TTS),
        settings=_MAYA_SETTINGS,
        stt_fallback=dict(LOCAL_STT),
        llm_fallback=dict(BRAINS["local"]),
        tts_fallback=dict(LOCAL_TTS),
    )


# The agent, plus the fully offline stack (the same providers the agent falls back to).
PRESETS: dict[str, Preset] = {
    "maya": _maya("maya", "qwen", "Brain: Cerebras qwen-3.8-27b (6.2/10, 0.30 s to first token, about $0.003 per exchange)."),
    "local": Preset(
        name="local",
        description=(
            "Everything on this laptop: Parakeet TDT 0.6B (sherpa-onnx) STT, Ollama qwen3:8b, "
            "Kokoro TTS. Free, private, offline; tools work, the conversation is 3.7/10 (docs/EVAL_REPORT.md)."
        ),
        stt=dict(LOCAL_STT),
        llm=dict(BRAINS["local"]),
        tts=dict(LOCAL_TTS),
    ),
}
DEFAULT_PRESET = "maya"
