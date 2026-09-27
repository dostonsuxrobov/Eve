"""Paths, turn-taking settings, the brain, the voices and the ElevenLabs plan.

The setup chosen on 2026-09-26 (owner): the brain and the ears on this laptop (Ollama 4B,
Parakeet), the voice from ElevenLabs on the owner's prepaid Creator plan. The voice was the
bottleneck of the fully local build (Chatterbox took 2.3-3.4 s to its first sound on the 6 GB
GPU); the brain costs 0.1-0.4 s at any size. Kokoro stays as the local voice and the automatic
fallback when ElevenLabs can't answer (no internet, credits spent).
"""
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
USAGE_FILE = ROOT / "usage.json"  # ElevenLabs credits spent, per billing month (eva.credits)

USER_AGENT = "eva-voice-agent/0.3"

# On this Windows box `localhost` resolves to ::1 first and costs ~2 s per request
# before falling back. Always use the IPv4 literal.
OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"

# The RunPod pilot (2026-09-27, deploy/runpod/): the brain in vLLM and the open voices in
# voice/server.py, on the same GPU box as the loop, so only her audio crosses the internet.
POD_LLM_URL = os.environ.get("EVA_POD_LLM_URL", "http://127.0.0.1:8100/v1")
VOICE_SERVER_URL = os.environ.get("EVA_VOICE_SERVER_URL", "http://127.0.0.1:8765")

# ElevenLabs, owner's plan (2026-09-26): Creator, billed annually, paid through 2027-07-28.
# 121k credits a month; unused credits roll over for up to two months (balance at most 3x).
# The key only has text_to_speech / speech_to_text (no models_read, voices_read, user_read),
# so the balance can't be read from the API: eva.credits counts what the responses report.
ELEVENLABS_VOICE = "bD9maNcCuQQS75DGuteM"  # the owner's pick, 2026-09-26
PLAN_MONTHLY_CREDITS = 121_000
PLAN_RENEWS_ON_DAY = 28  # the subscription date (2027-07-28): each month's credits arrive on the 28th
PLAN_EXPIRES = "2027-07-28"
CREDITS_WARN_AT = 0.8  # of the month's allowance


def key_file(name: str, env: str) -> str | None:
    """A key from ``env`` or the gitignored ``<name>`` file (its first line that looks like a key)."""
    if os.environ.get(env):
        return os.environ[env].strip()
    path = ROOT / name
    if not path.exists():
        return None
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]
    keyish = [ln for ln in lines if " " not in ln and len(ln) >= 20]
    return (keyish or lines or [None])[0]


def elevenlabs_key() -> str | None:
    """From ELEVENLABS_API_KEY or the gitignored elevenlabs_key.txt."""
    if os.environ.get("ELEVENLABS_API_KEY"):
        return os.environ["ELEVENLABS_API_KEY"].strip()
    path = ROOT / "elevenlabs_key.txt"
    return path.read_text(encoding="utf-8").strip() if path.exists() else None


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
    sentence_gap_ms: int = 220  # pause between sentence chunks: v3 clips are trimmed hot, so without one sentences butt together
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



# The brain: Qwen3 4B instruct on Ollama, with the small-brain scaffolding (eva/toolgate.py tool
# gate and routed questions, eva/guard.py speech guard): the 4B has the same failure classes as
# the 1B in the bench ("One sec, setting that timer" without a call, a call written as text).
BRAINS: dict[str, dict[str, Any]] = {
    "qwen4b": {"kind": "ollama", "model": "qwen3:4b-instruct-2507-q4_K_M", "persona": "eva", "tool_gate": True,
               "label": "Qwen3 4B instruct, on this laptop (3.2 GB on the GPU, 56 tok/s)"},
    # The cloud era's brain (6.2/10 in its eval, 6/6 tools, no scaffolding), open weights, on Cerebras.
    # Reasoning "low": off, it cut 25-40 % of very short replies mid-word (archived eval).
    "qwen27b": {"kind": "openai", "base_url": "https://api.cerebras.ai/v1", "model": "qwen-3.8-27b", "persona": "eva",
                "key": ("cerebras_api_key.txt", "CEREBRAS_API_KEY"), "extra_body": {"reasoning_effort": "low"},
                # 2000: at 800 a hard question (a recovery plan) spent it all on reasoning, nothing left to say
                "temperature": 0.7, "max_tokens": 2000,
                "label": "Qwen3.8-27B on Cerebras (open weights, cloud): the cloud era's brain"},
    # The same model served by vLLM on the RunPod pod in FP8 (deploy/runpod/). Thinking off through the
    # chat template; sampling from the model card.
    "qwen27b-pod": {"kind": "openai", "base_url": POD_LLM_URL, "model": "qwen27b", "persona": "eva",
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}, "top_p": 0.8, "top_k": 20,
                               "presence_penalty": 1.5},
                "temperature": 0.7, "max_tokens": 600,
                "label": "Qwen3.8-27B on the RunPod pod (vLLM, FP8)"},
}
DEFAULT_BRAIN = "qwen4b"

# The voices. ElevenLabs on the owner's voice; measured 2026-09-26 on one 64-character line:
# first audio v3 0.72 s, v3 Conversational 0.31 s, Flash v2.5 0.47 s (cold connections), billed
# 35 / 17 / 17 credits (the response's character-cost header). There is no "Flash v3".
VOICES: dict[str, dict[str, Any]] = {
    "v3": {"kind": "elevenlabs", "model_id": "eleven_v3", "voice": ELEVENLABS_VOICE,
           "label": "ElevenLabs v3: the most expressive, audio tags; ~0.7 s to first audio"},
    "v3conv": {"kind": "elevenlabs", "model_id": "eleven_v3_conversational", "voice": ELEVENLABS_VOICE,
               "label": "ElevenLabs v3 Conversational: v3's tags, faster (~0.3 s), half the credits"},
    "flash": {"kind": "elevenlabs", "model_id": "eleven_flash_v2_5", "voice": ELEVENLABS_VOICE,
              "label": "ElevenLabs Flash v2.5: the fastest, half the credits, no audio tags"},
    "kokoro": {"kind": "kokoro", "voice": "af_heart", "label": "Kokoro, on this laptop: no credits, flat"},
    # Open voices behind voice/server.py, for the pod (on this laptop Chatterbox took 2.3-3.4 s to
    # its first sound, .archive/local-variants/docs/MEASUREMENTS.md). Chatterbox clones
    # eva/assets/voices/eva.wav and follows her delivery cue; Orpheus needs Ollama next to it.
    "chatterbox": {"kind": "voice-server", "engine": "chatterbox", "voice": "eva",
                   "label": "Chatterbox 0.5B (open): cloned voice, emotion strength follows her cue"},
    "orpheus": {"kind": "voice-server", "engine": "orpheus", "voice": "tara",
                "label": "Orpheus 3B 'tara' (open): laughs, sighs, gasps inline"},
}
DEFAULT_VOICE = "v3"

LOCAL_STT: dict[str, Any] = {"kind": "parakeet"}
LOCAL_TTS: dict[str, Any] = {"kind": "kokoro", "voice": "af_heart"}  # the fallback voice

# The cloud era's tuned turn-taking for the ElevenLabs voice (the setup that felt right, archived
# config _MAYA_SETTINGS), plus one change: render one sentence ahead, not two, so an interruption
# wastes at most one unheard sentence of credits.
ELEVENLABS_SETTINGS = PipelineSettings(
    endpoint_silence_ms=500, filler_after_ms=800, backchannels=True,
    first_chunk_min_chars=40, min_chunk_chars=20, tts_parallelism=1,
)
# The open voices on the pod: the same turn-taking, a shorter first chunk (Chatterbox renders a
# whole chunk before its first sample, so a shorter one speaks sooner). Starting points, to measure.
VOICE_SERVER_SETTINGS = PipelineSettings(
    endpoint_silence_ms=500, filler_after_ms=800, backchannels=True,
    first_chunk_min_chars=24, min_chunk_chars=12, tts_parallelism=1,
)


@dataclass
class Preset:
    """One runnable stack: speech-to-text, a brain, a voice (and its fallback), a persona, settings."""

    name: str
    description: str
    stt: dict[str, Any]
    llm: dict[str, Any]
    tts: dict[str, Any]
    persona: str = "eva"
    settings: PipelineSettings = field(default_factory=PipelineSettings)
    tts_fallback: dict[str, Any] | None = None


def make_preset(brain: str = DEFAULT_BRAIN, voice: str = DEFAULT_VOICE) -> Preset:
    """The variant ``brain`` x ``voice`` (keys of BRAINS and VOICES)."""
    b, v = BRAINS[brain], VOICES[voice]
    llm = {k: val for k, val in b.items() if k not in ("label", "persona")}
    tts = {k: val for k, val in v.items() if k != "label"}
    cloud = v["kind"] == "elevenlabs"
    served = v["kind"] == "voice-server"
    return Preset(
        name=f"{brain}+{voice}",
        description=f"Parakeet STT, {b['label']}; voice {v['label']}.",
        stt=dict(LOCAL_STT),
        llm=llm,
        tts=tts,
        persona=b.get("persona", "eva"),
        settings=ELEVENLABS_SETTINGS if cloud else VOICE_SERVER_SETTINGS if served else PipelineSettings(),
        tts_fallback=dict(LOCAL_TTS) if cloud or served else None,
    )
