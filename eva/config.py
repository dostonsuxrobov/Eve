"""Paths, turn-taking settings, and the two tables a variant is made of: brains and voices.

Everything runs on this laptop (CLAUDE.md, 2026-09-25). A variant is one brain (an Ollama
model) and one voice (Kokoro in-process, or an expressive model behind the local voice
server, ``voice/server.py`` in ``.venv-voice``). ``run.py --brain X --voice Y`` picks one;
with neither it shows both lists.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
SAMPLES_DIR = ROOT / "samples"
MEMORY_FILE = ROOT / "memory.json"
NOTES_FILE = ROOT / "notes.json"  # eva.tools remember_note / recall_notes
VOICE_REFS_DIR = ROOT / "eva" / "assets" / "voices"  # reference clips for voice cloning

USER_AGENT = "eva-voice-agent/0.2"

# On this Windows box `localhost` resolves to ::1 first and costs ~2 s per request
# before falling back. Always use the IPv4 literal.
OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
VOICE_SERVER_URL = "http://127.0.0.1:8765"


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



# Brains: Ollama models, measured in docs/MEASUREMENTS.md (2026-09-25; "GPU" is nvidia-smi
# with the model loaded at an 8k context). Models of 2.5B and under get the short persona:
# on the full one they rambled, looped on "mm" or slid into a help-desk voice. "tools": False
# for a model whose Ollama template has none (Ollama refuses the request otherwise); "think":
# None for a model without a thinking mode (Ollama refuses the field).
BRAINS: dict[str, dict[str, Any]] = {
    "qwen4b": {"kind": "ollama", "model": "qwen3:4b-instruct-2507-q4_K_M", "persona": "eva",
               "label": "Qwen3 4B: fewest rule breaks in the bench (7/41); 3.2 GB, 56 tok/s, tools 4/6"},
    "qwen2b": {"kind": "ollama", "model": "qwen3.5:2b-q4_K_M", "persona": "eva_small", "tool_gate": True,
               "label": "Qwen 3.5 2B: 2.4 GB, 99 tok/s, rambles (51 words), tools 3/6"},
    "minicpm2b": {"kind": "ollama", "model": "openbmb/minicpm5-2b", "persona": "eva_small", "tool_gate": True,
                  "label": "MiniCPM5 2B: 1.7 GB, 92 tok/s, rambles (50 words), tools 3/6"},
    "lfm1b": {"kind": "ollama", "model": "LiquidAI/lfm2.5-1.2b-instruct:q8_0", "persona": "eva_small", "tool_gate": True,
              "label": "LFM2.5 1.2B: 1.4 GB, 123 tok/s, help-desk register, tools 2/6"},
    "minicpm1b": {"kind": "ollama", "model": "openbmb/minicpm5:q8_0", "persona": "eva_small", "tool_gate": True,
                  "label": "MiniCPM5 1B: 1.2 GB, 142 tok/s, best small tool caller (4/6), loops on 'mm'"},
    "gemma1b": {"kind": "ollama", "model": "gemma3:1b-it-qat", "persona": "eva_small", "tool_gate": True, "tools": False, "think": None,
                "label": "Gemma 3 1B: 1.2 GB, 123 tok/s, no tools"},
    "qwen08b": {"kind": "ollama", "model": "qwen3.5:0.8b", "persona": "eva_small", "tool_gate": True,
                "label": "Qwen 3.5 0.8B: 1.4 GB, 132 tok/s, often incoherent"},
}
DEFAULT_BRAIN = "minicpm1b"

# Voices. "voice-server" voices run in .venv-voice (PyTorch + CUDA) behind voice/server.py,
# which run.py starts on demand. Orpheus generates its audio tokens in Ollama next to the
# brain (2.1 GB at a 2k context) and the server decodes them with SNAC on the CPU; Chatterbox
# clones the reference clip in eva/assets/voices/ (PyTorch on the GPU: Turbo 3.1 GB reserved,
# the 0.5B model 3.6 GB, plus ~0.3 GB of CUDA context). Brain + voice must stay under ~5.7 GB
# of the 6 GB card, or Windows spills into system RAM and everything slows ~20x (eva/gpu.py).
VOICES: dict[str, dict[str, Any]] = {
    "kokoro": {"kind": "kokoro", "voice": "af_heart", "label": "Kokoro af_heart: fast, on the CPU, flat"},
    **{
        f"orpheus-{name}": {"kind": "voice-server", "engine": "orpheus", "voice": name,
                            "label": f"Orpheus 3B '{name}': laughs, sighs, gasps inline; 2.1 GB; ~1.1-1.8 s to first sound"}
        for name in ("tara", "leah", "jess", "mia", "zoe")
    },
    "chatterbox-turbo": {"kind": "voice-server", "engine": "chatterbox-turbo", "voice": "eva",
                         "label": "Chatterbox Turbo 350M: cloned voice, laughs; 3.4 GB (1B brains only); ~1 s a sentence"},
    "chatterbox": {"kind": "voice-server", "engine": "chatterbox", "voice": "eva",
                   "label": "Chatterbox 0.5B: cloned voice, emotion strength follows her cue; 3.9 GB (1B brains only); slow"},
}
DEFAULT_VOICE = "orpheus-tara"

LOCAL_STT: dict[str, Any] = {"kind": "parakeet"}


@dataclass
class Preset:
    """One runnable stack: speech-to-text, a brain, a voice, a persona, the turn-taking settings."""

    name: str
    description: str
    stt: dict[str, Any]
    llm: dict[str, Any]
    tts: dict[str, Any]
    persona: str = "eva"
    settings: PipelineSettings = field(default_factory=PipelineSettings)


def make_preset(brain: str = DEFAULT_BRAIN, voice: str = DEFAULT_VOICE) -> Preset:
    """The variant ``brain`` x ``voice`` (keys of BRAINS and VOICES)."""
    b, v = BRAINS[brain], VOICES[voice]
    llm = {k: val for k, val in b.items() if k not in ("label", "persona")}
    tts = {k: val for k, val in v.items() if k != "label"}
    return Preset(
        name=f"{brain}+{voice}",
        description=f"Parakeet STT, {b['label']}; voice {v['label']}.",
        stt=dict(LOCAL_STT),
        llm=llm,
        tts=tts,
        persona=b.get("persona", "eva"),
    )
