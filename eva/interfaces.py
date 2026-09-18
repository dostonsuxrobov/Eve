"""Shared contracts for every swappable component in Eva.

Every STT / LLM / TTS implementation MUST conform to these Protocols exactly so
that the pipeline, the factory and the benchmarks can treat them interchangeably.

Audio conventions (do not deviate):
  * Microphone / STT audio: int16 mono numpy array at 16 000 Hz.
  * TTS output: raw little-endian int16 mono PCM bytes at `tts.sample_rate`
    (24 000 Hz for ElevenLabs pcm_24000 and Kokoro; 16 000 for others).
  * All providers are asyncio-native. Blocking work (ONNX inference, whisper)
    must run in a thread via `asyncio.to_thread` so the audio loop never stalls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol, runtime_checkable

import numpy as np

MIC_SAMPLE_RATE = 16_000


# --------------------------------------------------------------------------- STT
@dataclass
class Transcript:
    text: str
    latency_s: float  # wall time spent inside transcribe()
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class STT(Protocol):
    name: str

    async def warmup(self) -> None:
        """Load models / open connections. Called once before the first turn."""

    async def transcribe(self, pcm: np.ndarray, sample_rate: int = MIC_SAMPLE_RATE) -> Transcript:
        """Transcribe one complete utterance (int16 mono)."""

    async def close(self) -> None: ...


@runtime_checkable
class StreamingSTT(STT, Protocol):
    """Optional low-latency extension: audio is streamed *while* the user speaks.

    The pipeline detects support with ``isinstance(stt, StreamingSTT)`` and then
    calls ``feed(frame)`` for every mic frame from speech onset (including the
    pre-speech buffer) and ``commit()`` at the VAD endpoint; ``commit()`` returns
    the transcript of everything fed since the previous commit.  ``discard()``
    abandons the audio fed so far (a blip that was not a turn, or a commit that a
    barge-in cancelled; ``keep_audio=True`` keeps the client copy for re-sending).
    ``transcribe()`` must keep working for utterances that were not streamed.
    """

    async def feed(self, pcm: np.ndarray) -> None: ...

    async def commit(self) -> Transcript: ...

    async def discard(self, keep_audio: bool = False) -> None: ...


# --------------------------------------------------------------------------- LLM
@dataclass
class LLMDelta:
    text: str


@dataclass
class LLMToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMDone:
    """End of one assistant turn.

    ``finish_reason`` is the backend's (``stop`` / ``tool_calls`` / ``length``) or
    ``"error"`` when the stream broke after events were already yielded (the error
    text is then in ``usage["error"]``).  ``usage`` may carry backend-specific keys
    (Cerebras ``time_info``, reasoning counters ...).
    """

    finish_reason: str
    ttft_s: float | None
    total_s: float
    usage: dict[str, Any] = field(default_factory=dict)


LLMEvent = LLMDelta | LLMToolCall | LLMDone


@runtime_checkable
class LLM(Protocol):
    name: str

    async def warmup(self) -> None:
        """Open a keep-alive connection / load the model so the first real turn is fast."""

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMEvent]:
        """Yield LLMDelta for text, LLMToolCall for each complete tool call (after
        arguments are fully accumulated), and exactly one LLMDone at the end.

        `messages` is OpenAI chat format. Tool results are appended by the caller as
        {"role": "tool", "tool_call_id": ..., "content": ...} messages.

        If the request cannot be started at all the implementation raises; once the
        first event has been yielded it never raises and ends with
        ``LLMDone(finish_reason="error")`` instead.
        """

    async def close(self) -> None: ...

    # Optional (not required by the Protocol): ``async def complete(messages, **kw) -> str``
    # for non-streaming helpers such as the memory summariser, which feature-detects it
    # and otherwise drains ``stream()``.


# --------------------------------------------------------------------------- TTS
@runtime_checkable
class TTS(Protocol):
    name: str
    sample_rate: int
    supports_audio_tags: bool  # True if inline tags like [laughs] / [sighs] are rendered

    async def warmup(self) -> None: ...

    def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Yield raw int16 mono PCM chunks at `sample_rate` as soon as they are ready.

        Implementations should start yielding within ~100-300 ms (streaming APIs) and
        MUST NOT buffer the whole utterance before the first yield unless the backend
        physically cannot stream.
        """

    async def close(self) -> None: ...


# ------------------------------------------------------------------------- Tools
@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for the arguments object
    fn: Any  # sync or async callable(**arguments) -> str
    spoken_hint: str | None = None  # e.g. "let me check" spoken while the tool runs

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ----------------------------------------------------------------------- Metrics
@dataclass
class TurnMetrics:
    """Wall-clock timestamps (time.perf_counter()) for one conversational turn."""

    speech_start: float | None = None
    speech_end: float | None = None
    stt_done: float | None = None
    llm_first_token: float | None = None
    tts_first_audio: float | None = None
    audio_started: float | None = None
    audio_finished: float | None = None
    interrupted: bool = False
    user_text: str = ""
    assistant_text: str = ""

    def response_latency(self) -> float | None:
        """The number that matters: user stops talking -> agent audio starts."""
        if self.speech_end is None or self.audio_started is None:
            return None
        return self.audio_started - self.speech_end

    def breakdown(self) -> dict[str, float | None]:
        def d(a: float | None, b: float | None) -> float | None:
            return None if a is None or b is None else round(b - a, 3)

        return {
            "stt": d(self.speech_end, self.stt_done),
            "llm_ttft": d(self.stt_done, self.llm_first_token),
            "tts_ttfa": d(self.llm_first_token, self.tts_first_audio),
            "playback_start": d(self.tts_first_audio, self.audio_started),
            "total": d(self.speech_end, self.audio_started),
        }
