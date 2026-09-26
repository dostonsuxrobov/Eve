"""Test doubles for the Eva pipeline.

Everything here lets ``eva.pipeline.VoiceAgent`` run end to end without a
microphone, a sound card, a network connection or any of the sibling modules
(``eva.audio``, ``eva.llm``, ``eva.tools``).  The doubles are *paced* like the
real thing (STT delay, LLM TTFT + token rate, TTS TTFA + realtime factor, a
player that drains in wall-clock time) so the timing logic of the pipeline -
fillers, barge-in truncation, lookahead - is exercised for real.

Also contains the *stand-ins* for sibling modules that may not exist yet
(``StandInChunker``, ``standin_clean_for_tts``, ``standin_strip_think``,
``standin_execute``).  The pipeline falls back to them automatically.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import inspect
import math
import re
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable

import numpy as np

from .config import PipelineSettings
from .interfaces import MIC_SAMPLE_RATE, LLMDelta, LLMDone, LLMToolCall, Tool, Transcript, TurnMetrics

try:  # use the real event classes when the audio package is present
    from .audio.vad import SpeechEnd, SpeechStart  # type: ignore
except Exception:  # pragma: no cover - sibling not built yet

    @dataclass
    class SpeechStart:  # type: ignore[no-redef]
        t: float

    @dataclass
    class SpeechEnd:  # type: ignore[no-redef]
        t: float
        pcm: np.ndarray
        duration_s: float
        speech_ms: float = 0.0


# --------------------------------------------------------------------------- STT
class MockSTT:
    """Returns the next scripted text after ``delay_s``.

    The text is consumed only when the call completes, so a transcription that is
    cancelled by a barge-in does not eat its script entry.
    """

    name = "mock-stt"

    def __init__(self, texts: list[str], delay_s: float = 0.3, *, heard_as: list[str | None] | None = None) -> None:
        self._texts = list(texts)
        self._i = 0
        self.delay_s = delay_s
        self.calls: list[dict[str, Any]] = []
        # per transcript: the language the STT reports (meta["language"], like Scribe's detection)
        self._heard_as = list(heard_as or [])

    def _meta(self, i: int, **extra: Any) -> dict[str, Any]:
        meta: dict[str, Any] = {"mock": True, **extra}
        if i < len(self._heard_as) and self._heard_as[i]:
            meta["language"] = self._heard_as[i]
        return meta

    async def warmup(self) -> None:
        return None

    async def transcribe(self, pcm: np.ndarray, sample_rate: int = MIC_SAMPLE_RATE) -> Transcript:
        t0 = time.perf_counter()
        await asyncio.sleep(self.delay_s)
        i = self._i
        text = self._texts[i] if i < len(self._texts) else ""
        self._i += 1
        self.calls.append({"samples": int(len(pcm)), "text": text, "t": t0})
        latency = time.perf_counter() - t0
        if hasattr(self, "last_batch_s"):
            self.last_batch_s = latency
        return Transcript(text=text, latency_s=latency, meta=self._meta(i))

    async def close(self) -> None:
        return None


class MockStreamingSTT(MockSTT):
    """``MockSTT`` plus the :class:`eva.interfaces.StreamingSTT` methods.

    ``feed()`` only counts samples; ``commit()`` returns the next scripted text after
    ``commit_delay_s`` (the measured realtime commit-to-text is ~0.3 s vs ~1.1 s for
    batch); ``discard()`` records what was thrown away.  ``transcribe()`` still works
    (batch path) and is what the pipeline uses for merged / queued utterances.
    """

    name = "mock-streaming-stt"

    def __init__(
        self, texts: list[str], delay_s: float = 0.3, commit_delay_s: float = 0.1, *, heard_as: list[str | None] | None = None
    ) -> None:
        super().__init__(texts, delay_s, heard_as=heard_as)
        self.commit_delay_s = commit_delay_s
        self.feeds = 0
        self.segment_samples = 0
        self.commits: list[dict[str, Any]] = []
        self.discards: list[dict[str, Any]] = []
        self.commit_cancelled: list[float] = []  # perf_counter of each commit() cancelled mid-flight
        self.last_batch_s: float | None = None  # latency of the last batch request (incl. a warmup probe), like the real STT
        self.on_partial: Any = None  # the pipeline sets this; tests call emit_partial()

    def emit_partial(self, text: str) -> None:
        if self.on_partial is not None:
            self.on_partial(text)

    async def feed(self, pcm: np.ndarray) -> None:
        self.feeds += 1
        self.segment_samples += int(len(pcm))
        await asyncio.sleep(0)

    async def commit(self) -> Transcript:
        t0 = time.perf_counter()
        samples, self.segment_samples = self.segment_samples, 0
        try:
            await asyncio.sleep(self.commit_delay_s)
        except asyncio.CancelledError:
            self.commit_cancelled.append(time.perf_counter())
            raise
        i = self._i
        text = self._texts[i] if i < len(self._texts) else ""
        self._i += 1
        self.commits.append({"samples": samples, "text": text, "t": t0})
        return Transcript(text=text, latency_s=time.perf_counter() - t0, meta=self._meta(i, mode="stream"))

    async def discard(self, keep_audio: bool = False) -> None:
        self.discards.append({"samples": self.segment_samples, "keep_audio": keep_audio})
        if not keep_audio:
            self.segment_samples = 0


# --------------------------------------------------------------------------- LLM
@dataclass
class ScriptedError:
    """A reply slot that makes ``stream()`` raise before yielding anything."""

    message: str = "mock llm failure"


@dataclass
class ScriptedToolCall:
    """A reply that ends in a tool call.  ``preface`` is spoken before the call."""

    name: str
    arguments: dict[str, Any]
    preface: str = ""
    id: str = "call_mock_1"


class MockLLM:
    """Streams scripted replies word by word with a realistic TTFT.

    ``replies`` is consumed one entry per ``stream()`` call.  A ``ScriptedToolCall``
    entry yields its preface, then one ``LLMToolCall``; the *next* entry is what the
    model says after the tool result comes back.
    """

    name = "mock-llm"

    def __init__(
        self,
        replies: list[str | ScriptedToolCall | ScriptedError],
        ttft_s: float = 0.4,
        token_delay_s: float = 0.02,
    ) -> None:
        self._replies = list(replies)
        self._i = 0
        self.ttft_s = ttft_s
        self.token_delay_s = token_delay_s
        self.calls: list[list[dict[str, Any]]] = []
        self.tools_offered: list[list[str] | None] = []  # tool names offered per request
        self.pings: list[float] = []  # perf_counter of every ping() (only if ``pingable``)
        self.pingable = False  # set True to expose ping() like OpenAICompatLLM
        self.ping_delay_s = 0.05

    def __getattr__(self, name: str) -> Any:
        if name == "ping" and self.__dict__.get("pingable"):
            return self._ping
        raise AttributeError(name)

    async def _ping(self) -> float:
        t0 = time.perf_counter()
        self.pings.append(t0)
        await asyncio.sleep(self.ping_delay_s)
        return time.perf_counter() - t0

    async def warmup(self) -> None:
        return None

    async def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[LLMDelta | LLMToolCall | LLMDone]:
        t0 = time.perf_counter()
        self.calls.append([dict(m) for m in messages])
        self.tools_offered.append(None if tools is None else [t["function"]["name"] for t in tools])
        reply: str | ScriptedToolCall | ScriptedError = self._replies[self._i] if self._i < len(self._replies) else "Okay."
        self._i += 1
        if isinstance(reply, ScriptedError):
            raise RuntimeError(reply.message)
        await asyncio.sleep(self.ttft_s)
        ttft = time.perf_counter() - t0
        text = reply.preface if isinstance(reply, ScriptedToolCall) else reply
        words = text.split(" ") if text else []
        for k, w in enumerate(words):
            yield LLMDelta(text=w if k == len(words) - 1 else w + " ")
            if k < len(words) - 1:
                await asyncio.sleep(self.token_delay_s)
        if isinstance(reply, ScriptedToolCall):
            yield LLMToolCall(id=reply.id, name=reply.name, arguments=dict(reply.arguments))
            yield LLMDone(finish_reason="tool_calls", ttft_s=ttft, total_s=time.perf_counter() - t0)
        else:
            yield LLMDone(finish_reason="stop", ttft_s=ttft, total_s=time.perf_counter() - t0)

    async def close(self) -> None:
        return None


# --------------------------------------------------------------------------- TTS
class MockTTS:
    """Yields a quiet tone paced like a streaming TTS.

    Audio duration is ``len(text) / chars_per_second`` (about normal speech rate).
    The first chunk arrives after ``ttfa_s``; the rest is spread so that the whole
    utterance is generated in ``duration * realtime_factor`` seconds.
    """

    name = "mock-tts"
    supports_audio_tags = False

    def __init__(
        self,
        sample_rate: int = 24_000,
        ttfa_s: float = 0.25,
        realtime_factor: float = 0.3,
        chars_per_second: float = 15.0,
        chunk_ms: int = 100,
        tone_hz: float = 220.0,
        amplitude: int = 600,
    ) -> None:
        self.sample_rate = sample_rate
        self.ttfa_s = ttfa_s
        self.realtime_factor = realtime_factor
        self.chars_per_second = chars_per_second
        self.chunk_ms = chunk_ms
        self.tone_hz = tone_hz
        self.amplitude = amplitude
        self.calls: list[str] = []

    async def warmup(self) -> None:
        return None

    def render(self, text: str) -> np.ndarray:
        """Synchronously build the int16 tone for ``text``."""
        seconds = max(0.25, len(text) / self.chars_per_second)
        n = int(seconds * self.sample_rate)
        t = np.arange(n, dtype=np.float32) / self.sample_rate
        env = np.minimum(1.0, np.minimum(t, seconds - t) * 20).astype(np.float32)  # 50 ms fades
        return (np.sin(2 * math.pi * self.tone_hz * t) * env * self.amplitude).astype(np.int16)

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        self.calls.append(text)
        audio = self.render(text)
        total_s = len(audio) / self.sample_rate
        chunk = max(1, int(self.sample_rate * self.chunk_ms / 1000))
        n_chunks = max(1, math.ceil(len(audio) / chunk))
        per_chunk = (total_s * self.realtime_factor) / n_chunks
        await asyncio.sleep(self.ttfa_s)
        for i in range(n_chunks):
            yield audio[i * chunk : (i + 1) * chunk].tobytes()
            if i < n_chunks - 1:
                await asyncio.sleep(per_chunk)

    async def synthesize_to_bytes(self, text: str) -> bytes:
        parts = [b async for b in self.synthesize(text)]
        return b"".join(parts)

    async def close(self) -> None:
        return None


# ------------------------------------------------------------------------ Player
class MockPlayer:
    """Drains audio in wall-clock time without a device.

    Implements the full ``eva.audio.player.Player`` API: ``write`` / ``stop`` /
    ``mark`` / ``played_samples`` / ``is_active`` / ``buffered_seconds`` /
    ``wait_until_done``.  ``writes`` keeps ``(t_write, t_play_start, samples)`` per
    write so tests can prove that nothing overlapped.
    """

    def __init__(self, sample_rate: int = 24_000, device: int | None = None, block_ms: int = 20) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self.block_ms = block_ms
        self._end_t = 0.0  # perf_counter when the buffer runs dry
        self._written = 0
        self._mark_written = 0
        self._dropped = 0
        self.started = False
        self.writes: list[tuple[float, float, int]] = []
        self.pcm: list[bytes] = []  # the bytes themselves, for tests that look at the audio
        self.stop_calls: list[tuple[float, int]] = []

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.started = False
        self._end_t = 0.0

    # -- accounting -----------------------------------------------------------
    def _remaining(self, now: float | None = None) -> int:
        now = time.perf_counter() if now is None else now
        return int(max(0.0, self._end_t - now) * self.sample_rate)

    def write(self, pcm: bytes) -> None:
        n = len(pcm) // 2
        now = time.perf_counter()
        start = max(now, self._end_t)
        self._end_t = start + n / self.sample_rate
        self._written += n
        self.writes.append((now, start, n))
        self.pcm.append(bytes(pcm))

    def mark(self) -> None:
        self._mark_written = self._written
        self._dropped = 0

    @property
    def played_samples(self) -> int:
        since_mark = self._written - self._mark_written - self._dropped
        return max(0, since_mark - self._remaining())

    def stop(self) -> int:
        now = time.perf_counter()
        played = self.played_samples
        self._dropped += self._remaining(now)
        self._end_t = now
        self.stop_calls.append((now, played))
        return played

    @property
    def is_active(self) -> bool:
        return time.perf_counter() < self._end_t

    @property
    def buffered_seconds(self) -> float:
        return max(0.0, self._end_t - time.perf_counter())

    async def wait_until_done(self) -> None:
        while True:
            rem = self._end_t - time.perf_counter()
            if rem <= 0:
                return
            await asyncio.sleep(min(rem, 0.05))


# --------------------------------------------------------------------- Segmenter
class ScriptedSegmenter:
    """Emits ``SpeechStart`` / ``SpeechEnd`` at scheduled wall-clock times.

    ``script`` is a list of ``(start_offset_s, duration_s)`` relative to the first
    ``feed()`` call; ``schedule()`` adds utterances at absolute ``perf_counter``
    times while running (used to place a barge-in relative to agent audio).
    ``pad_s`` is added to every emitted utterance's audio / ``duration_s`` (the real
    segmenter's pre-speech ring buffer + tail) while ``speech_ms`` stays the
    scripted speech length, as ``UtteranceSegmenter`` reports it.
    """

    def __init__(
        self,
        script: list[tuple[float, float]] | None = None,
        sample_rate: int = MIC_SAMPLE_RATE,
        threshold: float = 0.5,
        pad_s: float = 0.0,
    ) -> None:
        self._script = list(script or [])
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.pad_s = pad_s
        self.last_prob = 0.0
        self.speaking = False
        self._t0: float | None = None
        self._pending: list[tuple[float, float]] = []  # absolute (start, duration)
        self._speech_start_t = 0.0
        self._speech_duration = 0.0

    def schedule(self, start_at: float, duration_s: float) -> None:
        """Add an utterance starting at absolute ``perf_counter`` time ``start_at``."""
        self._pending.append((start_at, duration_s))
        self._pending.sort()

    def reset(self) -> None:
        self.speaking = False
        self.last_prob = 0.0

    @property
    def speaking_ms(self) -> float:
        if not self.speaking:
            return 0.0
        return (time.perf_counter() - self._speech_start_t) * 1000.0

    def feed(self, frame: np.ndarray) -> list[SpeechStart | SpeechEnd]:
        now = time.perf_counter()
        if self._t0 is None:
            self._t0 = now
            self._pending.extend((self._t0 + off, dur) for off, dur in self._script)
            self._pending.sort()
        events: list[SpeechStart | SpeechEnd] = []
        if self.speaking:
            self.last_prob = 0.95
            if now >= self._speech_start_t + self._speech_duration:
                self.speaking = False
                self.last_prob = 0.05
                total = self._speech_duration + self.pad_s
                n = int(total * self.sample_rate)
                events.append(
                    SpeechEnd(t=now, pcm=np.zeros(n, dtype=np.int16), duration_s=total, speech_ms=self._speech_duration * 1000.0)
                )
        elif self._pending and now >= self._pending[0][0]:
            _, dur = self._pending.pop(0)
            self.speaking = True
            self.last_prob = 0.95
            self._speech_start_t = now
            self._speech_duration = dur
            events.append(SpeechStart(t=now))
        else:
            self.last_prob = 0.05
        return events


async def silent_frames(
    duration_s: float, frame_ms: int = 20, sample_rate: int = MIC_SAMPLE_RATE
) -> AsyncIterator[np.ndarray]:
    """Real-time stream of silent 20 ms int16 frames (a stand-in for ``Mic.frames()``)."""
    n = int(sample_rate * frame_ms / 1000)
    frame = np.zeros(n, dtype=np.int16)
    t0 = time.perf_counter()
    i = 0
    while True:
        target = t0 + i * frame_ms / 1000
        delay = target - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        if time.perf_counter() - t0 >= duration_s:
            return
        yield frame
        i += 1


# --------------------------------------------------- stand-ins for sibling modules
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n+")


class StandInChunker:
    """Minimal ``eva.llm.chunker.SentenceChunker`` replacement.

    Cuts on sentence terminators; the *first* chunk may also be cut at a comma once
    ``first_chunk_min_chars`` have accumulated so the first TTS request goes out
    early.  Chunks shorter than ``min_chunk_chars`` are merged into the next one.
    """

    def __init__(self, first_chunk_min_chars: int = 14, min_chunk_chars: int = 6) -> None:
        self.first_chunk_min_chars = first_chunk_min_chars
        self.min_chunk_chars = min_chunk_chars
        self._buf = ""
        self._emitted = 0

    def feed(self, delta: str) -> list[str]:
        self._buf += delta
        out: list[str] = []
        search_from = 0  # boundaries before this offset were judged too short to cut at
        while True:
            m = _SENTENCE_END.search(self._buf, search_from)
            cut: int | None = None
            head = ""
            if m:
                cut = m.end()
                head = self._buf[: m.start()]
            elif self._emitted == 0 and len(self._buf) >= self.first_chunk_min_chars:
                k = self._buf.find(", ", max(search_from, self.first_chunk_min_chars - 6))
                if k >= 0:
                    cut = k + 2
                    head = self._buf[: k + 1]
            if cut is None:
                break
            chunk = head.strip()
            if not chunk:
                self._buf = self._buf[cut:]
                continue
            if len(chunk) < self.min_chunk_chars:
                if not self._buf[cut:].strip():
                    break  # too short and nothing after it yet: wait for more text (or flush)
                search_from = cut  # too short: merge with the next sentence
                continue
            self._buf = self._buf[cut:]
            search_from = 0
            out.append(chunk)
            self._emitted += 1
        return out

    def flush(self) -> list[str]:
        rest = self._buf.strip()
        self._buf = ""
        if rest:
            self._emitted += 1
            return [rest]
        return []


_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_TAG_RE = re.compile(r"\[[^\]]{1,30}\]")
_STAGE_RE = re.compile(r"\*[^*\n]{1,60}\*|\([^)\n]{0,40}(?:laughs|sighs|pause|chuckles)[^)\n]{0,40}\)", re.I)
_MD_RE = re.compile(r"[*_`#>]+")
_EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF️]")


def standin_strip_think(text: str) -> str:
    """Remove ``<think>...</think>`` blocks (and a dangling open block)."""
    text = _THINK_RE.sub("", text)
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text.strip()


def standin_clean_for_tts(text: str, keep_audio_tags: bool = False) -> str:
    """Strip markdown, emoji and stage directions; keep ``[tags]`` only if asked."""
    text = standin_strip_think(text)
    text = _STAGE_RE.sub("", text)
    if not keep_audio_tags:
        text = _TAG_RE.sub("", text)
    text = _MD_RE.sub("", text)
    text = _EMOJI_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


async def standin_execute(call: LLMToolCall, tools: list[Tool]) -> str:
    """Run ``call`` against ``tools`` (sync or async ``fn``)."""
    for tool in tools:
        if tool.name == call.name:
            result = tool.fn(**call.arguments)
            if inspect.isawaitable(result):
                result = await result
            return str(result)
    return f"error: unknown tool {call.name!r}"


# ------------------------------------------------------- test / bench harness
# Shared by tests/test_pipeline.py and bench/e2e_sim.py.
class EventLog:
    """Collects pipeline events with timestamps; optional live printing."""

    def __init__(self, verbose: bool = False, player: Any = None) -> None:
        self.events: list[tuple[float, str, dict[str, Any]]] = []
        self.verbose = verbose
        self.player = player
        self.t0 = time.perf_counter()
        self.hooks: list[Callable[[str, dict[str, Any]], None]] = []

    def __call__(self, name: str, data: dict[str, Any]) -> None:
        now = time.perf_counter()
        if name == "barge_in" and self.player is not None:
            data = {**data, "player_active_after_stop": bool(self.player.is_active)}
        self.events.append((now, name, data))
        if self.verbose and name not in ("state",):
            print(f"{now - self.t0:7.3f} {name:18} {json.dumps(data, default=str)[:160]}")
        for h in self.hooks:
            h(name, data)

    def first(self, name: str) -> tuple[float, dict[str, Any]] | None:
        for t, n, d in self.events:
            if n == name:
                return t, d
        return None

    def all(self, name: str) -> list[tuple[float, dict[str, Any]]]:
        return [(t, d) for t, n, d in self.events if n == name]


class Check:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.notes: list[str] = []

    def ok(self, cond: bool, msg: str) -> None:
        if not cond:
            self.failures.append(msg)

    def note(self, msg: str) -> None:
        self.notes.append(msg)


def mock_agent(
    *,
    stt: MockSTT | MockStreamingSTT,
    llm: MockLLM,
    tts: MockTTS,
    player: MockPlayer,
    segmenter: ScriptedSegmenter | None,
    frames: AsyncIterator[np.ndarray] | None,
    settings: PipelineSettings,
    log: EventLog,
    tools: list[Tool] | None = None,
    fillers: list[str] | None = None,
    max_turns: int | None = None,
    pending_events: asyncio.Queue | None = None,
    languages: list[str] | None = None,
    tool_filter: Any = None,
) -> Any:
    from .pipeline import VoiceAgent

    return VoiceAgent(
        stt,
        llm,
        tts,
        "You are Eva, a warm voice companion.",
        tools or [],
        settings,
        frames=frames,
        segmenter=segmenter,
        player=player,
        fillers=fillers,
        on_event=log,
        max_turns=max_turns,
        chunker_factory=lambda: StandInChunker(settings.first_chunk_min_chars, 6),
        sanitizer=standin_clean_for_tts,
        tool_executor=standin_execute,
        pending_events=pending_events if pending_events is not None else asyncio.Queue(),
        languages=languages,
        tool_filter=tool_filter,
    )


def settings_with(**over: Any) -> PipelineSettings:
    return dataclasses.replace(PipelineSettings(), **over)


def turn_rows(turns: list[TurnMetrics]) -> list[dict[str, Any]]:
    rows = []
    for m in turns:
        rows.append(
            {
                "user_text": m.user_text,
                "assistant_text": m.assistant_text,
                "interrupted": m.interrupted,
                "response_latency": None if m.response_latency() is None else round(m.response_latency(), 3),
                **{k: v for k, v in m.breakdown().items()},
                "audio_seconds": None
                if m.audio_started is None or m.audio_finished is None
                else round(m.audio_finished - m.audio_started, 3),
            }
        )
    return rows
