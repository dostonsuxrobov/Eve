"""The conversation loop: VAD events -> STT -> LLM (+tools) -> chunker -> TTS -> player.

State machine
-------------
``LISTENING``  no response in flight. Mic frames feed the segmenter, pending tool
               events (finished timers) are polled between frames.
``THINKING``   a response task is running (STT / LLM / first TTS chunk) but nothing
               is audible yet.
``SPEAKING``   audio (a filler or real speech) is in the player. With
               ``settings.echo_guard`` the VAD threshold is raised by 0.25 here so
               speaker bleed does not look like a barge-in.

Barge-in
--------
A ``SpeechStart`` while THINKING/SPEAKING is a *candidate*. It is confirmed once the
segmenter reports ``barge_in_min_speech_ms`` of continuous speech (a cough never
stops her). On confirmation: ``player.stop()`` (instant), cancel the response task
(which tears down every child task: LLM stream, TTS jobs, writer, filler timer),
map ``played_samples`` onto the per-chunk sample counts we tracked, keep only the
words that were actually heard and store them with `` [interrupted]``.
If nothing was heard yet (interrupted while thinking) the user message is taken
back and merged with the next utterance, so "Hey Eva ... how's it going" becomes
one turn instead of two.

Streaming STT
-------------
If the STT implements :class:`eva.interfaces.StreamingSTT` (``feed`` / ``commit`` /
``discard``, e.g. ``ElevenLabsRealtimeSTT``) every mic frame is streamed to it from
speech onset (a short ring buffer covers the pre-speech context and the
``min_speech_ms`` confirmation delay) and ``commit()`` is called at the endpoint,
which returns in ~0.3 s instead of the ~1.1 s batch round trip.  Utterances that
are not answered directly (a blip below ``barge_in_min_speech_ms``, a queued
utterance while barge-in is off, two utterances merged after a thinking-phase
interruption) are ``discard()``-ed on the stream and transcribed in batch instead.

Empty replies
-------------
Cerebras ``qwen-3.8-27b`` with reasoning disabled occasionally returns an empty
completion (measured 2-6 % on flat-mood histories).  A turn that produced no text
and no tool call is retried once, then once more with a trailing ``"Mm."``
assistant prefill, and finally a fixed spoken fallback so there is never dead air.

Every await in the response path is cancellable; child tasks live in
``_Turn.children`` and are cancelled + awaited in the task's ``finally``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Awaitable, Callable, Protocol

import numpy as np

from .config import PipelineSettings
from .interfaces import (
    LLM,
    MIC_SAMPLE_RATE,
    STT,
    TTS,
    LLMDelta,
    LLMDone,
    LLMToolCall,
    Tool,
    Transcript,
    TurnMetrics,
)

log = logging.getLogger("eva.pipeline")

MAX_TOOL_ROUNDS = 3
HISTORY_LIMIT = 30
ECHO_GUARD_BOOST = 0.25
PLAYER_LOOKAHEAD_S = 1.5  # never queue more than this much audio in the player
MIN_WRITE_MS = 60  # coalesce tiny TTS chunks (ElevenLabs http sends ~21 ms pieces) before writing
MIN_TRANSCRIPT_CHARS = 2
CARRY_GAP_S = 0.3  # silence inserted between two merged utterances
INTERRUPTED_MARK = " [interrupted]"
STREAM_RING_EXTRA_MS = 200  # ring buffer slack on top of prespeech + min_speech
STREAM_FEED_TIMEOUT_S = 0.25  # a feed() slower than this (socket reconnect) breaks the stream for the turn
STT_RACE_AFTER_S = 1.5  # a streaming commit() slower than this is raced against a batch request
FILLER_TTS_GRACE_S = 0.6  # extra wait before a filler when a TTS request is already running
EMPTY_REPLY_PREFILL = "Mm."
EMPTY_REPLY_TEXT = "Hm, sorry, I lost my train of thought there. Say that again?"
LLM_FAILURE_TEXT = "Sorry, I'm having trouble thinking right now. Give me a moment and try again."


class State(str, Enum):
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


EventHandler = Callable[[str, dict[str, Any]], None]
ToolExecutor = Callable[[LLMToolCall, list[Tool]], Awaitable[str]]
Sanitizer = Callable[[str, bool], str]


class Chunker(Protocol):
    def feed(self, delta: str) -> list[str]: ...

    def flush(self) -> list[str]: ...


class SegmenterLike(Protocol):
    """What the pipeline needs from ``eva.audio.vad.UtteranceSegmenter``."""

    speaking: bool
    speaking_ms: float
    threshold: float

    def feed(self, frame: np.ndarray) -> list[Any]: ...

    def reset(self) -> None: ...


class PlayerLike(Protocol):
    """What the pipeline needs from ``eva.audio.player.Player``."""

    played_samples: int
    is_active: bool
    buffered_seconds: float

    def write(self, pcm: bytes) -> None: ...

    def stop(self) -> int: ...

    def mark(self) -> None: ...

    async def wait_until_done(self) -> None: ...


# --------------------------------------------------------- sibling-module resolvers
def _resolve_chunker_factory(settings: PipelineSettings) -> Callable[[], Chunker]:
    try:
        from .llm.chunker import SentenceChunker  # type: ignore
    except ImportError:
        log.warning("eva.llm.chunker not available; using StandInChunker from eva.mocks")
        from .mocks import StandInChunker as SentenceChunker  # type: ignore
    return lambda: SentenceChunker(first_chunk_min_chars=settings.first_chunk_min_chars, min_chunk_chars=6)


def _resolve_sanitizer() -> Sanitizer:
    try:
        from .llm.sanitize import clean_for_tts  # type: ignore
    except ImportError:
        log.warning("eva.llm.sanitize not available; using standin_clean_for_tts from eva.mocks")
        from .mocks import standin_clean_for_tts as clean_for_tts  # type: ignore
    return clean_for_tts


def _resolve_tools_runtime() -> tuple[ToolExecutor, "asyncio.Queue[dict[str, Any]]"]:
    try:
        from . import tools as tools_mod  # type: ignore

        return tools_mod.execute, tools_mod.pending_events
    except ImportError:
        log.warning("eva.tools not available; using standin_execute and an empty event queue")
        from .mocks import standin_execute

        return standin_execute, asyncio.Queue()


# ------------------------------------------------------------------ helpers
def _partial_suffix_len(s: str, tag: str) -> int:
    """Length of the longest proper prefix of ``tag`` that ``s`` ends with."""
    for n in range(min(len(tag) - 1, len(s)), 0, -1):
        if s.endswith(tag[:n]):
            return n
    return 0


class _ThinkFilter:
    """Streaming filter that drops ``<think>...</think>`` blocks across delta boundaries."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self._buf = ""
        self._in = False

    def feed(self, delta: str) -> str:
        self._buf += delta
        out: list[str] = []
        while self._buf:
            tag = self.CLOSE if self._in else self.OPEN
            i = self._buf.find(tag)
            if i >= 0:
                if not self._in:
                    out.append(self._buf[:i])
                self._buf = self._buf[i + len(tag) :]
                self._in = not self._in
                continue
            keep = _partial_suffix_len(self._buf, tag)
            if not self._in:
                out.append(self._buf[: len(self._buf) - keep])
            self._buf = self._buf[len(self._buf) - keep :]
            break
        return "".join(out)

    def flush(self) -> str:
        out = "" if self._in else self._buf
        self._buf = ""
        return out


_JSON_OBJ_RE = re.compile(r"\{\s*\"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}")


def _recover_tool_calls(text: str, tools: list[Tool], seq: list[int]) -> tuple[str, list[LLMToolCall]]:
    """Turn tool calls that a model leaked as JSON *text* into real ``LLMToolCall``s.

    Seen from gpt-oss-120b on Cerebras after a tool result: ``{"text":"Call mom
    later tonight"}`` or ``{"function":"remember_note","arguments":{...}}`` inside the
    content stream.  Explicit forms carry the tool name; a bare argument object is
    accepted only when its keys match exactly one tool's required parameters.
    Returns the text with the JSON removed and the recovered calls.
    """
    calls: list[LLMToolCall] = []
    if "{" not in text:
        return text, calls
    by_name = {t.name: t for t in tools}

    def _sub(m: re.Match[str]) -> str:
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            return m.group(0)
        if not isinstance(obj, dict):
            return m.group(0)
        name = obj.get("function") or obj.get("name") or obj.get("tool")
        args = obj.get("arguments", obj.get("parameters"))
        if isinstance(name, dict):  # {"function": {"name": ..., "arguments": ...}}
            args = name.get("arguments", args)
            name = name.get("name")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = None
        if not (isinstance(name, str) and name in by_name):
            keys = set(obj)
            matches = [t for t in tools if keys and keys == set(t.parameters.get("required", []))]
            if len(matches) != 1:
                return " "  # not a tool call we can vouch for: never speak JSON
            name, args = matches[0].name, obj
        if not isinstance(args, dict):
            args = {}
        seq[0] += 1
        calls.append(LLMToolCall(id=f"recovered_{seq[0]}", name=name, arguments=args))
        return " "

    return _JSON_OBJ_RE.sub(_sub, text), calls


def _as_sentence(text: str) -> str:
    """"one sec" -> "One sec." so hints read and are spoken like a sentence."""
    text = " ".join(text.split())
    if not text:
        return text
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?…" else text + "."


def _is_speech_end(ev: Any) -> bool:
    return hasattr(ev, "pcm")


def _now() -> float:
    return time.perf_counter()


@dataclass
class _ChunkJob:
    """One TTS-sized piece of the reply and its audio accounting."""

    index: int
    round: int
    raw: str  # text as the LLM produced it (for the transcript)
    text: str  # sanitized text sent to TTS ("" -> nothing to say)
    is_hint: bool = False
    samples: int = 0  # samples written to the player so far
    audio: "asyncio.Queue[bytes | None]" = field(default_factory=asyncio.Queue)
    play_started: asyncio.Event = field(default_factory=asyncio.Event)
    error: BaseException | None = None

    @property
    def silent(self) -> bool:
        return not self.text


@dataclass
class _Turn:
    """Mutable state of one response turn (voice, typed or system event)."""

    metrics: TurnMetrics
    kind: str = "voice"
    pcm: np.ndarray | None = None
    user_text: str | None = None
    use_stream: bool = False  # transcript comes from stt.commit() (audio already streamed)
    stt_inflight: bool = False  # transcribe()/commit() is being awaited right now
    started_at: float = 0.0
    chunks: list[_ChunkJob] = field(default_factory=list)
    filler_samples: int = 0
    marked: bool = False
    audio_started: bool = False
    user_appended: bool = False
    hint_spoken: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    rounds_in_history: int = 0  # assistant tool-call messages already appended
    recovered_seq: list[int] = field(default_factory=lambda: [0])  # ids for recovered tool calls
    finished: bool = False
    error: str | None = None
    task: asyncio.Task[None] | None = None
    children: set[asyncio.Task[Any]] = field(default_factory=set)


class VoiceAgent:
    """The conversation loop.  See the module docstring for the state machine.

    ``segmenter`` and ``player`` are duck-typed against the ``eva.audio`` API so
    ``eva.mocks`` doubles can be injected.  ``fillers`` are pre-rendered by
    ``prepare()`` and played when nothing is audible ``filler_after_ms`` after the
    user stops; ``tool_hints`` are spoken while a tool without its own
    ``spoken_hint`` runs.  ``chunker_factory`` / ``sanitizer`` /
    ``tool_executor`` / ``pending_events`` default to the sibling modules and fall
    back to stand-ins when those are missing.
    """

    def __init__(
        self,
        stt: STT,
        llm: LLM,
        tts: TTS,
        system_prompt: str,
        tools: list[Tool],
        settings: PipelineSettings,
        *,
        frames: AsyncIterator[np.ndarray] | None,
        segmenter: SegmenterLike | None,
        player: PlayerLike,
        fillers: list[str] | None = None,
        on_event: EventHandler | None = None,
        max_turns: int | None = None,
        tool_hints: list[str] | None = None,
        chunker_factory: Callable[[], Chunker] | None = None,
        sanitizer: Sanitizer | None = None,
        tool_executor: ToolExecutor | None = None,
        pending_events: "asyncio.Queue[dict[str, Any]] | None" = None,
    ) -> None:
        self.stt, self.llm, self.tts = stt, llm, tts
        self.system_prompt = system_prompt
        self.tools = list(tools)
        self.settings = settings
        self.frames = frames
        self.segmenter = segmenter
        self.player = player
        self.on_event = on_event
        self.max_turns = max_turns

        self.chunker_factory = chunker_factory or _resolve_chunker_factory(settings)
        self.sanitizer: Sanitizer = sanitizer or _resolve_sanitizer()
        if tool_executor is None or pending_events is None:
            ex, q = _resolve_tools_runtime()
            tool_executor = tool_executor or ex
            pending_events = pending_events if pending_events is not None else q
        self.tool_executor: ToolExecutor = tool_executor
        self.pending_events: "asyncio.Queue[dict[str, Any]]" = pending_events

        self.messages: list[dict[str, Any]] = []  # history without the system prompt
        self.turns: list[TurnMetrics] = []
        self.state = State.LISTENING
        self._fillers_text = [f for f in (fillers or []) if f and f.strip()]
        self._tool_hints = [h for h in (tool_hints or []) if h and h.strip()]
        self._tool_hint_i = 0
        self._fillers_audio: list[bytes] = []
        self._filler_i = 0
        self._prepared = False
        self._turn_seq = 0
        self._turns_done = 0
        self._response: _Turn | None = None
        self._barge_candidate: float | None = None
        self._last_speech_start: float | None = None
        self._base_threshold: float | None = None
        self._carry_pcm: np.ndarray | None = None
        self._carry_text: str | None = None
        self._pending_utterance: Any | None = None
        self._stopping = False

        # streaming STT (feed while the user talks, commit at the endpoint)
        self.stt_streaming = all(callable(getattr(stt, m, None)) for m in ("feed", "commit", "discard"))
        ring_ms = settings.prespeech_buffer_ms + settings.min_speech_ms + STREAM_RING_EXTRA_MS
        self._recent: deque[np.ndarray] = deque(maxlen=max(1, math.ceil(ring_ms / 20)))
        self._stream_open = False  # frames are being fed to the STT right now
        self._stream_broken = False  # feed() failed for the utterance in progress -> batch
        self.stream_turns = 0  # turns transcribed through the streaming path (for reports)

    # ------------------------------------------------------------------ public
    @property
    def is_responding(self) -> bool:
        return self._response is not None

    async def prepare(self) -> None:
        """Pre-render the filler phrases in the active voice (idempotent)."""
        if self._prepared:
            return
        self._prepared = True
        for text in self._fillers_text:
            try:
                audio = await self._render_to_bytes(text)
            except Exception as e:  # a broken filler must not stop startup
                log.warning("filler %r failed to render: %s", text, e)
                continue
            if audio:
                self._fillers_audio.append(audio)
        self._emit("ready", {"fillers": len(self._fillers_audio)})

    async def run(self) -> list[TurnMetrics]:
        """Main loop: consume mic frames until they end, ``max_turns`` is reached or cancelled."""
        if self.frames is None or self.segmenter is None:
            raise RuntimeError("run() needs frames and a segmenter; use say() for text mode")
        await self.prepare()
        self._emit("listening", {})
        try:
            async for frame in self.frames:
                if self.state is State.LISTENING:
                    if self._pending_utterance is not None:
                        ev, self._pending_utterance = self._pending_utterance, None
                        await self._start_voice_turn(ev)
                    else:
                        await self.poll_pending_events()
                if self.stt_streaming:
                    self._recent.append(frame)
                fed = False
                for ev in self.segmenter.feed(frame):
                    fed |= await self._on_vad_event(ev)
                if self._stream_open and not fed:
                    await self._stt_feed(frame)
                if self._barge_candidate is not None:
                    await self._check_barge_in()
                if self._should_stop():
                    break
            turn = self._response
            if turn is not None and turn.task is not None and not turn.task.done():
                await asyncio.wait({turn.task})
        finally:
            await self._shutdown()
        return self.turns

    async def say(self, text: str) -> TurnMetrics:
        """Run one response turn for typed input (still speaks).  Waits for any
        in-flight turn first, then returns this turn's metrics."""
        await self.prepare()
        prev = self._response
        if prev is not None and prev.task is not None and not prev.task.done():
            await asyncio.wait({prev.task})
        metrics = TurnMetrics(speech_end=_now(), user_text=text)
        turn = _Turn(metrics=metrics, kind="text", user_text=text)
        self._launch(turn)
        assert turn.task is not None
        await asyncio.wait({turn.task})
        return metrics

    async def interrupt(self, reason: str = "manual") -> bool:
        """Stop the current response immediately (what a confirmed barge-in does)."""
        if self._response is None:
            return False
        await self._interrupt(reason)
        return True

    async def poll_pending_events(self) -> bool:
        """Start a response turn for one pending tool event (e.g. a finished timer)
        if nobody is talking.  Returns True if a turn was started."""
        if self._response is not None or self._stopping:
            return False
        if self.segmenter is not None and self.segmenter.speaking:
            return False
        try:
            ev = self.pending_events.get_nowait()
        except asyncio.QueueEmpty:
            return False
        text = self._event_to_user_text(ev)
        if not text:
            return False
        self._emit("pending_event", {"event": ev})
        turn = _Turn(metrics=TurnMetrics(speech_end=_now(), user_text=text), kind="event", user_text=text)
        self._launch(turn)
        return True

    # ----------------------------------------------------------- VAD handling
    async def _on_vad_event(self, ev: Any) -> bool:
        """Handle one segmenter event.  Returns True if the current frame was already
        fed to the streaming STT (as part of the ring buffer on SpeechStart)."""
        if _is_speech_end(ev):
            self._emit("speech_end", {"duration_s": round(float(ev.duration_s), 3)})
            streamed = self._stream_open and not self._stream_broken
            self._stream_open = False
            self._stream_broken = False
            self._recent.clear()  # never feed the tail of this utterance into the next one
            if self._response is None:
                await self._start_voice_turn(ev, streamed=streamed)
            elif self._barge_candidate is not None:
                self._barge_candidate = None
                if ev.duration_s * 1000 >= self.settings.barge_in_min_speech_ms:
                    await self._interrupt("barge-in", t_trigger=ev.t)
                    await self._start_voice_turn(ev, streamed=streamed)
                else:
                    self._emit("barge_in_ignored", {"duration_s": round(float(ev.duration_s), 3)})
                    await self._stt_discard(keep_audio=False)
            else:
                # response in flight, barge-in disabled: answer it after this turn (in batch)
                self._pending_utterance = ev
                self._emit("utterance_queued", {"duration_s": round(float(ev.duration_s), 3)})
                await self._stt_discard(keep_audio=False)
            return False
        # SpeechStart
        self._last_speech_start = float(ev.t)
        self._emit("speech_start", {"state": self.state.value})
        if self._response is not None and self.settings.barge_in:
            self._barge_candidate = float(ev.t)
            self._emit("barge_in_candidate", {"state": self.state.value})
        if self.stt_streaming and not self._stopping:
            # stream the pre-speech context + the frames that confirmed the onset
            self._stream_open = True
            self._stream_broken = False
            ring = list(self._recent)
            if ring:
                await self._stt_feed(np.concatenate(ring))
            return True
        return False

    async def _stt_feed(self, pcm: np.ndarray) -> None:
        """Feed audio to the streaming STT; a failure switches this utterance to batch."""
        if self._stream_broken:
            return
        try:
            await asyncio.wait_for(self.stt.feed(pcm), timeout=STREAM_FEED_TIMEOUT_S)  # type: ignore[attr-defined]
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._stream_broken = True
            log.warning("streaming stt feed failed (%s); this utterance will use batch", e)
            self._emit("stt_stream_broken", {"error": repr(e)})

    async def _stt_discard(self, *, keep_audio: bool) -> None:
        if not self.stt_streaming:
            return
        try:
            await self.stt.discard(keep_audio=keep_audio)  # type: ignore[attr-defined]
        except Exception as e:  # never let STT housekeeping kill the loop
            log.warning("streaming stt discard failed: %s", e)

    async def _check_barge_in(self) -> None:
        if self._response is None or self.segmenter is None:
            self._barge_candidate = None
            return
        if not self.segmenter.speaking:
            return
        speaking_ms = self.segmenter.speaking_ms
        min_ms = self.settings.barge_in_min_speech_ms
        if speaking_ms >= min_ms:
            # the moment the user had spoken exactly min_ms (segmenters may count speech
            # from before they emit SpeechStart, so derive it from speaking_ms itself)
            t_trigger = _now() - (speaking_ms - min_ms) / 1000.0
            self._barge_candidate = None
            await self._interrupt("barge-in", t_trigger=t_trigger)

    async def _start_voice_turn(self, ev: Any, *, streamed: bool = False) -> None:
        pcm = np.asarray(ev.pcm, dtype=np.int16)
        use_stream = streamed
        if self._carry_pcm is not None:
            gap = np.zeros(int(CARRY_GAP_S * MIC_SAMPLE_RATE), dtype=np.int16)
            pcm = np.concatenate([self._carry_pcm, gap, pcm])
            self._carry_pcm = None
            self._emit("utterance_merged", {"how": "audio"})
            use_stream = False  # the merged audio goes through batch as a whole
        if streamed and not use_stream:
            await self._stt_discard(keep_audio=False)
        metrics = TurnMetrics(speech_start=self._last_speech_start, speech_end=float(ev.t))
        self._launch(_Turn(metrics=metrics, kind="voice", pcm=pcm, use_stream=use_stream))

    # ------------------------------------------------------------ turn control
    def _launch(self, turn: _Turn) -> None:
        self._turn_seq += 1
        turn.started_at = _now()
        self._response = turn
        self._set_state(State.THINKING)
        turn.task = asyncio.create_task(self._run_turn(turn), name=f"eva-turn-{self._turn_seq}")

    def _spawn(self, turn: _Turn, coro: Awaitable[Any], name: str) -> asyncio.Task[Any]:
        task = asyncio.ensure_future(coro)
        task.set_name(name)
        turn.children.add(task)
        task.add_done_callback(turn.children.discard)
        return task

    async def _run_turn(self, turn: _Turn) -> None:
        try:
            self._spawn(turn, self._filler_timer(turn), "eva-filler")
            if turn.pcm is not None:
                text = await self._transcribe(turn)
                if not text:
                    if self._carry_text is None:
                        return  # nothing to answer
                    text = ""  # re-answer the carried question
            else:
                text = turn.user_text or ""
            text = self._append_user(text)
            turn.user_appended = True
            turn.metrics.user_text = text
            await self._generate_and_speak(turn)
            self._finish_turn(turn, interrupted=False)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("turn failed")
            self._emit("error", {"where": "turn", "error": repr(e)})
            self._finish_turn(turn, interrupted=False, error=repr(e))
        finally:
            for child in list(turn.children):
                child.cancel()
            if turn.children:
                await asyncio.gather(*turn.children, return_exceptions=True)
            if self._response is turn:
                self._response = None
            self._set_state(State.LISTENING)

    async def _transcribe(self, turn: _Turn) -> str:
        assert turn.pcm is not None
        turn.stt_inflight = True
        try:
            if turn.use_stream:
                tr = await self._commit_with_race(turn)
            else:
                tr = await self._transcribe_batch(turn.pcm)
        finally:
            turn.stt_inflight = False
        turn.metrics.stt_done = _now()
        text = (tr.text or "").strip()
        self._emit(
            "stt",
            {
                "text": text,
                "latency_s": round(tr.latency_s, 3),
                "samples": int(len(turn.pcm)),
                "mode": "stream" if turn.use_stream else "batch",
                "fallback": tr.meta.get("fallback"),
            },
        )
        if len(text) < MIN_TRANSCRIPT_CHARS:
            self._emit("stt_empty", {"text": text})
            return ""
        return text

    async def _commit_with_race(self, turn: _Turn) -> Transcript:
        """``stt.commit()`` with a batch request racing it after ``STT_RACE_AFTER_S``.

        The realtime commit returns in ~0.3 s typically but has a server-side tail of
        2-4 s (measured 2 of 3 turns in one run).  Once it is late, a batch request on
        the utterance audio is started as well and the first transcript wins; the
        loser is cancelled (a cancelled commit drops its socket via ``discard``).
        A commit that fails outright falls back to batch too.
        """
        assert turn.pcm is not None
        commit = asyncio.ensure_future(self.stt.commit())  # type: ignore[attr-defined]
        batch: asyncio.Task[Transcript] | None = None
        try:
            done, _ = await asyncio.wait({commit}, timeout=STT_RACE_AFTER_S)
            if not done:
                self._emit("stt_race", {"after_s": STT_RACE_AFTER_S})
                batch = asyncio.ensure_future(self._transcribe_batch(turn.pcm))
                done, _ = await asyncio.wait({commit, batch}, return_when=asyncio.FIRST_COMPLETED)
                if commit in done and commit.exception() is not None and batch not in done:
                    done, _ = await asyncio.wait({batch})
            if commit.done() and commit.exception() is None:
                tr = commit.result()
                self.stream_turns += 1
                tr.meta.setdefault("won", "stream")
                return tr
            if batch is not None and batch.done() and batch.exception() is None:
                tr = batch.result()
                tr.meta["won"] = "batch"
                self._emit("stt_race_won", {"by": "batch", "latency_s": round(tr.latency_s, 3)})
                return tr
            exc = commit.exception() if commit.done() else None
            log.warning("streaming stt commit failed (%s); falling back to batch", exc)
            self._emit("error", {"where": "stt_stream", "error": repr(exc)})
            if batch is not None:
                return await batch
            return await self._transcribe_batch(turn.pcm)
        finally:
            for task in (commit, batch):
                if task is not None and not task.done():
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass
            if commit.cancelled():
                await self._stt_discard(keep_audio=self._stream_open)

    async def _transcribe_batch(self, pcm: np.ndarray) -> Transcript:
        """Whole-utterance transcription; a streaming STT may offer a dedicated batch entry point."""
        fn = getattr(self.stt, "transcribe_batch", None) if self.stt_streaming else None
        if fn is not None:
            return await fn(pcm, MIC_SAMPLE_RATE)
        return await self.stt.transcribe(pcm, MIC_SAMPLE_RATE)

    async def _generate_and_speak(self, turn: _Turn) -> None:
        schemas = [t.openai_schema() for t in self.tools] or None
        jobs: "asyncio.Queue[_ChunkJob | None]" = asyncio.Queue()
        writer = self._spawn(turn, self._writer_loop(turn, jobs), "eva-writer")
        final_text = ""
        last_results: list[str] = []
        round_no = 0
        empty_attempts = 0
        prefill: str | None = None
        while True:
            use_tools = schemas if round_no < MAX_TOOL_ROUNDS else None
            try:
                round_text, tool_calls = await self._llm_round(turn, jobs, round_no, use_tools, prefill=prefill)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # the request could not even start (auth, network, HTTP error after retry)
                log.warning("llm failed: %s", e)
                self._emit("error", {"where": "llm", "error": repr(e)})
                if not turn.chunks:
                    final_text = LLM_FAILURE_TEXT
                    self._enqueue(turn, jobs, round_no, final_text)
                turn.error = repr(e)
                break
            prefill = None
            if not tool_calls and not round_text.strip() and not turn.chunks and not last_results:
                # nothing said, nothing called: Cerebras qwen (reasoning off) does this now and then
                empty_attempts += 1
                if empty_attempts == 1:
                    self._emit("llm_empty", {"round": round_no, "action": "retry"})
                    continue
                if empty_attempts == 2:
                    self._emit("llm_empty", {"round": round_no, "action": "prefill"})
                    prefill = EMPTY_REPLY_PREFILL
                    continue
                self._emit("llm_empty", {"round": round_no, "action": "fallback"})
                round_text = EMPTY_REPLY_TEXT
                self._enqueue(turn, jobs, round_no, round_text)
            if not tool_calls:
                final_text = round_text
                break
            if not turn.chunks:  # nothing spoken yet: cover the wait
                hint = self._hint_for(tool_calls)
                if hint:
                    self._enqueue(turn, jobs, round_no, hint, is_hint=True)
                    turn.hint_spoken = hint
                    round_text = hint
            self.messages.append(
                {
                    "role": "assistant",
                    "content": round_text,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }
                        for tc in tool_calls
                    ],
                }
            )
            turn.rounds_in_history += 1
            last_results = []
            for tc in tool_calls:
                result = await self._execute_tool(turn, tc)
                last_results.append(result)
                self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            round_no += 1
        if not final_text and last_results:
            # the model went quiet after the tool: read the result out instead of silence
            final_text = last_results[-1]
            self._enqueue(turn, jobs, round_no, final_text)
        jobs.put_nowait(None)
        await writer
        await self.player.wait_until_done()
        turn.metrics.audio_finished = _now()
        if final_text:
            self.messages.append({"role": "assistant", "content": final_text})
            self._trim_history()

    async def _llm_round(
        self,
        turn: _Turn,
        jobs: "asyncio.Queue[_ChunkJob | None]",
        round_no: int,
        use_tools: list[dict[str, Any]] | None,
        *,
        prefill: str | None = None,
    ) -> tuple[str, list[LLMToolCall]]:
        """One LLM request: stream deltas into the chunker, collect tool calls.

        ``prefill`` is sent as a trailing assistant message (Cerebras continues it;
        a trailing *system* message is rejected with HTTP 400) and spoken in front
        of the continuation, but only if the model actually continues.
        """
        chunker = self.chunker_factory()
        think = _ThinkFilter()
        tool_calls: list[LLMToolCall] = []
        raw_parts: list[str] = []
        m = turn.metrics
        prefill_pending = prefill
        seen: set[str] = {j.raw.strip().lower() for j in turn.chunks if j.raw}

        def accept(chunk: str) -> None:
            chunk, recovered = _recover_tool_calls(chunk, self.tools, turn.recovered_seq)
            if recovered:
                self._emit("tool_call_recovered", {"names": [c.name for c in recovered]})
                tool_calls.extend(recovered)
            chunk = chunk.strip()
            if not chunk:
                return
            key = chunk.lower()
            if key in seen:  # gpt-oss repeats whole sentences after tool results
                self._emit("chunk_dropped", {"reason": "duplicate", "text": chunk})
                return
            seen.add(key)
            raw_parts.append(chunk)
            self._enqueue(turn, jobs, round_no, chunk)

        def push(text: str) -> None:
            nonlocal prefill_pending
            if prefill_pending:
                text, prefill_pending = prefill_pending + " " + text.lstrip(), None
            for c in chunker.feed(text):
                accept(c)

        messages = self._messages_for_llm()
        if prefill:
            messages.append({"role": "assistant", "content": prefill})
        stream = self.llm.stream(messages, tools=use_tools)
        try:
            async for ev in stream:
                if isinstance(ev, LLMDelta):
                    text = think.feed(ev.text)
                    if not text:
                        continue
                    if m.llm_first_token is None:
                        m.llm_first_token = _now()
                        self._emit("llm_first_token", {"round": round_no})
                    push(text)
                elif isinstance(ev, LLMToolCall):
                    if m.llm_first_token is None:
                        m.llm_first_token = _now()
                    tool_calls.append(ev)
                elif isinstance(ev, LLMDone):
                    self._emit("llm_done", {"round": round_no, "finish": ev.finish_reason, "usage": ev.usage})
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:  # pragma: no cover
                    pass
        tail = think.flush()
        if tail:
            push(tail)
        for c in chunker.flush():
            accept(c)
        return " ".join(p.strip() for p in raw_parts if p.strip()), tool_calls

    def _enqueue(
        self, turn: _Turn, jobs: "asyncio.Queue[_ChunkJob | None]", round_no: int, raw: str, *, is_hint: bool = False
    ) -> None:
        text = self.sanitizer(raw, self.tts.supports_audio_tags).strip()
        job = _ChunkJob(index=len(turn.chunks), round=round_no, raw=raw.strip(), text=text, is_hint=is_hint)
        turn.chunks.append(job)
        if job.silent:
            job.audio.put_nowait(None)
        else:
            self._spawn(turn, self._synth(turn, job), f"eva-tts-{job.index}")
        jobs.put_nowait(job)

    async def _synth(self, turn: _Turn, job: _ChunkJob) -> None:
        """Synthesize one chunk, at most ``tts_parallelism`` chunks ahead of playback."""
        gate = job.index - max(1, self.settings.tts_parallelism)
        min_bytes = int(self.tts.sample_rate * 2 * MIN_WRITE_MS / 1000)
        buf = bytearray()
        try:
            if gate >= 0:
                await turn.chunks[gate].play_started.wait()
            async for b in self.tts.synthesize(job.text):
                if not b:
                    continue
                if turn.metrics.tts_first_audio is None:
                    turn.metrics.tts_first_audio = _now()
                    self._emit("tts_first_audio", {"chunk": job.index})
                buf += b
                if len(buf) >= min_bytes:
                    job.audio.put_nowait(bytes(buf))
                    buf.clear()
            if buf:
                job.audio.put_nowait(bytes(buf))
                buf.clear()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            job.error = e
            if buf:
                job.audio.put_nowait(bytes(buf))
            log.warning("tts failed on %r: %s", job.text, e)
            self._emit("error", {"where": "tts", "error": repr(e), "chunk": job.index})
        finally:
            job.audio.put_nowait(None)

    async def _writer_loop(self, turn: _Turn, jobs: "asyncio.Queue[_ChunkJob | None]") -> None:
        """Strictly ordered playback: chunk i is fully written before chunk i+1 starts."""
        while True:
            job = await jobs.get()
            if job is None:
                return
            while True:
                b = await job.audio.get()
                if b is None:
                    break
                while self.player.buffered_seconds > PLAYER_LOOKAHEAD_S:
                    await asyncio.sleep(0.02)
                self._write_real(turn, job, b)
                if not job.play_started.is_set():
                    job.play_started.set()
            if not job.play_started.is_set():
                job.play_started.set()

    def _write_real(self, turn: _Turn, job: _ChunkJob, b: bytes) -> None:
        if not turn.marked:
            self.player.mark()
            turn.marked = True
        if not turn.audio_started:
            turn.audio_started = True
            turn.metrics.audio_started = _now()
            self._set_state(State.SPEAKING)
            self._emit(
                "audio_start",
                {
                    "latency_s": turn.metrics.response_latency(),
                    "after_filler": turn.filler_samples > 0,
                    "chunk": job.index,
                },
            )
        self.player.write(b)
        job.samples += len(b) // 2

    async def _filler_timer(self, turn: _Turn) -> None:
        if self.settings.filler_after_ms <= 0 or not self._fillers_audio:
            return
        base = turn.metrics.speech_end if turn.metrics.speech_end is not None else turn.started_at
        delay = base + self.settings.filler_after_ms / 1000.0 - _now()
        if delay > 0:
            await asyncio.sleep(delay)
        if turn.audio_started or turn.finished:
            return
        # If the LLM is already producing text (a TTS request follows within a chunk)
        # the real audio is at most a TTFA away (~0.2-0.35 s) while a filler costs
        # ~0.5-1.3 s of extra wait, so hold off a little longer; only a stalled LLM /
        # TTS gets the filler after the grace period.
        deadline = _now() + FILLER_TTS_GRACE_S
        while (turn.metrics.llm_first_token is not None or any(not j.silent for j in turn.chunks)) and _now() < deadline:
            await asyncio.sleep(0.05)
            if turn.audio_started or turn.finished:
                return
        audio = self._fillers_audio[self._filler_i % len(self._fillers_audio)]
        self._filler_i += 1
        if not turn.marked:
            self.player.mark()
            turn.marked = True
        self.player.write(audio)
        n = len(audio) // 2
        turn.filler_samples += n
        self._set_state(State.SPEAKING)
        self._emit("filler", {"index": self._filler_i - 1, "seconds": round(n / self.tts.sample_rate, 3), "after_s": round(_now() - base, 3)})

    def _hint_for(self, tool_calls: list[LLMToolCall]) -> str | None:
        """What to say while a tool runs: the tool's own hint, else a persona tool
        hint ("one sec"), else a filler phrase."""
        by_name = {t.name: t for t in self.tools}
        for tc in tool_calls:
            tool = by_name.get(tc.name)
            if tool is not None and tool.spoken_hint:
                return _as_sentence(tool.spoken_hint)
        if self._tool_hints:
            text = self._tool_hints[self._tool_hint_i % len(self._tool_hints)]
            self._tool_hint_i += 1
            return _as_sentence(text)
        if self._fillers_text:
            text = self._fillers_text[self._filler_i % len(self._fillers_text)]
            self._filler_i += 1
            return _as_sentence(text)
        return None

    async def _execute_tool(self, turn: _Turn, tc: LLMToolCall) -> str:
        self._emit("tool_call", {"name": tc.name, "arguments": tc.arguments})
        t0 = _now()
        try:
            result = str(await self.tool_executor(tc, self.tools))
        except Exception as e:
            log.warning("tool %s failed: %s", tc.name, e)
            result = f"error: {e}"
        turn.tool_calls.append({"name": tc.name, "arguments": tc.arguments, "result": result})
        self._emit("tool_result", {"name": tc.name, "result": result, "seconds": round(_now() - t0, 3)})
        return result

    # ------------------------------------------------------------- interrupts
    async def _interrupt(self, reason: str, t_trigger: float | None = None) -> None:
        turn = self._response
        if turn is None:
            return
        t0 = _now()
        played = self.player.stop()
        if not turn.marked:
            played = 0
        t_stopped = _now()
        stt_was_inflight = turn.stt_inflight and turn.use_stream
        if turn.task is not None and not turn.task.done():
            turn.task.cancel()
            await asyncio.wait({turn.task})
        if stt_was_inflight:
            # the cancelled commit() left the server with an uncommitted segment: drop that
            # socket but keep (and re-send) whatever the user has said since
            await self._stt_discard(keep_audio=self._stream_open)
        turn.metrics.audio_finished = t_stopped
        self._emit(
            "barge_in",
            {
                "reason": reason,
                "state_before": "speaking" if turn.audio_started or turn.filler_samples else "thinking",
                "played_samples": played,
                "played_s": round(played / self.tts.sample_rate, 3),
                "stop_ms": round((t_stopped - t0) * 1000, 2),
                "reaction_ms": None if t_trigger is None else round((t_stopped - t_trigger) * 1000, 1),
                "player_active": bool(self.player.is_active),
            },
        )
        self._finish_turn(turn, interrupted=True, played=played)

    def _heard_text(self, turn: _Turn, played_samples: int) -> tuple[str, str]:
        """Map played samples onto chunk boundaries -> (all heard words, heard words
        not yet stored in history, i.e. from LLM rounds after the last tool round)."""
        real_played = max(0, played_samples - turn.filler_samples)
        cum = 0
        heard: list[tuple[int, list[str]]] = []
        for job in turn.chunks:
            words = job.raw.split()
            if job.silent:
                heard.append((job.round, words))
                continue
            if job.samples == 0:
                break
            if cum + job.samples <= real_played:
                heard.append((job.round, words))
                cum += job.samples
                continue
            frac = (real_played - cum) / job.samples
            heard.append((job.round, words[: int(round(frac * len(words)))]))
            break
        all_words = [w for _, ws in heard for w in ws]
        # rounds < rounds_in_history are already stored as tool-call messages
        new_words = [w for r, ws in heard if r >= turn.rounds_in_history for w in ws]
        return " ".join(all_words), " ".join(new_words)

    def _finish_turn(self, turn: _Turn, *, interrupted: bool, played: int = 0, error: str | None = None) -> None:
        if turn.finished:
            return
        turn.finished = True
        m = turn.metrics
        if m.audio_finished is None:
            m.audio_finished = _now()
        m.interrupted = interrupted
        error = error or turn.error
        turn.error = error
        if interrupted:
            heard_all, heard_last = self._heard_text(turn, played)
            m.assistant_text = heard_all
            if heard_all.strip() or turn.rounds_in_history:
                self.messages.append({"role": "assistant", "content": (heard_last + INTERRUPTED_MARK).strip()})
                self._trim_history()
            else:
                # nothing was heard: this was not a turn. Take the question back so it
                # can be merged with the next utterance (text if STT finished, else audio).
                if turn.user_appended:
                    self._carry_text = self._pop_last_user()
                    self._emit("utterance_carried", {"text": self._carry_text})
                elif turn.pcm is not None:
                    self._carry_pcm = turn.pcm
                    self._emit("utterance_carried", {"samples": int(len(turn.pcm))})
                self._emit("turn_cancelled", {"kind": turn.kind, "user_text": m.user_text})
                return
        else:
            m.assistant_text = " ".join(j.raw for j in turn.chunks if j.raw)
        self.turns.append(m)
        self._turns_done += 1
        self._emit(
            "turn",
            {
                "kind": turn.kind,
                "user_text": m.user_text,
                "assistant_text": m.assistant_text,
                "interrupted": interrupted,
                "error": error,
                "breakdown": m.breakdown(),
                "response_latency": None if m.response_latency() is None else round(m.response_latency(), 3),
                "hint_spoken": turn.hint_spoken,
                "tool_calls": turn.tool_calls,
                "filler_played": turn.filler_samples > 0,
                "chunks": len(turn.chunks),
            },
        )

    async def _shutdown(self) -> None:
        self._stopping = True
        turn = self._response
        if turn is not None and turn.task is not None and not turn.task.done():
            played = self.player.stop() if turn.marked else 0
            turn.task.cancel()
            await asyncio.wait({turn.task})
            self._finish_turn(turn, interrupted=True, played=played)
        else:
            try:
                self.player.stop()
            except Exception:  # pragma: no cover
                pass
        self._set_state(State.LISTENING)
        aclose = getattr(self.frames, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:  # pragma: no cover
                pass

    # ------------------------------------------------------------------ misc
    def _should_stop(self) -> bool:
        return self.max_turns is not None and self._turns_done >= self.max_turns and self._response is None

    def _set_state(self, new: State) -> None:
        if new is self.state:
            return
        old, self.state = self.state, new
        seg = self.segmenter
        if seg is not None and self.settings.echo_guard:
            if new is State.SPEAKING and self._base_threshold is None:
                self._base_threshold = seg.threshold
                seg.threshold = min(0.98, self._base_threshold + ECHO_GUARD_BOOST)
            elif new is not State.SPEAKING and self._base_threshold is not None:
                seg.threshold = self._base_threshold
                self._base_threshold = None
        self._emit("state", {"from": old.value, "to": new.value})

    def _emit(self, name: str, data: dict[str, Any]) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(name, data)
        except Exception:  # a broken UI callback must never kill the loop
            log.exception("on_event(%s) failed", name)

    async def _render_to_bytes(self, text: str) -> bytes:
        fn = getattr(self.tts, "synthesize_to_bytes", None)
        if fn is not None:
            return bytes(await fn(text))
        return b"".join([b async for b in self.tts.synthesize(text)])

    @staticmethod
    def _event_to_user_text(ev: Any) -> str:
        if not isinstance(ev, dict):
            return f"[system: {ev}]"
        kind = ev.get("type")
        if kind == "timer":
            label = ev.get("label") or "timer"
            return f"[system: the timer '{label}' just finished - tell the user naturally]"
        if kind == "reminder":
            return f"[system: reminder due: {ev.get('text') or ev.get('label')} - tell the user naturally]"
        msg = ev.get("message") or ev.get("text")
        return f"[system: {msg}]" if msg else f"[system: {json.dumps(ev)}]"

    def _append_user(self, text: str) -> str:
        if self._carry_text:
            text = (self._carry_text + " " + text).strip()
            self._carry_text = None
            self._emit("utterance_merged", {"how": "text", "text": text})
        self.messages.append({"role": "user", "content": text})
        self._trim_history()
        return text

    def _pop_last_user(self) -> str | None:
        if self.messages and self.messages[-1].get("role") == "user":
            return str(self.messages.pop()["content"])
        return None

    def _trim_history(self) -> None:
        if len(self.messages) <= HISTORY_LIMIT:
            return
        self.messages = self.messages[-HISTORY_LIMIT:]
        while self.messages and self.messages[0].get("role") != "user":
            self.messages.pop(0)

    def _messages_for_llm(self) -> list[dict[str, Any]]:
        return [{"role": "system", "content": self.system_prompt}, *self.messages]
