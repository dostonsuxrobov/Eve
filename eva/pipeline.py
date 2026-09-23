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
segmenter reports ``barge_in_min_speech_ms`` of speech-positive VAD windows
(``speaking_ms`` while the user talks, ``SpeechEnd.speech_ms`` if the utterance
ended first; never the utterance *length*, which includes the pre-speech ring
buffer and the trailing silence), so a cough or a "mm" never stops her.  On
confirmation: ``turn.cancelled`` is set, ``player.stop()`` (instant), the response
task is cancelled (which tears down every child task: LLM stream, TTS jobs,
writer, filler timer) and awaited, then ``player.stop()`` once more so that a
writer or filler wake-up that was already scheduled in the same event-loop
iteration can never leave audio behind (they also check ``turn.cancelled``
before writing).  ``played_samples`` is mapped onto the per-chunk sample counts
we tracked, only the words that were actually heard are kept and stored with
`` [interrupted]``.  If nothing was heard yet (interrupted while thinking) the
user message is taken back and merged with the next utterance, so "Hey Eva ...
how's it going" becomes one turn instead of two.  A barge-in that lands while a
tool is executing answers every unanswered ``tool_call_id`` with a synthetic
``role: tool`` message before the ``[interrupted]`` assistant message, so the
history always satisfies the chat contract.

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
A ``commit()`` that has not returned by the commit deadline is cancelled (its
socket dropped) and the utterance audio is sent to the batch endpoint *after
that*, never concurrently: two Scribe requests in flight at once were measured to
slow both down 10-30x (``bench/out/e2e_cloud-fast_race.json``).  The deadline is
``STT_COMMIT_DEADLINE_S`` when the service is fast and grows with the slowest
recent Scribe round trip (last commit, last batch request, the warmup probe) x
``STT_COMMIT_DEADLINE_FACTOR``: the batch endpoint is slow whenever the commits
are (same account state, measured 4-10x on both), so a fixed 2.5 s deadline paid
2.5 + 14 s for an 8 s utterance whose commits were returning in 1.7-2 s
(``bench/out/verify_cloud_run.json``).

Tool rounds
-----------
At most ``MAX_TOOL_ROUNDS`` tool rounds per turn: the round after the last one is
requested without tools and any tool call it still produces (native, or leaked
as JSON text and recovered) is dropped instead of executed, so a model that
keeps emitting calls can never loop and re-run its tools.

LLM keep-alive
--------------
If the LLM exposes ``ping()`` (``OpenAICompatLLM`` does: ``GET /models`` on the
pooled connection) it is called after ``LLM_KEEPALIVE_S`` of LLM idleness while
no response is in flight.  The Cerebras TTFT tail (2-3 s client-side while the
server reports 0.13-0.18 s) matches the cold-connection cost measured in
``DESIGN.md``; a turn never starts a request while a ping is in flight.

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

from .audio.echo import EchoDetector
from .audio.envelope import fade_in, fade_out, room_tone, split_tail
from .config import PipelineSettings
from .delivery import (
    detect_lang,
    echo_similarity,
    extract_cue,
    is_hesitation,
    looks_hallucinated,
    looks_incomplete,
    looks_like_echo,
)
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
from .llm.chunker import SentenceChunker
from .llm.sanitize import clean_for_tts
from . import tools as tools_mod

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
STT_COMMIT_DEADLINE_S = 2.5  # base deadline: a streaming commit() slower than this is cancelled, then ONE batch request follows
STT_COMMIT_DEADLINE_FACTOR = 2.0  # ... but never sooner than this x the slowest recent Scribe round trip (see docstring)
LLM_KEEPALIVE_S = 15.0  # ping the LLM's pooled connection after this much LLM idleness (0 = off)
TOOL_CANCELLED_RESULT = "cancelled: the user interrupted before the tool finished; do not assume it ran"
FILLER_TTS_GRACE_S = 0.6  # extra wait before a filler when a TTS request is already running
ECHO_WINDOW_S = 2.0  # an utterance starting this soon after her audio ended may be her own echo
ECHO_PARTIAL_SIM = 0.6  # a partial / final transcript this similar to what she is saying is her echo
ECHO_MIN_WORD_LEN = 3  # partial words shorter than this are not evidence of a person talking
LATE_COMMIT_TIMEOUT_S = 2.0  # an unconfirmed onset that ended: wait this long for its final text
ECHO_STORM_COUNT = 3  # echo-classified utterances within ECHO_STORM_WINDOW_S ...
ECHO_STORM_WINDOW_S = 20.0
ECHO_STORM_HOLD_S = 30.0  # ... make barge-in demand the final transcript for this long
_LETTERS_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
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


# ------------------------------------------------------------ default collaborators
def _default_chunker_factory(settings: PipelineSettings) -> Callable[[], Chunker]:
    return lambda: SentenceChunker(
        first_chunk_min_chars=settings.first_chunk_min_chars,
        min_chunk_chars=settings.min_chunk_chars,
    )


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
_SPEAKABLE_RE = re.compile(r"[^\W_]", re.UNICODE)  # at least one letter or digit


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


def _by_lang(v: "list[str] | dict[str, list[str]] | None") -> dict[str, list[str]]:
    """Normalise ``["mm", "hmm"]`` or ``{"en": [...], "ru": [...]}`` to a per-language dict."""
    if not v:
        return {}
    if isinstance(v, dict):
        return {k: [x for x in (vals or []) if x and x.strip()] for k, vals in v.items() if vals}
    return {"en": [x for x in v if x and x.strip()]}


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
    cue: str | None = None  # delivery cue ("warm", "teasing" ...) split off the raw text
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
    reply_cue: str | None = None  # the first delivery cue of the reply; every later chunk inherits it
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    rounds_in_history: int = 0  # assistant tool-call messages already appended
    recovered_seq: list[int] = field(default_factory=lambda: [0])  # ids for recovered tool calls
    finished: bool = False
    cancelled: bool = False  # set before the task is cancelled: no child may write audio after this
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
    ``tool_executor`` / ``pending_events`` default to the sibling modules (tests inject
    doubles).
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
        fillers: "list[str] | dict[str, list[str]] | None" = None,
        on_event: EventHandler | None = None,
        max_turns: int | None = None,
        tool_hints: "list[str] | dict[str, list[str]] | None" = None,
        backchannels: "list[str] | dict[str, list[str]] | None" = None,
        chunker_factory: Callable[[], Chunker] | None = None,
        sanitizer: Sanitizer | None = None,
        tool_executor: ToolExecutor | None = None,
        pending_events: "asyncio.Queue[dict[str, Any]] | None" = None,
        languages: "list[str] | None" = None,
    ) -> None:
        self.stt, self.llm, self.tts = stt, llm, tts
        # the session's language codes; a transcript the STT labels outside them is a
        # mishearing and never switches her language (see _follow_lang)
        self.languages = {c.lower() for c in languages or ()}
        self.system_prompt = system_prompt
        self.tools = list(tools)
        self.settings = settings
        self.frames = frames
        self.segmenter = segmenter
        self.player = player
        self.on_event = on_event
        self.max_turns = max_turns

        self.chunker_factory = chunker_factory or _default_chunker_factory(settings)
        self.sanitizer: Sanitizer = sanitizer or clean_for_tts
        self.tool_executor: ToolExecutor = tool_executor or tools_mod.execute
        self.pending_events: "asyncio.Queue[dict[str, Any]]" = (
            pending_events if pending_events is not None else tools_mod.pending_events
        )

        self.messages: list[dict[str, Any]] = []  # history without the system prompt
        self.turns: list[TurnMetrics] = []
        self.state = State.LISTENING
        # Spoken-language handling: fillers, tool hints and backchannels are kept per
        # language ("en", "ru", ...) and chosen by the language of the user's last turn.
        self._fillers_by_lang = _by_lang(fillers)
        self._hints_by_lang = _by_lang(tool_hints)
        self._backchannels_by_lang = _by_lang(backchannels)
        self._fillers_text = self._fillers_by_lang.get("en") or next(iter(self._fillers_by_lang.values()), [])
        self._tool_hints = self._hints_by_lang.get("en") or next(iter(self._hints_by_lang.values()), [])
        self._tool_hint_i = 0
        self._fillers_audio: list[bytes] = []  # audio for the current language (see _select_lang)
        self._fillers_audio_by_lang: dict[str, list[bytes]] = {}
        self._backchannel_audio_by_lang: dict[str, list[bytes]] = {}
        self._filler_i = 0
        self._backchannel_i = 0
        self._user_lang = "en"
        self._dip_ms = 0.0
        self._last_backchannel_t = -1e9
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
        self._last_spoken = ""  # what she said last (heard part), for the self-echo gate
        self._last_audio_end: float | None = None
        self._audible_since: float | None = None  # when the current SPEAKING state began
        # barge-in while she is audible: evidence gathered per VAD onset (see _check_barge_in)
        self._partial = ""  # latest partial transcript from the streaming STT for this onset
        self._cand_echo_hits = 0
        self._cand_echo_checks = 0
        self._cand_echo_reported = False
        self._late_task: asyncio.Task[None] | None = None
        self._echo_times: deque[float] = deque()
        self._storm_until = 0.0
        self._echo: EchoDetector | None = (
            EchoDetector(player, MIC_SAMPLE_RATE, max_lag_s=float(getattr(player, "echo_max_lag_s", 0.35)))
            if settings.echo_detector and callable(getattr(player, "played_since", None))
            else None
        )
        # Whisper-class STTs invent phrases on silence; Scribe / Parakeet do not (see eva.delivery)
        self._whisper_class = "whisper" in str(getattr(stt, "name", "")).lower()
        self._stopping = False
        self.end_requested = False  # set by an end_session event (the end_conversation tool)

        # streaming STT (feed while the user talks, commit at the endpoint)
        self.stt_streaming = all(callable(getattr(stt, m, None)) for m in ("feed", "commit", "discard"))
        if self.stt_streaming:
            try:
                stt.on_partial = self._on_partial  # type: ignore[attr-defined]
            except Exception:  # an STT without partials: barge-in falls back to the wait rule
                pass
        ring_ms = settings.prespeech_buffer_ms + settings.min_speech_ms + STREAM_RING_EXTRA_MS
        self._recent: deque[np.ndarray] = deque(maxlen=max(1, math.ceil(ring_ms / 20)))
        self._stream_open = False  # frames are being fed to the STT right now
        self._stream_broken = False  # feed() failed for the utterance in progress -> batch
        self.stream_turns = 0  # turns transcribed through the streaming path (for reports)
        self._stt_recent_s: dict[str, float] = {}  # latency of the last "commit" / "batch" Scribe request
        # LLM keep-alive (see module docstring)
        self._llm_last_t = _now()
        self._keepalive_task: asyncio.Task[None] | None = None
        self._ping_task: asyncio.Task[Any] | None = None
        self.pings = 0

    # ------------------------------------------------------------------ public
    @property
    def is_responding(self) -> bool:
        return self._response is not None

    async def prepare(self) -> None:
        """Pre-render the filler phrases in the active voice (idempotent)."""
        if self._prepared:
            return
        self._prepared = True
        self._start_keepalive()
        for lang, texts in self._fillers_by_lang.items():
            for text in texts:
                try:
                    audio = await self._render_to_bytes(text)
                except Exception as e:  # a broken filler must not stop startup
                    log.warning("filler %r failed to render: %s", text, e)
                    continue
                if audio:
                    self._fillers_audio_by_lang.setdefault(lang, []).append(audio)
        if self.settings.backchannels:
            for lang, texts in self._backchannels_by_lang.items():
                for text in texts:
                    try:
                        audio = await self._render_to_bytes(text)
                    except Exception as e:
                        log.warning("backchannel %r failed to render: %s", text, e)
                        continue
                    if audio:
                        self._backchannel_audio_by_lang.setdefault(lang, []).append(audio)
        self._select_lang(self._user_lang)
        self._emit(
            "ready",
            {
                "fillers": sum(len(v) for v in self._fillers_audio_by_lang.values()),
                "backchannels": sum(len(v) for v in self._backchannel_audio_by_lang.values()),
            },
        )

    def _follow_lang(self, text: str, heard_as: str | None) -> None:
        """Follow the user into the language of ``text`` unless the STT heard it as a
        language outside the session's (Scribe labelled Russian / English speech as
        Dutch, ``ja``, ``mk``): then her language, fillers and hints stay put."""
        heard = (heard_as or "").lower()[:2]  # Scribe answers ISO 639-1 ("en"); 639-3 starts the same for en/ru
        if heard and self.languages and heard not in self.languages:
            self._emit("stt_foreign", {"lang": heard_as, "kept": self._user_lang, "text": text[:80]})
            return
        lang = detect_lang(text, default=self._user_lang)
        if lang != self._user_lang:
            self._select_lang(lang)
            self._emit("language", {"lang": lang})

    def _select_lang(self, lang: str) -> None:
        """Switch fillers / hints to ``lang`` (falls back to any language that has them)."""
        self._user_lang = lang
        self._fillers_audio = self._fillers_audio_by_lang.get(lang) or next(
            iter(self._fillers_audio_by_lang.values()), []
        )
        self._fillers_text = self._fillers_by_lang.get(lang) or next(iter(self._fillers_by_lang.values()), [])
        self._tool_hints = self._hints_by_lang.get(lang) or next(iter(self._hints_by_lang.values()), [])

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
                if self._echo is not None:
                    self._echo.push_mic(frame)
                if self.settings.backchannels:
                    self._maybe_backchannel(frame_ms=1000.0 * len(frame) / 16_000)
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

    async def close(self) -> None:
        """Stop background housekeeping (the LLM keep-alive).  ``run()`` calls it on
        exit; text-mode callers (``say()`` only) call it themselves."""
        self._stopping = True
        for task in (self._keepalive_task, self._ping_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        self._keepalive_task = None
        self._ping_task = None

    # --------------------------------------------------------- LLM keep-alive
    def _start_keepalive(self) -> None:
        if self._keepalive_task is not None or LLM_KEEPALIVE_S <= 0 or self._stopping:
            return
        if not callable(getattr(self.llm, "ping", None)):
            return
        self._keepalive_task = asyncio.create_task(self._keepalive_loop(), name="eva-keepalive")

    async def _keepalive_loop(self) -> None:
        """``llm.ping()`` whenever the LLM has been idle for ``LLM_KEEPALIVE_S`` and no
        response is in flight, so the pooled keep-alive connection is never stale
        when the next turn needs it."""
        while not self._stopping:
            idle = _now() - self._llm_last_t
            if self._response is not None or (self.segmenter is not None and self.segmenter.speaking):
                # a turn is running, or one is about to start: a ping now could still be in
                # flight when the commit returns and the completion has to wait for it
                await asyncio.sleep(0.25)
                continue
            if idle < LLM_KEEPALIVE_S:
                await asyncio.sleep(max(0.1, LLM_KEEPALIVE_S - idle))
                continue
            self._ping_task = asyncio.ensure_future(self.llm.ping())  # type: ignore[attr-defined]
            try:
                secs = await self._ping_task
                self.pings += 1
                self._emit("llm_ping", {"seconds": None if not isinstance(secs, (int, float)) else round(secs, 3)})
            except asyncio.CancelledError:
                raise
            except Exception as e:  # a failed ping is only a missed warm-up
                log.debug("llm ping failed: %s", e)
            finally:
                self._ping_task = None
                self._llm_last_t = _now()

    async def _await_ping(self) -> None:
        """Never start a completion beside a ping: with the one warm connection busy the
        request would open a second, cold one."""
        ping = self._ping_task
        if ping is not None and not ping.done():
            await asyncio.wait({ping}, timeout=1.0)

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
        if isinstance(ev, dict) and ev.get("type") == "end_session":
            # The goodbye was spoken in the turn that called the tool; now stop the loop.
            self.end_requested = True
            self._emit("session_end", {"reason": ev.get("reason") or ""})
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
                # speech-positive VAD time, not the utterance length: the segmenter seeds
                # every utterance with up to prespeech_buffer_ms of ring audio and keeps a
                # 150 ms tail, so duration_s is >= ~0.5 s for any blip that got this far
                speech_ms = float(getattr(ev, "speech_ms", 0.0) or 0.0) or ev.duration_s * 1000
                if speech_ms < self.settings.barge_in_min_speech_ms:
                    self._emit(
                        "barge_in_ignored",
                        {"reason": "blip", "duration_s": round(float(ev.duration_s), 3), "speech_ms": round(speech_ms, 1)},
                    )
                    await self._stt_discard(keep_audio=False)
                elif not self._needs_words():
                    await self._interrupt("barge-in", t_trigger=ev.t)
                    await self._start_voice_turn(ev, streamed=streamed)
                else:
                    # she is audible and the onset never proved itself: decide on the final text
                    verdict = self._words_verdict()
                    if verdict == "user":
                        await self._interrupt("barge-in", t_trigger=ev.t)
                        await self._start_voice_turn(ev, streamed=streamed)
                    elif verdict == "echo" or not streamed:
                        self._note_echo("barge_in_ignored", {"reason": "echo", "partial": self._partial[:80], "echo_ratio": self._echo_ratio()})
                        await self._stt_discard(keep_audio=False)
                    else:
                        self._late_check(ev)
            else:
                # response in flight, barge-in disabled: answer it after this turn (in batch)
                self._pending_utterance = ev
                self._emit("utterance_queued", {"duration_s": round(float(ev.duration_s), 3)})
                await self._stt_discard(keep_audio=False)
            return False
        # SpeechStart
        self._last_speech_start = float(ev.t)
        self._partial = ""
        self._cand_echo_hits = self._cand_echo_checks = 0
        self._cand_echo_reported = False
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
        if speaking_ms < min_ms:
            return
        if self._needs_words():
            if self._echo is not None:
                v = self._echo.check()
                self._cand_echo_checks += 1
                self._cand_echo_hits += int(v.is_echo)
            verdict = self._words_verdict(speaking_ms=speaking_ms)
            if verdict != "user":
                if verdict == "echo" and not self._cand_echo_reported:
                    self._cand_echo_reported = True
                    self._emit("barge_in_echo", {"partial": self._partial[:80], "echo_ratio": self._echo_ratio(), "speaking_ms": round(speaking_ms)})
                return  # keep listening: the evidence may still turn into a person talking
        # the moment the user had spoken exactly min_ms (segmenters may count speech
        # from before they emit SpeechStart, so derive it from speaking_ms itself)
        t_trigger = _now() - (speaking_ms - min_ms) / 1000.0
        self._barge_candidate = None
        await self._interrupt("barge-in", t_trigger=t_trigger)

    # ------------------------------------------------- barge-in while she is audible
    def _on_partial(self, text: str) -> None:
        self._partial = text or ""

    def _needs_words(self) -> bool:
        """True while she is audible and the words rule is on: a VAD onset is not enough."""
        turn = self._response
        if turn is None or self.settings.barge_in_confirm != "words":
            return False
        return bool(turn.audio_started or turn.filler_samples)

    def _reply_text(self) -> str:
        """What she is saying in the current turn (all chunks so far) plus the filler phrases."""
        turn = self._response
        parts = [j.raw for j in turn.chunks if j.raw] if turn is not None else []
        return " ".join(parts + list(self._fillers_text))

    def _echo_ratio(self) -> float | None:
        if not self._cand_echo_checks:
            return None
        return round(self._cand_echo_hits / self._cand_echo_checks, 2)

    def _words_verdict(self, *, speaking_ms: float | None = None) -> str:
        """``"user"`` (a person is talking over her), ``"echo"`` (the mic hears the speakers)
        or ``"unknown"`` (no evidence yet), from the partial transcript and the echo detector."""
        ratio = self._echo_ratio()
        signal_echo = ratio is not None and self._cand_echo_checks >= 3 and ratio >= 0.6
        if not self.stt_streaming:
            # no partials to read: the echo detector is the only evidence there is
            return "echo" if signal_echo else "user"
        words = [w for w in _LETTERS_RE.findall(self._partial) if len(w) >= ECHO_MIN_WORD_LEN]
        if words:
            if echo_similarity(self._partial, self._reply_text()) >= ECHO_PARTIAL_SIM or is_hesitation(self._partial):
                return "echo"
            if signal_echo:
                return "echo"
            if _now() < self._storm_until:
                return "unknown"  # in a storm only the final transcript may interrupt her
            return "user"
        if signal_echo:
            return "echo"
        if speaking_ms is not None and speaking_ms >= self.settings.barge_in_words_wait_ms and _now() >= self._storm_until:
            return "user"  # sustained speech and a silent STT: assume a person
        return "unknown"

    def _late_check(self, ev: Any) -> None:
        """The onset ended before proving itself: fetch its final transcript, then decide.

        She keeps talking meanwhile. A real interjection ("wait!") that the partials
        missed interrupts her a few hundred milliseconds late; her echo is dropped.
        """
        if self._late_task is not None and not self._late_task.done():
            self._late_task.cancel()
        self._late_task = asyncio.create_task(self._late_check_run(ev), name="eva-late-bargein")

    async def _late_check_run(self, ev: Any) -> None:
        try:
            tr = await asyncio.wait_for(self.stt.commit(), timeout=LATE_COMMIT_TIMEOUT_S)  # type: ignore[attr-defined]
        except (asyncio.TimeoutError, Exception) as e:
            self._emit("barge_in_ignored", {"reason": f"late text unavailable: {type(e).__name__}"})
            await self._stt_discard(keep_audio=False)
            return
        text = (tr.text or "").strip()
        words = [w for w in _LETTERS_RE.findall(text) if len(w) >= ECHO_MIN_WORD_LEN]
        reply = self._reply_text() or self._last_spoken
        if not words or is_hesitation(text) or echo_similarity(text, reply) >= ECHO_PARTIAL_SIM:
            self._note_echo("barge_in_ignored", {"reason": "echo (final)", "text": text[:80]})
            return
        turn = self._response
        if turn is None:
            return  # she finished meanwhile: it is answered as a normal turn below
        await self._interrupt("barge-in (late)", t_trigger=float(ev.t))
        self._follow_lang(text, tr.meta.get("language"))
        metrics = TurnMetrics(speech_start=self._last_speech_start, speech_end=float(ev.t))
        self._emit("stt", {"text": text, "latency_s": round(tr.latency_s, 3), "samples": int(len(ev.pcm)), "mode": "stream", "fallback": None})
        self._launch(_Turn(metrics=metrics, kind="voice", user_text=text))

    def _note_echo(self, event: str, data: dict[str, Any]) -> None:
        """Record an echo classification; three within ECHO_STORM_WINDOW_S start a storm."""
        now = _now()
        self._echo_times.append(now)
        while self._echo_times and now - self._echo_times[0] > ECHO_STORM_WINDOW_S:
            self._echo_times.popleft()
        self._emit(event, data)
        if len(self._echo_times) >= ECHO_STORM_COUNT and now >= self._storm_until:
            self._storm_until = now + ECHO_STORM_HOLD_S
            self._emit("echo_storm", {"hold_s": ECHO_STORM_HOLD_S, "count": len(self._echo_times)})

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
                elif await self._user_went_on(turn, text):
                    # They paused mid-thought and continued: keep the text for merging
                    # with the next utterance instead of answering half a sentence.
                    self._carry_text = ((self._carry_text or "") + " " + text).strip()
                    self._emit("utterance_carried", {"text": self._carry_text, "why": "incomplete"})
                    return
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

    async def _user_went_on(self, turn: _Turn, text: str) -> bool:
        """After an unfinished-looking transcript, wait ``incomplete_grace_ms`` for the user
        to resume. True if they did (speech detected again) before the grace ran out."""
        grace = self.settings.incomplete_grace_ms
        if grace <= 0 or turn.kind != "voice" or self.segmenter is None or not looks_incomplete(text):
            return False
        self._emit("stt_incomplete", {"text": text, "grace_ms": grace})
        deadline = _now() + grace / 1000.0
        while _now() < deadline:
            if getattr(self.segmenter, "speaking", False) or self._barge_candidate is not None:
                self._barge_candidate = None
                return True
            await asyncio.sleep(0.02)
        return False

    async def _transcribe(self, turn: _Turn) -> str:
        assert turn.pcm is not None
        turn.stt_inflight = True
        try:
            if turn.use_stream:
                tr = await self._commit_then_batch(turn)
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
        reason = looks_hallucinated(text, tr.meta, whisper_class=self._whisper_class)
        if reason is not None:
            # Noise, breaths and fan hum make Whisper-class models invent "Thank you." etc.
            self._emit("stt_phantom", {"text": text, "reason": reason})
            return ""
        if is_hesitation(text):
            # "Uh." / "Hmm.": thinking, not a turn. Answering it is the "ignoring me" feel.
            self._emit("stt_hesitation", {"text": text})
            return ""
        if self.settings.self_echo_gate and self._last_spoken and self._last_audio_end is not None:
            started = turn.metrics.speech_start if turn.metrics.speech_start is not None else turn.started_at
            near = started - self._last_audio_end < ECHO_WINDOW_S
            during = self._audible_since is not None and started >= self._audible_since - 0.2
            # fuzzy matching only for an utterance that began while she was playing: after she
            # stopped, a short reply that repeats her words is an answer, not echo
            echo = (during and looks_like_echo(text, self._last_spoken, min_words=2, fuzzy=ECHO_PARTIAL_SIM)) or (
                near and looks_like_echo(text, self._last_spoken)
            )
            if echo:
                # the mic heard her own reply through the speakers (echo canceller still converging)
                self._note_echo("stt_echo", {"text": text, "spoken": self._last_spoken})
                return ""
        self._follow_lang(text, tr.meta.get("language"))
        return text

    async def _commit_then_batch(self, turn: _Turn) -> Transcript:
        """``stt.commit()`` with a deadline; on timeout or failure ONE batch request follows.

        The realtime commit returns in 0.15-0.4 s typically but has a server-side
        tail of 2-4 s.  Never two Scribe requests at once: a batch request started
        while a commit was still being served made both crawl (4 / 31 / 7.5 s per
        turn, ``bench/out/e2e_cloud-fast_race.json``, the account's concurrency
        limit).  So a late commit is cancelled first, which drops its socket via
        ``discard()``, and only then is the utterance audio sent to the batch
        endpoint (the STT waits for the socket to close before posting).
        """
        assert turn.pcm is not None
        deadline = self._commit_deadline()
        commit = asyncio.ensure_future(self.stt.commit())  # type: ignore[attr-defined]
        try:
            done, _ = await asyncio.wait({commit}, timeout=deadline)
        finally:
            if not commit.done():  # deadline passed, or this turn was cancelled (barge-in)
                commit.cancel()
                try:
                    await commit
                except (asyncio.CancelledError, Exception):
                    pass
            if commit.cancelled():
                await self._stt_discard(keep_audio=self._stream_open)
        if not commit.cancelled() and commit.exception() is None:
            tr = commit.result()
            self.stream_turns += 1
            tr.meta.setdefault("path", "stream")
            self._stt_recent_s["batch" if tr.meta.get("fallback") else "commit"] = float(tr.latency_s)
            return tr
        if commit.cancelled():
            reason = f"commit slower than {deadline:.1f} s"
            self._emit("stt_commit_timeout", {"after_s": round(deadline, 3), "base_s": STT_COMMIT_DEADLINE_S})
        else:
            exc = commit.exception()
            reason = f"commit failed: {type(exc).__name__}"
            log.warning("streaming stt commit failed (%s); falling back to batch", exc)
            self._emit("error", {"where": "stt_stream", "error": repr(exc)})
        tr = await self._transcribe_batch(turn.pcm)
        tr.meta["fallback"] = f"batch after {reason}"
        tr.meta["path"] = "batch"
        self._emit("stt_fallback", {"reason": reason, "latency_s": round(tr.latency_s, 3)})
        return tr

    def _commit_deadline(self) -> float:
        """How long to wait for a streaming commit before ONE batch request replaces it.

        Cancelling costs the time already waited plus a full batch round trip, and the
        batch endpoint is slow whenever the commits are, so the deadline is at least
        ``STT_COMMIT_DEADLINE_FACTOR`` x the slowest recent Scribe round trip: the last
        commit, the last batch request, or the STT's own warmup probe
        (``last_batch_s``).  Fast service: ``STT_COMMIT_DEADLINE_S``.
        """
        recent = [self._stt_recent_s.get("commit"), self._stt_recent_s.get("batch"), getattr(self.stt, "last_batch_s", None)]
        slowest = max((float(v) for v in recent if v), default=0.0)
        return max(float(STT_COMMIT_DEADLINE_S), STT_COMMIT_DEADLINE_FACTOR * slowest)

    async def _transcribe_batch(self, pcm: np.ndarray) -> Transcript:
        """Whole-utterance transcription; a streaming STT may offer a dedicated batch entry point."""
        fn = getattr(self.stt, "transcribe_batch", None) if self.stt_streaming else None
        tr = await (fn(pcm, MIC_SAMPLE_RATE) if fn is not None else self.stt.transcribe(pcm, MIC_SAMPLE_RATE))
        self._stt_recent_s["batch"] = float(tr.latency_s)
        return tr

    async def _generate_and_speak(self, turn: _Turn) -> None:
        begin = getattr(self.tts, "begin_turn", None)
        if callable(begin):
            begin()  # hybrid first-chunk model + prosodic continuity restart per reply
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
            if tool_calls and use_tools is None:
                # tools were not offered this round (the cap is reached, or there are none):
                # a call the model produced anyway is never executed, so the loop ends here
                self._emit("tool_calls_dropped", {"round": round_no, "names": [tc.name for tc in tool_calls]})
                tool_calls = []
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
            final_call = self._is_final_call(tool_calls)
            spoke_with_call = bool(round_text.strip())
            if not turn.chunks and not final_call:  # nothing spoken yet: cover the wait
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
            answered: set[str] = set()
            try:
                for tc in tool_calls:
                    result = await self._execute_tool(turn, tc)
                    last_results.append(result)
                    self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                    answered.add(tc.id)
            except asyncio.CancelledError:
                # a barge-in while a tool ran: every tool_call_id in the assistant message
                # above must still be answered or the history breaks the chat contract
                for tc in tool_calls:
                    if tc.id not in answered:
                        self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": TOOL_CANCELLED_RESULT})
                raise
            round_no += 1
            if final_call and spoke_with_call:
                # end_conversation: the goodbye was said with the call. A round after the
                # result would only say it a second time (measured: "Bye." ... "See you.").
                self._emit("tool_round_final", {"names": [tc.name for tc in tool_calls]})
                last_results = []
                break
        if not final_text and last_results:
            # the model went quiet after the tool: read the result out instead of silence
            final_text = last_results[-1]
            self._enqueue(turn, jobs, round_no, final_text)
        jobs.put_nowait(None)
        await writer
        end_turn = getattr(self.player, "end_turn", None)
        if callable(end_turn):
            end_turn()  # a remote player may hold a jitter buffer: nothing more is coming, drain it
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
        await self._await_ping()
        self._llm_last_t = _now()
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
            self._llm_last_t = _now()
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
        cue, body = extract_cue(raw)
        if is_hint:
            cue = None
        elif cue is None:
            cue = turn.reply_cue  # one voice for the whole reply: a chunk without a cue must not reset it
        elif turn.reply_cue is None:
            turn.reply_cue = cue
        text = self.sanitizer(body, self.tts.supports_audio_tags).strip()
        if text and not _SPEAKABLE_RE.search(text):
            text = ""  # brace / punctuation debris ("} }"): nothing to say, and ElevenLabs answers 400
        job = _ChunkJob(index=len(turn.chunks), round=round_no, raw=raw.strip(), text=text, is_hint=is_hint, cue=cue)
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
            if job.cue and getattr(self.tts, "supports_cues", False):
                stream = self.tts.synthesize(job.text, cue=job.cue)  # type: ignore[call-arg]
            else:
                stream = self.tts.synthesize(job.text)
            async for b in stream:
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
        """Strictly ordered playback: chunk i is fully written before chunk i+1 starts.

        Shapes the turn's envelope (``eva.audio.envelope``): a short lead-in silence and
        fade-in on the first audio, an optional gap between sentence chunks, and a
        fade-out plus tail silence after the last chunk. The last ``fade_out_ms`` of
        audio is held back until the writer knows whether more follows, so the fade
        lands exactly on the final word.
        """
        s, sr = self.settings, self.tts.sample_rate
        fade_out_ms = getattr(s, "fade_out_ms", 0)
        edge_ms = getattr(s, "chunk_edge_ms", 0)
        tone = getattr(s, "room_tone_dbfs", None)
        held: bytes = b""  # last fade_out_ms of audio not yet written
        held_job: _ChunkJob | None = None
        first_write = True
        while True:
            job = await jobs.get()
            if job is None:
                if held and held_job is not None:
                    tail = room_tone(getattr(s, "tail_ms", 0), sr, tone)
                    self._write_real(turn, held_job, fade_out(held, fade_out_ms, sr) + tail)
                return
            gap_ms = getattr(s, "sentence_gap_ms", 0)
            first_of_job = True
            while True:
                b = await job.audio.get()
                if b is None:
                    break
                while self.player.buffered_seconds > PLAYER_LOOKAHEAD_S:
                    await asyncio.sleep(0.02)
                if held and held_job is not None:
                    if held_job is not job:
                        # a chunk boundary: soften both hot edges (v3 clips start and end at
                        # -30..-40 dBFS) and leave the pause a sentence break has
                        gap = room_tone(gap_ms, sr, tone) if (gap_ms and not job.is_hint) else b""
                        self._write_real(turn, held_job, fade_out(held, edge_ms, sr) + gap)
                        b = fade_in(b, edge_ms, sr)
                    else:
                        # the same chunk continuing: the held tail is just the previous piece's
                        # last fade_out_ms, written back untouched. (Fading it here too, on every
                        # ~80 ms piece, was the 12 Hz tremolo heard as "a bad connection".)
                        self._write_real(turn, held_job, held)
                    held, held_job = b"", None
                if first_write:
                    first_write = False
                    lead = b"" if turn.filler_samples else room_tone(getattr(s, "lead_in_ms", 0), sr, tone)
                    b = lead + fade_in(b, getattr(s, "fade_in_ms", 0), sr)
                first_of_job = False
                if fade_out_ms:
                    b, held = split_tail(b, fade_out_ms, sr)
                    held_job = job
                if b:
                    self._write_real(turn, job, b)
                if not job.play_started.is_set():
                    job.play_started.set()
            if not job.play_started.is_set():
                job.play_started.set()

    def _write_real(self, turn: _Turn, job: _ChunkJob, b: bytes) -> None:
        if turn.cancelled:
            return  # the interrupt already ran player.stop(); this wake-up was scheduled before it
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
        if turn.audio_started or turn.finished or turn.cancelled:
            return
        # If the LLM is already producing text (a TTS request follows within a chunk)
        # the real audio is at most a TTFA away (~0.2-0.35 s) while a filler costs
        # ~0.5-1.3 s of extra wait, so hold off a little longer; only a stalled LLM /
        # TTS gets the filler after the grace period.
        deadline = _now() + FILLER_TTS_GRACE_S
        while (turn.metrics.llm_first_token is not None or any(not j.silent for j in turn.chunks)) and _now() < deadline:
            await asyncio.sleep(0.05)
            if turn.audio_started or turn.finished or turn.cancelled:
                return
        if turn.cancelled:
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

    def _maybe_backchannel(self, *, frame_ms: float) -> None:
        """Play a short "mm-hm" at a natural dip inside a LONG user utterance.

        Conditions: LISTENING with no reply in flight, the user has been speaking for
        ``backchannel_after_ms``, the VAD probability just dipped below the end
        threshold for ``backchannel_dip_ms`` (a breath or a comma-pause, not the end of
        the turn), and the last backchannel was ``backchannel_min_gap_s`` ago. Headphones
        recommended: through speakers the sound reaches the mic and the STT.
        """
        seg = self.segmenter
        audio_list = self._backchannel_audio_by_lang.get(self._user_lang) or next(
            iter(self._backchannel_audio_by_lang.values()), []
        )
        if not audio_list or seg is None or self.state is not State.LISTENING or self._response is not None:
            self._dip_ms = 0.0
            return
        prob = getattr(seg, "last_prob", None)
        if not getattr(seg, "speaking", False) or prob is None:
            self._dip_ms = 0.0
            return
        if getattr(seg, "speaking_ms", 0.0) < self.settings.backchannel_after_ms:
            return
        end_thr = getattr(seg, "end_threshold", None)
        if end_thr is None:
            end_thr = max(float(seg.threshold) - 0.15, 0.02)
        if prob < end_thr:
            self._dip_ms += frame_ms
        else:
            self._dip_ms = 0.0
            return
        now = _now()
        if self._dip_ms < self.settings.backchannel_dip_ms or now - self._last_backchannel_t < self.settings.backchannel_min_gap_s:
            return
        audio = audio_list[self._backchannel_i % len(audio_list)]
        self._backchannel_i += 1
        self._last_backchannel_t = now
        self._dip_ms = 0.0
        self.player.write(audio)
        self._emit("backchannel", {"index": self._backchannel_i - 1, "seconds": round(len(audio) / 2 / self.tts.sample_rate, 2)})

    def _is_final_call(self, tool_calls: list[LLMToolCall]) -> bool:
        """True if every call in the round is to a ``Tool.final`` tool (end_conversation)."""
        by_name = {t.name: t for t in self.tools}
        return bool(tool_calls) and all(getattr(by_name.get(tc.name), "final", False) for tc in tool_calls)

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
        turn.cancelled = True  # before stop(): a writer/filler wake-up already scheduled must not write
        played = self.player.stop()
        if not turn.marked:
            played = 0
        t_stopped = _now()
        stt_was_inflight = turn.stt_inflight and turn.use_stream
        if turn.task is not None and not turn.task.done():
            turn.task.cancel()
            await asyncio.wait({turn.task})
        # every child (writer, TTS jobs, filler timer) is cancelled and awaited by the
        # task's finally, so nothing can write after this second stop
        self.player.stop()
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
        if m.assistant_text.strip():
            self._last_spoken = m.assistant_text
            self._last_audio_end = m.audio_finished
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
        if self._late_task is not None and not self._late_task.done():
            self._late_task.cancel()
        turn = self._response
        if turn is not None and turn.task is not None and not turn.task.done():
            turn.cancelled = True
            played = self.player.stop() if turn.marked else 0
            turn.task.cancel()
            await asyncio.wait({turn.task})
            self.player.stop()
            self._finish_turn(turn, interrupted=True, played=played)
        else:
            try:
                self.player.stop()
            except Exception:  # pragma: no cover
                pass
        self._set_state(State.LISTENING)
        await self.close()
        aclose = getattr(self.frames, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:  # pragma: no cover
                pass

    # ------------------------------------------------------------------ misc
    def _should_stop(self) -> bool:
        if self.end_requested and self._response is None:
            return True
        return self.max_turns is not None and self._turns_done >= self.max_turns and self._response is None

    def _set_state(self, new: State) -> None:
        if new is self.state:
            return
        old, self.state = self.state, new
        if new is State.SPEAKING:
            self._audible_since = _now()
        elif old is State.SPEAKING:
            self._last_audio_end = _now()
            self._audible_since = None
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
        """Render a filler / backchannel, cached on disk per (tts name, text) so a restart
        costs no TTS credits. Cache dir: ``models/filler_cache`` (gitignored)."""
        import hashlib

        from .config import MODELS_DIR

        cache_dir = MODELS_DIR / "filler_cache"
        key = hashlib.sha1(f"{self.tts.name}|{self.tts.sample_rate}|{text}".encode("utf-8")).hexdigest()
        path = cache_dir / f"{key}.pcm"
        try:
            if path.exists() and path.stat().st_size > 0:
                return path.read_bytes()
        except OSError:
            pass
        begin = getattr(self.tts, "begin_turn", None)
        if callable(begin):
            begin()
        fn = getattr(self.tts, "synthesize_to_bytes", None)
        if fn is not None:
            audio = bytes(await fn(text))
        else:
            audio = b"".join([b async for b in self.tts.synthesize(text)])
        if audio:
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                path.write_bytes(audio)
            except OSError as e:
                log.debug("filler cache write failed: %s", e)
        return audio

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
        if kind == "session_start":
            name = ev.get("user_name") or "them"
            lang = str(ev.get("language") or {"ru": "Russian", "en": "English"}.get(str(ev.get("lang") or "en"), "English"))
            return (
                f"[system: the conversation just started. Say hello to {name} in {lang} the way you "
                "greet someone you talk to every day: one short, unhurried sentence in your own words, "
                "no 'how can I help', no task talk, no question needed. Then wait for them.]"
            )
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
