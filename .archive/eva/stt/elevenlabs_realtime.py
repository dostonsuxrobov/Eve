"""ElevenLabs realtime Scribe STT over websocket (``scribe_v2_realtime``).

Endpoint (verified Sept 2026; this key is permitted):
    wss://api.elevenlabs.io/v1/speech-to-text/realtime
        ?model_id=scribe_v2_realtime&audio_format=pcm_16000&commit_strategy=manual
    header:  xi-api-key
    client -> {"message_type": "input_audio_chunk", "audio_base_64": <b64 pcm>,
               "commit": bool, "sample_rate": 16000}
    server -> session_started | partial_transcript | committed_transcript
              | committed_transcript_with_timestamps | <error types> | warning

Two ways to use :class:`ElevenLabsRealtimeSTT`:

1. ``transcribe(pcm)`` - the common :class:`eva.interfaces.STT` contract.  The whole
   utterance is burst-streamed in 100 ms chunks, the last chunk carries
   ``commit=True`` and we await the committed transcript.  Measured: this is *not*
   faster than the batch endpoint (the server transcribes at roughly real time and
   only starts after ~1-2 s of audio); it exists for parity tests.

2. ``feed(frame)`` while the user speaks + ``commit()`` at the VAD endpoint - the way
   the pipeline should use it.  Because the server has already processed the audio,
   ``commit()`` returns in 0.1-0.5 s (measured) instead of the 0.7-1.3 s batch round
   trip.  Partial transcripts arrive through ``on_partial``.

Measured server quirks handled here:
  * Commits with < 0.3 s of uncommitted audio are ignored (``commit_throttled``):
    short segments are padded with silence to 0.35 s.
  * A session keeps "context" across commits; when a long utterance repeats the
    content of the previous segment the transcriber stalls after two partials and
    commits ``""`` (reproduced 6/6 times; never on a fresh socket).  Therefore the
    socket is rotated after every commit (``rotate_sessions=True``): the fresh one is
    opened in the background while the LLM/TTS run, so no turn pays the handshake.
  * If a commit still returns ``""`` for voiced audio (RMS above ``voiced_rms``) the
    utterance is re-run through the batch endpoint (``fallback_to_batch=True``).
  * The server closes a session that receives no audio for roughly 15 s (measured:
    alive after 12 s idle, closed after 20 s).  A pre-opened socket waiting for the
    next turn therefore sends 100 ms of silence every ``keepalive_s`` (8 s) while
    idle; measured to keep a session alive for 40 s with the commit still correct.
    Should the server close an idle session anyway, a replacement is pre-opened
    immediately so the next ``feed()`` never pays the handshake.
  * Without ``language_code`` the server picks from ~90 languages; on noisy 1.2 s
    fragments it labelled English as ``ja`` and silence tails as ``mk`` (2026-09-23).
    ``language_code`` + ``secondary_languages`` (en + [ru] or ru + [en], same results)
    boxes it in: the ``ja`` case came back as English, clean speech was unchanged. It is
    a bias, not a wall (a ``mk`` label survived), so ``language_detection`` reports the
    label (``meta["language"]``) and the pipeline refuses to switch on a foreign one.
    With detection on, ``committed_transcript_with_timestamps`` (``language_code``, no
    words) arrives before ``committed_transcript``.
  * Never two Scribe requests at once.  A batch request sent while a realtime
    commit was still being served made both crawl (4 / 31 / 7.5 s per turn,
    ``bench/out/e2e_cloud-fast_race.json``; the account's concurrency limit).  One
    ``asyncio.Lock`` therefore serializes ``commit()`` and ``transcribe_batch()``,
    a batch request waits (bounded) until retired sockets have finished closing,
    a new socket is only opened once the retired one is closed, and ``warmup()``
    makes its batch call before it opens the first realtime session.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from collections.abc import Callable, Sequence
from typing import Any
from urllib.parse import urlencode

import numpy as np
import websockets
from websockets.asyncio.client import ClientConnection

from ..config import USER_AGENT
from ..interfaces import MIC_SAMPLE_RATE, Transcript
from .elevenlabs_scribe import ElevenLabsScribeSTT, as_int16_mono, ensure_ssl_context, is_voiced

log = logging.getLogger(__name__)

REALTIME_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
_ERROR_TYPES = frozenset(
    {
        "error", "auth_error", "quota_exceeded", "commit_throttled", "unaccepted_terms",
        "rate_limited", "queue_overflow", "resource_exhausted", "session_time_limit_exceeded",
        "input_error", "invalid_request", "chunk_size_exceeded", "insufficient_audio_activity",
        "transcriber_error",
    }
)
# Errors that just mean "nothing to transcribe": map to an empty transcript, keep the session.
# commit_throttled = "only X s of uncommitted audio; need at least 0.3 s" (measured message).
_SOFT_ERRORS = frozenset({"insufficient_audio_activity", "commit_throttled"})
# The server ignores commits with < 0.3 s of uncommitted audio; pad short segments to this.
MIN_COMMIT_AUDIO_S = 0.35


class RealtimeSTTError(RuntimeError):
    """Raised for auth / quota / protocol failures on the realtime websocket."""

    def __init__(self, message: str, error_type: str | None = None):
        super().__init__(message)
        self.error_type = error_type


class RealtimeSession:
    """One open websocket to Scribe realtime: ``feed()`` audio, ``commit()`` for text.

    Use from one asyncio task at a time. ``commit`` is serialized with a lock.
    """

    def __init__(
        self,
        ws: ClientConnection,
        *,
        sample_rate: int,
        chunk_ms: int,
        include_timestamps: bool,
        on_partial: Callable[[str], None] | None,
        commit_timeout_s: float,
        keepalive_s: float = 8.0,
        on_closed: Callable[["RealtimeSession"], None] | None = None,
        language_detection: bool = False,
    ) -> None:
        self._ws = ws
        self._language_detection = language_detection
        self._keepalive_s = keepalive_s
        self._on_closed = on_closed
        self._last_send_t = time.perf_counter()
        self.keepalives_sent = 0
        self.sample_rate = sample_rate
        self._chunk_samples = max(1, sample_rate * chunk_ms // 1000)
        self._include_timestamps = include_timestamps
        self._on_partial = on_partial
        self._commit_timeout_s = commit_timeout_s
        self._pending = np.zeros(0, dtype=np.int16)
        self.fed_since_commit = 0  # samples sent (or pending) since the last commit
        self.commits_done = 0
        self._commits: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._ts_event = asyncio.Event()
        self._last_words: list[dict[str, Any]] = []
        self._last_language: str | None = None
        self._lock = asyncio.Lock()
        self.session_id: str | None = None
        self.config: dict[str, Any] = {}
        self.latest_partial: str = ""
        self.partials_this_segment: int = 0
        self.closed = False
        self.error: RealtimeSTTError | None = None
        self._reader = asyncio.create_task(self._read_loop(), name="scribe-rt-reader")
        self._keepalive: asyncio.Task[None] | None = (
            asyncio.create_task(self._keepalive_loop(), name="scribe-rt-keepalive") if keepalive_s > 0 else None
        )

    async def _keepalive_loop(self) -> None:
        """Send 100 ms of silence whenever nothing has been sent for ``keepalive_s``."""
        silence = np.zeros(self.sample_rate // 10, dtype=np.int16)
        try:
            while not self.closed:
                wait = self._keepalive_s - (time.perf_counter() - self._last_send_t)
                if wait > 0:
                    await asyncio.sleep(wait)
                    continue
                await self._send_chunk(silence, commit=False)
                self.keepalives_sent += 1
        except (asyncio.CancelledError, websockets.ConnectionClosed):
            pass
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("scribe realtime keepalive stopped: %s", exc)

    # --------------------------------------------------------------- inbound
    async def _read_loop(self) -> None:
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                mt = msg.get("message_type", "")
                if mt == "partial_transcript":
                    self.latest_partial = msg.get("text", "")
                    self.partials_this_segment += 1
                    if self._on_partial:
                        try:
                            self._on_partial(self.latest_partial)
                        except Exception:  # never let a UI callback kill the reader
                            log.exception("on_partial callback raised")
                elif mt == "committed_transcript":
                    self._commits.put_nowait(msg)
                elif mt == "committed_transcript_with_timestamps":
                    self._last_words = msg.get("words") or []
                    self._last_language = msg.get("language_code")
                    self._ts_event.set()
                elif mt == "session_started":
                    self.session_id = msg.get("session_id")
                    self.config = msg.get("config") or {}
                elif mt == "warning":
                    log.warning("scribe realtime warning: %s", msg.get("warning"))
                elif mt in _ERROR_TYPES or "error" in mt:
                    err = RealtimeSTTError(f"{mt}: {msg.get('error') or msg}", mt)
                    log.warning("scribe realtime %s", err)
                    if mt in _SOFT_ERRORS:
                        self._commits.put_nowait({"message_type": "committed_transcript", "text": "", "soft_error": mt})
                    else:
                        self.error = err
                        self._commits.put_nowait({"message_type": "__error__", "error": err})
                else:
                    log.debug("scribe realtime unhandled message %s", mt)
        except websockets.ConnectionClosed as exc:
            log.info("scribe realtime connection closed: %s", exc)
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("scribe realtime reader crashed: %s", exc)
            self.error = RealtimeSTTError(str(exc))
        finally:
            self.closed = True
            self._commits.put_nowait({"message_type": "__closed__"})  # wake any waiter
            if self._on_closed is not None:
                try:
                    self._on_closed(self)
                except Exception:  # pragma: no cover
                    log.exception("on_closed callback raised")

    # -------------------------------------------------------------- outbound
    async def _send_chunk(self, chunk: np.ndarray, commit: bool) -> None:
        payload = {
            "message_type": "input_audio_chunk",
            "audio_base_64": base64.b64encode(chunk.tobytes()).decode("ascii") if len(chunk) else "",
            "commit": commit,
            "sample_rate": self.sample_rate,
        }
        self._last_send_t = time.perf_counter()
        await self._ws.send(json.dumps(payload))

    async def feed(self, pcm: np.ndarray) -> None:
        """Queue audio (any length, e.g. 20 ms mic frames); sends whenever a full chunk is ready."""
        if self.closed:
            raise self.error or RealtimeSTTError("session is closed", "closed")
        pcm = as_int16_mono(pcm)
        self.fed_since_commit += len(pcm)
        buf = np.concatenate([self._pending, pcm]) if len(self._pending) else pcm
        n_full = (len(buf) // self._chunk_samples) * self._chunk_samples
        for i in range(0, n_full, self._chunk_samples):
            await self._send_chunk(buf[i : i + self._chunk_samples], commit=False)
        self._pending = buf[n_full:]

    async def commit(self) -> Transcript:
        """Flush pending audio with ``commit=True`` and await the committed transcript.

        ``latency_s`` is commit-call -> committed text (what the pipeline experiences
        when audio was streamed live).
        """
        async with self._lock:
            if self.closed:
                raise self.error or RealtimeSTTError("session is closed", "closed")
            while not self._commits.empty():  # drop stale items from an earlier segment
                self._commits.get_nowait()
            self._ts_event.clear()
            self._last_words = []
            self._last_language = None
            t0 = time.perf_counter()
            min_samples = int(MIN_COMMIT_AUDIO_S * self.sample_rate)
            if self.fed_since_commit < min_samples:
                # Server rejects (commit_throttled) commits with < 0.3 s of new audio: pad with silence.
                pad = np.zeros(min_samples - self.fed_since_commit, dtype=np.int16)
                self._pending = np.concatenate([self._pending, pad])
            await self._send_chunk(self._pending, commit=True)
            self._pending = np.zeros(0, dtype=np.int16)
            self.fed_since_commit = 0
            try:
                item = await asyncio.wait_for(self._commits.get(), timeout=self._commit_timeout_s)
            except asyncio.TimeoutError as exc:
                raise RealtimeSTTError(f"no committed transcript within {self._commit_timeout_s}s", "timeout") from exc
            if item["message_type"] == "__error__":
                raise item["error"]
            if item["message_type"] == "__closed__":
                raise self.error or RealtimeSTTError("connection closed before commit completed", "closed")
            meta: dict[str, Any] = {
                "session_id": self.session_id,
                "partials": self.partials_this_segment,
                "last_partial": self.latest_partial,
                "text_s": round(time.perf_counter() - t0, 4),
            }
            if item.get("soft_error"):
                meta["soft_error"] = item["soft_error"]
            if (self._include_timestamps or self._language_detection) and item.get("text"):
                # The timestamps message follows within ~20 ms (with language detection on it
                # carries language_code and arrives first); wait briefly, never block a turn.
                try:
                    await asyncio.wait_for(self._ts_event.wait(), timeout=0.3)
                except asyncio.TimeoutError:
                    pass
            meta["words"] = self._last_words
            meta["language"] = self._last_language
            self.latest_partial = ""
            self.partials_this_segment = 0
            self.commits_done += 1
            return Transcript(text=(item.get("text") or "").strip(), latency_s=time.perf_counter() - t0, meta=meta)

    async def close(self) -> None:
        self.closed = True
        self._on_closed = None  # a deliberate close never triggers a replacement
        self._reader.cancel()
        if self._keepalive is not None:
            self._keepalive.cancel()
        try:
            await self._ws.close()
        except Exception:
            pass
        for task in (self._reader, self._keepalive):
            if task is None:
                continue
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


class ElevenLabsRealtimeSTT:
    """Realtime Scribe as an :class:`eva.interfaces.STT` plus a streaming feed/commit API.

    Args:
        api_key: ElevenLabs API key.
        model_id: ``scribe_v2_realtime`` (the only realtime model advertised).
        language: optional ``language_code`` query param.
        secondary_languages: other languages the speaker uses (``secondary_languages``);
            with ``language`` they box the recogniser into that set (see module doc).
            The batch fallback is only pinned to ``language`` when there are none.
        language_detection: request ``include_language_detection``; the detected code
            is returned as ``meta["language"]``.
        chunk_ms: audio chunk per websocket message (100 ms measured best for bursts).
        include_timestamps: also request ``committed_transcript_with_timestamps``.
        rotate_sessions: open a fresh websocket after every commit (in the background)
            instead of reusing one; avoids the repeated-content stall (see module doc).
        fallback_to_batch: if a commit returns "" for voiced audio, re-transcribe the
            buffered utterance with the batch endpoint (``scribe_v2``).
        voiced_rms: int16 RMS above which audio counts as voiced for the fallback.
        on_partial: callback receiving partial text as it streams in.
        commit_timeout_s: max wait for a committed transcript.
        min_audio_ms: ``transcribe()`` returns an empty transcript without a round trip
            for audio shorter than this.
    """

    def __init__(
        self,
        api_key: str,
        model_id: str = "scribe_v2_realtime",
        language: str | None = None,
        *,
        secondary_languages: Sequence[str] = (),
        language_detection: bool = False,
        chunk_ms: int = 100,
        include_timestamps: bool = False,
        rotate_sessions: bool = True,
        fallback_to_batch: bool = True,
        voiced_rms: float = 200.0,
        on_partial: Callable[[str], None] | None = None,
        commit_timeout_s: float = 15.0,
        connect_timeout_s: float = 10.0,
        min_audio_ms: int = 100,
        keepalive_s: float = 8.0,
    ) -> None:
        if not api_key:
            raise ValueError("ElevenLabs api_key is required")
        self.keepalive_s = keepalive_s
        self.api_key = api_key
        self.model_id = model_id
        self.language = language
        self.secondary_languages = [c for c in secondary_languages if c and c != language]
        self.language_detection = language_detection
        self.chunk_ms = chunk_ms
        self.include_timestamps = include_timestamps
        self.rotate_sessions = rotate_sessions
        self.fallback_to_batch = fallback_to_batch
        self.voiced_rms = voiced_rms
        self.on_partial = on_partial
        self.commit_timeout_s = commit_timeout_s
        self.connect_timeout_s = connect_timeout_s
        self.min_audio_ms = min_audio_ms
        self.name = f"elevenlabs/{model_id}"
        self._session: RealtimeSession | None = None
        self._next: asyncio.Task[RealtimeSession] | None = None  # background pre-connect
        self._segment: list[np.ndarray] = []  # audio fed since the last commit (resend + fallback)
        self._sent_chunks = 0  # how many entries of _segment the current session has received
        self._closing: set[asyncio.Task[None]] = set()  # retired sockets being closed in the background
        self._fallback: ElevenLabsScribeSTT | None = None
        # One Scribe request (realtime commit or batch) in flight at a time: see module doc.
        self._api_lock = asyncio.Lock()
        self.close_wait_s = 1.5  # max wait for a retired socket to close before the next request / socket
        self.last_connect_s: float | None = None
        # latency of the most recent batch request (incl. the warmup probe): the pipeline
        # reads it to size the commit deadline, since batch is slow whenever commits are
        self.last_batch_s: float | None = None
        self.sample_rate = MIC_SAMPLE_RATE
        self._closed_flag = False

    # ---------------------------------------------------------------- session
    def _url(self, sample_rate: int) -> str:
        params: list[tuple[str, str]] = [
            ("model_id", self.model_id),
            ("audio_format", f"pcm_{sample_rate}"),
            ("commit_strategy", "manual"),
            ("include_timestamps", "true" if self.include_timestamps else "false"),
        ]
        if self.language_detection:
            params.append(("include_language_detection", "true"))
        if self.language:
            params.append(("language_code", self.language))
            # an array is the repeated key (verified: session_started echoes ["ru"])
            params += [("secondary_languages", c) for c in self.secondary_languages]
        return f"{REALTIME_URL}?{urlencode(params)}"

    async def open_session(self, sample_rate: int = MIC_SAMPLE_RATE) -> RealtimeSession:
        """Open a new websocket and wait for ``session_started``.

        A retired socket that is still closing is awaited first (bounded by
        ``close_wait_s``) so the server never sees two sessions from us at once.
        """
        t0 = time.perf_counter()
        await self._await_closing()
        ssl_ctx = await ensure_ssl_context()  # shared; avoids ~150 ms of blocking per socket
        try:
            ws = await asyncio.wait_for(
                websockets.connect(
                    self._url(sample_rate),
                    ssl=ssl_ctx,
                    additional_headers={"xi-api-key": self.api_key, "User-Agent": USER_AGENT},
                    max_size=None,
                    open_timeout=self.connect_timeout_s,
                ),
                timeout=self.connect_timeout_s,
            )
        except websockets.InvalidStatus as exc:
            body = exc.response.body.decode("utf-8", "replace") if exc.response.body else ""
            raise RealtimeSTTError(f"websocket rejected: HTTP {exc.response.status_code} {body[:300]}", "http") from exc
        session = RealtimeSession(
            ws,
            sample_rate=sample_rate,
            chunk_ms=self.chunk_ms,
            include_timestamps=self.include_timestamps,
            language_detection=self.language_detection,
            on_partial=self._forward_partial,  # reads self.on_partial at call time: the pipeline sets it after warmup
            commit_timeout_s=self.commit_timeout_s,
            keepalive_s=self.keepalive_s,
            on_closed=self._session_closed,
        )
        # session_started is the first frame (measured: arrives with the handshake).
        for _ in range(100):
            if session.session_id or session.closed:
                break
            await asyncio.sleep(0.01)
        if session.closed:
            raise session.error or RealtimeSTTError("connection closed during session start", "closed")
        self.last_connect_s = time.perf_counter() - t0
        log.info("%s session %s open in %.3f s", self.name, session.session_id, self.last_connect_s)
        return session

    def _forward_partial(self, text: str) -> None:
        cb = self.on_partial
        if cb is not None:
            cb(text)

    def _session_closed(self, session: RealtimeSession) -> None:
        """The server (or the network) closed a socket we did not close ourselves.

        If it was the idle current session or the pre-opened spare, pre-open a
        replacement right away so the next utterance does not pay the handshake.
        A session that dies mid-utterance is handled by ``_flush`` (reconnect + resend).
        """
        if self._closed_flag:
            return
        if self._session is session and not self._segment:
            self._session = None
            self._sent_chunks = 0
            self._close_in_background(session)
            self._preconnect()
        elif self._next is not None and self._next.done() and not self._next.cancelled():
            if self._next.exception() is None and self._next.result() is session:
                self._next = None
                self._close_in_background(session)
                self._preconnect()

    def _preconnect(self) -> None:
        """Start opening the next session in the background (idempotent)."""
        if self._next is None or (self._next.done() and self._next.exception() is not None):
            self._next = asyncio.create_task(self.open_session(self.sample_rate), name="scribe-rt-preconnect")

    async def _current(self) -> RealtimeSession:
        """Return a live session, taking the pre-connected one or opening a new one."""
        if self._session is not None:
            if not self._session.closed:
                return self._session
            # died on its own (server close / network): the replacement must get the whole segment
            dead, self._session = self._session, None
            self._sent_chunks = 0
            self._close_in_background(dead)
        if self._next is not None:
            task, self._next = self._next, None
            try:
                self._session = await task
                return self._session
            except Exception as exc:
                log.warning("%s pre-connected session failed: %s", self.name, exc)
        self._session = await self.open_session(self.sample_rate)
        return self._session

    def _close_in_background(self, session: RealtimeSession) -> None:
        task = asyncio.create_task(session.close(), name="scribe-rt-close")
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)

    async def _await_closing(self) -> None:
        """Wait (at most ``close_wait_s``) until retired sockets have finished closing.

        Called before a batch request and before a new socket is opened, so that a
        session the server may still count as active never overlaps with the next
        request (measured: overlapping requests are served 10-30x slower).
        """
        pending = [t for t in self._closing if not t.done()]
        if pending:
            await asyncio.wait(pending, timeout=self.close_wait_s)

    async def _retire(self, session: RealtimeSession) -> None:
        if self._session is session:
            self._session = None
        self._sent_chunks = 0  # whatever the next session gets must start from the segment start
        self._close_in_background(session)

    async def _flush(self, session: RealtimeSession) -> None:
        """Send the not-yet-sent part of the current segment; reconnect + resend on a dead socket."""
        for attempt in range(2):
            if self._sent_chunks >= len(self._segment):
                return
            unsent = self._segment[self._sent_chunks :]
            try:
                await session.feed(np.concatenate(unsent) if len(unsent) > 1 else unsent[0])
                self._sent_chunks = len(self._segment)
                return
            except (websockets.ConnectionClosed, RealtimeSTTError) as exc:
                log.warning("%s session died during feed (%s); reconnecting", self.name, exc)
                await self._retire(session)
                if attempt == 0:
                    session = await self._current()
                else:
                    raise

    # ------------------------------------------------------------ streaming API
    async def feed(self, pcm: np.ndarray) -> None:
        """Stream a mic frame (any length) of the utterance in progress.

        Never blocks on a socket that is still connecting (e.g. right after a rotation):
        frames are buffered and flushed as soon as the session is up or at ``commit()``.
        """
        self._segment.append(as_int16_mono(pcm))
        session = self._session if self._session is not None and not self._session.closed else None
        if session is None:
            if self._next is not None and not self._next.done():
                return  # background connect in flight; keep buffering
            session = await self._current()  # adopt the finished pre-connect, or open one (first use)
        await self._flush(session)

    async def commit(self) -> Transcript:
        """Finish the utterance: await the committed text, then rotate the socket.

        Holds ``_api_lock`` for the whole call, so a batch fallback never overlaps
        the realtime commit and no other Scribe request can start meanwhile.  The
        replacement socket is pre-opened only after any batch fallback returned.
        """
        async with self._api_lock:
            return await self._commit_locked()

    async def _commit_locked(self) -> Transcript:
        t0 = time.perf_counter()
        segment = np.concatenate(self._segment) if self._segment else np.zeros(0, dtype=np.int16)
        session: RealtimeSession | None = None
        try:
            session = await self._current()
            await self._flush(session)
            session = await self._current()  # _flush may have reconnected
            self._segment, self._sent_chunks = [], 0
            tr = await session.commit()
        except Exception as exc:  # socket died, could not reconnect, server error, timeout ...
            self._segment, self._sent_chunks = [], 0
            if session is not None:
                await self._retire(session)
            try:
                if self.fallback_to_batch and self._is_voiced(segment):
                    log.warning("%s commit failed (%s: %s); falling back to batch", self.name, type(exc).__name__, exc)
                    tr = await self._batch(segment)  # waits for the retired socket to close first
                    tr.meta["fallback"] = f"batch after {type(exc).__name__}"
                    tr.latency_s = time.perf_counter() - t0
                    return tr
                raise
            finally:
                self._preconnect()
        if self.rotate_sessions:
            await self._retire(session)
        try:
            tr.meta["model_id"] = self.model_id
            tr.meta["audio_s"] = round(len(segment) / self.sample_rate, 3)
            if not tr.text and self.fallback_to_batch and self._is_voiced(segment):
                log.warning(
                    "%s empty commit for voiced %.1f s audio; falling back to batch", self.name, len(segment) / self.sample_rate
                )
                fb = await self._batch(segment)
                fb.meta.update({"fallback": "batch after empty commit", "realtime_meta": tr.meta})
                fb.latency_s = time.perf_counter() - t0
                return fb
            tr.latency_s = time.perf_counter() - t0
            return tr
        finally:
            if self.rotate_sessions:
                self._preconnect()

    async def discard(self, keep_audio: bool = False) -> None:
        """Abandon the utterance in progress on the server side.

        Used by the pipeline when a segment turns out not to be a turn (a blip that
        never reached ``barge_in_min_speech_ms``, a queued utterance that will be
        re-transcribed in batch, or a ``commit()`` that was cancelled by a barge-in).
        The current socket is retired in the background and a fresh one pre-opened,
        so the server's uncommitted buffer can never leak into the next commit.

        ``keep_audio=True`` keeps the client-side copy of the frames fed since the
        last commit: they are re-sent on the fresh socket at the next ``feed()`` /
        ``commit()`` (a barge-in that cancelled the previous commit while the user
        was already talking again).  ``False`` drops them.
        """
        if not keep_audio:
            self._segment = []
        session, self._session = self._session, None
        self._sent_chunks = 0
        if session is not None:
            self._close_in_background(session)
        self._preconnect()

    @property
    def stream_pending_s(self) -> float:
        """Seconds of audio fed since the last commit (not yet committed)."""
        return sum(len(a) for a in self._segment) / self.sample_rate

    async def transcribe_batch(self, pcm: np.ndarray, sample_rate: int = MIC_SAMPLE_RATE) -> Transcript:
        """Transcribe a complete utterance with the batch endpoint (``scribe_v2``).

        The pipeline uses this instead of :meth:`transcribe` (burst realtime, slower)
        whenever the streaming path was not used for an utterance, e.g. two
        utterances merged after a thinking-phase interruption.
        """
        t0 = time.perf_counter()
        async with self._api_lock:  # never alongside a realtime commit
            tr = await self._batch(as_int16_mono(pcm), sample_rate)
        tr.meta["mode"] = "batch"
        tr.latency_s = time.perf_counter() - t0
        return tr

    def _is_voiced(self, pcm: np.ndarray) -> bool:
        return is_voiced(pcm, self.sample_rate, rms_threshold=self.voiced_rms)

    async def _batch(self, pcm: np.ndarray, sample_rate: int | None = None) -> Transcript:
        """One batch request (caller holds ``_api_lock``); retired sockets are closed first."""
        if self._fallback is None:
            # a boxed session (language + secondaries) must not pin the batch call to one language
            pinned = self.language if not self.secondary_languages else None
            self._fallback = ElevenLabsScribeSTT(self.api_key, model_id="scribe_v2", language=pinned)
        await self._await_closing()
        t0 = time.perf_counter()
        tr = await self._fallback.transcribe(pcm, sample_rate or self.sample_rate)
        self.last_batch_s = time.perf_counter() - t0
        return tr

    # --------------------------------------------------------------- protocol
    async def warmup(self) -> None:
        """Open (and keep) a websocket so the first turn skips the handshake."""
        t0 = time.perf_counter()
        self._closed_flag = False
        try:
            if self.fallback_to_batch:  # warm the batch TLS connection first, with no realtime session open
                async with self._api_lock:
                    await self._batch(np.zeros(int(0.4 * self.sample_rate), dtype=np.int16))
            await self._current()
            log.info("%s warmup: %.3f s", self.name, time.perf_counter() - t0)
        except Exception as exc:
            log.warning("%s warmup failed after %.3f s: %s", self.name, time.perf_counter() - t0, exc)

    async def transcribe(self, pcm: np.ndarray, sample_rate: int = MIC_SAMPLE_RATE) -> Transcript:
        """Burst-stream a complete utterance and await the committed transcript."""
        t0 = time.perf_counter()
        pcm = as_int16_mono(pcm)
        audio_s = len(pcm) / sample_rate
        if sample_rate != self.sample_rate:  # a different rate needs a socket with matching audio_format
            await self.close()
            self.sample_rate = sample_rate
        if audio_s * 1000 < self.min_audio_ms:
            return Transcript(
                text="",
                latency_s=time.perf_counter() - t0,
                meta={"model_id": self.model_id, "audio_s": round(audio_s, 3), "mode": "burst", "skipped": "too_short"},
            )
        await self.feed(pcm)
        tr = await self.commit()
        tr.meta["mode"] = "burst"
        tr.latency_s = time.perf_counter() - t0
        return tr

    async def close(self) -> None:
        self._closed_flag = True
        if self._next is not None:
            task, self._next = self._next, None
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            if task.done() and not task.cancelled() and task.exception() is None:
                await task.result().close()
        if self._session is not None:
            session, self._session = self._session, None
            await session.close()
        if self._closing:
            await asyncio.gather(*self._closing, return_exceptions=True)
        if self._fallback is not None:
            await self._fallback.close()
            self._fallback = None
        self._segment, self._sent_chunks = [], 0
