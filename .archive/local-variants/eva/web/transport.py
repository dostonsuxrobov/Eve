"""Mic source and player over a WebSocket, for the phone (or any browser) client.

The pipeline only needs two things from the audio side: an async iterator of 20 ms
int16 16 kHz mic frames, and a ``PlayerLike`` (``write`` / ``stop`` / ``mark`` /
``wait_until_done`` / ``played_samples`` / ``is_active`` / ``buffered_seconds``).
Here both are backed by one socket:

* :class:`WebMic` takes binary messages of int16 PCM at whatever rate the browser's
  AudioContext runs (48 kHz on most phones; the client says so in its ``hello``),
  resamples to 16 kHz and re-frames to 20 ms.
* :class:`WebPlayer` sends Eva's audio (int16 24 kHz) as binary messages the moment
  the pipeline writes it; the browser queues and plays it. ``stop()`` sends a flush
  message and returns how many samples the phone has *probably* played (elapsed
  time since the first write after ``mark()``, minus an assumed transport latency),
  which the pipeline maps onto the words the user heard. The browser reports its
  real played count in its ``stopped`` reply, which corrects the estimate for the
  next time. It keeps a timestamped history of what it sent, so the echo detector
  works the same way as with the local player, with a wider lag window.

Wire protocol (both directions on one socket):

client -> server
    text ``{"type": "hello", "sampleRate": 48000}`` once, then binary int16 PCM
    frames at that rate; text ``{"type": "stopped", "played": N}`` after a flush;
    text ``{"type": "interrupt"}`` (a tap on the stop button); ``{"type": "bye"}``.
server -> client
    text ``{"type": "config", "sampleRate": 24000, "prebufferS": 0.25, "roomToneDbfs": null}``;
    binary int16 PCM 24 kHz; text ``{"type": "stop"}`` flush now; ``{"type": "eot"}`` the
    reply's audio is complete (drain the jitter buffer); ``{"type": "event", "name": ...,
    "data": ...}`` every pipeline event the UI shows.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from typing import Any, AsyncIterator, Awaitable, Callable

import numpy as np

from ..audio.mic import LinearResampler
from ..interfaces import MIC_SAMPLE_RATE

log = logging.getLogger("eva.web")

Send = Callable[[str | bytes], Awaitable[None]]
FRAME_MS = 20
TRANSPORT_LATENCY_S = 0.20  # assumed socket + output latency for the played estimate
PREBUFFER_S = 0.25  # the page holds this much before it starts a reply (Wi-Fi jitter); added to the estimate


class WebMic:
    """Frames from the browser: resampled to 16 kHz, re-framed to 20 ms."""

    def __init__(self, sample_rate: int = MIC_SAMPLE_RATE) -> None:
        self.frame_samples = MIC_SAMPLE_RATE * FRAME_MS // 1000
        self._queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=500)
        self._resampler: LinearResampler | None = None
        self._pending = np.zeros(0, dtype=np.int16)
        self._closed = False
        self.dropped = 0
        self.received_samples = 0
        self.set_rate(sample_rate)

    def set_rate(self, sample_rate: int) -> None:
        self.sample_rate = int(sample_rate)
        self._resampler = None if self.sample_rate == MIC_SAMPLE_RATE else LinearResampler(self.sample_rate, MIC_SAMPLE_RATE)

    def push(self, data: bytes) -> None:
        """A binary message from the client (int16 mono at ``sample_rate``)."""
        if self._closed or len(data) < 2:
            return
        block = np.frombuffer(data[: len(data) - (len(data) % 2)], dtype=np.int16)
        self.received_samples += int(block.size)
        if self._resampler is not None:
            block = np.clip(self._resampler.process(block), -32768, 32767).astype(np.int16)
        if self._pending.size:
            block = np.concatenate([self._pending, block])
        n = self.frame_samples
        full = (block.size // n) * n
        for i in range(0, full, n):
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                    self.dropped += 1
                except asyncio.QueueEmpty:
                    pass
            self._queue.put_nowait(block[i : i + n].copy())
        self._pending = block[full:].copy()

    def stop(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put_nowait(None)

    async def frames(self) -> AsyncIterator[np.ndarray]:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item


class WebPlayer:
    """``PlayerLike`` that streams to the browser and estimates what was heard."""

    def __init__(self, send: Send, sample_rate: int, *, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self.sample_rate = int(sample_rate)
        self._send = send
        self._loop = loop or asyncio.get_event_loop()
        self._written = 0  # samples sent since mark()
        self._first_write_t: float | None = None
        self._last_write_t: float | None = None
        self._stopped_at: float | None = None
        self._closed = False
        self._tasks: set[asyncio.Task[Any]] = set()
        self._history: deque[tuple[float, bytes]] = deque()
        self.played_history_s = 3.0
        self.echo_max_lag_s = 0.8  # network + jitter buffer: wider than the local player's window
        self.reported_played: int | None = None  # the browser's own count after the last stop
        self.latency_s = TRANSPORT_LATENCY_S + PREBUFFER_S
        self.prebuffer_s = PREBUFFER_S
        self.room_tone_dbfs: float | None = None  # sent to the page: its idle fill level

    # -- sending ---------------------------------------------------------------------
    def _post(self, payload: str | bytes) -> None:
        if self._closed:
            return
        task = self._loop.create_task(self._safe_send(payload))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _safe_send(self, payload: str | bytes) -> None:
        try:
            await self._send(payload)
        except Exception as e:  # the socket went away: the session ends on its own
            log.debug("web send failed: %s", e)

    # -- PlayerLike ------------------------------------------------------------------
    def start(self) -> None:
        self._post(json.dumps({
            "type": "config", "sampleRate": self.sample_rate, "roomToneDbfs": self.room_tone_dbfs,
            "prebufferS": self.prebuffer_s,
        }))

    def end_turn(self) -> None:
        """No more audio for this reply: the page may drain what it holds below its prebuffer."""
        self._post(json.dumps({"type": "eot"}))

    def write(self, pcm: bytes | bytearray | memoryview | np.ndarray) -> None:
        if self._closed:
            return
        if isinstance(pcm, np.ndarray):
            pcm = pcm.astype(np.int16).tobytes()
        data = bytes(pcm)
        if not data:
            return
        now = time.perf_counter()
        if self._first_write_t is None:
            self._first_write_t = now
        self._last_write_t = now
        self._written += len(data) // 2
        self._history.append((now, data))
        horizon = now - self.played_history_s
        while self._history and self._history[0][0] < horizon:
            self._history.popleft()
        self._post(data)

    def played_since(self, t: float) -> list[tuple[float, bytes]]:
        return [item for item in self._history if item[0] >= t]

    @property
    def played_samples(self) -> int:
        """Samples the phone has probably played since mark(): time-based, capped by what was sent."""
        if self._first_write_t is None:
            return 0
        end = self._stopped_at if self._stopped_at is not None else time.perf_counter()
        elapsed = max(0.0, end - self._first_write_t - self.latency_s)
        return min(self._written, int(elapsed * self.sample_rate))

    @property
    def buffered_seconds(self) -> float:
        return max(0.0, (self._written - self.played_samples) / self.sample_rate)

    @property
    def is_active(self) -> bool:
        return self._written > self.played_samples

    def mark(self) -> None:
        self._written = 0
        self._first_write_t = None
        self._stopped_at = None
        self.reported_played = None

    def stop(self) -> int:
        played = self.played_samples
        self._stopped_at = time.perf_counter()
        self._post(json.dumps({"type": "stop"}))
        self._written = played  # nothing after this will be heard
        return played

    async def wait_until_done(self, poll_s: float = 0.02) -> None:
        while self.is_active and not self._closed:
            await asyncio.sleep(poll_s)
        if not self._closed:
            await asyncio.sleep(0.05)

    def note_stopped(self, played: int) -> None:
        """The browser's own count after a flush: refine the latency estimate for next time."""
        self.reported_played = int(played)

    def send_event(self, name: str, data: dict[str, Any]) -> None:
        self._post(json.dumps({"type": "event", "name": name, "data": data}, default=str))

    def close(self) -> None:
        self._closed = True
        for t in list(self._tasks):
            t.cancel()


__all__ = ["WebMic", "WebPlayer"]
