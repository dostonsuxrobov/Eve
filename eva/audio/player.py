"""Low-latency int16 mono output with instant ``stop()`` and played-sample accounting.

The sounddevice callback pulls from a ``bytearray`` guarded by a ``threading.Lock``
and writes zeros when the buffer is empty, so the stream never underruns and
``stop()`` (which clears the buffer under the lock) takes effect at the next
callback, i.e. within one block (``block_ms``) plus the host API's own output
latency.  ``played_samples`` counts real (non-silence) samples handed to the
device since the last :meth:`mark`, which the pipeline uses to work out how much
of an utterance the user actually heard before barging in.
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import Any

import numpy as np

__all__ = ["Player"]


class Player:
    """Streaming PCM player.  ``write()`` from any thread, ``stop()`` drops everything."""

    def __init__(self, sample_rate: int, device: int | None = None, block_ms: int = 20) -> None:
        self.sample_rate = int(sample_rate)
        self.device = device
        self.block_ms = int(block_ms)
        self.block_samples = max(1, int(round(self.sample_rate * self.block_ms / 1000)))
        self._lock = threading.Lock()
        self._buf = bytearray()
        self._played = 0  # samples pulled by the callback since mark()
        self._played_total = 0
        self._first_audio_t: float | None = None  # perf_counter of the first real block since mark()
        self._last_audio_t: float | None = None  # perf_counter of the most recent real block
        self._last_callback_t: float | None = None  # perf_counter of the most recent callback (any)
        self.callbacks = 0
        self._mark_t: float | None = None
        self.callback_errors = 0
        self._stream: Any = None
        self._closed = False
        # What actually went to the device, with the time it left: (perf_counter, int16 bytes)
        # per callback block, about the last `played_history_s` seconds. The echo detector
        # (eva.audio.echo) correlates mic frames against this.
        self.played_history_s = 2.0
        self._history: deque[tuple[float, bytes]] = deque()
        # Idle fill: room tone at this level instead of digital zero (None = zeros), so the
        # background does not switch on and off with her replies and the amp never sleeps.
        self.room_tone_dbfs: float | None = None
        self._tone: np.ndarray | None = None
        self._tone_pos = 0

    # -- lifecycle -------------------------------------------------------------------
    @staticmethod
    def _wasapi_default_output() -> int | None:
        """Index of the WASAPI endpoint for the default output device (Windows), else None."""
        import sounddevice as sd

        try:
            for api in sd.query_hostapis():
                if "WASAPI" in str(api.get("name", "")):
                    d = api.get("default_output_device", -1)
                    return int(d) if d is not None and int(d) >= 0 else None
        except Exception:
            return None
        return None

    def start(self) -> None:
        """Open and start the output stream (idempotent).

        With ``device=None`` the WASAPI endpoint of the default output device is
        tried first (measured ~40 ms output latency here vs ~100 ms for MME), with
        PortAudio's auto-convert so any TTS sample rate works in shared mode.
        Falls back to the plain default device.
        """
        if self._stream is not None:
            return
        import sounddevice as sd

        base: dict[str, Any] = dict(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.block_samples,
            callback=self._callback,
        )
        attempts: list[dict[str, Any]] = []
        if self.device is None:
            w = self._wasapi_default_output()
            if w is not None:
                attempts.append(dict(base, device=w, latency="low", extra_settings=sd.WasapiSettings(auto_convert=True)))
        attempts.append(dict(base, device=self.device, latency="low"))
        attempts.append(dict(base, device=self.device))
        last: Exception | None = None
        for kw in attempts:
            try:
                stream = sd.OutputStream(**kw)
                stream.start()
                self._stream = stream
                break
            except Exception as e:  # try the next configuration
                last = e
        if self._stream is None:
            raise RuntimeError(f"could not open output stream: {last!r}")
        self._closed = False

    @property
    def device_name(self) -> str:
        """Human-readable name + host API of the open device."""
        s = self._stream
        if s is None:
            return "(closed)"
        import sounddevice as sd

        info = sd.query_devices(s.device)
        return f"{info['name']} [{sd.query_hostapis(info['hostapi'])['name']}]"

    def close(self) -> None:
        """Drop buffered audio and close the device."""
        self._closed = True
        with self._lock:
            self._buf.clear()
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    def __enter__(self) -> "Player":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def output_latency_s(self) -> float | None:
        """PortAudio's reported output latency for the open stream (seconds)."""
        s = self._stream
        return float(s.latency) if s is not None else None

    # -- audio thread ------------------------------------------------------------------
    def _callback(self, outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if status:
            self.callback_errors += 1
        want = frames * 2
        with self._lock:
            self.callbacks += 1
            self._last_callback_t = time.perf_counter()
            n = min(want, len(self._buf))
            chunk = bytes(self._buf[:n]) if n else b""
            if n:
                del self._buf[:n]
            samples = n // 2
            self._played += samples
            self._played_total += samples
            if samples:
                now = time.perf_counter()
                if self._first_audio_t is None:
                    self._first_audio_t = now
                self._last_audio_t = now
                self._history.append((now, chunk))
                horizon = now - self.played_history_s
                while self._history and self._history[0][0] < horizon:
                    self._history.popleft()
        if samples:
            outdata[:samples, 0] = np.frombuffer(chunk[: samples * 2], dtype=np.int16)
        if samples < frames:
            outdata[samples:, 0] = self._idle_fill(frames - samples)

    def _idle_fill(self, n: int) -> np.ndarray | int:
        """Room tone for ``n`` samples of idle output, or 0 (zeros)."""
        if self.room_tone_dbfs is None:
            return 0
        if self._tone is None:
            from .envelope import room_tone

            self._tone = np.frombuffer(room_tone(2000, self.sample_rate, self.room_tone_dbfs), dtype=np.int16)
        out = np.empty(n, dtype=np.int16)
        pos = self._tone_pos
        filled = 0
        while filled < n:
            take = min(n - filled, self._tone.size - pos)
            out[filled : filled + take] = self._tone[pos : pos + take]
            filled += take
            pos = (pos + take) % self._tone.size
        self._tone_pos = pos
        return out

    # -- producer API --------------------------------------------------------------------
    def write(self, pcm: bytes | bytearray | memoryview | np.ndarray) -> None:
        """Queue int16 mono PCM bytes. Thread-safe; returns immediately."""
        if self._closed:
            return
        if isinstance(pcm, np.ndarray):
            if pcm.dtype != np.int16:
                pcm = np.clip(pcm * 32767.0, -32768, 32767).astype(np.int16) if np.issubdtype(pcm.dtype, np.floating) else pcm.astype(np.int16)
            pcm = pcm.tobytes()
        with self._lock:
            self._buf += pcm

    def mark(self) -> None:
        """Start counting played samples for a new utterance."""
        with self._lock:
            self._played = 0
            self._first_audio_t = None
            self._last_audio_t = None
            self._mark_t = time.perf_counter()

    def stop(self) -> int:
        """Discard all buffered audio now. Returns samples played since mark()."""
        with self._lock:
            self._buf.clear()
            return self._played

    # -- introspection -------------------------------------------------------------------
    @property
    def played_samples(self) -> int:
        with self._lock:
            return self._played

    @property
    def played_seconds(self) -> float:
        return self.played_samples / self.sample_rate

    @property
    def first_audio_t(self) -> float | None:
        """perf_counter() when the first real samples since mark() went to the device."""
        with self._lock:
            return self._first_audio_t

    @property
    def last_audio_t(self) -> float | None:
        with self._lock:
            return self._last_audio_t

    @property
    def last_callback_t(self) -> float | None:
        """perf_counter() of the most recent device callback, real audio or silence."""
        with self._lock:
            return self._last_callback_t

    def played_since(self, t: float) -> list[tuple[float, bytes]]:
        """Blocks handed to the device at or after perf_counter ``t`` (newest last)."""
        with self._lock:
            return [item for item in self._history if item[0] >= t]

    @property
    def buffered_samples(self) -> int:
        with self._lock:
            return len(self._buf) // 2

    @property
    def buffered_seconds(self) -> float:
        return self.buffered_samples / self.sample_rate

    @property
    def is_active(self) -> bool:
        """True while buffered audio remains to be played."""
        with self._lock:
            return len(self._buf) >= 2

    async def wait_until_done(self, poll_s: float = 0.01) -> None:
        """Resolve once the buffer has drained and the last block has left the callback."""
        while self.is_active and not self._closed:
            await asyncio.sleep(poll_s)
        # the final block was handed to the device at most one block ago; let it play out
        if not self._closed:
            await asyncio.sleep(self.block_ms / 1000)
