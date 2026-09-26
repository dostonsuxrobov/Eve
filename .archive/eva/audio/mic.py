"""Microphone sources that yield 20 ms int16 mono frames at 16 kHz.

Three sources share one frame shape so the pipeline can be driven by a real
microphone, by a wav file (for benchmarks) or by a scripted sequence of
utterances (for barge-in simulations):

* :class:`Mic`         - sounddevice InputStream -> asyncio.Queue -> ``frames()``
* :class:`FileMic`     - wav / numpy array, optionally paced in real time
* :class:`ScriptedMic` - list of (audio, gap_seconds) items played back to back

Every frame is a fresh ``np.ndarray`` of dtype int16 with exactly
``sample_rate * frame_ms / 1000`` samples (320 for the defaults).
"""
from __future__ import annotations

import asyncio
import math
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["Mic", "FileMic", "ScriptedMic", "LinearResampler", "load_pcm"]


# --------------------------------------------------------------------------- utils
class LinearResampler:
    """Stateful linear-interpolation resampler for streaming int16/float audio.

    Keeps the last input sample and the fractional read position between
    calls so block boundaries are seamless.  Good enough for 44.1/48 kHz
    microphones -> 16 kHz speech; not meant for hi-fi.
    """

    def __init__(self, src_rate: int, dst_rate: int) -> None:
        if src_rate <= 0 or dst_rate <= 0:
            raise ValueError("rates must be positive")
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        self._step = src_rate / dst_rate
        self._pos = 0.0  # fractional read position relative to the buffer start
        self._last: float | None = None  # last input sample (history for interpolation)

    def reset(self) -> None:
        self._pos = 0.0
        self._last = None

    def process(self, x: np.ndarray) -> np.ndarray:
        """Resample one block. Returns float32 samples at ``dst_rate``."""
        x = np.asarray(x)
        if x.ndim != 1:
            x = x.reshape(-1)
        x = x.astype(np.float32, copy=False)
        if x.size == 0:
            return np.empty(0, np.float32)
        if self._last is not None:
            x = np.concatenate(([self._last], x))
        n_in = x.shape[0]
        # read positions strictly below the last sample so idx + 1 is valid
        positions = np.arange(self._pos, n_in - 1, self._step, dtype=np.float64)
        if positions.size:
            idx = positions.astype(np.int64)
            frac = (positions - idx).astype(np.float32)
            y = x[idx] * (1.0 - frac) + x[idx + 1] * frac
            next_pos = positions[-1] + self._step
        else:
            y = np.empty(0, np.float32)
            next_pos = self._pos
        # the next block is prefixed with x[-1], so shift the position accordingly
        self._pos = next_pos - (n_in - 1)
        self._last = float(x[-1])
        return y


def _to_int16(x: np.ndarray) -> np.ndarray:
    """Convert any mono/stereo numeric array to a 1-D int16 array."""
    x = np.asarray(x)
    if x.ndim == 2:
        x = x[:, 0] if x.shape[1] <= 8 else x[0, :]
    if x.dtype == np.int16:
        return np.ascontiguousarray(x)
    if np.issubdtype(x.dtype, np.floating):
        return np.clip(x * 32767.0, -32768, 32767).astype(np.int16)
    if x.dtype == np.int32:
        return (x >> 16).astype(np.int16)
    if x.dtype == np.uint8:
        return ((x.astype(np.int16) - 128) << 8).astype(np.int16)
    return x.astype(np.int16)


def load_pcm(source: str | Path | np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Load a wav file or array as int16 mono at ``sample_rate`` (resampling if needed).

    A numpy array is assumed to already be at ``sample_rate``; float arrays are
    expected in -1..1 and are converted to int16.
    """
    if isinstance(source, np.ndarray):
        return _to_int16(source)
    import soundfile as sf  # local import keeps module import cheap

    data, sr = sf.read(str(source), dtype="int16", always_2d=True)
    pcm = np.ascontiguousarray(data[:, 0])
    if sr != sample_rate:
        y = LinearResampler(sr, sample_rate).process(pcm.astype(np.float32))
        pcm = np.clip(y, -32768, 32767).astype(np.int16)
    return pcm


def _mic_hint() -> str:
    """Extra diagnosis for the common Windows failure: Store Python has no microphone capability."""
    import sys

    base = getattr(sys, "_base_executable", sys.executable) or ""
    if "WindowsApps" in base:
        return (
            "\n  This interpreter is the Microsoft Store Python (a packaged app without the 'microphone' capability), "
            "so Windows denies audio capture to it. Either allow it under Settings > Privacy & security > Microphone "
            "('Python 3.x' entry) or rebuild the venv on a non-Store CPython (uv python install 3.13)."
        )
    return ""


def _frame_samples(sample_rate: int, frame_ms: int) -> int:
    n = sample_rate * frame_ms / 1000
    if n != int(n) or n <= 0:
        raise ValueError(f"frame_ms={frame_ms} is not a whole number of samples at {sample_rate} Hz")
    return int(n)


async def _paced_frames(
    pcm: np.ndarray, frame_samples: int, frame_s: float, realtime: bool
) -> AsyncIterator[np.ndarray]:
    """Yield consecutive frames of ``pcm`` (zero-padded to a whole frame), paced or not."""
    n_frames = math.ceil(len(pcm) / frame_samples)
    if len(pcm) < n_frames * frame_samples:
        pcm = np.concatenate([pcm, np.zeros(n_frames * frame_samples - len(pcm), np.int16)])
    t0 = time.perf_counter()
    for i in range(n_frames):
        if realtime:
            target = t0 + i * frame_s
            delay = target - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
        else:
            await asyncio.sleep(0)  # let other tasks run
        yield pcm[i * frame_samples : (i + 1) * frame_samples].copy()


# ------------------------------------------------------------------------------ Mic
class Mic:
    """Real microphone.  ``async for frame in mic.frames(): ...``

    The sounddevice callback only copies the block and hands it to the event
    loop via ``call_soon_threadsafe``; resampling / re-framing (when the device
    refuses 16 kHz) and queueing happen on the loop thread, so the audio
    callback never blocks.  ``start()`` must be called from inside a running
    event loop (``frames()`` calls it for you if you forgot).
    """

    def __init__(self, device: int | None = None, sample_rate: int = 16000, frame_ms: int = 20) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = _frame_samples(sample_rate, frame_ms)
        self.native_rate: int = sample_rate  # rate the device was actually opened at
        self.dropped_frames = 0  # frames discarded because the consumer fell behind
        self.callback_errors = 0  # PortAudio status flags seen (overflow etc.)
        self._max_queue = max(50, int(10_000 / frame_ms))  # ~10 s of audio
        self._stream: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[np.ndarray | None] | None = None
        self._resampler: LinearResampler | None = None
        self._pending = np.empty(0, np.int16)
        self._closed = False

    # -- lifecycle ---------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._stream is not None and not self._closed

    def start(self) -> None:
        """Open the input stream. Idempotent. Needs a running event loop."""
        if self._stream is not None:
            return
        import sounddevice as sd

        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=self._max_queue)
        self._closed = False
        self._pending = np.empty(0, np.int16)

        rate = self.sample_rate
        try:
            sd.check_input_settings(device=self.device, channels=1, samplerate=rate, dtype="int16")
            self._resampler = None
        except Exception:
            info = sd.query_devices(self.device, "input")
            rate = int(round(info["default_samplerate"]))
            self._resampler = LinearResampler(rate, self.sample_rate)
        self.native_rate = rate
        blocksize = int(round(rate * self.frame_ms / 1000))

        kwargs: dict[str, Any] = dict(
            device=self.device,
            channels=1,
            samplerate=rate,
            dtype="int16",
            blocksize=blocksize,
            callback=self._callback,
        )
        try:
            self._stream = sd.InputStream(latency="low", **kwargs)
            self._stream.start()
        except Exception:
            try:
                self._stream = sd.InputStream(**kwargs)
                self._stream.start()
            except Exception as e:
                self._stream = None
                raise RuntimeError(f"could not open microphone (device={self.device}): {e!r}" + _mic_hint()) from e

    def stop(self) -> None:
        """Stop the stream and terminate ``frames()``. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        self._post(None)

    async def __aenter__(self) -> "Mic":
        self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.stop()

    # -- audio thread -> loop thread ---------------------------------------------
    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if status:
            self.callback_errors += 1
        # copy: PortAudio reuses the buffer after we return
        self._post(indata[:, 0].copy())

    def _post(self, block: np.ndarray | None) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._ingest, block)
        except RuntimeError:  # loop closing
            pass

    def _ingest(self, block: np.ndarray | None) -> None:
        """Runs on the event-loop thread: resample, re-frame and enqueue."""
        q = self._queue
        if q is None:
            return
        if block is None:
            self._put(None)
            return
        if self._resampler is not None:
            y = self._resampler.process(block)
            block = np.clip(y, -32768, 32767).astype(np.int16)
        if self._pending.size:
            block = np.concatenate([self._pending, block])
        n = self.frame_samples
        full = (len(block) // n) * n
        for i in range(0, full, n):
            self._put(block[i : i + n])
        self._pending = block[full:]

    def _put(self, item: np.ndarray | None) -> None:
        q = self._queue
        if q is None:
            return
        if q.full():
            try:
                q.get_nowait()
                self.dropped_frames += 1
            except asyncio.QueueEmpty:
                pass
        q.put_nowait(item)

    # -- consumer -----------------------------------------------------------------
    async def frames(self) -> AsyncIterator[np.ndarray]:
        """Yield int16 frames of exactly ``frame_samples`` until :meth:`stop`."""
        if self._stream is None and not self._closed:
            self.start()
        q = self._queue
        assert q is not None
        while True:
            item = await q.get()
            if item is None:
                return
            yield item


# -------------------------------------------------------------------------- FileMic
class FileMic:
    """Plays a wav file / array as mic frames, with leading and trailing silence.

    ``realtime=True`` paces frames on the wall clock (one frame per ``frame_ms``);
    ``realtime=False`` yields as fast as the consumer takes them (benchmarks).
    """

    def __init__(
        self,
        source: str | Path | np.ndarray,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        realtime: bool = True,
        leading_silence_s: float = 0.5,
        trailing_silence_s: float = 2.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = _frame_samples(sample_rate, frame_ms)
        self.realtime = realtime
        self.speech = load_pcm(source, sample_rate)
        lead = np.zeros(int(round(leading_silence_s * sample_rate)), np.int16)
        tail = np.zeros(int(round(trailing_silence_s * sample_rate)), np.int16)
        self.pcm = np.concatenate([lead, self.speech, tail])
        self.speech_start_s = len(lead) / sample_rate
        self.speech_end_s = (len(lead) + len(self.speech)) / sample_rate
        self._stopped = False

    @property
    def duration_s(self) -> float:
        return len(self.pcm) / self.sample_rate

    def start(self) -> None:  # API symmetry with Mic
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True

    async def __aenter__(self) -> "FileMic":
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.stop()

    async def frames(self) -> AsyncIterator[np.ndarray]:
        async for frame in _paced_frames(self.pcm, self.frame_samples, self.frame_ms / 1000, self.realtime):
            if self._stopped:
                return
            yield frame


# ----------------------------------------------------------------------- ScriptedMic
class ScriptedMic:
    """Sequence of ``(audio, gap_seconds)`` items streamed back to back in real time.

    Each item's audio is followed by ``gap_seconds`` of silence.  Used to simulate
    a user who talks, waits, and talks again (or barges in) with wall-clock timing.
    ``item_starts_s`` gives the offset of each item from the first frame so a
    simulation can correlate events with the script.
    """

    def __init__(
        self,
        items: list[tuple[str | Path | np.ndarray, float]],
        sample_rate: int = 16000,
        frame_ms: int = 20,
        realtime: bool = True,
        leading_silence_s: float = 0.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = _frame_samples(sample_rate, frame_ms)
        self.realtime = realtime
        parts: list[np.ndarray] = []
        self.item_starts_s: list[float] = []
        pos = int(round(leading_silence_s * sample_rate))
        if pos:
            parts.append(np.zeros(pos, np.int16))
        for source, gap in items:
            pcm = load_pcm(source, sample_rate)
            self.item_starts_s.append(pos / sample_rate)
            parts.append(pcm)
            pos += len(pcm)
            gap_n = int(round(max(0.0, gap) * sample_rate))
            if gap_n:
                parts.append(np.zeros(gap_n, np.int16))
                pos += gap_n
        self.pcm = np.concatenate(parts) if parts else np.empty(0, np.int16)
        self._stopped = False
        self.t0: float | None = None  # perf_counter of the first frame

    @property
    def duration_s(self) -> float:
        return len(self.pcm) / self.sample_rate

    def start(self) -> None:
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True

    async def __aenter__(self) -> "ScriptedMic":
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.stop()

    async def frames(self) -> AsyncIterator[np.ndarray]:
        self.t0 = time.perf_counter()
        async for frame in _paced_frames(self.pcm, self.frame_samples, self.frame_ms / 1000, self.realtime):
            if self._stopped:
                return
            yield frame

