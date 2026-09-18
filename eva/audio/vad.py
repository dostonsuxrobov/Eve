"""Silero VAD (onnxruntime, no torch) and the utterance segmenter built on it.

* :class:`SileroVAD` wraps ``models/silero_vad.onnx`` (Silero v5).  It takes one
  512-sample window (32 ms at 16 kHz) and returns the speech probability while
  carrying the RNN ``state`` and the 64-sample audio context the v5 export
  expects between calls.  (Without the context prefix this export returns
  ~0.0 for everything; verified with onnxruntime.)
* :class:`UtteranceSegmenter` turns a stream of 20 ms int16 frames into
  :class:`SpeechStart` / :class:`SpeechEnd` events using ``PipelineSettings``:
  pre-speech ring buffer, minimum speech length, endpoint silence, maximum
  utterance length, and hysteresis (end threshold = threshold - 0.15).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import MODELS_DIR, PipelineSettings

__all__ = ["SileroVAD", "SpeechStart", "SpeechEnd", "UtteranceSegmenter", "VAD_WINDOW"]

VAD_WINDOW = 512  # samples per Silero window at 16 kHz (32 ms)
_HYSTERESIS = 0.15
_TAIL_KEEP_MS = 150  # trailing silence kept in the emitted utterance


class SileroVAD:
    """Streaming Silero v5 VAD over onnxruntime (CPU, single thread).

    ``vad(window)`` -> speech probability in 0..1.  ``window`` must contain
    exactly 512 samples (16 kHz) or 256 samples (8 kHz), int16 or float32.
    """

    def __init__(self, model_path: str | Path = MODELS_DIR / "silero_vad.onnx", sample_rate: int = 16000) -> None:
        import onnxruntime as ort

        if sample_rate not in (8000, 16000):
            raise ValueError("Silero VAD supports 8000 or 16000 Hz")
        self.sample_rate = sample_rate
        self.window_size = 512 if sample_rate == 16000 else 256
        self.context_size = 64 if sample_rate == 16000 else 32
        self.model_path = Path(model_path)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(str(self.model_path), opts, providers=["CPUExecutionProvider"])
        names = {i.name for i in self._session.get_inputs()}
        if not {"input", "state", "sr"} <= names:
            raise RuntimeError(f"unexpected Silero model inputs: {sorted(names)} (need input/state/sr)")
        self._sr = np.array(sample_rate, dtype=np.int64)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(self.context_size, dtype=np.float32)
        self._buf = np.empty((1, self.context_size + self.window_size), dtype=np.float32)
        self.last_prob = 0.0

    def reset(self) -> None:
        """Forget RNN state and audio context (call between unrelated streams)."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(self.context_size, dtype=np.float32)
        self.last_prob = 0.0

    def __call__(self, window: np.ndarray) -> float:
        window = np.asarray(window)
        if window.ndim != 1:
            window = window.reshape(-1)
        if window.shape[0] != self.window_size:
            raise ValueError(f"expected {self.window_size} samples, got {window.shape[0]}")
        if window.dtype == np.int16:
            x = window.astype(np.float32) / 32768.0
        else:
            x = window.astype(np.float32, copy=False)
        buf = self._buf
        buf[0, : self.context_size] = self._context
        buf[0, self.context_size :] = x
        out, state = self._session.run(None, {"input": buf, "state": self._state, "sr": self._sr})
        self._state = state
        self._context = x[-self.context_size :]
        self.last_prob = float(out[0, 0])
        return self.last_prob


@dataclass
class SpeechStart:
    t: float  # time.perf_counter() when speech was confirmed (after min_speech_ms)


@dataclass
class SpeechEnd:
    t: float  # time.perf_counter() when the endpoint was detected
    pcm: np.ndarray  # int16 utterance incl. pre-speech buffer, trailing silence <= 150 ms
    duration_s: float  # len(pcm) / sample_rate


class UtteranceSegmenter:
    """State machine: 20 ms frames in, SpeechStart / SpeechEnd events out.

    Feed frames of any length (typically 320 samples); they are re-buffered into
    512-sample VAD windows.  ``threshold`` may be changed at runtime (the
    pipeline raises it while the agent speaks when ``echo_guard`` is on).
    """

    def __init__(self, settings: PipelineSettings, vad: SileroVAD | None = None) -> None:
        self.settings = settings
        self.vad = vad or SileroVAD()
        self.sample_rate = self.vad.sample_rate
        self.window = self.vad.window_size
        self.window_ms = 1000.0 * self.window / self.sample_rate
        self._threshold = float(settings.vad_threshold)
        # windows needed to confirm / end speech, rounded up so we never under-count
        self._min_speech_windows = max(1, math.ceil(settings.min_speech_ms / self.window_ms))
        self._endpoint_windows = max(1, math.ceil(settings.endpoint_silence_ms / self.window_ms))
        self._prespeech_samples = int(settings.prespeech_buffer_ms * self.sample_rate / 1000)
        self._max_utt_samples = int(settings.max_utterance_s * self.sample_rate)
        self._tail_keep = int(_TAIL_KEEP_MS * self.sample_rate / 1000)
        self._pending = np.empty(0, np.int16)
        self._ring: list[np.ndarray] = []
        self._ring_samples = 0
        self._utt: list[np.ndarray] = []
        self._utt_samples = 0
        self._last_speech_samples = 0  # utterance length at the last speech-positive window
        self._speech_windows = 0  # speech-positive windows in the current run/utterance
        self._silence_windows = 0
        self._candidate = False  # speech seen but not yet confirmed for min_speech_ms
        self.speaking = False
        self.last_prob = 0.0

    # -- knobs -----------------------------------------------------------------------
    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = float(min(max(value, 0.0), 1.0))

    @property
    def end_threshold(self) -> float:
        return max(self._threshold - _HYSTERESIS, 0.02)

    @property
    def speaking_ms(self) -> float:
        """Milliseconds of speech-positive audio in the current utterance (0 when idle)."""
        if not (self.speaking or self._candidate):
            return 0.0
        return self._speech_windows * self.window_ms

    @property
    def silence_ms(self) -> float:
        """Silence accumulated since the last speech window while speaking."""
        return self._silence_windows * self.window_ms if self.speaking else 0.0

    def reset(self) -> None:
        self.vad.reset()
        self._pending = np.empty(0, np.int16)
        self._ring.clear()
        self._ring_samples = 0
        self._reset_utt()
        self.last_prob = 0.0

    def _reset_utt(self) -> None:
        self._utt = []
        self._utt_samples = 0
        self._last_speech_samples = 0
        self._speech_windows = 0
        self._silence_windows = 0
        self._candidate = False
        self.speaking = False

    # -- feeding ---------------------------------------------------------------------
    def feed(self, frame: np.ndarray) -> list[SpeechStart | SpeechEnd]:
        """Consume one mic frame and return zero or more events."""
        frame = np.asarray(frame)
        if frame.ndim != 1:
            frame = frame.reshape(-1)
        if frame.dtype != np.int16:
            if np.issubdtype(frame.dtype, np.floating):
                frame = np.clip(frame * 32767.0, -32768, 32767).astype(np.int16)
            else:
                frame = frame.astype(np.int16)
        buf = np.concatenate([self._pending, frame]) if self._pending.size else frame
        events: list[SpeechStart | SpeechEnd] = []
        n = self.window
        full = (len(buf) // n) * n
        for i in range(0, full, n):
            ev = self._process_window(buf[i : i + n])
            if ev is not None:
                events.append(ev)
        self._pending = buf[full:].copy()
        return events

    def _push_ring(self, window: np.ndarray) -> None:
        self._ring.append(window)
        self._ring_samples += len(window)
        while self._ring and self._ring_samples - len(self._ring[0]) >= self._prespeech_samples:
            self._ring_samples -= len(self._ring.pop(0))

    def _process_window(self, window: np.ndarray) -> SpeechStart | SpeechEnd | None:
        prob = self.vad(window)
        self.last_prob = prob
        now = time.perf_counter()

        if not self.speaking and not self._candidate:
            if prob > self._threshold:
                # onset: seed the utterance with the pre-speech ring buffer
                pre = np.concatenate(self._ring) if self._ring else np.empty(0, np.int16)
                if len(pre) > self._prespeech_samples:
                    pre = pre[-self._prespeech_samples :]
                self._utt = [pre.copy(), window.copy()] if len(pre) else [window.copy()]
                self._utt_samples = len(pre) + len(window)
                self._last_speech_samples = self._utt_samples
                self._speech_windows = 1
                self._silence_windows = 0
                self._candidate = True
                if self._speech_windows >= self._min_speech_windows:
                    self._candidate, self.speaking = False, True
                    return SpeechStart(t=now)
            else:
                self._push_ring(window.copy())
            return None

        # candidate or speaking: accumulate audio
        self._utt.append(window.copy())
        self._utt_samples += len(window)
        is_speech = prob > (self._threshold if self._candidate else self.end_threshold)

        if self._candidate:
            if is_speech:
                self._speech_windows += 1
                self._last_speech_samples = self._utt_samples
                if self._speech_windows >= self._min_speech_windows:
                    self._candidate, self.speaking = False, True
                    return SpeechStart(t=now)
            else:
                # blip: keep what we buffered as pre-speech context and go idle
                for w in self._utt:
                    self._push_ring(w)
                self._reset_utt()
            return None

        # speaking
        if is_speech:
            self._speech_windows += 1
            self._silence_windows = 0
            self._last_speech_samples = self._utt_samples
        else:
            self._silence_windows += 1

        if self._silence_windows >= self._endpoint_windows:
            return self._finish(now)
        if self._utt_samples >= self._max_utt_samples:
            return self._finish(now)
        return None

    def _tighten(self, full: np.ndarray, last_speech: int) -> int:
        """Back the VAD's last-speech point off over near-silent audio.

        Silero keeps reporting speech for 1-2 windows after the energy has died
        (it looks at context), so walk back in 10 ms hops while the hop is at
        least 45 dB below the utterance's loudest hop, at most 3 windows.
        """
        hop = self.sample_rate // 100
        if last_speech < 2 * hop or len(full) < 2 * hop:
            return last_speech
        x = full[: (len(full) // hop) * hop].astype(np.float32).reshape(-1, hop)
        rms = np.sqrt((x * x).mean(axis=1))
        floor = float(rms.max()) * 10 ** (-45 / 20)
        end = last_speech
        limit = max(0, last_speech - 3 * self.window)
        while end - hop >= limit:
            seg = full[end - hop : end].astype(np.float32)
            if math.sqrt(float((seg * seg).mean())) >= floor:
                break
            end -= hop
        return end

    def _finish(self, now: float) -> SpeechEnd:
        full = np.concatenate(self._utt) if self._utt else np.empty(0, np.int16)
        last_speech = self._tighten(full, self._last_speech_samples)
        end = min(len(full), last_speech + self._tail_keep)
        pcm = full[:end]
        # the audio after the last speech is the true recent past: it becomes the
        # pre-speech context for the next utterance instead of the stale ring
        tail = full[self._last_speech_samples :]
        self._ring.clear()
        self._ring_samples = 0
        if len(tail):
            self._push_ring(tail[-self._prespeech_samples :].copy())
        self._reset_utt()
        return SpeechEnd(t=now, pcm=pcm, duration_s=len(pcm) / self.sample_rate)
