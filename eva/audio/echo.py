"""Is what the microphone hears what the speakers are playing?

Through speakers Eva's own voice reaches the mic. The laptop's echo canceller
removes most of it, but what leaks is enough for the VAD to call it speech, for a
barge-in to cut her off, and for the STT to transcribe the garbled residue as a
"user" turn she then answers (a live session on 2026-09-19 looped like that for
twelve turns). Text heuristics catch most of it; this catches it at the signal.

:class:`EchoDetector` keeps the last second of what the player actually handed to
the device (``Player.played_since``), resampled to the mic rate, and for a probe of
recent mic audio computes the peak normalised cross-correlation against that
reference over acoustic lags of ``min_lag_s`` .. ``max_lag_s``. Echo correlates
(after AEC the residual is still a filtered copy of the playback); the user's own
voice over the top does not. The verdict is a score in 0..1 and a boolean at
``threshold``. Cost: one FFT correlation per call, well under a millisecond.

Limits: a loud user talking exactly over her may still read as partly echo (their
voice adds uncorrelated energy, which lowers the score, so this errs towards
"user"); playback that is pure silence gives no reference and the score is 0.
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

__all__ = ["EchoDetector", "EchoVerdict"]


class EchoVerdict:
    __slots__ = ("score", "lag_s", "is_echo", "reference_s")

    def __init__(self, score: float, lag_s: float, is_echo: bool, reference_s: float) -> None:
        self.score = score
        self.lag_s = lag_s
        self.is_echo = is_echo
        self.reference_s = reference_s

    def __repr__(self) -> str:
        return f"EchoVerdict(score={self.score:.2f}, lag={self.lag_s * 1000:.0f}ms, echo={self.is_echo})"


class EchoDetector:
    def __init__(
        self,
        player: Any,
        mic_rate: int = 16_000,
        *,
        probe_s: float = 0.16,
        min_lag_s: float = 0.0,
        max_lag_s: float = 0.35,
        threshold: float = 0.30,
    ) -> None:
        self.player = player
        self.mic_rate = mic_rate
        self.probe_n = int(probe_s * mic_rate)
        self.min_lag_s = min_lag_s
        self.max_lag_s = max_lag_s
        self.threshold = threshold
        self._mic = np.zeros(0, dtype=np.float32)
        self.last: EchoVerdict | None = None

    # ------------------------------------------------------------- mic input
    def push_mic(self, frame: np.ndarray) -> None:
        """Append a mic frame (int16 or float, mic rate); keeps the last ``probe_s``."""
        x = np.asarray(frame)
        x = (x.astype(np.float32) / 32768.0) if x.dtype == np.int16 else x.astype(np.float32)
        self._mic = np.concatenate([self._mic, x])[-self.probe_n :]

    # -------------------------------------------------------------- reference
    def _reference(self, now: float) -> np.ndarray:
        """Playback that could be arriving at the mic now: [now - max_lag - probe, now - min_lag], at mic rate."""
        span = self.max_lag_s + self.probe_n / self.mic_rate + 0.05
        blocks = [b for b in self.player.played_since(now - span) if b[0] <= now]
        if not blocks:
            return np.zeros(0, dtype=np.float32)
        raw = np.frombuffer(b"".join(b for _, b in blocks), dtype=np.int16).astype(np.float32) / 32768.0
        sr = int(getattr(self.player, "sample_rate", self.mic_rate))
        if sr != self.mic_rate and raw.size > 1:
            n_out = int(raw.size * self.mic_rate / sr)
            raw = np.interp(np.linspace(0.0, raw.size - 1, n_out, dtype=np.float64), np.arange(raw.size), raw).astype(np.float32)
        return raw

    # ---------------------------------------------------------------- verdict
    def check(self, now: float | None = None) -> EchoVerdict:
        """Score the current mic probe against the recent playback."""
        now = time.perf_counter() if now is None else now
        probe = self._mic
        ref = self._reference(now)
        verdict = EchoVerdict(0.0, 0.0, False, ref.size / self.mic_rate)
        if probe.size < self.probe_n // 2 or ref.size < probe.size:
            self.last = verdict
            return verdict
        p = probe - probe.mean()
        r = ref - ref.mean()
        pn = float(np.sqrt((p * p).sum()))
        if pn < 1e-6 or float(np.abs(r).max()) < 1e-4:
            self.last = verdict
            return verdict
        n = 1 << int(math.ceil(math.log2(r.size + p.size)))
        # cross-correlation of the reference with the probe at every lag, via FFT
        corr = np.fft.irfft(np.fft.rfft(r, n) * np.conj(np.fft.rfft(p, n)), n)[: r.size - p.size + 1]
        # normalise per lag by the reference window energy so a loud stretch cannot win by itself
        csum = np.concatenate([[0.0], np.cumsum(r * r)])
        win = csum[p.size : p.size + corr.size] - csum[: corr.size]
        norm = np.sqrt(np.maximum(win, 1e-9)) * pn
        ncc = np.abs(corr) / norm
        k = int(np.argmax(ncc))
        score = float(min(1.0, ncc[k]))
        # lag: the probe ends at `now`; reference index k+p.size is where the probe aligned
        lag_s = (r.size - (k + p.size)) / self.mic_rate
        verdict = EchoVerdict(score, lag_s, score >= self.threshold and self.min_lag_s <= lag_s <= self.max_lag_s + 0.05, ref.size / self.mic_rate)
        self.last = verdict
        return verdict
