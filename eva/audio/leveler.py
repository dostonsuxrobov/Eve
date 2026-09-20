"""Loudness leveling across TTS sources (model x voice).

Measured on the ``maya`` preset (voiced RMS, 50 ms hops above -41 dBFS): the Flash
first-chunk model renders at -21 dBFS, ElevenLabs v3 on the English voice at -16.6,
the Russian voice at -25.4, and a ``[quiet]`` sentence on v3 at -22. So every reply
opened 4-5 dB quieter than its second sentence and Russian sat 9 dB under English:
the "volume dials up from the start to the middle of every reply" complaint.

:class:`Leveler` learns the typical level of each *source* (an EMA over finished
clips, keyed by ``"model/voice"``) and scales the next clip of that source towards a
common target. The gain is fixed for the whole clip, so nothing pumps or ramps
inside a sentence, and it is derived from the source's typical level rather than
the clip's own, so an intentionally quiet sentence stays quieter than its
neighbours (expressive dynamics survive; only the offsets between sources go). A
source that has never been heard starts from a seed level or unity gain. Gain is
bounded, and a soft knee above ``knee`` of full scale keeps a raised voice from
hard-clipping.

All audio is little-endian int16 mono PCM bytes.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["Leveler", "ClipLeveler", "voiced_rms_dbfs"]

FULL_SCALE = 32767.0
_HOP_MS = 50
_VOICED_DBFS = -41.0  # a 50 ms hop above this counts as speech


def voiced_rms_dbfs(pcm: bytes | np.ndarray, sample_rate: int) -> tuple[float | None, float]:
    """(mean voiced RMS in dBFS or None if nothing voiced, peak as a fraction of full scale)."""
    a = np.frombuffer(pcm, dtype=np.int16) if isinstance(pcm, (bytes, bytearray, memoryview)) else np.asarray(pcm)
    a = a.astype(np.float32)
    if a.size == 0:
        return None, 0.0
    hop = max(1, sample_rate * _HOP_MS // 1000)
    n = (a.size // hop) * hop
    peak = float(np.abs(a).max()) / FULL_SCALE
    if n == 0:
        return None, peak
    rms = np.sqrt((a[:n].reshape(-1, hop) ** 2).mean(axis=1))
    voiced = rms[rms > FULL_SCALE * 10 ** (_VOICED_DBFS / 20)]
    if voiced.size == 0:
        return None, peak
    return 20 * math.log10(float(voiced.mean()) / FULL_SCALE), peak


_CEILING = 0.98  # the knee's asymptote, a little under full scale


def _soft_knee(a: np.ndarray, knee: float) -> np.ndarray:
    """Compress |x| above ``knee`` (fraction of full scale) with a tanh curve into (knee, 0.98)."""
    lim = knee * FULL_SCALE
    over = np.abs(a) > lim
    if not over.any():
        return a
    x = a[over]
    span = (_CEILING - knee) * FULL_SCALE
    a[over] = np.sign(x) * (lim + span * np.tanh((np.abs(x) - lim) / span))
    return a


class Leveler:
    """Per-source loudness memory; hand out a :class:`ClipLeveler` per clip."""

    def __init__(
        self,
        sample_rate: int,
        target_dbfs: float = -19.0,
        seeds_dbfs: dict[str, float] | None = None,
        *,
        max_gain_db: float = 7.0,
        min_gain_db: float = -6.0,
        alpha: float = 0.35,
        knee: float = 0.8,
    ) -> None:
        self.sample_rate = sample_rate
        self.target_dbfs = target_dbfs
        self.seeds = dict(seeds_dbfs or {})
        self.max_gain = 10 ** (max_gain_db / 20)
        self.min_gain = 10 ** (min_gain_db / 20)
        self.alpha = alpha
        self.knee = knee
        self.levels: dict[str, float] = {}  # learned voiced level per source (dBFS)
        self.clips = 0

    def level_for(self, key: str) -> float | None:
        """Learned level, else the most specific seed ("model/voice", then "model")."""
        if key in self.levels:
            return self.levels[key]
        if key in self.seeds:
            return self.seeds[key]
        head = key.split("/", 1)[0]
        return self.seeds.get(head)

    def gain_for(self, key: str) -> float:
        level = self.level_for(key)
        if level is None:
            return 1.0
        return float(min(self.max_gain, max(self.min_gain, 10 ** ((self.target_dbfs - level) / 20))))

    def begin(self, key: str) -> "ClipLeveler":
        return ClipLeveler(self, key, self.gain_for(key))

    def _learn(self, key: str, level_dbfs: float) -> None:
        prev = self.levels.get(key)
        self.levels[key] = level_dbfs if prev is None else prev + self.alpha * (level_dbfs - prev)
        self.clips += 1


class ClipLeveler:
    """Scales one clip's chunks by a fixed gain and reports the clip's level at the end."""

    def __init__(self, owner: Leveler, key: str, gain: float) -> None:
        self.owner = owner
        self.key = key
        self.gain = gain
        self._carry = b""
        self._parts: list[np.ndarray] = []

    def process(self, chunk: bytes) -> bytes:
        data = self._carry + chunk
        cut = len(data) - (len(data) % 2)
        data, self._carry = data[:cut], data[cut:]
        if not data:
            return b""
        a = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        self._parts.append(a)
        if self.gain == 1.0:
            return data
        a = a * self.gain
        if self.gain > 1.0:
            a = _soft_knee(a, self.owner.knee)
        return np.clip(a, -32768, 32767).astype(np.int16).tobytes()

    def finish(self) -> float | None:
        """Learn this clip's (pre-gain) level; returns it in dBFS, or None if it had no speech."""
        if not self._parts:
            return None
        level, _ = voiced_rms_dbfs(np.concatenate(self._parts), self.owner.sample_rate)
        if level is not None:
            self.owner._learn(self.key, level)
        return level
