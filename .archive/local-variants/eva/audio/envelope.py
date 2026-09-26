"""Amplitude shaping for spoken turns: lead-in, fade-in, fade-out, tail.

TTS services trim silence at both ends of every clip, so a reply otherwise starts at
full amplitude the instant the endpoint fires and stops dead on the last sample. A few
tens of milliseconds of raised-cosine fade at each end plus a short breath of silence
before and after make the same audio feel like a person starting and finishing a
thought instead of a sound file being played.

All functions take and return raw little-endian int16 mono PCM bytes.
"""
from __future__ import annotations

import numpy as np


def silence(ms: float, sample_rate: int) -> bytes:
    n = max(0, int(round(sample_rate * ms / 1000.0)))
    return bytes(n * 2)


_rng = np.random.default_rng(7)


def room_tone(ms: float, sample_rate: int, dbfs: float | None) -> bytes:
    """``ms`` of faint, soft (low-passed) noise at ``dbfs``; digital silence when ``dbfs`` is None.

    TTS clips carry their own floor; gaps of digital zero between them make the background
    switch on and off with every reply. A constant bed at the clips' own floor hides that.
    """
    n = max(0, int(round(sample_rate * ms / 1000.0)))
    if n == 0 or dbfs is None:
        return bytes(n * 2)
    x = _rng.standard_normal(n + 8).astype(np.float32)
    x = np.convolve(x, np.ones(8, dtype=np.float32) / 8.0, mode="valid")[:n]  # gentle low-pass
    rms = float(np.sqrt((x * x).mean())) or 1.0
    x *= (32767.0 * 10 ** (dbfs / 20.0)) / rms
    return x.astype(np.int16).tobytes()


def _ramp(n: int) -> np.ndarray:
    """Raised-cosine ramp 0 -> 1 over ``n`` samples (smoother than linear at the ends)."""
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    x = np.linspace(0.0, np.pi, n, dtype=np.float32)
    return (0.5 - 0.5 * np.cos(x)).astype(np.float32)


def fade_in(pcm: bytes, ms: float, sample_rate: int) -> bytes:
    """Apply a fade-in over the first ``ms`` of ``pcm`` (shorter if the clip is shorter)."""
    n = min(len(pcm) // 2, int(round(sample_rate * ms / 1000.0)))
    if n <= 0:
        return pcm
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32).copy()
    a[:n] *= _ramp(n)
    return a.astype(np.int16).tobytes()


def fade_out(pcm: bytes, ms: float, sample_rate: int) -> bytes:
    """Apply a fade-out over the last ``ms`` of ``pcm``."""
    n = min(len(pcm) // 2, int(round(sample_rate * ms / 1000.0)))
    if n <= 0:
        return pcm
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32).copy()
    a[-n:] *= _ramp(n)[::-1]
    return a.astype(np.int16).tobytes()


def split_tail(pcm: bytes, ms: float, sample_rate: int) -> tuple[bytes, bytes]:
    """Split ``pcm`` into (head, tail) where tail is the last ``ms`` (whole samples)."""
    n = min(len(pcm) // 2, int(round(sample_rate * ms / 1000.0)))
    if n <= 0:
        return pcm, b""
    cut = len(pcm) - n * 2
    return pcm[:cut], pcm[cut:]


__all__ = ["silence", "room_tone", "fade_in", "fade_out", "split_tail"]
