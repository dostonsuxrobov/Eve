"""Local speech-to-text with faster-whisper (CTranslate2).

Implements :class:`eva.interfaces.STT`.  The model is loaded in :meth:`warmup`
and every blocking call (model load, inference) runs in a worker thread via
``asyncio.to_thread`` so the audio loop never stalls.

Device handling
---------------
``device="auto"`` tries ``cuda`` first and falls back to ``cpu``.  CTranslate2
happily *constructs* a CUDA model on a machine without the cuBLAS / cuDNN DLLs
and only raises on the first forward pass, so the probe here runs a tiny real
transcription before declaring a device usable.  The outcome is cached at module
level so a second instance does not pay for the failed CUDA attempt again.
The resolved device is exposed as :attr:`FasterWhisperSTT.device`.

Hallucination / latency guards
------------------------------
Whisper is known to invent text ("Thank you.", "you", ...) on silence or pure
tones.  faster-whisper drops a segment when ``no_speech_prob > no_speech_threshold``
AND ``avg_logprob < log_prob_threshold``; both knobs are exposed here and the
per-segment scores are put in ``Transcript.meta["segment_scores"]`` for diagnosis.
Temperature fallback is disabled (``temperature=0.0``) so a hard utterance never
triggers up to six extra decoding passes; latency stays predictable.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import MODELS_DIR
from ..interfaces import MIC_SAMPLE_RATE, Transcript

# huggingface_hub prints a long symlink warning on every download on Windows.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

log = logging.getLogger(__name__)

# Module-level memo: None = unknown, True = CUDA works, False = CUDA is broken.
_CUDA_USABLE: bool | None = None
_CUDA_LOCK = threading.Lock()


def _to_float32(pcm: np.ndarray, sample_rate: int) -> np.ndarray:
    """int16 (or float) mono -> float32 in [-1, 1] at 16 kHz."""
    a = np.asarray(pcm)
    if a.ndim > 1:
        a = a.mean(axis=1)
    if a.dtype == np.int16:
        f = a.astype(np.float32) / 32768.0
    elif a.dtype.kind == "f":
        f = a.astype(np.float32)
    else:  # int32 / other ints
        f = a.astype(np.float32) / float(np.iinfo(a.dtype).max)
    if sample_rate != MIC_SAMPLE_RATE and f.size:
        n_out = int(round(f.size * MIC_SAMPLE_RATE / sample_rate))
        x_old = np.linspace(0.0, 1.0, num=f.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        f = np.interp(x_new, x_old, f).astype(np.float32)
    return np.ascontiguousarray(f)


class FasterWhisperSTT:
    """faster-whisper backend.

    Args:
        model: Whisper size or CTranslate2 model path (``base.en``, ``small.en`` ...).
        device: ``"auto"`` (cuda then cpu), ``"cuda"`` or ``"cpu"``.
        compute_type: CTranslate2 compute type; ``int8`` is fastest on CPU.
        cpu_threads: 0 lets CTranslate2 pick (all physical cores).
        download_root: where model weights are stored; defaults to ``models/faster-whisper``.
        language: forced language passed to ``transcribe``.
        no_speech_threshold / logprob_threshold: hallucination guard (see module doc).
        temperature: single value = no fallback passes; a list enables whisper's fallback.
    """

    def __init__(
        self,
        model: str = "base.en",
        device: str = "auto",
        compute_type: str = "int8",
        cpu_threads: int = 0,
        download_root: str | os.PathLike[str] | None = None,
        language: str = "en",
        no_speech_threshold: float = 0.6,
        logprob_threshold: float = -1.0,
        temperature: float | list[float] = 0.0,
    ) -> None:
        self.model_name = model
        self.requested_device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.download_root = Path(download_root) if download_root else MODELS_DIR / "faster-whisper"
        self.language = language
        self.no_speech_threshold = no_speech_threshold
        self.logprob_threshold = logprob_threshold
        self.temperature = temperature

        self.name = f"faster-whisper/{model}"
        self.device: str | None = None  # resolved in warmup()
        self.load_time_s: float | None = None
        self.load_log: list[str] = []  # human-readable trace of what happened during warmup
        self._model: Any = None

    # ------------------------------------------------------------------ load
    def _try_device(self, device: str) -> Any:
        """Construct the model on `device` and run a real 0.5 s probe. Raises on failure."""
        from faster_whisper import WhisperModel

        t0 = time.perf_counter()
        m = WhisperModel(
            self.model_name,
            device=device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
            download_root=str(self.download_root),
        )
        t_load = time.perf_counter() - t0
        # A tiny probe: CTranslate2 reports missing CUDA DLLs only on the first forward pass.
        probe = np.zeros(MIC_SAMPLE_RATE // 2, dtype=np.float32)
        t0 = time.perf_counter()
        segs, _ = m.transcribe(probe, beam_size=1, language=self.language, without_timestamps=True)
        for _ in segs:  # generator -> force decode
            pass
        t_probe = time.perf_counter() - t0
        self.load_log.append(f"{device}: load {t_load:.2f}s, probe {t_probe:.3f}s -> OK")
        return m

    def _load(self) -> None:
        global _CUDA_USABLE
        self.download_root.mkdir(parents=True, exist_ok=True)
        t_start = time.perf_counter()
        if self.requested_device == "auto":
            candidates = ["cuda", "cpu"]
        else:
            candidates = [self.requested_device]

        last_err: Exception | None = None
        for dev in candidates:
            if dev == "cuda" and self.requested_device == "auto":
                with _CUDA_LOCK:
                    if _CUDA_USABLE is False:
                        self.load_log.append("cuda: skipped (known unusable in this process)")
                        continue
            t_dev = time.perf_counter()
            try:
                self._model = self._try_device(dev)
                self.device = dev
                if dev == "cuda":
                    with _CUDA_LOCK:
                        _CUDA_USABLE = True
                break
            except Exception as e:  # noqa: BLE001 - we want to fall through to cpu
                last_err = e
                msg = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
                self.load_log.append(f"{dev}: FAILED after {time.perf_counter() - t_dev:.2f}s ({type(e).__name__}: {msg})")
                log.warning("faster-whisper %s on %s failed: %s", self.model_name, dev, msg)
                self._model = None
                if dev == "cuda":
                    with _CUDA_LOCK:
                        _CUDA_USABLE = False
        if self._model is None:
            raise RuntimeError(
                f"faster-whisper could not load {self.model_name!r} on any of {candidates}: {last_err}"
            ) from last_err
        self.load_time_s = time.perf_counter() - t_start
        log.info("faster-whisper %s ready on %s in %.2fs", self.model_name, self.device, self.load_time_s)

    async def warmup(self) -> None:
        """Load the model (downloading weights on first use) and resolve the device."""
        if self._model is not None:
            return
        await asyncio.to_thread(self._load)

    # ------------------------------------------------------------ transcribe
    def _transcribe_sync(self, audio: np.ndarray) -> tuple[str, dict[str, Any]]:
        segments, info = self._model.transcribe(
            audio,
            beam_size=1,
            language=self.language,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
            no_speech_threshold=self.no_speech_threshold,
            log_prob_threshold=self.logprob_threshold,
            temperature=self.temperature,
        )
        texts: list[str] = []
        scores: list[dict[str, float]] = []
        for seg in segments:  # generator: decoding happens here
            text = seg.text.strip()
            scores.append(
                {"no_speech_prob": round(float(seg.no_speech_prob), 3), "avg_logprob": round(float(seg.avg_logprob), 3)}
            )
            if text:
                texts.append(text)
        meta: dict[str, Any] = {
            "backend": "faster-whisper",
            "model": self.model_name,
            "device": self.device,
            "compute_type": self.compute_type,
            "segments": len(scores),
            "segment_scores": scores,
            "language_probability": round(float(info.language_probability), 3),
        }
        return " ".join(texts).strip(), meta

    async def transcribe(self, pcm: np.ndarray, sample_rate: int = MIC_SAMPLE_RATE) -> Transcript:
        """Transcribe one complete utterance (int16 mono). Inference runs in a thread."""
        if self._model is None:
            await self.warmup()
        t0 = time.perf_counter()
        audio = _to_float32(pcm, sample_rate)
        if audio.size == 0:
            return Transcript(text="", latency_s=0.0, meta={"backend": "faster-whisper", "empty_input": True})
        text, meta = await asyncio.to_thread(self._transcribe_sync, audio)
        latency = time.perf_counter() - t0
        audio_s = audio.size / MIC_SAMPLE_RATE
        meta["audio_s"] = round(audio_s, 3)
        meta["rtf"] = round(latency / audio_s, 3) if audio_s else None
        return Transcript(text=text, latency_s=latency, meta=meta)

    async def close(self) -> None:
        """Release the model."""
        self._model = None
