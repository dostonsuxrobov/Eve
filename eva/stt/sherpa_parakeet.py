"""Local speech-to-text with NVIDIA Parakeet TDT 0.6B v2 (int8) via sherpa-onnx.

Implements :class:`eva.interfaces.STT`.  Parakeet is a NeMo TDT transducer;
sherpa-onnx runs the int8 ONNX export on CPU with punctuation and casing
included in the output.  The 0.6B model is considerably more accurate than
whisper base/small and, in int8, decodes a few seconds of speech in well under
a second on a modern laptop CPU.

Model files
-----------
On first :meth:`warmup` the archive
``sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2`` (~470 MB) is downloaded
from the sherpa-onnx GitHub release into ``models/`` and extracted with
``tarfile`` (bz2).  Both steps show progress on stderr and are skipped when the
extracted model directory already contains the encoder / decoder / joiner /
tokens files.  ``model_dir`` can point at an already extracted directory.

Short-input guard
-----------------
Measured on this model: clips shorter than ~0.5 s of silence or room noise
decode to backchannel words ("Mm.", "Yeah.", "Okay.") while 1 s or more of the
same audio decodes to nothing.  Inputs shorter than ``min_audio_s`` (default
1.5 s) are therefore zero-padded at the end before decoding; real speech in a
short clip is unaffected ("Hey" in 300 ms stays "Hey").

All blocking work (download, extraction, recognizer construction, decoding)
runs in a worker thread via ``asyncio.to_thread``.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import MODELS_DIR, USER_AGENT
from ..interfaces import MIC_SAMPLE_RATE, Transcript

log = logging.getLogger(__name__)

PARAKEET_ARCHIVE_NAME = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
PARAKEET_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    f"{PARAKEET_ARCHIVE_NAME}.tar.bz2"
)


def _progress(label: str, done: int, total: int | None, t0: float) -> None:
    """Single-line stderr progress: `label 123.4/470.0 MB 45% 12.3 MB/s`."""
    mb = done / 1e6
    dt = max(time.perf_counter() - t0, 1e-6)
    if total:
        pct = 100.0 * done / total
        msg = f"\r{label} {mb:7.1f}/{total / 1e6:.1f} MB {pct:5.1f}% {mb / dt:5.1f} MB/s"
    else:
        msg = f"\r{label} {mb:7.1f} MB {mb / dt:5.1f} MB/s"
    sys.stderr.write(msg)
    sys.stderr.flush()


def _model_files(model_dir: Path) -> dict[str, Path] | None:
    """Locate encoder/decoder/joiner/tokens inside `model_dir`; None if any is missing.

    Prefers int8 files when several variants exist.
    """
    if not model_dir.is_dir():
        return None

    def pick(stem: str) -> Path | None:
        cands = sorted(model_dir.glob(f"{stem}*.onnx"))
        if not cands:
            return None
        int8 = [c for c in cands if "int8" in c.name]
        return (int8 or cands)[0]

    enc, dec, joi = pick("encoder"), pick("decoder"), pick("joiner")
    tokens = model_dir / "tokens.txt"
    if enc and dec and joi and tokens.exists():
        return {"encoder": enc, "decoder": dec, "joiner": joi, "tokens": tokens}
    return None


def download_file(url: str, dest: Path, label: str = "download") -> None:
    """Stream `url` to `dest` (via a .part file) with a progress line. Blocking."""
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    t0 = time.perf_counter()
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0, read=120.0), headers={"User-Agent": USER_AGENT}) as client:
        with client.stream("GET", url) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length") or 0) or None
            done = 0
            last = 0.0
            with open(part, "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    now = time.perf_counter()
                    if now - last > 0.25:
                        _progress(label, done, total, t0)
                        last = now
            _progress(label, done, total, t0)
            sys.stderr.write("\n")
    if total and done != total:
        part.unlink(missing_ok=True)
        raise IOError(f"short download: {done} of {total} bytes")
    part.replace(dest)


def extract_tar_bz2(archive: Path, into: Path, label: str = "extract") -> None:
    """Extract a .tar.bz2 with a per-member progress line. Blocking."""
    into.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with tarfile.open(archive, "r:bz2") as tf:
        members = tf.getmembers()
        total = sum(m.size for m in members) or None
        done = 0
        for m in members:
            tf.extract(m, path=into, filter="data")
            done += m.size
            _progress(label, done, total, t0)
    sys.stderr.write("\n")


class SherpaParakeetSTT:
    """Parakeet TDT 0.6B v2 int8 through ``sherpa_onnx.OfflineRecognizer``.

    Args:
        model_dir: directory with encoder/decoder/joiner ``.onnx`` + ``tokens.txt``.
            ``None`` -> ``models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8`` (auto-download).
        num_threads: onnxruntime intra-op threads.
        provider: ``"cpu"`` (the venv's onnxruntime is CPU-only).
        decoding_method: ``greedy_search`` (fastest) or ``modified_beam_search``.
        min_audio_s: zero-pad shorter inputs to this length (hallucination guard, see module doc).
    """

    def __init__(
        self,
        model_dir: str | os.PathLike[str] | None = None,
        num_threads: int = 4,
        provider: str = "cpu",
        decoding_method: str = "greedy_search",
        min_audio_s: float = 1.5,
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir else MODELS_DIR / PARAKEET_ARCHIVE_NAME
        self.num_threads = num_threads
        self.provider = provider
        self.decoding_method = decoding_method
        self.min_audio_s = min_audio_s

        self.name = "parakeet-tdt-0.6b-v2-int8"
        self.device = provider
        self.load_time_s: float | None = None
        self.load_log: list[str] = []
        self.files: dict[str, Path] | None = None
        self._rec: Any = None

    # ------------------------------------------------------------------ load
    def _ensure_model(self) -> dict[str, Path]:
        files = _model_files(self.model_dir)
        if files:
            self.load_log.append("model files present, download skipped")
            return files

        # Only auto-download into the canonical models/ location.
        archive = MODELS_DIR / f"{PARAKEET_ARCHIVE_NAME}.tar.bz2"
        if not archive.exists():
            log.info("downloading %s", PARAKEET_URL)
            t0 = time.perf_counter()
            download_file(PARAKEET_URL, archive, label=f"[parakeet] {archive.name}")
            self.load_log.append(f"downloaded {archive.stat().st_size / 1e6:.1f} MB in {time.perf_counter() - t0:.1f}s")
        else:
            self.load_log.append("archive present, download skipped")

        t0 = time.perf_counter()
        extract_tar_bz2(archive, MODELS_DIR, label="[parakeet] extracting")
        self.load_log.append(f"extracted in {time.perf_counter() - t0:.1f}s")

        extracted = MODELS_DIR / PARAKEET_ARCHIVE_NAME
        files = _model_files(extracted)
        if not files:
            listing = sorted(p.name for p in extracted.glob("*")) if extracted.exists() else []
            raise FileNotFoundError(f"Parakeet files not found after extraction in {extracted}; contents: {listing}")
        if extracted != self.model_dir:
            self.model_dir = extracted
        return files

    def _load(self) -> None:
        import sherpa_onnx

        t_start = time.perf_counter()
        self.files = self._ensure_model()
        t0 = time.perf_counter()
        self._rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(self.files["encoder"]),
            decoder=str(self.files["decoder"]),
            joiner=str(self.files["joiner"]),
            tokens=str(self.files["tokens"]),
            num_threads=self.num_threads,
            sample_rate=MIC_SAMPLE_RATE,
            feature_dim=80,
            decoding_method=self.decoding_method,
            model_type="nemo_transducer",
            provider=self.provider,
        )
        self.load_log.append(f"recognizer built in {time.perf_counter() - t0:.2f}s ({self.provider}, {self.num_threads} threads)")
        # First decode is slower (onnxruntime lazy init); do it now, not on the user's first turn.
        t0 = time.perf_counter()
        self._decode(np.zeros(MIC_SAMPLE_RATE // 2, dtype=np.float32))
        self.load_log.append(f"probe decode {time.perf_counter() - t0:.3f}s")
        self.load_time_s = time.perf_counter() - t_start
        log.info("parakeet ready in %.2fs (%s)", self.load_time_s, "; ".join(self.load_log))

    async def warmup(self) -> None:
        """Download/extract the model if needed, build the recognizer, run one probe decode."""
        if self._rec is not None:
            return
        await asyncio.to_thread(self._load)

    # ------------------------------------------------------------ transcribe
    def _decode(self, audio: np.ndarray, sample_rate: int = MIC_SAMPLE_RATE) -> tuple[str, dict[str, Any]]:
        """Blocking decode of float32 audio in [-1, 1]; sherpa resamples if needed."""
        stream = self._rec.create_stream()
        stream.accept_waveform(sample_rate, audio)
        self._rec.decode_stream(stream)
        res = stream.result
        meta: dict[str, Any] = {
            "backend": "sherpa-onnx",
            "model": self.name,
            "device": self.provider,
            "tokens": len(getattr(res, "tokens", []) or []),
        }
        return res.text.strip(), meta

    async def transcribe(self, pcm: np.ndarray, sample_rate: int = MIC_SAMPLE_RATE) -> Transcript:
        """Transcribe one complete utterance (int16 mono). Decoding runs in a thread."""
        if self._rec is None:
            await self.warmup()
        t0 = time.perf_counter()
        a = np.asarray(pcm)
        if a.ndim > 1:
            a = a.mean(axis=1)
        if a.dtype == np.int16:
            audio = a.astype(np.float32) / 32768.0
        elif a.dtype.kind == "f":
            audio = a.astype(np.float32)
        else:
            audio = a.astype(np.float32) / float(np.iinfo(a.dtype).max)
        if audio.size == 0:
            return Transcript(text="", latency_s=0.0, meta={"backend": "sherpa-onnx", "empty_input": True})
        audio_s = audio.size / sample_rate
        min_len = int(self.min_audio_s * sample_rate)
        padded_ms = 0
        if audio.size < min_len:
            padded_ms = round((min_len - audio.size) * 1000 / sample_rate)
            audio = np.concatenate([audio, np.zeros(min_len - audio.size, dtype=np.float32)])
        audio = np.ascontiguousarray(audio)
        text, meta = await asyncio.to_thread(self._decode, audio, sample_rate)
        latency = time.perf_counter() - t0
        meta["audio_s"] = round(audio_s, 3)
        if padded_ms:
            meta["padded_ms"] = padded_ms
        meta["rtf"] = round(latency / audio_s, 3) if audio_s else None
        return Transcript(text=text, latency_s=latency, meta=meta)

    async def close(self) -> None:
        """Release the recognizer."""
        self._rec = None
