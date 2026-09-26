"""Local text-to-speech with Kokoro-82M through kokoro-onnx (CPU onnxruntime).

Implements :class:`eva.interfaces.TTS`. Audio is raw little-endian int16 mono PCM
at 24 000 Hz, yielded one sentence at a time so the first chunk is ready well
before the whole utterance has been rendered.

Model choice (measured on this laptop, Ryzen 7 7840HS, onnxruntime 1.30 CPU):

* ``kokoro-v1.0.onnx`` (fp32, 325 MB): ~0.5 s for 3.4 s of speech, RTF ~0.15.
* ``kokoro-v1.0.fp16.onnx``: no faster than fp32 on CPU and spams ORT
  "can't constant fold Reciprocal" warnings. Not used by default.
* ``kokoro-v1.0.int8.onnx`` (both the v1.0 and v1.1 release builds): 5-15x
  SLOWER than fp32 here (RTF 0.85-2.1) because the dynamic-quantised MatMuls
  fall back to slow kernels. Not usable in real time on this CPU.

So the default is fp32. Files are downloaded on first ``warmup()`` from the
kokoro-onnx GitHub releases into ``models/``.

Streaming strategy: kokoro-onnx 0.6 has ``create_stream`` but it only splits
at the 510-phoneme model limit, so for a normal sentence it yields exactly one
chunk after the *whole* text is rendered. This module instead splits the text
into sentences itself and renders each with ``Kokoro.create`` in a worker
thread, one sentence ahead of the consumer. Every sentence comes back with the
model's own ~0.2 s trailing pause, so chunks concatenate naturally.

Phonemisation uses the espeak-ng DLL bundled by ``espeakng-loader``; no system
espeak-ng install is needed on Windows (verified).
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import threading
import time
import unicodedata
from pathlib import Path
from typing import AsyncIterator, Final

import numpy as np

from ..config import MODELS_DIR, USER_AGENT

log = logging.getLogger("eva.tts.kokoro")

SAMPLE_RATE: Final[int] = 24_000

RELEASE_BASE: Final[str] = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/"
)
MODEL_FILES: Final[dict[str, str]] = {
    "fp32": "kokoro-v1.0.onnx",
    "fp16": "kokoro-v1.0.fp16.onnx",
    "int8": "kokoro-v1.0.int8.onnx",
}
VOICES_FILE: Final[str] = "voices-v1.0.bin"
DEFAULT_MODEL: Final[str] = "fp32"

# Sentence boundary: terminal punctuation (optionally followed by a closing
# quote/bracket) then whitespace. Decimal points and ellipses inside a run are
# not followed by whitespace so they survive.
_SENTENCE_RE = re.compile(r"(?<=[.!?…])[\"'”’)\]]*\s+")
# Common abbreviations whose trailing period must not end a sentence.
_ABBREV_RE = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|St|Mt|vs|etc|e\.g|i\.e|Jr|Sr|Prof)\.$", re.IGNORECASE)
_VOICE_PART_RE = re.compile(r"^\s*([A-Za-z]{2}_[A-Za-z]+)\s*(?:\*\s*([0-9]*\.?[0-9]+))?\s*$")


def strip_unspeakable(text: str) -> str:
    """Drop emoji / pictographs / private-use characters.

    espeak-ng reads emoji out loud by name ("slightly smiling face"), which is
    never wanted from a voice agent, so anything in the Unicode "other symbol"
    or surrogate/private/unassigned categories is removed. Letters, digits,
    punctuation, currency and math symbols are kept.
    """
    return "".join(
        ch for ch in text if unicodedata.category(ch) not in ("So", "Cs", "Co", "Cn")
    )


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentence-sized chunks for incremental synthesis.

    Emoji are stripped, whitespace is normalised, empty pieces dropped, and a
    piece ending in a known abbreviation ("Dr.") is glued to the next one. Text
    without terminal punctuation comes back as a single chunk.
    """
    text = " ".join(strip_unspeakable(text).split())
    if not text:
        return []
    pieces = [p.strip() for p in _SENTENCE_RE.split(text) if p.strip()]
    merged: list[str] = []
    for piece in pieces:
        if merged and _ABBREV_RE.search(merged[-1]):
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return merged


def float_to_int16_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 [-1, 1] samples to little-endian int16 PCM bytes."""
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def _download(url: str, dest: Path) -> None:
    """Stream ``url`` into ``dest`` (via a .part file so a failed run leaves nothing)."""
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    log.info("downloading %s -> %s", url, dest)
    t0 = time.perf_counter()
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0, read=120.0),
                      headers={"User-Agent": USER_AGENT}) as client:
        with client.stream("GET", url) as r:
            r.raise_for_status()
            expected = int(r.headers.get("content-length", 0) or 0)
            n = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
                    n += len(chunk)
    if expected and n != expected:
        tmp.unlink(missing_ok=True)
        raise IOError(f"short download for {dest.name}: {n} of {expected} bytes")
    tmp.replace(dest)
    log.info("downloaded %s (%.1f MB) in %.1fs", dest.name, n / 1e6, time.perf_counter() - t0)


class KokoroTTS:
    """Kokoro-82M local TTS conforming to :class:`eva.interfaces.TTS`.

    Args:
        voice: a Kokoro voice name (``af_heart``, ``af_bella``, ``bf_emma``,
            ``am_michael``...) or a blend such as ``"af_heart*0.7+af_bella*0.3"``.
        speed: 0.5-2.0 speaking-rate multiplier.
        model: ``"fp32"`` (default, fastest here), ``"fp16"``, ``"int8"`` or an
            explicit path to a Kokoro ONNX file.
        intra_threads: onnxruntime ``intra_op_num_threads``; ``None`` lets ORT
            pick (physical cores, which measured best). 4-8 is the useful range.
        lang: ``"en-us"`` or ``"en-gb"`` (phonemiser accent; British voices
            pair naturally with ``en-gb``).
        models_dir: where model files live / are downloaded to.
        lookahead: sentences rendered ahead of the consumer inside one
            ``synthesize`` call.
    """

    name: str = "kokoro"
    sample_rate: int = SAMPLE_RATE
    supports_audio_tags: bool = False

    def __init__(
        self,
        voice: str = "af_heart",
        speed: float = 1.0,
        *,
        model: str = DEFAULT_MODEL,
        intra_threads: int | None = None,
        lang: str = "en-us",
        models_dir: Path | str | None = None,
        lookahead: int = 1,
    ) -> None:
        self.voice = voice
        self.speed = float(speed)
        self.model = model
        self.intra_threads = intra_threads
        self.lang = lang
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.lookahead = max(1, int(lookahead))
        self.name = f"kokoro/{voice}"

        self._kokoro = None  # kokoro_onnx.Kokoro after warmup
        self._style: np.ndarray | None = None  # resolved (possibly blended) voice vector
        self._infer_lock = threading.Lock()  # one ORT run at a time: CPU has no spare cores
        self.load_time_s: float | None = None  # session build only
        self.warmup_time_s: float | None = None  # whole warmup() incl. download + first synth

    # ------------------------------------------------------------------ paths
    @property
    def model_path(self) -> Path:
        if self.model in MODEL_FILES:
            return self.models_dir / MODEL_FILES[self.model]
        return Path(self.model)

    @property
    def voices_path(self) -> Path:
        return self.models_dir / VOICES_FILE

    def _ensure_files(self) -> None:
        """Download the ONNX model and voices pack if they are missing."""
        if self.model in MODEL_FILES and not self.model_path.exists():
            _download(RELEASE_BASE + MODEL_FILES[self.model], self.model_path)
        if not self.voices_path.exists():
            _download(RELEASE_BASE + VOICES_FILE, self.voices_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Kokoro model not found: {self.model_path}")

    # --------------------------------------------------------------- lifecycle
    def _build(self) -> None:
        """Blocking: create the ORT session, load voices, resolve the style vector."""
        import onnxruntime as rt
        from kokoro_onnx import Kokoro

        so = rt.SessionOptions()
        so.log_severity_level = 3  # silence per-node constant-folding warnings
        if self.intra_threads:
            so.intra_op_num_threads = int(self.intra_threads)
        provider = os.environ.get("ONNX_PROVIDER") or "CPUExecutionProvider"
        t0 = time.perf_counter()
        sess = rt.InferenceSession(str(self.model_path), so, providers=[provider])
        self._kokoro = Kokoro.from_session(sess, str(self.voices_path))
        self.load_time_s = time.perf_counter() - t0
        self._style = self._resolve_voice(self.voice)

    def _resolve_voice(self, spec: str) -> np.ndarray:
        """Turn ``"af_heart"`` or ``"af_heart*0.7+af_bella*0.3"`` into a style array."""
        assert self._kokoro is not None
        available = self._kokoro.get_voices()
        parts: list[tuple[str, float]] = []
        for raw in spec.split("+"):
            m = _VOICE_PART_RE.match(raw)
            if not m:
                raise ValueError(f"bad Kokoro voice spec {raw!r}")
            name, weight = m.group(1), float(m.group(2) or 1.0)
            if name not in available:
                english = [v for v in available if v[:3] in ("af_", "am_", "bf_", "bm_")]
                raise ValueError(f"unknown Kokoro voice {name!r}; English voices: {english}")
            parts.append((name, weight))
        total = sum(w for _, w in parts)
        style = sum(
            self._kokoro.get_voice_style(n).astype(np.float32) * (w / total) for n, w in parts
        )
        return np.asarray(style, dtype=np.float32)

    async def warmup(self) -> None:
        """Download files if needed, build the session and run one tiny synthesis.

        The throwaway synthesis pays the one-time espeak-ng backend init
        (~175 ms measured) and ORT arena allocation so the first real turn is
        not slower than the rest.
        """
        t0 = time.perf_counter()
        await asyncio.to_thread(self._ensure_files)
        await asyncio.to_thread(self._build)
        await asyncio.to_thread(self._synth_chunk, "Hi.")
        self.warmup_time_s = time.perf_counter() - t0
        log.info(
            "kokoro ready: %s voice=%s load=%.2fs warmup=%.2fs threads=%s",
            self.model_path.name, self.voice, self.load_time_s or 0.0,
            self.warmup_time_s, self.intra_threads or "auto",
        )

    async def close(self) -> None:
        self._kokoro = None
        self._style = None

    def available_voices(self) -> list[str]:
        """Voice names in the loaded voices pack (empty before warmup)."""
        return [] if self._kokoro is None else list(self._kokoro.get_voices())

    def set_voice(self, voice: str) -> None:
        """Switch voice (or blend) without rebuilding the session."""
        self.voice = voice
        self.name = f"kokoro/{voice}"
        if self._kokoro is not None:
            self._style = self._resolve_voice(voice)

    # --------------------------------------------------------------- synthesis
    def _synth_chunk(self, text: str) -> bytes:
        """Blocking: render one chunk to int16 PCM bytes (empty if unspeakable)."""
        if self._kokoro is None or self._style is None:
            raise RuntimeError("KokoroTTS.warmup() must be awaited before synthesize()")
        with self._infer_lock:
            try:
                audio, _sr = self._kokoro.create(
                    text, voice=self._style, speed=self.speed, lang=self.lang
                )
            except ValueError as e:  # no phonemes in vocabulary, e.g. only emoji
                log.warning("kokoro skipped %r: %s", text, e)
                return b""
        return float_to_int16_bytes(audio)

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Yield int16 PCM chunks, one per sentence, as soon as each is rendered.

        Inference runs in worker threads via ``asyncio.to_thread``; a producer
        task keeps ``lookahead`` sentences in flight so the loop and playback
        never wait on the CPU. Abandoning the iterator cancels the producer; the
        chunk already inside ORT finishes, nothing further starts.
        """
        chunks = split_sentences(text)
        if not chunks:
            return

        queue: asyncio.Queue[bytes | BaseException | None] = asyncio.Queue(maxsize=self.lookahead)

        async def produce() -> None:
            try:
                for chunk in chunks:
                    pcm = await asyncio.to_thread(self._synth_chunk, chunk)
                    await queue.put(pcm)
            except asyncio.CancelledError:
                raise
            except BaseException as e:  # hand the failure to the consumer
                await queue.put(e)
            else:
                await queue.put(None)

        task = asyncio.create_task(produce(), name="kokoro-produce")
        try:
            while True:
                item = await queue.get()
                if item is None:
                    return
                if isinstance(item, BaseException):
                    raise item
                if item:
                    yield item
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def synthesize_to_bytes(self, text: str) -> bytes:
        """Render the whole ``text`` and return it as one int16 PCM byte string."""
        parts: list[bytes] = []
        async for pcm in self.synthesize(text):
            parts.append(pcm)
        return b"".join(parts)

    async def synthesize_one_shot(self, text: str) -> bytes:
        """Render ``text`` with a single ``Kokoro.create`` call (no sentence split).

        Lower total CPU time than :meth:`synthesize` (one fixed per-call cost
        instead of one per sentence) but first audio only arrives at the end.
        Used by the benchmark to quantify the trade-off.
        """
        return await asyncio.to_thread(self._synth_chunk, " ".join(strip_unspeakable(text).split()))


__all__ = ["KokoroTTS", "split_sentences", "strip_unspeakable", "float_to_int16_bytes", "SAMPLE_RATE"]
