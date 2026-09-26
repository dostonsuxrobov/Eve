"""ElevenLabs Scribe batch speech-to-text (``POST /v1/speech-to-text``).

One complete utterance (int16 mono PCM) is wrapped into an in-memory WAV and
posted as multipart form data.  A single keep-alive ``httpx.AsyncClient`` is
reused for every request so that only the very first call pays for DNS + TLS;
``warmup()`` performs that first call with 0.4 s of silence.

Model ids this key accepts (from the API's own error message, Sept 2026):
``scribe_v1``, ``scribe_v1_experimental``, ``scribe_v2``, ``scribe_v2_medical``.
``scribe_v2_realtime`` is websocket-only (see ``elevenlabs_realtime.py``).

Silence / near-silence returns HTTP 200 with ``text == ""`` (no exception).
Hard failures (auth, quota, malformed request, exhausted retries) raise
:class:`ScribeError`; the pipeline decides whether to swallow them.
"""
from __future__ import annotations

import asyncio
import io
import logging
import ssl
import time
from typing import Any

import httpx
import numpy as np
import soundfile as sf

from ..config import USER_AGENT
from ..interfaces import MIC_SAMPLE_RATE, Transcript

log = logging.getLogger(__name__)

STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
_RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
_SSL_CONTEXT: ssl.SSLContext | None = None


def get_ssl_context() -> ssl.SSLContext:
    """One process-wide TLS context (creating one costs ~150 ms of blocking CPU, measured).

    Shared by the batch httpx client and every realtime websocket so that reconnects
    and socket rotations never stall the event loop.  Prefer :func:`ensure_ssl_context`
    from async code.
    """
    global _SSL_CONTEXT
    if _SSL_CONTEXT is None:
        _SSL_CONTEXT = httpx.create_ssl_context()
    return _SSL_CONTEXT


async def ensure_ssl_context() -> ssl.SSLContext:
    """Build the shared TLS context in a worker thread (no-op once cached)."""
    if _SSL_CONTEXT is None:
        await asyncio.to_thread(get_ssl_context)
    return get_ssl_context()


class ScribeError(RuntimeError):
    """Raised when the Scribe API returns a non-retryable error or retries are exhausted."""

    def __init__(self, message: str, status: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status = status
        self.body = body


def pcm_to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Encode int16 mono PCM as a 16-bit WAV file in memory (no ffmpeg needed)."""
    buf = io.BytesIO()
    sf.write(buf, pcm, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def is_voiced(pcm: np.ndarray, sample_rate: int = MIC_SAMPLE_RATE, *, rms_threshold: float = 200.0, min_s: float = 0.3) -> bool:
    """Cheap energy gate: True if the int16 audio is at least ``min_s`` long and its RMS exceeds the threshold."""
    if len(pcm) < int(min_s * sample_rate):
        return False
    rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)))
    return rms > rms_threshold


def as_int16_mono(pcm: np.ndarray) -> np.ndarray:
    """Coerce any numpy audio array to a contiguous 1-D int16 array."""
    arr = np.asarray(pcm)
    if arr.ndim == 2:  # (frames, channels) -> mono
        arr = arr.mean(axis=1)
    if arr.dtype != np.int16:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, -1.0, 1.0) * 32767.0
        arr = arr.astype(np.int16)
    return np.ascontiguousarray(arr)


class ElevenLabsScribeSTT:
    """Batch Scribe STT conforming to :class:`eva.interfaces.STT`.

    Args:
        api_key: ElevenLabs API key (``xi-api-key``).
        model_id: ``scribe_v1`` (default) or ``scribe_v2``; both measured to work.
        language: optional ISO-639-1/3 code passed as ``language_code``.  Leave
            ``None`` to let Scribe auto-detect (measured cost: none).
        timeout_s: read timeout for one request.
        retries: extra attempts on transport errors / 5xx / 429.
        min_audio_ms: utterances shorter than this are not sent at all and
            return an empty transcript (the API requires >= 100 ms).
        retry_empty: re-send once when the API returns ``""`` for audio that is
            clearly voiced (measured: 1 in ~40 calls did that during a server hiccup).
    """

    def __init__(
        self,
        api_key: str,
        model_id: str = "scribe_v1",
        language: str | None = None,
        *,
        timeout_s: float = 15.0,
        retries: int = 1,
        min_audio_ms: int = 100,
        retry_empty: bool = True,
    ) -> None:
        if not api_key:
            raise ValueError("ElevenLabs api_key is required")
        self.api_key = api_key
        self.model_id = model_id
        self.language = language
        self.timeout_s = timeout_s
        self.retries = max(0, retries)
        self.min_audio_ms = min_audio_ms
        self.retry_empty = retry_empty
        self.name = f"elevenlabs/{model_id}"
        self._client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()  # one in-flight request per instance keeps the pool tiny

    # ------------------------------------------------------------------ client
    def _client_or_create(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                verify=get_ssl_context(),
                headers={"xi-api-key": self.api_key, "User-Agent": USER_AGENT},
                timeout=httpx.Timeout(connect=5.0, read=self.timeout_s, write=self.timeout_s, pool=5.0),
                # httpx's default keepalive_expiry is 5 s, which would drop the warm TLS
                # connection between conversational turns. Keep it for minutes; httpcore
                # detects a server-side close at checkout and reconnects transparently.
                limits=httpx.Limits(max_connections=2, max_keepalive_connections=2, keepalive_expiry=300.0),
            )
        return self._client

    # --------------------------------------------------------------- protocol
    async def warmup(self) -> None:
        """Open the keep-alive TLS connection by transcribing 0.4 s of silence."""
        t0 = time.perf_counter()
        silence = np.zeros(int(0.4 * MIC_SAMPLE_RATE), dtype=np.int16)
        try:
            await ensure_ssl_context()
            tr = await self.transcribe(silence)
            log.info("%s warmup: %.3f s (text=%r)", self.name, time.perf_counter() - t0, tr.text)
        except Exception as exc:  # warmup must never take the pipeline down
            log.warning("%s warmup failed after %.3f s: %s", self.name, time.perf_counter() - t0, exc)

    async def transcribe(self, pcm: np.ndarray, sample_rate: int = MIC_SAMPLE_RATE) -> Transcript:
        """Transcribe one complete utterance. Returns stripped text plus timing/meta."""
        t0 = time.perf_counter()
        pcm = as_int16_mono(pcm)
        audio_s = len(pcm) / sample_rate
        meta: dict[str, Any] = {"model_id": self.model_id, "audio_s": round(audio_s, 3)}

        if audio_s * 1000 < self.min_audio_ms:
            meta["skipped"] = "too_short"
            return Transcript(text="", latency_s=time.perf_counter() - t0, meta=meta)

        wav = pcm_to_wav_bytes(pcm, sample_rate)
        data: dict[str, str] = {
            "model_id": self.model_id,
            "tag_audio_events": "false",
            "diarize": "false",
        }
        if self.language:
            data["language_code"] = self.language
        files = {"file": ("utterance.wav", wav, "audio/wav")}

        async with self._lock:
            payload = await self._post_with_retry(data, files, meta)
            if self.retry_empty and not (payload.get("text") or "").strip() and is_voiced(pcm, sample_rate):
                log.warning("%s returned empty text for voiced %.1f s audio; retrying once", self.name, audio_s)
                meta["retried_empty"] = True
                payload = await self._post_with_retry(data, files, meta)

        words = payload.get("words") or []
        meta.update(
            {
                "language": payload.get("language_code"),
                "language_probability": payload.get("language_probability"),
                "words": words,
                "transcription_id": payload.get("transcription_id"),
            }
        )
        text = (payload.get("text") or "").strip()
        return Transcript(text=text, latency_s=time.perf_counter() - t0, meta=meta)

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    # ---------------------------------------------------------------- helpers
    async def _post_with_retry(
        self, data: dict[str, str], files: dict[str, Any], meta: dict[str, Any]
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            client = self._client_or_create()
            t_http = time.perf_counter()
            try:
                resp = await client.post(STT_URL, data=data, files=files)
            except httpx.TransportError as exc:
                # Typically a keep-alive socket the server already closed; retry on a fresh one.
                last_error = exc
                log.warning("%s transport error (attempt %d): %s", self.name, attempt + 1, exc)
                await self.close()
                continue
            meta["http_s"] = round(time.perf_counter() - t_http, 4)
            meta["trace_id"] = resp.headers.get("x-trace-id")
            meta["region"] = resp.headers.get("x-region")
            meta["character_cost"] = resp.headers.get("character-cost")
            meta["fiat_cost_usd"] = resp.headers.get("fiat-cost-before-overages")
            meta["attempts"] = attempt + 1
            if resp.status_code == 200:
                return resp.json()
            body = resp.text[:500]
            if resp.status_code in _RETRY_STATUSES and attempt < self.retries:
                last_error = ScribeError(f"HTTP {resp.status_code}", resp.status_code, body)
                log.warning("%s HTTP %d (attempt %d): %s", self.name, resp.status_code, attempt + 1, body)
                await asyncio.sleep(0.2 * (attempt + 1))
                continue
            raise ScribeError(f"Scribe HTTP {resp.status_code}: {body}", resp.status_code, body)
        raise ScribeError(f"Scribe request failed after {self.retries + 1} attempts: {last_error}") from last_error
