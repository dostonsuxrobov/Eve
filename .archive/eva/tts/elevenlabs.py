"""ElevenLabs text-to-speech provider (HTTP ``stream`` endpoint).

Implements :class:`eva.interfaces.TTS`. Output is raw little-endian int16 mono PCM at
24 000 Hz (``output_format=pcm_24000``), so no ffmpeg / mp3 decoding is required.

``POST /v1/text-to-speech/{voice}/stream`` on a keep-alive ``httpx`` client. Bytes are
yielded as they arrive; a carry byte guarantees every yielded chunk has an even length
(whole int16 samples). The ``stream-input`` websocket transport that used to live here
was measured slower than HTTP (it is reopened per reply) and cannot serve ``eleven_v3``;
it was removed on 2026-09-19 and is in git history if a persistent socket is ever tried.

Per reply (``begin_turn()``):
* the FIRST chunk may go to ``first_chunk_model`` (Flash under v3) for a fast onset;
* the voice is picked per chunk by script (``voices_by_lang``);
* consecutive chunks carry ``previous_text`` / ``previous_request_ids`` so a reply keeps
  one prosodic line (Flash / Turbo only: v3 rejects both fields);
* every chunk is loudness-leveled per model x voice (:mod:`eva.audio.leveler`).

Voice settings: for flash / turbo a warm conversational default is
``stability=0.45, similarity_boost=0.75, style=0.0``. ``eleven_v3`` (and
``eleven_v3_conversational``, which shares its limits: no ``optimize_streaming_latency``,
no ``previous_text``, verified 2026-09-23) only accepts the stability presets ``0.0`` (creative), ``0.5`` (natural) and ``1.0`` (robust); any other
value is snapped to the nearest preset.
"""
from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

from ..audio.leveler import Leveler
from ..config import USER_AGENT
from ..delivery import cue_settings, detect_lang, strip_tags

API_HOST = "api.elevenlabs.io"
HTTP_BASE = f"https://{API_HOST}"
SAMPLE_RATE = 24_000
OUTPUT_FORMAT = "pcm_24000"
BYTES_PER_SECOND = SAMPLE_RATE * 2  # int16 mono

# ElevenLabs v3 accepts only these stability presets (creative / natural / robust).
V3_STABILITY_PRESETS = (0.0, 0.5, 1.0)

# Warm, conversational defaults for flash / turbo (per DESIGN.md persona notes).
DEFAULT_STABILITY = 0.45
DEFAULT_SIMILARITY = 0.75
DEFAULT_STYLE = 0.0
DEFAULT_V3_STABILITY = 0.5  # "natural"

# Loudness leveling (eva.audio.leveler): every clip is scaled towards LEVEL_TARGET_DBFS
# by the learned level of its source ("model/voice"). Seeds are voiced-RMS levels
# measured on 2026-09-19 so the very first sentence of a session is already close.
LEVEL_TARGET_DBFS = -19.0
LEVEL_SEEDS_DBFS: dict[str, float] = {
    "eleven_flash_v2_5": -21.0,
    "eleven_v3": -16.6,
    "eleven_v3/yMBZR4SLoc24wOJLWAB2": -25.4,  # eva_ru
    # 2026-09-23, two cued lines per voice (raw): -20.3 / -16.4 and -20.7 / -20.3
    "eleven_v3_conversational": -18.4,
    "eleven_v3_conversational/yMBZR4SLoc24wOJLWAB2": -20.5,  # eva_ru
}


class ElevenLabsError(RuntimeError):
    """Non-200 answer from the API. ``code`` keeps the raw API error slug."""

    def __init__(self, status: int, message: str, *, model_id: str, code: str | None = None) -> None:
        super().__init__(f"ElevenLabs http {model_id}: HTTP {status} {code or ''}: {message}")
        self.status = status
        self.message = message
        self.model_id = model_id
        self.code = code


def _even(data: bytes, carry: bytearray) -> bytes:
    """Return ``carry + data`` trimmed to an even length; the odd byte stays in ``carry``."""
    if carry:
        data = bytes(carry) + data
        carry.clear()
    if len(data) & 1:
        carry.append(data[-1])
        data = data[:-1]
    return data


@dataclass
class SynthStats:
    """Timing of the most recent :meth:`ElevenLabsTTS.synthesize` call."""

    ttfa_s: float | None = None  # call -> first PCM chunk yielded
    total_s: float | None = None  # call -> last chunk
    audio_s: float = 0.0  # seconds of PCM produced
    bytes: int = 0
    chunks: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class ElevenLabsTTS:
    """ElevenLabs TTS conforming to :class:`eva.interfaces.TTS` (24 kHz int16 PCM)."""

    sample_rate: int = SAMPLE_RATE

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        model_id: str = "eleven_flash_v2_5",
        stability: float | None = None,
        similarity_boost: float | None = None,
        style: float | None = None,
        speed: float | None = None,
        *,
        use_speaker_boost: bool | None = None,
        timeout_s: float = 30.0,
        optimize_streaming_latency: int | None = 3,
        voices_by_lang: dict[str, str] | None = None,
        first_chunk_model: str | None = None,
        continuity: bool = True,
        level_dbfs: float | None = LEVEL_TARGET_DBFS,
    ) -> None:
        """
        voices_by_lang: optional ``{"ru": voice_id, ...}``; each synthesize() call picks the
            voice by the script of its text (``eva.delivery.detect_lang``), falling back
            to ``voice_id``.
        first_chunk_model: optional model used for the FIRST chunk after ``begin_turn()``
            (e.g. ``eleven_flash_v2_5`` under ``eleven_v3``): fast onset, expressive rest.
        continuity: pass ``previous_text`` / ``previous_request_ids`` within a turn so
            consecutive sentences keep one prosodic line instead of restarting each time.
        level_dbfs: equalise loudness across models and voices towards this voiced level
            (``None`` = off). Flash, v3 and the per-language voices differ by up to 9 dB
            otherwise, which is heard as the volume climbing inside every reply.
        """
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.voices_by_lang = {k.lower(): v for k, v in (voices_by_lang or {}).items()}
        self.first_chunk_model = first_chunk_model
        self.continuity = continuity
        self.leveler: Leveler | None = (
            Leveler(SAMPLE_RATE, level_dbfs, LEVEL_SEEDS_DBFS) if level_dbfs is not None else None
        )
        self.supports_cues: bool = True  # synthesize(text, cue=...) maps cues to settings
        # Per-turn continuity state (reset by begin_turn()).
        self._turn_chunk = 0
        self._prev_text: str | None = None
        self._prev_ids: list[tuple[str, str]] = []  # (model_id, request_id) of recent chunks
        self.supports_audio_tags: bool = model_id.startswith("eleven_v3")
        self.name = f"elevenlabs/{model_id}/{voice_id}"
        self.last = SynthStats()
        self.calls = 0

        self._stability = stability
        self._similarity = similarity_boost
        self._style = style
        self._speed = speed
        self._speaker_boost = use_speaker_boost
        self._timeout_s = timeout_s
        self._osl = optimize_streaming_latency

        self._http: httpx.AsyncClient | None = None
        self._closed = False

    # ------------------------------------------------------------------ settings
    def voice_settings(self, model_id: str | None = None, cue: str | None = None) -> dict[str, Any]:
        """The ``voice_settings`` object sent to the API (model-aware, cue-aware).

        A delivery ``cue`` only changes settings on tag-less models; v3 receives the cue
        inline as a tag instead.
        """
        model_id = model_id or self.model_id
        if model_id.startswith("eleven_v3"):  # eleven_v3: stability presets only
            stab = DEFAULT_V3_STABILITY if self._stability is None else self._stability
            stab = min(V3_STABILITY_PRESETS, key=lambda p: abs(p - stab))
            vs: dict[str, Any] = {
                "stability": stab,
                "similarity_boost": DEFAULT_SIMILARITY if self._similarity is None else self._similarity,
            }
            if self._style is not None:
                vs["style"] = self._style
        else:
            vs = {
                "stability": DEFAULT_STABILITY if self._stability is None else self._stability,
                "similarity_boost": DEFAULT_SIMILARITY if self._similarity is None else self._similarity,
                "style": DEFAULT_STYLE if self._style is None else self._style,
            }
        if self._speed is not None:
            vs["speed"] = self._speed
        if self._speaker_boost is not None:
            vs["use_speaker_boost"] = self._speaker_boost
        if cue and not model_id.startswith("eleven_v3"):
            vs = cue_settings(cue, vs)
        return vs

    # ---------------------------------------------------------------- per turn
    def begin_turn(self) -> None:
        """Start a new reply: the next chunk is the turn's first (hybrid model, no continuity)."""
        self._turn_chunk = 0
        self._prev_text = None
        self._prev_ids = []

    def voice_for(self, text: str) -> str:
        """Voice id for ``text`` by script (``voices_by_lang``), default ``voice_id``."""
        if not self.voices_by_lang:
            return self.voice_id
        lang = detect_lang(text, default="")
        return self.voices_by_lang.get(lang, self.voice_id)

    # ------------------------------------------------------------------- public
    async def warmup(self) -> None:
        """Establish and cache the TLS connection with a free ``OPTIONS`` request.

        Measured here: an OPTIONS on the stream URL answers 200 in ~0.45 s and the next
        synthesis starts streaming in ~0.17 s instead of ~0.8-3 s on a cold client.
        """
        client = self._http_client()
        with contextlib.suppress(httpx.HTTPError):
            await client.options(self._http_path())

    def synthesize(self, text: str, *, cue: str | None = None) -> AsyncIterator[bytes]:
        """Stream int16 PCM chunks for ``text``.

        ``cue`` is a delivery cue from ``eva.delivery`` (``"warm"``, ``"teasing"`` ...):
        on Flash/Turbo it becomes voice settings for this chunk; on v3 it is prepended
        as an inline tag if the text does not already start with one.
        """
        return self._synthesize(text, cue=cue)

    async def synthesize_to_bytes(self, text: str) -> bytes:
        """Render ``text`` completely (for pre-rendering fillers)."""
        parts: list[bytes] = []
        async for chunk in self._synthesize(text):
            parts.append(chunk)
        return b"".join(parts)

    async def close(self) -> None:
        self._closed = True
        if self._http is not None:
            client, self._http = self._http, None
            await client.aclose()

    # ------------------------------------------------------------------ helpers
    async def _synthesize(self, text: str, *, cue: str | None = None) -> AsyncIterator[bytes]:
        if self._closed:
            raise RuntimeError("ElevenLabsTTS is closed")
        self.calls += 1
        if not text.strip():
            self.last = SynthStats(ttfa_s=None, total_s=0.0)
            return
        chunk_no = self._turn_chunk
        self._turn_chunk += 1
        model_id = self.model_id
        if chunk_no == 0 and self.first_chunk_model:
            model_id = self.first_chunk_model
        voice_id = self.voice_for(text)
        if model_id.startswith("eleven_v3"):
            if cue and not text.lstrip().startswith("["):
                text = f"[{cue}] {text}"
        else:
            text = strip_tags(text) or text  # a tag-less model would read "[warm]" aloud
        if not text.strip():
            return
        clip = self.leveler.begin(f"{model_id}/{voice_id}") if self.leveler is not None else None
        async for chunk in self._synthesize_http(text, model_id=model_id, voice_id=voice_id, cue=cue):
            yield clip.process(chunk) if clip else chunk
        if clip:
            clip.finish()

    def _http_client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=HTTP_BASE,
                headers={
                    "xi-api-key": self.api_key,
                    "User-Agent": USER_AGENT,
                    "Accept": "audio/*, application/json",
                },
                timeout=httpx.Timeout(self._timeout_s, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=4, max_connections=8, keepalive_expiry=300.0),
            )
        return self._http

    def _http_path(self, voice_id: str | None = None) -> str:
        return f"/v1/text-to-speech/{voice_id or self.voice_id}/stream"

    def _http_params(self, model_id: str | None = None) -> dict[str, Any]:
        model_id = model_id or self.model_id
        params: dict[str, Any] = {"output_format": OUTPUT_FORMAT}
        # eleven_v3 answers HTTP 400 if optimize_streaming_latency is present.
        if self._osl is not None and not model_id.startswith("eleven_v3"):
            params["optimize_streaming_latency"] = self._osl
        return params

    async def _synthesize_http(
        self,
        text: str,
        *,
        model_id: str | None = None,
        voice_id: str | None = None,
        cue: str | None = None,
    ) -> AsyncIterator[bytes]:
        client = self._http_client()
        model_id = model_id or self.model_id
        voice_id = voice_id or self.voice_id
        body: dict[str, Any] = {
            "text": text,
            "model_id": model_id,
            "voice_settings": self.voice_settings(model_id, cue),
        }
        if self.continuity and not model_id.startswith("eleven_v3"):
            # Prosodic continuity across the sentences of one reply. Request ids are only
            # reused for the same model, and eleven_v3 accepts neither previous_text nor
            # previous_request_ids yet (HTTP 400 unsupported_model, verified 2026-09-18).
            if self._prev_text:
                body["previous_text"] = self._prev_text[-300:]
            ids = [rid for m, rid in self._prev_ids if m == model_id][-3:]
            if ids:
                body["previous_request_ids"] = ids
        t0 = time.perf_counter()
        stats = SynthStats(extra={"model_id": model_id, "voice_id": voice_id, "cue": cue})
        carry = bytearray()
        async with client.stream(
            "POST", self._http_path(voice_id), params=self._http_params(model_id), json=body
        ) as resp:
            if resp.status_code != 200:
                raw = await resp.aread()
                msg = _err_message(raw)
                if resp.status_code == 400 and "previous_" in msg and (
                    "previous_text" in body or "previous_request_ids" in body
                ):
                    # This model does not take continuity fields: retry once without them
                    # and stop sending them for the rest of the session.
                    self.continuity = False
                    body.pop("previous_text", None)
                    body.pop("previous_request_ids", None)
                    async for out in self._synthesize_http(text, model_id=model_id, voice_id=voice_id, cue=cue):
                        yield out
                    return
                raise ElevenLabsError(resp.status_code, msg, model_id=model_id, code=_err_code(raw))
            rid = resp.headers.get("request-id")
            if rid:
                self._prev_ids = (self._prev_ids + [(model_id, rid)])[-6:]
            self._prev_text = text
            async for data in resp.aiter_bytes():
                out = _even(data, carry)
                if not out:
                    continue
                if stats.ttfa_s is None:
                    stats.ttfa_s = time.perf_counter() - t0
                stats.chunks += 1
                stats.bytes += len(out)
                yield out
        stats.total_s = time.perf_counter() - t0
        stats.audio_s = stats.bytes / BYTES_PER_SECOND
        self.last = stats


def _err_code(raw: bytes) -> str | None:
    """The API error slug (``detail.status`` / ``detail.code``) from an error body, if any."""
    try:
        detail = json.loads(raw.decode("utf-8", "replace")).get("detail")
    except (ValueError, AttributeError):
        return None
    if isinstance(detail, dict):
        v = detail.get("status") or detail.get("code")
        return str(v) if v else None
    return None


def _err_message(raw: bytes) -> str:
    """Pull the human message out of an ElevenLabs error body."""
    try:
        detail = json.loads(raw.decode("utf-8", "replace")).get("detail")
    except (ValueError, AttributeError):
        return raw.decode("utf-8", "replace")[:300]
    if isinstance(detail, dict):
        return str(detail.get("message") or detail)
    if isinstance(detail, list):
        return "; ".join(str(d.get("msg", d)) if isinstance(d, dict) else str(d) for d in detail)
    return str(detail)


__all__ = ["ElevenLabsTTS", "ElevenLabsError", "SynthStats", "SAMPLE_RATE"]
