"""ElevenLabs text-to-speech provider: websocket ``stream-input`` and HTTP ``stream`` modes.

Implements :class:`eva.interfaces.TTS`. Output is raw little-endian int16 mono PCM at
24 000 Hz (``output_format=pcm_24000``), so no ffmpeg / mp3 decoding is required.

Two transport modes:

``ws``
    ``wss://api.elevenlabs.io/v1/text-to-speech/{voice}/stream-input`` with
    ``auto_mode=true`` (the low-latency setting: no chunk schedule, audio is generated as
    soon as a sentence / flush arrives). One websocket is used per :meth:`synthesize`
    call. To hide the ~200-300 ms handshake, a *one-slot connection pool* keeps the NEXT
    websocket pre-opened (and already initialised with the API key and voice settings)
    in the background: :meth:`warmup` opens the first one and every :meth:`synthesize`
    call immediately starts opening its successor. While a pooled socket sits idle a
    keep-alive task sends a single space every few seconds so ElevenLabs' inactivity
    timeout never fires.

``http``
    ``POST /v1/text-to-speech/{voice}/stream`` on a keep-alive ``httpx`` client. Bytes
    are yielded as they arrive; a carry byte guarantees every yielded chunk has an even
    length (whole int16 samples).

If the websocket endpoint rejects the model (``eleven_v3`` is only served by the
text-to-dialogue websocket, so ``stream-input`` answers HTTP 400 ``unsupported_model``)
the provider falls back to ``http`` automatically and records why in
:attr:`ElevenLabsTTS.fallback_reason`.

Voice settings: for flash / turbo a warm conversational default is
``stability=0.45, similarity_boost=0.75, style=0.0``. ``eleven_v3`` only accepts the
stability presets ``0.0`` (creative), ``0.5`` (natural) and ``1.0`` (robust); any other
value is snapped to the nearest preset.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator
from urllib.parse import urlencode

import httpx
import websockets
from websockets.asyncio.client import ClientConnection, connect as ws_connect
from websockets.protocol import State

from ..config import USER_AGENT
from ..delivery import cue_settings, detect_lang, strip_tags

API_HOST = "api.elevenlabs.io"
HTTP_BASE = f"https://{API_HOST}"
WS_BASE = f"wss://{API_HOST}"
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


class ElevenLabsError(RuntimeError):
    """A failure reported by the ElevenLabs API (HTTP 400/401/403/422/429 or a ws error frame).

    ``status`` is the HTTP status; websocket error frames (close code 1008 with an
    ``error`` slug such as ``invalid_api_key``) are mapped to the closest HTTP status so
    callers can treat both transports alike. ``code`` keeps the raw API error slug.
    """

    def __init__(
        self, status: int, message: str, *, mode: str, model_id: str, code: str | None = None
    ) -> None:
        super().__init__(f"ElevenLabs {mode} {model_id}: HTTP {status} {code or ''}: {message}")
        self.status = status
        self.message = message
        self.mode = mode
        self.model_id = model_id
        self.code = code


def _ws_error_status(slug: str) -> int:
    """Map a stream-input error slug to an HTTP-like status."""
    slug = slug.lower()
    if "api_key" in slug or "unauthor" in slug or "permission" in slug or "auth" in slug:
        return 401
    if "quota" in slug or "limit" in slug or "too_many" in slug:
        return 429
    if "model" in slug or "voice" in slug or "not_found" in slug:
        return 422
    return 400


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
class _PooledWS:
    """A pre-opened, pre-initialised websocket waiting in the one-slot pool."""

    ws: ClientConnection
    opened_at: float
    setup_s: float  # handshake + init message send time
    keepalive: asyncio.Task[None] | None = None

    @property
    def is_open(self) -> bool:
        return self.ws.state is State.OPEN


@dataclass
class SynthStats:
    """Timing of the most recent :meth:`ElevenLabsTTS.synthesize` call."""

    mode: str = ""
    ttfa_s: float | None = None  # call -> first PCM chunk yielded
    total_s: float | None = None  # call -> last chunk
    audio_s: float = 0.0  # seconds of PCM produced
    bytes: int = 0
    chunks: int = 0
    setup_hidden: bool | None = None  # ws only: was the socket pre-opened?
    ws_setup_s: float | None = None  # ws only: handshake cost of the socket that was used
    extra: dict[str, Any] = field(default_factory=dict)


class ElevenLabsTTS:
    """ElevenLabs TTS conforming to :class:`eva.interfaces.TTS` (24 kHz int16 PCM)."""

    sample_rate: int = SAMPLE_RATE

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        model_id: str = "eleven_flash_v2_5",
        mode: str = "ws",
        stability: float | None = None,
        similarity_boost: float | None = None,
        style: float | None = None,
        speed: float | None = None,
        *,
        use_speaker_boost: bool | None = None,
        inactivity_timeout: int = 180,
        keepalive_s: float = 12.0,
        timeout_s: float = 30.0,
        optimize_streaming_latency: int | None = 3,
        voices_by_lang: dict[str, str] | None = None,
        first_chunk_model: str | None = None,
        continuity: bool = True,
    ) -> None:
        """
        voices_by_lang: optional ``{"ru": voice_id, ...}``; each synthesize() call picks the
            voice by the script of its text (``eva.delivery.detect_lang``), falling back
            to ``voice_id``. Only honoured in http mode (the pooled websocket is bound to
            one voice).
        first_chunk_model: optional model used for the FIRST chunk after ``begin_turn()``
            (e.g. ``eleven_flash_v2_5`` under ``eleven_v3``): fast onset, expressive rest.
        continuity: pass ``previous_text`` / ``previous_request_ids`` within a turn so
            consecutive sentences keep one prosodic line instead of restarting each time.
        """
        if mode not in ("ws", "http"):
            raise ValueError(f"mode must be 'ws' or 'http', got {mode!r}")
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.voices_by_lang = {k.lower(): v for k, v in (voices_by_lang or {}).items()}
        self.first_chunk_model = first_chunk_model
        self.continuity = continuity
        self.supports_cues: bool = True  # synthesize(text, cue=...) maps cues to settings
        # Per-turn continuity state (reset by begin_turn()).
        self._turn_chunk = 0
        self._prev_text: str | None = None
        self._prev_ids: list[tuple[str, str]] = []  # (model_id, request_id) of recent chunks
        self.requested_mode = mode
        self.mode = mode  # effective mode; may flip to "http" on fallback
        self.supports_audio_tags: bool = model_id.startswith("eleven_v3")
        self.name = f"elevenlabs/{model_id}/{voice_id}/{mode}"
        self.fallback_reason: str | None = None
        self.last = SynthStats()
        self.calls = 0

        self._stability = stability
        self._similarity = similarity_boost
        self._style = style
        self._speed = speed
        self._speaker_boost = use_speaker_boost
        self._inactivity_timeout = max(1, min(int(inactivity_timeout), 180))
        self._keepalive_s = keepalive_s
        self._timeout_s = timeout_s
        self._osl = optimize_streaming_latency

        self._http: httpx.AsyncClient | None = None
        self._next_ws: asyncio.Future[_PooledWS] | None = None
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
        """Open the first connection so the first real turn pays no setup cost.

        ``ws``: pre-open (and initialise) the first pooled websocket; if the endpoint
        rejects the model, switch to ``http`` and warm that instead.
        ``http``: create the keep-alive client and perform one cheap request so the
        TLS connection is established and cached in the pool.
        """
        if self.mode == "ws":
            try:
                pooled = await self._open_pooled()
            except _WSRejected as exc:
                self._fallback(f"warmup: {exc}")
            else:
                self._stash_next(pooled)
                return
        await self._warm_http()

    def synthesize(self, text: str, *, cue: str | None = None) -> AsyncIterator[bytes]:
        """Stream int16 PCM chunks for ``text`` (see module docstring for modes).

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
        """Cancel the pooled websocket and close the HTTP client."""
        self._closed = True
        task, self._next_ws = self._next_ws, None
        if task is not None:
            if task.done() and not task.cancelled() and task.exception() is None:
                await self._discard(task.result())
            else:
                task.cancel()
                with contextlib.suppress(BaseException):
                    await task
        if self._http is not None:
            client, self._http = self._http, None
            await client.aclose()

    # ------------------------------------------------------------------ helpers
    def _fallback(self, reason: str) -> None:
        self.mode = "http"
        self.fallback_reason = reason
        self.name = f"elevenlabs/{self.model_id}/{self.voice_id}/http(fallback)"

    async def _synthesize(self, text: str, *, cue: str | None = None) -> AsyncIterator[bytes]:
        if self._closed:
            raise RuntimeError("ElevenLabsTTS is closed")
        self.calls += 1
        if not text.strip():
            self.last = SynthStats(mode=self.mode, ttfa_s=None, total_s=0.0)
            return
        chunk_no = self._turn_chunk
        self._turn_chunk += 1
        model_id = self.model_id
        if chunk_no == 0 and self.first_chunk_model:
            model_id = self.first_chunk_model
        voice_id = self.voice_for(text) if self.mode == "http" else self.voice_id
        if model_id.startswith("eleven_v3"):
            if cue and not text.lstrip().startswith("["):
                text = f"[{cue}] {text}"
        else:
            text = strip_tags(text) or text  # a tag-less model would read "[warm]" aloud
        if not text.strip():
            return
        if self.mode == "ws" and model_id == self.model_id and voice_id == self.voice_id and cue is None:
            try:
                async with contextlib.aclosing(self._synthesize_ws(text)) as gen:
                    async for chunk in gen:
                        yield chunk
                return
            except _WSRejected as exc:
                self._fallback(str(exc))
        async for chunk in self._synthesize_http(text, model_id=model_id, voice_id=voice_id, cue=cue):
            yield chunk

    # ---------------------------------------------------------------------- HTTP
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

    async def _warm_http(self) -> None:
        """Establish and cache the TLS connection with a free ``OPTIONS`` request.

        Measured here: an OPTIONS on the stream URL answers 200 in ~0.45 s and the next
        synthesis starts streaming in ~0.17 s instead of ~0.8-3 s on a cold client.
        (A 401 from ``/v1/models`` also warms the socket but far less reliably.)
        """
        client = self._http_client()
        with contextlib.suppress(httpx.HTTPError):
            await client.options(self._http_path())

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
        stats = SynthStats(mode="http", extra={"model_id": model_id, "voice_id": voice_id, "cue": cue})
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
                raise ElevenLabsError(
                    resp.status_code,
                    msg,
                    mode="http",
                    model_id=model_id,
                    code=_err_code(raw),
                )
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

    # ------------------------------------------------------------------------ WS
    def _ws_url(self) -> str:
        q = {
            "model_id": self.model_id,
            "output_format": OUTPUT_FORMAT,
            "auto_mode": "true",
            "inactivity_timeout": self._inactivity_timeout,
        }
        return f"{WS_BASE}/v1/text-to-speech/{self.voice_id}/stream-input?{urlencode(q)}"

    async def _open_pooled(self) -> _PooledWS:
        """Connect, send the init message (key + voice settings) and start keep-alive."""
        t0 = time.perf_counter()
        try:
            ws = await ws_connect(
                self._ws_url(),
                additional_headers={"xi-api-key": self.api_key, "User-Agent": USER_AGENT},
                open_timeout=15.0,
                max_size=None,
            )
        except websockets.InvalidStatus as exc:
            status = exc.response.status_code
            msg = _err_message(bytes(exc.response.body or b""))
            if status in (401, 403):
                raise ElevenLabsError(status, msg, mode="ws", model_id=self.model_id) from exc
            raise _WSRejected(f"ws handshake HTTP {status}: {msg}") from exc
        except websockets.InvalidHandshake as exc:
            raise _WSRejected(f"ws handshake failed: {exc}") from exc
        init = {
            "text": " ",
            "voice_settings": self.voice_settings(),
            "xi_api_key": self.api_key,
        }
        await ws.send(json.dumps(init))
        pooled = _PooledWS(ws=ws, opened_at=t0, setup_s=time.perf_counter() - t0)
        pooled.keepalive = asyncio.create_task(self._keepalive(pooled))
        return pooled

    async def _keepalive(self, pooled: _PooledWS) -> None:
        """Send a lone space periodically so the idle socket is not closed for inactivity."""
        try:
            while pooled.is_open:
                await asyncio.sleep(self._keepalive_s)
                await pooled.ws.send(json.dumps({"text": " "}))
        except (websockets.ConnectionClosed, asyncio.CancelledError):
            pass

    def _stash_next(self, pooled: _PooledWS) -> None:
        fut: asyncio.Future[_PooledWS] = asyncio.get_running_loop().create_future()
        fut.set_result(pooled)
        self._next_ws = fut

    def _spawn_next(self) -> None:
        if self._closed or self.mode != "ws":
            return
        task = asyncio.create_task(self._open_pooled())
        task.add_done_callback(_retrieve_exception)
        self._next_ws = task

    async def _discard(self, pooled: _PooledWS) -> None:
        if pooled.keepalive is not None:
            pooled.keepalive.cancel()
            with contextlib.suppress(BaseException):
                await pooled.keepalive
        with contextlib.suppress(Exception):
            await pooled.ws.close()

    async def _take_ws(self) -> tuple[_PooledWS, bool]:
        """Take the pooled socket (or open one inline) and start pre-opening the next."""
        task, self._next_ws = self._next_ws, None
        self._spawn_next()
        hidden = task is not None
        pooled: _PooledWS | None = None
        if task is not None:
            try:
                pooled = await task
            except _WSRejected:
                raise
            except ElevenLabsError:
                raise
            except Exception:
                pooled = None
            if pooled is not None and not pooled.is_open:
                await self._discard(pooled)
                pooled = None
        if pooled is None:
            hidden = False
            pooled = await self._open_pooled()
        if pooled.keepalive is not None:
            pooled.keepalive.cancel()
            with contextlib.suppress(BaseException):
                await pooled.keepalive
            pooled.keepalive = None
        return pooled, hidden

    async def _synthesize_ws(self, text: str) -> AsyncIterator[bytes]:
        t0 = time.perf_counter()
        stats = SynthStats(mode="ws")
        if not text.endswith(" "):
            text += " "  # ElevenLabs asks for a trailing space on streamed text
        pooled, hidden = await self._take_ws()
        stats.setup_hidden = hidden
        stats.ws_setup_s = pooled.setup_s
        ws = pooled.ws
        carry = bytearray()
        try:
            await ws.send(json.dumps({"text": text, "flush": True}))
            await ws.send(json.dumps({"text": ""}))
            async for raw in ws:
                frame = json.loads(raw)
                if frame.get("error") or frame.get("code"):
                    slug = str(frame.get("error") or frame.get("code"))
                    raise ElevenLabsError(
                        _ws_error_status(slug),
                        str(frame.get("message") or slug),
                        mode="ws",
                        model_id=self.model_id,
                        code=slug,
                    )
                audio = frame.get("audio")
                if audio:
                    out = _even(base64.b64decode(audio), carry)
                    if out:
                        if stats.ttfa_s is None:
                            stats.ttfa_s = time.perf_counter() - t0
                        stats.chunks += 1
                        stats.bytes += len(out)
                        yield out
                if frame.get("isFinal"):
                    break
        except websockets.ConnectionClosedOK:
            pass  # server closed after the final frame
        except websockets.ConnectionClosedError as exc:
            if stats.bytes == 0:
                raise _WSRejected(f"ws closed before audio: {exc}") from exc
            raise
        finally:
            with contextlib.suppress(Exception):
                await ws.close()
        stats.total_s = time.perf_counter() - t0
        stats.audio_s = stats.bytes / BYTES_PER_SECOND
        self.last = stats


class _WSRejected(RuntimeError):
    """The stream-input endpoint refused this model / request; fall back to HTTP."""


def _retrieve_exception(task: asyncio.Future[Any]) -> None:
    """Swallow background-open failures so asyncio does not log 'never retrieved'."""
    if not task.cancelled():
        task.exception()


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
