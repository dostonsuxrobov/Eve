"""Cloud-to-local failover for the three providers.

Each wrapper holds a *primary* (cloud) and a *backup* (local) provider and looks like
the primary to the pipeline. When the primary cannot answer, the request is served by
the backup instead and the primary is marked *down* for ``cooldown_s``; after that the
next request probes it again. So an internet outage, a 5xx, or a request that hangs
before its first result all degrade to the local stack within one turn, and the cloud
comes back on its own once it answers again.

What counts as "cannot answer" is deliberately narrow: an exception before the first
result, or no first result within ``first_result_timeout_s``. A failure *after* the
first token / byte / transcript is not retried here; half a reply from two different
voices or models would be worse than the pipeline's own handling (it already speaks
what it got and marks the stream as broken).

Backups are warmed in the background after the primary (``warmup()`` returns as soon
as the primary is ready) so startup and the greeting are not delayed by loading
Parakeet, Kokoro or the Ollama model; the first fallback turn pays whatever is left.

Events (``on_event(name, data)``, wired to the console by ``run.py``):
``failover`` when a request is served by the backup, ``recovered`` when the primary
answers again.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Awaitable, Callable

import numpy as np

from .interfaces import MIC_SAMPLE_RATE, LLMDelta, LLMDone, LLMEvent, LLMToolCall, Transcript

log = logging.getLogger("eva.failover")

EventHandler = Callable[[str, dict[str, Any]], None]
_OWN = frozenset({"primary", "backup", "health", "active", "sample_rate", "name"})  # never proxied

DEFAULT_COOLDOWN_S = 20.0  # short: a transient tail must not park a minute of talk on the 4B model
LLM_FIRST_TOKEN_TIMEOUT_S = 5.0  # Cerebras cold-connection tail is 2-3 s; 5 s is "not answering"
STT_TIMEOUT_S = 6.0  # the pipeline's own commit deadline (2.5 s+) runs first; this bounds the batch retry
TTS_FIRST_BYTE_TIMEOUT_S = 4.0  # v3 first audio is 0.5-0.9 s, Flash 0.2 s


class _Health:
    """Down / up bookkeeping shared by the three wrappers."""

    def __init__(self, kind: str, primary_name: str, backup_name: str, cooldown_s: float, on_event: EventHandler | None) -> None:
        self.kind = kind
        self.primary_name = primary_name
        self.backup_name = backup_name
        self.cooldown_s = cooldown_s
        self.on_event = on_event
        self.down_since: float | None = None
        self.failovers = 0

    @property
    def primary_ok(self) -> bool:
        """True if the primary should be tried: healthy, or down long enough to probe again."""
        return self.down_since is None or time.perf_counter() - self.down_since >= self.cooldown_s

    def failed(self, reason: str) -> None:
        first = self.down_since is None
        self.down_since = time.perf_counter()
        self.failovers += 1
        log.warning("%s: %s not answering (%s); using %s", self.kind, self.primary_name, reason, self.backup_name)
        self._emit("failover", {"kind": self.kind, "from": self.primary_name, "to": self.backup_name, "reason": reason, "first": first})

    def recovered(self) -> None:
        if self.down_since is None:
            return
        self.down_since = None
        log.info("%s: %s answering again", self.kind, self.primary_name)
        self._emit("recovered", {"kind": self.kind, "name": self.primary_name})

    def _emit(self, name: str, data: dict[str, Any]) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(name, data)
        except Exception:  # a UI callback must never break a provider
            log.exception("on_event(%s) failed", name)


def _warm_in_background(backup: Any, label: str) -> asyncio.Task[None]:
    async def go() -> None:
        try:
            await backup.warmup()
            log.info("%s backup %s warm", label, getattr(backup, "name", backup))
        except Exception as e:  # a broken backup must not break the session
            log.warning("%s backup %s failed to warm: %s", label, getattr(backup, "name", backup), e)

    return asyncio.create_task(go(), name=f"eva-warm-{label}")


async def _first_then_rest(
    agen: AsyncIterator[Any], timeout_s: float
) -> tuple[Any, AsyncIterator[Any]]:
    """Await the first item of ``agen`` with a timeout; return it and the rest of the stream.

    Raises whatever the stream raises before its first item, or ``asyncio.TimeoutError``.
    """
    it = agen.__aiter__()
    first = await asyncio.wait_for(it.__anext__(), timeout=timeout_s)
    return first, it


async def _aclose(agen: Any) -> None:
    aclose = getattr(agen, "aclose", None)
    if aclose is not None:
        try:
            await aclose()
        except Exception:
            pass


# ------------------------------------------------------------------------ LLM
class FailoverLLM:
    """``eva.interfaces.LLM`` over a primary and a backup brain."""

    def __init__(
        self,
        primary: Any,
        backup: Any,
        *,
        first_token_timeout_s: float = LLM_FIRST_TOKEN_TIMEOUT_S,
        cooldown_s: float = DEFAULT_COOLDOWN_S,
        on_event: EventHandler | None = None,
    ) -> None:
        self.primary = primary
        self.backup = backup
        self.first_token_timeout_s = first_token_timeout_s
        self.health = _Health("llm", primary.name, backup.name, cooldown_s, on_event)
        self._warm_task: asyncio.Task[None] | None = None

    @property
    def name(self) -> str:
        return self.primary.name if self.health.down_since is None else f"{self.backup.name} (fallback for {self.primary.name})"

    @property
    def active(self) -> Any:
        return self.primary if self.health.down_since is None else self.backup

    async def warmup(self) -> None:
        await self.primary.warmup()
        self._warm_task = _warm_in_background(self.backup, "llm")

    async def ping(self) -> float:
        """Keep-alive for the active brain; while down, a successful ping brings the primary back."""
        if self.health.down_since is not None and self.health.primary_ok and callable(getattr(self.primary, "ping", None)):
            try:
                secs = await asyncio.wait_for(self.primary.ping(), timeout=self.first_token_timeout_s)
            except (asyncio.TimeoutError, Exception):
                secs = None
            if secs is None:
                self.health.down_since = time.perf_counter()  # still down: restart the cooldown
            else:
                self.health.recovered()
            return float(secs) if secs is not None else 0.0
        ping = getattr(self.active, "ping", None)
        secs = await ping() if callable(ping) else None
        return float(secs) if isinstance(secs, (int, float)) else 0.0

    async def complete(self, messages: list[dict[str, Any]], **kw: Any) -> str:
        if self.health.primary_ok and callable(getattr(self.primary, "complete", None)):
            try:
                out = await asyncio.wait_for(self.primary.complete(messages, **kw), timeout=self.first_token_timeout_s * 3)
                self.health.recovered()
                return out
            except Exception as e:
                self.health.failed(f"complete: {type(e).__name__}")
        complete = getattr(self.backup, "complete", None)
        if callable(complete):
            return await complete(messages, **kw)
        parts: list[str] = []
        async for ev in self.backup.stream(messages, None):
            if isinstance(ev, LLMDelta):
                parts.append(ev.text)
        return "".join(parts)

    async def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[LLMEvent]:
        if self.health.primary_ok:
            agen = self.primary.stream(messages, tools=tools)
            try:
                first, rest = await _first_then_rest(agen, self.first_token_timeout_s)
            except asyncio.CancelledError:
                await _aclose(agen)
                raise
            except Exception as e:
                await _aclose(agen)
                reason = "no first token in %.0f s" % self.first_token_timeout_s if isinstance(e, asyncio.TimeoutError) else type(e).__name__
                self.health.failed(reason)
            else:
                self.health.recovered()
                yield first
                async for ev in rest:
                    yield ev
                return
        async for ev in self.backup.stream(messages, tools=tools):
            yield ev

    async def close(self) -> None:
        if self._warm_task is not None and not self._warm_task.done():
            self._warm_task.cancel()
        await asyncio.gather(self.primary.close(), self.backup.close(), return_exceptions=True)

    def __getattr__(self, item: str) -> Any:  # anything else (model, extra_body ...) from the active brain
        if item.startswith("_") or item in _OWN:
            raise AttributeError(item)
        return getattr(self.active, item)


# ------------------------------------------------------------------------ STT
class FailoverSTT:
    """``eva.interfaces.StreamingSTT`` over a streaming cloud STT and a batch local STT.

    While the primary is up every call goes to it. ``feed`` / ``commit`` failures make
    the pipeline fall back to a batch transcription of the utterance
    (``transcribe_batch``), which is where the switch to the local model happens, so a
    dead cloud costs at most one late transcript. While the primary is down ``feed``
    raises at once (the pipeline then buffers the utterance for batch) and every batch
    call goes straight to the backup.
    """

    def __init__(
        self,
        primary: Any,
        backup: Any,
        *,
        timeout_s: float = STT_TIMEOUT_S,
        cooldown_s: float = DEFAULT_COOLDOWN_S,
        on_event: EventHandler | None = None,
    ) -> None:
        self.primary = primary
        self.backup = backup
        self.timeout_s = timeout_s
        self.health = _Health("stt", primary.name, backup.name, cooldown_s, on_event)
        self.sample_rate = getattr(primary, "sample_rate", MIC_SAMPLE_RATE)
        self._warm_task: asyncio.Task[None] | None = None

    @property
    def name(self) -> str:
        return self.primary.name if self.health.down_since is None else f"{self.backup.name} (fallback for {self.primary.name})"

    async def warmup(self) -> None:
        await self.primary.warmup()
        self._warm_task = _warm_in_background(self.backup, "stt")

    # streaming path: primary only
    async def feed(self, pcm: np.ndarray) -> None:
        if self.health.down_since is not None:
            raise RuntimeError(f"{self.primary.name} is down; batch")
        await self.primary.feed(pcm)

    async def commit(self) -> Transcript:
        if self.health.down_since is not None:
            raise RuntimeError(f"{self.primary.name} is down; batch")
        try:
            tr = await asyncio.wait_for(self.primary.commit(), timeout=self.timeout_s)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.health.failed(f"commit: {type(e).__name__}")
            raise
        self.health.recovered()
        return tr

    async def discard(self, keep_audio: bool = False) -> None:
        if self.health.down_since is None:
            await self.primary.discard(keep_audio=keep_audio)

    # batch path: primary if it may be up, else (or on failure) the backup
    async def transcribe_batch(self, pcm: np.ndarray, sample_rate: int = MIC_SAMPLE_RATE) -> Transcript:
        if self.health.primary_ok:
            fn = getattr(self.primary, "transcribe_batch", None) or self.primary.transcribe
            try:
                tr = await asyncio.wait_for(fn(pcm, sample_rate), timeout=self.timeout_s)
                self.health.recovered()
                return tr
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.health.failed(f"batch: {type(e).__name__}")
        tr = await self.backup.transcribe(pcm, sample_rate)
        tr.meta["fallback_provider"] = self.backup.name
        return tr

    async def transcribe(self, pcm: np.ndarray, sample_rate: int = MIC_SAMPLE_RATE) -> Transcript:
        return await self.transcribe_batch(pcm, sample_rate)

    async def close(self) -> None:
        if self._warm_task is not None and not self._warm_task.done():
            self._warm_task.cancel()
        await asyncio.gather(self.primary.close(), self.backup.close(), return_exceptions=True)

    def __getattr__(self, item: str) -> Any:  # last_batch_s and friends come from the primary
        if item.startswith("_") or item in _OWN:
            raise AttributeError(item)
        return getattr(self.primary, item)


# ------------------------------------------------------------------------ TTS
class FailoverTTS:
    """``eva.interfaces.TTS`` over a cloud voice and a local one (same sample rate)."""

    def __init__(
        self,
        primary: Any,
        backup: Any,
        *,
        first_byte_timeout_s: float = TTS_FIRST_BYTE_TIMEOUT_S,
        cooldown_s: float = DEFAULT_COOLDOWN_S,
        on_event: EventHandler | None = None,
    ) -> None:
        if primary.sample_rate != backup.sample_rate:
            raise ValueError(f"TTS fallback sample rate {backup.sample_rate} != primary {primary.sample_rate}")
        self.primary = primary
        self.backup = backup
        self.sample_rate = primary.sample_rate
        self.first_byte_timeout_s = first_byte_timeout_s
        self.health = _Health("tts", primary.name, backup.name, cooldown_s, on_event)
        self._warm_task: asyncio.Task[None] | None = None

    @property
    def active(self) -> Any:
        return self.primary if self.health.down_since is None else self.backup

    @property
    def name(self) -> str:
        return self.primary.name if self.health.down_since is None else f"{self.backup.name} (fallback for {self.primary.name})"

    @property
    def supports_audio_tags(self) -> bool:
        return bool(getattr(self.active, "supports_audio_tags", False))

    @property
    def supports_cues(self) -> bool:
        return bool(getattr(self.active, "supports_cues", False))

    async def warmup(self) -> None:
        await self.primary.warmup()
        self._warm_task = _warm_in_background(self.backup, "tts")

    def begin_turn(self) -> None:
        for p in (self.primary, self.backup):
            begin = getattr(p, "begin_turn", None)
            if callable(begin):
                begin()

    def synthesize(self, text: str, *, cue: str | None = None) -> AsyncIterator[bytes]:
        return self._synthesize(text, cue=cue)

    async def _synthesize(self, text: str, *, cue: str | None = None) -> AsyncIterator[bytes]:
        if self.health.primary_ok:
            agen = self._call(self.primary, text, cue)
            try:
                first, rest = await _first_then_rest(agen, self.first_byte_timeout_s)
            except asyncio.CancelledError:
                await _aclose(agen)
                raise
            except StopAsyncIteration:
                self.health.recovered()
                return  # nothing to say (empty text)
            except Exception as e:
                await _aclose(agen)
                reason = "no audio in %.0f s" % self.first_byte_timeout_s if isinstance(e, asyncio.TimeoutError) else type(e).__name__
                self.health.failed(reason)
            else:
                self.health.recovered()
                yield first
                async for b in rest:
                    yield b
                return
        async for b in self._call(self.backup, text, cue):
            yield b

    @staticmethod
    def _call(provider: Any, text: str, cue: str | None) -> AsyncIterator[bytes]:
        if cue and getattr(provider, "supports_cues", False):
            return provider.synthesize(text, cue=cue)
        return provider.synthesize(text)

    async def synthesize_to_bytes(self, text: str) -> bytes:
        return b"".join([b async for b in self._synthesize(text)])

    async def close(self) -> None:
        if self._warm_task is not None and not self._warm_task.done():
            self._warm_task.cancel()
        await asyncio.gather(self.primary.close(), self.backup.close(), return_exceptions=True)

    def __getattr__(self, item: str) -> Any:
        if item.startswith("_") or item in _OWN:
            raise AttributeError(item)
        return getattr(self.active, item)


__all__ = ["FailoverLLM", "FailoverSTT", "FailoverTTS", "DEFAULT_COOLDOWN_S"]
