"""Streaming chat-completions client for any OpenAI-compatible backend.

Used for Cerebras (``https://api.cerebras.ai/v1``), Ollama (``http://127.0.0.1:11434/v1``)
and anything else that speaks the ``/chat/completions`` SSE protocol.

Design notes
------------
* One ``httpx.AsyncClient`` per instance with HTTP keep-alive so the TLS handshake is
  paid once at :meth:`OpenAICompatLLM.warmup` and not on the first real turn.
* :meth:`OpenAICompatLLM.stream` yields :class:`~eva.interfaces.LLMDelta` for content,
  one :class:`~eva.interfaces.LLMToolCall` per *completed* tool call (arguments fully
  accumulated and JSON-parsed) and exactly one :class:`~eva.interfaces.LLMDone`.
* Reasoning deltas (``delta.reasoning`` / ``delta.reasoning_content`` / ``delta.thinking``)
  are never surfaced as text but are counted into ``LLMDone.usage``.
* A leaked ``<think>...</think>`` block at the start of the content stream is filtered
  out on the fly (some local models emit it even when told not to).
* ``stream_options: {include_usage: true}`` is requested; if a backend rejects it with
  HTTP 400 the request is retried without it and the backend is remembered as not
  supporting it.
* One retry with a short backoff on 429 / 5xx / connection-level errors, but only if
  nothing has been yielded yet (never duplicates text the caller already consumed).
  If the failure happens mid-stream the generator ends with
  ``LLMDone(finish_reason="error")`` and the error text in ``usage["error"]``.

Error contract
--------------
If the request cannot be started at all (retries exhausted) :meth:`stream` raises the
underlying ``httpx`` exception or :class:`LLMHTTPError`; the caller decides what to say.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, AsyncIterator

import httpx

from ..config import USER_AGENT, Keys
from ..interfaces import LLMDelta, LLMDone, LLMEvent, LLMToolCall

log = logging.getLogger("eva.llm")

_RETRYABLE_EXC = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.PoolTimeout,
)

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_THINK_BLOCK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>\s*", re.DOTALL | re.IGNORECASE)
_THINK_UNCLOSED_RE = re.compile(r"<think(?:ing)?>.*\Z", re.DOTALL | re.IGNORECASE)


class LLMHTTPError(RuntimeError):
    """Non-2xx response from the backend (after retry)."""

    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status
        self.body = body


def strip_think_text(text: str) -> str:
    """Remove any ``<think>...</think>`` blocks (closed or unclosed) from a full string."""
    text = _THINK_BLOCK_RE.sub("", text)
    text = _THINK_UNCLOSED_RE.sub("", text)
    return text.lstrip("\n")


class _ThinkFilter:
    """Incremental filter that drops a leading ``<think>...</think>`` block from a stream.

    Deltas may split the tag across boundaries, so text is held back until it is clear
    whether the stream starts with a think block.  Once past the block (or once it is
    clear there is none) the filter is transparent.
    """

    def __init__(self) -> None:
        self._state = "start"  # start | in_think | normal
        self._buf = ""
        self.reasoning_chars = 0

    def feed(self, delta: str) -> str:
        if self._state == "normal":
            return delta
        self._buf += delta
        out = ""
        while True:
            if self._state == "start":
                stripped = self._buf.lstrip()
                if not stripped:
                    return out  # only whitespace so far; keep waiting
                if stripped.lower().startswith(_THINK_OPEN):
                    self._state = "in_think"
                    self._buf = stripped[len(_THINK_OPEN) :]
                    continue
                if _THINK_OPEN.startswith(stripped[: len(_THINK_OPEN)].lower()):
                    return out  # could still become "<think>": hold
                self._state = "normal"
                out += self._buf
                self._buf = ""
                return out
            if self._state == "in_think":
                idx = self._buf.lower().find(_THINK_CLOSE)
                if idx < 0:
                    # Keep only a tail long enough to detect a split closing tag.
                    keep = len(_THINK_CLOSE) - 1
                    self.reasoning_chars += max(0, len(self._buf) - keep)
                    self._buf = self._buf[-keep:] if keep else ""
                    return out
                self.reasoning_chars += idx
                rest = self._buf[idx + len(_THINK_CLOSE) :].lstrip("\n")
                self._state = "normal"
                self._buf = ""
                out += rest
                return out
            return out

    def flush(self) -> str:
        """Return whatever is still held back (called at end of stream)."""
        if self._state == "start":
            rest, self._buf = self._buf, ""
            self._state = "normal"
            return rest
        self._buf = ""
        return ""


def _parse_tool_arguments(raw: str) -> dict[str, Any]:
    """Parse tool-call arguments tolerantly: empty / invalid / non-object -> ``{}``."""
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        val = json.loads(raw)
    except json.JSONDecodeError:
        # Some models emit a trailing garbage char or two concatenated objects; try the
        # first balanced object.
        depth = 0
        for i, ch in enumerate(raw):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        val = json.loads(raw[: i + 1])
                        break
                    except json.JSONDecodeError:
                        return {}
        else:
            return {}
    if isinstance(val, str):  # double-encoded
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            return {}
    return val if isinstance(val, dict) else {}


def assistant_tool_call_message(text: str, calls: list[LLMToolCall]) -> dict[str, Any]:
    """Build the assistant message that must precede ``role: tool`` results."""
    msg: dict[str, Any] = {"role": "assistant", "content": text or None}
    if calls:
        msg["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
            }
            for c in calls
        ]
    return msg


class OpenAICompatLLM:
    """Streaming OpenAI-compatible chat client implementing :class:`eva.interfaces.LLM`."""

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str,
        model: str,
        extra_body: dict[str, Any] | None = None,
        max_tokens: int = 400,
        temperature: float = 0.8,
        *,
        connect_timeout: float = 5.0,
        read_timeout: float = 60.0,
        keepalive_expiry: float = 120.0,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.extra_body = dict(extra_body or {})
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._supports_stream_options: bool | None = None  # unknown until first stream
        self.warmup_s: float | None = None
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
            },
            timeout=httpx.Timeout(connect=connect_timeout, read=read_timeout, write=10.0, pool=5.0),
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2, keepalive_expiry=keepalive_expiry),
            http2=False,
        )

    # ------------------------------------------------------------------ helpers
    def _body(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        stream: bool,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": stream,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        body.update(self.extra_body)
        if overrides:
            body.update(overrides)
        return body

    async def _read_error(self, resp: httpx.Response) -> str:
        try:
            raw = await resp.aread()
            return raw.decode("utf-8", "replace")
        except Exception:  # pragma: no cover - best effort
            return "<unreadable body>"

    # --------------------------------------------------------------- lifecycle
    async def warmup(self) -> None:
        """Establish the (TLS) keep-alive connection with a 1-token completion."""
        t0 = time.perf_counter()
        try:
            await self.complete(
                [{"role": "user", "content": "hi"}],
                max_tokens=1,
                temperature=0.0,
            )
        except Exception as e:  # warmup must never crash the pipeline
            log.warning("%s warmup failed: %s", self.name, e)
        self.warmup_s = time.perf_counter() - t0
        log.info("%s warmup took %.3f s", self.name, self.warmup_s)

    async def ping(self) -> float:
        """Cheap keep-alive request (GET /models); returns seconds taken.

        Call every ~60 s while idle so the pooled connection is not closed by the
        server's idle timeout; a stale socket costs a full reconnect on the next turn.
        """
        t0 = time.perf_counter()
        try:
            r = await self._client.get("/models")
            r.raise_for_status()
        except Exception as e:
            log.debug("%s ping failed: %s", self.name, e)
        return time.perf_counter() - t0

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------- completion
    async def complete(self, messages: list[dict[str, Any]], **kw: Any) -> str:
        """Non-streaming helper (memory summarisation etc.). Returns the content string.

        Keyword arguments are merged into the request body (``max_tokens``,
        ``temperature``, ``tools`` ...).  ``<think>`` blocks are stripped.
        """
        tools = kw.pop("tools", None)
        body = self._body(messages, tools, stream=False, overrides=kw)
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                r = await self._client.post("/chat/completions", json=body)
                if r.status_code >= 400:
                    raise LLMHTTPError(r.status_code, r.text)
                data = r.json()
                msg = data["choices"][0]["message"]
                return strip_think_text(msg.get("content") or "")
            except _RETRYABLE_EXC as e:
                last_exc = e
            except LLMHTTPError as e:
                last_exc = e
                if not (e.status == 429 or e.status >= 500):
                    raise
            if attempt == 0:
                await asyncio.sleep(0.5)
        assert last_exc is not None
        raise last_exc

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMEvent]:
        """Stream one assistant turn. See module docstring for the event contract."""
        t0 = time.perf_counter()
        ttft: float | None = None
        finish_reason = "stop"
        usage: dict[str, Any] = {}
        reasoning_deltas = 0
        reasoning_chars = 0
        content_deltas = 0
        yielded_any = False
        think = _ThinkFilter()
        # index -> accumulating tool call
        pending: dict[int, dict[str, Any]] = {}
        order: list[int] = []
        emitted_idx: set[int] = set()

        def _finish_call(idx: int) -> LLMToolCall | None:
            if idx in emitted_idx:
                return None
            tc = pending.get(idx)
            if not tc or not tc.get("name"):
                return None
            emitted_idx.add(idx)
            return LLMToolCall(
                id=tc.get("id") or f"call_{idx}",
                name=tc["name"],
                arguments=_parse_tool_arguments(tc.get("arguments", "")),
            )

        attempt = 0
        while True:
            attempt += 1
            use_so = self._supports_stream_options is not False
            overrides = {"stream_options": {"include_usage": True}} if use_so else None
            body = self._body(messages, tools, stream=True, overrides=overrides)
            try:
                async with self._client.stream("POST", "/chat/completions", json=body) as resp:
                    if resp.status_code >= 400:
                        err = await self._read_error(resp)
                        if resp.status_code == 400 and use_so and "stream_options" in err:
                            log.info("%s rejects stream_options; retrying without", self.name)
                            self._supports_stream_options = False
                            attempt -= 1  # a capability probe, not a failed attempt
                            continue
                        if (resp.status_code == 429 or resp.status_code >= 500) and attempt == 1:
                            wait = 0.75
                            ra = resp.headers.get("retry-after")
                            if ra and ra.isdigit():
                                wait = min(float(ra), 5.0)
                            log.warning("%s HTTP %s, retrying in %.2fs: %s", self.name, resp.status_code, wait, err[:200])
                            await asyncio.sleep(wait)
                            continue
                        raise LLMHTTPError(resp.status_code, err)
                    if use_so and self._supports_stream_options is None:
                        self._supports_stream_options = True

                    sse_done = False
                    async for line in resp.aiter_lines():
                        # NOTE: never `break` out of this loop.  Leaving the body unread
                        # makes httpx close the socket instead of returning it to the
                        # keep-alive pool, which costs a TCP+TLS handshake per turn.
                        if sse_done or not line or line.startswith(":"):
                            continue
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            sse_done = True
                            continue
                        try:
                            obj = json.loads(payload)
                        except json.JSONDecodeError:
                            log.debug("%s bad SSE json: %r", self.name, payload[:120])
                            continue
                        if obj.get("usage"):
                            usage.update(obj["usage"])
                        if obj.get("time_info"):
                            usage["time_info"] = obj["time_info"]
                        choices = obj.get("choices") or []
                        if not choices:
                            continue
                        ch = choices[0]
                        if ch.get("finish_reason"):
                            finish_reason = ch["finish_reason"]
                        delta = ch.get("delta") or {}

                        # reasoning: count, never surface
                        for key in ("reasoning", "reasoning_content", "thinking"):
                            r = delta.get(key)
                            if r:
                                reasoning_deltas += 1
                                reasoning_chars += len(r)

                        # tool calls
                        for tc in delta.get("tool_calls") or []:
                            idx = tc.get("index", 0)
                            if idx not in pending:
                                # a new index means every earlier one is complete
                                for prev in order:
                                    call = _finish_call(prev)
                                    if call is not None:
                                        yielded_any = True
                                        yield call
                                pending[idx] = {"id": None, "name": None, "arguments": ""}
                                order.append(idx)
                            slot = pending[idx]
                            if tc.get("id"):
                                slot["id"] = tc["id"]
                            fn = tc.get("function") or {}
                            if fn.get("name") and not slot["name"]:
                                slot["name"] = fn["name"]
                            if fn.get("arguments"):
                                slot["arguments"] += fn["arguments"]
                            if ttft is None:
                                ttft = time.perf_counter() - t0

                        # content
                        text = delta.get("content")
                        if text:
                            text = think.feed(text)
                            if text:
                                content_deltas += 1
                                if ttft is None:
                                    ttft = time.perf_counter() - t0
                                yielded_any = True
                                yield LLMDelta(text)
                break  # stream finished normally
            except _RETRYABLE_EXC as e:
                if yielded_any:
                    log.warning("%s stream broke mid-response: %s", self.name, e)
                    finish_reason = "error"
                    usage["error"] = repr(e)
                    break
                if attempt == 1:
                    log.warning("%s connection error, retrying: %s", self.name, e)
                    await asyncio.sleep(0.5)
                    continue
                raise
            except httpx.ReadTimeout as e:
                log.warning("%s read timeout: %s", self.name, e)
                if not yielded_any and attempt == 1:
                    continue
                finish_reason = "error"
                usage["error"] = repr(e)
                break

        # flush anything the think filter held back
        tail = think.flush()
        if tail:
            content_deltas += 1
            if ttft is None:
                ttft = time.perf_counter() - t0
            yield LLMDelta(tail)
        for idx in order:
            call = _finish_call(idx)
            if call is not None:
                yield call
        reasoning_chars += think.reasoning_chars
        usage["reasoning_deltas"] = reasoning_deltas
        usage["reasoning_chars"] = reasoning_chars
        usage["content_deltas"] = content_deltas
        usage["tool_calls"] = len(emitted_idx)
        if order and finish_reason == "stop":
            finish_reason = "tool_calls"
        yield LLMDone(
            finish_reason=finish_reason,
            ttft_s=ttft,
            total_s=time.perf_counter() - t0,
            usage=usage,
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"OpenAICompatLLM({self.name!r}, model={self.model!r}, extra={self.extra_body})"


# ------------------------------------------------------------------ factories
def make_cerebras(model: str, keys: Keys, reasoning: str | None = None, **kw: Any) -> OpenAICompatLLM:
    """Build a Cerebras client with the same extra_body rules as :func:`eva.factory.build_llm`.

    ``reasoning``: ``None``/``"none"`` disables reasoning for qwen; for gpt-oss the
    effort is one of ``low`` / ``medium`` / ``high`` (default ``low``).
    """
    from ..factory import build_llm

    cfg: dict[str, Any] = {"kind": "cerebras", "model": model, "reasoning": reasoning}
    cfg.update(kw)
    return build_llm(cfg, keys)  # type: ignore[return-value]


def make_ollama(model: str, **kw: Any) -> OpenAICompatLLM:
    """Build an Ollama client (127.0.0.1, never localhost) mirroring the factory."""
    from ..factory import build_llm

    cfg: dict[str, Any] = {"kind": "ollama", "model": model}
    cfg.update(kw)
    return build_llm(cfg, Keys(cerebras=None, elevenlabs=None))  # type: ignore[return-value]
