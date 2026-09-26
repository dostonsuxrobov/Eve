"""Ollama's native ``/api/chat`` as an :class:`eva.interfaces.LLM`.

Why not the OpenAI-compatible endpoint (``eva.llm.openai_compat``) for Ollama too?
Hybrid "thinking" models (Qwen3 8B / 14B / 32B, the non-2507 builds) reason before
every reply, and the only switch that reaches them is the native ``think`` field:
measured on ``qwen3:8b`` (Ollama 0.34), ``/api/chat`` with ``think: false`` answered
"Say hi in five words" with 9 tokens in 0.84 s, while ``/v1/chat/completions`` ignored
``think`` (and the ``/no_think`` prompt switch) and spent 736 tokens thinking.

Protocol (verified): NDJSON stream of ``{"message": {"role", "content", "thinking",
"tool_calls"}, "done": bool, ...}``; the last line carries ``done_reason``,
``prompt_eval_count`` and ``eval_count``. Tool calls arrive whole, arguments already
parsed. ``keep_alive`` keeps the model loaded between turns.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator

import httpx

from ..config import USER_AGENT
from ..interfaces import LLMDelta, LLMDone, LLMEvent, LLMToolCall

log = logging.getLogger("eva.llm.ollama")

_RETRYABLE_EXC = (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError, httpx.PoolTimeout)


class OllamaHTTPError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status


class OllamaNativeLLM:
    """Streaming chat over Ollama's native API with ``think`` control."""

    def __init__(
        self,
        name: str,
        base_url: str,
        model: str,
        *,
        think: bool | None = False,
        max_tokens: int = 300,
        temperature: float = 0.8,
        keep_alive: str = "30m",
        options: dict[str, Any] | None = None,
        connect_timeout: float = 5.0,
        read_timeout: float = 120.0,
    ) -> None:
        self.name = name
        self.model = model
        self.think = think
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.keep_alive = keep_alive
        self.options = dict(options or {})
        self.warmup_s: float | None = None
        root = base_url.rstrip("/")
        if root.endswith("/v1"):  # accept the OpenAI-style base too
            root = root[: -len("/v1")]
        self._client = httpx.AsyncClient(
            base_url=root,
            headers={"User-Agent": USER_AGENT},
            timeout=httpx.Timeout(connect=connect_timeout, read=read_timeout, write=10.0, pool=5.0),
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2, keepalive_expiry=120.0),
        )

    # ------------------------------------------------------------------ helpers
    def _body(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, stream: bool, **over: Any) -> dict[str, Any]:
        opts = {"temperature": over.pop("temperature", self.temperature), "num_predict": over.pop("max_tokens", self.max_tokens), **self.options}
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [self._convert(m) for m in messages],
            "stream": stream,
            "keep_alive": self.keep_alive,
            "options": opts,
        }
        if self.think is not None:
            body["think"] = self.think
        if tools:
            body["tools"] = tools
        body.update(over)
        return body

    @staticmethod
    def _convert(m: dict[str, Any]) -> dict[str, Any]:
        """OpenAI-format history -> Ollama's (tool call arguments as objects, tool results plain)."""
        out: dict[str, Any] = {"role": m.get("role", "user"), "content": m.get("content") or ""}
        calls = m.get("tool_calls")
        if calls:
            conv = []
            for c in calls:
                fn = c.get("function") or {}
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args else {}
                    except ValueError:
                        args = {}
                conv.append({"function": {"name": fn.get("name"), "arguments": args}})
            out["tool_calls"] = conv
        return out

    # --------------------------------------------------------------- lifecycle
    async def warmup(self) -> None:
        """Load the model (a 1-token request) so the first turn pays no load time."""
        t0 = time.perf_counter()
        try:
            await self.complete([{"role": "user", "content": "hi"}], max_tokens=1)
        except Exception as e:
            log.warning("%s warmup failed: %s", self.name, e)
        self.warmup_s = time.perf_counter() - t0
        log.info("%s warmup took %.3f s", self.name, self.warmup_s)

    async def ping(self) -> float | None:
        t0 = time.perf_counter()
        try:
            r = await self._client.get("/api/tags")
            r.raise_for_status()
        except Exception as e:
            log.debug("%s ping failed: %s", self.name, e)
            return None
        return time.perf_counter() - t0

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------- completion
    async def complete(self, messages: list[dict[str, Any]], **kw: Any) -> str:
        tools = kw.pop("tools", None)
        body = self._body(messages, tools, stream=False, **kw)
        last: Exception | None = None
        for attempt in range(2):
            try:
                r = await self._client.post("/api/chat", json=body)
                if r.status_code >= 400:
                    raise OllamaHTTPError(r.status_code, r.text)
                return str((r.json().get("message") or {}).get("content") or "")
            except _RETRYABLE_EXC as e:
                last = e
            except OllamaHTTPError as e:
                last = e
                if e.status < 500:
                    raise
            if attempt == 0:
                await asyncio.sleep(0.5)
        assert last is not None
        raise last

    async def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> AsyncIterator[LLMEvent]:
        t0 = time.perf_counter()
        ttft: float | None = None
        usage: dict[str, Any] = {}
        finish = "stop"
        thinking_chars = 0
        content_deltas = 0
        calls = 0
        yielded_any = False
        body = self._body(messages, tools, stream=True)
        attempt = 0
        while True:
            attempt += 1
            try:
                async with self._client.stream("POST", "/api/chat", json=body) as resp:
                    if resp.status_code >= 400:
                        err = (await resp.aread()).decode("utf-8", "replace")
                        if resp.status_code >= 500 and attempt == 1:
                            await asyncio.sleep(0.5)
                            continue
                        raise OllamaHTTPError(resp.status_code, err)
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if obj.get("error"):
                            raise OllamaHTTPError(500, str(obj["error"]))
                        msg = obj.get("message") or {}
                        if msg.get("thinking"):
                            thinking_chars += len(msg["thinking"])
                        for tc in msg.get("tool_calls") or []:
                            fn = tc.get("function") or {}
                            args = fn.get("arguments") or {}
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except ValueError:
                                    args = {}
                            calls += 1
                            if ttft is None:
                                ttft = time.perf_counter() - t0
                            yielded_any = True
                            yield LLMToolCall(id=f"call_{calls}", name=str(fn.get("name")), arguments=args if isinstance(args, dict) else {})
                        text = msg.get("content")
                        if text:
                            content_deltas += 1
                            if ttft is None:
                                ttft = time.perf_counter() - t0
                            yielded_any = True
                            yield LLMDelta(text)
                        if obj.get("done"):
                            finish = str(obj.get("done_reason") or "stop")
                            usage = {
                                "prompt_tokens": obj.get("prompt_eval_count"),
                                "completion_tokens": obj.get("eval_count"),
                                "time_info": {
                                    "load_s": round((obj.get("load_duration") or 0) / 1e9, 3),
                                    "prompt_eval_s": round((obj.get("prompt_eval_duration") or 0) / 1e9, 3),
                                    "eval_s": round((obj.get("eval_duration") or 0) / 1e9, 3),
                                },
                            }
                break
            except _RETRYABLE_EXC as e:
                if yielded_any:
                    finish = "error"
                    usage["error"] = repr(e)
                    break
                if attempt == 1:
                    log.warning("%s connection error, retrying: %s", self.name, e)
                    await asyncio.sleep(0.5)
                    continue
                raise
            except httpx.ReadTimeout as e:
                if not yielded_any and attempt == 1:
                    continue
                finish = "error"
                usage["error"] = repr(e)
                break
        if finish == "length":
            finish = "length"
        if calls and finish == "stop":
            finish = "tool_calls"
        usage["reasoning_chars"] = thinking_chars
        usage["content_deltas"] = content_deltas
        usage["tool_calls"] = calls
        yield LLMDone(finish_reason=finish, ttft_s=ttft, total_s=time.perf_counter() - t0, usage=usage)

    def __repr__(self) -> str:  # pragma: no cover
        return f"OllamaNativeLLM({self.name!r}, model={self.model!r}, think={self.think})"


__all__ = ["OllamaNativeLLM", "OllamaHTTPError"]
