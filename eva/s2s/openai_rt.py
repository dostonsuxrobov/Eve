"""OpenAI Realtime (GA) over a raw WebSocket: 24 kHz PCM in and out, semantic turn detection,
function tools, barge-in by truncating what the caller never heard.

Protocol per developers.openai.com/api/docs/guides/realtime-conversations (read 2026-09-27):
``session.update`` with ``session.type = "realtime"``; audio in ``input_audio_buffer.append``;
``response.output_audio.delta`` out; function calls read from ``response.done``; results as
``function_call_output`` items followed by ``response.create``.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any

from websockets.asyncio.client import connect

from ..config import key_file
from . import PRICES

log = logging.getLogger("eva.s2s.openai")
URL = "wss://api.openai.com/v1/realtime?model={model}"


def tool_schema(tools: list[Any]) -> list[dict[str, Any]]:
    return [{"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters} for t in tools]


def cost_of(model: str, usage: dict[str, Any]) -> float:
    p = PRICES.get(model)
    if not p or not usage:
        return 0.0
    i = usage.get("input_token_details") or {}
    o = usage.get("output_token_details") or {}
    cached = i.get("cached_tokens_details") or {}
    c_text, c_audio = cached.get("text_tokens", 0), cached.get("audio_tokens", 0)
    return (max(0, i.get("text_tokens", 0) - c_text) * p["text_in"] + c_text * p["text_in_cached"]
            + max(0, i.get("audio_tokens", 0) - c_audio) * p["audio_in"] + c_audio * p["audio_in_cached"]
            + o.get("text_tokens", 0) * p["text_out"] + o.get("audio_tokens", 0) * p["audio_out"]) / 1e6


class OpenAIRealtime:
    provider = "openai"
    in_rate = 24_000
    out_rate = 24_000

    def __init__(self, model: str, instructions: str, tools: list[Any], *, voice: str = "marin",
                 turn_detection: str | None = "semantic_vad", eagerness: str = "auto", reasoning: str | None = "low",
                 text_only: bool = False) -> None:
        self.model = model
        self.name = f"openai/{model}"
        self.instructions = instructions
        self.tools = tools
        self.voice = voice
        self.turn_detection = turn_detection
        self.eagerness = eagerness
        self.reasoning = reasoning
        self.text_only = text_only
        self.events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.ws: Any = None
        self._reader: asyncio.Task[None] | None = None
        self.audio_item: str | None = None  # the assistant item whose audio is playing (for truncate)
        self.cost = 0.0

    async def connect(self) -> None:
        key = key_file("openai_key.txt", "OPENAI_API_KEY")
        if not key:
            raise RuntimeError("no OpenAI key: put it in openai_key.txt or OPENAI_API_KEY")
        self.ws = await connect(URL.format(model=self.model), additional_headers={"Authorization": f"Bearer {key}"},
                                max_size=None, ping_interval=None)
        td: dict[str, Any] | None = None
        if self.turn_detection == "semantic_vad":
            td = {"type": "semantic_vad", "eagerness": self.eagerness, "create_response": True, "interrupt_response": True}
        elif self.turn_detection == "server_vad":
            td = {"type": "server_vad", "silence_duration_ms": 500, "create_response": True, "interrupt_response": True}
        session: dict[str, Any] = {
            "type": "realtime", "instructions": self.instructions,
            "output_modalities": ["text"] if self.text_only else ["audio"],
            "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}, "noise_reduction": {"type": "near_field"},
                                "transcription": {"model": "gpt-4o-mini-transcribe", "language": "en"}, "turn_detection": td},
                      "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": self.voice}},
            "tools": tool_schema(self.tools), "tool_choice": "auto",
        }
        if self.reasoning and self.model.startswith("gpt-realtime-2"):
            session["reasoning"] = {"effort": self.reasoning}
        await self._send({"type": "session.update", "session": session})
        self._reader = asyncio.create_task(self._read())

    async def _send(self, obj: dict[str, Any]) -> None:
        await self.ws.send(json.dumps(obj))

    async def _read(self) -> None:
        put = self.events.put_nowait
        try:
            async for raw in self.ws:
                ev = json.loads(raw)
                t = ev.get("type", "")
                if t in ("response.output_audio.delta", "response.audio.delta"):
                    self.audio_item = ev.get("item_id")
                    put({"type": "audio", "pcm": base64.b64decode(ev["delta"])})
                elif t in ("response.output_audio_transcript.delta", "response.output_text.delta", "response.audio_transcript.delta"):
                    put({"type": "text_out", "delta": ev.get("delta", "")})
                elif t == "conversation.item.input_audio_transcription.completed":
                    put({"type": "text_in", "text": ev.get("transcript", "").strip()})
                elif t == "input_audio_buffer.speech_started":
                    put({"type": "speech_started"})
                elif t == "input_audio_buffer.speech_stopped":
                    put({"type": "speech_stopped"})
                elif t == "response.done":
                    r = ev.get("response") or {}
                    calls = [(o["call_id"], o["name"], json.loads(o.get("arguments") or "{}"))
                             for o in r.get("output") or [] if o.get("type") == "function_call"]
                    usage = r.get("usage") or {}
                    c = cost_of(self.model, usage)
                    self.cost += c
                    if calls and r.get("status") == "completed":
                        put({"type": "tool_calls", "calls": calls})
                    put({"type": "turn_done", "usage": usage, "cost": c, "status": r.get("status", ""),
                         "more": bool(calls) and r.get("status") == "completed"})
                elif t == "error":
                    err = ev.get("error") or {}
                    put({"type": "error", "message": f"{err.get('code')}: {err.get('message')}"})
        except Exception as e:  # noqa: BLE001 - the socket closing ends the call
            put({"type": "error", "message": f"connection closed: {e!r}"})
        put({"type": "closed"})

    async def send_audio(self, pcm: bytes) -> None:
        await self._send({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()})

    async def send_text(self, text: str, role: str = "user") -> None:
        kind = "input_text"
        await self._send({"type": "conversation.item.create", "item": {"type": "message", "role": role, "content": [{"type": kind, "text": text}]}})
        await self._send({"type": "response.create"})

    async def send_tool_results(self, results: list[tuple[str, str, str]]) -> None:
        for call_id, _name, output in results:
            await self._send({"type": "conversation.item.create", "item": {"type": "function_call_output", "call_id": call_id, "output": output}})
        await self._send({"type": "response.create"})

    async def truncate(self, played_ms: int) -> None:
        """The caller cut in: keep only what they heard of her last item in the conversation."""
        if self.audio_item:
            await self._send({"type": "conversation.item.truncate", "item_id": self.audio_item, "content_index": 0,
                              "audio_end_ms": max(0, int(played_ms))})
            self.audio_item = None

    async def close(self) -> None:
        if self.ws is not None:
            await self.ws.close()
        if self._reader is not None:
            self._reader.cancel()
