"""Gemini Live over a raw WebSocket (API key): 16 kHz PCM in, 24 kHz out, Google's own turn
detection and barge-in, function tools.

Protocol per ai.google.dev/api/live (read 2026-09-27): a ``setup`` message first, then
``setupComplete``; ``realtimeInput.audio`` in; ``serverContent`` with audio parts, transcriptions,
``interrupted`` and ``turnComplete`` out; ``toolCall`` answered with ``toolResponse``. Messages may
arrive as binary frames holding JSON.
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

log = logging.getLogger("eva.s2s.gemini")
URL = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={key}"


def _gschema(s: dict[str, Any]) -> dict[str, Any]:
    """JSON schema -> the API's Schema (upper-case types)."""
    out: dict[str, Any] = {}
    if "type" in s:
        out["type"] = str(s["type"]).upper()
    if "description" in s:
        out["description"] = s["description"]
    if "properties" in s:
        out["properties"] = {k: _gschema(v) for k, v in s["properties"].items()}
    if s.get("required"):
        out["required"] = list(s["required"])
    if "items" in s:
        out["items"] = _gschema(s["items"])
    if "enum" in s:
        out["enum"] = s["enum"]
    return out


def declarations(tools: list[Any], behavior: str | None) -> list[dict[str, Any]]:
    decls = []
    for t in tools:
        d: dict[str, Any] = {"name": t.name, "description": t.description}
        if t.parameters.get("properties"):
            d["parameters"] = _gschema(t.parameters)
        if behavior:
            d["behavior"] = behavior
        decls.append(d)
    return decls


def cost_of(model: str, usage: dict[str, Any]) -> float:
    p = PRICES.get(model)
    if not p or not usage:
        return 0.0

    def by(key: str) -> dict[str, int]:
        return {d.get("modality", ""): d.get("tokenCount", 0) for d in usage.get(key) or []}

    pin, pout = by("promptTokensDetails"), by("responseTokensDetails")
    if not pin and not pout:  # no breakdown: count everything as text in and audio out
        return (usage.get("promptTokenCount", 0) * p["text_in"] + usage.get("responseTokenCount", 0) * p["audio_out"]) / 1e6
    thoughts = usage.get("thoughtsTokenCount", 0)
    return (pin.get("AUDIO", 0) * p["audio_in"] + (pin.get("TEXT", 0) + usage.get("toolUsePromptTokenCount", 0)) * p["text_in"]
            + pout.get("AUDIO", 0) * p["audio_out"] + (pout.get("TEXT", 0) + thoughts) * p["text_out"]) / 1e6


class GeminiLive:
    provider = "gemini"
    in_rate = 16_000
    out_rate = 24_000

    def __init__(self, model: str, instructions: str, tools: list[Any], *, voice: str = "Aoede",
                 behavior: str | None = "BLOCKING", thinking: str | None = None, **_: Any) -> None:
        self.model = model
        self.name = f"gemini/{model}"
        self.instructions = instructions
        self.tools = tools
        self.voice = voice
        self.behavior = None if "extended-thinking" in model else behavior  # extended thinking: non-blocking only
        self.thinking = thinking or ("LOW" if "extended-thinking" in model else None)
        self.events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.ws: Any = None
        self._reader: asyncio.Task[None] | None = None
        self._usage: dict[str, Any] = {}
        self.cost = 0.0

    async def connect(self) -> None:
        key = key_file("google_api_key.txt", "GEMINI_API_KEY")
        if not key:
            raise RuntimeError("no Google key: put it in google_api_key.txt or GEMINI_API_KEY")
        self.ws = await connect(URL.format(key=key), max_size=None, ping_interval=None)
        gen: dict[str, Any] = {"responseModalities": ["AUDIO"],
                               "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": self.voice}}}}
        if self.thinking:
            gen["thinkingConfig"] = {"thinkingLevel": self.thinking}
        setup = {"setup": {
            "model": f"models/{self.model}", "generationConfig": gen,
            "systemInstruction": {"parts": [{"text": self.instructions}]},
            "inputAudioTranscription": {}, "outputAudioTranscription": {},
            "realtimeInputConfig": {"activityHandling": "START_OF_ACTIVITY_INTERRUPTS"},
            "contextWindowCompression": {"slidingWindow": {}},
        }}
        if self.tools:  # an empty declaration list is refused
            setup["setup"]["tools"] = [{"functionDeclarations": declarations(self.tools, self.behavior)}]
        await self.ws.send(json.dumps(setup))
        first = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=20))
        if "setupComplete" not in first:
            raise RuntimeError(f"Gemini Live setup failed: {str(first)[:300]}")
        self._reader = asyncio.create_task(self._read())

    async def _read(self) -> None:
        put = self.events.put_nowait
        try:
            async for raw in self.ws:
                msg = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
                if "usageMetadata" in msg:
                    self._usage = msg["usageMetadata"]
                sc = msg.get("serverContent")
                if sc:
                    for part in (sc.get("modelTurn") or {}).get("parts") or []:
                        data = (part.get("inlineData") or {}).get("data")
                        if data:
                            put({"type": "audio", "pcm": base64.b64decode(data)})
                    if (sc.get("outputTranscription") or {}).get("text"):
                        put({"type": "text_out", "delta": sc["outputTranscription"]["text"]})
                    if (sc.get("inputTranscription") or {}).get("text"):
                        put({"type": "text_in_part", "text": sc["inputTranscription"]["text"]})
                    if sc.get("interrupted"):
                        put({"type": "interrupted"})
                    if sc.get("turnComplete"):
                        c = cost_of(self.model, self._usage)
                        self.cost += c
                        put({"type": "turn_done", "usage": self._usage, "cost": c, "status": "completed", "more": False})
                        self._usage = {}
                tc = msg.get("toolCall")
                if tc:
                    put({"type": "tool_calls", "calls": [(f.get("id", ""), f["name"], f.get("args") or {}) for f in tc.get("functionCalls") or []]})
                if "toolCallCancellation" in msg:
                    put({"type": "tool_cancelled", "ids": msg["toolCallCancellation"].get("ids", [])})
                if "goAway" in msg:
                    put({"type": "error", "message": f"server going away in {msg['goAway'].get('timeLeft')}"})
        except Exception as e:  # noqa: BLE001
            put({"type": "error", "message": f"connection closed: {e!r}"})
        put({"type": "closed"})

    async def send_audio(self, pcm: bytes) -> None:
        await self.ws.send(json.dumps({"realtimeInput": {"audio": {"data": base64.b64encode(pcm).decode(), "mimeType": "audio/pcm;rate=16000"}}}))

    async def send_text(self, text: str, role: str = "user") -> None:
        await self.ws.send(json.dumps({"clientContent": {"turns": [{"role": "user", "parts": [{"text": text}]}], "turnComplete": True}}))

    async def send_tool_results(self, results: list[tuple[str, str, str]]) -> None:
        responses = []
        for call_id, name, output in results:
            try:
                body = json.loads(output)
            except ValueError:
                body = output
            responses.append({"id": call_id, "name": name, "response": {"result": body}})
        await self.ws.send(json.dumps({"toolResponse": {"functionResponses": responses}}))

    async def truncate(self, played_ms: int) -> None:  # Gemini drops what wasn't played by itself
        return None

    async def close(self) -> None:
        if self.ws is not None:
            await self.ws.close()
        if self._reader is not None:
            self._reader.cancel()
