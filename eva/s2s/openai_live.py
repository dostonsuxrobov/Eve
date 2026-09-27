"""OpenAI GPT-Live (``gpt-live-1``) over a raw WebSocket: a full-duplex voice model that listens
and speaks at once and delegates reasoning and tools to a Responses backend model (gpt-5.6-luna by
default, ``sol`` for harder work). Our tools run here; their results go back as Responses items.

Protocol per developers.openai.com/api/docs/guides/voice-websockets?api=live, live-delegation and
live-conversations (read 2026-09-27): ``session.start`` first, then ``session.started``;
``session.input_audio.append`` in; ``session.output_audio.delta`` and the transcript deltas out;
backend events nested in ``response.event``; a function call is read from a nested
``response.output_item.done`` and answered with ``response.item.create`` + ``response.create``.
There is no end-of-turn event: a turn ends here when her audio has been quiet for ``quiet_s``
with no backend work running.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import Any

from websockets.asyncio.client import connect

from ..config import key_file

log = logging.getLogger("eva.s2s.live")
URL = "wss://api.openai.com/v1/live/sessions"
VOICE_PER_MIN = 0.05  # $ per minute of session, billed per second
BACKEND_PRICES = {"gpt-5.6-luna": (0.20, 0.02, 1.20), "gpt-5.6-terra": (2.0, 0.20, 12.0), "gpt-5.6-sol": (4.0, 0.40, 20.0)}
LIVE_NOTE = (
    "\n\nHow you work on this call. You are the voice. You can't see Red Oak's systems yourself: every "
    "lookup, price, booking, update or message goes to your backend, which has the tools and the "
    "database. Delegate as soon as you know what's needed, say a short natural line while it works, "
    "and speak only the facts it returns. Never state a number, a name, a time or a load number the "
    "backend hasn't given you in this call."
)


class OpenAILive:
    provider = "openai-live"
    in_rate = 24_000
    out_rate = 24_000

    def __init__(self, model: str, instructions: str, tools: list[Any], *, voice: str = "gleam",
                 backend: str = "gpt-5.6-luna", reasoning: str = "low", quiet_s: float = 1.2,
                 backend_instructions: str | None = None, **_: Any) -> None:
        self.model = model
        self.backend = backend
        self.name = f"openai/{model}+{backend}"
        self.instructions = instructions
        # with its own backend prompt the voice prompt already carries the delegation policy; without,
        # the backend gets the voice prompt and the voice a note to delegate (the first version)
        self.backend_instructions = backend_instructions
        self.tools = tools
        self.voice = voice
        self.reasoning = reasoning
        self.quiet_s = quiet_s
        self.cover_after_s = 2.0
        self.events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.ws: Any = None
        self._reader: asyncio.Task[None] | None = None
        self._watch: asyncio.Task[None] | None = None
        self._pending_calls: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}
        self._busy: set[str] = set()  # backend responses still running
        self._last_out = 0.0
        self._spoke = False
        self._seconds = 0.0
        self._voice_cost_at_turn = 0.0
        self.backend_cost = 0.0
        self._turn_backend = 0.0

    @property
    def cost(self) -> float:
        return self._seconds / 60 * VOICE_PER_MIN + self.backend_cost

    async def connect(self) -> None:
        key = key_file("openai_key.txt", "OPENAI_API_KEY")
        if not key:
            raise RuntimeError("no OpenAI key: put it in openai_key.txt or OPENAI_API_KEY")
        self.ws = await connect(URL, additional_headers={"Authorization": f"Bearer {key}"}, max_size=None, ping_interval=None)
        backend: dict[str, Any] = {
            "model": self.backend, "instructions": self.backend_instructions or self.instructions,
            "tools": [{"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters} for t in self.tools],
            "tool_choice": "auto", "parallel_tool_calls": True,
        }
        if self.reasoning:
            backend["reasoning"] = {"effort": self.reasoning}
        await self._send({"type": "session.start", "event_id": "start", "session": {
            "model": self.model, "instructions": self.instructions + ("" if self.backend_instructions else LIVE_NOTE),
            "audio": {"format": {"type": "audio/pcm", "rate": 24000}, "output": {"voice": self.voice}},
            "delegation": {"type": "responses", "responses": backend},
        }})
        while True:
            ev = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=20))
            if ev.get("type") == "session.started":
                break
            if ev.get("type") == "error":
                raise RuntimeError(f"GPT-Live session failed: {ev.get('error')}")
        self._reader = asyncio.create_task(self._read())
        self._watch = asyncio.create_task(self._watchdog())

    async def _send(self, obj: dict[str, Any]) -> None:
        await self.ws.send(json.dumps(obj))

    def _out(self) -> None:
        self._last_out = time.perf_counter()
        self._spoke = True

    async def _watchdog(self) -> None:
        """No end-of-turn event on GPT-Live: her side is done when she's been quiet for a moment
        and no backend work is running."""
        while True:
            await asyncio.sleep(0.2)
            if self._spoke and not self._busy and not self._pending_calls and time.perf_counter() - self._last_out > self.quiet_s:
                self._spoke = False
                voice = self._seconds / 60 * VOICE_PER_MIN
                c = voice - self._voice_cost_at_turn + self._turn_backend
                self._voice_cost_at_turn, self._turn_backend = voice, 0.0
                self.events.put_nowait({"type": "turn_done", "usage": {"seconds": self._seconds}, "cost": c,
                                        "status": "completed", "more": False})

    async def _read(self) -> None:
        put = self.events.put_nowait
        try:
            async for raw in self.ws:
                ev = json.loads(raw)
                t = ev.get("type", "")
                if t == "session.output_audio.delta":
                    self._out()
                    put({"type": "audio", "pcm": base64.b64decode(ev["delta"])})
                elif t == "session.output_transcript.delta":
                    self._out()
                    put({"type": "text_out", "delta": ev.get("delta", "")})
                elif t == "session.input_transcript.delta":
                    put({"type": "text_in_part", "text": ev.get("delta", "")})
                elif t == "session.usage.updated":
                    self._seconds = float((ev.get("usage") or {}).get("seconds", self._seconds))
                elif t == "session.delegation.created":
                    rid = ev.get("response_id") or ev.get("delegation_id") or "?"
                    self._busy.add(rid)
                    asyncio.create_task(self._cover_the_wait(time.perf_counter()))
                elif t == "response.event":
                    self._backend(ev.get("event") or {})
                elif t == "error":
                    err = ev.get("error") or {}
                    put({"type": "error", "message": f"{err.get('code')}: {err.get('message')}"})
                elif t == "session.closed":
                    self._seconds = float((ev.get("usage") or {}).get("seconds", self._seconds))
                    break
        except Exception as e:  # noqa: BLE001
            put({"type": "error", "message": f"connection closed: {e!r}"})
        put({"type": "closed"})

    async def _cover_the_wait(self, since: float) -> None:
        """A backend answer can take a while (12.6 s of silence on a broker's counter-offer, 2026-09-27):
        if she hasn't said anything 2 s after handing work off, have her say she's checking."""
        await asyncio.sleep(self.cover_after_s)
        if self._busy and self._last_out < since:
            try:
                await self._send({"type": "session.commentary.append", "delegation_id": None,
                                  "content": "Still checking that in the system; tell the caller briefly, in your own words."})
                self.events.put_nowait({"type": "covered_wait"})
            except Exception:  # noqa: BLE001 - the call may be closing
                pass

    def _backend(self, e: dict[str, Any]) -> None:
        t = e.get("type", "")
        resp = e.get("response") or {}
        rid = resp.get("id") or e.get("response_id") or "?"
        if t == "response.created":
            self._busy.add(rid)
        elif t == "response.output_item.done":
            item = e.get("item") or {}
            if item.get("type") == "function_call":
                args = json.loads(item.get("arguments") or "{}")
                self._pending_calls.setdefault(rid, []).append((item["call_id"], item["name"], args))
        elif t in ("response.completed", "response.done", "response.failed", "response.incomplete", "response.cancelled"):
            usage = resp.get("usage") or {}
            p = BACKEND_PRICES.get(self.backend)
            if p and usage:
                cached = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
                c = ((usage.get("input_tokens", 0) - cached) * p[0] + cached * p[1] + usage.get("output_tokens", 0) * p[2]) / 1e6
                self.backend_cost += c
                self._turn_backend += c
            self._busy.discard(rid)
            self._busy = {b for b in self._busy if b != "?"}
            calls = self._pending_calls.pop(rid, None)
            if calls is None and self._pending_calls:  # response id not echoed: take whatever is waiting
                calls = [c for v in self._pending_calls.values() for c in v]
                self._pending_calls.clear()
            if calls and t in ("response.completed", "response.done"):
                self.events.put_nowait({"type": "tool_calls", "calls": calls})

    async def send_audio(self, pcm: bytes) -> None:
        await self._send({"type": "session.input_audio.append", "audio": base64.b64encode(pcm).decode()})

    async def send_text(self, text: str, role: str = "user") -> None:
        """An instruction for the voice (the greeting); GPT-Live takes the caller's words as audio."""
        await self._send({"type": "session.instructions.append", "delegation_id": None, "content": text.strip("[]")})

    async def send_tool_results(self, results: list[tuple[str, str, str]]) -> None:
        for call_id, _name, output in results:
            await self._send({"type": "response.item.create", "item": {"type": "function_call_output", "call_id": call_id, "output": output}})
        await self._send({"type": "response.create"})

    async def truncate(self, played_ms: int) -> None:  # full duplex: it hears the caller and stops itself
        return None

    async def close(self) -> None:
        if self.ws is not None:
            try:
                await self._send({"type": "session.close"})
                await asyncio.sleep(0.5)
            except Exception:  # noqa: BLE001
                pass
            await self.ws.close()
        for task in (self._reader, self._watch):
            if task is not None:
                task.cancel()
