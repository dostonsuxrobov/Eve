"""One live call with a speech-to-speech client: our tools run for the model, barge-in stops the
player, and every turn is recorded with its latency, tool calls and cost.

Used by run_s2s.py (the laptop's mic and speakers) and bench/dispatch_eval.py (typed caller lines,
her audio collected, not played).
"""
from __future__ import annotations

import asyncio
import time

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable

from ..interfaces import LLMToolCall
from ..tools import execute

EventHandler = Callable[[str, dict[str, Any]], None]


@dataclass
class Turn:
    heard: str = ""
    said: str = ""
    tools: list[dict[str, Any]] = field(default_factory=list)
    t_start: float = 0.0  # when the caller's line was sent (typed) or their speech ended (voice)
    first_audio_s: float | None = None
    first_text_s: float | None = None
    done_s: float | None = None
    audio_s: float = 0.0
    cost: float = 0.0
    interrupted: bool = False
    status: str = ""
    voiced: list[float] = field(default_factory=list)  # perf_counter of every chunk of her audio above -45 dBFS


class Call:
    def __init__(self, client: Any, tools: list[Any], *, player: Any = None, on_event: EventHandler | None = None) -> None:
        self.client = client
        self.tools = tools
        self.player = player
        self.on_event = on_event or (lambda name, data: None)
        self.turns: list[Turn] = []
        self.turn = Turn(t_start=time.perf_counter())
        self.pending_tools = 0
        self.last_event_t = time.perf_counter()
        self.turn_done = asyncio.Event()
        self.ended = False
        self.closed = False
        self.errors: list[str] = []
        self._task: asyncio.Task[None] | None = None
        self._heard_parts: list[str] = []
        self._audio_bytes = 0

    @property
    def cost(self) -> float:
        return getattr(self.client, "cost", 0.0)

    async def start(self) -> None:
        await self.client.connect()
        self._task = asyncio.create_task(self._loop())

    def mark_start(self, t: float | None = None) -> None:
        """A new caller turn begins now (typed line sent, or the local VAD heard them stop)."""
        if self.turn.said or self.turn.tools or self.turn.first_audio_s is not None:
            self._finish()
        self.turn = Turn(t_start=t if t is not None else time.perf_counter())
        self.turn_done.clear()

    def _finish(self) -> None:
        if self._heard_parts and not self.turn.heard:
            self.turn.heard = "".join(self._heard_parts).strip()
        self._heard_parts = []
        self.turn.audio_s = self._audio_bytes / 2 / self.client.out_rate
        self._audio_bytes = 0
        self.turns.append(self.turn)
        self.on_event("turn", {"turn": self.turn})

    async def _loop(self) -> None:
        c = self.client
        while True:
            ev = await c.events.get()
            self.last_event_t = time.perf_counter()
            t = ev["type"]
            now = time.perf_counter() - self.turn.t_start
            if t == "audio":
                pcm = np.frombuffer(ev["pcm"], np.int16)
                if pcm.size and float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2))) > 184:  # -45 dBFS: speech, not silence
                    self.turn.voiced.append(self.last_event_t)
                if self.turn.first_audio_s is None:
                    self.turn.first_audio_s = now
                    self.on_event("first_audio", {"s": now})
                self._audio_bytes += len(ev["pcm"])
                if self.player is not None:
                    self.player.write(ev["pcm"])
            elif t == "text_out":
                if self.turn.first_text_s is None:
                    self.turn.first_text_s = now
                self.turn.said += ev["delta"]
                self.on_event("text_out", {"delta": ev["delta"]})
            elif t == "text_in":
                self.turn.heard = ev["text"]
                self.on_event("text_in", {"text": ev["text"]})
            elif t == "text_in_part":
                self._heard_parts.append(ev["text"])
            elif t in ("speech_started", "interrupted"):
                if self.player is not None and self.player.buffered_samples() > 0:
                    played_ms = int(self.player.played_seconds() * 1000)
                    self.player.stop()
                    self.turn.interrupted = True
                    await c.truncate(played_ms)
                    self.on_event("barge_in", {"played_ms": played_ms})
            elif t == "tool_calls":
                self.pending_tools += 1
                asyncio.create_task(self._run_tools(ev["calls"]))
            elif t == "turn_done":
                self.turn.cost += ev.get("cost", 0.0)
                self.turn.status = ev.get("status", "")
                if not ev.get("more") and self.pending_tools == 0:
                    self.turn.done_s = now
                    self.turn_done.set()
                    self.on_event("turn_done", {"cost": ev.get("cost", 0.0), "status": self.turn.status})
            elif t == "error":
                self.errors.append(ev["message"])
                self.on_event("error", {"message": ev["message"]})
            elif t == "closed":
                self.closed = True
                self.turn_done.set()
                return

    async def _run_tools(self, calls: list[tuple[str, str, dict[str, Any]]]) -> None:
        async def one(call_id: str, name: str, args: dict[str, Any]) -> tuple[str, str, str]:
            t0 = time.perf_counter()
            self.on_event("tool", {"name": name, "args": args})
            result = await execute(LLMToolCall(call_id, name, args), self.tools)
            ms = int((time.perf_counter() - t0) * 1000)
            self.turn.tools.append({"name": name, "args": args, "result": result, "ms": ms})
            self.on_event("tool_result", {"name": name, "ms": ms, "result": result})
            if name == "end_conversation":
                self.ended = True
            return call_id, name, result

        try:
            results = await asyncio.gather(*(one(*c) for c in calls))
            await self.client.send_tool_results(list(results))
        finally:
            self.pending_tools -= 1

    async def ask(self, text: str, *, quiet_s: float = 1.5, timeout_s: float = 90.0) -> Turn:
        """Send a typed caller line; return her whole answer (tool rounds included) once the line goes quiet."""
        self.mark_start()
        self.turn.heard = text
        await self.client.send_text(text)
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline and not self.closed:
            try:
                await asyncio.wait_for(self.turn_done.wait(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if self.pending_tools == 0 and time.perf_counter() - self.last_event_t >= quiet_s:
                break
            await asyncio.sleep(0.2)
        turn = self.turn
        self._finish()
        self.turn = Turn(t_start=time.perf_counter())
        return turn

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
        await self.client.close()
