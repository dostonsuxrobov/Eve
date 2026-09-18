"""End-to-end simulator for the Eva pipeline.

Two modes:

``--mock``
    Runs the pipeline against ``eva.mocks`` doubles (no audio device, no network,
    no sibling modules) and asserts the behaviours that matter: metrics on a
    normal 2-turn chat, barge-in truncation + stop latency, tool-call hints,
    fillers, timer events, typed input and thinking-phase merges.

real mode (default)
    Builds the preset's real STT/LLM/TTS via ``eva.factory``, the real Silero
    segmenter and either the real ``Player`` (``--speakers``) or a ``MockPlayer``
    (``--silent``), then feeds the sample utterances as mic frames.  Prints a
    per-turn table and writes ``bench/out/e2e_<preset>.json``.

Examples::

    .venv/Scripts/python.exe bench/e2e_sim.py --mock
    .venv/Scripts/python.exe bench/e2e_sim.py --preset cloud-fast \
        --utterances samples/user_hello.wav,samples/user_rough_day.wav --gap 6 --silent
    .venv/Scripts/python.exe bench/e2e_sim.py --preset cloud-fast \
        --utterances samples/user_hello.wav,samples/user_task.wav --barge-in-at 1.2 --speakers
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.markup import escape  # noqa: E402
from rich.table import Table  # noqa: E402

from eva.config import PRESETS, SAMPLES_DIR, PipelineSettings, load_keys  # noqa: E402
from eva.interfaces import MIC_SAMPLE_RATE, Tool, TurnMetrics  # noqa: E402
from eva.mocks import (  # noqa: E402
    MockLLM,
    MockPlayer,
    MockSTT,
    MockStreamingSTT,
    MockTTS,
    ScriptedError,
    ScriptedSegmenter,
    ScriptedToolCall,
    StandInChunker,
    silent_frames,
    standin_clean_for_tts,
    standin_execute,
)
from eva.pipeline import EMPTY_REPLY_PREFILL, EMPTY_REPLY_TEXT, INTERRUPTED_MARK, LLM_FAILURE_TEXT, VoiceAgent  # noqa: E402

console = Console()
OUT_DIR = ROOT / "bench" / "out"


# ------------------------------------------------------------------ helpers
class EventLog:
    """Collects pipeline events with timestamps; optional live printing."""

    def __init__(self, verbose: bool = False, player: Any = None) -> None:
        self.events: list[tuple[float, str, dict[str, Any]]] = []
        self.verbose = verbose
        self.player = player
        self.t0 = time.perf_counter()
        self.hooks: list[Callable[[str, dict[str, Any]], None]] = []

    def __call__(self, name: str, data: dict[str, Any]) -> None:
        now = time.perf_counter()
        if name == "barge_in" and self.player is not None:
            data = {**data, "player_active_after_stop": bool(self.player.is_active)}
        self.events.append((now, name, data))
        if self.verbose and name not in ("state",):
            console.print(f"[dim]{now - self.t0:7.3f}[/] {name:18} {escape(json.dumps(data, default=str)[:160])}")
        for h in self.hooks:
            h(name, data)

    def first(self, name: str) -> tuple[float, dict[str, Any]] | None:
        for t, n, d in self.events:
            if n == name:
                return t, d
        return None

    def all(self, name: str) -> list[tuple[float, dict[str, Any]]]:
        return [(t, d) for t, n, d in self.events if n == name]


def _mock_agent(
    *,
    stt: MockSTT | MockStreamingSTT,
    llm: MockLLM,
    tts: MockTTS,
    player: MockPlayer,
    segmenter: ScriptedSegmenter | None,
    frames: AsyncIterator[np.ndarray] | None,
    settings: PipelineSettings,
    log: EventLog,
    tools: list[Tool] | None = None,
    fillers: list[str] | None = None,
    max_turns: int | None = None,
    pending_events: asyncio.Queue | None = None,
) -> VoiceAgent:
    return VoiceAgent(
        stt,
        llm,
        tts,
        "You are Eva, a warm voice companion.",
        tools or [],
        settings,
        frames=frames,
        segmenter=segmenter,
        player=player,
        fillers=fillers,
        on_event=log,
        max_turns=max_turns,
        chunker_factory=lambda: StandInChunker(settings.first_chunk_min_chars, 6),
        sanitizer=standin_clean_for_tts,
        tool_executor=standin_execute,
        pending_events=pending_events if pending_events is not None else asyncio.Queue(),
    )


def _settings(**over: Any) -> PipelineSettings:
    return dataclasses.replace(PipelineSettings(), **over)


def _turn_rows(turns: list[TurnMetrics]) -> list[dict[str, Any]]:
    rows = []
    for m in turns:
        rows.append(
            {
                "user_text": m.user_text,
                "assistant_text": m.assistant_text,
                "interrupted": m.interrupted,
                "response_latency": None if m.response_latency() is None else round(m.response_latency(), 3),
                **{k: v for k, v in m.breakdown().items()},
                "audio_seconds": None
                if m.audio_started is None or m.audio_finished is None
                else round(m.audio_finished - m.audio_started, 3),
            }
        )
    return rows


def print_turn_table(title: str, rows: list[dict[str, Any]]) -> None:
    table = Table(title=title, show_lines=True)
    for col in ("#", "user", "assistant", "stt", "llm_ttft", "tts_ttfa", "total", "int"):
        table.add_column(col, overflow="fold", max_width=None if col in ("user", "assistant") else 9)

    def f(v: float | None) -> str:
        return "-" if v is None else f"{v:.3f}"

    for i, r in enumerate(rows, 1):
        table.add_row(
            str(i),
            escape(r["user_text"]),
            escape(r["assistant_text"]),
            f(r["stt"]),
            f(r["llm_ttft"]),
            f(r["tts_ttfa"]),
            f(r["total"]),
            "yes" if r["interrupted"] else "",
        )
    console.print(table)


# --------------------------------------------------------------- mock tests
class Check:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.notes: list[str] = []

    def ok(self, cond: bool, msg: str) -> None:
        if not cond:
            self.failures.append(msg)

    def note(self, msg: str) -> None:
        self.notes.append(msg)


async def test_normal_two_turns(verbose: bool) -> tuple[Check, dict[str, Any]]:
    """(a) two utterances, two replies, full metrics on each."""
    c = Check()
    player = MockPlayer(24_000)
    log = EventLog(verbose, player)
    stt = MockSTT(["Hey Eva, how's it going? I just got home from work.", "Honestly, today was rough."], delay_s=0.3)
    llm = MockLLM(
        ["Hey! Welcome home. How was your day?", "Oh no. That sounds heavy. Want to talk about it?"],
        ttft_s=0.4,
        token_delay_s=0.02,
    )
    tts = MockTTS(ttfa_s=0.25, realtime_factor=0.3)
    seg = ScriptedSegmenter(script=[(0.3, 1.2), (6.5, 1.5)])
    settings = _settings(filler_after_ms=1500)
    agent = _mock_agent(
        stt=stt, llm=llm, tts=tts, player=player, segmenter=seg, frames=silent_frames(30), settings=settings,
        log=log, fillers=["hmm"], max_turns=2,
    )
    turns = await agent.run()
    c.ok(len(turns) == 2, f"expected 2 turns, got {len(turns)}")
    for i, m in enumerate(turns):
        b = m.breakdown()
        c.ok(all(v is not None for v in b.values()), f"turn {i}: breakdown has None: {b}")
        c.ok(not m.interrupted, f"turn {i}: unexpectedly interrupted")
        lat = m.response_latency()
        c.ok(lat is not None and 0.85 <= lat <= 1.3, f"turn {i}: latency {lat} outside 0.85-1.3 s (stt .3 + ttft .4 + ttfa .25)")
        c.ok(m.assistant_text == llm._replies[i], f"turn {i}: assistant text {m.assistant_text!r}")
    c.ok(not log.all("filler"), "filler fired although audio arrived before 1.5 s")
    c.ok(len(agent.messages) == 4, f"history should have 4 messages, has {len(agent.messages)}")
    c.ok(agent.messages[1]["role"] == "assistant" and agent.messages[1]["content"] == llm._replies[0], "history[1] wrong")
    # echo guard: threshold raised by 0.25 while speaking and restored after
    states = [d for _, d in log.all("state")]
    c.ok(any(s["to"] == "speaking" for s in states), "never entered SPEAKING")
    c.ok(abs(seg.threshold - 0.5) < 1e-9, f"threshold not restored: {seg.threshold}")
    c.ok(not player.is_active, "player still active at the end")
    return c, {"turns": _turn_rows(turns)}


async def test_barge_in(verbose: bool) -> tuple[Check, dict[str, Any]]:
    """(b) user starts talking 1.2 s into a long reply."""
    c = Check()
    long_reply = (
        "So here's the thing about today. I was thinking about what you said yesterday and honestly "
        "it stuck with me for a while. There's something about the way you described that meeting "
        "that made me wonder whether the real issue is the project or the way people talk to you "
        "about it. Anyway, I'm curious what you think now that you've had some sleep."
    )
    player = MockPlayer(24_000)
    log = EventLog(verbose, player)
    stt = MockSTT(["Tell me what you think.", "Wait, hold on a second."], delay_s=0.3)
    llm = MockLLM([long_reply, "Sure, go ahead."], ttft_s=0.4, token_delay_s=0.015)
    tts = MockTTS(ttfa_s=0.25, realtime_factor=0.3)
    seg = ScriptedSegmenter(script=[(0.3, 1.0)])
    settings = _settings(filler_after_ms=1500, barge_in_min_speech_ms=300)
    barge_at = 1.2

    def hook(name: str, data: dict[str, Any]) -> None:
        if name == "audio_start" and not log.all("barge_in"):
            seg.schedule(time.perf_counter() + barge_at, 1.5)

    log.hooks.append(hook)
    agent = _mock_agent(
        stt=stt, llm=llm, tts=tts, player=player, segmenter=seg, frames=silent_frames(30), settings=settings,
        log=log, max_turns=2,
    )
    turns = await agent.run()
    c.ok(len(turns) == 2, f"expected 2 turns, got {len(turns)}")
    if not turns:
        return c, {}
    first = turns[0]
    c.ok(first.interrupted, "first turn not marked interrupted")
    bi = log.first("barge_in")
    c.ok(bi is not None, "no barge_in event")
    stop_ms = reaction_ms = None
    if bi is not None:
        _, d = bi
        stop_ms, reaction_ms = d["stop_ms"], d["reaction_ms"]
        c.ok(stop_ms is not None and stop_ms < 100, f"player.stop() took {stop_ms} ms")
        c.ok(reaction_ms is not None and reaction_ms < 100, f"stopped {reaction_ms} ms after the 300 ms confirmation point (>100)")
        c.ok(d.get("player_active_after_stop") is False, "player still active right after barge-in")
        c.ok(d["state_before"] == "speaking", f"barge-in should hit while speaking, was {d['state_before']}")
        played_s = d["played_s"]
        c.ok(1.3 <= played_s <= 1.8, f"played {played_s}s, expected ~1.2 s + 0.3 s confirmation")
    stored = [m for m in agent.messages if m["role"] == "assistant"]
    c.ok(len(stored) == 2, f"expected 2 assistant messages, got {len(stored)}")
    if stored:
        text = stored[0]["content"]
        c.ok(text.endswith(INTERRUPTED_MARK), f"stored text does not end with [interrupted]: {text!r}")
        heard = text[: -len(INTERRUPTED_MARK)]
        c.ok(bool(heard.strip()), "heard text is empty")
        c.ok(long_reply.startswith(heard), f"heard text is not a prefix of the reply: {heard!r}")
        n = len(heard.split())
        c.ok(1 <= n <= 8, f"heard {n} words for ~1.5 s of audio at 15 chars/s (expected 1-8)")
        c.note(f"heard {n} words: {heard!r}")
    c.ok(turns[1].user_text == "Wait, hold on a second." and not turns[1].interrupted, "second turn wrong")
    c.ok(not player.is_active, "player still active at the end")
    # no leaked tasks
    leaked = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and (t.get_name().startswith("eva-"))]
    c.ok(not leaked, f"leaked tasks: {[t.get_name() for t in leaked]}")
    return c, {"turns": _turn_rows(turns), "stop_ms": stop_ms, "reaction_ms": reaction_ms, "heard": first.assistant_text}


async def test_tool_call(verbose: bool) -> tuple[Check, dict[str, Any]]:
    """(c) a timer tool call with its spoken hint covering the wait."""
    c = Check()
    calls: list[dict[str, Any]] = []

    async def set_timer(minutes: int, label: str = "timer") -> str:
        calls.append({"minutes": minutes, "label": label})
        await asyncio.sleep(0.2)  # a slow tool, so the hint matters
        return f"Timer '{label}' set for {minutes} minutes."

    tools = [
        Tool(
            name="set_timer",
            description="Set a countdown timer.",
            parameters={"type": "object", "properties": {"minutes": {"type": "integer"}, "label": {"type": "string"}}, "required": ["minutes"]},
            fn=set_timer,
            spoken_hint="One sec.",
        )
    ]
    player = MockPlayer(24_000)
    log = EventLog(verbose, player)
    stt = MockSTT(["Can you set a timer for five minutes?"], delay_s=0.3)
    llm = MockLLM(
        [ScriptedToolCall("set_timer", {"minutes": 5, "label": "five minutes"}), "Done. Five minutes, starting now."],
        ttft_s=0.4,
    )
    tts = MockTTS(ttfa_s=0.25, realtime_factor=0.3)
    seg = ScriptedSegmenter(script=[(0.3, 1.5)])
    settings = _settings(filler_after_ms=1500)
    agent = _mock_agent(
        stt=stt, llm=llm, tts=tts, player=player, segmenter=seg, frames=silent_frames(30), settings=settings,
        log=log, tools=tools, max_turns=1,
    )
    turns = await agent.run()
    c.ok(len(turns) == 1, f"expected 1 turn, got {len(turns)}")
    c.ok(calls == [{"minutes": 5, "label": "five minutes"}], f"tool not executed as expected: {calls}")
    tevt = log.first("turn")
    c.ok(tevt is not None and tevt[1]["hint_spoken"] == "One sec.", "spoken hint missing from turn event")
    c.ok(tts.calls[:1] == ["One sec."], f"hint was not the first thing synthesized: {tts.calls}")
    roles = [m["role"] for m in agent.messages]
    c.ok(roles == ["user", "assistant", "tool", "assistant"], f"history roles {roles}")
    c.ok(agent.messages[1].get("tool_calls", [{}])[0].get("function", {}).get("name") == "set_timer", "tool_calls missing in history")
    c.ok(agent.messages[-1]["content"] == "Done. Five minutes, starting now.", f"final text {agent.messages[-1]['content']!r}")
    if turns:
        c.ok(turns[0].assistant_text == "One sec. Done. Five minutes, starting now.", f"assistant_text {turns[0].assistant_text!r}")
        c.ok(len(llm.calls) == 2, f"expected 2 LLM calls, got {len(llm.calls)}")
        c.ok(llm.calls[1][-1]["role"] == "tool", "second LLM call did not end with the tool result")
    # the writer keeps audio strictly ordered: hint fully before the answer
    w = player.writes
    c.ok(all(w[i][1] + w[i][2] / 24_000 <= w[i + 1][1] + 1e-6 for i in range(len(w) - 1)), "audio writes overlapped")
    return c, {"turns": _turn_rows(turns), "tts_calls": tts.calls}


async def test_filler(verbose: bool) -> tuple[Check, dict[str, Any]]:
    """(d) LLM TTFT 1.5 s -> a filler fires at 0.9 s and never overlaps real audio."""
    c = Check()
    player = MockPlayer(24_000)
    log = EventLog(verbose, player)
    stt = MockSTT(["What do you think I should do about it?"], delay_s=0.3)
    llm = MockLLM(["Honestly, I think you should sleep on it first."], ttft_s=1.5)
    tts = MockTTS(ttfa_s=0.25, realtime_factor=0.3)
    seg = ScriptedSegmenter(script=[(0.3, 1.5)])
    settings = _settings(filler_after_ms=900)
    agent = _mock_agent(
        stt=stt, llm=llm, tts=tts, player=player, segmenter=seg, frames=silent_frames(30), settings=settings,
        log=log, fillers=["hmm", "let me think"], max_turns=1,
    )
    turns = await agent.run()
    c.ok(len(turns) == 1, f"expected 1 turn, got {len(turns)}")
    ready = log.first("ready")
    c.ok(ready is not None and ready[1]["fillers"] == 2, "fillers were not pre-rendered")
    fe = log.first("filler")
    c.ok(fe is not None, "filler did not fire")
    se = log.first("speech_end")
    if fe is not None and se is not None:
        after = fe[0] - se[0]
        c.ok(0.85 <= after <= 1.05, f"filler fired {after:.3f}s after speech end (expected ~0.9)")
    a = log.first("audio_start")
    c.ok(a is not None and a[1]["after_filler"] is True, "audio_start not flagged after_filler")
    w = player.writes
    c.ok(len(w) >= 2, f"expected filler + speech writes, got {len(w)}")
    if len(w) >= 2:
        filler_end = w[0][1] + w[0][2] / 24_000
        c.ok(w[1][1] >= filler_end - 1e-6, "real audio overlapped the filler")
        c.note(f"filler {w[0][2] / 24_000:.2f}s, real audio queued {w[1][0] - w[0][0]:.3f}s after filler write")
    if turns:
        lat = turns[0].response_latency()
        c.ok(lat is not None and lat >= 1.9, f"latency {lat} should reflect the 1.5 s TTFT")
    return c, {"turns": _turn_rows(turns), "filler_after_s": None if fe is None or se is None else round(fe[0] - se[0], 3)}


async def test_timer_event(verbose: bool) -> tuple[Check, dict[str, Any]]:
    """(e) a pending timer event while LISTENING triggers a spoken turn."""
    c = Check()
    player = MockPlayer(24_000)
    log = EventLog(verbose, player)
    q: asyncio.Queue = asyncio.Queue()
    stt = MockSTT([])
    llm = MockLLM(["Hey, your tea timer just went off."], ttft_s=0.3)
    tts = MockTTS()
    seg = ScriptedSegmenter(script=[])
    agent = _mock_agent(
        stt=stt, llm=llm, tts=tts, player=player, segmenter=seg, frames=silent_frames(15), settings=_settings(),
        log=log, max_turns=1, pending_events=q,
    )

    async def fire() -> None:
        await asyncio.sleep(0.5)
        q.put_nowait({"type": "timer", "label": "tea"})

    asyncio.create_task(fire())
    turns = await agent.run()
    c.ok(len(turns) == 1, f"expected 1 turn, got {len(turns)}")
    if turns:
        c.ok("timer 'tea'" in turns[0].user_text, f"user_text {turns[0].user_text!r}")
        c.ok(turns[0].assistant_text == "Hey, your tea timer just went off.", "assistant text wrong")
    c.ok(agent.messages[0]["role"] == "user" and agent.messages[0]["content"].startswith("[system:"), "system-style user message missing")
    return c, {"turns": _turn_rows(turns)}


async def test_say(verbose: bool) -> tuple[Check, dict[str, Any]]:
    """(f) typed input via say() with no mic/segmenter."""
    c = Check()
    player = MockPlayer(24_000)
    log = EventLog(verbose, player)
    llm = MockLLM(["Good to hear from you. What's up?"], ttft_s=0.3)
    agent = _mock_agent(
        stt=MockSTT([]), llm=llm, tts=MockTTS(), player=player, segmenter=None, frames=None, settings=_settings(), log=log,
    )
    m = await agent.say("hi eva")
    lat = m.response_latency()
    c.ok(lat is not None and 0.5 <= lat <= 0.8, f"say() latency {lat} (expected ttft .3 + ttfa .25)")
    c.ok(m.assistant_text == "Good to hear from you. What's up?", f"say() text {m.assistant_text!r}")
    c.ok(agent.messages[0]["content"] == "hi eva", "typed text not in history")
    return c, {"turns": _turn_rows([m])}


async def test_thinking_merge(verbose: bool) -> tuple[Check, dict[str, Any]]:
    """(g) the user resumes talking while Eva is still thinking (STT in flight):
    the first utterance is cancelled and merged with the second one."""
    c = Check()
    player = MockPlayer(24_000)
    log = EventLog(verbose, player)
    stt = MockSTT(["Hey Eva how's it going"], delay_s=1.0)
    llm = MockLLM(["Pretty good! How are you?"], ttft_s=0.4)
    tts = MockTTS()
    seg = ScriptedSegmenter(script=[(0.3, 0.8), (1.5, 1.0)])
    agent = _mock_agent(
        stt=stt, llm=llm, tts=tts, player=player, segmenter=seg, frames=silent_frames(15), settings=_settings(),
        log=log, max_turns=1,
    )
    turns = await agent.run()
    c.ok(len(turns) == 1, f"expected 1 turn, got {len(turns)}")
    bi = log.first("barge_in")
    c.ok(bi is not None and bi[1]["state_before"] == "thinking", "no thinking-phase barge-in")
    c.ok(log.first("utterance_carried") is not None, "utterance was not carried")
    c.ok(len(stt.calls) == 1, f"expected exactly 1 completed STT call, got {len(stt.calls)}")
    if stt.calls:
        n = stt.calls[0]["samples"]
        expect = int((0.8 + 0.3 + 1.0) * MIC_SAMPLE_RATE)
        c.ok(abs(n - expect) < 0.05 * MIC_SAMPLE_RATE, f"merged pcm has {n} samples, expected ~{expect}")
    c.ok(len(agent.messages) == 2, f"history should be user+assistant, has {[m['role'] for m in agent.messages]}")
    return c, {"turns": _turn_rows(turns)}


async def test_no_barge_in_queue(verbose: bool) -> tuple[Check, dict[str, Any]]:
    """(h) with barge_in=False, talking over Eva never stops her; the utterance is
    answered once she is done."""
    c = Check()
    player = MockPlayer(24_000)
    log = EventLog(verbose, player)
    reply = "Let me tell you about my day, it was long but honestly pretty good in the end."
    stt = MockSTT(["How was your day?", "Nice, glad to hear it."], delay_s=0.3)
    llm = MockLLM([reply, "Thanks for asking."], ttft_s=0.4)
    tts = MockTTS()
    seg = ScriptedSegmenter(script=[(0.3, 1.0)])
    settings = _settings(barge_in=False, filler_after_ms=0)

    def hook(name: str, data: dict[str, Any]) -> None:
        if name == "audio_start" and not log.all("utterance_queued"):
            seg.schedule(time.perf_counter() + 0.8, 1.2)

    log.hooks.append(hook)
    agent = _mock_agent(
        stt=stt, llm=llm, tts=tts, player=player, segmenter=seg, frames=silent_frames(30), settings=settings,
        log=log, max_turns=2,
    )
    turns = await agent.run()
    c.ok(len(turns) == 2, f"expected 2 turns, got {len(turns)}")
    c.ok(not log.all("barge_in"), "barge-in happened although disabled")
    c.ok(log.first("utterance_queued") is not None, "utterance was not queued")
    if len(turns) == 2:
        c.ok(not turns[0].interrupted and turns[0].assistant_text == reply, "first reply was cut")
        c.ok(turns[1].user_text == "Nice, glad to hear it.", f"queued utterance not answered: {turns[1].user_text!r}")
        c.ok(turns[1].speech_end is not None and turns[0].audio_finished is not None and turns[1].speech_end < turns[0].audio_finished, "queued utterance should have ended while Eva was still speaking")
    return c, {"turns": _turn_rows(turns)}


async def test_streaming_stt(verbose: bool) -> tuple[Check, dict[str, Any]]:
    """(i) a StreamingSTT gets every frame from onset and commit() at the endpoint;
    a blip below barge_in_min_speech_ms is discarded, not committed."""
    c = Check()
    player = MockPlayer(24_000)
    log = EventLog(verbose, player)
    stt = MockStreamingSTT(["Hey Eva, quick one.", "And another thing."], delay_s=1.0, commit_delay_s=0.1)
    llm = MockLLM(["Sure, go ahead.", "Yeah?"], ttft_s=0.3)
    tts = MockTTS(ttfa_s=0.2, realtime_factor=0.3)
    # utterance, then a 0.1 s blip while Eva speaks (ignored), then a real second utterance
    seg = ScriptedSegmenter(script=[(0.3, 1.2), (6.0, 1.0)])
    settings = _settings(filler_after_ms=0, barge_in_min_speech_ms=300)

    def hook(name: str, data: dict[str, Any]) -> None:
        if name == "audio_start" and not log.all("barge_in_ignored") and len(log.all("audio_start")) == 1:
            seg.schedule(time.perf_counter() + 0.2, 0.1)

    log.hooks.append(hook)
    agent = _mock_agent(
        stt=stt, llm=llm, tts=tts, player=player, segmenter=seg, frames=silent_frames(30), settings=settings,
        log=log, max_turns=2,
    )
    c.ok(agent.stt_streaming, "pipeline did not detect the streaming STT")
    turns = await agent.run()
    c.ok(len(turns) == 2, f"expected 2 turns, got {len(turns)}")
    c.ok(len(stt.commits) == 2, f"expected 2 commits, got {len(stt.commits)}")
    c.ok(len(stt.calls) == 0, f"transcribe() should not be used on the streaming path, called {len(stt.calls)}x")
    if stt.commits:
        n = stt.commits[0]["samples"]
        # ring (prespeech 300 + min_speech 200 + 200 slack = 700 ms) + 1.2 s of speech frames
        lo, hi = int(1.2 * MIC_SAMPLE_RATE), int((1.2 + 0.75) * MIC_SAMPLE_RATE)
        c.ok(lo <= n <= hi, f"committed {n} samples, expected between {lo} and {hi}")
    ig = log.first("barge_in_ignored")
    c.ok(ig is not None, "the 0.1 s blip was not ignored")
    c.ok(any(not d["keep_audio"] for d in stt.discards), f"blip audio was not discarded: {stt.discards}")
    ev = log.first("stt")
    c.ok(ev is not None and ev[1].get("mode") == "stream", "stt event not flagged as stream")
    if turns:
        b = turns[0].breakdown()
        c.ok(b["stt"] is not None and b["stt"] < 0.3, f"stt stage {b['stt']} should be ~0.1 s (commit), not the 1.0 s batch delay")
        c.ok(turns[0].user_text == "Hey Eva, quick one.", f"user_text {turns[0].user_text!r}")
    c.ok(agent.stream_turns == 2, f"stream_turns {agent.stream_turns}")
    return c, {"turns": _turn_rows(turns), "commits": stt.commits, "discards": stt.discards, "feeds": stt.feeds}


async def test_streaming_thinking_merge(verbose: bool) -> tuple[Check, dict[str, Any]]:
    """(j) streaming STT + the user resumes while commit() is in flight: the commit is
    cancelled, the socket dropped with keep_audio, and the merged utterance goes batch."""
    c = Check()
    player = MockPlayer(24_000)
    log = EventLog(verbose, player)
    stt = MockStreamingSTT(["Hey Eva how's it going"], delay_s=0.5, commit_delay_s=1.0)
    llm = MockLLM(["Pretty good! How are you?"], ttft_s=0.3)
    seg = ScriptedSegmenter(script=[(0.3, 0.8), (1.5, 1.0)])
    agent = _mock_agent(
        stt=stt, llm=llm, tts=MockTTS(), player=player, segmenter=seg, frames=silent_frames(15), settings=_settings(),
        log=log, max_turns=1,
    )
    turns = await agent.run()
    c.ok(len(turns) == 1, f"expected 1 turn, got {len(turns)}")
    c.ok(log.first("utterance_carried") is not None, "utterance was not carried")
    c.ok(any(d["keep_audio"] for d in stt.discards), f"cancelled commit should discard with keep_audio: {stt.discards}")
    c.ok(len(stt.calls) == 1, f"merged utterance should be transcribed in batch exactly once, got {len(stt.calls)}")
    if stt.calls:
        n = stt.calls[0]["samples"]
        expect = int((0.8 + 0.3 + 1.0) * MIC_SAMPLE_RATE)
        c.ok(abs(n - expect) < 0.05 * MIC_SAMPLE_RATE, f"merged pcm has {n} samples, expected ~{expect}")
    c.ok(len([x for x in stt.commits]) == 0, f"no commit should have completed, got {stt.commits}")
    return c, {"turns": _turn_rows(turns), "discards": stt.discards}


async def test_empty_reply(verbose: bool) -> tuple[Check, dict[str, Any]]:
    """(k) an empty LLM completion is retried, then prefilled with "Mm.", then replaced
    by a spoken fallback; never dead air."""
    c = Check()
    # first turn: empty, empty -> prefill continuation works
    player = MockPlayer(24_000)
    log = EventLog(verbose, player)
    llm = MockLLM(["", "", "Okay, here we go.", "", "", ""], ttft_s=0.1)
    agent = _mock_agent(
        stt=MockSTT([]), llm=llm, tts=MockTTS(ttfa_s=0.1), player=player, segmenter=None, frames=None,
        settings=_settings(), log=log,
    )
    m1 = await agent.say("say something")
    c.ok(len(llm.calls) == 3, f"expected 3 LLM calls (empty, retry, prefill), got {len(llm.calls)}")
    if len(llm.calls) >= 3:
        last = llm.calls[2][-1]
        c.ok(last.get("role") == "assistant" and last.get("content") == EMPTY_REPLY_PREFILL, f"third call not prefilled: {last}")
    c.ok(m1.assistant_text == f"{EMPTY_REPLY_PREFILL} Okay, here we go.", f"assistant_text {m1.assistant_text!r}")
    c.ok(len(log.all("llm_empty")) == 2, f"expected 2 llm_empty events, got {len(log.all('llm_empty'))}")
    c.ok(agent.messages[-1]["content"] == f"{EMPTY_REPLY_PREFILL} Okay, here we go.", "history missing the prefilled reply")
    # second turn: everything empty -> fallback line
    m2 = await agent.say("and now?")
    c.ok(len(llm.calls) == 6, f"expected 6 LLM calls in total, got {len(llm.calls)}")
    c.ok(m2.assistant_text == EMPTY_REPLY_TEXT, f"fallback not spoken: {m2.assistant_text!r}")
    c.ok(m2.audio_started is not None, "fallback produced no audio")
    c.ok(len(agent.messages) == 4, f"history should be 4 messages, is {[x['role'] for x in agent.messages]}")
    return c, {"turns": _turn_rows([m1, m2])}


async def test_llm_failure(verbose: bool) -> tuple[Check, dict[str, Any]]:
    """(l) the LLM request cannot start at all: a spoken fallback, an error event, and
    the next turn works normally."""
    c = Check()
    player = MockPlayer(24_000)
    log = EventLog(verbose, player)
    llm = MockLLM([ScriptedError("HTTP 503"), "Back again."], ttft_s=0.1)
    agent = _mock_agent(
        stt=MockSTT([]), llm=llm, tts=MockTTS(ttfa_s=0.1), player=player, segmenter=None, frames=None,
        settings=_settings(), log=log,
    )
    m1 = await agent.say("hello?")
    c.ok(m1.assistant_text == LLM_FAILURE_TEXT, f"fallback not spoken: {m1.assistant_text!r}")
    c.ok(m1.audio_started is not None, "fallback produced no audio")
    err = log.first("error")
    c.ok(err is not None and err[1].get("where") == "llm", "no llm error event")
    t = log.first("turn")
    c.ok(t is not None and t[1].get("error"), "turn event should carry the error")
    m2 = await agent.say("still there?")
    c.ok(m2.assistant_text == "Back again." and not m2.interrupted, f"second turn wrong: {m2.assistant_text!r}")
    return c, {"turns": _turn_rows([m1, m2])}


async def test_commit_deadline(verbose: bool) -> tuple[Check, dict[str, Any]]:
    """(m) streaming STT whose commit() is slower than STT_COMMIT_DEADLINE_S: the commit
    is cancelled and its socket discarded BEFORE exactly one batch request is made
    (never two Scribe requests in flight), and the turn still gets answered."""
    import eva.pipeline as pl

    c = Check()
    player = MockPlayer(24_000)
    log = EventLog(verbose, player)
    stt = MockStreamingSTT(["Hey Eva, quick one."], delay_s=0.3, commit_delay_s=5.0)
    llm = MockLLM(["Sure, go ahead."], ttft_s=0.3)
    seg = ScriptedSegmenter(script=[(0.3, 1.0)])
    old = pl.STT_COMMIT_DEADLINE_S
    pl.STT_COMMIT_DEADLINE_S = 0.4
    try:
        agent = _mock_agent(
            stt=stt, llm=llm, tts=MockTTS(), player=player, segmenter=seg, frames=silent_frames(10),
            settings=_settings(filler_after_ms=0), log=log, max_turns=1,
        )
        turns = await agent.run()
    finally:
        pl.STT_COMMIT_DEADLINE_S = old
    c.ok(len(turns) == 1, f"expected 1 turn, got {len(turns)}")
    c.ok(log.first("stt_commit_timeout") is not None, "no stt_commit_timeout event")
    c.ok(log.first("stt_fallback") is not None, "no stt_fallback event")
    c.ok(len(stt.commits) == 0, f"no commit should have completed, got {stt.commits}")
    c.ok(len(stt.calls) == 1, f"exactly one batch request expected, got {len(stt.calls)}")
    c.ok(len(stt.commit_cancelled) == 1, f"the late commit should have been cancelled once, got {stt.commit_cancelled}")
    if stt.commit_cancelled and stt.calls:
        c.ok(
            stt.commit_cancelled[0] <= stt.calls[0]["t"],
            f"batch request started {stt.calls[0]['t'] - stt.commit_cancelled[0]:+.3f} s relative to the commit cancel: must not overlap",
        )
    c.ok(any(not d["keep_audio"] for d in stt.discards), f"cancelled commit's socket should be discarded: {stt.discards}")
    if turns:
        c.ok(turns[0].user_text == "Hey Eva, quick one.", f"user_text {turns[0].user_text!r}")
        b = turns[0].breakdown()
        c.ok(b["stt"] is not None and 0.6 <= b["stt"] <= 1.2, f"stt stage {b['stt']} should be ~deadline 0.4 + batch 0.3 s")
    c.ok(agent.stream_turns == 0, f"stream_turns {agent.stream_turns} (the turn went batch)")
    ev = log.first("stt")
    c.ok(ev is not None and (ev[1].get("fallback") or "").startswith("batch after commit slower"), f"stt event fallback: {ev}")
    return c, {"turns": _turn_rows(turns), "calls": stt.calls, "discards": stt.discards, "commit_cancelled": stt.commit_cancelled}


MOCK_TESTS = [
    ("a_normal_two_turns", test_normal_two_turns),
    ("b_barge_in", test_barge_in),
    ("c_tool_call", test_tool_call),
    ("d_filler", test_filler),
    ("e_timer_event", test_timer_event),
    ("f_say_text_mode", test_say),
    ("g_thinking_merge", test_thinking_merge),
    ("h_no_barge_in_queue", test_no_barge_in_queue),
    ("i_streaming_stt", test_streaming_stt),
    ("j_streaming_thinking_merge", test_streaming_thinking_merge),
    ("k_empty_reply", test_empty_reply),
    ("l_llm_failure", test_llm_failure),
    ("m_commit_deadline", test_commit_deadline),
]


async def run_mock_suite(verbose: bool, only: str | None) -> int:
    results: dict[str, Any] = {}
    table = Table(title="e2e_sim --mock", show_lines=False)
    table.add_column("test")
    table.add_column("result")
    table.add_column("details", overflow="fold")
    failed = 0
    for name, fn in MOCK_TESTS:
        if only and only not in name:
            continue
        console.print(f"[bold]{name}[/]")
        t0 = time.perf_counter()
        try:
            check, data = await fn(verbose)
        except Exception as e:  # a crash is a failure, keep going
            check, data = Check(), {"exception": repr(e)}
            check.failures.append(f"exception: {e!r}")
            traceback.print_exc()
        dt = time.perf_counter() - t0
        if verbose and data.get("turns"):
            print_turn_table(name, data["turns"])
        ok = not check.failures
        failed += 0 if ok else 1
        details = "; ".join(check.failures) if check.failures else "; ".join(check.notes)
        table.add_row(name, "[green]PASS[/]" if ok else "[red]FAIL[/]", f"{escape(details)} ({dt:.1f}s)")
        results[name] = {"pass": ok, "failures": check.failures, "notes": check.notes, "seconds": round(dt, 2), **data}
    console.print(table)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "e2e_mock.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    console.print(f"wrote {out}")
    return 1 if failed else 0


# ---------------------------------------------------------------- real mode
class ScenarioMic:
    """Mic stand-in that plays wav utterances at controlled moments.

    Utterance 0 starts 0.5 s in.  With ``barge_in_at`` set, utterance 1 starts that
    many seconds after the agent's first audio.  Every other utterance starts ``gap``
    seconds after the agent finished answering the previous one (a turn or a
    cancelled/merged turn).  Silence frames fill the rest so the VAD sees a
    continuous stream; the stream ends 1.5 s after the last answer.
    """

    def __init__(self, paths: list[Path], gap: float, barge_in_at: float | None, frame_ms: int = 20) -> None:
        self.paths = paths
        self.gap = gap
        self.barge_in_at = barge_in_at
        self.frame_ms = frame_ms
        self.n = int(MIC_SAMPLE_RATE * frame_ms / 1000)
        self.last_answer_at: float | None = None
        self.awaiting = False  # an utterance was played and not answered yet
        self.ended = False  # ... and the VAD has seen its end
        self.audio_started_at: float | None = None
        self.log: list[tuple[float, str]] = []

    def on_event(self, name: str, data: dict[str, Any]) -> None:
        if name == "speech_end":
            self.ended = True
        elif name in ("turn", "turn_cancelled"):
            if self.ended:  # an answer that came after the utterance ended is its answer
                self.awaiting = False
                self.ended = False
                self.last_answer_at = time.perf_counter()
        elif name == "audio_start" and self.audio_started_at is None:
            self.audio_started_at = time.perf_counter()

    @staticmethod
    def load(path: Path) -> np.ndarray:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="int16", always_2d=True)
        pcm = data[:, 0]
        if sr != MIC_SAMPLE_RATE:  # crude decimation/interpolation; samples are 16 kHz anyway
            idx = np.round(np.arange(0, len(pcm), sr / MIC_SAMPLE_RATE)).astype(int)
            pcm = pcm[idx[idx < len(pcm)]]
        return np.ascontiguousarray(pcm)

    async def frames(self) -> AsyncIterator[np.ndarray]:
        silence = np.zeros(self.n, dtype=np.int16)
        t0 = time.perf_counter()
        i = 0
        u = 0
        playing: np.ndarray | None = None
        pos = 0
        while True:
            target = t0 + i * self.frame_ms / 1000
            delay = target - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            i += 1
            now = time.perf_counter()
            if playing is None:
                if u >= len(self.paths):
                    if not self.awaiting and self.last_answer_at is not None and now >= self.last_answer_at + 1.5:
                        self.log.append((round(now - t0, 3), "end of scenario"))
                        return
                    if now - t0 > 120:
                        self.log.append((round(now - t0, 3), "gave up waiting for the last answer"))
                        return
                else:
                    if u == 0:
                        due = now >= t0 + 0.5
                    elif self.barge_in_at is not None and u == 1:
                        due = self.audio_started_at is not None and now >= self.audio_started_at + self.barge_in_at
                    else:
                        due = not self.awaiting and self.last_answer_at is not None and now >= self.last_answer_at + self.gap
                    if due:
                        playing = self.load(self.paths[u])
                        pos = 0
                        self.awaiting = True
                        self.ended = False
                        self.log.append((round(now - t0, 3), f"utterance {u} start: {self.paths[u].name}"))
                        u += 1
            if playing is not None:
                frame = playing[pos : pos + self.n]
                pos += self.n
                if len(frame) < self.n:
                    frame = np.concatenate([frame, np.zeros(self.n - len(frame), dtype=np.int16)])
                    playing = None
                    self.log.append((round(now - t0, 3), f"utterance {u - 1} end"))
                yield frame
            else:
                yield silence


async def run_real(args: argparse.Namespace) -> int:
    from eva.factory import build_llm, build_stt, build_tts

    preset = PRESETS[args.preset]
    keys = load_keys()
    settings = dataclasses.replace(preset.settings)
    if args.no_barge_in:
        settings.barge_in = False
    paths = [Path(p) if Path(p).exists() else SAMPLES_DIR / p for p in args.utterances.split(",") if p.strip()]
    for p in paths:
        if not p.exists():
            console.print(f"[red]missing utterance {escape(str(p))}[/]")
            return 2

    console.print(f"[bold]preset[/] {preset.name}: {preset.description}")
    tts_cfg = dict(preset.tts)
    if args.tts_mode:
        tts_cfg["mode"] = args.tts_mode
    stt, llm, tts = build_stt(preset.stt, keys), build_llm(preset.llm, keys), build_tts(tts_cfg, keys)

    persona_name = args.persona or preset.persona
    fillers: list[str] = []
    tool_hints: list[str] = []
    try:
        from eva.tools import get_tools
        from eva.tools import tool_notes as _tool_notes

        tools = get_tools()
    except Exception as e:
        console.print(f"[yellow]eva.tools unavailable ({escape(repr(e))}); no tools[/]")
        tools = []

        def _tool_notes(ts: list[Tool]) -> str:
            return "\n".join(f"- {t.name}: {t.description}" for t in ts)
    memory_text = ""
    try:
        from eva.config import MEMORY_FILE
        from eva.memory import Memory

        mem = Memory(MEMORY_FILE)
        mem.load()
        memory_text = mem.as_prompt_text()
    except Exception as e:
        console.print(f"[yellow]eva.memory unavailable ({escape(repr(e))})[/]")
    try:
        from eva.personas import load_persona, render

        persona = load_persona(persona_name)
        fillers = list(persona.fillers)
        tool_hints = [h for h in (getattr(persona, "tool_hints", None) or []) if isinstance(h, str)]
        system_prompt = render(
            persona,
            supports_audio_tags=tts.supports_audio_tags,
            memory_text=memory_text,
            now=datetime.now().strftime("%A %d %B %Y, %H:%M"),
            user_name=args.user_name or "",
            tool_notes=_tool_notes(tools),
        )
    except Exception as e:
        console.print(f"[yellow]eva.personas unavailable ({escape(repr(e))}); using a basic prompt[/]")
        system_prompt = (
            "You are Eva, a warm, emotionally intelligent voice companion. Speak in short natural "
            "sentences, no lists, no markdown, no emoji. Attune to feelings first. Use the tools when asked."
        )
    if args.mute_fillers:
        fillers = []

    from eva.audio.vad import UtteranceSegmenter

    segmenter = UtteranceSegmenter(settings)
    if args.speakers:
        from eva.audio.player import Player

        player: Any = Player(tts.sample_rate, device=settings.output_device)
    else:
        player = MockPlayer(tts.sample_rate)
    player.start()

    mic = ScenarioMic(paths, gap=args.gap, barge_in_at=args.barge_in_at)
    log = EventLog(verbose=True, player=player)
    log.hooks.append(mic.on_event)

    t0 = time.perf_counter()
    await asyncio.gather(stt.warmup(), llm.warmup(), tts.warmup())
    warm_s = time.perf_counter() - t0
    console.print(f"warmup {warm_s:.2f}s")

    agent = VoiceAgent(
        stt, llm, tts, system_prompt, tools, settings,
        frames=mic.frames(), segmenter=segmenter, player=player, fillers=fillers, tool_hints=tool_hints, on_event=log,
        max_turns=args.max_turns,
    )
    t0 = time.perf_counter()
    await agent.prepare()
    console.print(f"fillers pre-rendered in {time.perf_counter() - t0:.2f}s")
    try:
        turns = await asyncio.wait_for(agent.run(), timeout=args.timeout)
    except asyncio.TimeoutError:
        console.print("[red]timed out[/]")
        turns = agent.turns
    finally:
        try:
            player.close()
        except Exception:
            pass
        await asyncio.gather(stt.close(), llm.close(), tts.close(), return_exceptions=True)

    print_turn_table(f"e2e {preset.name}", _turn_rows(turns))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else OUT_DIR / f"e2e_{preset.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "preset": preset.name,
                "stt": getattr(stt, "name", "?"),
                "llm": getattr(llm, "name", "?"),
                "tts": getattr(tts, "name", "?"),
                "settings": dataclasses.asdict(settings),
                "utterances": [str(p) for p in paths],
                "barge_in_at": args.barge_in_at,
                "warmup_s": round(warm_s, 3),
                "stream_turns": getattr(agent, "stream_turns", 0),
                "turns": _turn_rows(turns),
                "history": agent.messages,
                "events": [(round(t - log.t0, 3), n, d) for t, n, d in log.events if n != "state"],
                "mic_log": mic.log,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    console.print(f"wrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", help="run the mock test suite (no devices, no network)")
    ap.add_argument("--only", help="run only mock tests whose name contains this")
    ap.add_argument("--verbose", "-v", action="store_true", help="print every pipeline event")
    ap.add_argument("--preset", default="cloud-fast", choices=sorted(PRESETS))
    ap.add_argument("--utterances", default="samples/user_hello.wav,samples/user_rough_day.wav,samples/user_task.wav")
    ap.add_argument("--gap", type=float, default=6.0, help="seconds after a turn before the next utterance")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--silent", action="store_true", help="MockPlayer (default)")
    g.add_argument("--speakers", action="store_true", help="real Player on the default output device")
    ap.add_argument("--persona")
    ap.add_argument("--user-name")
    ap.add_argument("--barge-in-at", type=float, help="play utterance #2 this many seconds into the first reply")
    ap.add_argument("--no-barge-in", action="store_true")
    ap.add_argument("--mute-fillers", action="store_true")
    ap.add_argument("--max-turns", type=int)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--out", help="JSON output path (default bench/out/e2e_<preset>.json)")
    ap.add_argument("--tts-mode", choices=["ws", "http"], help="override the preset's ElevenLabs transport")
    ap.add_argument("--stt-commit-deadline", type=float, help="override eva.pipeline.STT_COMMIT_DEADLINE_S (seconds before a late commit is abandoned for batch)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING, format="%(name)s %(levelname)s %(message)s")
    if args.stt_commit_deadline is not None:
        import eva.pipeline as _pl

        _pl.STT_COMMIT_DEADLINE_S = args.stt_commit_deadline
    if args.mock:
        return asyncio.run(run_mock_suite(args.verbose, args.only))
    return asyncio.run(run_real(args))


if __name__ == "__main__":
    sys.exit(main())
