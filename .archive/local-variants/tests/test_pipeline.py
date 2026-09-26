"""The pipeline's behaviour on doubles: no devices, no network.

Each scenario builds a ``VoiceAgent`` on ``eva.mocks`` doubles and asserts what
matters: metrics on a normal 2-turn chat, barge-in truncation + stop latency,
tool-call hints, fillers, timer events, typed input, thinking-phase merges, no
audio after a barge-in stop, the tool round cap, history validity after a
barge-in during a tool, the speech-time barge-in filter, the adaptive commit
deadline, the LLM keep-alive and the final-tool goodbye.

Run: ``.venv/Scripts/python.exe -m pytest tests -q`` (``-k name`` for one).
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np
import pytest

from eva.interfaces import MIC_SAMPLE_RATE, Tool
from eva.mocks import (
    Check,
    EventLog,
    MockLLM,
    MockPlayer,
    MockSTT,
    MockStreamingSTT,
    MockTTS,
    ScriptedError,
    ScriptedSegmenter,
    ScriptedToolCall,
    mock_agent as _mock_agent,
    settings_with as _settings,
    silent_frames,
    turn_rows as _turn_rows,
)
from eva.pipeline import EMPTY_REPLY_PREFILL, EMPTY_REPLY_TEXT, INTERRUPTED_MARK, LLM_FAILURE_TEXT

async def scenario_normal_two_turns(verbose: bool) -> tuple[Check, dict[str, Any]]:
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


async def scenario_barge_in(verbose: bool) -> tuple[Check, dict[str, Any]]:
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


async def scenario_tool_call(verbose: bool) -> tuple[Check, dict[str, Any]]:
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


async def scenario_filler(verbose: bool) -> tuple[Check, dict[str, Any]]:
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


async def scenario_timer_event(verbose: bool) -> tuple[Check, dict[str, Any]]:
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


async def scenario_say(verbose: bool) -> tuple[Check, dict[str, Any]]:
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


async def scenario_thinking_merge(verbose: bool) -> tuple[Check, dict[str, Any]]:
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


async def scenario_no_barge_in_queue(verbose: bool) -> tuple[Check, dict[str, Any]]:
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


async def scenario_streaming_stt(verbose: bool) -> tuple[Check, dict[str, Any]]:
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


async def scenario_streaming_thinking_merge(verbose: bool) -> tuple[Check, dict[str, Any]]:
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


async def scenario_empty_reply(verbose: bool) -> tuple[Check, dict[str, Any]]:
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


async def scenario_llm_failure(verbose: bool) -> tuple[Check, dict[str, Any]]:
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


async def scenario_commit_deadline(verbose: bool) -> tuple[Check, dict[str, Any]]:
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


async def scenario_no_audio_after_stop(verbose: bool) -> tuple[Check, dict[str, Any]]:
    """(n) a writer wake-up that is already in the event loop's ready queue when the
    barge-in stops the player must not write audio after player.stop(): Eva must not
    keep talking over the user.  The race is forced deterministically: the TTS sets an
    event right before yielding a chunk, so the interrupter and the pipeline writer
    (woken by that chunk) run in the same loop iteration, interrupter first."""
    c = Check()
    player = MockPlayer(24_000)
    log = EventLog(verbose, player)
    long_reply = (
        "So here is the thing about today, and I want to tell it properly because it matters. "
        "I was thinking about what you said yesterday and honestly it stuck with me for a while, "
        "long enough that I went back over the whole conversation twice. Anyway, tell me what you think."
    )
    llm = MockLLM([long_reply, "Okay, go on."], ttft_s=0.1, token_delay_s=0.02)
    go = asyncio.Event()

    class RacingTTS(MockTTS):
        async def synthesize(self, text: str) -> AsyncIterator[bytes]:
            i = 0
            async for b in super().synthesize(text):
                i += 1
                if i == 4 and not go.is_set():
                    go.set()  # the interrupter's wake-up is queued now, the writer's right after
                yield b

    tts = RacingTTS(ttfa_s=0.1, realtime_factor=0.3, chunk_ms=100)
    agent = _mock_agent(
        stt=MockSTT([]), llm=llm, tts=tts, player=player, segmenter=None, frames=None,
        settings=_settings(filler_after_ms=0), log=log,
    )

    async def interrupter() -> None:
        await go.wait()
        await agent.interrupt("test")
        c.ok(not player.is_active, "player still active when interrupt() returned")

    itask = asyncio.create_task(interrupter())
    m1 = await agent.say("tell me a long story")
    await itask
    c.ok(go.is_set(), "the race was never armed (TTS produced fewer than 4 chunks)")
    c.ok(m1.interrupted, "turn was not interrupted")
    c.ok(len(player.stop_calls) >= 2, f"expected stop() before and after the task ended, got {len(player.stop_calls)} calls")
    if player.stop_calls:
        t_stop = player.stop_calls[0][0]
        late = [w for w in player.writes if w[0] > t_stop]
        c.ok(not late, f"{len(late)} write(s) reached the player after stop(): {[round(w[2] / 24_000, 3) for w in late]} s")
        c.note(f"{len(player.writes)} writes before stop, {len(late)} after")
    c.ok(not player.is_active, "player active after the interrupted turn")
    stored = [x for x in agent.messages if x["role"] == "assistant"]
    c.ok(len(stored) == 1 and stored[0]["content"].endswith(INTERRUPTED_MARK), f"history after interrupt: {stored}")
    m2 = await agent.say("okay")
    c.ok(m2.assistant_text == "Okay, go on." and not m2.interrupted, f"next turn wrong: {m2.assistant_text!r}")
    leaked = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and t.get_name().startswith("eva-") and t.get_name() != "eva-keepalive"]
    c.ok(not leaked, f"leaked tasks: {[t.get_name() for t in leaked]}")
    return c, {"turns": _turn_rows([m1, m2]), "writes": len(player.writes), "stop_calls": len(player.stop_calls)}


async def scenario_tool_round_cap(verbose: bool) -> tuple[Check, dict[str, Any]]:
    """(o) a model that answers every tool result with another tool call is stopped at
    MAX_TOOL_ROUNDS: the extra call is dropped, not executed, and the turn ends."""
    import eva.pipeline as pl

    c = Check()
    calls: list[dict[str, Any]] = []

    async def set_timer(minutes: int, label: str = "timer") -> str:
        calls.append({"minutes": minutes, "label": label})
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
    llm = MockLLM([ScriptedToolCall("set_timer", {"minutes": 5, "label": "tea"}, id=f"call_{i}") for i in range(8)], ttft_s=0.05)
    agent = _mock_agent(
        stt=MockSTT([]), llm=llm, tts=MockTTS(ttfa_s=0.05), player=player, segmenter=None, frames=None,
        settings=_settings(filler_after_ms=0), log=log, tools=tools,
    )
    m = await asyncio.wait_for(agent.say("set a tea timer"), timeout=20)
    c.ok(len(llm.calls) == pl.MAX_TOOL_ROUNDS + 1, f"expected {pl.MAX_TOOL_ROUNDS + 1} LLM rounds, got {len(llm.calls)}")
    c.ok(len(calls) == pl.MAX_TOOL_ROUNDS, f"tool executed {len(calls)}x, expected {pl.MAX_TOOL_ROUNDS}")
    dropped = log.first("tool_calls_dropped")
    c.ok(dropped is not None and dropped[1]["names"] == ["set_timer"], f"the extra call was not dropped: {dropped}")
    c.ok(m.assistant_text.endswith("Timer 'tea' set for 5 minutes."), f"the last tool result should be spoken: {m.assistant_text!r}")
    roles = [x["role"] for x in agent.messages]
    c.ok(roles == ["user"] + ["assistant", "tool"] * pl.MAX_TOOL_ROUNDS + ["assistant"], f"history roles {roles}")
    return c, {"turns": _turn_rows([m]), "llm_calls": len(llm.calls), "tool_calls": len(calls)}


async def scenario_barge_in_during_tool(verbose: bool) -> tuple[Check, dict[str, Any]]:
    """(p) a barge-in while a tool is executing: the assistant tool_calls message already
    in the history gets a synthetic tool result for every call id before the
    [interrupted] message, so the next LLM request is a valid chat sequence."""
    c = Check()
    finished: list[float] = []

    async def slow_tool(city: str = "here") -> str:
        await asyncio.sleep(1.5)
        finished.append(time.perf_counter())
        return f"Weather in {city}: 19 C, clear."

    tools = [
        Tool(
            name="get_weather",
            description="Weather now.",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": []},
            fn=slow_tool,
            spoken_hint="Let me check the weather.",
        )
    ]
    player = MockPlayer(24_000)
    log = EventLog(verbose, player)
    stt = MockSTT(["What is the weather like?", "Actually never mind."], delay_s=0.2)
    llm = MockLLM([ScriptedToolCall("get_weather", {"city": "Lisbon"}, id="call_w1"), "Sure, no problem."], ttft_s=0.2)
    tts = MockTTS(ttfa_s=0.1)
    seg = ScriptedSegmenter(script=[(0.3, 1.0)])
    settings = _settings(filler_after_ms=0, barge_in_min_speech_ms=300)

    def hook(name: str, data: dict[str, Any]) -> None:
        if name == "tool_call":  # the user talks over the hint, 0.2 s into the 1.5 s tool
            seg.schedule(time.perf_counter() + 0.2, 1.0)

    log.hooks.append(hook)
    agent = _mock_agent(
        stt=stt, llm=llm, tts=tts, player=player, segmenter=seg, frames=silent_frames(20), settings=settings,
        log=log, tools=tools, max_turns=2,
    )
    turns = await agent.run()
    c.ok(len(turns) == 2, f"expected 2 turns, got {len(turns)}")
    c.ok(bool(turns) and turns[0].interrupted, "first turn not interrupted")
    c.ok(not finished, "the tool ran to completion although the turn was cancelled")
    roles = [x["role"] for x in agent.messages]
    c.ok(roles == ["user", "assistant", "tool", "assistant", "user", "assistant"], f"history roles {roles}")
    # every tool_call_id is answered by the tool messages that immediately follow
    for i, msg in enumerate(agent.messages):
        for tc in msg.get("tool_calls") or []:
            following = [x for x in agent.messages[i + 1 :] if x["role"] == "tool"]
            c.ok(any(x.get("tool_call_id") == tc["id"] for x in following), f"tool call {tc['id']} has no tool result")
            nxt = agent.messages[i + 1] if i + 1 < len(agent.messages) else {}
            c.ok(nxt.get("role") == "tool", f"message after tool_calls is {nxt.get('role')!r}, not 'tool'")
    tool_msgs = [x for x in agent.messages if x["role"] == "tool"]
    c.ok(bool(tool_msgs) and "cancelled" in tool_msgs[0]["content"], f"synthetic result missing: {tool_msgs}")
    if len(llm.calls) >= 2:
        sent = [x["role"] for x in llm.calls[1]]
        c.ok(sent[1:] == ["user", "assistant", "tool", "assistant", "user"], f"second LLM request roles {sent}")
    c.ok(len(turns) == 2 and turns[1].assistant_text == "Sure, no problem.", "second turn wrong")
    return c, {"turns": _turn_rows(turns), "history_roles": roles}


async def scenario_blip_with_padding_ignored(verbose: bool) -> tuple[Check, dict[str, Any]]:
    """(q) the real segmenter's SpeechEnd carries ~0.45 s of pre-speech ring + tail around
    even a 0.1 s blip; the barge-in filter must look at the speech time, not the
    utterance length, or every cough stops her at the endpoint."""
    c = Check()
    player = MockPlayer(24_000)
    log = EventLog(verbose, player)
    reply = "Let me tell you about my day, it was long but honestly pretty good in the end, all things considered."
    stt = MockSTT(["How was your day?"], delay_s=0.2)
    llm = MockLLM([reply], ttft_s=0.2)
    seg = ScriptedSegmenter(script=[(0.3, 1.0)], pad_s=0.45)  # every utterance is padded like UtteranceSegmenter's
    settings = _settings(filler_after_ms=0, barge_in_min_speech_ms=300)

    def hook(name: str, data: dict[str, Any]) -> None:
        if name == "audio_start" and len(log.all("speech_start")) == 1:
            seg.schedule(time.perf_counter() + 0.2, 0.1)  # a 0.1 s cough while Eva speaks

    log.hooks.append(hook)
    agent = _mock_agent(
        stt=stt, llm=llm, tts=MockTTS(ttfa_s=0.1), player=player, segmenter=seg, frames=silent_frames(20), settings=settings,
        log=log, max_turns=1,
    )
    turns = await agent.run()
    c.ok(len(turns) == 1, f"expected 1 turn, got {len(turns)}")
    ends = log.all("speech_end")
    c.ok(len(ends) == 2 and abs(ends[1][1]["duration_s"] - 0.55) < 0.02, f"blip SpeechEnd should report ~0.55 s of audio: {ends}")
    ig = log.first("barge_in_ignored")
    c.ok(ig is not None and ig[1].get("speech_ms") == 100.0, f"the padded 0.1 s blip was not ignored: {ig}")
    c.ok(not log.all("barge_in"), "the cough interrupted her")
    c.ok(bool(turns) and not turns[0].interrupted and turns[0].assistant_text == reply, "reply was cut")
    return c, {"turns": _turn_rows(turns)}


async def scenario_adaptive_commit_deadline(verbose: bool) -> tuple[Check, dict[str, Any]]:
    """(r) when the batch endpoint is known to be slow (the STT's warmup probe / last batch
    request) a late commit is NOT cancelled at the base deadline: the deadline grows to
    STT_COMMIT_DEADLINE_FACTOR x that latency, so the turn waits for the commit instead
    of paying the deadline plus a slow batch request."""
    import eva.pipeline as pl

    c = Check()
    player = MockPlayer(24_000)
    log = EventLog(verbose, player)
    stt = MockStreamingSTT(["Hey Eva, quick one."], delay_s=1.5, commit_delay_s=1.0)
    stt.last_batch_s = 0.9  # what a slow warmup probe would have measured
    llm = MockLLM(["Sure, go ahead."], ttft_s=0.3)
    seg = ScriptedSegmenter(script=[(0.3, 1.0)])
    old = pl.STT_COMMIT_DEADLINE_S
    pl.STT_COMMIT_DEADLINE_S = 0.4  # the base deadline alone would cancel this 1.0 s commit
    try:
        agent = _mock_agent(
            stt=stt, llm=llm, tts=MockTTS(), player=player, segmenter=seg, frames=silent_frames(10),
            settings=_settings(filler_after_ms=0), log=log, max_turns=1,
        )
        c.ok(abs(agent._commit_deadline() - 0.9 * pl.STT_COMMIT_DEADLINE_FACTOR) < 1e-9, f"deadline {agent._commit_deadline()} != factor x last batch")
        turns = await agent.run()
    finally:
        pl.STT_COMMIT_DEADLINE_S = old
    c.ok(len(turns) == 1, f"expected 1 turn, got {len(turns)}")
    c.ok(log.first("stt_commit_timeout") is None, "the commit was cancelled although batch is known to be slow")
    c.ok(len(stt.commits) == 1 and len(stt.calls) == 0, f"commits {len(stt.commits)}, batch calls {len(stt.calls)} (expected 1 / 0)")
    if turns:
        b = turns[0].breakdown()
        c.ok(b["stt"] is not None and 0.95 <= b["stt"] <= 1.3, f"stt stage {b['stt']} should be the 1.0 s commit, not deadline + 1.5 s batch")
    c.ok(agent.stream_turns == 1, f"stream_turns {agent.stream_turns}")
    c.ok(agent._stt_recent_s.get("commit") is not None and agent._commit_deadline() >= 2 * 0.9, "recent commit latency not tracked")
    return c, {"turns": _turn_rows(turns), "commits": stt.commits}


async def scenario_llm_keepalive(verbose: bool) -> tuple[Check, dict[str, Any]]:
    """(s) an LLM with ping() is pinged after LLM_KEEPALIVE_S of idleness, never while a
    response is in flight, and the keep-alive task is closed with the agent."""
    import eva.pipeline as pl

    c = Check()
    player = MockPlayer(24_000)
    log = EventLog(verbose, player)
    llm = MockLLM(["Hi there."], ttft_s=0.2)
    llm.pingable = True
    seg = ScriptedSegmenter(script=[(0.6, 0.5)])
    old = pl.LLM_KEEPALIVE_S
    pl.LLM_KEEPALIVE_S = 0.3
    try:
        agent = _mock_agent(
            stt=MockSTT(["hi"], delay_s=0.2), llm=llm, tts=MockTTS(ttfa_s=0.1), player=player, segmenter=seg,
            # 4 s, not 3: the turn's audio ends at ~2.9 s and no ping may start before that, so
            # 3 s left a 0.1 s window for the idle ping that a loaded machine missed (red since d982caa)
            frames=silent_frames(4.0), settings=_settings(filler_after_ms=0), log=log,
        )
        turns = await agent.run()
    finally:
        pl.LLM_KEEPALIVE_S = old
    c.ok(len(turns) == 1, f"expected 1 turn, got {len(turns)}")
    c.ok(len(llm.pings) >= 2, f"expected pings while idle, got {len(llm.pings)}")
    n_ev = len(log.all("llm_ping"))
    c.ok(len(llm.pings) - 1 <= n_ev <= len(llm.pings), f"{n_ev} llm_ping events for {len(llm.pings)} pings (at most one may be cut off by the shutdown)")
    if turns:
        t = turns[0]
        busy = [p for p in llm.pings if t.speech_end is not None and t.audio_finished is not None and t.speech_end <= p <= t.audio_finished]
        c.ok(not busy, f"{len(busy)} ping(s) while a response was in flight")
    c.ok(agent._keepalive_task is None, "keep-alive task not closed by run()")
    return c, {"turns": _turn_rows(turns), "pings": len(llm.pings)}


async def scenario_final_tool_goodbye(verbose: bool) -> tuple[Check, dict[str, Any]]:
    """(t) end_conversation is a ``final`` tool: when the goodbye came with the call
    there is no LLM round after the result (measured live: "Bye." then "See you around."
    twice), no "one sec" hint covers it, the tool result is never read out, and the
    end_session event stops the loop.  A call *without* a goodbye still gets one round."""
    c = Check()
    q: asyncio.Queue = asyncio.Queue()

    def end_conversation(reason: str = "") -> str:
        q.put_nowait({"type": "end_session", "reason": reason})
        return "OK: the session ends right after this reply. Say one short, warm goodbye now, nothing else."

    tools = [
        Tool(
            name="end_conversation", description="End the conversation.",
            parameters={"type": "object", "properties": {"reason": {"type": "string"}}, "required": []},
            fn=end_conversation, spoken_hint=None, final=True,
        )
    ]
    # 1. goodbye spoken with the call -> exactly one LLM request, one goodbye, session ends
    player = MockPlayer(24_000)
    log = EventLog(verbose, player)
    stt = MockSTT(["Okay I have to go, bye."], delay_s=0.2)
    llm = MockLLM(
        [ScriptedToolCall("end_conversation", {"reason": "bye"}, preface="Alright, take care. Bye.", id="call_e1"), "See you around."],
        ttft_s=0.2,
    )
    tts = MockTTS(ttfa_s=0.1)
    seg = ScriptedSegmenter(script=[(0.3, 1.0)])
    agent = _mock_agent(
        stt=stt, llm=llm, tts=tts, player=player, segmenter=seg, frames=silent_frames(20),
        settings=_settings(filler_after_ms=0), log=log, tools=tools, fillers=["one sec"], pending_events=q,
    )
    turns = await agent.run()
    c.ok(len(turns) == 1, f"expected 1 turn, got {len(turns)}")
    c.ok(len(llm.calls) == 1, f"expected 1 LLM request (no round after a final tool), got {len(llm.calls)}")
    c.ok(bool(turns) and turns[0].assistant_text == "Alright, take care. Bye.", f"assistant_text {turns[0].assistant_text if turns else None!r}")
    spoken = [t for t in tts.calls if t != "one sec"]  # "one sec" is the filler pre-render at prepare()
    c.ok(" ".join(spoken) == "Alright, take care. Bye.", f"synthesized {tts.calls}")
    roles = [m["role"] for m in agent.messages]
    c.ok(roles == ["user", "assistant", "tool"], f"history roles {roles}")
    c.ok(agent.end_requested, "end_session event did not stop the loop")
    c.ok(log.first("tool_round_final") is not None, "tool_round_final event missing")

    # 2. the model called the tool silently -> one round follows, no hint in front of it
    q2: asyncio.Queue = asyncio.Queue()
    tools[0].fn = lambda reason="": (q2.put_nowait({"type": "end_session"}), "OK: say one short goodbye now.")[1]
    player2 = MockPlayer(24_000)
    log2 = EventLog(verbose, player2)
    llm2 = MockLLM([ScriptedToolCall("end_conversation", {}, id="call_e2"), "Bye, take care."], ttft_s=0.2)
    tts2 = MockTTS(ttfa_s=0.1)
    agent2 = _mock_agent(
        stt=MockSTT(["Gotta go."], delay_s=0.2), llm=llm2, tts=tts2, player=player2, segmenter=ScriptedSegmenter(script=[(0.3, 1.0)]),
        frames=silent_frames(20), settings=_settings(filler_after_ms=0), log=log2, tools=tools, fillers=["one sec"], pending_events=q2,
    )
    turns2 = await agent2.run()
    c.ok(len(llm2.calls) == 2, f"silent call: expected 2 LLM requests, got {len(llm2.calls)}")
    spoken2 = [t for t in tts2.calls if t != "one sec"]
    c.ok(spoken2 == ["Bye, take care."], f"silent call: synthesized {tts2.calls} (a hint must not precede a goodbye)")
    c.ok(bool(turns2) and turns2[0].assistant_text == "Bye, take care.", f"silent call: assistant_text {turns2[0].assistant_text if turns2 else None!r}")
    c.ok(agent2.end_requested, "silent call: end_session event did not stop the loop")
    return c, {"turns": _turn_rows(turns) + _turn_rows(turns2), "llm_calls": [len(llm.calls), len(llm2.calls)]}


REPLY = "So here's the thing about long days, they don't really end, they just sort of fade out until you notice you're on the couch."


async def _speaking_barge_in(
    verbose: bool, partial: str | None, late_text: str, *, storm: bool = False, onset_s: float = 1.0
) -> tuple[Check, Any, Any, Any]:
    """Eva says REPLY; 0.3 s into it an ``onset_s`` VAD onset happens. ``partial`` is what the streaming STT
    reports 0.15 s after the onset (None = no partial at all); ``late_text`` is the final transcript."""
    c = Check()
    player = MockPlayer(24_000)
    log = EventLog(verbose, player)
    stt = MockStreamingSTT(["How was your day?", late_text], delay_s=0.2, commit_delay_s=0.1)
    llm = MockLLM([REPLY, "Sure, what's up?"], ttft_s=0.2)
    tts = MockTTS(ttfa_s=0.1, realtime_factor=0.3)
    seg = ScriptedSegmenter(script=[(0.3, 1.0)])
    settings = _settings(filler_after_ms=0, barge_in_min_speech_ms=300)

    def hook(name: str, data: dict[str, Any]) -> None:
        if name == "audio_start" and len(log.all("audio_start")) == 1:
            seg.schedule(time.perf_counter() + 0.3, onset_s)
        if name == "barge_in_candidate" and partial is not None:
            asyncio.get_running_loop().call_later(0.15, stt.emit_partial, partial)

    log.hooks.append(hook)
    agent = _mock_agent(
        stt=stt, llm=llm, tts=tts, player=player, segmenter=seg, frames=silent_frames(9), settings=settings,
        log=log, max_turns=2,
    )
    if storm:
        agent._storm_until = time.perf_counter() + 30
    c.ok(agent.stt_streaming, "streaming STT not detected")
    turns = await asyncio.wait_for(agent.run(), timeout=25)
    return c, log, turns, agent


async def scenario_echo_partial_does_not_interrupt(verbose: bool) -> tuple[Check, dict[str, Any]]:
    """(u) while she is audible, a VAD onset whose partial transcript is a fuzzy copy of what she is
    saying (her own voice through the speakers) must not cut her off, and is never answered."""
    c, log, turns, agent = await _speaking_barge_in(verbose, "the thing about long days they don't", "the thing about long days")
    c.ok(len(turns) == 1, f"expected 1 turn, got {len(turns)}")
    c.ok(bool(turns) and not turns[0].interrupted, "she was interrupted by her own echo")
    c.ok(bool(turns) and turns[0].assistant_text == REPLY, "reply not spoken in full")
    ig = log.first("barge_in_ignored")
    c.ok(ig is not None and ig[1].get("reason") == "echo", f"echo onset not ignored: {ig}")
    c.ok(len(log.all("stt")) == 1, "the echo was transcribed as a user turn")
    return c, {"turns": _turn_rows(turns)}


async def scenario_real_words_interrupt(verbose: bool) -> tuple[Check, dict[str, Any]]:
    """(v) the same onset with real words in the partial interrupts her at once and is answered."""
    c, log, turns, agent = await _speaking_barge_in(verbose, "wait wait stop that", "Wait, stop, I have a question.")
    c.ok(len(turns) == 2, f"expected 2 turns, got {len(turns)}")
    c.ok(bool(turns) and turns[0].interrupted, "she was not interrupted by real words")
    bi = log.first("barge_in")
    c.ok(bi is not None and bi[1]["reason"] == "barge-in", f"no immediate barge-in: {bi}")
    c.ok(len(turns) == 2 and turns[1].user_text == "Wait, stop, I have a question.", "the interjection was not answered")
    return c, {"turns": _turn_rows(turns)}


async def scenario_late_check_on_final_text(verbose: bool) -> tuple[Check, dict[str, Any]]:
    """(w) no partial ever arrives (a short interjection the STT was slow on): the onset ends
    unconfirmed, the final transcript is fetched, real words interrupt her late and are answered;
    an echo final transcript is dropped."""
    c, log, turns, agent = await _speaking_barge_in(verbose, None, "Wait, I have a question.")
    c.ok(len(turns) == 2, f"expected 2 turns, got {len(turns)}")
    bi = log.first("barge_in")
    c.ok(bi is not None and bi[1]["reason"] == "barge-in (late)", f"expected a late barge-in: {bi}")
    c.ok(len(turns) == 2 and turns[1].user_text == "Wait, I have a question.", "late interjection not answered")
    c2, log2, turns2, _ = await _speaking_barge_in(verbose, None, "they just sort of fade out")
    c.ok(len(turns2) == 1 and not turns2[0].interrupted, "an echo final transcript interrupted her")
    ig = log2.first("barge_in_ignored")
    c.ok(ig is not None and "echo" in str(ig[1].get("reason")), f"echo final text not ignored: {ig}")
    return c, {"turns": _turn_rows(turns) + _turn_rows(turns2)}


async def scenario_storm_needs_final_text(verbose: bool) -> tuple[Check, dict[str, Any]]:
    """(x) in an echo storm a partial with words is not enough: only the final transcript interrupts."""
    c, log, turns, agent = await _speaking_barge_in(verbose, "wait wait stop that", "Wait, stop, I have a question.", storm=True)
    bi = log.first("barge_in")
    c.ok(bi is not None and bi[1]["reason"] == "barge-in (late)", f"storm: expected only the late path, got {bi}")
    c.ok(len(turns) == 2 and turns[1].user_text == "Wait, stop, I have a question.", "storm: interjection lost")
    return c, {"turns": _turn_rows(turns)}


async def scenario_echo_transcript_after_she_stopped(verbose: bool) -> tuple[Check, dict[str, Any]]:
    """(zb) live log 2026-09-25, laptop speakers: her echo ran 1.5 s with no partial, counted as a
    person and cut her off; its transcript ("You'll probably smell bad when you stop asking" for
    "You'll probably feel it more when you") was then answered as the user. The self-echo gate
    only ran its fuzzy rule while she was still audible, and by transcription time she never is.
    An utterance that began while she was audible and reads as a garbled copy of her words is dropped."""
    c, log, turns, agent = await _speaking_barge_in(
        verbose, None, "So here is the thing about long days.", onset_s=2.0
    )
    c.ok(bool(turns) and turns[0].interrupted, "setup: the sustained onset should have cut her off")
    c.ok(len(turns) == 1, f"her own echo was answered as a user turn: {[t.user_text for t in turns]}")
    c.ok(log.first("stt_echo") is not None, "no stt_echo line for the dropped transcript")
    return c, {"turns": _turn_rows(turns)}


async def scenario_continuous_audio_inside_a_chunk(verbose: bool) -> tuple[Check, dict[str, Any]]:
    """(y) the writer releases audio piece by piece (each ~80 ms). Inside one chunk the pieces must
    join without any fade: a fade applied at every release was a 12 Hz tremolo ("a bad connection")."""
    c = Check()
    player = MockPlayer(24_000)
    log = EventLog(verbose, player)
    stt = MockSTT(["Tell me a long one."], delay_s=0.2)
    text = "Well, here is a long steady sentence that keeps going for a while so the tone runs on and on without a break."
    llm = MockLLM([text], ttft_s=0.2)
    tts = MockTTS(ttfa_s=0.1, realtime_factor=0.3)  # a constant-amplitude tone with 50 ms fades only at the clip's own ends
    seg = ScriptedSegmenter(script=[(0.3, 1.0)])
    settings = _settings(filler_after_ms=0, first_chunk_min_chars=200, min_chunk_chars=200)  # one chunk
    agent = _mock_agent(
        stt=stt, llm=llm, tts=tts, player=player, segmenter=seg, frames=silent_frames(12), settings=settings,
        log=log, max_turns=1,
    )
    turns = await agent.run()
    c.ok(len(turns) == 1, f"expected 1 turn, got {len(turns)}")
    audio = np.frombuffer(b"".join(player.pcm), dtype=np.int16).astype(np.float32)
    sr = 24_000
    lead, tail = int(sr * (settings.lead_in_ms + settings.fade_in_ms + 80) / 1000), int(sr * (settings.tail_ms + settings.fade_out_ms + 80) / 1000)
    body = audio[lead : audio.size - tail]
    hop = sr // 50  # 20 ms
    rms = np.sqrt((body[: body.size // hop * hop].reshape(-1, hop) ** 2).mean(axis=1))
    ref = float(np.median(rms))
    dips = int((rms < 0.5 * ref).sum())
    c.ok(ref > 100, f"tone too quiet to judge ({ref:.0f})")
    c.ok(dips == 0, f"{dips} of {rms.size} 20 ms hops inside the chunk dipped below half the median level (tremolo)")
    c.note(f"{rms.size} hops, median {ref:.0f}, min {rms.min():.0f}")
    return c, {"turns": _turn_rows(turns)}


async def scenario_foreign_language_label_keeps_language(verbose: bool) -> tuple[Check, dict[str, Any]]:
    """(z) live log 2026-09-23: Scribe heard Russian / English as Dutch ("Nee, het is niet.")
    and she switched to English fillers and answered in Dutch. A transcript the STT labels
    outside the session's languages must not switch her language; it is still answered
    (the persona rule treats it as a mishearing), and an in-box label still switches."""
    c = Check()
    player = MockPlayer(24_000)
    log = EventLog(verbose, player)
    stt = MockSTT(
        ["Привет, как дела?", "Nee, het is niet.", "Okay, English now."], delay_s=0.1, heard_as=["ru", "nl", "en"]
    )
    llm = MockLLM(["Привет!", "Не расслышала, повтори?", "Sure."], ttft_s=0.1)
    tts = MockTTS(ttfa_s=0.1, realtime_factor=0.3)
    seg = ScriptedSegmenter(script=[(0.3, 1.0), (4.0, 1.0), (7.5, 1.0)])
    agent = _mock_agent(
        stt=stt, llm=llm, tts=tts, player=player, segmenter=seg, frames=silent_frames(30),
        settings=_settings(filler_after_ms=0), log=log, max_turns=3, languages=["en", "ru"],
    )
    seen: list[str] = []
    log.hooks.append(lambda name, data: seen.append(f"{name}:{data.get('lang')}") if name in ("language", "stt_foreign") else None)
    turns = await agent.run()
    c.ok(len(turns) == 3, f"expected 3 turns (the misheard one is still answered), got {len(turns)}")
    c.ok(seen == ["language:ru", "stt_foreign:nl", "language:en"], f"language events {seen}")
    foreign = log.first("stt_foreign")
    c.ok(foreign is not None and foreign[1].get("kept") == "ru", f"stt_foreign should keep ru: {foreign}")
    c.ok(agent._user_lang == "en", f"final language {agent._user_lang}")
    return c, {"turns": _turn_rows(turns), "events": seen}

async def scenario_english_session_never_switches(verbose: bool) -> tuple[Check, dict[str, Any]]:
    """(za) English first: in a one-language session a Cyrillic transcript (Scribe can
    still write Russian speech in Cyrillic under an English hint) is answered, but never
    switches her language, fillers or hints to Russian."""
    c = Check()
    player = MockPlayer(24_000)
    log = EventLog(verbose, player)
    stt = MockSTT(["Привет, как дела?", "Okay, how are you?"], delay_s=0.1, heard_as=["en", "en"])
    llm = MockLLM(["Sorry, I only caught part of that. English?", "Good, you?"], ttft_s=0.1)
    tts = MockTTS(ttfa_s=0.1, realtime_factor=0.3)
    seg = ScriptedSegmenter(script=[(0.3, 1.0), (4.0, 1.0)])
    agent = _mock_agent(
        stt=stt, llm=llm, tts=tts, player=player, segmenter=seg, frames=silent_frames(30),
        settings=_settings(filler_after_ms=0), log=log, max_turns=2, languages=["en"],
    )
    turns = await agent.run()
    c.ok(len(turns) == 2, f"expected 2 turns, got {len(turns)}")
    c.ok(not log.all("language"), f"an English-only session switched language: {log.all('language')}")
    c.ok(agent._user_lang == "en", f"language {agent._user_lang}")
    return c, {"turns": _turn_rows(turns)}

async def scenario_tool_gate(verbose: bool) -> tuple[Check, dict[str, Any]]:
    """(zc) a small brain is offered only the tools the user's line points at: in the first
    end-to-end run MiniCPM5 1B checked the clock after "how's it going?" and hung up in the
    middle of "today was rough" (2026-09-25)."""
    from eva.toolgate import gate_tools

    c = Check()

    def tool(name: str) -> Tool:
        return Tool(name=name, description=name, parameters={"type": "object", "properties": {}, "required": []},
                    fn=lambda **_: "ok", spoken_hint=None, final=name == "end_conversation")

    tools = [tool("end_conversation"), tool("get_current_time"), tool("set_timer")]
    player = MockPlayer(24_000)
    log = EventLog(verbose, player)
    llm = MockLLM(["Hey. Good to hear you.", "Rough one. I'm sorry.", "It's half past nine."], ttft_s=0.05)
    agent = _mock_agent(stt=MockSTT([]), llm=llm, tts=MockTTS(), player=player, segmenter=None, frames=None,
                        settings=_settings(filler_after_ms=0), log=log, tools=tools, tool_filter=gate_tools)
    await agent.say("Hey Eva, how's it going? I just got home from work.")
    await agent.say("Honestly, today was rough. I don't know what to do.")
    await agent.say("What time is it?")
    c.ok(llm.tools_offered[0] is None, f"small talk was offered {llm.tools_offered[0]}")
    c.ok(llm.tools_offered[1] is None, f"a rough day was offered {llm.tools_offered[1]}")
    c.ok(llm.tools_offered[2] == ["get_current_time"], f"'what time is it' was offered {llm.tools_offered[2]}")
    c.ok(len(log.all("tools_gated")) == 3, "every narrowed turn prints why")
    # the same through the microphone: a batch-transcribed line keeps its text on the metrics
    # (the first live run offered nothing to "set a timer ... remind me" this way)
    player2 = MockPlayer(24_000)
    log2 = EventLog(verbose, player2)
    llm2 = MockLLM(["On it."], ttft_s=0.05)
    agent2 = _mock_agent(stt=MockSTT(["Can you set a timer for five minutes?"], delay_s=0.1), llm=llm2, tts=MockTTS(),
                         player=player2, segmenter=ScriptedSegmenter(script=[(0.3, 0.6)]), frames=silent_frames(3.0),
                         settings=_settings(filler_after_ms=0), log=log2, tools=tools, tool_filter=gate_tools)
    await agent2.run()
    c.ok(bool(llm2.tools_offered) and llm2.tools_offered[0] == ["set_timer"], f"spoken timer request was offered {llm2.tools_offered}")
    return c, {"offered": llm.tools_offered + llm2.tools_offered}


async def scenario_tool_router(verbose: bool) -> tuple[Check, dict[str, Any]]:
    """(zd) a plain weather question is answered from a real result: the loop calls the tool
    before the brain speaks, and doesn't offer it again for that line (replays 2026-09-26)."""
    from eva.toolgate import ToolGate

    c = Check()
    ran: list[tuple[str, dict[str, Any]]] = []

    def weather(city: str = "") -> str:
        ran.append(("get_weather", {"city": city}))
        return f"In {city}: drizzle, 59 degrees."

    tools = [Tool(name="get_weather", description="weather", parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
                  fn=weather, spoken_hint=None)]
    gate = ToolGate(home_city="Philadelphia")
    player = MockPlayer(24_000)
    log = EventLog(verbose, player)
    llm = MockLLM(["Drizzle and fifty-nine. Take a jacket."], ttft_s=0.05)
    agent = _mock_agent(stt=MockSTT([]), llm=llm, tts=MockTTS(), player=player, segmenter=None, frames=None,
                        settings=_settings(filler_after_ms=0), log=log, tools=tools, tool_filter=gate.filter)
    agent.tool_router = gate.route
    agent.tool_executor = lambda tc, ts: _run_tool(tc, ts)
    m = await agent.say("Can you check the weather for me please?")
    c.ok(len(log.all("tools_routed")) == 1, "the weather question was not routed")
    c.ok(llm.tools_offered[0] is None, f"get_weather was offered again after routing: {llm.tools_offered[0]}")
    roles = [msg["role"] for msg in llm.calls[0]]
    c.ok(roles[-2:] == ["assistant", "tool"] and "Philadelphia" in llm.calls[0][-1]["content"],
         f"the brain did not see the result before speaking: {roles[-3:]}")
    c.ok(m.assistant_text == "Drizzle and fifty-nine. Take a jacket.", f"reply {m.assistant_text!r}")
    return c, {"history": [msg["role"] for msg in agent.messages]}


async def _run_tool(tc: Any, tools: list[Tool]) -> str:
    tool = next(t for t in tools if t.name == tc.name)
    return str(tool.fn(**tc.arguments))


SCENARIOS = [
    ("a_normal_two_turns", scenario_normal_two_turns),
    ("b_barge_in", scenario_barge_in),
    ("c_tool_call", scenario_tool_call),
    ("d_filler", scenario_filler),
    ("e_timer_event", scenario_timer_event),
    ("f_say_text_mode", scenario_say),
    ("g_thinking_merge", scenario_thinking_merge),
    ("h_no_barge_in_queue", scenario_no_barge_in_queue),
    ("i_streaming_stt", scenario_streaming_stt),
    ("j_streaming_thinking_merge", scenario_streaming_thinking_merge),
    ("k_empty_reply", scenario_empty_reply),
    ("l_llm_failure", scenario_llm_failure),
    ("m_commit_deadline", scenario_commit_deadline),
    ("n_no_audio_after_stop", scenario_no_audio_after_stop),
    ("o_tool_round_cap", scenario_tool_round_cap),
    ("p_barge_in_during_tool", scenario_barge_in_during_tool),
    ("q_blip_with_padding_ignored", scenario_blip_with_padding_ignored),
    ("r_adaptive_commit_deadline", scenario_adaptive_commit_deadline),
    ("s_llm_keepalive", scenario_llm_keepalive),
    ("t_final_tool_goodbye", scenario_final_tool_goodbye),
    ("u_echo_partial_does_not_interrupt", scenario_echo_partial_does_not_interrupt),
    ("v_real_words_interrupt", scenario_real_words_interrupt),
    ("w_late_check_on_final_text", scenario_late_check_on_final_text),
    ("x_storm_needs_final_text", scenario_storm_needs_final_text),
    ("y_continuous_audio_inside_a_chunk", scenario_continuous_audio_inside_a_chunk),
    ("z_foreign_language_label_keeps_language", scenario_foreign_language_label_keeps_language),
    ("za_english_session_never_switches", scenario_english_session_never_switches),
    ("zb_echo_transcript_after_she_stopped", scenario_echo_transcript_after_she_stopped),
    ("zc_tool_gate", scenario_tool_gate),
    ("zd_tool_router", scenario_tool_router),
]


@pytest.mark.parametrize("name,scenario", SCENARIOS, ids=[n for n, _ in SCENARIOS])
def test_scenario(name: str, scenario: Any) -> None:
    check, data = asyncio.run(scenario(False))
    assert not check.failures, "; ".join(check.failures)
