#!/usr/bin/env python
"""Replay real conversations through the real loop and count how the brain fails.

    .venv/Scripts/python.exe bench/replay.py                          # minicpm1b, 8 sessions
    .venv/Scripts/python.exe bench/replay.py --brain qwen2b --sessions 5

The user's lines (from the owner's live sessions) go through ``VoiceAgent`` exactly as typed
turns do: the session's persona, memory (a copy, never saved), tool gate, sanitizer and
chunker, and the real brain in Ollama. The voice and the player are silent stand-ins that
record what would have been *spoken*; tools answer with fixed stand-in results, so nothing
touches the network. Each session starts with the greeting event, like run.py.

Counted per reply, on the spoken text:
  identity   she calls herself the user ("I'm Doston")
  repeat     a sentence of 4+ words she already said earlier in the session
  markup     angle-bracket markup or a tool's name spoken out loud
  tool talk  narrating a tool ("call the get_weather function")
  promise    announcing an action with no tool call behind it ("I'll look that up")
  wrong city a weather call for another city than the one in her memory
Writes bench/out/replay_<brain>.json with every reply.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import re
import shutil
import sys
import tempfile
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, AsyncIterator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eva.config import BRAINS, make_preset  # noqa: E402
from eva.mocks import EventLog, MockPlayer, MockSTT, MockTTS  # noqa: E402
from eva.pipeline import VoiceAgent  # noqa: E402
from eva.session import build_session  # noqa: E402

sys.path.insert(0, str(ROOT / "bench"))
from common import PROMISE_RE  # noqa: E402

OUT = ROOT / "bench" / "out"

# The owner's lines from the 2026-09-26 sessions (minicpm1b and qwen27b, both on Chatterbox).
LINES = [
    "How is everything so far?",
    "Yeah, can you check the weather for me please?",
    "What do you think? Should I go outside?",
    "What's the difference between the tree and the house?",
    "I don't know. Um sometimes I think about the meaning of life.",
    "You're right. What can you do for me?",
]
HOME_CITY = "phil"  # memory: "Lives in Philadelphia, Pennsylvania."

STAND_IN = {
    "get_weather": "In Philadelphia, it's 59 degrees Fahrenheit and light drizzle. Today's high is 68, and there's an 81 percent chance of rain.",
    "get_current_time": "It's Saturday, 9:12 am.",
    "set_timer": "OK: timer set.",
    "remember_note": "OK: saved.",
    "recall_notes": "No saved notes yet.",
    "open_url": "OK: opened.",
    "end_conversation": "OK: the session ends right after this reply. Say one short, warm goodbye now, nothing else.",
}
TOOL_NAMES = tuple(STAND_IN)


class RecordingTTS(MockTTS):
    """Silent, instant voice that keeps every chunk it was asked to say."""

    supports_cues = True

    def __init__(self) -> None:
        super().__init__(ttfa_s=0.0, realtime_factor=0.0)
        self.spoken: list[str] = []

    async def synthesize(self, text: str, cue: str | None = None) -> AsyncIterator[bytes]:  # type: ignore[override]
        self.spoken.append(text)
        async for pcm in super().synthesize(text):
            yield pcm


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 4]


def check(reply: str, earlier: list[str], user_name: str, calls: list[dict[str, Any]]) -> list[str]:
    low = reply.lower()
    flags = []
    if re.search(rf"\b(i'?m|i am|my name is|this is)\s+{re.escape(user_name.lower())}\b", low):
        flags.append("identity")
    if any(SequenceMatcher(None, s.lower(), e.lower()).ratio() >= 0.85 for s in sentences(reply) for e in earlier):
        flags.append("repeat")
    if re.search(r"<[^>]{1,60}>", reply) or any(n in low for n in TOOL_NAMES):
        flags.append("markup")
    if re.search(r"\b(call|calling|use|using) (the )?\w+ (function|tool)\b", low):
        flags.append("tool talk")
    if not calls and PROMISE_RE.search(low.replace("’", "'")):
        flags.append("promise")
    for c in calls:
        if c["name"] == "get_weather" and HOME_CITY not in json.dumps(c.get("arguments", {})).lower():
            flags.append("wrong city")
    return flags


async def one_session(brain: str, voice: str, memory: Path, user_name: str) -> list[dict[str, Any]]:
    s = build_session(make_preset(brain, voice), user_name=user_name, memory_path=memory)
    tts = RecordingTTS()
    calls: list[dict[str, Any]] = []

    async def executor(tc: Any, tools: Any) -> str:
        calls.append({"name": tc.name, "arguments": tc.arguments})
        return STAND_IN.get(tc.name, "OK.")

    settings = dataclasses.replace(s.preset.settings, filler_after_ms=0)
    log = EventLog(False)
    agent = VoiceAgent(MockSTT([]), s.llm, tts, s.system_prompt, s.tools, settings, frames=None, segmenter=None,
                       player=MockPlayer(tts.sample_rate), on_event=log, tool_executor=executor,
                       languages=s.plan.codes, **s.agent_kwargs())
    replies: list[dict[str, Any]] = []
    earlier: list[str] = []
    turns = [None, *LINES]
    for line in turns:
        tts.spoken.clear()
        n_drop, n_retry = len(log.all("sentence_dropped")), len(log.all("reply_retry"))
        calls.clear()
        if line is None:
            agent.pending_events.put_nowait(s.greeting_event(user_name))
            await agent.poll_pending_events()
            await asyncio.sleep(0)
            while agent._response is not None:  # the greeting runs as a background turn
                await asyncio.sleep(0.05)
        else:
            await agent.say(line)
        spoken = " ".join(tts.spoken).strip()
        flags = check(spoken, earlier, user_name, calls)
        drops = [d for _, d in log.all("sentence_dropped")[n_drop:]] if log.all("sentence_dropped") else []
        replies.append({"user": line or "(greeting)", "spoken": spoken, "calls": list(calls), "flags": flags,
                        "dropped": drops, "retried": len(log.all("reply_retry")) > n_retry})
        earlier += sentences(spoken)
    await agent.close()
    await s.llm.close()
    return replies


async def amain(args: argparse.Namespace) -> int:
    tmp = Path(tempfile.mkdtemp())
    memory = tmp / "memory.json"
    shutil.copy(ROOT / "memory.json", memory)  # her real memory, read only: a copy is what gets touched
    sessions = []
    for i in range(args.sessions):
        replies = await one_session(args.brain, args.voice, memory, args.user_name)
        sessions.append(replies)
        bad = sum(1 for r in replies if r["flags"])
        print(f"session {i + 1}: {bad}/{len(replies)} replies flagged " + " | ".join(
            f"{r['user'][:18]!r}: {','.join(r['flags'])}" for r in replies if r["flags"]), flush=True)
    kinds = ("identity", "repeat", "markup", "tool talk", "promise", "wrong city")
    total = sum(len(s) for s in sessions)
    counts = {k: sum(1 for s in sessions for r in s if k in r["flags"]) for k in kinds}
    any_bad = sum(1 for s in sessions for r in s if r["flags"])
    print(f"\n{args.brain}: {any_bad}/{total} replies flagged; " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    n_dropped = sum(len(r.get("dropped", [])) for ses in sessions for r in ses)
    n_retried = sum(1 for ses in sessions for r in ses if r.get("retried"))
    n_silent = sum(1 for ses in sessions for r in ses if not r["spoken"])
    print(f"guard: {n_dropped} sentences dropped, {n_retried} replies retried, {n_silent} replies with nothing spoken")
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"replay_{args.brain}.json"
    out.write_text(json.dumps({"brain": args.brain, "voice": args.voice, "counts": counts, "flagged": any_bad,
                               "replies": total, "sessions": sessions}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brain", default="minicpm1b", choices=[b for b, c in BRAINS.items() if c["kind"] == "ollama"])
    ap.add_argument("--voice", default="chatterbox", help="decides the persona's delivery rules (the voice itself is silent here)")
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--user-name", default="Doston")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
