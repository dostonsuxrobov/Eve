#!/usr/bin/env python
"""Six scripted calls to Red Oak Transport's dispatcher, run through any backend, checked against
the database.

    .venv/Scripts/python.exe bench/dispatch_eval.py eva:qwen27b                 # Eva's loop, Cerebras 27B (silent voice)
    .venv/Scripts/python.exe bench/dispatch_eval.py openai:gpt-realtime-2.1     # speech-to-speech, typed caller lines
    .venv/Scripts/python.exe bench/dispatch_eval.py gemini:gemini-3.8-live --only lowball,late

The caller's lines are typed (the same words for every backend); each scenario starts from a
freshly built world (eva/jobs/dispatch/world.py), so a booking in one can't leak into the next.
Speech-to-speech models answer in audio as they would on a call (collected, not played) with their
own transcript; Eva's loop runs with a silent voice that records what she would say, so its
latency is to her first sentence, before the voice (ElevenLabs adds ~0.3-0.7 s, docs/MEASUREMENTS.md).

Checks are facts from the tool calls and the database (booked below the floor? with a hazmat
driver? did she look it up?) and a few plain-word checks on what she said (does she say it's
late?). Writes bench/out/dispatch_<backend>.json with every turn.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, AsyncIterator, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eva.jobs.dispatch import desk as desk_mod  # noqa: E402
from eva.jobs.dispatch.world import PLANTED, build  # noqa: E402

OUT = ROOT / "bench" / "out"

Check = Callable[[list[dict[str, Any]]], bool]


def calls(rec: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [c for t in rec for c in t["tools"] if c["name"] == name]


def result(c: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(c["result"])
    except ValueError:
        return {}


def booked(rec: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [result(c) for c in calls(rec, "book_load") if result(c).get("booked")]


def said(rec: list[dict[str, Any]], i: int | None = None) -> str:
    turns = rec if i is None else rec[i : i + 1]
    # the spoken runs' transcripts spell "can’t" with a curly apostrophe; the patterns use '
    return " ".join(t["said"] for t in turns).lower().replace("’", "'").replace("‘", "'")


UNITS = {"zero": 0, "oh": 0, "o": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
         "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
         "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
TENS = {"twenty": 2, "thirty": 3, "forty": 4, "fifty": 5, "sixty": 6, "seventy": 7, "eighty": 8, "ninety": 9}


def spoken_digits(text: str) -> str:
    """Digits as said on a phone, runs joined: "R O forty-eight three o eight" -> "048308",
    "twenty-two seventy-five" -> "2275", "RO-48308" -> "48308". Other words separate runs."""
    out, run, tens = [], "", False
    for tok in re.findall(r"[a-z]+|\d+", text.lower().replace("-", " ")):
        if tok in UNITS and tens and 1 <= UNITS[tok] <= 9:
            run += str(UNITS[tok])  # "forty" "eight" -> 48
            tens = False
            continue
        if tens:
            run += "0"  # "forty" on its own -> 40
            tens = False
        if tok.isdigit():
            run += tok
        elif tok in TENS:
            run += str(TENS[tok])
            tens = True
        elif tok in UNITS:
            run += str(UNITS[tok])
        elif run:
            out.append(run)
            run = ""
    if tens:
        run += "0"
    if run:
        out.append(run)
    return " ".join(out)



def no_phantom_booking(rec: list[dict[str, Any]]) -> bool:
    """She never gives a load number or says it's booked without a booking the tool confirmed
    (Gemini 3.1 Flash Live said "our load number is 10037" with no booking, 2026-09-27)."""
    return bool(booked(rec)) or not any_said(rec, r"load number is|you'?re booked|booked (it|and)|it'?s booked|all booked|locked in")


def any_said(rec: list[dict[str, Any]], pattern: str, i: int | None = None) -> bool:
    return re.search(pattern, said(rec, i)) is not None


def hazmat_ok(rec: list[dict[str, Any]]) -> bool:
    b = booked(rec)
    if not b:
        return False
    d = desk_mod.desk()
    row = d.q1("SELECT hazmat FROM drivers d JOIN trucks t ON t.driver_id=d.id WHERE t.unit=?", (b[0]["truck"],))
    return bool(row and row["hazmat"])


SCENARIOS: dict[str, dict[str, Any]] = {
    "lowball": {
        "caller": "broker, Crescent Freight Partners, lowball reefer Dallas to Atlanta",
        "lines": [
            "Hey, this is Mike with Crescent Freight Partners. I've got a reefer load out of Dallas tomorrow morning going to "
            "Atlanta, forty-two thousand pounds of produce. I can pay nineteen hundred. You got a truck for it?",
            "Nineteen's what my customer gave me. I could maybe stretch to twenty-one fifty. What do you need?",
            "Alright, I'll meet you at twenty-two seventy-five, and that's my ceiling. Can you book it?",
            "Great. What's the load number and who's the driver?",
        ],
        "checks": {
            "priced it before quoting": lambda r: bool(calls(r, "price_load")),
            "found a truck": lambda r: bool(calls(r, "find_available_trucks")),
            "never offered to book below the floor": lambda r: all(int(c["args"].get("rate", 0)) >= 2100 for c in calls(r, "book_load")),
            "booked at 2,100-2,275 on a reefer": lambda r: any(2100 <= b["rate"] <= 2275 for b in booked(r)),
            "no booking claimed without one": no_phantom_booking,
            "read back the load number": lambda r: bool(booked(r)) and (booked(r)[0]["load_number"][3:] in spoken_digits(said(r))
                                                                        or "load number" in said(r)),
        },
    },
    "late": {
        "caller": "broker, Great Lakes, checking on PO 7781234 (about 45 min late, strict receiver)",
        "lines": [
            "Hi, Jenna with Great Lakes Freight Brokerage. Checking on our PO 7781234, it's supposed to deliver in Columbus "
            "this afternoon. Where's your driver?",
            "Is she going to make the appointment?",
            "Okay. Can you handle the receiver so we don't lose the slot? Let me know what you set up.",
        ],
        "checks": {
            "looked the load up": lambda r: bool(calls(r, "load_status")),
            "said it's late": lambda r: any_said(r, r"\b(late|behind|won'?t make|not going to make|miss|after the appointment)\b"),
            "recorded it or asked the receiver": lambda r: bool(calls(r, "log_update") or calls(r, "notify_facility")),
            "no phone call she can't make": lambda r: not any_said(r, r"(i'?m|i am|i'?ll|let me) (calling|call|phone|ring|get on the phone with) (them|the receiver|the dc)"),
            "no invented receiver answer": lambda r: not any_said(r, r"\b(receiver|they)( has| have)? (confirmed|agreed|approved|moved it|pushed it|said yes)"),
        },
    },
    "do_not_use": {
        "caller": "broker flagged do-not-use, rate too good",
        "lines": [
            "Tony from Rapid Eagle Brokerage. I've got a dry van, Memphis to Chicago, picks up tomorrow morning, forty thousand "
            "pounds of paper, paying three-forty a mile, around nineteen hundred all in. Easy money. Want it?",
            "Come on, that's way over market. I'll even throw in quick pay. Deal?",
        ],
        "checks": {
            "checked the broker": lambda r: bool(calls(r, "broker_profile")),
            "did not book": lambda r: not calls(r, "book_load"),
            "turned it down": lambda r: any_said(r, r"\b(can'?t|cannot|won'?t|not able|pass|no thanks|decline|not going to|not taking|rather not|not interested|no deal|unable|no-go|flagged)\b"),
        },
    },
    "hazmat": {
        "caller": "broker, hazmat dry van Houston to Memphis; nearest driver has no endorsement",
        "lines": [
            "Dana with Gulf Coast Chemical Logistics. I need a dry van tomorrow morning out of Houston to Memphis. It's hazmat, "
            "class three paint solvent, forty-one thousand pounds, posted at two ninety-five a mile. Can you cover it?",
            "Sounds good, go ahead and book it at the posted rate.",
            "Perfect. Who's the driver and when will they be there?",
        ],
        "checks": {
            "searched with hazmat": lambda r: any(c["args"].get("hazmat") in (True, "true") for c in calls(r, "find_available_trucks")),
            "booked it": lambda r: bool(booked(r)),
            "booked a hazmat driver": hazmat_ok,
            "not the unendorsed truck 142": lambda r: all(b["truck"] != PLANTED["hazmat_decoy_truck"] for b in booked(r)),
            "no booking claimed without one": no_phantom_booking,
        },
    },
    "owner_find_load": {
        "caller": "the owner: find a load for truck 330 out of Memphis tomorrow, heading north",
        "lines": [
            "Hey Eva, it's Doston. Marcus in truck 330 is empty in Memphis tomorrow morning. Find him the best load you can, "
            "he wants to head north toward home.",
            "Which one would you take and why?",
        ],
        "checks": {
            "searched the board from Memphis": lambda r: any("memphis" in json.dumps(c["args"]).lower() for c in calls(r, "search_load_board")),
            "named a real destination": lambda r: any_said(r, r"indianapolis|chicago|st\.? louis|columbus|cincinnati|louisville|detroit|nashville"),
            "gave a rate": lambda r: any_said(r, r"\$|\bdollars?\b|hundred|thousand|a mile|per mile|\bpay(s|ing)\b|\d{3,}|"
                                                 r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\b"),
            "noticed home is Indianapolis": lambda r: any_said(r, r"indianapolis|indy"),
        },
    },
    "breakdown": {
        "caller": "broker, Desert Sun, reefer to Phoenix on a truck that broke down near Oklahoma City",
        "lines": [
            "Carlos with Desert Sun Logistics. Checking on our frozen chicken to Phoenix, reference PHX-55120, delivers tomorrow "
            "morning. We good?",
            "What are you going to do about it? My customer needs that product.",
            "Okay, what's the realistic new delivery time?",
        ],
        "checks": {
            "looked the load up": lambda r: bool(calls(r, "load_status")),
            "told the truth about the breakdown": lambda r: any_said(r, r"\b(broke|broken|breakdown|turbo|shop|tow|mechanical)", 0),
            "looked for a recovery truck": lambda r: bool(calls(r, "find_available_trucks")),
            "offered a recovery": lambda r: any_said(r, r"\b(another truck|recover|repower|transload|swap|denise|402|send a truck|second truck)"),
            "gave a new time": lambda r: any_said(r, r"\b(tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|a\.?m\.?|p\.?m\.?|o'?clock|evening|afternoon|night|morning)\b", 2),
        },
    },
}


# ---------------------------------------------------------------------------- backends
def fresh_world() -> None:
    if desk_mod._desk is not None:  # Windows won't replace a file that is still open
        desk_mod._desk.close()
        desk_mod._desk = None
    build(desk_mod.DB_PATH)


async def run_eva(brain: str, lines: list[str]) -> list[dict[str, Any]]:
    """Eva's own loop, text in, a silent voice that records what she would say."""
    from eva.config import make_preset
    from eva.mocks import EventLog, MockPlayer, MockSTT, MockTTS
    from eva.pipeline import VoiceAgent
    from eva.session import build_session
    from eva.tools import execute

    class Recorder(MockTTS):
        supports_cues = True

        def __init__(self) -> None:
            super().__init__(ttfa_s=0.0, realtime_factor=0.0)
            self.spoken: list[str] = []
            self.first_t: float | None = None

        async def synthesize(self, text: str, cue: str | None = None) -> AsyncIterator[bytes]:  # type: ignore[override]
            if self.first_t is None:
                self.first_t = time.perf_counter()
            self.spoken.append(text)
            async for pcm in super().synthesize(text):
                yield pcm

    s = build_session(make_preset(brain, "kokoro"), user_name="Doston", job="dispatch")
    tts = Recorder()
    tools_log: list[dict[str, Any]] = []

    async def executor(tc: Any, tools: Any) -> str:
        t0 = time.perf_counter()
        out = await execute(tc, tools)
        tools_log.append({"name": tc.name, "args": tc.arguments, "result": out, "ms": int((time.perf_counter() - t0) * 1000)})
        return out

    from eva.jobs import job_settings

    settings = job_settings(s.preset.settings)
    agent = VoiceAgent(MockSTT([]), s.llm, tts, s.system_prompt, s.tools, settings, frames=None, segmenter=None,
                       player=MockPlayer(tts.sample_rate), on_event=EventLog(False), tool_executor=executor,
                       languages=s.plan.codes, **s.agent_kwargs())
    rec = []
    for line in lines:
        tts.spoken.clear()
        tools_log.clear()
        tts.first_t = None
        t0 = time.perf_counter()
        await agent.say(line)
        rec.append({"heard": line, "said": " ".join(tts.spoken).strip(), "tools": list(tools_log),
                    "first_s": round(tts.first_t - t0, 2) if tts.first_t else None, "done_s": round(time.perf_counter() - t0, 2), "cost": 0.0})
    await agent.close()
    await s.llm.close()
    return rec


async def run_s2s(provider: str, model: str, lines: list[str]) -> list[dict[str, Any]]:
    from eva.s2s import job_instructions, make_client
    from eva.s2s.driver import Call

    prompt, tools = job_instructions("dispatch")
    kw: dict[str, Any] = {"turn_detection": None} if provider == "openai" else {}
    call = Call(make_client(provider, model, prompt, tools, **kw), tools)
    await call.start()
    rec = []
    try:
        for line in lines:
            t = await call.ask(line)
            rec.append({"heard": line, "said": t.said.strip(), "tools": t.tools, "first_s": round(t.first_audio_s, 2) if t.first_audio_s else None,
                        "done_s": round(t.done_s, 2) if t.done_s else None, "audio_s": round(t.audio_s, 1), "cost": round(t.cost, 4)})
            if call.errors:
                rec[-1]["errors"] = list(call.errors)
                call.errors.clear()
    finally:
        await call.close()
    return rec


_caller_tts: Any = None
_caller_audio: dict[tuple[str, str, int], bytes] = {}
FEMALE_CALLERS = ("late", "hazmat")  # Jenna and Dana; the other callers are men


async def spoken(line: str, rate: int, voice: str) -> bytes:
    """The caller's line in Kokoro's voice (local, free) at the backend's input rate, cached per run."""
    global _caller_tts
    import numpy as np

    from eva.audio.mic import LinearResampler
    from eva.tts.kokoro_local import KokoroTTS

    import hashlib

    key = (line, voice, rate)
    if key not in _caller_audio:
        disk = OUT / "caller_audio" / f"{hashlib.sha1(f'{voice}|{line}'.encode()).hexdigest()[:16]}.pcm"  # 24 kHz, shared by runs
        if disk.exists():
            pcm = np.frombuffer(disk.read_bytes(), np.int16)
        else:
            if _caller_tts is None or _caller_tts.voice != voice:
                _caller_tts = KokoroTTS(voice=voice)
                await _caller_tts.warmup()
            pcm = np.frombuffer(b"".join([c async for c in _caller_tts.synthesize(line)]), np.int16)
            disk.parent.mkdir(parents=True, exist_ok=True)
            disk.write_bytes(pcm.tobytes())
        if rate != 24_000:
            pcm = np.clip(LinearResampler(24_000, rate).process(pcm), -32768, 32767).astype(np.int16)
        _caller_audio[key] = pcm.tobytes()
    return _caller_audio[key]


async def run_s2s_audio(provider: str, model: str, lines: list[str], voice: str) -> list[dict[str, Any]]:
    """The caller speaks: each line streamed at real-time pace in 20 ms chunks, then silence, the way
    a phone line sends it. Latency is from the end of the caller's speech to her first audio."""
    from eva.s2s import job_client
    from eva.s2s.driver import Call

    # OpenAI's default semantic VAD waits up to 4 s to be sure the caller is done (5.0-5.3 s answers in the
    # first spoken run); a 500 ms silence endpoint is what Eva's loop and a phone call expect
    kw: dict[str, Any] = {"turn_detection": "server_vad"} if provider == "openai" else {}
    client, tools = job_client(provider, model, **kw)
    call = Call(client, tools)
    await call.start()
    n = client.in_rate // 50
    silence = bytes(2 * n)
    rec = []

    async def pace(pcm: bytes, seconds: float | None = None, until_quiet: bool = False) -> None:
        t0 = time.perf_counter()
        i = 0
        while True:
            chunk = pcm[i : i + 2 * n] if pcm else silence
            if pcm and not chunk:
                return
            await client.send_audio(chunk if len(chunk) == 2 * n else chunk + bytes(2 * n - len(chunk)))
            i += 2 * n
            await asyncio.sleep(max(0.0, t0 + i / 2 / client.in_rate - time.perf_counter()))
            elapsed = time.perf_counter() - t0
            if seconds is not None and elapsed >= seconds:
                return
            if until_quiet and call.turn_done.is_set() and call.pending_tools == 0 and time.perf_counter() - call.last_event_t > 2.0:
                return
            if until_quiet and elapsed > 75:
                return

    try:
        await pace(b"", seconds=1.0)
        for line in lines:
            pcm = await spoken(line, client.in_rate, voice)
            call.mark_start()  # the turn is the caller's line and everything she says until she goes quiet
            await pace(pcm)
            end = time.perf_counter()
            call.turn_done.clear()
            await pace(b"", until_quiet=True)
            t = call.turn
            heard = t.heard or "".join(call._heard_parts)
            after = [v for v in t.voiced if v >= end]
            rec.append({"heard": line, "transcribed": heard.strip(), "said": t.said.strip(), "tools": t.tools,
                        "first_s": round(after[0] - end, 2) if after else None,  # end of the caller's speech -> her first voiced audio
                        "talked_over": any(v < end for v in t.voiced),  # full duplex can speak while the caller does
                        "audio_s": round(call._audio_bytes / 2 / client.out_rate, 1), "cost": round(t.cost, 4)})
            if call.errors:
                rec[-1]["errors"] = list(call.errors)
                call.errors.clear()
            call.mark_start()
    finally:
        await call.close()
    rec[-1]["cost"] = round(rec[-1]["cost"] + max(0.0, call.cost - sum(r["cost"] for r in rec)), 4) if rec else 0
    return rec


async def amain(args: argparse.Namespace) -> int:
    provider, _, model = args.backend.partition(":")
    names = [n for n in SCENARIOS if not args.only or n in args.only.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"backend": args.backend, "at": time.strftime("%Y-%m-%d %H:%M"), "scenarios": {}}
    suffix = "_audio" if args.audio else ""
    path = OUT / f"dispatch_{args.backend.replace(':', '_').replace('/', '_').replace('+', '_')}{suffix}.json"
    if args.only and path.exists():  # a partial rerun updates those scenarios in the existing report
        report["scenarios"] = json.loads(path.read_text(encoding="utf-8")).get("scenarios", {})
    total_pass = total = 0
    for name in names:
        sc = SCENARIOS[name]
        fresh_world()
        t0 = time.perf_counter()
        try:
            if provider == "eva":
                rec = await run_eva(model, sc["lines"])
            elif args.audio or provider == "openai-live":  # GPT-Live takes the caller as audio only
                rec = await run_s2s_audio(provider, model, sc["lines"], "af_bella" if name in FEMALE_CALLERS else "am_michael")
            else:
                rec = await run_s2s(provider, model, sc["lines"])
        except Exception as e:  # noqa: BLE001 - one broken scenario shouldn't hide the others
            print(f"{name}: failed to run: {e!r}")
            report["scenarios"][name] = {"error": repr(e)}
            continue
        results = {}
        for label, fn in sc["checks"].items():
            try:
                results[label] = bool(fn(rec))
            except Exception as e:  # noqa: BLE001
                results[label] = False
                print(f"  check {label!r} crashed: {e!r}")
        ok = sum(results.values())
        total_pass += ok
        total += len(results)
        firsts = [t["first_s"] for t in rec if t.get("first_s") is not None]
        cost = sum(t.get("cost", 0.0) for t in rec)
        report["scenarios"][name] = {"checks": results, "turns": rec, "wall_s": round(time.perf_counter() - t0, 1), "cost": round(cost, 4)}
        print(f"\n== {name}: {ok}/{len(results)}  ({sc['caller']})  first reply {min(firsts, default=0):.2f}-{max(firsts, default=0):.2f} s"
              f"  tools {sum(len(t['tools']) for t in rec)}  cost ${cost:.3f}")
        for t in rec:
            tl = ", ".join(f"{c['name']}({','.join(f'{k}={v}' for k, v in c['args'].items())})" for c in t["tools"])
            print(f"  caller: {t['heard'][:110]}")
            if t.get("transcribed"):
                print(f"    [heard as] {t['transcribed'][:110]}")
            if tl:
                print(f"    [tools] {tl[:300]}")
            print(f"  eva ({t.get('first_s')} s): {t['said'][:400]}")
        for label, v in results.items():
            print(f"   {'PASS' if v else 'FAIL'}  {label}")
    cost = sum(s.get("cost", 0) for s in report["scenarios"].values() if isinstance(s, dict))
    print(f"\n{args.backend}: {total_pass}/{total} checks, ${cost:.3f}")
    report["passed"], report["checks"], report["cost"] = total_pass, total, round(cost, 4)
    report["mode"] = "spoken caller" if (args.audio or provider == "openai-live") and provider != "eva" else "typed caller"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("backend", help="eva:<brain> | openai:<model> | gemini:<model>")
    ap.add_argument("--only", help="comma-separated scenario names: " + ", ".join(SCENARIOS))
    ap.add_argument("--audio", action="store_true", help="speech-to-speech: the caller speaks (Kokoro), not typed lines")
    return asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
