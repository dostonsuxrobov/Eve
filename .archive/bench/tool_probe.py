#!/usr/bin/env python
"""Native tool-calling probe for the brains: can the model *call* Eva's tools?

The conversation eval measures whether a brain fakes tool use in words; this probe
measures the other half: with the real tool schemas and the real persona prompt, does
it emit native tool calls, with the right names and arguments, only when a tool is
needed, and how fast?

    .venv/Scripts/python.exe bench/tool_probe.py --brain local,coder-7b,local-8b
    .venv/Scripts/python.exe bench/tool_probe.py --brain qwen        # Cerebras (a few calls)

Each case is one user turn against a fresh history. Scoring per case: expected tool
names present (with a check on the key argument where one matters), no unexpected
call, and some spoken text where a person would say something. The total is the
number of cases fully passed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eva.config import BRAINS, load_keys  # noqa: E402
from eva.factory import build_llm  # noqa: E402
from eva.interfaces import LLMDelta, LLMDone, LLMToolCall  # noqa: E402
from eva.personas import load_persona, now_string, render  # noqa: E402
from eva.pipeline import _recover_tool_calls  # noqa: E402
from eva.tools import get_tools, tool_notes  # noqa: E402

CASES: list[dict[str, Any]] = [
    {
        "id": "timer_and_note",
        "user": "Oh, can you set a timer for eight minutes? And remind me to call my mom later tonight.",
        "expect": {"set_timer": lambda a: int(a.get("seconds", 0)) == 480, "remember_note": lambda a: "mom" in json.dumps(a).lower()},
    },
    {"id": "time", "user": "What time is it right now?", "expect": {"get_current_time": lambda a: True}},
    {"id": "weather", "user": "Is it going to rain in Philadelphia today?", "expect": {"get_weather": lambda a: "phil" in json.dumps(a).lower()}},
    {"id": "goodbye", "user": "Okay, I have to go. Bye Eva.", "expect": {"end_conversation": lambda a: True}, "want_text": True},
    {"id": "no_tool_vent", "user": "I'm just so tired today. Everything took twice as long as it should have.", "expect": {}, "want_text": True},
    {"id": "no_tool_question", "user": "Do you think I should text my ex back? He messaged me today after a year.", "expect": {}, "want_text": True},
]


async def run_brain(name: str, keys: Any, *, quiet: bool) -> dict[str, Any]:
    cfg = BRAINS[name]
    llm = build_llm(cfg, keys)
    tools = get_tools()
    persona = load_persona("eva")
    system_prompt = render(persona, supports_audio_tags=False, memory_text="- Has a cat named Miso.\n- Calls mom on Sundays.",
                           now=now_string(), user_name="Sam", tool_notes=tool_notes(tools), delivery_cues=False)
    schemas = [t.openai_schema() for t in tools]
    await llm.warmup()
    results = []
    for case in CASES:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": case["user"]}]
        t0 = time.perf_counter()
        first: float | None = None
        text = ""
        calls: list[LLMToolCall] = []
        done: LLMDone | None = None
        try:
            async for ev in llm.stream(messages, tools=schemas):
                if isinstance(ev, LLMDelta):
                    if first is None:
                        first = time.perf_counter() - t0
                    text += ev.text
                elif isinstance(ev, LLMToolCall):
                    if first is None:
                        first = time.perf_counter() - t0
                    calls.append(ev)
                elif isinstance(ev, LLMDone):
                    done = ev
        except Exception as e:
            results.append({"id": case["id"], "error": repr(e)})
            continue
        total = time.perf_counter() - t0
        # a call leaked as JSON text counts, but is flagged (the pipeline recovers it the same way)
        cleaned, recovered = _recover_tool_calls(text, tools, [0])
        leaked = bool(recovered)
        calls += recovered
        got = {c.name: c.arguments for c in calls}
        expect = case["expect"]
        missing = [n for n in expect if n not in got]
        wrong_args = [n for n, chk in expect.items() if n in got and not chk(got[n])]
        unexpected = [n for n in got if n not in expect]
        need_text = case.get("want_text", False)
        no_text = need_text and not cleaned.strip()
        passed = not (missing or wrong_args or unexpected or no_text)
        row = {
            "id": case["id"], "pass": passed, "calls": {k: v for k, v in got.items()}, "leaked_json": leaked,
            "missing": missing, "wrong_args": wrong_args, "unexpected": unexpected, "no_text": no_text,
            "text": cleaned.strip()[:160], "ttft_s": None if first is None else round(first, 3), "total_s": round(total, 3),
            "finish": done.finish_reason if done else None,
            "reasoning_chars": (done.usage.get("reasoning_chars") if done else None),
        }
        results.append(row)
        if not quiet:
            flag = "PASS" if passed else "FAIL"
            print(f"  [{flag}] {case['id']:18} calls={json.dumps(row['calls'])}{' LEAKED-JSON' if leaked else ''} "
                  f"ttft={row['ttft_s']} total={row['total_s']}s text={row['text'][:90]!r}")
    await llm.close()
    score = sum(1 for r in results if r.get("pass"))
    return {"brain": name, "model": cfg.get("model"), "score": f"{score}/{len(CASES)}", "results": results}


async def amain(args: argparse.Namespace) -> int:
    keys = load_keys()
    names = [b.strip() for b in args.brain.split(",") if b.strip()]
    out: list[dict[str, Any]] = []
    for name in names:
        print(f"=== {name} ({BRAINS[name].get('model')}) ===")
        out.append(await run_brain(name, keys, quiet=args.quiet))
    print("\nbrain            score   ttft median   leaked-json")
    for r in out:
        tt = sorted(x["ttft_s"] for x in r["results"] if x.get("ttft_s") is not None)
        med = tt[len(tt) // 2] if tt else None
        leaked = sum(1 for x in r["results"] if x.get("leaked_json"))
        print(f"{r['brain']:16} {r['score']:7} {med!s:13} {leaked}")
    path = ROOT / "bench" / "out" / "tool_probe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"wrote {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brain", default="local", help="comma list of names from eva.config.BRAINS")
    ap.add_argument("--quiet", action="store_true")
    return asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
