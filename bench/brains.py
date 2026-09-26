#!/usr/bin/env python
"""Small-brain bench: every candidate plays the same example conversations and the tool
probe with the Eva persona, alone on the GPU, and we record what it costs and how it talks.

    .venv/Scripts/python.exe bench/brains.py                  # every candidate in bench/common.py
    .venv/Scripts/python.exe bench/brains.py --models qwen3.5:0.8b,gemma3:1b-it-qat
    .venv/Scripts/python.exe bench/brains.py --scenarios rough_day,sick_pet --no-tools

Per model: unload everything else, load it with an 8k context, read its VRAM from
``ollama ps``, play the English scenarios in bench/scenarios.json (its own replies go back
into the history, as in a real conversation), then the six tool cases of the cloud-era
probe. Written to bench/out/brains/: one transcript (.md) and result (.json) per model;
compare.md (every model's reply under each user line) and summary.md / summary.json (the
numbers) are rebuilt after each run from every model's saved result.

The style flags are what a script can see (length, emoji, lists, promised actions, banned
phrases). Honesty, warmth and "feel" need a reader: that's the transcripts.
Everything runs locally: $0, no quota.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from common import (BENCH_MEMORY, BENCH_USER, CANDIDATES, PROMISE_RE, ROOT, TOOL_SCHEMAS, TOOLS, chat, client,
                    eva_prompt, load, opening, show, style_flags, think_setting, unload_all, vram)

OUT = ROOT / "bench" / "out" / "brains"

# The cloud-era tool probe (.archive/bench/tool_probe.py): one user turn each, fresh history.
TOOL_CASES: list[dict[str, Any]] = [
    {"id": "timer_and_note", "user": "Oh, can you set a timer for eight minutes? And remind me to call my mom later tonight.",
     "expect": {"set_timer": lambda a: int(a.get("seconds", 0)) == 480, "remember_note": lambda a: "mom" in json.dumps(a).lower()}},
    {"id": "time", "user": "What time is it right now?", "expect": {"get_current_time": lambda a: True}},
    {"id": "weather", "user": "Is it going to rain in Philadelphia today?", "expect": {"get_weather": lambda a: "phil" in json.dumps(a).lower()}},
    {"id": "goodbye", "user": "Okay, I have to go. Bye Eva.", "expect": {"end_conversation": lambda a: True}, "want_text": True},
    {"id": "no_tool_vent", "user": "I'm just so tired today. Everything took twice as long as it should have.", "expect": {}, "want_text": True},
    {"id": "no_tool_question", "user": "Do you think I should text my ex back? He messaged me today after a year.", "expect": {}, "want_text": True},
]
CLAIMED_RE = re.compile(r"\b(timer('s| is)? set|set (a|the|your) timer|noted|got it|done|saved)\b")
# (substring of a style flag, column label) for the flag-count table
FLAG_KINDS = (("emoji", "emoji"), ("*action*", "*action*"), ("[bracket]", "[bracket]"), ("list/markdown", "list"),
              ("questions", ">1 question"), ("words", ">40 words"), ("promises an action", "promise"),
              ('"', "banned phrase"), ("empty", "empty"), ("same opening", "same opening"),
              ("think tags", "think tags"), ("cut at token cap", "cut"), ("ERROR", "error"))


def _check(fn: Any, args: Any) -> bool:
    try:
        return bool(fn(args if isinstance(args, dict) else {}))
    except (TypeError, ValueError):
        return False


def _median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def _gpu_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def run_model(c: Any, model: str, note: str, scenarios: list[dict[str, Any]], *, do_tools: bool) -> dict[str, Any]:
    info = show(c, model)
    caps = list(info.get("capabilities") or [])
    details = info.get("details") or {}
    think = think_setting(caps)
    unload_all(c)
    load_s = load(c, model)
    mem = vram(c, model)
    gpu_mib = _gpu_used_mib()
    print(f"  loaded in {load_s:.1f} s; {mem['size_vram'] / 2**30:.2f} GB in VRAM of {mem['size'] / 2**30:.2f} GB "
          f"(ctx {mem['context']}); GPU total {gpu_mib} MiB; capabilities {', '.join(caps)}", flush=True)

    prompt = eva_prompt(user_name=BENCH_USER, memory=BENCH_MEMORY, tools=False)
    turns: list[dict[str, Any]] = []
    for sc in scenarios:
        seed = list(sc.get("seed_history") or [])
        messages: list[dict[str, Any]] = [{"role": "system", "content": prompt}, *seed]
        prev_open = next((opening(m["content"]) for m in reversed(seed) if m["role"] == "assistant"), "")
        for i, user in enumerate(sc["turns"]):
            messages.append({"role": "user", "content": user})
            rep = chat(c, model, messages, think=think)
            text = rep.text.strip()
            flags = style_flags(text, tools_offered=False)
            op = opening(text)
            if op and op == prev_open:
                flags.append("same opening")
            prev_open = op
            if rep.thinking:
                flags.append(f"thought {len(rep.thinking)} chars")
            if rep.done_reason == "length":
                flags.append("cut at token cap")
            if rep.error:
                flags.append("ERROR " + rep.error)
            turns.append({"scenario": sc["id"], "turn": i, "user": user, "reply": text, "flags": flags,
                          "ttft_s": rep.ttft_s, "tok_s": rep.tok_per_s, "words": len(text.split()),
                          "prompt_tokens": rep.prompt_count, "prompt_s": rep.prompt_s, "total_s": rep.total_s})
            messages.append({"role": "assistant", "content": text})
        print(f"  {sc['id']:18} " + " | ".join(f"{t['ttft_s'] or 0:.2f}s {t['words']}w" for t in turns if t["scenario"] == sc["id"]), flush=True)

    probe: list[dict[str, Any]] | None = None
    if do_tools and "tools" in caps:
        probe = []
        tprompt = eva_prompt(user_name=BENCH_USER, memory="- Has a cat named Miso.\n- Calls mom on Sundays.", tools=True)
        for case in TOOL_CASES:
            rep = chat(c, model, [{"role": "system", "content": tprompt}, {"role": "user", "content": case["user"]}],
                       tools=TOOL_SCHEMAS, think=think)
            text = rep.text.strip()
            low = text.lower().replace("’", "'")
            got = {tc["name"]: tc["arguments"] for tc in rep.tool_calls}
            expect = case["expect"]
            missing = [n for n in expect if n not in got]
            wrong = [n for n, fn in expect.items() if n in got and not _check(fn, got[n])]
            unexpected = [n for n in got if n not in expect]
            no_text = bool(case.get("want_text")) and not text
            # a call written out as text ("end_conversation{...}", "<tool_call>"): nothing runs it
            leaked = "<tool_call" in low or '"arguments"' in low or any(t["name"] in low for t in TOOLS)
            said_not_done = bool(missing) and bool(PROMISE_RE.search(low) or CLAIMED_RE.search(low))
            passed = not (missing or wrong or unexpected or no_text or rep.error)
            probe.append({"id": case["id"], "pass": passed, "calls": got, "missing": missing, "wrong_args": wrong,
                          "unexpected": unexpected, "no_text": no_text, "leaked": leaked, "said_not_done": said_not_done,
                          "text": text, "ttft_s": rep.ttft_s, "error": rep.error})
            mark = "PASS" if passed else "FAIL"
            extra = " said-it-didn't-call" if said_not_done else ""
            extra += " leaked" if leaked else ""
            print(f"  [{mark}] {case['id']:17} calls={json.dumps(got)}{extra} text={text[:70]!r}", flush=True)

    ttfts = [t["ttft_s"] for t in turns if t["ttft_s"] is not None]
    counts = {label: sum(1 for t in turns if any(key in f for f in t["flags"])) for key, label in FLAG_KINDS}
    return {
        "model": model, "note": note, "capabilities": caps, "params": details.get("parameter_size"),
        "quant": details.get("quantization_level"), "family": details.get("family"), "think": think,
        "load_s": round(load_s, 2), "vram_gb": round(mem["size_vram"] / 2**30, 2), "size_gb": round(mem["size"] / 2**30, 2),
        "context": mem["context"], "gpu_total_mib": gpu_mib,
        "ttft_first_s": ttfts[0] if ttfts else None, "ttft_median_s": _median(ttfts[1:]),
        "ttft_max_s": max(ttfts[1:]) if len(ttfts) > 1 else None,
        "tok_s_median": _median([t["tok_s"] for t in turns if t["tok_s"]]),
        "words_median": _median([t["words"] for t in turns]),
        "turns": len(turns), "flagged_turns": sum(1 for t in turns if any(not f.startswith("thought") for f in t["flags"])),
        "flag_counts": counts,
        "probe_score": None if probe is None else f"{sum(1 for p in probe if p['pass'])}/{len(probe)}",
        "probe_said_not_done": None if probe is None else sum(1 for p in probe if p["said_not_done"]),
        "probe": probe, "transcript": turns,
    }


# ------------------------------------------------------------------ reports
def _slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def _fmt(x: float | None, spec: str = ".2f") -> str:
    return "–" if x is None else format(x, spec)


def write_transcript(r: dict[str, Any], scenarios: dict[str, dict[str, Any]]) -> Path:
    lines = [f"# {r['model']}", "", f"{r['note']}. {r['params']} {r['quant']}, {r['family']}; capabilities: {', '.join(r['capabilities'])}.",
             f"VRAM {r['vram_gb']} GB (ctx {r['context']}), load {r['load_s']} s, first reply {_fmt(r['ttft_first_s'])} s "
             f"(whole persona prompt), then median {_fmt(r['ttft_median_s'])} s to the first word, {_fmt(r['tok_s_median'], '.0f')} tok/s.", ""]
    current = None
    for t in r["transcript"]:
        if t["scenario"] != current:
            current = t["scenario"]
            sc = scenarios[current]
            lines += ["", f"## {sc['title']} (`{current}`)", "", f"_Good looks like: {sc['what_good_looks_like']}_", ""]
            for m in sc.get("seed_history") or []:
                lines.append(f"> _(seeded {m['role']})_ {m['content']}")
            if sc.get("seed_history"):
                lines.append("")
        flags = f" · **{', '.join(t['flags'])}**" if t["flags"] else ""
        lines += [f"**Sam:** {t['user']}", "", f"**Eva:** {t['reply'] or '_(nothing)_'}", "",
                  f"<sub>{_fmt(t['ttft_s'])} s to first word · {_fmt(t['tok_s'], '.0f')} tok/s · {t['words']} words{flags}</sub>", ""]
    if r["probe"] is not None:
        lines += ["", f"## Tool probe: {r['probe_score']}", ""]
        for p in r["probe"]:
            lines.append(f"- {'PASS' if p['pass'] else 'FAIL'} `{p['id']}` calls={json.dumps(p['calls'])}"
                         f"{' **said it, didn’t call**' if p['said_not_done'] else ''}{' **leaked**' if p['leaked'] else ''}: {p['text'] or '_(no text)_'}")
    else:
        lines += ["", "## Tool probe: n/a (no tool support in its Ollama template)"]
    path = OUT / f"{_slug(r['model'])}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_compare(results: list[dict[str, Any]], scenarios: list[dict[str, Any]]) -> Path:
    lines = ["# Every model's reply under each user line", "",
             "Each model hears its own earlier replies, so later turns drift apart; read a model's own transcript for the whole conversation.", ""]
    for sc in scenarios:
        lines += [f"## {sc['title']} (`{sc['id']}`)", "", f"_Good looks like: {sc['what_good_looks_like']}_", ""]
        for i, user in enumerate(sc["turns"]):
            lines += [f"**Sam:** {user}", ""]
            for r in results:
                t = next((t for t in r["transcript"] if t["scenario"] == sc["id"] and t["turn"] == i), None)
                if t is None:
                    continue
                flags = f" _({', '.join(t['flags'])})_" if t["flags"] else ""
                lines.append(f"- `{r['model']}`: {t['reply'] or '_(nothing)_'}{flags}")
            lines.append("")
    path = OUT / "compare.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_summary(results: list[dict[str, Any]]) -> Path:
    head = ("| model | params | VRAM GB | load s | first reply s | to first word, median s | tok/s | words/reply | "
            "flagged turns | tool probe | said, didn't call |")
    lines = ["# Small-brain bench", "", f"{time.strftime('%Y-%m-%d %H:%M')}, Ollama at 127.0.0.1, ctx {results[0]['context'] if results else '?'}, "
             "one model on the GPU at a time. 'First reply' includes reading the whole persona prompt; later turns reuse it from cache.", "",
             head, "|" + "---|" * (head.count("|") - 1)]
    for r in results:
        lines.append(f"| `{r['model']}` | {r['params']} {r['quant']} | {r['vram_gb']} | {r['load_s']} | {_fmt(r['ttft_first_s'])} | "
                     f"{_fmt(r['ttft_median_s'])} (max {_fmt(r['ttft_max_s'])}) | {_fmt(r['tok_s_median'], '.0f')} | {_fmt(r['words_median'], '.0f')} | "
                     f"{r['flagged_turns']}/{r['turns']} | {r['probe_score'] or 'n/a'} | {r['probe_said_not_done'] if r['probe_said_not_done'] is not None else '–'} |")
    lines += ["", "Flag counts (turns with at least one):", ""]
    kinds = [label for _, label in FLAG_KINDS]
    lines.append("| model | " + " | ".join(kinds) + " |")
    lines.append("|---|" + "---|" * len(kinds))
    for r in results:
        lines.append(f"| `{r['model']}` | " + " | ".join(str(r["flag_counts"][k]) for k in kinds) + " |")
    path = OUT / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="", help="comma list of Ollama tags (default: every candidate)")
    ap.add_argument("--scenarios", default="", help="comma list of scenario ids (default: all)")
    ap.add_argument("--no-tools", action="store_true", help="skip the tool probe")
    ap.add_argument("--report-only", action="store_true", help="rebuild compare.md / summary.md from the saved results")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

    notes = dict(CANDIDATES)
    models = [m.strip() for m in args.models.split(",") if m.strip()] or [m for m, _ in CANDIDATES]
    data = json.loads((ROOT / "bench" / "scenarios.json").read_text(encoding="utf-8"))["scenarios"]
    wanted = {s.strip() for s in args.scenarios.split(",") if s.strip()}
    scenarios = [s for s in data if not wanted or s["id"] in wanted]
    OUT.mkdir(parents=True, exist_ok=True)

    ran = 0
    with client() as c:
        for model in [] if args.report_only else models:
            print(f"=== {model}", flush=True)
            try:
                r = run_model(c, model, notes.get(model, ""), scenarios, do_tools=not args.no_tools)
            except Exception as e:  # a model that won't load must not stop the others
                print(f"  FAILED: {e!r}", flush=True)
                continue
            ran += 1
            (OUT / f"{_slug(model)}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            print(f"  -> {write_transcript(r, {s['id']: s for s in scenarios})}", flush=True)
        if not args.report_only:
            unload_all(c)
    # the reports cover every model benched so far (one JSON each), in candidate order
    order = [m for m, _ in CANDIDATES] + [m for m in models if m not in notes]
    results = [json.loads(p.read_text(encoding="utf-8")) for p in (OUT / f"{_slug(m)}.json" for m in order) if p.exists()]
    if results:
        write_compare(results, scenarios)
        print(Path(write_summary(results)).read_text(encoding="utf-8"))
    return 0 if results and (ran or args.report_only) else 1


if __name__ == "__main__":
    sys.exit(main())
