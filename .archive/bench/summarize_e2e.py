#!/usr/bin/env python
"""Median latency per stage from ``bench/out/e2e_*.json`` (the simulator's output).

    .venv/Scripts/python.exe bench/summarize_e2e.py                 # every e2e_*.json, one row each
    .venv/Scripts/python.exe bench/summarize_e2e.py --group preset  # rows merged per preset name

Numbers are medians over every non-interrupted, error-free turn of a file. The
historical per-preset tables from the selection phase (seven presets, transport
variants, throttled runs) live in ``docs/MEASUREMENTS.md``; every number there was
produced by the previous version of this script and is not recomputed here.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "out"
STAGES = ("stt", "llm_ttft", "tts_ttfa", "total")


def load(path: Path) -> tuple[list[dict], dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    meta = {k: d.get(k) for k in ("preset", "lang", "outage", "stt", "llm", "tts")}
    turns = [t for t in d.get("turns", []) if not (t.get("interrupted") or t.get("error") or t.get("total") is None)]
    return turns, meta


def medians(turns: list[dict]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for st in STAGES:
        vals = [float(t[st]) for t in turns if t.get(st) is not None]
        out[st] = round(statistics.median(vals), 3) if vals else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", choices=["file", "preset"], default="file")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    files = sorted(Path(args.out_dir).glob("e2e_*.json"))
    if not files:
        print(f"no e2e_*.json under {args.out_dir}", file=sys.stderr)
        return 1
    groups: dict[str, list[dict]] = defaultdict(list)
    metas: dict[str, dict] = {}
    for f in files:
        turns, meta = load(f)
        key = meta.get("preset") or f.stem if args.group == "preset" else f.stem
        groups[str(key)].extend(turns)
        metas[str(key)] = meta
    fmt = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
    print(f"{'run':40} {'turns':>5} {'stt':>7} {'ttft':>7} {'ttfa':>7} {'total':>7}  stack")
    for key, turns in groups.items():
        m = medians(turns)
        meta = metas[key]
        stack = " + ".join(str(meta.get(k) or "?").split("/")[0] for k in ("stt", "llm", "tts"))
        extra = f" [outage {','.join(meta['outage'])}]" if meta.get("outage") else ""
        print(f"{key:40} {len(turns):5d} {fmt(m['stt']):>7} {fmt(m['llm_ttft']):>7} {fmt(m['tts_ttfa']):>7} {fmt(m['total']):>7}  {stack}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
