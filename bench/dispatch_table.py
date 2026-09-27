#!/usr/bin/env python
"""One table from every bench/out/dispatch_*.json: checks, time to her first word, words per reply,
cost per call, and which checks failed.

    .venv/Scripts/python.exe bench/dispatch_table.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.dispatch_eval import SCENARIOS  # noqa: E402

OUT = ROOT / "bench" / "out"


def main() -> int:
    rows = []
    for f in sorted(OUT.glob("dispatch_*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        turns = [t for s in r["scenarios"].values() if isinstance(s, dict) for t in s.get("turns", [])]
        if not turns:
            continue
        ok = tot = 0
        failed = []
        for name, sc in r["scenarios"].items():
            if "turns" not in sc:
                failed.append(f"{name}: did not run")
                continue
            for label, fn in SCENARIOS[name]["checks"].items():
                tot += 1
                try:
                    good = bool(fn(sc["turns"]))
                except Exception:  # noqa: BLE001
                    good = False
                ok += good
                if not good:
                    failed.append(f"{name}: {label}")
        firsts = sorted(t["first_s"] for t in turns if t.get("first_s") is not None)
        words = [len(t["said"].split()) for t in turns]
        silent = sum(1 for t in turns if not t["said"].strip())
        calls = sum(1 for s in r["scenarios"].values() if "turns" in s)
        cost = sum(s.get("cost", 0) for s in r["scenarios"].values() if isinstance(s, dict))
        p90 = firsts[int(0.9 * (len(firsts) - 1))] if firsts else None
        rows.append((r["backend"], r.get("mode", "typed caller"), f"{ok}/{tot}",
                     f"{statistics.median(firsts):.2f}" if firsts else "-", f"{p90:.2f}" if p90 is not None else "-",
                     int(statistics.median(words)), silent, f"{cost / max(calls, 1):.3f}", "; ".join(failed)))
    head = ("backend", "caller", "checks", "first word s (median)", "p90", "words/reply", "silent turns", "$ per call", "failed")
    print(" | ".join(head))
    for row in sorted(rows, key=lambda x: (x[1], x[0])):
        print(" | ".join(str(c) for c in row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
