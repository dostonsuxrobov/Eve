#!/usr/bin/env python
"""Per-preset median latency table from the ``bench/out/e2e_*.json`` files.

    .venv/Scripts/python.exe bench/summarize_e2e.py            # the seven presets
    .venv/Scripts/python.exe bench/summarize_e2e.py --variants # plus the cloud-fast experiments

Numbers are medians over every non-interrupted turn of the files that count for a
preset (``cloud-fast`` = ``e2e_cloud-fast_run1..3.json``; the other presets their
single ``e2e_<preset>.json``).  ``e2e_cloud-fast_race.json`` (batch request raced
against the realtime commit), ``e2e_cloud-fast_run4.json`` and
``e2e_cloud-fast_run5.json`` (both taken while the account's STT was throttled) are
never used.  Every number is measured, nothing is estimated.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "out"
STAGES = ("stt", "llm_ttft", "tts_ttfa", "total")
PRESET_FILES: dict[str, list[str]] = {
    "cloud-fast": ["e2e_cloud-fast_run1.json", "e2e_cloud-fast_run2.json", "e2e_cloud-fast_run3.json"],
    "cloud-smart": ["e2e_cloud-smart.json"],
    "expressive": ["e2e_expressive.json"],
    "local-stt": ["e2e_local-stt.json"],
    "local-brain": ["e2e_local-brain.json"],
    "fully-local": ["e2e_fully-local.json"],
    "parakeet-local": ["e2e_parakeet-local.json"],
}
VARIANT_FILES: dict[str, list[str]] = {
    "cloud-fast (tts ws)": ["e2e_cloud-fast_tts-ws.json"],
    "cloud-fast (tts http)": ["e2e_cloud-fast_tts-http.json"],
    "cloud-fast (real speakers)": ["e2e_cloud-fast_speakers.json"],
    "cloud-fast (batch scribe_v1, old)": ["e2e_cloud-fast.json"],
    "cloud-fast (bargein run)": ["e2e_cloud-fast_bargein.json"],
    "cloud-fast run4 (throttled, excluded)": ["e2e_cloud-fast_run4.json"],
    "cloud-fast race (excluded)": ["e2e_cloud-fast_race.json"],
    "cloud-fast run5 (verification, STT throttled, excluded)": ["e2e_cloud-fast_run5.json"],
}


def load_turns(files: list[str]) -> tuple[list[dict], dict]:
    turns: list[dict] = []
    meta: dict = {}
    for name in files:
        path = OUT_DIR / name
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        meta = {k: d.get(k) for k in ("stt", "llm", "tts")}
        for t in d.get("turns", []):
            if t.get("interrupted") or t.get("error") or t.get("total") is None:
                continue
            turns.append(t)
    return turns, meta


def summarize(label: str, files: list[str]) -> dict | None:
    turns, meta = load_turns(files)
    if not turns:
        return None
    row = {"preset": label, "n": len(turns), **meta}
    for s in STAGES:
        vals = [t[s] for t in turns if t.get(s) is not None]
        row[s] = statistics.median(vals) if vals else None
    row["max_total"] = max(t["total"] for t in turns)
    stage_meds = {s: row[s] for s in ("stt", "llm_ttft", "tts_ttfa") if row[s] is not None}
    row["dominant"] = max(stage_meds, key=stage_meds.get) if stage_meds else "?"
    return row


def fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.2f}"


def table(rows: list[dict]) -> str:
    out = [
        "| preset | turns | median response (s) | max (s) | STT (s) | LLM TTFT (s) | TTS TTFA (s) | dominant stage |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| {r['preset']} | {r['n']} | **{fmt(r['total'])}** | {fmt(r['max_total'])} | {fmt(r['stt'])} | "
            f"{fmt(r['llm_ttft'])} | {fmt(r['tts_ttfa'])} | {r['dominant']} |"
        )
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", action="store_true", help="also print the cloud-fast experiment files")
    ap.add_argument("--json", action="store_true", help="print rows as JSON instead of markdown")
    args = ap.parse_args()

    rows = [r for r in (summarize(k, v) for k, v in PRESET_FILES.items()) if r]
    if args.json:
        print(json.dumps(rows, indent=1))
        return 0
    print(table(rows))
    print()
    print("Stack per preset (from the files):")
    for k, files in PRESET_FILES.items():
        _, meta = load_turns(files)
        if meta:
            print(f"- {k}: {meta.get('stt')} + {meta.get('llm')} + {meta.get('tts')}")
    if args.variants:
        print()
        print("cloud-fast experiment files (not part of the table above):")
        print()
        print(table([r for r in (summarize(k, v) for k, v in VARIANT_FILES.items()) if r]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
