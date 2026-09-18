"""Benchmark the local STT backends on the sample utterances.

Runs each backend (faster-whisper base.en, faster-whisper small.en, Parakeet TDT
0.6B v2 int8) against every ``samples/*.wav`` several times, prints the text and
latency of every run, computes a word error rate against the expected texts,
checks that 0.5 s of silence and 0.5 s of a 440 Hz tone produce (near-)empty
text, and prints a summary table with model load time and RSS growth.

Usage (from the project root)::

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe bench/test_stt_local.py
    ... --backends whisper-base,parakeet --repeats 5 --device cpu --threads 8

Results are also written to ``bench/out/stt_local_results.json``.
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eva.config import SAMPLES_DIR  # noqa: E402
from eva.interfaces import MIC_SAMPLE_RATE, STT  # noqa: E402

OUT_DIR = ROOT / "bench" / "out"

EXPECTED: dict[str, str] = {
    "user_hello": "Hey Eva, how's it going? I just got home from work.",
    "user_rough_day": (
        "Honestly, today was rough. My manager pulled me into a meeting and basically "
        "said the project might get cancelled. I don't know what to do."
    ),
    "user_task": "Can you set a timer for five minutes and remind me to call my mom later tonight?",
}

# Whisper likes digits ("5 minutes"); the reference says "five". Treat those as equal.
_NUM_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten", "11": "eleven",
    "12": "twelve", "13": "thirteen", "14": "fourteen", "15": "fifteen", "16": "sixteen",
    "17": "seventeen", "18": "eighteen", "19": "nineteen", "20": "twenty",
}


# ------------------------------------------------------------------ helpers
def normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation, spell out small numbers -> word list."""
    t = text.lower().replace("’", "'")
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    words = [w.strip("'") for w in t.split()]
    return [_NUM_WORDS.get(w, w) for w in words if w]


def wer(ref: str, hyp: str) -> float:
    """Word error rate = Levenshtein(ref words, hyp words) / len(ref words)."""
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1] / len(r)


def rss_mb() -> float | None:
    """Resident set size of this process in MB (psutil if present, else ctypes/resource)."""
    try:
        import psutil  # type: ignore

        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        pass
    if sys.platform == "win32":
        import ctypes
        import ctypes.wintypes as wt

        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wt.DWORD),
                ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wt.HANDLE  # without this the 64-bit handle is truncated
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
        psapi.GetProcessMemoryInfo.restype = wt.BOOL
        if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return pmc.WorkingSetSize / 1e6
        return None
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3
    except Exception:  # noqa: BLE001
        return None


def load_samples() -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    for p in sorted(SAMPLES_DIR.glob("*.wav")):
        pcm, sr = sf.read(p, dtype="int16")
        if pcm.ndim > 1:
            pcm = pcm[:, 0]
        if sr != MIC_SAMPLE_RATE:
            raise SystemExit(f"{p.name}: expected {MIC_SAMPLE_RATE} Hz, got {sr}")
        out.append((p.stem, np.ascontiguousarray(pcm)))
    if not out:
        raise SystemExit(f"no wav files in {SAMPLES_DIR}")
    return out


def synthetic_inputs() -> list[tuple[str, np.ndarray]]:
    n = MIC_SAMPLE_RATE // 2  # 0.5 s
    silence = np.zeros(n, dtype=np.int16)
    t = np.arange(n) / MIC_SAMPLE_RATE
    tone = (0.3 * 32767 * np.sin(2 * np.pi * 440.0 * t)).astype(np.int16)
    return [("silence_0.5s", silence), ("tone_440hz_0.5s", tone)]


def make_backend(key: str, device: str, threads: int | None) -> STT:
    if key == "whisper-base":
        from eva.stt.faster_whisper_local import FasterWhisperSTT

        return FasterWhisperSTT(model="base.en", device=device, compute_type="int8", cpu_threads=threads or 0)
    if key == "whisper-small":
        from eva.stt.faster_whisper_local import FasterWhisperSTT

        return FasterWhisperSTT(model="small.en", device=device, compute_type="int8", cpu_threads=threads or 0)
    if key == "parakeet":
        from eva.stt.sherpa_parakeet import SherpaParakeetSTT

        return SherpaParakeetSTT(num_threads=threads or 4)
    raise SystemExit(f"unknown backend {key!r}; choose from whisper-base, whisper-small, parakeet")


# --------------------------------------------------------------------- bench
async def bench_backend(
    key: str,
    stt: STT,
    samples: list[tuple[str, np.ndarray]],
    repeats: int,
) -> dict[str, Any]:
    print(f"\n=== {key}  ({stt.name}) ===")
    assert isinstance(stt, STT), f"{key} does not satisfy interfaces.STT"

    gc.collect()
    rss0 = rss_mb()
    t0 = time.perf_counter()
    await stt.warmup()
    load_s = time.perf_counter() - t0
    rss1 = rss_mb()
    device = getattr(stt, "device", "?")
    for line in getattr(stt, "load_log", []):
        print(f"  load: {line}")
    print(f"  loaded in {load_s:.2f}s on {device}; RSS {rss0 and round(rss0)} -> {rss1 and round(rss1)} MB")

    per_sample: dict[str, dict[str, Any]] = {}
    for stem, pcm in samples:
        lat: list[float] = []
        texts: list[str] = []
        audio_s = len(pcm) / MIC_SAMPLE_RATE
        for i in range(repeats):
            tr = await stt.transcribe(pcm, MIC_SAMPLE_RATE)
            lat.append(tr.latency_s)
            texts.append(tr.text)
            print(f"  {stem:<16} run{i + 1} {tr.latency_s * 1000:7.1f} ms  {tr.text!r}")
        ref = EXPECTED.get(stem)
        w = wer(ref, texts[-1]) if ref else None
        per_sample[stem] = {
            "audio_s": round(audio_s, 3),
            "latency_ms": [round(x * 1000, 1) for x in lat],
            "median_ms": round(statistics.median(lat) * 1000, 1),
            "min_ms": round(min(lat) * 1000, 1),
            "rtf": round(statistics.median(lat) / audio_s, 3),
            "text": texts[-1],
            "wer": None if w is None else round(w, 3),
            "consistent": len(set(texts)) == 1,
        }
        if ref is not None:
            print(f"  {stem:<16} WER {w * 100:5.1f}%  median {per_sample[stem]['median_ms']:.0f} ms  rtf {per_sample[stem]['rtf']}")

    synthetic: dict[str, dict[str, Any]] = {}
    for label, pcm in synthetic_inputs():
        tr = await stt.transcribe(pcm, MIC_SAMPLE_RATE)
        clean = len(normalize(tr.text)) == 0
        synthetic[label] = {"text": tr.text, "latency_ms": round(tr.latency_s * 1000, 1), "clean": clean}
        print(f"  {label:<16} {tr.latency_s * 1000:7.1f} ms  {tr.text!r}  -> {'clean' if clean else 'HALLUCINATED'}")

    rss_peak = rss_mb()
    await stt.close()
    gc.collect()

    wers = [v["wer"] for v in per_sample.values() if v["wer"] is not None]
    return {
        "backend": key,
        "name": stt.name,
        "device": device,
        "load_s": round(load_s, 2),
        "rss_before_mb": None if rss0 is None else round(rss0),
        "rss_after_load_mb": None if rss1 is None else round(rss1),
        "rss_delta_mb": None if rss0 is None or rss1 is None else round(rss1 - rss0),
        "rss_peak_mb": None if rss_peak is None else round(rss_peak),
        "mean_wer": round(sum(wers) / len(wers), 3) if wers else None,
        "samples": per_sample,
        "synthetic": synthetic,
    }


def print_summary(results: list[dict[str, Any]], samples: list[tuple[str, np.ndarray]]) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console(width=max(140, Console().width))  # avoid column truncation when piped
    table = Table(title="Local STT summary (median latency per sample, ms)", show_lines=False)
    table.add_column("backend")
    table.add_column("device")
    table.add_column("load s", justify="right")
    table.add_column("RSS +MB", justify="right")
    for stem, pcm in samples:
        table.add_column(f"{stem}\n({len(pcm) / MIC_SAMPLE_RATE:.1f}s)", justify="right")
    table.add_column("mean WER", justify="right")
    table.add_column("silence", justify="center")
    table.add_column("tone", justify="center")
    for r in results:
        row = [
            r["backend"],
            str(r["device"]),
            f"{r['load_s']:.2f}",
            "?" if r["rss_delta_mb"] is None else str(r["rss_delta_mb"]),
        ]
        for stem, _ in samples:
            s = r["samples"].get(stem)
            row.append("-" if s is None else f"{s['median_ms']:.0f}")
        row.append("-" if r["mean_wer"] is None else f"{r['mean_wer'] * 100:.1f}%")
        for k in ("silence_0.5s", "tone_440hz_0.5s"):
            syn = r["synthetic"].get(k)
            row.append("-" if syn is None else ("ok" if syn["clean"] else f"'{syn['text'][:18]}'"))
        table.add_row(*row)
    console.print(table)


def recommend(results: list[dict[str, Any]]) -> str:
    """Pick the backend with the best accuracy among those fast enough for a voice agent."""
    shortest = None
    for r in results:
        for stem, s in r["samples"].items():
            if shortest is None or s["audio_s"] < shortest[1]:
                shortest = (stem, s["audio_s"])
    if not shortest:
        return "no data"
    stem = shortest[0]

    def ok(r: dict[str, Any]) -> bool:
        s = r["samples"].get(stem)
        return bool(s) and s["median_ms"] <= 600 and all(v["clean"] for v in r["synthetic"].values())

    candidates = [r for r in results if ok(r)] or results
    best = min(candidates, key=lambda r: (r["mean_wer"] if r["mean_wer"] is not None else 9, r["samples"][stem]["median_ms"]))
    s = best["samples"][stem]
    return (
        f"{best['backend']} ({best['name']} on {best['device']}): mean WER "
        f"{(best['mean_wer'] or 0) * 100:.1f}%, {s['median_ms']:.0f} ms on the {s['audio_s']:.1f}s utterance, "
        f"load {best['load_s']:.1f}s, +{best['rss_delta_mb']} MB RSS"
    )


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backends", default="whisper-base,whisper-small,parakeet", help="comma separated")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="auto", help="faster-whisper device: auto | cuda | cpu")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (whisper cpu_threads / parakeet num_threads)")
    ap.add_argument("--out", default=str(OUT_DIR / "stt_local_results.json"))
    args = ap.parse_args()

    samples = load_samples()
    results: list[dict[str, Any]] = []
    for key in [k.strip() for k in args.backends.split(",") if k.strip()]:
        stt = make_backend(key, args.device, args.threads)
        try:
            results.append(await bench_backend(key, stt, samples, args.repeats))
        except Exception as e:  # noqa: BLE001 - keep benchmarking the others
            print(f"  !! {key} failed: {type(e).__name__}: {e}")
            results.append({"backend": key, "name": getattr(stt, "name", key), "device": "?", "load_s": float("nan"),
                            "rss_delta_mb": None, "mean_wer": None, "samples": {}, "synthetic": {}, "error": str(e)})

    print()
    print_summary([r for r in results if r["samples"]], samples)
    ok_results = [r for r in results if r["samples"]]
    if ok_results:
        print(f"\nRecommended default: {recommend(ok_results)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"repeats": args.repeats, "device_arg": args.device, "threads": args.threads,
                               "results": results}, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
