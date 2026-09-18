"""Benchmark the local Kokoro TTS (eva/tts/kokoro_local.py) on this machine.

Measures, with real synthesis runs:

* model load time (ORT session build) and full warmup time;
* per voice: time-to-first-audio (TTFA) of the streaming ``synthesize`` path,
  total synthesis time, rendered audio length and real-time factor (RTF =
  synthesis wall time / audio duration), plus the same for the one-shot path
  (single ``Kokoro.create`` call, no sentence split) to show the trade-off;
* the worst event-loop stall observed while inference runs (proves the loop is
  not blocked by ONNX work);
* the fixed per-call floor using a one-word utterance ("Mm.");
* the effect of onnxruntime ``intra_op_num_threads`` on speed.

Writes ``samples/out/kokoro_<voice>.wav`` for every voice and a JSON summary to
``bench/out/tts_local.json``.

Run:  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe bench/test_tts_local.py
      [--runs 3] [--other-runs 2] [--voices af_heart,af_bella,...]
      [--threads 1,2,4,6,8,16] [--no-threads] [--model fp32|fp16|int8|<path>]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from eva.config import SAMPLES_DIR  # noqa: E402
from eva.tts.kokoro_local import SAMPLE_RATE, KokoroTTS  # noqa: E402

TEXT = "Hey, I'm right here. Rough day, huh? Tell me what happened."
FLOOR_TEXT = "Mm."
DEFAULT_VOICES = ["af_heart", "af_bella", "af_nicole", "bf_emma", "am_michael"]
OUT_DIR = ROOT / "bench" / "out"
WAV_DIR = SAMPLES_DIR / "out"

console = Console()


class LoopStallMonitor:
    """Tick every 5 ms and record the largest gap beyond that; a proxy for loop blocking."""

    def __init__(self, period: float = 0.005) -> None:
        self.period = period
        self.max_stall = 0.0
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def _run(self) -> None:
        last = time.perf_counter()
        while not self._stop.is_set():
            await asyncio.sleep(self.period)
            now = time.perf_counter()
            self.max_stall = max(self.max_stall, now - last - self.period)
            last = now

    async def __aenter__(self) -> "LoopStallMonitor":
        self._task = asyncio.create_task(self._run())
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._stop.set()
        if self._task:
            await self._task


async def measure_stream(tts: KokoroTTS, text: str) -> dict[str, Any]:
    """One streaming synthesis: TTFA, total, audio seconds, RTF, chunk count, loop stall."""
    chunks: list[bytes] = []
    t0 = time.perf_counter()
    ttfa: float | None = None
    async with LoopStallMonitor() as mon:
        async for pcm in tts.synthesize(text):
            if ttfa is None:
                ttfa = time.perf_counter() - t0
            chunks.append(pcm)
    total = time.perf_counter() - t0
    pcm_all = b"".join(chunks)
    audio_s = len(pcm_all) / 2 / SAMPLE_RATE
    return {
        "ttfa_ms": round((ttfa or total) * 1000, 1),
        "total_ms": round(total * 1000, 1),
        "audio_s": round(audio_s, 3),
        "rtf": round(total / audio_s, 4) if audio_s else None,
        "chunks": len(chunks),
        "loop_stall_ms": round(mon.max_stall * 1000, 1),
        "pcm": pcm_all,
    }


async def measure_one_shot(tts: KokoroTTS, text: str) -> dict[str, Any]:
    """One single-call synthesis (no sentence split): total, audio seconds, RTF."""
    t0 = time.perf_counter()
    pcm = await tts.synthesize_one_shot(text)
    total = time.perf_counter() - t0
    audio_s = len(pcm) / 2 / SAMPLE_RATE
    return {
        "total_ms": round(total * 1000, 1),
        "audio_s": round(audio_s, 3),
        "rtf": round(total / audio_s, 4) if audio_s else None,
    }


def write_wav(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.frombuffer(pcm, dtype="<i2"), SAMPLE_RATE, subtype="PCM_16")


def med(values: list[float]) -> float:
    return round(statistics.median(values), 1)


async def bench_voices(args: argparse.Namespace, results: dict[str, Any]) -> None:
    """Load once, then run every voice through the streaming and one-shot paths."""
    tts = KokoroTTS(voice=args.voices[0], model=args.model, intra_threads=args.intra)
    t0 = time.perf_counter()
    await tts.warmup()
    results["load"] = {
        "model": tts.model_path.name,
        "model_mb": round(tts.model_path.stat().st_size / 1e6, 1),
        "session_build_s": round(tts.load_time_s or 0.0, 3),
        "warmup_total_s": round(tts.warmup_time_s or 0.0, 3),
        "warmup_wall_s": round(time.perf_counter() - t0, 3),
        "intra_threads": tts.intra_threads or "auto",
    }
    console.print(
        f"[bold]model[/] {results['load']['model']} ({results['load']['model_mb']} MB): "
        f"session build {results['load']['session_build_s']} s, full warmup "
        f"{results['load']['warmup_total_s']} s, threads={results['load']['intra_threads']}"
    )

    # Fixed per-call floor: a one-word utterance.
    floor = [await measure_one_shot(tts, FLOOR_TEXT) for _ in range(3)]
    results["floor"] = {
        "text": FLOOR_TEXT,
        "median_ms": med([f["total_ms"] for f in floor]),
        "audio_s": floor[0]["audio_s"],
    }

    table = Table(title=f"Kokoro voices: {TEXT!r}")
    for col in ("voice", "runs", "TTFA ms (med/min)", "stream total ms", "audio s",
                "stream RTF", "one-shot ms", "one-shot RTF", "loop stall ms", "wav"):
        table.add_column(col, justify="right" if col not in ("voice", "wav") else "left")

    results["voices"] = {}
    for voice in args.voices:
        tts.set_voice(voice)
        n_runs = args.runs if voice == args.voices[0] else args.other_runs
        streams, shots = [], []
        for _ in range(n_runs):
            streams.append(await measure_stream(tts, TEXT))
            shots.append(await measure_one_shot(tts, TEXT))
        wav = WAV_DIR / f"kokoro_{voice}.wav"
        write_wav(wav, streams[-1]["pcm"])
        entry = {
            "runs": n_runs,
            "stream": [{k: v for k, v in s.items() if k != "pcm"} for s in streams],
            "one_shot": shots,
            "ttfa_ms_median": med([s["ttfa_ms"] for s in streams]),
            "ttfa_ms_min": min(s["ttfa_ms"] for s in streams),
            "stream_total_ms_median": med([s["total_ms"] for s in streams]),
            "stream_rtf_median": round(statistics.median([s["rtf"] for s in streams]), 3),
            "one_shot_ms_median": med([s["total_ms"] for s in shots]),
            "one_shot_rtf_median": round(statistics.median([s["rtf"] for s in shots]), 3),
            "audio_s": streams[-1]["audio_s"],
            "loop_stall_ms_max": max(s["loop_stall_ms"] for s in streams),
            "wav": str(wav.relative_to(ROOT)),
        }
        results["voices"][voice] = entry
        table.add_row(
            voice, str(n_runs),
            f"{entry['ttfa_ms_median']:.0f} / {entry['ttfa_ms_min']:.0f}",
            f"{entry['stream_total_ms_median']:.0f}", f"{entry['audio_s']:.2f}",
            f"{entry['stream_rtf_median']:.3f}", f"{entry['one_shot_ms_median']:.0f}",
            f"{entry['one_shot_rtf_median']:.3f}", f"{entry['loop_stall_ms_max']:.1f}",
            entry["wav"],
        )
    console.print(table)
    console.print(
        f"per-call floor ({FLOOR_TEXT!r}): median {results['floor']['median_ms']:.0f} ms "
        f"for {results['floor']['audio_s']:.2f} s of audio"
    )
    await tts.close()


async def bench_threads(args: argparse.Namespace, results: dict[str, Any]) -> None:
    """Rebuild the session per intra-op thread count and time one-shot + streaming."""
    table = Table(title="onnxruntime intra_op_num_threads sweep (af_heart)")
    for col in ("threads", "session build s", "one-shot ms (med)", "one-shot RTF",
                "TTFA ms (med)", "stream total ms (med)"):
        table.add_column(col, justify="right")
    results["threads"] = {}
    for n in args.threads:
        tts = KokoroTTS(voice="af_heart", model=args.model, intra_threads=n or None)
        await tts.warmup()
        shots = [await measure_one_shot(tts, TEXT) for _ in range(3)]
        streams = [await measure_stream(tts, TEXT) for _ in range(3)]
        entry = {
            "session_build_s": round(tts.load_time_s or 0.0, 3),
            "one_shot_ms_median": med([s["total_ms"] for s in shots]),
            "one_shot_rtf_median": round(statistics.median([s["rtf"] for s in shots]), 3),
            "ttfa_ms_median": med([s["ttfa_ms"] for s in streams]),
            "stream_total_ms_median": med([s["total_ms"] for s in streams]),
        }
        key = "auto" if not n else str(n)
        results["threads"][key] = entry
        table.add_row(
            key, f"{entry['session_build_s']:.2f}", f"{entry['one_shot_ms_median']:.0f}",
            f"{entry['one_shot_rtf_median']:.3f}", f"{entry['ttfa_ms_median']:.0f}",
            f"{entry['stream_total_ms_median']:.0f}",
        )
        await tts.close()
    console.print(table)


def verdict(results: dict[str, Any]) -> str:
    """Summarise usability: RTF comfortably < 1 and TTFA in the sub-half-second range."""
    heart = results["voices"].get("af_heart") or next(iter(results["voices"].values()))
    rtf = heart["stream_rtf_median"]
    ttfa = heart["ttfa_ms_median"]
    if rtf < 0.5 and ttfa < 500:
        grade = "USABLE in real time"
    elif rtf < 1.0:
        grade = "marginal: real time but with little headroom"
    else:
        grade = "NOT real time"
    return (
        f"{grade}: af_heart streaming RTF {rtf:.3f}, TTFA {ttfa:.0f} ms, "
        f"one-shot RTF {heart['one_shot_rtf_median']:.3f}; "
        f"load {results['load']['session_build_s']:.2f} s"
    )


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=3, help="runs for the first voice (default 3)")
    ap.add_argument("--other-runs", type=int, default=2, help="runs for the other voices")
    ap.add_argument("--voices", default=",".join(DEFAULT_VOICES))
    ap.add_argument("--model", default="fp32", help="fp32 | fp16 | int8 | path")
    ap.add_argument("--intra", type=int, default=None, help="intra_op threads for the voice run")
    ap.add_argument("--threads", default="1,2,4,6,8,16,0", help="sweep list; 0 = ORT auto")
    ap.add_argument("--no-threads", action="store_true", help="skip the thread sweep")
    args = ap.parse_args()
    args.voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    args.threads = [int(t) for t in args.threads.split(",") if t.strip()]

    results: dict[str, Any] = {
        "machine": {
            "cpu": platform.processor(),
            "cpu_count": os.cpu_count(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "text": TEXT,
        "sample_rate": SAMPLE_RATE,
    }
    import onnxruntime as rt

    results["machine"]["onnxruntime"] = rt.__version__

    await bench_voices(args, results)
    if not args.no_threads:
        await bench_threads(args, results)
    results["verdict"] = verdict(results)
    console.print(f"\n[bold]{results['verdict']}[/]")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "tts_local.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    console.print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
