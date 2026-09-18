"""Benchmark the ElevenLabs cloud STT backends on the sample utterances.

Backends (all real network calls):
  * batch:scribe_v1, batch:scribe_v2   - POST /v1/speech-to-text (ElevenLabsScribeSTT)
  * rt-burst                            - scribe_v2_realtime websocket via transcribe():
                                          whole utterance sent at once, then commit
  * rt-paced                            - scribe_v2_realtime streaming feed()/commit() fed
                                          with 20 ms frames at real-time pace (like the mic);
                                          the latency reported is commit() -> final text,
                                          which is what the pipeline would experience.
                                          Sockets rotate after each commit + batch fallback.
  * rt-paced-persist (opt-in)           - same, but one websocket reused for every
                                          utterance and no fallback (shows the server's
                                          repeated-content stall).

For every backend each sample is transcribed N times; text, latency and a word error
rate (WER) against the expected transcript are printed, followed by a summary table.
0.5 s of digital silence is also transcribed and must yield empty text without raising.

Usage:
    set PYTHONIOENCODING=utf-8
    .venv/Scripts/python.exe bench/test_stt_cloud.py [--reps 3] [--backends batch:scribe_v1,rt-paced]
                                                     [--idle-test] [--json bench/out/stt_cloud.json]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
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

from eva.config import SAMPLES_DIR, load_keys  # noqa: E402
from eva.interfaces import MIC_SAMPLE_RATE, STT, Transcript  # noqa: E402
from eva.stt.elevenlabs_realtime import ElevenLabsRealtimeSTT  # noqa: E402
from eva.stt.elevenlabs_scribe import ElevenLabsScribeSTT  # noqa: E402

EXPECTED: dict[str, str] = {
    "user_hello": "Hey Eva, how's it going? I just got home from work.",
    "user_rough_day": (
        "Honestly, today was rough. My manager pulled me into a meeting and basically said "
        "the project might get cancelled. I don't know what to do."
    ),
    "user_task": "Can you set a timer for five minutes and remind me to call my mom later tonight?",
}

DEFAULT_BACKENDS = ["batch:scribe_v1", "batch:scribe_v2", "rt-burst", "rt-paced"]
ALL_BACKENDS = DEFAULT_BACKENDS + ["rt-paced-persist"]


# ------------------------------------------------------------------------- WER
_WORD_RE = re.compile(r"[a-z0-9']+")


def normalize(text: str) -> list[str]:
    """Lowercase, drop punctuation, split into words; 'cancelled'/'canceled' treated equal."""
    words = _WORD_RE.findall(text.lower().replace("’", "'"))
    return ["canceled" if w == "cancelled" else w for w in words]


def word_error_rate(ref: str, hyp: str) -> float:
    """Levenshtein word distance / reference length (0.0 = perfect)."""
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


# --------------------------------------------------------------------- runners
def load_samples() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in EXPECTED:
        pcm, sr = sf.read(SAMPLES_DIR / f"{name}.wav", dtype="int16")
        assert sr == MIC_SAMPLE_RATE, f"{name}: expected 16 kHz, got {sr}"
        out[name] = pcm
    return out


async def run_paced(stt: ElevenLabsRealtimeSTT, pcm: np.ndarray) -> Transcript:
    """Feed 20 ms frames at wall-clock pace (simulating the mic), then commit."""
    frame = MIC_SAMPLE_RATE // 50
    t0 = time.perf_counter()
    for i in range(0, len(pcm), frame):
        await stt.feed(pcm[i : i + frame])
        delay = t0 + (i + frame) / MIC_SAMPLE_RATE - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
    tr = await stt.commit()
    tr.meta["stream_wall_s"] = round(time.perf_counter() - t0, 3)
    tr.meta["mode"] = "paced"
    return tr


async def bench_backend(
    label: str, key: str, samples: dict[str, np.ndarray], reps: int
) -> dict[str, Any]:
    """Run one backend over every sample; returns a result dict for the summary/JSON."""
    result: dict[str, Any] = {"backend": label, "runs": [], "silence": None, "warmup_s": None, "error": None}
    stt: STT
    paced = label.startswith("rt-paced")
    if label.startswith("batch:"):
        stt = ElevenLabsScribeSTT(key, model_id=label.split(":", 1)[1])
    elif label in ("rt-burst", "rt-paced"):
        stt = ElevenLabsRealtimeSTT(key, include_timestamps=True)
    elif label == "rt-paced-persist":
        stt = ElevenLabsRealtimeSTT(key, include_timestamps=True, rotate_sessions=False, fallback_to_batch=False)
    else:
        raise ValueError(f"unknown backend {label}")
    print(f"\n=== {label}  ({stt.name}) ===")

    t = time.perf_counter()
    await stt.warmup()
    result["warmup_s"] = round(time.perf_counter() - t, 3)
    print(f"warmup: {result['warmup_s']:.3f} s")

    async def transcribe(pcm: np.ndarray) -> Transcript:
        if paced:
            return await run_paced(stt, pcm)  # type: ignore[arg-type]
        return await stt.transcribe(pcm)

    try:
        for name, pcm in samples.items():
            for rep in range(reps):
                try:
                    tr = await transcribe(pcm)
                    wer = word_error_rate(EXPECTED[name], tr.text)
                    run = {
                        "sample": name, "rep": rep, "latency_s": round(tr.latency_s, 3), "wer": round(wer, 3),
                        "text": tr.text, "audio_s": round(len(pcm) / MIC_SAMPLE_RATE, 2),
                        "words": len(tr.meta.get("words") or []), "language": tr.meta.get("language"),
                        "partials": tr.meta.get("partials"), "stream_wall_s": tr.meta.get("stream_wall_s"),
                        "fallback": tr.meta.get("fallback"),
                    }
                    fb = f"  [{tr.meta['fallback']}]" if tr.meta.get("fallback") else ""
                    print(f"  {name:15s} #{rep}  {tr.latency_s:6.3f} s  WER {wer:5.3f}  {tr.text!r}{fb}")
                except Exception as exc:
                    run = {"sample": name, "rep": rep, "error": f"{type(exc).__name__}: {exc}"}
                    print(f"  {name:15s} #{rep}  ERROR {type(exc).__name__}: {exc}")
                result["runs"].append(run)
                if paced:
                    await asyncio.sleep(0.5)  # avoid rapid successive commits (server throttles)

        # silence must return empty text, no exception
        silence = np.zeros(int(0.5 * MIC_SAMPLE_RATE), dtype=np.int16)
        try:
            tr = await transcribe(silence)
            ok = tr.text == ""
            result["silence"] = {"latency_s": round(tr.latency_s, 3), "text": tr.text, "ok": ok}
            print(f"  {'silence 0.5s':15s}     {tr.latency_s:6.3f} s  text={tr.text!r}  {'OK' if ok else 'FAIL (non-empty)'}")
        except Exception as exc:
            result["silence"] = {"error": f"{type(exc).__name__}: {exc}", "ok": False}
            print(f"  {'silence 0.5s':15s}     ERROR {type(exc).__name__}: {exc}")
    finally:
        await stt.close()
    return result


async def idle_test(key: str, samples: dict[str, np.ndarray], idles: list[float]) -> list[dict[str, Any]]:
    """Does the keep-alive HTTP connection survive N seconds idle between turns?"""
    print("\n=== idle keep-alive test (batch:scribe_v2) ===")
    stt = ElevenLabsScribeSTT(key, model_id="scribe_v2")
    await stt.warmup()
    pcm = samples["user_hello"]
    out = []
    tr = await stt.transcribe(pcm)
    print(f"  idle   0 s -> {tr.latency_s:.3f} s")
    out.append({"idle_s": 0, "latency_s": round(tr.latency_s, 3)})
    for idle in idles:
        await asyncio.sleep(idle)
        tr = await stt.transcribe(pcm)
        print(f"  idle {idle:3.0f} s -> {tr.latency_s:.3f} s (attempts={tr.meta.get('attempts')})")
        out.append({"idle_s": idle, "latency_s": round(tr.latency_s, 3), "attempts": tr.meta.get("attempts")})
    await stt.close()
    return out


# --------------------------------------------------------------------- summary
def summarize(results: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 100)
    print("SUMMARY  (latency = transcribe() wall time; for rt-paced = commit() -> final text)")
    print("=" * 100)
    hdr = f"{'backend':16s} {'sample':15s} {'n':>2s} {'mean s':>7s} {'min s':>7s} {'max s':>7s} {'WER':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for res in results:
        per_sample: dict[str, list[dict[str, Any]]] = {}
        for run in res["runs"]:
            per_sample.setdefault(run["sample"], []).append(run)
        all_lat: list[float] = []
        all_wer: list[float] = []
        for name, runs in per_sample.items():
            ok = [r for r in runs if "error" not in r]
            errs = len(runs) - len(ok)
            if not ok:
                print(f"{res['backend']:16s} {name:15s} {len(runs):2d}  all {errs} runs failed")
                continue
            lat = [r["latency_s"] for r in ok]
            wer = [r["wer"] for r in ok]
            all_lat += lat
            all_wer += wer
            suffix = f"  ({errs} errors)" if errs else ""
            print(
                f"{res['backend']:16s} {name:15s} {len(ok):2d} {statistics.mean(lat):7.3f} {min(lat):7.3f} "
                f"{max(lat):7.3f} {statistics.mean(wer):6.3f}{suffix}"
            )
        sil = res.get("silence") or {}
        sil_txt = "OK" if sil.get("ok") else f"FAIL {sil.get('error') or sil.get('text')!r}"
        n_fb = sum(1 for r in res["runs"] if r.get("fallback"))
        fb_txt = f"   batch fallbacks: {n_fb}" if n_fb else ""
        if all_lat:
            print(
                f"{res['backend']:16s} {'ALL':15s} {len(all_lat):2d} {statistics.mean(all_lat):7.3f} {min(all_lat):7.3f} "
                f"{max(all_lat):7.3f} {statistics.mean(all_wer):6.3f}   warmup {res['warmup_s']:.3f} s   silence: {sil_txt}{fb_txt}"
            )
        print("-" * len(hdr))


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--backends", default=",".join(DEFAULT_BACKENDS), help="comma separated subset of " + ",".join(ALL_BACKENDS))
    ap.add_argument("--idle-test", action="store_true", help="also measure keep-alive survival after 10/30/65 s idle")
    ap.add_argument("--json", default=str(ROOT / "bench" / "out" / "stt_cloud.json"))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    key = load_keys().elevenlabs
    if not key:
        print("elevenlabs_key.txt / ELEVENLABS_API_KEY missing", file=sys.stderr)
        return 2
    samples = load_samples()
    results: list[dict[str, Any]] = []
    for label in [b.strip() for b in args.backends.split(",") if b.strip()]:
        try:
            results.append(await bench_backend(label, key, samples, args.reps))
        except Exception as exc:
            print(f"  backend {label} failed to run: {type(exc).__name__}: {exc}")
            results.append({"backend": label, "runs": [], "silence": None, "warmup_s": None, "error": f"{type(exc).__name__}: {exc}"})
    idle = await idle_test(key, samples, [10, 30, 65]) if args.idle_test else None
    summarize(results)

    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "reps": args.reps, "results": results, "idle_test": idle}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {out_path}")
    failed = any(r.get("error") for r in results) or any(not (r.get("silence") or {}).get("ok") for r in results)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
