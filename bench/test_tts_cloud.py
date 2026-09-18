"""Benchmark ElevenLabs TTS: models x voices x transport modes.

Measures time-to-first-audio (TTFA) and total synthesis time for one conversational
sentence, writes every rendering to ``samples/out/tts_<model>_<voice>_<mode>.wav``
(24 kHz int16) for listening, checks that ``eleven_v3`` renders audio tags, and prints a
table with the median TTFA per combination plus a data-driven recommendation.

Run::

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe bench/test_tts_cloud.py
    ... --models eleven_flash_v2_5 --voices sarah lily --modes ws --runs 5

Results are also written to ``bench/out/tts_cloud.json``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eva.config import EL_VOICES, load_keys  # noqa: E402
from eva.tts.elevenlabs import SAMPLE_RATE, ElevenLabsError, ElevenLabsTTS  # noqa: E402

SENTENCE = "Hey, I'm right here. Rough day, huh? Tell me what happened."
TAGGED = "[laughs] Okay, okay... [sighs] that's genuinely rough. I'm here."
UNTAGGED = "Okay, okay... that's genuinely rough. I'm here."

DEFAULT_MODELS = ["eleven_flash_v2_5", "eleven_turbo_v2_5", "eleven_v3"]
DEFAULT_VOICES = ["sarah", "rachel", "lily", "jessica", "charlotte"]
DEFAULT_MODES = ["ws", "http"]

OUT_WAV = ROOT / "samples" / "out"
OUT_JSON = ROOT / "bench" / "out" / "tts_cloud.json"


@dataclass
class Run:
    ttfa_s: float | None
    total_s: float | None
    audio_s: float
    chunks: int
    setup_hidden: bool | None = None


@dataclass
class Combo:
    model: str
    voice: str
    mode: str  # requested
    effective_mode: str = ""
    fallback_reason: str | None = None
    warmup_s: float | None = None
    runs: list[Run] = field(default_factory=list)
    error: str | None = None
    status: int | None = None
    wav: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and any(r.ttfa_s is not None for r in self.runs)

    def median(self, key: str) -> float | None:
        vals = [getattr(r, key) for r in self.runs if getattr(r, key) is not None]
        return statistics.median(vals) if vals else None


async def timed_synthesis(tts: ElevenLabsTTS, text: str) -> tuple[Run, bytes]:
    """Synthesize ``text`` once, timing from the call to first / last chunk."""
    t0 = time.perf_counter()
    first: float | None = None
    parts: list[bytes] = []
    async for chunk in tts.synthesize(text):
        if first is None:
            first = time.perf_counter() - t0
        assert len(chunk) % 2 == 0, "odd-length PCM chunk"
        parts.append(chunk)
    total = time.perf_counter() - t0
    pcm = b"".join(parts)
    run = Run(
        ttfa_s=first,
        total_s=total,
        audio_s=len(pcm) / (SAMPLE_RATE * 2),
        chunks=len(parts),
        setup_hidden=tts.last.setup_hidden,
    )
    return run, pcm


def write_wav(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.frombuffer(pcm, dtype=np.int16)
    sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")


async def bench_combo(
    key: str, model: str, voice: str, mode: str, runs: int, gap_s: float, text: str
) -> Combo:
    combo = Combo(model=model, voice=voice, mode=mode)
    tts = ElevenLabsTTS(api_key=key, voice_id=EL_VOICES[voice], model_id=model, mode=mode)
    pcm_last = b""
    try:
        t0 = time.perf_counter()
        await tts.warmup()
        combo.warmup_s = time.perf_counter() - t0
        for i in range(runs):
            if i:
                await asyncio.sleep(gap_s)  # let the pre-opened socket finish its handshake
            run, pcm = await timed_synthesis(tts, text)
            combo.runs.append(run)
            pcm_last = pcm
            print(
                f"  {model:<18} {voice:<9} {mode:<4} run{i + 1}: ttfa={run.ttfa_s:.3f}s "
                f"total={run.total_s:.3f}s audio={run.audio_s:.2f}s chunks={run.chunks}"
                + (f" hidden={run.setup_hidden}" if run.setup_hidden is not None else ""),
                flush=True,
            )
    except ElevenLabsError as exc:
        combo.error = f"{exc.mode} HTTP {exc.status} {exc.code or ''}: {exc.message}".strip()
        combo.status = exc.status
        print(f"  {model:<18} {voice:<9} {mode:<4} ERROR {combo.error}", flush=True)
    except Exception as exc:  # network / protocol problems: record, keep going
        combo.error = f"{type(exc).__name__}: {exc}"
        print(f"  {model:<18} {voice:<9} {mode:<4} ERROR {combo.error}", flush=True)
    finally:
        combo.effective_mode = tts.mode
        combo.fallback_reason = tts.fallback_reason
        await tts.close()
    if pcm_last:
        path = OUT_WAV / f"tts_{model}_{voice}_{mode}.wav"
        write_wav(path, pcm_last)
        combo.wav = str(path.relative_to(ROOT))
    return combo


async def bench_cold(key: str, model: str, voice: str, mode: str) -> Run | str:
    """One synthesis on a fresh instance with NO warmup: shows the setup cost the pool hides."""
    tts = ElevenLabsTTS(api_key=key, voice_id=EL_VOICES[voice], model_id=model, mode=mode)
    try:
        run, _ = await timed_synthesis(tts, SENTENCE)
        return run
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    finally:
        await tts.close()


async def tag_test(key: str, voice: str) -> dict[str, Any]:
    """Render the tagged line with eleven_v3 (http) and an untagged control; compare durations."""
    result: dict[str, Any] = {"voice": voice, "text": TAGGED}
    tts = ElevenLabsTTS(api_key=key, voice_id=EL_VOICES[voice], model_id="eleven_v3", mode="http")
    try:
        await tts.warmup()
        run, pcm = await timed_synthesis(tts, TAGGED)
        path = OUT_WAV / f"tts_eleven_v3_{voice}_tags.wav"
        write_wav(path, pcm)
        result.update(tagged=asdict(run), wav=str(path.relative_to(ROOT)))
        run2, pcm2 = await timed_synthesis(tts, UNTAGGED)
        path2 = OUT_WAV / f"tts_eleven_v3_{voice}_notags.wav"
        write_wav(path2, pcm2)
        result.update(untagged=asdict(run2), wav_control=str(path2.relative_to(ROOT)))
        # Tags are non-verbal sounds: a rendered [laughs]/[sighs] adds audible time versus
        # the same words without tags, and a v3 render of ~50 chars should be 2-12 s.
        plausible = 2.0 <= run.audio_s <= 12.0
        result.update(
            duration_plausible=plausible,
            tags_add_s=round(run.audio_s - run2.audio_s, 2),
            rendered=plausible and run.audio_s > run2.audio_s,
        )
    except ElevenLabsError as exc:
        result["error"] = f"http HTTP {exc.status} {exc.code or ''}: {exc.message}"
    finally:
        await tts.close()
    return result


def fmt(v: float | None, nd: int = 3) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def print_table(combos: list[Combo]) -> None:
    """Plain fixed-width table (safe for cp1252 consoles and log scraping)."""
    print(
        f"{'model':<18} {'voice':<9} {'mode':<10} {'warmup':>6} {'ttfa_med':>8} {'ttfa_min':>8} "
        f"{'total_med':>9} {'audio_s':>7} {'runs':>4}  status"
    )
    for c in sorted(combos, key=lambda c: (c.model, c.mode, c.median("ttfa_s") or 9e9)):
        mode = c.mode if c.effective_mode in ("", c.mode) else f"{c.mode}->{c.effective_mode}"
        ttfas = [r.ttfa_s for r in c.runs if r.ttfa_s is not None]
        print(
            f"{c.model:<18} {c.voice:<9} {mode:<10} {fmt(c.warmup_s, 2):>6} {fmt(c.median('ttfa_s')):>8} "
            f"{fmt(min(ttfas) if ttfas else None):>8} {fmt(c.median('total_s')):>9} "
            f"{fmt(c.median('audio_s'), 2):>7} {len(c.runs):>4}  {'ok' if c.ok else (c.error or 'no audio')}"
        )


def recommend(combos: list[Combo]) -> dict[str, Any]:
    """Pick the lowest-median-TTFA working combo, preferring the warm premade voices."""
    ok = [c for c in combos if c.ok and c.median("ttfa_s") is not None]
    if not ok:
        return {"note": "no working combination"}
    def med(c: Combo) -> float:
        return c.median("ttfa_s") or 9e9

    by_model_mode: dict[str, list[float]] = {}
    for c in ok:
        by_model_mode.setdefault(f"{c.model}/{c.effective_mode or c.mode}", []).append(med(c))
    summary = {k: round(statistics.median(v), 3) for k, v in by_model_mode.items()}
    fastest = min(ok, key=med)
    warm = [c for c in ok if c.voice in ("sarah", "lily") and c.model == "eleven_flash_v2_5"]
    pick = min(warm, key=med) if warm else fastest
    # Anything within 50 ms of the pick is a coin flip on this network; list it.
    near = [
        f"{c.model}/{c.voice}/{c.mode} ttfa={fmt(c.median('ttfa_s'))}"
        for c in ok
        if c is not pick and med(c) - med(pick) <= 0.05
    ]
    return {
        "median_ttfa_by_model_mode": summary,
        "fastest_combo": f"{fastest.model}/{fastest.voice}/{fastest.effective_mode or fastest.mode} "
        f"ttfa={fmt(fastest.median('ttfa_s'))}",
        "recommended": {
            "model_id": pick.model,
            "voice": pick.voice,
            "mode": pick.mode,
            "median_ttfa_s": round(pick.median("ttfa_s") or 0, 3),
            "why": "lowest median TTFA among the warm premade voices (sarah/lily) on flash; "
            "listen to samples/out/*.wav to confirm naturalness",
            "within_50ms": near,
        },
    }


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--voices", nargs="+", default=DEFAULT_VOICES, choices=sorted(EL_VOICES))
    ap.add_argument("--modes", nargs="+", default=DEFAULT_MODES, choices=["ws", "http"])
    ap.add_argument("--runs", type=int, default=3, help="runs per combo for eleven_flash_v2_5")
    ap.add_argument("--runs-other", type=int, default=1, help="runs per combo for other models")
    ap.add_argument("--gap", type=float, default=0.5, help="seconds between runs (like a real turn gap)")
    ap.add_argument("--text", default=SENTENCE)
    ap.add_argument("--no-tags", action="store_true", help="skip the eleven_v3 audio-tag test")
    ap.add_argument("--no-cold", action="store_true", help="skip the cold (no warmup) reference runs")
    ap.add_argument("--json", type=Path, default=OUT_JSON)
    args = ap.parse_args()
    args.json = args.json.resolve()

    key = load_keys().elevenlabs
    if not key:
        print("elevenlabs_key.txt / ELEVENLABS_API_KEY missing", file=sys.stderr)
        return 2

    combos: list[Combo] = []
    t_start = time.perf_counter()
    for model in args.models:
        n = args.runs if model == "eleven_flash_v2_5" else args.runs_other
        for voice in args.voices:
            for mode in args.modes:
                combos.append(await bench_combo(key, model, voice, mode, n, args.gap, args.text))

    cold: dict[str, Any] = {}
    if not args.no_cold:
        print("cold reference (fresh instance, no warmup):", flush=True)
        for mode in args.modes:
            r = await bench_cold(key, args.models[0], args.voices[0], mode)
            cold[f"{args.models[0]}/{args.voices[0]}/{mode}"] = asdict(r) if isinstance(r, Run) else r
            print(f"  {mode}: {r if isinstance(r, str) else f'ttfa={r.ttfa_s:.3f}s total={r.total_s:.3f}s'}", flush=True)

    tags: dict[str, Any] = {}
    if not args.no_tags and "eleven_v3" in args.models:
        print("eleven_v3 audio-tag test:", flush=True)
        tags = await tag_test(key, args.voices[0])
        print("  " + json.dumps({k: v for k, v in tags.items() if k != "text"}), flush=True)

    print_table(combos)
    rec = recommend(combos)
    print("recommendation:", json.dumps(rec, indent=2))
    def mm(c: Combo) -> str:
        return f"{c.model}/{c.mode}" + ("" if c.effective_mode in ("", c.mode) else f"->{c.effective_mode}")

    works = sorted({mm(c) for c in combos if c.ok})
    fails = sorted({f"{mm(c)}: {c.error}" for c in combos if not c.ok and mm(c) not in works})
    bad_voices = sorted({f"{c.voice} ({EL_VOICES[c.voice]}): {c.error}" for c in combos if not c.ok and mm(c) in works})
    print("working model/mode combos:", works)
    if fails:
        print("failing model/mode combos:", fails)
    if bad_voices:
        print("voices failing on otherwise-working combos:", bad_voices)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sentence": args.text,
        "sample_rate": SAMPLE_RATE,
        "elapsed_s": round(time.perf_counter() - t_start, 1),
        "combos": [
            {**asdict(c), "median_ttfa_s": c.median("ttfa_s"), "median_total_s": c.median("total_s"), "ok": c.ok}
            for c in combos
        ],
        "cold_reference": cold,
        "v3_tag_test": tags,
        "working": works,
        "failing": fails,
        "failing_voices": bad_voices,
        "recommendation": rec,
    }
    args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.json} and {sum(1 for c in combos if c.wav)} wav files under samples/out/")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
