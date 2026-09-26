#!/usr/bin/env python
"""Listening set: the same emotional lines in every voice, to judge by ear.

    .venv/Scripts/python.exe bench/voices.py                         # every voice in eva/config.py
    .venv/Scripts/python.exe bench/voices.py --voices orpheus-tara,chatterbox-turbo

Each line goes through the exact path the loop uses (the voice's TTS class, the delivery
cue and inline sounds as the brain would write them), and is saved as
bench/out/voices/<line>_<voice>.wav with an index, listening.md, that has the timings:
time to first audio, whole clip, and real-time factor. Lines are in Eva's generic tags
([laughs], [sighs]); each voice renders the sounds it can and drops the rest.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from eva.config import VOICES  # noqa: E402
from eva.delivery import extract_cue  # noqa: E402
from eva.factory import build_tts  # noqa: E402
from eva.gpu import free_ollama, gpu_used_mib  # noqa: E402
from eva.llm.sanitize import clean_for_tts  # noqa: E402

OUT = ROOT / "bench" / "out" / "voices"

# (id, line as the brain would write it: an optional leading cue, generic sounds inline)
LINES: list[tuple[str, str]] = [
    ("warm", "[warm] Hey, it's really good to hear your voice. How was today?"),
    ("excited", "[excited] Wait, you got the job? That's amazing! I knew it."),
    ("sad", "[sad] Oh no. I'm really sorry. That sounds like such a heavy day."),
    ("teasing", "[teasing] [laughs] Okay, you did not just burn the pasta again."),
    ("sigh", "[thoughtful] [sighs] Yeah... some days are just like that."),
    ("serious", "[serious] Call the emergency vet now. I'm right here with you."),
]


async def render(voice: str) -> list[dict]:
    tts = build_tts(VOICES[voice])
    t0 = time.perf_counter()
    await tts.warmup()
    load_s = time.perf_counter() - t0
    rows = []
    for line_id, line in LINES:
        cue, body = extract_cue(line)
        text = clean_for_tts(body, tts.supports_audio_tags)
        t0 = time.perf_counter()
        first = None
        parts: list[bytes] = []
        stream = tts.synthesize(text, cue=cue) if getattr(tts, "supports_cues", False) else tts.synthesize(text)
        async for pcm in stream:
            if first is None:
                first = time.perf_counter() - t0
            parts.append(pcm)
        total = time.perf_counter() - t0
        audio = np.frombuffer(b"".join(parts), dtype="<i2")
        path = OUT / f"{line_id}_{voice}.wav"
        sf.write(str(path), audio, tts.sample_rate, subtype="PCM_16")
        dur = len(audio) / tts.sample_rate
        rows.append({"voice": voice, "line": line_id, "text": line, "sent": text, "cue": cue, "first_s": first,
                     "total_s": total, "audio_s": dur, "file": path.name})
        print(f"  {voice:16} {line_id:8} first {first or 0:5.2f} s  {dur:4.1f} s of audio in {total:5.2f} s", flush=True)
    rows.append({"voice": voice, "load_s": load_s, "gpu_mib": gpu_used_mib()})
    await tts.close()
    return rows


async def amain(voices: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    freed = free_ollama(set())
    if freed:
        print(f"freed the GPU: unloaded {', '.join(freed)}")
    results = []
    for voice in voices:
        print(f"=== {voice}", flush=True)
        results += await render(voice)
    lines = ["# Listening set", "", f"{time.strftime('%Y-%m-%d %H:%M')}. The same lines in every voice; files in this folder.",
             "", "| line | text |", "|---|---|"]
    lines += [f"| {lid} | {txt} |" for lid, txt in LINES]
    lines += ["", "| voice | line | first audio | clip | made in | x real time | file |", "|---|---|---|---|---|---|---|"]
    for r in results:
        if "line" in r:
            lines.append(f"| {r['voice']} | {r['line']} | {r['first_s'] or 0:.2f} s | {r['audio_s']:.1f} s | {r['total_s']:.2f} s | "
                         f"{r['audio_s'] / r['total_s'] if r['total_s'] else 0:.2f} | {r['file']} |")
    lines += ["", "| voice | load | GPU in use after load |", "|---|---|---|"]
    lines += [f"| {r['voice']} | {r['load_s']:.1f} s | {r['gpu_mib']} MiB |" for r in results if "load_s" in r]
    (OUT / "listening.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'listening.md'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voices", default="", help="comma list of voices (default: all)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    voices = [v.strip() for v in args.voices.split(",") if v.strip()] or list(VOICES)
    asyncio.run(amain(voices))
    return 0


if __name__ == "__main__":
    sys.exit(main())
