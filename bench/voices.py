#!/usr/bin/env python
"""Hear every voice a speech-to-speech model offers, on the same dispatcher line, and pick by ear.

    .venv/Scripts/python.exe bench/voices.py render gemini        # 30 Gemini Live voices (~$0.10)
    .venv/Scripts/python.exe bench/voices.py render openai        # 10 OpenAI realtime voices (~$0.15)
    .venv/Scripts/python.exe bench/voices.py render live          # 12 GPT-Live voices (~$0.20)
    .venv/Scripts/python.exe bench/voices.py play gemini          # play them one by one, name first
    .venv/Scripts/python.exe bench/voices.py play live --only gleam,delta

Clips go to samples/out/voices/<provider>/<voice>.wav (24 kHz). Then talk with the one you like:
``dispatch.py gemini-live --voice Puck``, ``dispatch.py openai-live --voice delta``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

OUT = ROOT / "samples" / "out" / "voices"
LINE = ("Red Oak dispatch, this is Eva. Yeah, I've got your load right here. She's about forty minutes out, running a "
        "little behind on I seventy, so I've already asked the receiver to hold the slot. I'll call you the second she checks in.")
SAY = "You are a voice actor reading one line for a phone call. Read the line you're given aloud, word for word, in a " \
      "natural, relaxed, friendly voice, like an experienced dispatcher on a busy day. Say nothing else."

VOICES = {
    "gemini": ("gemini", "gemini-3.8-live", "Zephyr Puck Charon Kore Fenrir Leda Orus Aoede Callirrhoe Autonoe Enceladus Iapetus "
               "Umbriel Algieba Despina Erinome Algenib Rasalgethi Laomedeia Achernar Alnilam Schedar Gacrux Pulcherrima "
               "Achird Zubenelgenubi Vindemiatrix Sadachbia Sadaltager Sulafat".split()),
    "openai": ("openai", "gpt-realtime-2.1", "marin cedar alloy ash ballad coral echo sage shimmer verse".split()),
    "live": ("openai-live", "gpt-live-1", "gleam meridian delta cinder willow stone vesper quartz ripple beacon marin cedar".split()),
}


async def render_one(provider: str, model: str, voice: str) -> tuple[bytes, float]:
    from eva.s2s import make_client

    kw = {"turn_detection": None} if provider == "openai" else {}
    c = make_client(provider, model, SAY, [], voice=voice, **kw)
    await c.connect()
    n = c.in_rate // 50
    silence = bytes(2 * n)
    if provider == "openai-live":
        await c.send_text(f"Say this now, word for word, then stop and listen: {LINE}")
    else:
        await c.send_text(LINE)
    audio = bytearray()
    t0 = time.perf_counter()
    last = t0
    done = False
    while time.perf_counter() - t0 < 40:
        await c.send_audio(silence)  # the line stays open, as on a call
        await asyncio.sleep(0.02)
        while not c.events.empty():
            ev = c.events.get_nowait()
            if ev["type"] == "audio":
                audio += ev["pcm"]
                last = time.perf_counter()
            elif ev["type"] == "turn_done":
                done = True
            elif ev["type"] == "error":
                print(f"  {voice}: {ev['message']}")
        if audio and (done or time.perf_counter() - last > 2.5):
            break
    cost = getattr(c, "cost", 0.0)
    await c.close()
    return bytes(audio), cost


def trim(a: np.ndarray, sr: int = 24000) -> np.ndarray:
    """Cut the silence around the line: GPT-Live streams audio all the time, silence included."""
    hop = sr // 50
    n = len(a) // hop
    if n == 0:
        return a[:0]
    rms = np.sqrt((a[: n * hop].astype(np.float32).reshape(n, hop) ** 2).mean(axis=1))
    voiced = np.nonzero(rms > 184)[0]  # -45 dBFS
    if voiced.size == 0:
        return a[:0]
    return a[max(0, voiced[0] - 5) * hop : min(len(a), (voiced[-1] + 10) * hop)]


async def render(which: str, only: list[str] | None) -> None:
    import soundfile as sf

    provider, model, voices = VOICES[which]
    folder = OUT / which
    folder.mkdir(parents=True, exist_ok=True)
    total = 0.0
    for v in voices:
        if only and v not in only:
            continue
        try:
            pcm, cost = await render_one(provider, model, v)
        except Exception as e:  # noqa: BLE001 - one voice failing shouldn't stop the others
            print(f"  {v}: failed: {e!r}")
            continue
        total += cost
        a = trim(np.frombuffer(pcm, np.int16))
        if not len(a):
            print(f"  {v}: silent (no audio above -45 dBFS)")
            continue
        sf.write(str(folder / f"{v}.wav"), a, 24000, subtype="PCM_16")
        print(f"  {which}/{v}: {len(a) / 24000:.1f} s", flush=True)
    (folder / "line.txt").write_text(LINE, encoding="utf-8")
    print(f"{which}: {folder} (${total:.3f})")


def play(which: str, only: list[str] | None) -> None:
    import sounddevice as sd
    import soundfile as sf

    for f in sorted((OUT / which).glob("*.wav"), key=lambda p: VOICES[which][2].index(p.stem) if p.stem in VOICES[which][2] else 99):
        if only and f.stem not in only:
            continue
        a, sr = sf.read(str(f), dtype="int16")
        print(f"{f.stem} ({len(a) / sr:.1f} s)", flush=True)
        sd.play(a, sr)
        sd.wait()
        time.sleep(0.6)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["render", "play"])
    ap.add_argument("provider", choices=list(VOICES))
    ap.add_argument("--only", help="comma-separated voice names")
    a = ap.parse_args()
    only = a.only.split(",") if a.only else None
    if a.action == "render":
        asyncio.run(render(a.provider, only))
    else:
        play(a.provider, only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
