#!/usr/bin/env python
"""A caller's voice, not typed lines, through a speech-to-speech backend: the model's own turn
detection, transcription and tool calls on real audio, and the time from the end of the caller's
speech to her first audio.

    .venv/Scripts/python.exe bench/s2s_audio_check.py gemini:gemini-3.8-live
    .venv/Scripts/python.exe bench/s2s_audio_check.py openai:gpt-realtime-2.1-mini

The caller's lines are spoken by Kokoro (local, free) and streamed at real-time pace in 20 ms
chunks, followed by silence, the way a phone line sends them. Her audio is collected, not played.
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

LINES = [
    "Hi, this is Jenna with Great Lakes Freight Brokerage. Can you check on P O seven seven eight one two three four for me?",
    "Is she going to make the appointment?",
]


async def amain(backend: str) -> int:
    from eva.audio.mic import LinearResampler
    from eva.jobs.dispatch.world import build
    from eva.s2s import job_instructions, make_client
    from eva.s2s.driver import Call
    from eva.tts.kokoro_local import KokoroTTS

    build()
    provider, _, model = backend.partition(":")
    prompt, tools = job_instructions("dispatch")
    client = make_client(provider, model, prompt, tools)
    call = Call(client, tools)
    tts = KokoroTTS(voice="af_bella")
    await tts.warmup()
    await call.start()
    chunk_ms = 20
    n = client.in_rate * chunk_ms // 1000
    silence = np.zeros(n, np.int16).tobytes()
    results = []
    for line in LINES:
        pcm = np.frombuffer(b"".join([c async for c in tts.synthesize(line)]), np.int16)
        if client.in_rate != tts.sample_rate:
            pcm = np.clip(LinearResampler(tts.sample_rate, client.in_rate).process(pcm), -32768, 32767).astype(np.int16)
        t0 = time.perf_counter()
        for i in range(0, len(pcm), n):
            await client.send_audio(pcm[i : i + n].tobytes())
            await asyncio.sleep(max(0.0, t0 + (i + n) / client.in_rate - time.perf_counter()))
        speech_end = time.perf_counter()
        call.mark_start(speech_end)
        # keep the line open with silence until she has answered and gone quiet
        while time.perf_counter() - speech_end < 45:
            await client.send_audio(silence)
            await asyncio.sleep(chunk_ms / 1000)
            quiet = time.perf_counter() - call.last_event_t
            if call.turn_done.is_set() and call.pending_tools == 0 and quiet > 2.0:
                break
        t = call.turn
        heard = t.heard or "".join(call._heard_parts)
        results.append((line, heard, t.said.strip(), t.first_audio_s, [c["name"] for c in t.tools], t.cost))
        print(f"\ncaller: {line}\n  heard: {heard.strip()!r}\n  tools: {[c['name'] for c in t.tools]}\n"
              f"  eva ({t.first_audio_s if t.first_audio_s is None else round(t.first_audio_s, 2)} s after the caller stopped): {t.said.strip()[:300]}")
    await call.close()
    await tts.close()
    lat = [r[3] for r in results if r[3] is not None]
    print(f"\n{backend}: end of speech -> first audio {', '.join(f'{x:.2f}' for x in lat)} s; cost ${call.cost:.4f}; errors {call.errors or 'none'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("backend")
    return asyncio.run(amain(ap.parse_args().backend))


if __name__ == "__main__":
    sys.exit(main())
