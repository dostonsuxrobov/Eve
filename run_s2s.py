#!/usr/bin/env python
"""Talk to a speech-to-speech model (OpenAI Realtime, Gemini Live) on this laptop's mic and
speakers, as Red Oak Transport's dispatcher (or any job), with the same tools as Eva's loop.

    .venv/Scripts/python.exe run_s2s.py openai:gpt-realtime-2.1
    .venv/Scripts/python.exe run_s2s.py openai:gpt-realtime-2.1-mini --voice cedar
    .venv/Scripts/python.exe run_s2s.py gemini:gemini-3.8-live --voice Aoede
    .venv/Scripts/python.exe run_s2s.py gemini:gemini-3.8-live --text        # type your lines, she still speaks

Headphones: the model hears the room, and its own voice from the speakers would count as you
talking. Every turn prints what it heard, what she said, the tools she used, the time from the end
of your speech (this laptop's VAD) to her first audio, and what the turn cost. Ctrl-C ends.
The database resets with ``--fresh`` (python -m eva.jobs.dispatch.world); bookings otherwise persist.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.markup import escape  # noqa: E402

console = Console(highlight=False)


def printer(state: dict[str, Any]):
    def on_event(name: str, data: dict[str, Any]) -> None:
        if name == "text_out":
            if not state.get("speaking"):
                console.print("[bold magenta]eva:[/] ", end="")
                state["speaking"] = True
            console.print(escape(data["delta"]), end="")
        elif name == "text_in" and data["text"]:
            console.print(f"\n[bold cyan]you:[/] {escape(data['text'])}")
        elif name == "tool":
            args = ", ".join(f"{k}={v}" for k, v in data["args"].items())
            console.print(f"\n[dim]  -> {data['name']}({escape(args)})[/]", end="")
        elif name == "tool_result":
            console.print(f"[dim] {data['ms']} ms[/]")
        elif name in ("speech_started", "speech_stopped"):
            console.print(f"\n[dim]  ({'heard you start' if name == 'speech_started' else 'you stopped'})[/]", end="")
        elif name == "barge_in":
            console.print(f"\n[dim]  (you cut in after {data['played_ms'] / 1000:.1f} s of her audio)[/]")
        elif name == "turn_done":
            state["speaking"] = False
            lat = state.get("latency")
            cost = data.get("cost", 0.0)
            state["total"] = state.get("total", 0.0) + cost
            bits = [f"first audio {lat:.2f} s after you stopped" if lat is not None else None,
                    f"${cost:.4f} this turn, ${state['total']:.3f} so far"]
            console.print(f"\n[dim]  {' | '.join(b for b in bits if b)}[/]")
            state["latency"] = None
        elif name == "first_audio":
            if state.get("speech_end_t"):
                state["latency"] = time.perf_counter() - state["speech_end_t"]
                state["speech_end_t"] = None
        elif name == "error":
            console.print(f"\n[red]{escape(data['message'])}[/]")
    return on_event


async def amain(args: argparse.Namespace) -> int:
    from eva.audio.mic import LinearResampler, Mic
    from eva.audio.player import Player
    from eva.audio.vad import SileroVAD
    from eva.s2s import job_client
    from eva.s2s.driver import Call

    provider, _, model = args.backend.partition(":")
    if args.web:
        return await serve_phone(args, provider, model)
    if args.fresh:
        from eva.jobs.dispatch.world import build

        build()
    kw: dict[str, Any] = {}
    if args.voice:
        kw["voice"] = args.voice
    if provider == "openai":
        # a 500 ms silence endpoint like Eva's loop; OpenAI's semantic VAD (--eagerness) waited up to 4 s
        # to be sure the caller was done: 5.0-5.3 s answers on the spoken dispatch calls (2026-09-27)
        kw.update({"turn_detection": "semantic_vad", "eagerness": args.eagerness} if args.eagerness else {"turn_detection": "server_vad"})
    client, tools = job_client(provider, model, args.job, args.user_name, **kw)
    player = Player(client.out_rate)
    player.start()
    state: dict[str, Any] = {}
    log_dir = ROOT / "bench" / "out" / "sessions"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{args.backend.replace(':', '_').replace('+', '_')}.jsonl"
    log_file = open(log_path, "a", encoding="utf-8")  # noqa: SIM115 - closed at the end of the call
    t_log = time.perf_counter()

    def log(ev: dict[str, Any]) -> None:  # every event but the audio bytes, for "why didn't she ...?"
        row = {"t": round(time.perf_counter() - t_log, 3), **{k: v for k, v in ev.items() if k != "pcm"}}
        if ev["type"] == "audio":
            row["bytes"] = len(ev["pcm"])
        log_file.write(json.dumps(row, default=str) + "\n")

    def on_event(name: str, data: dict[str, Any]) -> None:
        show(name, data)
        if name in ("tool", "tool_result", "barge_in", "error", "turn_done"):
            log({"type": f"call.{name}", **{k: v for k, v in data.items() if k != "turn"}})

    show = printer(state)
    call = Call(client, tools, player=player, on_event=on_event, log=log)
    console.print(f"[bold]{client.name}[/] as the {args.job} desk. Connecting...")
    await call.start()
    console.print("[dim]connected. Headphones on; Ctrl-C to end.[/]")
    # she answers the phone first
    greet = "[The phone line just connected. Answer it the way a dispatcher does: one short sentence with the company and your name.]"
    await client.send_text(greet)

    mic: Mic | None = None
    try:
        if args.text:
            loop = asyncio.get_running_loop()
            while not call.closed and not call.ended:
                line = await loop.run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                if line.strip():
                    state["speech_end_t"] = time.perf_counter()
                    call.mark_start()
                    await client.send_text(line.strip())
        else:
            mic = Mic(device=args.input_device, sample_rate=16000, frame_ms=32)  # 512 samples: one Silero window per frame
            vad = SileroVAD()
            up = LinearResampler(16000, client.in_rate) if client.in_rate != 16000 else None
            speaking, silence_ms = False, 0
            levels: list[float] = []
            async for frame in mic.frames():
                if len(levels) < 94:  # the first ~3 s: is the mic hearing anything at all?
                    levels.append(float(np.sqrt(np.mean(frame.astype(np.float32) ** 2))))
                    if len(levels) == 94:
                        db = lambda x: 20 * np.log10(max(x, 1.0) / 32768)  # noqa: E731
                        loud = db(max(levels))
                        console.print(f"\n[dim]  mic ({mic.native_rate} Hz): average {db(float(np.mean(levels))):.0f} dBFS, "
                                      f"loudest {loud:.0f} dBFS over the first 3 s[/]")
                        if loud < -55:
                            console.print("[yellow]  the mic looks silent: pick another with --input-device "
                                          "(python run.py --list-devices)[/]")
                if call.closed or call.ended:
                    break
                p = vad(frame)
                if p > 0.6:
                    speaking, silence_ms = True, 0
                elif speaking:
                    silence_ms += 32
                    if silence_ms >= 200:  # our own end-of-speech mark, only for timing
                        speaking = False
                        state["speech_end_t"] = time.perf_counter() - 0.2
                        call.mark_start(state["speech_end_t"])
                pcm = frame if up is None else np.clip(up.process(frame), -32768, 32767).astype(np.int16)
                await client.send_audio(pcm.astype("<i2").tobytes())
        if call.ended:
            await asyncio.sleep(max(0.5, player.buffered_seconds + 0.3))
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        if mic is not None:
            mic.stop()
        await call.close()
        player.close()
        log_file.close()
        console.print(f"[dim]every event of this call: {log_path}[/]")
        console.print(f"\n[bold]session cost ${call.cost:.3f}[/] over {len(call.turns)} turns ({client.name})")
    return 0


async def serve_phone(args: argparse.Namespace, provider: str, model: str) -> int:
    """The phone page (https://<laptop-ip>:8443) talking to this model; tools run here."""
    from eva.s2s import job_client
    from eva.s2s.driver import Call
    from eva.web.s2s_server import serve_s2s

    kw: dict[str, Any] = {"voice": args.voice} if args.voice else {}
    if provider == "openai":
        kw.update({"turn_detection": "semantic_vad", "eagerness": args.eagerness} if args.eagerness else {"turn_detection": "server_vad"})
    state: dict[str, Any] = {}
    show = printer(state)

    def on_console(name: str, data: dict[str, Any]) -> None:
        if name == "web_ready":
            console.print(f"\n[bold green]On your phone (same Wi-Fi), open:[/] [bold]{data['url']}[/]")
            console.print("[dim]  the certificate is this laptop's own: tap 'Show details' / 'Advanced' and continue to the site; "
                          "then Start and allow the microphone. Ctrl-C here ends it.[/]")
        elif name == "web_hello":
            console.print(f"\n[green]phone connected[/] [dim]{escape(data.get('ua', ''))}[/]")
        elif name == "call_over":
            console.print(f"\n[bold]call over: ${data['cost']:.3f}[/] [dim]({data['turns']} turns; every event: {data['log']})[/]")
        else:
            show(name, data)

    def make_call(player: Any, on_event: Any, log: Any) -> tuple[Any, Any]:
        if args.fresh:
            from eva.jobs.dispatch.world import build

            build()
        client, tools = job_client(provider, model, args.job, args.user_name, **kw)
        return Call(client, tools, player=player, on_event=on_event, log=log), client

    log_dir = ROOT / "bench" / "out" / "sessions"
    log_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]{args.backend}[/] as the {args.job} desk, for the phone.")
    try:
        await serve_s2s(make_call, printer=on_console, name=args.backend.replace(":", "_").replace("+", "_"),
                        log_dir=log_dir, port=args.port, tls=not args.no_tls)
    except asyncio.CancelledError:
        pass
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("backend", help="openai:gpt-realtime-2.1 | openai-live:gpt-live-1 | gemini:gemini-3.8-live | ... (dispatch.py --list)")
    ap.add_argument("--job", default="dispatch")
    ap.add_argument("--voice", help="OpenAI: marin, cedar, ...; Gemini: Aoede, Kore, Puck, ...")
    ap.add_argument("--eagerness", choices=["low", "medium", "high", "auto"], help="OpenAI: use semantic VAD with this eagerness instead of a 500 ms silence endpoint")
    ap.add_argument("--text", action="store_true", help="type your lines instead of talking")
    ap.add_argument("--fresh", action="store_true", help="rebuild the dispatch world first")
    ap.add_argument("--user-name", default="Doston")
    ap.add_argument("--input-device", type=int, help="microphone index (python run.py --list-devices)")
    ap.add_argument("--web", action="store_true", help="serve the phone page instead of this laptop's mic: https://<laptop-ip>:8443")
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--no-tls", action="store_true", help="plain http (browsers then allow the mic only on localhost)")
    args = ap.parse_args()
    try:
        return asyncio.run(amain(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
