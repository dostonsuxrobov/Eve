#!/usr/bin/env python
"""Eva CLI - talk to the voice agent.

    .venv/Scripts/python.exe run.py --preset cloud-fast --persona eva
    .venv/Scripts/python.exe run.py --list-devices
    .venv/Scripts/python.exe run.py --text          # type instead of talk (Eva still speaks)
    .venv/Scripts/python.exe run.py --once "hey eva, how's it going"   # one typed turn, then exit

Ctrl-C ends the session: the memory summariser runs on the transcript, then all
streams are closed.  Wear headphones for reliable barge-in; on speakers the echo
guard raises the VAD threshold while Eva talks, which also makes her a bit harder
to interrupt.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rich.console import Console  # noqa: E402
from rich.markup import escape  # noqa: E402

from eva.config import MEMORY_FILE, PRESETS, PipelineSettings, load_keys  # noqa: E402

console = Console(highlight=False)
log = logging.getLogger("eva.run")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="cloud-fast", choices=sorted(PRESETS), help="which stack to run")
    ap.add_argument("--persona", help="persona name (default: the preset's)")
    ap.add_argument("--voice", help="TTS voice override (ElevenLabs name/id or Kokoro voice)")
    ap.add_argument("--list-devices", action="store_true", help="print audio devices and exit")
    ap.add_argument("--input-device", type=int)
    ap.add_argument("--output-device", type=int)
    ap.add_argument("--no-barge-in", action="store_true", help="never interrupt Eva while she speaks")
    ap.add_argument("--text", action="store_true", help="typed input mode (still speaks)")
    ap.add_argument("--once", metavar="TEXT", help="text mode: speak the reply to this one line, then exit")
    ap.add_argument("--user-name")
    ap.add_argument("--mute-fillers", action="store_true", help="no 'hmm' while thinking")
    ap.add_argument("--no-greeting", action="store_true", help="don't have her say hello when the session starts")
    ap.add_argument("--lang", default="en", choices=["en", "ru"], help="language of the opening greeting and fillers (she follows you afterwards)")
    ap.add_argument("--debug", action="store_true", help="verbose logging + every pipeline event")
    return ap.parse_args(argv)


# ------------------------------------------------------------------ UI
class StatusPrinter:
    """One compact rich line per pipeline event."""

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug
        self.t0 = time.perf_counter()

    def __call__(self, name: str, data: dict[str, Any]) -> None:
        if name == "listening":
            console.print("[dim]listening...[/]")
        elif name == "speech_start" and data.get("state") == "listening":
            console.print("[dim]  (hearing you)[/]")
        elif name == "stt":
            console.print(f"[bold cyan]you:[/] {escape(data['text'])}  [dim]{data['latency_s']:.2f}s[/]")
        elif name == "stt_empty":
            console.print("[dim]  (nothing to transcribe)[/]")
        elif name == "filler":
            console.print(f"[dim]  (filler #{data['index']} at +{data['after_s']:.2f}s)[/]")
        elif name == "tool_call":
            console.print(f"[yellow]tool:[/] {escape(data['name'])}({escape(str(data['arguments']))})")
        elif name == "tool_result":
            console.print(f"[yellow]  -> [/]{escape(str(data['result']))}  [dim]{data['seconds']:.2f}s[/]")
        elif name == "barge_in":
            console.print(
                f"[magenta]interrupted[/] after {data['played_s']:.2f}s of audio "
                f"[dim](stop {data['stop_ms']:.1f} ms, {data['state_before']})[/]"
            )
        elif name == "utterance_merged":
            console.print("[dim]  (merged with what you said before)[/]")
        elif name == "pending_event":
            console.print(f"[yellow]event:[/] {escape(str(data['event']))}")
        elif name == "turn":
            b = data["breakdown"]
            flag = f" [magenta]{escape('[interrupted]')}[/]" if data["interrupted"] else ""
            console.print(f"[bold green]eva:[/] {escape(data['assistant_text'])}{flag}")

            def f(v: float | None) -> str:
                return "-" if v is None else f"{v:.2f}"

            console.print(
                f"[dim]  stt {f(b['stt'])} | ttft {f(b['llm_ttft'])} | ttfa {f(b['tts_ttfa'])} "
                f"| total {f(b['total'])} s{' | filler' if data.get('filler_played') else ''}[/]"
            )
            if data.get("error"):
                console.print(f"[red]  error: {escape(str(data['error']))}[/]")
            console.print("[dim]listening...[/]")
        elif name == "error":
            console.print(f"[red]error ({data.get('where')}): {escape(str(data.get('error')))}[/]")
        elif self.debug and name != "state":
            console.print(f"[dim]{time.perf_counter() - self.t0:7.2f} {name} {escape(str(data))}[/]")


def list_devices() -> None:
    import sounddevice as sd

    console.print(sd.query_devices())
    d = sd.default.device
    console.print(f"default input {d[0]}, default output {d[1]}")


async def stdin_lines() -> AsyncIterator[str]:
    """Yield typed lines without blocking the loop (daemon reader thread, so Ctrl-C
    and shutdown never wait on a stuck ``input()``)."""
    loop = asyncio.get_running_loop()
    q: asyncio.Queue[str | None] = asyncio.Queue()

    def reader() -> None:
        try:
            for line in sys.stdin:
                loop.call_soon_threadsafe(q.put_nowait, line.rstrip("\r\n"))
        except Exception:
            pass
        loop.call_soon_threadsafe(q.put_nowait, None)

    threading.Thread(target=reader, name="eva-stdin", daemon=True).start()
    while True:
        line = await q.get()
        if line is None:
            return
        yield line


# ---------------------------------------------------------------- session
async def amain(args: argparse.Namespace) -> int:
    from eva.factory import build_llm, build_stt, build_tts
    from eva.memory import Memory
    from eva.personas import load_persona, render
    from eva.pipeline import VoiceAgent
    from eva.tools import get_tools
    from eva.tools import tool_notes as _tool_notes

    preset = PRESETS[args.preset]
    settings: PipelineSettings = dataclasses.replace(preset.settings)
    if args.no_barge_in:
        settings.barge_in = False
    if args.input_device is not None:
        settings.input_device = args.input_device
    if args.output_device is not None:
        settings.output_device = args.output_device
    if args.mute_fillers:
        settings.filler_after_ms = 0
    keys = load_keys()

    console.print(f"[bold]Eva[/] preset [cyan]{preset.name}[/]: {preset.description}")
    tts_cfg = dict(preset.tts)
    if args.voice:
        tts_cfg["voice"] = args.voice
    stt = build_stt(preset.stt, keys)
    llm = build_llm(preset.llm, keys)
    tts = build_tts(tts_cfg, keys)

    persona = load_persona(args.persona or preset.persona)
    memory = Memory(MEMORY_FILE)
    memory.load()
    tools = get_tools()
    system_prompt = render(
        persona,
        supports_audio_tags=tts.supports_audio_tags,
        memory_text=memory.as_prompt_text(),
        now=datetime.now().strftime("%A %d %B %Y, %H:%M"),
        user_name=args.user_name or "",
        tool_notes=_tool_notes(tools),
        delivery_cues=bool(getattr(tts, "supports_cues", False)),
    )
    fillers = {} if args.mute_fillers else persona.fillers_by_lang()
    tool_hints = persona.tool_hints_by_lang()
    backchannels = persona.backchannels_by_lang() if settings.backchannels else {}

    # Open the audio devices BEFORE any network warmup: a missing/denied microphone
    # should fail fast without spending API calls or leaving warmup tasks dangling.
    from eva.audio.player import Player

    player = Player(tts.sample_rate, device=settings.output_device)
    player.start()
    mic = None
    segmenter = None
    frames = None
    if not args.text:
        from eva.audio.mic import Mic
        from eva.audio.vad import UtteranceSegmenter

        mic = Mic(device=settings.input_device)
        try:
            mic.start()
        except Exception as e:
            player.close()
            console.print(f"[red]{e}[/]")
            return 2
        frames = mic.frames()
        segmenter = UtteranceSegmenter(settings)

    t0 = time.perf_counter()
    try:
        with console.status("warming up stt / llm / tts..."):
            await asyncio.gather(stt.warmup(), llm.warmup(), tts.warmup())
    except BaseException:
        if mic is not None:
            mic.stop()
        player.close()
        await asyncio.gather(stt.close(), llm.close(), tts.close(), return_exceptions=True)
        raise
    console.print(f"[dim]warm in {time.perf_counter() - t0:.2f}s: {stt.name} + {llm.name} + {tts.name}[/]")

    agent = VoiceAgent(
        stt,
        llm,
        tts,
        system_prompt,
        tools,
        settings,
        frames=frames,
        segmenter=segmenter,
        player=player,
        fillers=fillers,
        tool_hints=tool_hints,
        backchannels=backchannels,
        on_event=StatusPrinter(debug=args.debug),
    )
    t0 = time.perf_counter()
    await agent.prepare()
    if hasattr(agent, "_select_lang"):
        agent._select_lang(args.lang)
    if fillers:
        n = sum(len(v) for v in fillers.values())
        console.print(f"[dim]{n} fillers pre-rendered in {time.perf_counter() - t0:.2f}s[/]")
    if not args.no_greeting and not args.once:
        # She opens the conversation (one short LLM turn) instead of sitting in silence.
        agent.pending_events.put_nowait({"type": "session_start", "user_name": args.user_name or "", "lang": args.lang})
        await agent.poll_pending_events()
    console.print("[dim]Ctrl-C to end the session" + (" | type and press Enter; an empty line interrupts her" if args.text else "; headphones recommended for barge-in") + "[/]")

    try:
        if args.once:
            m = await agent.say(args.once)
            await player.wait_until_done()
            lat = m.response_latency()
            console.print(f"[dim]one turn done: response latency {'-' if lat is None else f'{lat:.2f}'} s[/]")
        elif args.text:
            await text_session(agent)
        else:
            await agent.run()
    except asyncio.CancelledError:
        pass  # Ctrl-C: asyncio.run cancels the main task; fall through to the cleanup
    finally:
        console.print("\n[dim]ending session...[/]")
        try:
            if mic is not None:
                mic.stop()
        except Exception:
            pass
        try:
            await agent.close()  # stops the LLM keep-alive pings (run() does this itself)
        except Exception:
            pass
        try:
            player.stop()
        except Exception:
            pass
        if agent.messages:
            try:
                with console.status("updating memory..."):
                    await asyncio.wait_for(memory.update_from_transcript(llm, agent.messages), timeout=25)
                save = getattr(memory, "save", None)
                if save is not None:
                    save()
                console.print(f"[dim]memory saved to {MEMORY_FILE}[/]")
            except Exception as e:
                console.print(f"[red]memory update failed: {escape(repr(e))}[/]")
        try:
            player.close()
        except Exception:
            pass
        await asyncio.gather(stt.close(), llm.close(), tts.close(), return_exceptions=True)
        console.print("[dim]bye.[/]")
    return 0


async def text_session(agent: Any) -> None:
    """Typed input: each line is a turn; an empty line interrupts a reply in progress.
    Timer events are still delivered while idle; the session ends by itself once she has
    said goodbye (the end_conversation tool)."""
    ended = asyncio.Event()

    async def event_poller() -> None:
        while not ended.is_set():
            await asyncio.sleep(0.25)
            try:
                await agent.poll_pending_events()
            except Exception:
                log.exception("pending event failed")
            if getattr(agent, "end_requested", False):
                ended.set()

    poller = asyncio.create_task(event_poller())
    say_task: asyncio.Task[Any] | None = None
    lines = stdin_lines().__aiter__()
    try:
        console.print("[bold cyan]you:[/] ", end="")
        while not ended.is_set():
            next_line = asyncio.ensure_future(lines.__anext__())
            end_wait = asyncio.ensure_future(ended.wait())
            done, _ = await asyncio.wait({next_line, end_wait}, return_when=asyncio.FIRST_COMPLETED)
            end_wait.cancel()
            if next_line not in done:
                next_line.cancel()
                break
            try:
                line = next_line.result().strip()
            except StopAsyncIteration:
                break
            if say_task is not None and not say_task.done():
                await agent.interrupt("typed")
                await asyncio.wait({say_task})
            if line:
                say_task = asyncio.create_task(agent.say(line))
            console.print("[bold cyan]you:[/] ", end="")
        if say_task is not None:
            await asyncio.wait({say_task})
    finally:
        ended.set()
        poller.cancel()
        if say_task is not None and not say_task.done():
            await agent.interrupt("shutdown")
            await asyncio.wait({say_task})


def _enable_ctrl_c_on_windows() -> None:
    """A process started with CREATE_NEW_PROCESS_GROUP (some launchers, IDE runners,
    job-control shells) inherits "Ctrl-C ignored"; undo that so Ctrl-C reaches asyncio."""
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleCtrlHandler(None, False)
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.once:
        args.text = True
    _enable_ctrl_c_on_windows()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logging.getLogger("phonemizer").setLevel(logging.ERROR)  # Kokoro G2P "words count mismatch" noise
    if args.list_devices:
        list_devices()
        return 0
    try:
        # asyncio.run installs a SIGINT handler that cancels the main task (works on
        # Windows too); amain's finally block does the async cleanup before the
        # KeyboardInterrupt is re-raised here.
        return asyncio.run(amain(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
