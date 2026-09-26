#!/usr/bin/env python
"""Eva CLI - talk to the voice agent. Everything runs on this laptop.

    .venv/Scripts/python.exe run.py --user-name Doston            # pick a brain and a voice from the lists
    .venv/Scripts/python.exe run.py --brain minicpm1b --voice orpheus-tara --user-name Doston
    .venv/Scripts/python.exe run.py --list                        # every brain and voice
    .venv/Scripts/python.exe run.py --web --tls                   # talk from your phone: https://<laptop-ip>:8443
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
from pathlib import Path
from typing import Any, AsyncIterator

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rich.console import Console  # noqa: E402
from rich.markup import escape  # noqa: E402

from eva.config import BRAINS, DEFAULT_BRAIN, DEFAULT_VOICE, MEMORY_FILE, VOICES, PipelineSettings, make_preset  # noqa: E402
from eva.lang import DEFAULT_MODE as DEFAULT_LANG_MODE, modes as lang_modes  # noqa: E402

console = Console(highlight=False)
log = logging.getLogger("eva.run")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brain", choices=list(BRAINS), help="which brain (no --brain and no --voice: pick from a list)")
    ap.add_argument("--voice", choices=list(VOICES), help="which voice")
    ap.add_argument("--list", action="store_true", help="print every brain and voice and exit")
    ap.add_argument("--persona", help="persona name (default: the brain's, eva or eva_small)")
    ap.add_argument("--list-devices", action="store_true", help="print audio devices and exit")
    ap.add_argument("--input-device", type=int)
    ap.add_argument("--output-device", type=int)
    ap.add_argument("--no-barge-in", action="store_true", help="never interrupt Eva while she speaks")
    ap.add_argument("--text", action="store_true", help="typed input mode (still speaks)")
    ap.add_argument("--once", metavar="TEXT", help="text mode: speak the reply to this one line, then exit")
    ap.add_argument("--user-name")
    ap.add_argument("--mute-fillers", action="store_true", help="no 'hmm' while thinking")
    ap.add_argument("--no-greeting", action="store_true", help="don't have her say hello when the session starts")
    ap.add_argument("--lang", default=DEFAULT_LANG_MODE, choices=lang_modes(), help=f"default {DEFAULT_LANG_MODE}: English only while Russian is frozen (CLAUDE.md); auto: follow you between languages; ru: locked to Russian")
    ap.add_argument("--web", action="store_true", help="serve the phone/browser client instead of using this machine's mic and speakers")
    ap.add_argument("--port", type=int, help="port for --web (default 8080, or 8443 with --tls)")
    ap.add_argument("--tls", action="store_true", help="--web over https with a self-signed certificate (browsers need it for the mic)")
    ap.add_argument("--echo-gates", choices=["auto", "on", "off"], default="auto",
                    help="the speaker-echo defences (words-confirmed barge-in, echo detector, echo gate): auto = on for this machine's mic, off for --web")
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
        elif name == "stt_phantom":
            console.print(f"[dim]  (ignored as noise: {escape(data['reason'])})[/]")
        elif name == "stt_hesitation":
            console.print("[dim]  (just a hesitation, waiting for you)[/]")
        elif name == "stt_echo":
            console.print("[dim]  (that was my own voice through the mic, ignored)[/]")
        elif name == "barge_in_echo":
            console.print(f"[dim]  (hearing myself through the speakers, not stopping; echo {data.get('echo_ratio')})[/]")
        elif name == "barge_in_ignored":
            reason = data.get("reason", "")
            if reason != "blip":
                console.print(f"[dim]  (not a real interruption: {escape(str(reason))})[/]")
        elif name == "echo_storm":
            console.print(
                "[yellow]the speakers keep reaching the mic: for the next "
                f"{data['hold_s']:.0f} s only a full transcript can interrupt me. Headphones, or lower the volume.[/]"
            )
        elif name == "web_ready":
            console.print(f"[bold green]open on your phone:[/] [bold]{data['url']}[/]   [dim](this machine: {data['local']})[/]")
            console.print(f"[dim]echo gates {'on' if data.get('echo_gates') else 'off (the phone cancels its own echo)'}[/]")
            if data.get("tls"):
                console.print("[dim]self-signed certificate: accept the browser warning once (Safari: Show details -> visit this website);"
                              " Chrome on Android: install " + data["url"] + "/cert.pem or use chrome://flags/#unsafely-treat-insecure-origin-as-secure[/]")
            console.print("[dim]Ctrl-C stops the server[/]")
        elif name == "web_client":
            console.print(f"[dim]phone {data['state']}: {escape(str(data['peer']))}[/]")
        elif name == "phone_stats":
            console.print(
                f"[dim]phone: buffer {data.get('buffer_ms')} ms (hold {data.get('hold_ms')}) | drops {data.get('drops')} | "
                f"rtt {data.get('rtt_ms')} ms | echo cancel {'on' if data.get('aec', True) else 'off'}[/]"
            )
        elif name == "web_hello":
            console.print(f"[dim]phone: echo cancel {'on' if data['aec'] else 'off'}, buffer {data['prebuffer_s'] * 1000:.0f} ms, echo gates {'on' if data['gates'] else 'off'}[/]")
        elif name == "memory_saved":
            console.print(f"[dim]memory saved ({data['facts']} facts)[/]")
        elif name == "stt_incomplete":
            console.print(f"[dim]  (sounds unfinished, giving you {data['grace_ms']} ms)[/]")
        elif name == "utterance_carried":
            console.print("[dim]  (holding that, go on...)[/]")
        elif name == "stt_commit_timeout":
            console.print(f"[dim]  (transcription slow: no answer in {data['after_s']:.1f}s, retrying in batch)[/]")
        elif name == "stt_fallback":
            console.print(f"[dim]  (batch transcription took {data['latency_s']:.2f}s)[/]")
        elif name == "language":
            console.print(f"[dim]  (switching to {data['lang']})[/]")
        elif name == "stt_foreign":
            console.print(f"[dim]  (heard as {escape(str(data['lang']))}, not one of ours: probably misheard, staying in {data['kept']})[/]")
        elif name == "failover":
            if data.get("first", True):
                console.print(f"[yellow]{data['kind']}: {escape(data['from'])} not answering ({escape(data['reason'])}); using {escape(data['to'])}[/]")
        elif name == "recovered":
            console.print(f"[green]{data['kind']}: {escape(data['name'])} is back[/]")
        elif name == "filler":
            console.print(f"[dim]  (filler #{data['index']} at +{data['after_s']:.2f}s)[/]")
        elif name == "tools_gated":
            offered = ", ".join(data["offered"]) or "none"
            console.print(f"[dim]  (tools offered for this line: {escape(offered)})[/]")
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


def print_variants() -> None:
    console.print("[bold]brains[/] (--brain)")
    for i, (key, b) in enumerate(BRAINS.items(), 1):
        console.print(f"  {i}. [cyan]{key:10}[/] {escape(b['label'])}{'  [dim](default)[/]' if key == DEFAULT_BRAIN else ''}")
    console.print("[bold]voices[/] (--voice)")
    for i, (key, v) in enumerate(VOICES.items(), 1):
        console.print(f"  {i}. [cyan]{key:16}[/] {escape(v['label'])}{'  [dim](default)[/]' if key == DEFAULT_VOICE else ''}")


def pick_variant() -> tuple[str, str]:
    """Ask for a brain and a voice by number (Enter keeps the default)."""
    print_variants()

    def ask(what: str, keys: list[str], default: str) -> str:
        while True:
            raw = input(f"{what} [number, Enter = {default}]: ").strip()
            if not raw:
                return default
            if raw.isdigit() and 1 <= int(raw) <= len(keys):
                return keys[int(raw) - 1]
            if raw in keys:
                return raw

    return ask("brain", list(BRAINS), DEFAULT_BRAIN), ask("voice", list(VOICES), DEFAULT_VOICE)


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
    from eva.pipeline import VoiceAgent
    from eva.session import build_session

    preset = make_preset(args.brain or DEFAULT_BRAIN, args.voice or DEFAULT_VOICE)
    settings: PipelineSettings = dataclasses.replace(preset.settings)
    if args.no_barge_in:
        settings.barge_in = False
    if args.input_device is not None:
        settings.input_device = args.input_device
    if args.output_device is not None:
        settings.output_device = args.output_device
    if args.mute_fillers:
        settings.filler_after_ms = 0
    if args.echo_gates == "off":
        settings.barge_in_confirm, settings.echo_detector, settings.self_echo_gate = "vad", False, False

    console.print(f"[bold]Eva[/] [cyan]{preset.name}[/]: {preset.description}")
    printer = StatusPrinter(debug=args.debug)
    session = build_session(
        preset, lang=args.lang, persona=args.persona, user_name=args.user_name or "",
        mute_fillers=args.mute_fillers, on_event=printer,
    )
    stt, llm, tts = session.stt, session.llm, session.tts
    plan, memory = session.plan, session.memory
    if session.dropped_facts:
        # A misheard "my name is ..." or the assistant saying the name once used to end
        # up in memory as "The user's name is Doster." / "Has a friend named Doston."
        console.print(f"[dim]memory: dropped {len(session.dropped_facts)} stale name fact(s): {escape('; '.join(session.dropped_facts))}[/]")
    if plan.locked and session.persona.lang != plan.mode:
        console.print(f"[dim]no {plan.mode} version of persona {session.persona.name!r}; using the English prompt with a locked-language rule[/]")
    if plan.mode == DEFAULT_LANG_MODE == "en":
        console.print("[dim]English only: Russian is frozen until English is done (--lang auto brings it back)[/]")
    console.print(f"[dim]language: {plan.mode} | persona: {session.persona.name} ({session.persona.lang}) | brain: {llm.name}[/]")
    if args.web:
        from eva.web.server import serve_web

        port = args.port or (8443 if args.tls else 8080)
        try:
            await serve_web(
                session, settings, port=port, tls=args.tls, user_name=args.user_name or "",
                greeting=not args.no_greeting, echo_gates={"auto": None, "on": True, "off": False}[args.echo_gates],
                printer=printer,
            )
        except asyncio.CancelledError:
            pass
        finally:
            console.print("\n[dim]stopping the server...[/]")
            await asyncio.gather(stt.close(), llm.close(), tts.close(), return_exceptions=True)
            console.print("[dim]bye.[/]")
        return 0
    if not args.text:
        console.print(
            f"[dim]barge-in: {settings.barge_in_confirm} (while I talk, your words must show up in the transcript"
            f"{' and the echo detector must not hear the speakers' if settings.echo_detector else ''})[/]"
        )

    # Open the audio devices BEFORE any network warmup: a missing/denied microphone
    # should fail fast without spending API calls or leaving warmup tasks dangling.
    from eva.audio.player import Player

    player = Player(tts.sample_rate, device=settings.output_device)
    player.room_tone_dbfs = settings.room_tone_dbfs
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

    from eva.gpu import GPU_TOTAL_MIB, GPU_WARN_MIB, free_ollama, gpu_used_mib

    freed = free_ollama(session.ollama_models)
    if freed:
        console.print(f"[dim]freed the GPU for this variant: unloaded {escape(', '.join(freed))}[/]")
    t0 = time.perf_counter()
    try:
        with console.status("warming up stt / llm / tts (a voice's first run downloads it)..."):
            await asyncio.gather(stt.warmup(), llm.warmup(), tts.warmup())
    except BaseException:
        if mic is not None:
            mic.stop()
        player.close()
        await asyncio.gather(stt.close(), llm.close(), tts.close(), return_exceptions=True)
        raise
    console.print(f"[dim]warm in {time.perf_counter() - t0:.2f}s: {stt.name} + {llm.name} + {tts.name}[/]")
    used = gpu_used_mib()
    if used is not None:
        console.print(f"[dim]GPU: {used} of {GPU_TOTAL_MIB} MiB in use[/]")
        if used > GPU_WARN_MIB:
            console.print("[yellow]the GPU is nearly full: past it Windows spills into system RAM and everything slows"
                          " down about 20x. A smaller brain or voice (run.py --list) keeps it fast.[/]")

    agent = VoiceAgent(
        stt,
        llm,
        tts,
        session.system_prompt,
        session.tools,
        settings,
        frames=frames,
        segmenter=segmenter,
        player=player,
        fillers=session.fillers,
        tool_hints=session.tool_hints,
        backchannels=session.backchannels,
        on_event=printer,
        languages=plan.codes,
        tool_filter=session.tool_filter,
    )
    t0 = time.perf_counter()
    await agent.prepare()
    agent._select_lang(plan.primary.code)
    if session.fillers:
        n = sum(len(v) for v in session.fillers.values())
        console.print(f"[dim]{n} fillers pre-rendered in {time.perf_counter() - t0:.2f}s[/]")
    if not args.no_greeting and not args.once:
        # She opens the conversation (one short LLM turn) instead of sitting in silence.
        agent.pending_events.put_nowait(session.greeting_event(args.user_name or ""))
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
                    await asyncio.wait_for(
                        memory.update_from_transcript(llm, agent.messages, user_name=args.user_name or ""),
                        timeout=25,
                    )
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
    if args.list:
        print_variants()
        return 0
    if args.brain is None and args.voice is None and not args.once and sys.stdin.isatty():
        args.brain, args.voice = pick_variant()
    try:
        # asyncio.run installs a SIGINT handler that cancels the main task (works on
        # Windows too); amain's finally block does the async cleanup before the
        # KeyboardInterrupt is re-raised here.
        return asyncio.run(amain(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
