"""End-to-end simulator for the Eva pipeline (real providers, sample utterances).

Builds a preset through ``eva.session.build_session`` (so it composes exactly what
``run.py`` runs: providers with their local fallbacks, the language plan, the
persona, the memory), the real Silero segmenter, and either the real ``Player``
(``--speakers``) or a ``MockPlayer`` (``--silent``), then feeds the sample
utterances as mic frames at controlled moments. Prints a per-turn table and writes
``bench/out/e2e_<preset>.json``.

``--outage llm,stt,tts`` points the named cloud providers at a dead host so the
``eva.failover`` switch to the local stack can be exercised end to end.

The behavioural test suite on doubles lives in ``tests/`` (``python -m pytest tests``).

Examples::

    .venv/Scripts/python.exe bench/e2e_sim.py --utterances samples/user_hello.wav,samples/user_rough_day.wav --gap 6 --silent
    .venv/Scripts/python.exe bench/e2e_sim.py --utterances samples/user_hello.wav,samples/user_task.wav --barge-in-at 1.2 --speakers
    .venv/Scripts/python.exe bench/e2e_sim.py --lang ru --utterances samples/user_ru_rough_day.wav
    .venv/Scripts/python.exe bench/e2e_sim.py --outage llm,stt,tts --utterances samples/user_hello.wav
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, AsyncIterator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.markup import escape  # noqa: E402
from rich.table import Table  # noqa: E402

from eva.config import BRAINS, DEFAULT_PRESET, PRESETS, SAMPLES_DIR, load_keys  # noqa: E402
from eva.lang import modes as lang_modes  # noqa: E402
from eva.interfaces import MIC_SAMPLE_RATE  # noqa: E402
from eva.mocks import EventLog, MockPlayer, turn_rows as _turn_rows  # noqa: E402
from eva.pipeline import VoiceAgent  # noqa: E402

console = Console()
OUT_DIR = ROOT / "bench" / "out"
DEAD_HOST = "https://127.0.0.1:9"  # nothing listens here: connection refused at once


def print_turn_table(title: str, rows: list[dict[str, Any]]) -> None:
    table = Table(title=title, show_lines=True)
    for col in ("#", "user", "assistant", "stt", "llm_ttft", "tts_ttfa", "total", "int"):
        table.add_column(col, overflow="fold", max_width=None if col in ("user", "assistant") else 9)

    def f(v: float | None) -> str:
        return "-" if v is None else f"{v:.3f}"

    for i, r in enumerate(rows, 1):
        table.add_row(
            str(i),
            escape(r["user_text"]),
            escape(r["assistant_text"]),
            f(r["stt"]),
            f(r["llm_ttft"]),
            f(r["tts_ttfa"]),
            f(r["total"]),
            "yes" if r["interrupted"] else "",
        )
    console.print(table)


def simulate_outage(kinds: set[str]) -> None:
    """Point the cloud providers named in ``kinds`` at a dead host (before building)."""
    if "llm" in kinds:
        import eva.factory as _f

        _f.CEREBRAS_BASE_URL = DEAD_HOST
    if "tts" in kinds:
        import eva.tts.elevenlabs as _el

        _el.HTTP_BASE = DEAD_HOST
    if "stt" in kinds:
        import eva.stt.elevenlabs_realtime as _rt
        import eva.stt.elevenlabs_scribe as _sc

        _rt.REALTIME_URL = "wss://127.0.0.1:9/v1/speech-to-text/realtime"
        _sc.STT_URL = f"{DEAD_HOST}/v1/speech-to-text"



class ScenarioMic:
    """Mic stand-in that plays wav utterances at controlled moments.

    Utterance 0 starts 0.5 s in.  With ``barge_in_at`` set, utterance 1 starts that
    many seconds after the agent's first audio.  Every other utterance starts ``gap``
    seconds after the agent finished answering the previous one (a turn or a
    cancelled/merged turn).  Silence frames fill the rest so the VAD sees a
    continuous stream; the stream ends 1.5 s after the last answer.
    """

    def __init__(self, paths: list[Path], gap: float, barge_in_at: float | None, frame_ms: int = 20) -> None:
        self.paths = paths
        self.gap = gap
        self.barge_in_at = barge_in_at
        self.frame_ms = frame_ms
        self.n = int(MIC_SAMPLE_RATE * frame_ms / 1000)
        self.last_answer_at: float | None = None
        self.awaiting = False  # an utterance was played and not answered yet
        self.ended = False  # ... and the VAD has seen its end
        self.audio_started_at: float | None = None
        self.log: list[tuple[float, str]] = []

    def on_event(self, name: str, data: dict[str, Any]) -> None:
        if name == "speech_end":
            self.ended = True
        elif name in ("turn", "turn_cancelled"):
            if self.ended:  # an answer that came after the utterance ended is its answer
                self.awaiting = False
                self.ended = False
                self.last_answer_at = time.perf_counter()
        elif name == "audio_start" and self.audio_started_at is None:
            self.audio_started_at = time.perf_counter()

    @staticmethod
    def load(path: Path) -> np.ndarray:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="int16", always_2d=True)
        pcm = data[:, 0]
        if sr != MIC_SAMPLE_RATE:  # crude decimation/interpolation; samples are 16 kHz anyway
            idx = np.round(np.arange(0, len(pcm), sr / MIC_SAMPLE_RATE)).astype(int)
            pcm = pcm[idx[idx < len(pcm)]]
        return np.ascontiguousarray(pcm)

    async def frames(self) -> AsyncIterator[np.ndarray]:
        silence = np.zeros(self.n, dtype=np.int16)
        t0 = time.perf_counter()
        i = 0
        u = 0
        playing: np.ndarray | None = None
        pos = 0
        while True:
            target = t0 + i * self.frame_ms / 1000
            delay = target - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            i += 1
            now = time.perf_counter()
            if playing is None:
                if u >= len(self.paths):
                    if not self.awaiting and self.last_answer_at is not None and now >= self.last_answer_at + 1.5:
                        self.log.append((round(now - t0, 3), "end of scenario"))
                        return
                    if now - t0 > 120:
                        self.log.append((round(now - t0, 3), "gave up waiting for the last answer"))
                        return
                else:
                    if u == 0:
                        due = now >= t0 + 0.5
                    elif self.barge_in_at is not None and u == 1:
                        due = self.audio_started_at is not None and now >= self.audio_started_at + self.barge_in_at
                    else:
                        due = not self.awaiting and self.last_answer_at is not None and now >= self.last_answer_at + self.gap
                    if due:
                        playing = self.load(self.paths[u])
                        pos = 0
                        self.awaiting = True
                        self.ended = False
                        self.log.append((round(now - t0, 3), f"utterance {u} start: {self.paths[u].name}"))
                        u += 1
            if playing is not None:
                frame = playing[pos : pos + self.n]
                pos += self.n
                if len(frame) < self.n:
                    frame = np.concatenate([frame, np.zeros(self.n - len(frame), dtype=np.int16)])
                    playing = None
                    self.log.append((round(now - t0, 3), f"utterance {u - 1} end"))
                yield frame
            else:
                yield silence


async def run_real(args: argparse.Namespace) -> int:
    from eva.session import build_session

    preset = PRESETS[args.preset]
    keys = load_keys()
    settings = dataclasses.replace(preset.settings)
    if args.no_barge_in:
        settings.barge_in = False
    if args.mute_fillers:
        settings.filler_after_ms = 0
    paths = [Path(p) if Path(p).exists() else SAMPLES_DIR / p for p in args.utterances.split(",") if p.strip()]
    for p in paths:
        if not p.exists():
            console.print(f"[red]missing utterance {escape(str(p))}[/]")
            return 2
    outage = {k.strip() for k in (args.outage or "").split(",") if k.strip()}
    if outage:
        simulate_outage(outage)
        console.print(f"[yellow]simulating a cloud outage for: {', '.join(sorted(outage))}[/]")

    console.print(f"[bold]preset[/] {preset.name}: {preset.description}")
    log = EventLog(verbose=True)
    session = build_session(
        preset, keys, lang=args.lang, brain=args.brain, persona=args.persona, user_name=args.user_name or "",
        fallbacks=not args.no_fallback, mute_fillers=args.mute_fillers, on_event=log,
    )
    stt, llm, tts = session.stt, session.llm, session.tts
    console.print(f"language {session.plan.mode} | persona {session.persona.name} ({session.persona.lang}) | {stt.name} + {llm.name} + {tts.name}")

    from eva.audio.vad import UtteranceSegmenter

    segmenter = UtteranceSegmenter(settings)
    if args.speakers:
        from eva.audio.player import Player

        player: Any = Player(tts.sample_rate, device=settings.output_device)
    else:
        player = MockPlayer(tts.sample_rate)
    player.start()
    log.player = player

    mic = ScenarioMic(paths, gap=args.gap, barge_in_at=args.barge_in_at)
    log.hooks.append(mic.on_event)

    t0 = time.perf_counter()
    await asyncio.gather(stt.warmup(), llm.warmup(), tts.warmup())
    warm_s = time.perf_counter() - t0
    console.print(f"warmup {warm_s:.2f}s")

    agent = VoiceAgent(
        stt, llm, tts, session.system_prompt, session.tools, settings,
        frames=mic.frames(), segmenter=segmenter, player=player, fillers=session.fillers,
        tool_hints=session.tool_hints, backchannels=session.backchannels, on_event=log, max_turns=args.max_turns,
        languages=session.plan.codes,
    )
    t0 = time.perf_counter()
    await agent.prepare()
    agent._select_lang(session.plan.primary.code)
    console.print(f"fillers pre-rendered in {time.perf_counter() - t0:.2f}s")
    try:
        turns = await asyncio.wait_for(agent.run(), timeout=args.timeout)
    except asyncio.TimeoutError:
        console.print("[red]timed out[/]")
        turns = agent.turns
    finally:
        try:
            player.close()
        except Exception:
            pass
        await asyncio.gather(stt.close(), llm.close(), tts.close(), return_exceptions=True)

    print_turn_table(f"e2e {preset.name}", _turn_rows(turns))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else OUT_DIR / f"e2e_{preset.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "preset": preset.name,
                "lang": session.plan.mode,
                "outage": sorted(outage),
                "stt": getattr(stt, "name", "?"),
                "llm": getattr(llm, "name", "?"),
                "tts": getattr(tts, "name", "?"),
                "settings": dataclasses.asdict(settings),
                "utterances": [str(p) for p in paths],
                "barge_in_at": args.barge_in_at,
                "warmup_s": round(warm_s, 3),
                "stream_turns": getattr(agent, "stream_turns", 0),
                "turns": _turn_rows(turns),
                "history": agent.messages,
                "events": [(round(t - log.t0, 3), n, d) for t, n, d in log.events if n != "state"],
                "mic_log": mic.log,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    console.print(f"wrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default=DEFAULT_PRESET, choices=sorted(PRESETS))
    ap.add_argument("--brain", choices=sorted(BRAINS), help="LLM override (default: the preset's)")
    ap.add_argument("--lang", default="auto", choices=lang_modes())
    ap.add_argument("--utterances", default="samples/user_hello.wav,samples/user_rough_day.wav,samples/user_task.wav")
    ap.add_argument("--gap", type=float, default=6.0, help="seconds after a turn before the next utterance")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--silent", action="store_true", help="MockPlayer (default)")
    g.add_argument("--speakers", action="store_true", help="real Player on the default output device")
    ap.add_argument("--persona")
    ap.add_argument("--user-name")
    ap.add_argument("--barge-in-at", type=float, help="play utterance #2 this many seconds into the first reply")
    ap.add_argument("--no-barge-in", action="store_true")
    ap.add_argument("--mute-fillers", action="store_true")
    ap.add_argument("--no-fallback", action="store_true", help="do not load the local backups")
    ap.add_argument("--outage", help="comma list of llm,stt,tts: point those cloud providers at a dead host")
    ap.add_argument("--max-turns", type=int)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--out", help="JSON output path (default bench/out/e2e_<preset>.json)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING, format="%(name)s %(levelname)s %(message)s")
    return asyncio.run(run_real(args))


if __name__ == "__main__":
    sys.exit(main())
