#!/usr/bin/env python
"""Red Oak Transport's dispatcher, every variant by one name.

    .venv/Scripts/python.exe dispatch.py --list
    .venv/Scripts/python.exe dispatch.py eva-v3                   # talk (Eva's loop, ElevenLabs v3 in the job's voice)
    .venv/Scripts/python.exe dispatch.py eva-v3 --voice flash
    .venv/Scripts/python.exe dispatch.py openai-realtime-2.1      # talk (speech-to-speech, headphones)
    .venv/Scripts/python.exe dispatch.py gemini-live --text       # type your lines, she speaks
    .venv/Scripts/python.exe dispatch.py --eval gemini-live-flash --audio   # the six scripted calls, spoken

Eva's loop hears with Parakeet on this laptop, thinks with Qwen3.8-27B on Cerebras and speaks with
ElevenLabs (v3 by default, ``--voice v3conv|flash``) in the dispatcher's own voice. Speech-to-speech
variants are one model that hears and speaks, with its own voice. All of them get the same
persona and the same tools on the same database.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# name: (kind, backend, what it is)
VARIANTS: dict[str, tuple[str, str, str]] = {
    # Eva's own loop: Parakeet ears, Cerebras brain, ElevenLabs voice (the dispatcher's own voice)
    "eva-v3": ("eva", "qwen27b", "Eva's loop: Parakeet + Qwen3.8-27B on Cerebras + ElevenLabs v3 (try --voice v3conv / flash)"),
    # OpenAI speech-to-speech, largest to smallest
    "openai-realtime-2.1": ("s2s", "openai:gpt-realtime-2.1", "OpenAI gpt-realtime-2.1 (newest, large, reasoning)"),
    "openai-realtime-2.1-mini": ("s2s", "openai:gpt-realtime-2.1-mini", "OpenAI gpt-realtime-2.1-mini (newest mini)"),
    "openai-realtime-2": ("s2s", "openai:gpt-realtime-2", "OpenAI gpt-realtime-2 (previous large, reasoning)"),
    "openai-realtime-1.5": ("s2s", "openai:gpt-realtime-1.5", "OpenAI gpt-realtime-1.5"),
    "openai-realtime": ("s2s", "openai:gpt-realtime", "OpenAI gpt-realtime (first GA, Aug 2025)"),
    "openai-realtime-mini": ("s2s", "openai:gpt-realtime-mini", "OpenAI gpt-realtime-mini (Dec 2025)"),
    "openai-live": ("s2s", "openai-live:gpt-live-1", "OpenAI GPT-Live 1: full duplex, tools through a gpt-5.6-luna backend"),
    "openai-live-sol": ("s2s", "openai-live:gpt-live-1+gpt-5.6-sol", "OpenAI GPT-Live 1 with the larger gpt-5.6-sol backend"),
    # Google speech-to-speech
    "gemini-live": ("s2s", "gemini:gemini-3.8-live", "Google Gemini 3.8 Live (newest)"),
    "gemini-live-thinking": ("s2s", "gemini:gemini-3.8-live-extended-thinking", "Google Gemini 3.8 Live, extended thinking"),
    "gemini-live-flash": ("s2s", "gemini:gemini-3.1-flash-live-preview", "Google Gemini 3.1 Flash Live (preview)"),
    "gemini-live-2.5": ("s2s", "gemini:gemini-2.5-flash-native-audio-latest", "Google Gemini 2.5 Flash native audio (older)"),
}


def eval_backend(name: str) -> str:
    kind, backend, _ = VARIANTS[name]
    return f"eva:{backend}" if kind == "eva" else backend


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variant", nargs="?", choices=list(VARIANTS))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--eval", metavar="VARIANT", choices=list(VARIANTS), help="run the six scripted calls on a variant")
    ap.add_argument("--voice", default="v3", help="Eva's loop: v3 (default), v3conv or flash")
    ap.add_argument("--text", action="store_true", help="type instead of talking")
    args, extra = ap.parse_known_args()
    py = sys.executable
    if args.list or not (args.variant or args.eval):
        for name, (kind, backend, what) in VARIANTS.items():
            print(f"  {name:22s} {what}")
        return 0
    if args.eval:
        return subprocess.call([py, str(ROOT / "bench" / "dispatch_eval.py"), eval_backend(args.eval), *extra])
    kind, backend, _ = VARIANTS[args.variant]
    if kind == "eva":
        cmd = [py, str(ROOT / "run.py"), "--job", "dispatch", "--brain", backend, "--voice", args.voice, "--user-name", "Doston"]
        if args.text:
            cmd.append("--text")
    else:
        cmd = [py, str(ROOT / "run_s2s.py"), backend] + (["--text"] if args.text else [])
    return subprocess.call(cmd + extra)


if __name__ == "__main__":
    sys.exit(main())
