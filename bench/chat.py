#!/usr/bin/env python
"""Talk to one brain candidate as Eva in the terminal, with its speed after every reply.

    .venv/Scripts/python.exe bench/chat.py                  # pick from the candidate list
    .venv/Scripts/python.exe bench/chat.py 3                # the third candidate
    .venv/Scripts/python.exe bench/chat.py 3 --name Doston --tools
    .venv/Scripts/python.exe bench/chat.py qwen3.5:0.8b --raw   # no persona: the bare model

In the chat: /reset starts a new conversation, /quit leaves (so does Ctrl+C). The dim
line after each reply is the time to her first word, the decode speed and any rule the
reply broke. With --tools she gets Eva's tools; each call is shown and answered by a
stand-in (the real clock for the time, "done" for timers and notes, "not connected" for
the weather), so you can see whether she calls or only says she did.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.markup import escape

from common import (CANDIDATES, TOOL_SCHEMAS, chat, client, eva_prompt, load, show, style_flags, think_setting,
                    unload_all, vram)

console = Console(highlight=False)


def dim(text: str) -> None:
    console.print(f"[dim]{escape(text)}[/dim]")


def stand_in_result(name: str, args: dict[str, Any]) -> str:
    """What a tool 'returns' in this test: enough for her to carry on honestly."""
    if name == "get_current_time":
        now = datetime.now()
        return f"It's {now:%A}, {now.hour % 12 or 12}:{now:%M} {'am' if now.hour < 12 else 'pm'}."
    if name == "set_timer":
        return f"OK: timer '{args.get('label', '')}' set for {args.get('seconds')} seconds."
    if name == "remember_note":
        return "OK: saved."
    if name == "recall_notes":
        return "No saved notes yet."
    if name == "get_weather":
        return "The weather service is not connected in this test: say plainly you can't check it right now."
    if name == "open_url":
        return "OK: opened."
    if name == "end_conversation":
        return "OK: the session ends right after this reply. Say one short, warm goodbye now, nothing else."
    return f"Unknown tool {name!r}."


def pick_model(arg: str | None) -> str:
    if arg and not arg.isdigit():
        return arg
    if not arg:
        for i, (model, note) in enumerate(CANDIDATES, 1):
            print(f"  {i}. {model:36} {note}")
        arg = input("which one? ").strip()
    return CANDIDATES[int(arg) - 1][0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="an Ollama tag, or the number of a candidate (no argument: pick from a list)")
    ap.add_argument("--name", default=None, help="your name as she should know it (default: she doesn't know it)")
    ap.add_argument("--tools", action="store_true", help="give her Eva's tools (stand-in results)")
    ap.add_argument("--raw", action="store_true", help="no persona, the bare model")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    model = pick_model(args.model)

    with client() as c:
        caps = list(show(c, model).get("capabilities") or [])
        think = think_setting(caps)
        tools = TOOL_SCHEMAS if args.tools and "tools" in caps else None
        if args.tools and tools is None:
            dim(f"{model} has no tool support in its Ollama template: chatting without tools")
        freed = unload_all(c, keep=model)
        if freed:
            dim(f"unloaded {', '.join(freed)} to give it the GPU")
        load_s = load(c, model)
        m = vram(c, model)
        system = None if args.raw else eva_prompt(user_name=args.name, memory="", tools=bool(tools))
        dim(f"{model}: loaded in {load_s:.1f} s, {m['size_vram'] / 2**30:.2f} GB in VRAM (ctx {m['context']}); "
            f"{'bare model' if args.raw else 'Eva persona'}{', tools' if tools else ''}. /reset starts over, /quit leaves.")
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []

        while True:
            try:
                user = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user:
                continue
            if user in ("/quit", "/exit"):
                break
            if user == "/reset":
                messages = messages[:1] if system else []
                dim("(new conversation)")
                continue
            messages.append({"role": "user", "content": user})
            for _ in range(3):  # a tool round, then her follow-up
                print("eva> ", end="", flush=True)
                rep = chat(c, model, messages, tools=tools, think=think, on_text=lambda s: print(s, end="", flush=True))
                print()
                text = rep.text.strip()
                flags = style_flags(text, tools_offered=bool(tools))
                if rep.tool_calls and "empty" in flags:
                    flags.remove("empty")  # a bare call is fine; the follow-up round speaks
                if rep.thinking:
                    flags.append(f"thought {len(rep.thinking)} chars first")
                if rep.error:
                    flags.append("ERROR " + rep.error)
                ttft = "–" if rep.ttft_s is None else f"{rep.ttft_s:.2f} s"
                speed = "–" if rep.tok_per_s is None else f"{rep.tok_per_s:.0f} tok/s"
                dim(f"{ttft} to first word · {speed} · {rep.eval_count} tokens · {len(text.split())} words"
                    + (" · " + ", ".join(flags) if flags else ""))
                turn: dict[str, Any] = {"role": "assistant", "content": text}
                if rep.tool_calls:
                    turn["tool_calls"] = [{"function": {"name": t["name"], "arguments": t["arguments"] or {}}} for t in rep.tool_calls]
                messages.append(turn)
                if not rep.tool_calls:
                    break
                for t in rep.tool_calls:
                    result = stand_in_result(t["name"], t["arguments"] or {})
                    dim(f"[tool] {t['name']}({json.dumps(t['arguments'] or {})}) -> {result}")
                    messages.append({"role": "tool", "content": result, "tool_name": t["name"]})
                if any(t["name"] == "end_conversation" for t in rep.tool_calls) and text:
                    dim("(she said goodbye and ended the call; in the real loop the session would stop here)")
                    break
        unload_all(c)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
