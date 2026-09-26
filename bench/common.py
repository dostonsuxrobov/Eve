"""Shared by the brain bench (``bench/brains.py``) and the chat tool (``bench/chat.py``):
the candidate list, a small streaming client for Ollama's native API, the Eva system
prompt, the tool schemas and the automatic style checks.

Ollama is reached at 127.0.0.1, never ``localhost``: on this laptop ``localhost`` resolves
to ::1 first and costs about 2 s per request.
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eva.personas import load_persona, now_string, render  # noqa: E402

OLLAMA = "http://127.0.0.1:11434"
NUM_CTX = 8192  # the Eva prompt is ~2.5k tokens; 8k leaves room for a long conversation
KEEP_ALIVE = "30m"

# Small brains to free VRAM for the voice (2026-09-25). Sizes and dates from ollama.com and
# the makers' pages that day; the notes are what each one is here to show.
CANDIDATES: list[tuple[str, str]] = [
    ("qwen3.5:2b-q4_K_M", "Qwen 3.5 2B (2026-03): best 2B on Artificial Analysis' index"),
    ("openbmb/minicpm5-2b", "MiniCPM5 2B (2026-09-07): the newest 2B"),
    ("LiquidAI/lfm2.5-1.2b-instruct:q8_0", "LFM2.5 1.2B (Liquid): built for on-device chat"),
    ("openbmb/minicpm5:q8_0", "MiniCPM5 1B (2026-05): claims best 1B"),  # the README's minicpm5-1b tag doesn't exist
    ("gemma3:1b-it-qat", "Gemma 3 1B QAT (2025): Google's conversational register"),
    ("qwen3.5:0.8b", "Qwen 3.5 0.8B (2026-03): the floor"),
    ("qwen3:4b-instruct-2507-q4_K_M", "reference: today's 4B brain"),
]

# Stand-in user for the scripted examples: the same facts as the cloud-era eval, so the
# transcripts in .archive/bench/out stay comparable.
BENCH_USER = "Sam"
BENCH_MEMORY = "\n".join(
    [
        "- Name is Sam. Prefers Sam, never Samuel.",
        "- Has a cat named Miso, a grey tabby, about four years old.",
        "- Works as a product designer at a small startup; the big project is a redesign of the onboarding flow.",
        "- Mom lives in another city; they usually call on Sunday evenings.",
        "- Has a sister called Priya.",
        "- Mentioned wanting to get back into running.",
    ]
)
NO_TOOLS_NOTE = (
    "No tools are connected in this session, so you can't set timers or reminders, "
    "check the time or weather, or look anything up. If asked, say so in one plain "
    "sentence and offer what you can do instead."
)

# Eva's tools as the cloud-era loop offered them (.archive/eva/tools.py); schemas only.
TOOLS: list[dict[str, Any]] = [
    {"name": "end_conversation",
     "description": ("End the conversation. Call this ONLY when the user themselves says goodbye, says they "
                     "have to go, or asks you to stop; say your goodbye in the same reply. Never call it "
                     "because of a silence, an error, a system message or anything you decided on your own."),
     "parameters": {"type": "object", "properties": {"reason": {"type": "string", "description": "Why the conversation ends, a few words."}}, "required": []}},
    {"name": "get_current_time",
     "description": "Get the current local date and time. Use when the user asks what time or day it is.",
     "parameters": {"type": "object", "properties": {}, "required": []}},
    {"name": "set_timer",
     "description": "Set a countdown timer that will notify the user when it ends. Convert the requested duration to seconds (two minutes = 120).",
     "parameters": {"type": "object", "properties": {
         "seconds": {"type": "integer", "description": "Duration in seconds (1 to 86400)."},
         "label": {"type": "string", "description": "Short name for the timer, e.g. 'tea'."}}, "required": ["seconds", "label"]}},
    {"name": "remember_note",
     "description": "Save a short note or reminder text the user wants remembered for later.",
     "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "The note to save, in the user's words."}}, "required": ["text"]}},
    {"name": "recall_notes",
     "description": "Read back the user's saved notes and reminders.",
     "parameters": {"type": "object", "properties": {}, "required": []}},
    {"name": "get_weather",
     "description": "Get the current weather and today's forecast for a city.",
     "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g. 'Boston' or 'Tashkent'."}}, "required": ["city"]}},
    {"name": "open_url",
     "description": "Open a website in the user's browser.",
     "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "Full URL, e.g. https://youtube.com"}}, "required": ["url"]}},
]
TOOL_SCHEMAS = [{"type": "function", "function": t} for t in TOOLS]


def tool_notes() -> str:
    """The ``{tool_notes}`` text the cloud-era loop used (its wording took the 27B from
    1-3/8 real timer calls to 8/8)."""
    lines = "\n".join(f"- {t['name']}: {t['description']}" for t in TOOLS)
    return (
        "Tools, and this rule matters most: when they ask for a timer, a reminder or "
        "note, the weather, the time or to open a link, or when they say goodbye or that "
        "they have to go, your reply MUST include the function call (you may add a short "
        "aside like 'one sec', or the goodbye itself). You physically cannot do any of "
        "these by talking, so a reply without the call means it did not happen. "
        "The tools:\n" + lines
    )


def eva_prompt(*, user_name: str | None, memory: str, tools: bool) -> str:
    """The Eva system prompt for a voice without audio tags (Kokoro-class), English only.

    Build it once per conversation: ``{now}`` sits in the first line, so a prompt rebuilt
    every minute would miss Ollama's prefix cache for the whole persona.
    """
    return render(load_persona("eva"), supports_audio_tags=False, memory_text=memory, now=now_string(),
                  user_name=user_name, tool_notes=tool_notes() if tools else NO_TOOLS_NOTE,
                  locked_language="English")


# ------------------------------------------------------------------ Ollama client
@dataclass
class Reply:
    """One streamed answer and Ollama's own timings for it."""

    text: str = ""
    thinking: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    ttft_s: float | None = None  # request -> first spoken character or tool call
    total_s: float = 0.0
    eval_count: int = 0
    eval_s: float = 0.0
    prompt_count: int = 0
    prompt_s: float = 0.0
    load_s: float = 0.0
    done_reason: str = ""
    error: str = ""

    @property
    def tok_per_s(self) -> float | None:
        return self.eval_count / self.eval_s if self.eval_s > 0 else None


def client() -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(300.0, connect=5.0))


def show(c: httpx.Client, model: str) -> dict[str, Any]:
    r = c.post(f"{OLLAMA}/api/show", json={"model": model})
    r.raise_for_status()
    return r.json()


def loaded(c: httpx.Client) -> list[dict[str, Any]]:
    r = c.get(f"{OLLAMA}/api/ps")
    r.raise_for_status()
    return r.json().get("models") or []


def full_tag(model: str) -> str:
    """``ollama ps`` lists an untagged pull as ``name:latest``."""
    return model if ":" in model.rsplit("/", 1)[-1] else model + ":latest"


def unload_all(c: httpx.Client, keep: str | None = None) -> list[str]:
    """Unload every model but ``keep`` and wait until they are gone (clean VRAM numbers)."""
    names = [m["name"] for m in loaded(c) if keep is None or m["name"] != full_tag(keep)]
    for name in names:
        c.post(f"{OLLAMA}/api/generate", json={"model": name, "keep_alive": 0})
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and any(m["name"] in names for m in loaded(c)):
        time.sleep(0.2)
    return names


def load(c: httpx.Client, model: str) -> float:
    """Load ``model`` with the bench's context size; returns the wall time it took."""
    t0 = time.perf_counter()
    r = c.post(f"{OLLAMA}/api/generate", json={"model": model, "keep_alive": KEEP_ALIVE, "options": {"num_ctx": NUM_CTX}})
    r.raise_for_status()
    return time.perf_counter() - t0


def vram(c: httpx.Client, model: str) -> dict[str, Any]:
    """What ``ollama ps`` says about ``model``: bytes in VRAM, total bytes, context."""
    for m in loaded(c):
        if full_tag(model) in (m["name"], m.get("model")):
            return {"size_vram": m.get("size_vram", 0), "size": m.get("size", 0), "context": m.get("context_length")}
    return {"size_vram": 0, "size": 0, "context": None}


def chat(
    c: httpx.Client,
    model: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    think: bool | None = None,
    num_predict: int = 256,
    on_text: Callable[[str], None] | None = None,
) -> Reply:
    """Stream one reply from ``/api/chat``. ``think`` is sent only when not None: Ollama
    refuses the field for models without a thinking mode."""
    body: dict[str, Any] = {
        "model": model, "messages": messages, "stream": True, "keep_alive": KEEP_ALIVE,
        "options": {"num_ctx": NUM_CTX, "num_predict": num_predict},
    }
    if tools:
        body["tools"] = tools
    if think is not None:
        body["think"] = think
    reply = Reply()
    t0 = time.perf_counter()
    try:
        with c.stream("POST", f"{OLLAMA}/api/chat", json=body) as r:
            if r.status_code != 200:
                reply.error = f"HTTP {r.status_code}: {r.read().decode('utf-8', 'replace')[:300]}"
                return reply
            for line in r.iter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    reply.error = str(chunk["error"])
                    break
                msg = chunk.get("message") or {}
                if msg.get("thinking"):
                    reply.thinking += msg["thinking"]
                text = msg.get("content") or ""
                if text:
                    if reply.ttft_s is None and text.strip():
                        reply.ttft_s = time.perf_counter() - t0
                    reply.text += text
                    if on_text:
                        on_text(text)
                for call in msg.get("tool_calls") or []:
                    if reply.ttft_s is None:
                        reply.ttft_s = time.perf_counter() - t0
                    fn = call.get("function") or {}
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass
                    reply.tool_calls.append({"name": fn.get("name"), "arguments": args})
                if chunk.get("done"):
                    reply.eval_count = int(chunk.get("eval_count") or 0)
                    reply.eval_s = (chunk.get("eval_duration") or 0) / 1e9
                    reply.prompt_count = int(chunk.get("prompt_eval_count") or 0)
                    reply.prompt_s = (chunk.get("prompt_eval_duration") or 0) / 1e9
                    reply.load_s = (chunk.get("load_duration") or 0) / 1e9
                    reply.done_reason = chunk.get("done_reason") or ""
    except httpx.HTTPError as e:
        reply.error = repr(e)
    reply.total_s = time.perf_counter() - t0
    return reply


def think_setting(capabilities: list[str]) -> bool | None:
    """``False`` for models with a thinking mode (a voice can't wait for it), else unset."""
    return False if "thinking" in capabilities else None


# ------------------------------------------------------------------ style checks
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\U0001F000-\U0001F2FF☀-➿⭐⭕]")
# An action announced in words; with no tools connected nothing is behind it.
PROMISE_RE = re.compile(
    r"\b(one sec|hang on|let me (check|look|find|pull|see if)|i'?ll (check|look|find|pull|set|remind|note)|"
    r"i'?m (looking|checking|setting|pulling)|setting (that|a|the|your) )"
)
# Phrases the persona forbids by name.
BANNED = (
    "let me know if you need", "anything else on your mind", "great question", "i hear you", "holding space",
    "that's so valid", "makes total sense", "take a breath", "i'm so sorry", "i can hear that", "decompress",
    "unpack", "sit with that", "self-care", "as an ai language model", "i'm so proud of you", "i'm so happy for you",
)


def style_flags(text: str, *, tools_offered: bool) -> list[str]:
    """Rule breaks a script can see. Honesty and warmth need a reader; these don't."""
    flags: list[str] = []
    low = text.lower().replace("’", "'")
    words = len(text.split())
    if not text.strip():
        flags.append("empty")
    if words > 40:
        flags.append(f"{words} words")
    if EMOJI_RE.search(text):
        flags.append("emoji")
    if re.search(r"\*[^*\n]+\*", text):
        flags.append("*action*")
    if re.search(r"\[[^\]\n]+\]", text):
        flags.append("[bracket]")
    if re.search(r"(?m)^\s*([-*•]|\d+[.)])\s", text) or "**" in text or re.search(r"(?m)^#", text):
        flags.append("list/markdown")
    if text.count("?") > 1:
        flags.append(f"{text.count('?')} questions")
    if "<think" in low or "</think" in low:
        flags.append("think tags")
    if not tools_offered and PROMISE_RE.search(low):
        flags.append("promises an action")
    flags += [f'"{p}"' for p in BANNED if p in low]
    return flags


def opening(text: str) -> str:
    """The first two words, lowercased, to catch replies that all start the same way."""
    return " ".join(re.findall(r"[a-z']+", text.lower().replace("’", "'"))[:2])
