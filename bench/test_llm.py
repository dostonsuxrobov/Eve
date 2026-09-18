"""Real-run benchmark + unit checks for eva.llm.* and eva.tools.

Runs (unless ``--skip-unit``) offline assertions for the chunker, sanitizer, tool
argument parsing and the timer event queue, then hits every requested backend:

* Cerebras ``qwen-3.8-27b`` (disable_reasoning)
* Cerebras ``gpt-oss-120b`` (reasoning_effort low)
* Ollama ``qwen3:4b-instruct-2507-q4_K_M``

For each backend: warmup time; N spoken-style chat turns (TTFT / total / tokens per
second, median); one tool-calling exchange executed through ``eva.tools.execute`` with
results fed back; the chunker over a streamed answer; and a leak report (markdown,
emoji, think tags, bracket tags, stage directions).

Usage::

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe bench/test_llm.py
    ... --models cerebras-qwen,ollama --turns 3

Results are also written to ``bench/out/test_llm_results.json``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eva import tools as eva_tools  # noqa: E402
from eva.config import load_keys  # noqa: E402
from eva.interfaces import LLMDelta, LLMDone, LLMToolCall  # noqa: E402
from eva.llm.chunker import SentenceChunker  # noqa: E402
from eva.llm.openai_compat import (  # noqa: E402
    LLMHTTPError,
    OpenAICompatLLM,
    _parse_tool_arguments,
    _ThinkFilter,
    assistant_tool_call_message,
    make_cerebras,
    make_ollama,
)
from eva.llm.sanitize import _EMOJI_RE, clean_for_tts, strip_think  # noqa: E402

OUT_DIR = ROOT / "bench" / "out"

SYSTEM_PROMPT = (
    "You are Eva, a warm, emotionally intelligent voice companion talking out loud with a "
    "close friend. Everything you say is read by a text-to-speech voice, so keep each reply "
    "to one to three short sentences, use contractions and a natural spoken rhythm, and never "
    "use lists, markdown, headings, emoji, or stage directions. Attune to feelings first and "
    "information second: mirror the person's energy, name feelings lightly, and ask at most one "
    "gentle question. Small backchannels like 'mm' or 'yeah' are fine, and you can use "
    "ellipses for a pause. Never claim a task is done unless a tool result says so."
)

TOOL_PROMPT = (
    SYSTEM_PROMPT
    + " You have tools for the current time, timers, notes, weather and opening websites; use "
    "them whenever they are needed, and when several are needed call them all. After the tool "
    "results come back, tell the person briefly and naturally what happened."
)

CHAT_TURNS = [
    "Hey Eva, how's it going? I just got home from work.",
    "Honestly, today was rough. My manager pulled me into a meeting and basically said the "
    "project might get cancelled. I don't know what to do.",
    "Yeah... I guess. I've been on this project for eight months, it's kind of my baby.",
    "Thanks. Anyway, what should I make for dinner? I've got eggs and some spinach.",
    "Sounds good. Okay, I'm gonna go cook. Talk later?",
]

TOOL_TURN = "what time is it, and set a two minute timer called tea"
CHUNKER_TURN = "Tell me about a small thing that made you happy recently, in a few sentences."

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "cerebras-qwen": {"kind": "cerebras", "model": "qwen-3.8-27b", "reasoning": "none"},
    "cerebras-gpt-oss": {"kind": "cerebras", "model": "gpt-oss-120b", "reasoning": "low"},
    "ollama": {"kind": "ollama", "model": "qwen3:4b-instruct-2507-q4_K_M"},
}


# ------------------------------------------------------------------ unit tests
def _chunk_all(text: str, step: int, **kw: Any) -> list[str]:
    ch = SentenceChunker(**kw)
    out: list[str] = []
    for i in range(0, len(text), step):
        out += ch.feed(text[i : i + step])
    out += ch.flush()
    return out


def unit_tests_sync() -> int:
    """Offline assertions; returns the number of checks passed."""
    n = 0

    def check(cond: bool, msg: str) -> None:
        nonlocal n
        assert cond, msg
        n += 1

    # --- chunker: early first chunk at a comma, then sentence enders (delta-size independent)
    text = "Honestly, that sounds really rough, I'm sorry. Do you want to talk about it? I'm here."
    want = ["Honestly, that sounds really rough,", "I'm sorry.", "Do you want to talk about it?", "I'm here."]
    for step in (1, 2, 5, 11, 200):
        check(_chunk_all(text, step) == want, f"early-cut chunks wrong at step {step}: {_chunk_all(text, step)}")
    # first chunk must be >= 14 chars before a comma is used
    check(_chunk_all("Hey, I'm here for you. Really.", 3) == ["Hey, I'm here for you.", "Really."], "comma too early")
    # numbers, abbreviations, initials
    got = _chunk_all("The price is 3.5 dollars. Dr. Smith said e.g. that J. K. Rowling wrote it. Then we left.", 4)
    check(got == ["The price is 3.5 dollars.", "Dr. Smith said e.g. that J. K. Rowling wrote it.", "Then we left."], f"abbrev: {got}")
    check(_chunk_all("It costs 1,000 dollars, which is a lot. Yes.", 3) == ["It costs 1,000 dollars,", "which is a lot.", "Yes."], "1,000 split")
    # short chunk merged into the next
    check(_chunk_all("Hi. How are you? Fine.", 1) == ["Hi. How are you?", "Fine."], "short merge")
    # ender runs, em dash, ellipsis
    check(_chunk_all("Wait—what?! No way! Okay... tell me more.", 2) == ["Wait—what?!", "No way!", "Okay...", "tell me more."], "runs")
    # audio tags stay attached to the following sentence
    got = _chunk_all("[laughs] Oh man, that's great. That's funny. [sighs] I know, right?", 3)
    check(got == ["[laughs] Oh man,", "that's great.", "That's funny.", "[sighs] I know, right?"], f"tags: {got}")
    check(_chunk_all("[clears throat. really] fine now. Done.", 3) == ["[clears throat. really] fine now.", "Done."], "no split inside tag")
    # newline cut, blank lines dropped, bullets become lines
    check(_chunk_all("Sure thing!\nHere's the thing.\n\n- apples\n- pears\n", 3) == ["Sure thing!", "Here's the thing.", "- apples", "- pears"], "newline")
    # closing quote after ender
    check(_chunk_all('He said "no." Then he left.', 2) == ['He said "no."', "Then he left."], "closing quote")
    # nothing lost, flush returns the tail, reset works
    ch = SentenceChunker()
    check(ch.feed("Hello there") == [], "premature emit")
    check(ch.flush() == ["Hello there"], "flush tail")
    check(ch.flush() == [], "flush twice")
    check(ch.pending == "", "pending after flush")
    # every character of the input survives (modulo whitespace)
    src = "One two three, four five six. Seven eight? Nine!\nTen."
    squash = lambda s: s.replace(" ", "").replace("\n", "")  # noqa: E731
    check(squash("".join(_chunk_all(src, 3))) == squash(src), "chars lost")

    # --- sanitizer
    md = "**Hello** there! Here's a *list*:\n- one\n- two\n1. three\n## Header\n`code` and ```py\nx=1\n``` done 😊👍"
    check(clean_for_tts(md, False) == "Hello there! Here's a list: one two three Header code and x=1 done", f"md: {clean_for_tts(md, False)!r}")
    sd = "*sighs* I know, *really* I do. *leans in closer* See https://example.com/x?y=1 ok [laughs] [interrupted] [Whispers] hi"
    check(clean_for_tts(sd, True) == "I know, really I do. See a link ok [laughs] [whispers] hi", f"tags kept: {clean_for_tts(sd, True)!r}")
    check(clean_for_tts(sd, False) == "I know, really I do. See a link ok hi", f"tags dropped: {clean_for_tts(sd, False)!r}")
    check(clean_for_tts("<think>\nplan\n</think>\nSure, “quoted” and ‘single’ (laughs) ok!!!!", False) == "Sure, \"quoted\" and 'single' ok!!", "think/quotes")
    check(clean_for_tts("Look at [this link](https://x.y) and [Laughs] then [clears throat] ⭐ ❤️ 🇺🇸 ➡️ end", True) == "Look at this link and [laughs] then [clears throat] end", "links/emoji")
    check(clean_for_tts("[pause] [giggles] [robot voice] ok", True) == "[pause] [giggles] ok", "whitelist")
    check(clean_for_tts("It\u2019s 7:17\u202fPM, a two\u2011minute timer \u201ctea\u201d", False) == "It's 7:17 PM, a two-minute timer \"tea\"", "typographic")
    check(clean_for_tts("", True) == "" and clean_for_tts("   \n ", False) == "", "empty")
    check(strip_think("<think>abc") == "" and strip_think("<think>a</think>\n\nHi") == "Hi" and strip_think("plain") == "plain", "strip_think")

    # --- streaming think filter (tags split across deltas)
    f = _ThinkFilter()
    out = "".join(f.feed(d) for d in ["<th", "ink>reason", "ing here</th", "ink>\nHel", "lo"]) + f.flush()
    check(out == "Hello", f"think filter: {out!r}")
    f = _ThinkFilter()
    out = "".join(f.feed(d) for d in ["He", "llo <b>"]) + f.flush()
    check(out == "Hello <b>", f"think filter passthrough: {out!r}")
    f = _ThinkFilter()
    out = "".join(f.feed(d) for d in ["<", "t"]) + f.flush()
    check(out == "<t", f"think filter partial flush: {out!r}")

    # --- tool argument parsing
    check(_parse_tool_arguments("") == {} and _parse_tool_arguments("garbage") == {}, "bad args")
    check(_parse_tool_arguments('{"seconds": 120, "label": "tea"}') == {"seconds": 120, "label": "tea"}, "args")
    check(_parse_tool_arguments('"{\\"a\\": 1}"') == {"a": 1}, "double-encoded args")
    check(_parse_tool_arguments("[1,2]") == {}, "non-object args")
    msg = assistant_tool_call_message("", [LLMToolCall("c1", "set_timer", {"seconds": 5, "label": "x"})])
    check(msg["content"] is None and msg["tool_calls"][0]["function"]["arguments"] == '{"seconds": 5, "label": "x"}', "assistant msg")

    # --- registry
    tl = eva_tools.get_tools()
    names = {t.name for t in tl}
    check(names == {"get_current_time", "set_timer", "remember_note", "recall_notes", "get_weather", "open_url"}, f"tools: {names}")
    sch = eva_tools.schemas(tl)
    check(all(s["type"] == "function" and "parameters" in s["function"] for s in sch), "schemas")
    return n


async def unit_tests_async() -> int:
    n = 0
    loop_q = eva_tools.set_event_loop()
    # unknown tool / missing args / type coercion / timer event on the queue
    r = await eva_tools.execute(LLMToolCall("x", "nope", {}))
    assert r.startswith("Error"), r
    n += 1
    r = await eva_tools.execute(LLMToolCall("x", "set_timer", {"label": "no seconds"}))
    assert r.startswith("Error") and "seconds" in r, r
    n += 1
    r = await eva_tools.execute(LLMToolCall("x", "set_timer", {"seconds": "1", "label": "unit"}))
    assert r.startswith("Timer 'unit' set for 1 second"), r
    n += 1
    assert any(t["label"] == "unit" for t in eva_tools.active_timers())
    n += 1
    t0 = time.perf_counter()
    ev = await asyncio.wait_for(loop_q.get(), timeout=3)
    dt = time.perf_counter() - t0
    assert ev["type"] == "timer" and ev["label"] == "unit" and 0.8 < dt < 1.6, (ev, dt)
    n += 1
    assert eva_tools.pending_events is loop_q and not eva_tools.active_timers()
    n += 1
    r = await eva_tools.execute(LLMToolCall("x", "get_current_time", {"bogus": 1}))
    assert r.startswith("It's ") and "20" in r, r
    n += 1
    # notes round trip against the real notes.json under ROOT, restored afterwards
    notes_path = eva_tools.NOTES_FILE
    before = notes_path.read_bytes() if notes_path.exists() else None
    try:
        r = await eva_tools.execute(LLMToolCall("x", "remember_note", {"text": "bench note: call mom tonight"}))
        assert r.startswith("Noted"), r
        n += 1
        r = await eva_tools.execute(LLMToolCall("x", "recall_notes", {}))
        assert "call mom tonight" in r, r
        n += 1
    finally:
        if before is None:
            notes_path.unlink(missing_ok=True)
        else:
            notes_path.write_bytes(before)
    # open_url: refuse non-web schemes without launching anything
    r = await eva_tools.execute(LLMToolCall("x", "open_url", {"url": "file:///C:/Windows"}))
    assert r.startswith("Error"), r
    n += 1
    # weather is a live call; failures are reported as an error string, never raised
    t0 = time.perf_counter()
    r = await eva_tools.execute(LLMToolCall("x", "get_weather", {"city": "Boston"}))
    print(f"  weather(Boston) in {time.perf_counter() - t0:.2f} s -> {r}")
    assert isinstance(r, str) and r, r
    n += 1
    return n


# ------------------------------------------------------- fake backend test
def _sse(*objs: dict[str, Any]) -> bytes:
    body = "".join(f"data: {json.dumps(o)}\n\n" for o in objs) + "data: [DONE]\n\n"
    return body.encode()


def _chunk(**delta: Any) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": delta}]}


class _FakeBackend:
    """Minimal scripted HTTP/1.1 keep-alive server speaking the chat-completions SSE dialect."""

    def __init__(self, script: list[tuple[int, bytes, str]]) -> None:
        self.script = list(script)  # (status, body, content-type)
        self.requests: list[dict[str, Any]] = []
        self.server: asyncio.AbstractServer | None = None
        self.port = 0

    async def start(self) -> None:
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        assert self.server is not None
        self.server.close()
        await self.server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                head = await reader.readuntil(b"\r\n\r\n")
                lines = head.decode().split("\r\n")
                headers = {k.lower(): v for k, v in (ln.split(": ", 1) for ln in lines[1:] if ": " in ln)}
                length = int(headers.get("content-length", "0"))
                raw = await reader.readexactly(length) if length else b""
                self.requests.append({"line": lines[0], "headers": headers, "body": json.loads(raw) if raw else None})
                if self.script:
                    status, body, ctype = self.script.pop(0)
                else:
                    status, body, ctype = 500, b'{"error":"script exhausted"}', "application/json"
                reason = {200: "OK", 400: "Bad Request", 429: "Too Many Requests", 500: "Internal Server Error"}.get(status, "X")
                head_out = (
                    f"HTTP/1.1 {status} {reason}\r\nContent-Type: {ctype}\r\n"
                    f"Content-Length: {len(body)}\r\nConnection: keep-alive\r\n\r\n"
                )
                writer.write(head_out.encode() + body)
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        finally:
            writer.close()


async def fake_backend_tests() -> int:
    """Exercise stream_options fallback, 429 retry, 5xx failure, split tool args, think leak."""
    n = 0
    good = _sse(
        _chunk(role="assistant"),
        _chunk(reasoning="let me think"),
        _chunk(content="<think>secret"),
        _chunk(content=" plan</think>Hel"),
        _chunk(content="lo there."),
        _chunk(tool_calls=[{"index": 0, "id": "c1", "type": "function", "function": {"name": "set_timer", "arguments": ""}}]),
        _chunk(tool_calls=[{"index": 0, "function": {"arguments": '{"seconds": 1'}}]),
        _chunk(tool_calls=[{"index": 0, "function": {"arguments": '20, "label": "tea"}'}}]),
        _chunk(tool_calls=[{"index": 1, "id": "c2", "type": "function", "function": {"name": "get_current_time", "arguments": "not json"}}]),
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 5, "completion_tokens": 9}},
    )
    plain = _sse(_chunk(content="Hi."), {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    script = [
        (400, b'{"error": {"message": "stream_options is not supported"}}', "application/json"),  # -> retry w/o
        (200, good, "text/event-stream"),
        (429, b'{"error": "slow down"}', "application/json"),  # -> retry once
        (200, plain, "text/event-stream"),
        (500, b"boom", "text/plain"),
        (500, b"boom again", "text/plain"),  # -> raises
        (200, b'{"choices": [{"message": {"role": "assistant", "content": "<think>x</think>\\nSummary."}}]}', "application/json"),
    ]
    fb = _FakeBackend(script)
    await fb.start()
    llm = OpenAICompatLLM("fake", f"http://127.0.0.1:{fb.port}/v1", "k", "m", {"disable_reasoning": True}, max_tokens=50)
    try:
        tools = [{"type": "function", "function": {"name": "set_timer", "parameters": {}}}]
        events = [ev async for ev in llm.stream([{"role": "user", "content": "x"}], tools)]
        text = "".join(e.text for e in events if isinstance(e, LLMDelta))
        calls = [e for e in events if isinstance(e, LLMToolCall)]
        done = [e for e in events if isinstance(e, LLMDone)]
        assert text == "Hello there.", text
        n += 1
        want = [("c1", "set_timer", {"seconds": 120, "label": "tea"}), ("c2", "get_current_time", {})]
        assert [(c.id, c.name, c.arguments) for c in calls] == want, calls
        n += 1
        assert len(done) == 1 and done[0].finish_reason == "tool_calls" and done[0].usage["completion_tokens"] == 9, done
        n += 1
        u = done[0].usage
        assert u["reasoning_deltas"] == 1 and u["reasoning_chars"] > 10 and done[0].ttft_s is not None, u
        n += 1
        assert len(fb.requests) == 2, len(fb.requests)
        assert "stream_options" in fb.requests[0]["body"] and "stream_options" not in fb.requests[1]["body"], "stream_options fallback"
        n += 1
        b1 = fb.requests[1]["body"]
        assert b1["disable_reasoning"] is True and b1["tool_choice"] == "auto" and b1["max_tokens"] == 50, b1
        n += 1
        h0 = fb.requests[0]["headers"]
        assert h0["user-agent"] == "eva-voice-agent/0.1" and h0["authorization"] == "Bearer k", h0
        n += 1
        # 429 then success
        events = [ev async for ev in llm.stream([{"role": "user", "content": "x"}])]
        assert "".join(e.text for e in events if isinstance(e, LLMDelta)) == "Hi." and len(fb.requests) == 4, "429 retry"
        n += 1
        # double 500 -> raises LLMHTTPError
        try:
            async for _ in llm.stream([{"role": "user", "content": "x"}]):
                pass
            raise AssertionError("expected LLMHTTPError")
        except LLMHTTPError as e:
            assert e.status == 500 and len(fb.requests) == 6, e
        n += 1
        # complete() strips think
        assert await llm.complete([{"role": "user", "content": "x"}]) == "Summary."
        n += 1
        assert all(r["line"].startswith("POST /v1/chat/completions") for r in fb.requests), fb.requests[0]["line"]
        n += 1
    finally:
        await llm.close()
        await fb.stop()
    return n


# --------------------------------------------------------------- model bench
_MD_RE = re.compile(r"(\*\*|__|^#{1,6}\s|^\s*[-*•]\s|^\s*\d+[.)]\s|`)", re.M)
_TAG_RE = re.compile(r"\[[^\]]+\]")
_STAGE_RE = re.compile(r"\*[^*\n]+\*")


def leak_report(texts: list[str]) -> dict[str, int]:
    rep = {"markdown": 0, "emoji": 0, "think": 0, "bracket_tags": 0, "stage_directions": 0, "sanitizer_changed": 0, "turns": len(texts)}
    for t in texts:
        rep["sanitizer_changed"] += clean_for_tts(t, False) != " ".join(t.split())
        rep["markdown"] += bool(_MD_RE.search(t))
        rep["emoji"] += bool(_EMOJI_RE.search(t))
        rep["think"] += "<think" in t.lower()
        rep["bracket_tags"] += bool(_TAG_RE.search(t))
        rep["stage_directions"] += bool(_STAGE_RE.search(t))
    return rep


async def collect(
    llm: OpenAICompatLLM,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    chunker: SentenceChunker | None = None,
) -> tuple[str, list[LLMToolCall], LLMDone, list[tuple[float, str]]]:
    """Drain one stream; return (text, tool calls, done, [(t_ready, chunk), ...])."""
    text_parts: list[str] = []
    calls: list[LLMToolCall] = []
    done: LLMDone | None = None
    chunks: list[tuple[float, str]] = []
    t0 = time.perf_counter()
    n_done = 0
    async for ev in llm.stream(messages, tools):
        if isinstance(ev, LLMDelta):
            text_parts.append(ev.text)
            if chunker is not None:
                for c in chunker.feed(ev.text):
                    chunks.append((time.perf_counter() - t0, c))
        elif isinstance(ev, LLMToolCall):
            calls.append(ev)
        elif isinstance(ev, LLMDone):
            n_done += 1
            done = ev
    assert done is not None and n_done == 1, f"expected exactly one LLMDone, got {n_done}"
    if chunker is not None:
        for c in chunker.flush():
            chunks.append((time.perf_counter() - t0, c))
    return "".join(text_parts), calls, done, chunks


def build(spec: dict[str, Any]) -> OpenAICompatLLM:
    if spec["kind"] == "cerebras":
        return make_cerebras(spec["model"], load_keys(), spec.get("reasoning"))
    return make_ollama(spec["model"])


async def bench_model(key: str, spec: dict[str, Any], n_turns: int) -> dict[str, Any]:
    llm = build(spec)
    res: dict[str, Any] = {"key": key, "name": llm.name, "extra_body": llm.extra_body, "errors": []}
    print(f"\n=== {llm.name}  extra_body={llm.extra_body}")
    try:
        # ---------------------------------------------------------- warmup
        await llm.warmup()
        res["warmup_s"] = round(llm.warmup_s or 0.0, 3)
        print(f"warmup: {res['warmup_s']:.3f} s")

        # ------------------------------------------------------ chat turns
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        turns: list[dict[str, Any]] = []
        texts: list[str] = []
        for i, user in enumerate(CHAT_TURNS[:n_turns], 1):
            messages.append({"role": "user", "content": user})
            text, calls, done, _ = await collect(llm, messages)
            messages.append({"role": "assistant", "content": text})
            texts.append(text)
            ct = done.usage.get("completion_tokens")
            gen_s = (done.total_s - (done.ttft_s or 0)) or 1e-9
            tps = round(ct / gen_s, 1) if ct else None
            ti = done.usage.get("time_info") or {}
            turn = {
                "i": i,
                "user": user,
                "reply": text,
                "clean": clean_for_tts(text, False),
                "ttft_s": round(done.ttft_s, 3) if done.ttft_s is not None else None,
                "total_s": round(done.total_s, 3),
                "completion_tokens": ct,
                "prompt_tokens": done.usage.get("prompt_tokens"),
                "reasoning_tokens": (done.usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                "reasoning_deltas": done.usage.get("reasoning_deltas"),
                "tok_per_s": tps,
                "finish_reason": done.finish_reason,
                "server_queue_s": ti.get("queue_time"),
                "server_total_s": ti.get("total_time"),
            }
            turns.append(turn)
            turn["gen_s"] = round(gen_s, 3)
            print(f"turn {i}: ttft={turn['ttft_s']} s total={turn['total_s']} s gen={gen_s:.3f} s tokens={ct} ({tps} tok/s) finish={done.finish_reason}")
            print(f"   user: {user}")
            print(f"   eva : {text!r}")
            if turn["clean"] != text.strip():
                print(f"   tts : {turn['clean']!r}")
        res["turns"] = turns
        ttfts = [t["ttft_s"] for t in turns if t["ttft_s"] is not None]
        totals = [t["total_s"] for t in turns]
        res["ttft_median_s"] = round(statistics.median(ttfts), 3) if ttfts else None
        res["ttft_min_s"] = round(min(ttfts), 3) if ttfts else None
        res["ttft_max_s"] = round(max(ttfts), 3) if ttfts else None
        res["total_median_s"] = round(statistics.median(totals), 3) if totals else None
        tpss = [t["tok_per_s"] for t in turns if t["tok_per_s"]]
        res["tok_per_s_median"] = round(statistics.median(tpss), 1) if tpss else None
        res["leaks_chat"] = leak_report(texts)
        print(f"chat: ttft median={res['ttft_median_s']} s (min {res['ttft_min_s']}, max {res['ttft_max_s']}), total median={res['total_median_s']} s, {res['tok_per_s_median']} tok/s")

        # ---------------------------------------------------- tool exchange
        tool_list = eva_tools.get_tools()
        tschemas = eva_tools.schemas(tool_list)
        tmsgs: list[dict[str, Any]] = [{"role": "system", "content": TOOL_PROMPT}, {"role": "user", "content": TOOL_TURN}]
        tool_log: list[dict[str, Any]] = []
        rounds: list[dict[str, Any]] = []
        final_text = ""
        t_start = time.perf_counter()
        for rnd in range(1, 5):
            text, calls, done, _ = await collect(llm, tmsgs, tschemas)
            rounds.append(
                {
                    "round": rnd,
                    "ttft_s": round(done.ttft_s, 3) if done.ttft_s is not None else None,
                    "total_s": round(done.total_s, 3),
                    "finish_reason": done.finish_reason,
                    "calls": [{"name": c.name, "arguments": c.arguments, "id": c.id} for c in calls],
                    "text": text,
                }
            )
            print(f"tool round {rnd}: ttft={rounds[-1]['ttft_s']} s total={done.total_s:.3f} s finish={done.finish_reason} calls={[(c.name, c.arguments) for c in calls]} text={text!r}")
            if not calls:
                final_text = text
                break
            tmsgs.append(assistant_tool_call_message(text, calls))
            for c in calls:
                result = await eva_tools.execute(c, tool_list)
                tool_log.append({"name": c.name, "arguments": c.arguments, "result": result})
                print(f"   exec {c.name}({c.arguments}) -> {result}")
                tmsgs.append({"role": "tool", "tool_call_id": c.id, "content": result})
        called = [c["name"] for c in tool_log]
        res["tool_exchange"] = {
            "rounds": rounds,
            "executed": tool_log,
            "final_reply": final_text,
            "final_clean": clean_for_tts(final_text, False),
            "wall_s": round(time.perf_counter() - t_start, 3),
            "called_time": "get_current_time" in called,
            "called_timer": "set_timer" in called,
            "timer_args_ok": any(c["name"] == "set_timer" and c["arguments"].get("seconds") == 120 and "tea" in str(c["arguments"].get("label", "")).lower() for c in tool_log),
            "ok": bool(final_text) and "get_current_time" in called and "set_timer" in called,
        }
        te = res["tool_exchange"]
        print(f"tool exchange: ok={te['ok']} time={te['called_time']} timer={te['called_timer']} args_ok={te['timer_args_ok']} wall={te['wall_s']} s")
        print(f"   final: {final_text!r}")
        # cancel the tea timer so it does not fire after the bench ends
        for t in eva_tools.active_timers():
            if t["label"].lower() == "tea":
                t["handle"].cancel()

        # ---------------------------------------------------------- chunker
        cmsgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": CHUNKER_TURN}]
        text, _, done, chunks = await collect(llm, cmsgs, None, SentenceChunker())
        res["chunker"] = {
            "ttft_s": round(done.ttft_s, 3) if done.ttft_s is not None else None,
            "total_s": round(done.total_s, 3),
            "first_chunk_ready_s": round(chunks[0][0], 3) if chunks else None,
            "n_chunks": len(chunks),
            "chunks": [(round(t, 3), c) for t, c in chunks],
            "text": text,
        }
        print(f"chunker: ttft={res['chunker']['ttft_s']} s first chunk ready at {res['chunker']['first_chunk_ready_s']} s, {len(chunks)} chunks, total {done.total_s:.3f} s")
        for t, c in chunks:
            print(f"   [{t:6.3f}s] {c!r}  ->  {clean_for_tts(c, False)!r}")
        res["leaks_all"] = leak_report(texts + [final_text, text])
        print(f"leaks (chat+tool+chunker replies): {res['leaks_all']}")
    except Exception as e:  # keep going with the other models
        logging.exception("bench failed for %s", llm.name)
        res["errors"].append(repr(e))
    finally:
        await llm.close()
    return res


def summary_table(results: list[dict[str, Any]]) -> str:
    cols = ["model", "warmup", "ttft med", "ttft min/max", "total med", "tok/s", "tools ok", "1st chunk", "md/emoji/think/tags/sanitized"]
    rows = []
    for r in results:
        if r.get("errors") and "turns" not in r:
            rows.append([r["name"], "ERR", "-", "-", "-", "-", "-", "-", r["errors"][0][:40]])
            continue
        te = r.get("tool_exchange") or {}
        lk = r.get("leaks_all") or {}
        ch = r.get("chunker") or {}
        rows.append(
            [
                r["name"],
                f"{r.get('warmup_s', 0):.2f}s",
                f"{r.get('ttft_median_s')}s",
                f"{r.get('ttft_min_s')}/{r.get('ttft_max_s')}s",
                f"{r.get('total_median_s')}s",
                str(r.get("tok_per_s_median")),
                f"{'yes' if te.get('ok') else 'NO'} ({len(te.get('rounds', []))}r, {te.get('wall_s')}s)",
                f"{ch.get('first_chunk_ready_s')}s/{ch.get('n_chunks')}ch",
                f"{lk.get('markdown', '?')}/{lk.get('emoji', '?')}/{lk.get('think', '?')}/{lk.get('bracket_tags', '?')}/{lk.get('sanitizer_changed', '?')} of {lk.get('turns', '?')}",
            ]
        )
    widths = [max(len(str(x)) for x in col) for col in zip(cols, *rows)] if rows else [len(c) for c in cols]
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    lines = [fmt.format(*cols), fmt.format(*["-" * w for w in widths])]
    lines += [fmt.format(*[str(x) for x in row]) for row in rows]
    return "\n".join(lines)


async def run(models: list[str], n_turns: int, skip_unit: bool) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not skip_unit:
        n = unit_tests_sync()
        print(f"unit (sync): {n} checks passed")
        n2 = await unit_tests_async()
        print(f"unit (async): {n2} checks passed")
        n3 = await fake_backend_tests()
        print(f"unit (fake backend: stream_options fallback, 429 retry, 5xx, split tool args, think leak): {n3} checks passed")
    for key in models:
        spec = MODEL_SPECS[key]
        results.append(await bench_model(key, spec, n_turns))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(MODEL_SPECS), help="comma-separated subset of " + ", ".join(MODEL_SPECS))
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--skip-unit", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR / "test_llm_results.json"))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in MODEL_SPECS]
    if unknown:
        ap.error(f"unknown models {unknown}; choose from {list(MODEL_SPECS)}")
    t0 = time.perf_counter()
    results = asyncio.run(run(models, args.turns, args.skip_unit))
    print("\n" + "=" * 100)
    print(summary_table(results))
    print(f"\nbench wall time {time.perf_counter() - t0:.1f} s")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
