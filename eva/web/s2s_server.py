"""The phone page (eva/web/static/index.html) talking to a speech-to-speech model instead of Eva's loop.

The phone sends its mic and plays what comes back, exactly as with Eva's loop (same page, same wire
protocol, eva/web/transport.py); this laptop relays the audio to the model (GPT-Live, OpenAI Realtime,
Gemini Live), runs the job's tools for it, shows the transcript on the phone and prints the usual dim
lines here. One call at a time.

    python run_s2s.py openai-live:gpt-live-1 --web      # https://<laptop-ip>:8443 on the phone
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from ..audio.mic import LinearResampler
from .server import STATIC, WebServer, ensure_certificate, lan_ip
from .transport import WebMic, WebPlayer

log = logging.getLogger("eva.web.s2s")
GREETING = "[The phone line just connected. Answer it the way a dispatcher does: one short sentence with the company and your name.]"


class S2SWebServer(WebServer):
    def __init__(self, make_call: Callable[..., Any], *, printer: Callable[[str, dict[str, Any]], None],
                 log_dir: Path, name: str) -> None:
        self.make_call = make_call  # (player, on_event, log) -> (Call, client)
        self.printer = printer
        self.log_dir = log_dir
        self.name = name
        self._busy = False
        self._page = (STATIC / "index.html").read_bytes()
        self._cert: Path | None = None

    async def _conversation(self, ws: ServerConnection) -> None:
        hello: dict[str, Any] = {}
        async for first in ws:
            if isinstance(first, bytes):
                continue
            with contextlib.suppress(ValueError):
                hello = json.loads(first)
            if hello.get("type") == "hello":
                break
        mic = WebMic(int(hello.get("sampleRate") or 48000))
        player: WebPlayer | None = None
        heard: list[str] = []
        said: list[str] = []
        log_path = self.log_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_phone_{self.name}.jsonl"
        log_file = open(log_path, "a", encoding="utf-8")  # noqa: SIM115 - closed below
        t0 = time.perf_counter()

        def write_log(ev: dict[str, Any]) -> None:
            row = {"t": round(time.perf_counter() - t0, 3), **{k: v for k, v in ev.items() if k != "pcm"}}
            if ev.get("type") == "audio":
                row["bytes"] = len(ev["pcm"])
            log_file.write(json.dumps(row, default=str) + "\n")

        def to_page(name: str, data: dict[str, Any]) -> None:
            if player is not None:
                player.send_event(name, data)

        def flush_heard() -> None:
            text = "".join(heard).strip()
            heard.clear()
            if text:
                to_page("stt", {"text": text})

        def on_event(name: str, data: dict[str, Any]) -> None:
            self.printer(name, data)
            if name in ("tool", "tool_result", "barge_in", "error", "turn_done"):
                write_log({"type": f"call.{name}", **{k: v for k, v in data.items() if k != "turn"}})
            if name == "text_in":
                heard.clear()
                to_page("stt", {"text": data["text"]})
            elif name == "text_in_part":
                heard.append(data["text"])
            elif name == "first_audio":
                flush_heard()
                to_page("state", {"to": "speaking"})
            elif name == "text_out":
                said.append(data["delta"])
            elif name == "tool":
                flush_heard()
                to_page("tool_call", {"name": data["name"], "arguments": data["args"]})
            elif name == "barge_in":
                to_page("barge_in", {})
            elif name == "turn_done":
                flush_heard()
                text = "".join(said).strip()
                said.clear()
                if text:
                    to_page("turn", {"assistant_text": text, "interrupted": False})
                to_page("state", {"to": "listening"})
                if player is not None:
                    player.end_turn()
            elif name == "error":
                to_page("error", {"where": self.name, "error": data.get("message", "")})

        call, client = self.make_call(None, on_event, write_log)
        player = WebPlayer(ws.send, client.out_rate)
        call.player = player
        if hello.get("prebufferS"):
            player.prebuffer_s = float(hello["prebufferS"])
            player.latency_s = 0.2 + player.prebuffer_s
        player.start()
        self.printer("web_hello", {"ua": str(hello.get("ua", ""))[:60]})
        up = LinearResampler(16000, client.in_rate) if client.in_rate != 16000 else None

        async def forward() -> None:
            async for frame in mic.frames():
                pcm = frame if up is None else np.clip(up.process(frame), -32768, 32767).astype(np.int16)
                await client.send_audio(pcm.astype("<i2").tobytes())

        fwd: asyncio.Task[None] | None = None
        try:
            await call.start()
            await client.send_text(GREETING)
            fwd = asyncio.create_task(forward())
            async for msg in ws:
                if isinstance(msg, bytes):
                    mic.push(msg)
                    continue
                with contextlib.suppress(ValueError):
                    m = json.loads(msg)
                    kind = m.get("type")
                    if kind == "ping":
                        await ws.send(json.dumps({"type": "pong", "t": m.get("t")}))
                    elif kind == "stopped":
                        player.note_stopped(int(m.get("played") or 0))
                    elif kind == "interrupt":
                        player.stop()
                    elif kind == "bye":
                        break
                if call.closed or call.ended:
                    await asyncio.sleep(max(0.5, player.buffered_seconds + 0.3))
                    break
        except ConnectionClosed:
            pass
        finally:
            mic.stop()
            if fwd is not None:
                fwd.cancel()
            await call.close()
            player.close()
            log_file.close()
            with contextlib.suppress(Exception):
                await ws.close()
            self.printer("call_over", {"cost": call.cost, "turns": len(call.turns), "log": str(log_path)})


async def serve_s2s(make_call: Callable[..., Any], *, printer: Callable[[str, dict[str, Any]], None], name: str,
                    log_dir: Path, port: int = 8443, tls: bool = True) -> None:
    server = S2SWebServer(make_call, printer=printer, log_dir=log_dir, name=name)
    ip = lan_ip()
    ssl_ctx = None
    if tls:
        import ssl

        cert, key = ensure_certificate(ip)
        server._cert = cert
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(str(cert), str(key))
    scheme = "https" if tls else "http"
    printer("web_ready", {"url": f"{scheme}://{ip}:{port}", "local": f"{scheme}://localhost:{port}"})
    async with serve(server.handle, "0.0.0.0", port, ssl=ssl_ctx, process_request=server.process_request,
                     max_size=2**20, ping_interval=20, ping_timeout=20):
        await asyncio.Future()


__all__ = ["S2SWebServer", "serve_s2s"]
