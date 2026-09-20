"""Serve Eva to a browser (the phone): the page and the audio socket on one port.

    python run.py --web                 # http://<laptop-ip>:8080  (mic works only from the laptop's own browser)
    python run.py --web --tls           # https://<laptop-ip>:8443 with a self-signed certificate (the phone)

Browsers allow the microphone only on ``https://`` (or ``localhost``), so for the phone
the server needs TLS. ``--tls`` generates a self-signed certificate for this machine's
LAN address with openssl (Git for Windows ships one) under ``models/web/`` the first
time; the phone must accept it once (Safari: "Show details" -> "visit this website";
Chrome on Android needs the certificate installed, or the URL added to
``chrome://flags/#unsafely-treat-insecure-origin-as-secure`` after which plain
``http://`` works). A tunnel with a real certificate (``cloudflared tunnel --url
http://localhost:8080``) is the zero-setup alternative and also works over LTE.

One conversation at a time: a second client gets ``{"type": "busy"}``. Each connection
builds a fresh ``VoiceAgent`` over the shared providers (warmed once at start), runs it
until the socket closes or she ends the call, then updates memory like the CLI does.
"""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import http
import json
import logging
import shutil
import socket
import ssl
import subprocess
import time
from pathlib import Path
from typing import Any

from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Response

from ..audio.vad import UtteranceSegmenter
from ..config import MODELS_DIR, PipelineSettings
from ..pipeline import VoiceAgent
from ..session import Session
from .transport import WebMic, WebPlayer

log = logging.getLogger("eva.web")
STATIC = Path(__file__).resolve().parent / "static"
CERT_DIR = MODELS_DIR / "web"
UI_EVENTS = frozenset(
    {
        "stt", "turn", "state", "filler", "barge_in", "barge_in_echo", "stt_echo", "stt_hesitation", "stt_phantom",
        "utterance_carried", "language", "tool_call", "failover", "recovered", "echo_storm", "session_end", "error",
    }
)


# --------------------------------------------------------------------- network helpers
def lan_ip() -> str:
    """This machine's address on the local network (the one a phone would use)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return str(s.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def find_openssl() -> str | None:
    found = shutil.which("openssl")
    if found:
        return found
    for cand in (
        r"C:\Program Files\Git\usr\bin\openssl.exe",
        r"C:\Program Files\Git\mingw64\bin\openssl.exe",
        r"C:\Program Files (x86)\Git\usr\bin\openssl.exe",
    ):
        if Path(cand).exists():
            return cand
    return None


def ensure_certificate(ip: str) -> tuple[Path, Path]:
    """``models/web/cert.pem`` + ``key.pem`` for ``ip`` (and localhost); generated once with openssl."""
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    cert, key, meta = CERT_DIR / "cert.pem", CERT_DIR / "key.pem", CERT_DIR / "cert.json"
    if cert.exists() and key.exists() and meta.exists():
        try:
            if json.loads(meta.read_text())["ip"] == ip:
                return cert, key
        except (ValueError, KeyError):
            pass
    exe = find_openssl()
    if exe is None:
        raise RuntimeError("openssl not found (Git for Windows ships one under C:\\Program Files\\Git\\usr\\bin); cannot make a certificate")
    subprocess.run(
        [
            exe, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-sha256", "-days", "825",
            "-keyout", str(key), "-out", str(cert), "-subj", "/CN=eva.local",
            "-addext", f"subjectAltName=IP:{ip},DNS:localhost,IP:127.0.0.1",
        ],
        check=True, capture_output=True,
    )
    meta.write_text(json.dumps({"ip": ip, "made": time.time()}))
    log.info("made a self-signed certificate for %s under %s", ip, CERT_DIR)
    return cert, key


# ------------------------------------------------------------------------- the server
class WebServer:
    def __init__(
        self, session: Session, settings: PipelineSettings, *, user_name: str, greeting: bool, printer: Any,
        gated_settings: PipelineSettings | None = None,
    ) -> None:
        self.session = session
        self.settings = settings  # for a phone that cancels its own echo (the default)
        self.gated_settings = gated_settings or settings  # for a phone with ?aec=0: the laptop's echo defences
        self.user_name = user_name
        self.greeting = greeting
        self.printer = printer
        self._busy = False
        self._page = (STATIC / "index.html").read_bytes()
        self._cert: Path | None = None

    # -- HTTP: the page, the certificate ------------------------------------------------
    @staticmethod
    def _http(status: http.HTTPStatus, content_type: str, body: bytes, **extra: str) -> Response:
        headers = Headers()
        headers["Content-Type"] = content_type
        headers["Content-Length"] = str(len(body))
        headers["Cache-Control"] = "no-store"
        headers["Connection"] = "close"
        for k, v in extra.items():
            headers[k.replace("_", "-")] = v
        return Response(status.value, status.phrase, headers, body)

    def process_request(self, connection: ServerConnection, request: Any) -> Any:
        path = request.path.split("?", 1)[0]
        if path == "/ws":
            return None  # continue with the WebSocket handshake
        if path in ("/", "/index.html"):
            return self._http(http.HTTPStatus.OK, "text/html; charset=utf-8", self._page)
        if path == "/cert.pem" and self._cert is not None:
            return self._http(
                http.HTTPStatus.OK, "application/x-x509-ca-cert", self._cert.read_bytes(),
                Content_Disposition='attachment; filename="eva-cert.pem"',
            )
        return self._http(http.HTTPStatus.NOT_FOUND, "text/plain", b"not found")

    # -- one conversation per socket ----------------------------------------------------
    async def handle(self, ws: ServerConnection) -> None:
        peer = getattr(ws, "remote_address", None)
        if self._busy:
            await ws.send(json.dumps({"type": "busy"}))
            await ws.close()
            return
        self._busy = True
        log.info("client connected: %s", peer)
        self.printer("web_client", {"peer": str(peer), "state": "connected"})
        try:
            await self._conversation(ws)
        finally:
            self._busy = False
            self.printer("web_client", {"peer": str(peer), "state": "gone"})

    async def _conversation(self, ws: ServerConnection) -> None:
        s = self.session
        # the client's hello decides the settings: without browser echo cancelling the
        # laptop's echo defences come back on
        hello: dict[str, Any] = {}
        async for first in ws:
            if isinstance(first, bytes):
                continue
            try:
                hello = json.loads(first)
            except ValueError:
                continue
            if hello.get("type") == "hello":
                break
        settings = self.settings if hello.get("aec", True) else self.gated_settings
        mic = WebMic(int(hello.get("sampleRate") or 48000))
        player = WebPlayer(ws.send, s.tts.sample_rate)
        player.room_tone_dbfs = settings.room_tone_dbfs
        if hello.get("prebufferS"):
            player.prebuffer_s = float(hello["prebufferS"])
            player.latency_s = 0.2 + player.prebuffer_s
        player.start()
        segmenter = UtteranceSegmenter(settings)
        log.info("hello: mic %s Hz, aec=%s, prebuffer %.2f s, %s", mic.sample_rate, hello.get("aec", True), player.prebuffer_s, str(hello.get("ua", ""))[:80])
        self.printer("web_hello", {"aec": hello.get("aec", True), "prebuffer_s": player.prebuffer_s, "gates": settings.barge_in_confirm == "words"})

        def on_event(name: str, data: dict[str, Any]) -> None:
            self.printer(name, data)
            if name in UI_EVENTS:
                player.send_event(name, data)

        agent = VoiceAgent(
            s.stt, s.llm, s.tts, s.system_prompt, s.tools, settings,
            frames=mic.frames(), segmenter=segmenter, player=player,
            fillers=s.fillers, tool_hints=s.tool_hints, backchannels=s.backchannels, on_event=on_event,
        )
        await agent.prepare()
        agent._select_lang(s.plan.primary.code)
        if self.greeting:
            # launch it now: run() only polls its event queue between mic frames
            agent.pending_events.put_nowait(s.greeting_event(self.user_name))
            await agent.poll_pending_events()
        run_task: asyncio.Task[Any] | None = asyncio.create_task(agent.run(), name="eva-web-run")
        try:
            async for msg in ws:
                if isinstance(msg, bytes):
                    mic.push(msg)
                    continue
                try:
                    m = json.loads(msg)
                except ValueError:
                    continue
                kind = m.get("type")
                if kind == "ping":
                    await ws.send(json.dumps({"type": "pong", "t": m.get("t")}))
                elif kind == "stopped":
                    player.note_stopped(int(m.get("played") or 0))
                elif kind == "interrupt":
                    await agent.interrupt("tap")
                elif kind == "bye":
                    break
                if run_task is not None and run_task.done():
                    break  # she ended the call (end_conversation) or the loop died
        except ConnectionClosed:
            pass
        finally:
            mic.stop()
            if run_task is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(run_task, timeout=10)
            else:
                await agent.close()
            player.close()
            with contextlib.suppress(Exception):
                await ws.close()
            if agent.messages:
                try:
                    await asyncio.wait_for(
                        s.memory.update_from_transcript(s.llm, agent.messages, user_name=self.user_name), timeout=25
                    )
                    s.memory.save()
                    self.printer("memory_saved", {"facts": len(s.memory.facts)})
                except Exception as e:
                    log.warning("memory update failed: %s", e)


async def serve_web(
    session: Session,
    settings: PipelineSettings,
    *,
    host: str = "0.0.0.0",
    port: int = 8080,
    tls: bool = False,
    user_name: str = "",
    greeting: bool = True,
    echo_gates: bool | None = None,
    printer: Any,
) -> None:
    """Warm the providers, then serve until cancelled (Ctrl-C).

    ``echo_gates``: True keeps the laptop's echo gating on for the phone too; None / False turns it off.
    """
    gated = dataclasses.replace(settings)  # the laptop's echo defences, for a phone with ?aec=0
    settings = dataclasses.replace(settings)
    if echo_gates is not True:
        # The phone's browser cancels its own echo (measured on an iPhone: none reached the STT),
        # and the laptop-mic gates only produced false positives there.
        settings = dataclasses.replace(settings, barge_in_confirm="vad", echo_detector=False, self_echo_gate=False)
    if echo_gates is False:
        gated = settings
    server = WebServer(session, settings, user_name=user_name, greeting=greeting, printer=printer, gated_settings=gated)
    ip = lan_ip()
    ssl_ctx: ssl.SSLContext | None = None
    if tls:
        cert, key = ensure_certificate(ip)
        server._cert = cert
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(str(cert), str(key))
    await asyncio.gather(session.stt.warmup(), session.llm.warmup(), session.tts.warmup())
    scheme = "https" if tls else "http"
    printer("web_ready", {"url": f"{scheme}://{ip}:{port}", "local": f"{scheme}://localhost:{port}", "tls": tls,
                          "echo_gates": settings.barge_in_confirm == "words"})
    async with serve(server.handle, host, port, ssl=ssl_ctx, process_request=server.process_request, max_size=2**20, ping_interval=20, ping_timeout=20):
        await asyncio.Future()  # until cancelled


__all__ = ["serve_web", "WebServer", "lan_ip", "ensure_certificate"]
