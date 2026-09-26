"""Expressive voices behind the local voice server (``voice/server.py``, run by ``.venv-voice``).

The server holds the PyTorch models (Orpheus' SNAC decoder, Chatterbox) in their own
environment, so their pinned dependencies never touch the main one. This client is what
the pipeline sees: a TTS that streams int16 PCM. It

* rewrites Eva's generic inline sounds (``[laughs]``, ``[sighs]``) into the engine's own
  spelling and drops any other bracketed text;
* maps her delivery cue (``[excited]``, ``[soft]`` ...) onto the engine's controls where it
  has any (Chatterbox's ``exaggeration`` / ``cfg_weight``);
* levels loudness per engine and voice (``eva.audio.leveler``);
* starts the server when it isn't running, and stops it on close when it started it.
"""
from __future__ import annotations

import asyncio
import logging
import re
import subprocess
import time
from typing import Any, AsyncIterator

import httpx

from ..audio.leveler import Leveler
from ..config import ROOT, VOICE_SERVER_URL
from ..delivery import normalise_tag

log = logging.getLogger(__name__)

SERVER_PY = ROOT / "voice" / "server.py"
SERVER_PYTHON = ROOT / ".venv-voice" / "Scripts" / "python.exe"
SERVER_LOG = ROOT / "voice" / "server.log"

# Chatterbox's emotion controls per cue. Starting points, not measurements: higher
# exaggeration is more expressive and faster, lower cfg_weight slows the pace back down
# (Resemble's own guidance). Tune by ear.
CUE_EXAGGERATION: dict[str, dict[str, float]] = {
    "warm": {"exaggeration": 0.55, "cfg_weight": 0.5},
    "soft": {"exaggeration": 0.40, "cfg_weight": 0.5},
    "gentle": {"exaggeration": 0.40, "cfg_weight": 0.5},
    "quiet": {"exaggeration": 0.35, "cfg_weight": 0.5},
    "sad": {"exaggeration": 0.45, "cfg_weight": 0.4},
    "thoughtful": {"exaggeration": 0.45, "cfg_weight": 0.45},
    "slow": {"exaggeration": 0.40, "cfg_weight": 0.35},
    "bright": {"exaggeration": 0.70, "cfg_weight": 0.4},
    "playful": {"exaggeration": 0.70, "cfg_weight": 0.4},
    "teasing": {"exaggeration": 0.70, "cfg_weight": 0.4},
    "excited": {"exaggeration": 0.85, "cfg_weight": 0.3},
    "curious": {"exaggeration": 0.60, "cfg_weight": 0.45},
    "amused": {"exaggeration": 0.65, "cfg_weight": 0.45},
    "serious": {"exaggeration": 0.40, "cfg_weight": 0.5},
    "flat": {"exaggeration": 0.30, "cfg_weight": 0.5},
}

# Per engine: output rate, how each generic sound is spelled, which sounds the persona
# offers the brain (few, so a small brain doesn't sprinkle them), and the cue controls.
ENGINES: dict[str, dict[str, Any]] = {
    "orpheus": {
        "sample_rate": 24_000,
        "ollama_model": "legraphista/Orpheus:3b-ft-q4_k_m",  # the same as Orpheus.model in voice/server.py
        "sounds": {"laughs": "<laugh>", "laughing": "<laugh>", "chuckles": "<chuckle>", "giggles": "<chuckle>",
                   "sighs": "<sigh>", "exhales": "<sigh>", "gasps": "<gasp>", "groans": "<groan>",
                   "yawns": "<yawn>", "sniffs": "<sniffle>", "coughs": "<cough>"},
        "offer": ["laughs", "chuckles", "sighs", "gasps"],
        "cues": None,
    },
    "chatterbox-turbo": {
        "sample_rate": 24_000,
        "sounds": {"laughs": "[laugh]", "laughing": "[laugh]", "chuckles": "[chuckle]", "giggles": "[chuckle]",
                   "coughs": "[cough]"},
        "offer": ["laughs", "chuckles"],
        "cues": None,
    },
    "chatterbox": {"sample_rate": 24_000, "sounds": {}, "offer": [], "cues": CUE_EXAGGERATION},
}

_TAG_RE = re.compile(r"\[([^\[\]]{1,30})\]")
_SPACES_RE = re.compile(r"[ \t]{2,}")


def engine_text(text: str, sounds: dict[str, str]) -> str:
    """``text`` with known sounds in the engine's spelling and every other ``[tag]`` dropped."""

    def swap(m: re.Match[str]) -> str:
        return sounds.get(normalise_tag(m.group(1)), "")

    return _SPACES_RE.sub(" ", _TAG_RE.sub(swap, text)).strip()


class VoiceServerTTS:
    """One engine and voice of the voice server, as a streaming TTS."""

    def __init__(self, engine: str, voice: str, base_url: str = VOICE_SERVER_URL, *,
                 level_dbfs: float | None = -19.0, start_server: bool = True) -> None:
        if engine not in ENGINES:
            raise ValueError(f"unknown voice-server engine {engine!r}; known: {', '.join(ENGINES)}")
        spec = ENGINES[engine]
        self.engine = engine
        self.voice = voice
        self.base_url = base_url.rstrip("/")
        self.name = f"{engine}/{voice}"
        self.sample_rate: int = spec["sample_rate"]
        self.sound_map: dict[str, str] = spec["sounds"]
        self.sound_tags: list[str] = list(spec["offer"])  # what the persona offers the brain
        self.supports_audio_tags: bool = bool(self.sound_map)
        self.cue_map: dict[str, dict[str, float]] | None = spec["cues"]
        self.supports_cues: bool = self.cue_map is not None
        self.ollama_model: str | None = spec.get("ollama_model")  # held in Ollama next to the brain
        self.start_server = start_server
        self._leveler = Leveler(self.sample_rate, target_dbfs=level_dbfs) if level_dbfs is not None else None
        self._client: httpx.AsyncClient | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self.load_s: float | None = None

    # --------------------------------------------------------------- lifecycle
    async def _healthy(self) -> bool:
        assert self._client is not None
        try:
            return (await self._client.get(f"{self.base_url}/health", timeout=2.0)).status_code == 200
        except httpx.HTTPError:
            return False

    async def warmup(self) -> None:
        """Start the server if needed, load the engine (first time: downloads its weights)
        and render one short line so the first real reply pays no CUDA start-up."""
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))
        if not await self._healthy():
            if not self.start_server:
                raise RuntimeError(f"voice server not reachable at {self.base_url}")
            if not SERVER_PYTHON.exists():
                raise RuntimeError(f"{SERVER_PYTHON} missing: the voice environment isn't installed (see README)")
            port = self.base_url.rsplit(":", 1)[-1]
            log.info("starting the voice server: %s (log %s)", SERVER_PY, SERVER_LOG)
            logf = open(SERVER_LOG, "ab")  # noqa: SIM115 - handed to the child process
            self._proc = subprocess.Popen([str(SERVER_PYTHON), str(SERVER_PY), "--port", port],
                                          cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 90
            while not await self._healthy():
                if self._proc.poll() is not None:
                    raise RuntimeError(f"voice server exited with code {self._proc.returncode}; see {SERVER_LOG}")
                if time.monotonic() > deadline:
                    raise RuntimeError(f"voice server did not come up in 90 s; see {SERVER_LOG}")
                await asyncio.sleep(0.5)
        t0 = time.perf_counter()
        r = await self._client.post(f"{self.base_url}/load", json={"engine": self.engine, "voice": self.voice},
                                    timeout=httpx.Timeout(3600.0, connect=5.0))
        if r.status_code != 200:
            raise RuntimeError(f"voice server could not load {self.name}: {r.text[:300]}")
        self.load_s = time.perf_counter() - t0
        async for _ in self.synthesize("Hi."):
            pass

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()  # frees the GPU; the next session starts it again
            try:
                await asyncio.to_thread(self._proc.wait, 10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    # --------------------------------------------------------------- synthesis
    def params_for(self, cue: str | None) -> dict[str, float]:
        if not self.cue_map or not cue:
            return {}
        return dict(self.cue_map.get(normalise_tag(cue), {}))

    async def synthesize(self, text: str, cue: str | None = None) -> AsyncIterator[bytes]:
        """Stream int16 PCM for ``text``. Leaving the loop early closes the request, and the
        server stops generating."""
        if self._client is None:
            raise RuntimeError("VoiceServerTTS.warmup() must be awaited before synthesize()")
        body = engine_text(text, self.sound_map)
        if not re.search(r"[A-Za-z0-9]", body):
            return
        payload = {"engine": self.engine, "voice": self.voice, "text": body, "params": self.params_for(cue)}
        clip = self._leveler.begin(self.name) if self._leveler else None
        async with self._client.stream("POST", f"{self.base_url}/speak", json=payload,
                                       timeout=httpx.Timeout(120.0, connect=5.0)) as r:
            if r.status_code != 200:
                raise RuntimeError(f"voice server {r.status_code}: {(await r.aread())[:300]!r}")
            async for chunk in r.aiter_bytes():
                out = clip.process(chunk) if clip else chunk
                if out:
                    yield out
        if clip:
            clip.finish()


__all__ = ["VoiceServerTTS", "ENGINES", "CUE_EXAGGERATION", "engine_text"]
