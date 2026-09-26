#!/usr/bin/env python
"""Eva's voice server: the expressive voices that need PyTorch, in their own environment.

Runs on ``.venv-voice`` (Python 3.13, torch 2.6 + CUDA 12.4, chatterbox-tts, snac), so their
pinned dependencies never touch the main environment. ``eva.tts.voice_server`` starts it
on demand; by hand:

    .venv-voice/Scripts/python.exe voice/server.py --port 8765
    .venv-voice/Scripts/python.exe voice/server.py --say "Hey. <laugh> Long day?" --engine orpheus --voice tara --out hey.wav

Endpoints (127.0.0.1 only):
    GET  /health                                    {"ok", "loaded", "device"}
    POST /load   {"engine", "voice"}                load an engine (first time: download it); unloads the others
    POST /speak  {"engine", "voice", "text", "params"}  raw int16 mono PCM at 24 kHz, streamed (chunked)

Engines:
    orpheus           Orpheus 3B generates SNAC codes in Ollama next to the brain; this server
                      decodes them on the GPU, frame by frame. It generates at about 0.86x real
                      time on this laptop (71 tok/s against 82 needed, 2026-09-25), so each clip
                      starts after a pre-roll sized from its estimated length, and never runs dry.
    chatterbox-turbo  Resemble's 350M model cloning eva/assets/voices/<voice>.wav; [laugh] [chuckle] [cough].
    chatterbox        Resemble's 0.5B model; ``exaggeration`` / ``cfg_weight`` per request.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

# Hugging Face's xet transfer stalled at 0 bytes on this connection (2026-09-25); plain
# HTTP downloads fine. Symlinks need Developer Mode on Windows: don't warn about it.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
REFS = ROOT / "eva" / "assets" / "voices"
OLLAMA = "http://127.0.0.1:11434"  # never localhost: +2 s per request on this laptop
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Orpheus' SNAC decoder runs on the CPU: on the GPU (4.7 ms a decode) it shares the card with
# Ollama's process, and the switching cut Orpheus from 71 to 49 tok/s (63 when batched), while
# 8 CPU threads decode a 7-frame window in 45 ms, 13 % of real time (2026-09-25).
SNAC_DEVICE = os.environ.get("EVA_SNAC_DEVICE", "cpu")
torch.set_num_threads(int(os.environ.get("EVA_SNAC_THREADS", "8")))
SR = 24_000


def log(msg: str) -> None:
    print(f"[voice {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def to_pcm(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


# --------------------------------------------------------------------------- Orpheus
_EMPTY = np.zeros(0, np.float32)


class EdgeTrim:
    """Streaming silence trim: drops a clip's leading silence and always holds back the most
    recent stretch of silence (up to ``hold_s``), which speech resuming releases and the end
    of the clip drops. Orpheus pads every clip with about half a second at each end
    (measured 2026-09-25: 0.55 s at -52 dBFS before the first word), which the pipeline's
    own sentence gap would otherwise double."""

    def __init__(self, sr: int = SR, thresh_dbfs: float = -45.0, keep_s: float = 0.05,
                 hold_s: float = 0.5, tail_s: float = 0.08) -> None:
        self.hop = sr // 50  # 20 ms
        self.thresh = 10 ** (thresh_dbfs / 20)
        self.keep, self.hold, self.tail = int(keep_s * sr), int(hold_s * sr), int(tail_s * sr)
        self.started = False
        self.held = _EMPTY

    def _voiced_hops(self, a: np.ndarray) -> np.ndarray:
        n = len(a) // self.hop
        if n == 0:
            return np.zeros(0, dtype=np.int64)
        rms = np.sqrt((a[: n * self.hop].reshape(n, self.hop) ** 2).mean(axis=1))
        return np.nonzero(rms > self.thresh)[0]

    def push(self, seg: np.ndarray) -> np.ndarray:
        a = np.concatenate([self.held, seg]) if len(self.held) else seg
        self.held = _EMPTY
        voiced = self._voiced_hops(a)
        if not self.started:
            if voiced.size == 0:
                self.held = a[-self.keep :]
                return _EMPTY
            self.started = True
            a = a[max(0, int(voiced[0]) * self.hop - self.keep) :]
            voiced = self._voiced_hops(a)
        end = (int(voiced[-1]) + 1) * self.hop if voiced.size else 0  # silence after this, so far
        cut = len(a) - self.hold if len(a) - end > self.hold else end
        self.held = a[cut:]
        return a[:cut]

    def finish(self) -> np.ndarray:
        return self.held[: self.tail] if self.started else _EMPTY


class Orpheus:
    """Orpheus 3B through Ollama + SNAC 24 kHz on the GPU, streamed."""

    name = "orpheus"
    sample_rate = SR
    model = "legraphista/Orpheus:3b-ft-q4_k_m"
    voices = ("tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe")
    frame = 2048  # samples per SNAC frame (7 codes), 85 ms
    realtime_tok_s = 7 * SR / 2048  # 82 codes per second of audio

    max_prefix_frames = 5
    # Decode every 4 new frames (340 ms), not every frame: each decode is a GPU switch away from
    # Ollama's process, and 12 a second cut generation from 71 to 49 tok/s (2026-09-25) though a
    # decode itself takes 4.7 ms. The pre-roll waits about this long anyway.
    decode_every = 4

    def __init__(self) -> None:
        self.snac: Any = None
        self.tok_s = 70.0  # learned generation speed (EMA); measured 70.6-70.8 alone on the GPU
        self.s_per_word = 0.38  # learned speech length per word, silence trimmed (EMA)
        # Silent frames put in the prompt after "start of speech": Orpheus opens every clip with
        # ~0.5 s of silence it has to generate first. Prefilled, they cost a prompt read instead:
        # first voiced frame 0.22-0.36 s after the request, against 0.57-1.25 s (2026-09-25, 3 takes each).
        self.prefix: list[int] = []

    def _silent_lead(self, codes: list[int]) -> list[int]:
        """The codes of the silent frames (at most ``max_prefix_frames``) a clip opens with."""
        frames = min(len(codes) // 7, 12)
        if frames == 0:
            return []
        audio = self._decode(codes, 0, frames)
        n = 0
        for f in range(frames):
            seg = audio[f * self.frame : (f + 1) * self.frame]
            if 20 * np.log10(max(float(np.sqrt((seg ** 2).mean())), 1e-9)) > -45:
                break
            n += 1
        return codes[: min(n, self.max_prefix_frames) * 7]

    def load(self, voice: str) -> None:
        from snac import SNAC

        if self.snac is None:
            self.snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(SNAC_DEVICE)
        if not self.prefix:  # one plain take (this also loads the model in Ollama) gives the silent lead
            codes: list[int] = []
            for n in self._codes("Hi there.", voice, prefill=False):
                c = n - 10 - (len(codes) % 7) * 4096
                if 0 <= c < 4096:
                    codes.append(c)
            self.prefix = self._silent_lead(codes)
            log(f"orpheus: prefilling {len(self.prefix) // 7} silent frames")
        for _ in self.speak("Hi.", voice, {}):  # warms the prefilled path
            pass

    def unload(self) -> None:
        self.snac = None
        try:
            _post(f"{OLLAMA}/api/generate", {"model": self.model, "keep_alive": 0}, timeout=10).read()
        except OSError:
            pass

    def _codes(self, text: str, voice: str, prefill: bool = True) -> Iterator[int]:
        """The numbers N of the <custom_token_N> Ollama streams, in order (after the prefill)."""
        prompt = f"<|audio|>{voice}: {text}<|eot_id|>"
        if prefill and self.prefix:
            # end of human, start of AI, start of speech, then the silent frames as tokens
            prompt += "<custom_token_4><custom_token_5><custom_token_1>" + "".join(
                f"<custom_token_{c + 10 + (i % 7) * 4096}>" for i, c in enumerate(self.prefix))
        body = {
            "model": self.model, "prompt": prompt, "raw": True, "stream": True,
            "keep_alive": "30m",
            # sampling as Canopy Labs recommends; the stop ends the clip at "end of speech" (without it
            # the model went on into a new utterance in another voice, 2026-09-25); ctx 2k keeps it at 2.1 GB
            "options": {"temperature": 0.6, "top_p": 0.9, "repeat_penalty": 1.1, "num_predict": 2400,
                        "num_ctx": 2048, "stop": ["<custom_token_2>"]},
        }
        resp = _post(f"{OLLAMA}/api/generate", body, timeout=120)
        buf = ""
        try:
            for line in resp:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                buf += chunk.get("response", "")
                last = 0
                for m in re.finditer(r"<custom_token_(\d+)>", buf):
                    yield int(m.group(1))
                    last = m.end()
                buf = buf[last:]
                if chunk.get("done"):
                    break
        finally:
            resp.close()  # a closed request stops Ollama generating

    def _decode(self, codes: list[int], f0: int, f1: int) -> np.ndarray:
        """Audio for frames [f0, f1): each frame's 7 codes spread over SNAC's 3 layers."""
        c = torch.tensor(codes[f0 * 7 : f1 * 7], dtype=torch.int32, device=SNAC_DEVICE).view(-1, 7)
        layers = [c[:, 0].unsqueeze(0), c[:, [1, 4]].reshape(1, -1), c[:, [2, 3, 5, 6]].reshape(1, -1)]
        with torch.inference_mode():
            audio = self.snac.decode(layers)
        return audio[0, 0].float().cpu().numpy()

    def speak(self, text: str, voice: str, params: dict[str, Any]) -> Iterator[bytes]:
        voice = voice if voice in self.voices else "tara"
        words = len(re.findall(r"[A-Za-z0-9']+", text))
        est_s = self.s_per_word * words + 1.0 * text.count("<") + 0.3
        speed = min(1.0, self.tok_s / self.realtime_tok_s)
        preroll_s = est_s * (1.0 - speed) + 0.1  # enough in hand that playback never catches up
        F = self.frame
        trim = EdgeTrim()
        codes: list[int] = list(self.prefix)  # context for the first decode windows, never played
        emitted = len(codes) // 7  # frames decoded and handed to the trim
        pending: list[np.ndarray] = []
        pending_s = 0.0
        sent_s = 0.0
        started = False
        t_first: float | None = None
        for n in self._codes(text, voice):
            code = n - 10 - (len(codes) % 7) * 4096
            if not 0 <= code < 4096:  # the start / end-of-speech markers, or a stray token
                continue
            if t_first is None:
                t_first = time.perf_counter()
            codes.append(code)
            if len(codes) % 7:
                continue
            total = len(codes) // 7
            upto = total - 2  # a frame is final once two frames follow it
            if upto - emitted < self.decode_every:
                continue
            # decode the new frames with one frame of context on the left and two on the right
            w0 = max(0, emitted - 1)
            audio = self._decode(codes, w0, total)
            seg = trim.push(audio[(emitted - w0) * F : (upto - w0) * F])
            emitted = upto
            if len(seg):
                pending.append(seg)
                pending_s += len(seg) / SR
            if not started and pending_s >= preroll_s:
                started = True
            if started and pending:
                out = np.concatenate(pending)
                sent_s += len(out) / SR
                yield to_pcm(out)
                pending.clear()
        total = len(codes) // 7
        if total > emitted:
            w0 = max(0, emitted - 1)
            audio = self._decode(codes, w0, total)
            pending.append(trim.push(audio[(emitted - w0) * F :]))
        pending.append(trim.finish())
        out = np.concatenate(pending) if pending else _EMPTY
        if len(out):
            sent_s += len(out) / SR
            yield to_pcm(out)
        generated = len(codes) - len(self.prefix)
        if t_first is not None and generated > 70:
            gen_s = time.perf_counter() - t_first
            self.tok_s = 0.7 * self.tok_s + 0.3 * (generated / max(gen_s, 1e-3))
            if words and sent_s:
                self.s_per_word = 0.8 * self.s_per_word + 0.2 * (sent_s / words)


# ------------------------------------------------------------------------ Chatterbox
class ChatterboxTurbo:
    """Resemble's Chatterbox Turbo (350M), cloning a reference clip; one sentence at a time."""

    name = "chatterbox-turbo"
    sample_rate = SR

    def __init__(self) -> None:
        self.model: Any = None
        self.ref: str | None = None

    def _new_model(self) -> Any:
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        model = ChatterboxTurboTTS.from_pretrained(device=DEVICE)
        # Its loudness normalisation multiplies by a NumPy float64 gain, which under NumPy 2
        # makes the reference float64 and the tokenizer then fails ("expected scalar type
        # Double but found Float"). Keep the normalisation, hand back float32.
        norm = model.norm_loudness
        model.norm_loudness = lambda wav, sr, target_lufs=-27: np.asarray(norm(wav, sr, target_lufs), dtype=np.float32)
        return model

    def _prepare(self, voice: str) -> None:
        if self.ref != voice:
            path = REFS / f"{voice}.wav"
            if not path.exists():
                raise FileNotFoundError(f"no reference clip {path} (5 s or longer)")
            self.model.prepare_conditionals(str(path))
            self.ref = voice

    def load(self, voice: str) -> None:
        if self.model is None:
            self.model = self._new_model()
        self._prepare(voice)
        for _ in self.speak("Hi there.", voice, {}):
            pass

    def unload(self) -> None:
        self.model = None
        self.ref = None

    def _generate(self, text: str, params: dict[str, Any]) -> Any:
        return self.model.generate(text)

    def speak(self, text: str, voice: str, params: dict[str, Any]) -> Iterator[bytes]:
        self._prepare(voice)
        with torch.inference_mode():
            wav = self._generate(text, params)
        audio = wav[0].float().cpu().numpy()
        step = SR // 5  # 200 ms pieces
        for i in range(0, len(audio), step):
            yield to_pcm(audio[i : i + step])


class Chatterbox(ChatterboxTurbo):
    """Resemble's original Chatterbox (0.5B): slower, but with emotion strength per request."""

    name = "chatterbox"

    def _new_model(self) -> Any:
        from chatterbox.tts import ChatterboxTTS

        return ChatterboxTTS.from_pretrained(device=DEVICE)

    def _generate(self, text: str, params: dict[str, Any]) -> Any:
        return self.model.generate(text, exaggeration=float(params.get("exaggeration", 0.5)),
                                   cfg_weight=float(params.get("cfg_weight", 0.5)))


ENGINES: dict[str, Any] = {e.name: e for e in (Orpheus(), ChatterboxTurbo(), Chatterbox())}
_loaded: str | None = None
_gpu = threading.Lock()  # one synthesis (and one load) at a time: they share the GPU


def load(engine: str, voice: str) -> float:
    global _loaded
    t0 = time.perf_counter()
    with _gpu:
        if _loaded and _loaded != engine:
            log(f"unloading {_loaded}")
            ENGINES[_loaded].unload()
            torch.cuda.empty_cache() if DEVICE == "cuda" else None
        log(f"loading {engine} ({voice})")
        ENGINES[engine].load(voice)
        _loaded = engine
    dt = time.perf_counter() - t0
    log(f"{engine} ready in {dt:.1f} s")
    return dt


def _post(url: str, body: dict[str, Any], timeout: float) -> Any:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


# ------------------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet: one line per /speak is enough
        pass

    def _json(self, code: int, obj: dict[str, Any]) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"ok": True, "loaded": _loaded, "device": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            engine = req.get("engine", "")
            if engine not in ENGINES:
                self._json(400, {"error": f"unknown engine {engine!r}"})
                return
            voice = req.get("voice", "")
            if self.path == "/load":
                self._json(200, {"ok": True, "load_s": load(engine, voice), "sample_rate": ENGINES[engine].sample_rate})
            elif self.path == "/speak":
                self._speak(engine, voice, req.get("text", ""), req.get("params") or {})
            else:
                self._json(404, {"error": "not found"})
        except Exception as e:  # report instead of dropping the connection
            log(f"error: {e!r}")
            try:
                self._json(500, {"error": repr(e)})
            except OSError:
                pass

    def _speak(self, engine: str, voice: str, text: str, params: dict[str, Any]) -> None:
        if _loaded != engine:
            load(engine, voice)
        t_arrive = time.perf_counter()
        first: float | None = None
        n = 0
        with _gpu:
            t0 = time.perf_counter()
            waited = t0 - t_arrive  # queued behind the sentence before it
            gen = ENGINES[engine].speak(text, voice, params)
            self.send_response(200)
            self.send_header("Content-Type", "audio/L16")
            self.send_header("X-Sample-Rate", str(ENGINES[engine].sample_rate))
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                for pcm in gen:
                    if not pcm:
                        continue
                    if first is None:
                        first = time.perf_counter() - t0
                    n += len(pcm)
                    self.wfile.write(f"{len(pcm):X}\r\n".encode() + pcm + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                log(f"{engine}: client left after {n / 2 / SR:.1f} s (barge-in)")
            except Exception as e:  # the response has started: end it cleanly, keep what was sent
                log(f"{engine}: failed mid-clip after {n / 2 / SR:.1f} s: {e!r}")
                try:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except OSError:
                    pass
            finally:
                gen.close()
        audio_s = n / 2 / SR
        total = time.perf_counter() - t0
        log(f"{engine}/{voice}: queued {waited:.2f} s, then first audio {first or 0:.2f} s, {audio_s:.1f} s of audio "
            f"in {total:.1f} s (x{audio_s / total if total else 0:.2f} real time) {params or ''} {text[:60]!r}")


def say(engine: str, voice: str, text: str, out: Path, params: dict[str, Any]) -> None:
    """Render one line to a WAV file and print its timings (the listening set, reference clips)."""
    import soundfile as sf

    load(engine, voice)
    t0 = time.perf_counter()
    first: float | None = None
    parts: list[bytes] = []
    for pcm in ENGINES[engine].speak(text, voice, params):
        if first is None:
            first = time.perf_counter() - t0
        parts.append(pcm)
    total = time.perf_counter() - t0
    audio = np.frombuffer(b"".join(parts), dtype="<i2")
    sf.write(str(out), audio, ENGINES[engine].sample_rate, subtype="PCM_16")
    dur = len(audio) / ENGINES[engine].sample_rate
    print(json.dumps({"engine": engine, "voice": voice, "out": str(out), "first_audio_s": round(first or 0, 3),
                      "audio_s": round(dur, 2), "total_s": round(total, 2), "x_realtime": round(dur / total, 2) if total else None}))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--say", help="render this text to --out and exit")
    ap.add_argument("--engine", default="orpheus", choices=sorted(ENGINES))
    ap.add_argument("--voice", default="tara")
    ap.add_argument("--out", type=Path, default=Path("say.wav"))
    ap.add_argument("--params", default="{}", help='JSON, e.g. \'{"exaggeration": 0.8}\'')
    args = ap.parse_args()
    if args.say:
        say(args.engine, args.voice, args.say, args.out, json.loads(args.params))
        return 0
    log(f"listening on 127.0.0.1:{args.port} ({DEVICE})")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
