#!/usr/bin/env bash
# Eva on a RunPod pod: install everything once (mostly downloads: ~15-25 min).
#
# The pod (2026-09-27): RTX PRO 6000 96 GB, template runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
# (CUDA 12.8: the Blackwell card needs it), container disk 200 GB, HTTP port 8000 (the phone page),
# TCP port 22 (SSH). Three environments, so pinned dependencies never meet:
#   venv-vllm   the brain: vLLM serving Qwen3.8-27B in FP8
#   venv-voice  the open voices: Chatterbox (its torch 2.6 pin overridden to 2.8: 2.6 has no
#               Blackwell kernels) + SNAC for Orpheus; Orpheus' tokens come from Ollama
#   venv-eva    the loop, Parakeet, Kokoro, the web server
# Run from the repo copy: bash deploy/runpod/setup.sh   (log: /workspace/logs/setup.log)
set -euo pipefail

W=/workspace
EVA="$(cd "$(dirname "$0")/../.." && pwd)"
LOG=$W/logs
mkdir -p "$LOG" "$W/hf"
export HF_HOME=$W/hf
export UV_CACHE_DIR=$W/uv-cache
export PATH="$HOME/.local/bin:$PATH"

step() { echo "[setup $(date +%H:%M:%S)] $*"; }

step "system packages"
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libportaudio2 libsndfile1 tmux curl zstd >/dev/null

if ! command -v uv >/dev/null; then
  step "uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi

step "brain weights (background): Qwen/Qwen3.8-27B-FP8"
if [ ! -f "$W/.weights-done" ]; then
  (uvx --from huggingface_hub hf download Qwen/Qwen3.8-27B-FP8 >"$LOG/weights.log" 2>&1 && touch "$W/.weights-done") &
  WEIGHTS_PID=$!
fi

step "venv-vllm"
[ -x "$W/venv-vllm/bin/python" ] || uv venv -q "$W/venv-vllm" --python 3.12
uv pip install -q --python "$W/venv-vllm/bin/python" vllm

step "venv-voice (torch 2.8, chatterbox, snac)"
[ -x "$W/venv-voice/bin/python" ] || uv venv -q "$W/venv-voice" --python 3.12
printf 'torch==2.8.0\ntorchaudio==2.8.0\n' >"$W/voice-overrides.txt"
uv pip install -q --python "$W/venv-voice/bin/python" --override "$W/voice-overrides.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
  chatterbox-tts snac soundfile

step "venv-eva"
[ -x "$W/venv-eva/bin/python" ] || uv venv -q "$W/venv-eva" --python 3.13
uv pip install -q --python "$W/venv-eva/bin/python" \
  numpy sounddevice soundfile httpx rich onnxruntime kokoro-onnx sherpa-onnx websockets pytest

step "Ollama + Orpheus (the second open voice)"
if ! command -v ollama >/dev/null; then
  curl -fsSL https://ollama.com/install.sh | sh >"$LOG/ollama-install.log" 2>&1
fi
if ! pgrep -x ollama >/dev/null; then
  (OLLAMA_HOST=127.0.0.1:11434 nohup ollama serve >"$LOG/ollama.log" 2>&1 &)
  sleep 3
fi
ollama pull "${EVA_ORPHEUS_MODEL:-legraphista/Orpheus:3b-ft-q4_k_m}" >"$LOG/orpheus-pull.log" 2>&1 || step "Orpheus pull failed (see orpheus-pull.log); Chatterbox still works"

step "Eva's own models (Parakeet, Kokoro, VAD) and the voices' weights"
cd "$EVA"
"$W/venv-eva/bin/python" - <<'PY'
from eva.factory import build_stt, build_tts
import asyncio
async def main():
    stt = build_stt({"kind": "parakeet"}); await stt.warmup(); await stt.close()
    tts = build_tts({"kind": "kokoro", "voice": "af_heart"}); await tts.warmup(); await tts.close()
asyncio.run(main())
print("parakeet + kokoro ready")
PY
"$W/venv-voice/bin/python" voice/server.py --say "Hey. It's me, testing the new voice." --engine chatterbox --voice eva --out "$LOG/chatterbox-test.wav"

if [ -n "${WEIGHTS_PID:-}" ]; then
  step "waiting for the brain weights"
  wait "$WEIGHTS_PID" || { step "weights download failed: see $LOG/weights.log"; exit 1; }
fi
step "done: now bash deploy/runpod/start.sh"
