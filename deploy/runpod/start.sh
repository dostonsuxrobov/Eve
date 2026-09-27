#!/usr/bin/env bash
# Start Eva on the pod: the brain (vLLM), the voice server, Ollama for Orpheus, then the loop
# serving the phone page on port 8000 (RunPod's proxy puts it on https, so the mic works):
#   https://<pod id>-8000.proxy.runpod.net
# Each runs in its own tmux window of the session "eva" (tmux attach -t eva); logs in /workspace/logs.
#   bash deploy/runpod/start.sh [voice]        voice: chatterbox (default), orpheus, kokoro
#   bash deploy/runpod/start.sh stop
set -euo pipefail

W=/workspace
EVA="$(cd "$(dirname "$0")/../.." && pwd)"
LOG=$W/logs
mkdir -p "$LOG"
export HF_HOME=$W/hf

if [ "${1:-}" = "stop" ]; then
  tmux kill-session -t eva 2>/dev/null || true
  pkill -f "vllm serve" 2>/dev/null || true
  pkill -f "voice/server.py" 2>/dev/null || true
  echo "stopped"
  exit 0
fi
VOICE="${1:-chatterbox}"

up() {  # wait for an HTTP health URL, at most $2 seconds
  local url=$1 limit=$2 t=0
  until curl -sf "$url" >/dev/null; do
    sleep 2; t=$((t + 2))
    if [ "$t" -ge "$limit" ]; then echo "not up after ${limit}s: $url"; return 1; fi
  done
}

tmux has-session -t eva 2>/dev/null || tmux new-session -d -s eva -n shell

if ! curl -sf http://127.0.0.1:8100/health >/dev/null; then
  # Qwen's own FP8 build halves the 27B (~28 GB) and its read per token; 0.55 of the card leaves ~43 GB for the
  # voices and a speech-to-speech model later. The vision encoder is never fed (no images).
  tmux new-window -t eva -n brain "$W/venv-vllm/bin/vllm serve Qwen/Qwen3.8-27B-FP8 --served-model-name qwen27b \
    --host 127.0.0.1 --port 8100 --max-model-len 32768 --gpu-memory-utilization 0.55 \
    --enable-auto-tool-choice --tool-call-parser hermes --limit-mm-per-prompt '{\"image\":0,\"video\":0}' \
    2>&1 | tee $LOG/vllm.log"
fi

if ! pgrep -x ollama >/dev/null && command -v ollama >/dev/null; then
  tmux new-window -t eva -n ollama "OLLAMA_HOST=127.0.0.1:11434 ollama serve 2>&1 | tee $LOG/ollama.log"
fi

if ! curl -sf http://127.0.0.1:8765/health >/dev/null; then
  # a big card: SNAC decodes on the GPU (on the laptop it had to share with Ollama and ran on the CPU)
  tmux new-window -t eva -n voice "cd $EVA && EVA_SNAC_DEVICE=cuda $W/venv-voice/bin/python voice/server.py --port 8765 \
    2>&1 | tee -a $LOG/voice.log"
fi

echo "waiting for the brain (first start compiles and loads ~28 GB: a few minutes) and the voice server..."
up http://127.0.0.1:8765/health 120
up http://127.0.0.1:8100/health 900

tmux kill-window -t eva:loop 2>/dev/null || true
tmux new-window -t eva -n loop "cd $EVA && EVA_VOICE_PYTHON=$W/venv-voice/bin/python PYTHONIOENCODING=utf-8 \
  $W/venv-eva/bin/python run.py --web --port 8000 --brain qwen27b --voice $VOICE --user-name Doston \
  2>&1 | tee -a $LOG/eva.log"
echo "Eva ($VOICE) is starting: https://${RUNPOD_POD_ID:-<pod id>}-8000.proxy.runpod.net  (tmux attach -t eva)"
