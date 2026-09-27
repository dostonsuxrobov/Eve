# Eva on a RunPod pod (the pilot, 2026-09-27)

The whole loop runs on one GPU box: ears (Parakeet), brain (Qwen3.8-27B in vLLM, FP8), voice
(Chatterbox or Orpheus in `voice/server.py`, Kokoro as the fallback). Only her audio crosses the
internet: the phone or laptop browser opens the page RunPod's proxy serves over https.

## The pod

* **GPU:** RTX PRO 6000, 96 GB (Secure Cloud, $2.09/h on 2026-09-27; stop it when done).
* **Template:** Runpod Pytorch 2.8.0 (`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`): CUDA 12.8,
  which the Blackwell card needs.
* **Overrides:** container disk 200 GB; HTTP ports 8888 and **8000**; TCP port 22; SSH terminal access on.
* The laptop's public key (`~/.ssh/runpod_ed25519.pub`) is in RunPod → Credentials → SSH Public Keys.

## Run

```
bash deploy/runpod/push.sh <ip> <port>                          # laptop -> /workspace/eva
ssh -p <port> -i ~/.ssh/runpod_ed25519 root@<ip>
mkdir -p /workspace/logs && bash /workspace/eva/deploy/runpod/setup.sh 2>&1 | tee /workspace/logs/setup.log  # once
bash /workspace/eva/deploy/runpod/start.sh [chatterbox|orpheus|kokoro]
```

Then open `https://<pod id>-8000.proxy.runpod.net` on the phone. `tmux attach -t eva` shows the
brain, the voice server and the loop (its console prints every silent decision, as on the laptop).
The container disk is erased when the pod stops: after a stop, `push.sh` and `setup.sh` again.
