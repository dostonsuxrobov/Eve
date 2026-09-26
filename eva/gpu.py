"""Keeping the 6 GB GPU for the variant that is running.

The brain and Orpheus live in Ollama, Chatterbox in the voice server's PyTorch. Ollama keeps a
model loaded for its keep-alive after the last request, so a brain from an earlier session can
still hold VRAM. On this Windows laptop an overflow does not fail: allocations spill into
system RAM and everything on the GPU slows down about twenty times (Chatterbox Turbo went from
2.2x to 0.05x real time with three stale Ollama models loaded, 2026-09-25). So a session
unloads what it doesn't use before it warms up, and says how full the card is afterwards.
"""
from __future__ import annotations

import subprocess

import httpx

from .config import OLLAMA_BASE_URL

OLLAMA = OLLAMA_BASE_URL.removesuffix("/v1")
GPU_TOTAL_MIB = 6141  # RTX 4050 laptop
GPU_WARN_MIB = 5700  # above this the next allocation may spill into system RAM


def full_tag(model: str) -> str:
    """``ollama ps`` lists an untagged model as ``name:latest``."""
    return model if ":" in model.rsplit("/", 1)[-1] else model + ":latest"


def ollama_loaded() -> list[dict]:
    try:
        return httpx.get(f"{OLLAMA}/api/ps", timeout=5).json().get("models") or []
    except (httpx.HTTPError, ValueError):
        return []


def free_ollama(keep: set[str]) -> list[str]:
    """Unload every Ollama model not in ``keep``; returns the names unloaded."""
    wanted = {full_tag(m) for m in keep}
    gone = []
    for m in ollama_loaded():
        if m["name"] not in wanted:
            try:
                httpx.post(f"{OLLAMA}/api/generate", json={"model": m["name"], "keep_alive": 0}, timeout=15)
                gone.append(m["name"])
            except httpx.HTTPError:
                pass
    return gone


def gpu_used_mib() -> int | None:
    """What ``nvidia-smi`` says is in use on the card (all processes)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


__all__ = ["GPU_TOTAL_MIB", "GPU_WARN_MIB", "free_ollama", "gpu_used_mib", "ollama_loaded", "full_tag"]
