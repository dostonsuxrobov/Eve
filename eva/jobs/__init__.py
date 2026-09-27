"""Jobs: a persona, a tool set and scenarios on the same loop (CLAUDE.md: new jobs are data,
not new agents). ``dispatch``: Red Oak Transport's truck dispatcher (eva/jobs/dispatch/)."""
from __future__ import annotations

from typing import Any

JOBS = ("dispatch",)
# Each job's own ElevenLabs voice (owner's pick, 2026-09-27), used on every ElevenLabs model.
JOB_VOICES = {"dispatch": "GZ4PpFJV8ikEGUtBrjK7"}


def job_tools(job: str) -> list[Any]:
    if job == "dispatch":
        from .dispatch.desk import dispatch_tools

        return dispatch_tools()
    raise ValueError(f"unknown job {job!r}; known: {', '.join(JOBS)}")


def job_persona(job: str) -> str:
    return {"dispatch": "dispatcher"}[job]
