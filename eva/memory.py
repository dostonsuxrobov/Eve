"""Tiny persistent facts store about the user + end-of-session summariser.

``Memory`` keeps a flat list of short durable facts ("Has a cat named Miso",
"Works as a product designer at a startup") in a JSON file.  The pipeline renders
them into the persona prompt through :meth:`Memory.as_prompt_text` and, at the end
of a session, calls :meth:`Memory.update_from_transcript` which asks the LLM for a
JSON array of *new* facts and merges them, deduplicated, capped at ``max_facts``.

The LLM is anything that satisfies :class:`eva.interfaces.LLM`.  The contract only
guarantees ``stream()``; if the concrete class also offers ``complete()`` (the
OpenAICompatLLM being built alongside this module is expected to) it is used
directly, otherwise the stream is drained.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config
from .interfaces import LLMDelta, LLMDone

MAX_FACTS_DEFAULT = 60
MAX_FACT_CHARS = 160
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z0-9']+")

EXTRACT_SYSTEM_PROMPT = (
    "You maintain a short list of durable facts about one person, gathered from their "
    "conversations with a voice assistant. Durable means it will still be true and "
    "useful in a week: their name and what they like to be called, people in their "
    "life (names and relationships), pets, work and projects, preferences and tastes, "
    "ongoing situations (a job decision, a sick pet, an upcoming trip), and how they "
    "like the assistant to behave. Do NOT record moods of the moment, small talk, "
    "things the assistant said, or anything already in the known list. Each fact is one "
    "plain third-person sentence under twenty words, e.g. \"Has a sister named Priya.\" "
    "Reply with ONLY a JSON array of strings. If there is nothing new, reply with []."
)


def _normalise(fact: str) -> str:
    return " ".join(_WORD_RE.findall(fact.lower()))


def _similar(a: str, b: str, threshold: float = 0.8) -> bool:
    """Token-Jaccard near-duplicate test on already normalised strings."""
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return a == b
    return len(ta & tb) / len(ta | tb) >= threshold


@dataclass
class Memory:
    """Persistent list of short facts about the user."""

    path: Path = field(default_factory=lambda: config.MEMORY_FILE)
    max_facts: int = MAX_FACTS_DEFAULT
    facts: list[str] = field(default_factory=list)
    updated_at: float | None = None

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    # ----------------------------------------------------------------- disk io
    def load(self) -> "Memory":
        """Read facts from ``path`` (missing or corrupt file -> empty memory)."""
        self.facts = []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return self
        raw = data.get("facts", []) if isinstance(data, dict) else data
        if isinstance(raw, list):
            for f in raw:
                if isinstance(f, str):
                    self.add(f)
        if isinstance(data, dict):
            self.updated_at = data.get("updated_at")
        return self

    def save(self) -> None:
        """Write facts to ``path`` atomically (write temp file, then replace)."""
        self.updated_at = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"facts": self.facts, "updated_at": self.updated_at}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    # --------------------------------------------------------------- mutation
    def add(self, fact: str) -> bool:
        """Add one fact if it is non-empty and not a (near) duplicate. Returns True if added."""
        fact = " ".join(fact.split()).strip().strip('"').strip()
        if not fact:
            return False
        if len(fact) > MAX_FACT_CHARS:
            fact = fact[: MAX_FACT_CHARS - 1].rstrip() + "…"
        norm = _normalise(fact)
        if not norm:
            return False
        for existing in self.facts:
            if _similar(norm, _normalise(existing)):
                return False
        self.facts.append(fact)
        if len(self.facts) > self.max_facts:
            # oldest facts fall off first
            del self.facts[: len(self.facts) - self.max_facts]
        return True

    def merge(self, facts: list[str]) -> list[str]:
        """Add many facts; returns the ones that were actually new."""
        return [f for f in facts if self.add(f)]

    def clear(self) -> None:
        self.facts = []

    # ---------------------------------------------------------------- render
    def as_prompt_text(self) -> str:
        """Facts as short lines for the ``{memory}`` slot ("" when empty)."""
        return "\n".join(f"- {f}" for f in self.facts)

    # ---------------------------------------------------------- summarising
    async def update_from_transcript(
        self,
        llm: Any,
        messages: list[dict[str, Any]],
        *,
        save: bool = True,
    ) -> list[str]:
        """Ask ``llm`` for new durable facts in ``messages`` and merge them.

        ``messages`` is the OpenAI-format history of the session (system / tool
        messages are ignored).  Returns the list of facts that were actually new.
        Never raises on model nonsense: unparsable output just yields no facts.
        """
        transcript = _transcript_text(messages)
        if not transcript.strip():
            return []
        known = "\n".join(f"- {f}" for f in self.facts) or "(none yet)"
        prompt = [
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Known facts:\n{known}\n\nConversation transcript:\n{transcript}\n\n"
                    "JSON array of NEW durable facts about the user:"
                ),
            },
        ]
        raw = await _complete(llm, prompt)
        new = self.merge(parse_fact_array(raw))
        if new and save:
            self.save()
        return new


# ------------------------------------------------------------------- helpers
def _transcript_text(messages: list[dict[str, Any]], max_chars: int = 12_000) -> str:
    lines: list[str] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            continue
        lines.append(f"{'User' if role == 'user' else 'Assistant'}: {content.strip()}")
    text = "\n".join(lines)
    return text[-max_chars:] if len(text) > max_chars else text


def parse_fact_array(raw: str) -> list[str]:
    """Pull a JSON array of strings out of model output (tolerates fences, prose, <think>)."""
    text = _THINK_RE.sub("", raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    candidates = []
    m = _JSON_ARRAY_RE.search(text)
    if m:
        candidates.append(m.group(0))
    candidates.append(text)
    for cand in candidates:
        try:
            data = json.loads(cand)
        except ValueError:
            continue
        if isinstance(data, list):
            out: list[str] = []
            for item in data:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict):
                    v = item.get("fact") or item.get("text")
                    if isinstance(v, str):
                        out.append(v)
            return out
    return []


async def _complete(llm: Any, messages: list[dict[str, Any]]) -> str:
    """Return the full text of one non-streaming completion from any LLM-like object."""
    complete = getattr(llm, "complete", None)
    if callable(complete):
        result = complete(messages)
        if hasattr(result, "__await__"):
            result = await result
        if isinstance(result, str):
            return result
        text = getattr(result, "text", None)
        if isinstance(text, str):
            return text
        return str(result)
    parts: list[str] = []
    async for ev in llm.stream(messages, None):
        if isinstance(ev, LLMDelta):
            parts.append(ev.text)
        elif isinstance(ev, LLMDone):
            break
    return "".join(parts)


__all__ = ["Memory", "parse_fact_array", "EXTRACT_SYSTEM_PROMPT", "MAX_FACTS_DEFAULT"]
