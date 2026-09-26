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


# Function words that do not distinguish two facts ("a meal" / "meals", "the user's ...").
_STOPWORDS = frozenset(
    {"a", "an", "the", "and", "or", "of", "to", "is", "are", "in", "on", "for", "with", "their",
     "they", "them", "user", "user's", "users", "his", "her", "he", "she", "has", "have", "that"}
)


def _normalise(fact: str) -> str:
    """Lowercase content words, plurals folded ("meals" -> "meal"), function words dropped."""
    words = []
    for w in _WORD_RE.findall(fact.lower()):
        if w in _STOPWORDS:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        words.append(w)
    return " ".join(words)


def _similar(a: str, b: str, threshold: float = 0.8) -> bool:
    """Token-Jaccard near-duplicate test on already normalised strings."""
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return a == b
    return len(ta & tb) / len(ta | tb) >= threshold


_VERBS = frozenset(
    "is was has had lives lived works worked speaks likes loves wants prefers considers mentioned does plays "
    "enjoys studies knows uses drinks eats hates feels thinks calls goes needs tries plans recently often "
    "usually has been will would can could".split()
)


def _lower_verb(fact: str) -> str:
    """"Is working on X" -> "is working on X"; "Priya is ..." stays as it is."""
    first = fact.split(" ", 1)[0].lower()
    return fact[0].lower() + fact[1:] if fact and first in _VERBS else fact


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
    def as_prompt_text(self, subject: str | None = None) -> str:
        """Facts as short lines for the ``{memory}`` slot ("" when empty).

        With ``subject`` (the user's name) each line says whose fact it is: facts are stored
        without one ("Is working on a project called Skynet"), and a 1B brain read them as its
        own ("I have been working on the Skynet project as well", 2026-09-26).
        """
        if not subject:
            return "\n".join(f"- {f}" for f in self.facts)
        return "\n".join(f"- {subject}: {_lower_verb(f)}" for f in self.facts)

    def drop_name_facts(self, user_name: str) -> list[str]:
        """Forget facts that state the user's name or file ``user_name`` as another person.

        When the caller knows the name (``--user-name``) such facts can only be wrong:
        the transcript is speech recognition, so "my name is Doston" arrives as
        "Doster", and the assistant addressing the user by name once produced "Has a
        friend named Doston."  Returns the facts that were removed.
        """
        user_name = " ".join((user_name or "").split())
        if not user_name:
            return []
        pat = re.compile(r"\bname is\b|\b(?:named|called|name)\s+" + re.escape(user_name) + r"\b", re.IGNORECASE)
        dropped = [f for f in self.facts if pat.search(f)]
        if dropped:
            self.facts = [f for f in self.facts if f not in dropped]
        return dropped

    # ---------------------------------------------------------- summarising
    async def update_from_transcript(
        self,
        llm: Any,
        messages: list[dict[str, Any]],
        *,
        user_name: str = "",
        assistant_name: str = "Eva",
        save: bool = True,
    ) -> list[str]:
        """Ask ``llm`` for new durable facts in ``messages`` and merge them.

        ``messages`` is the OpenAI-format history of the session (system / tool
        messages are ignored).  ``user_name`` is who the assistant was talking to when
        the caller knows it (``--user-name``): the extractor is told, so it neither
        records a misheard name nor files that name as a friend.  Returns the list of
        facts that were actually new.  Never raises on model nonsense: unparsable
        output just yields no facts.
        """
        transcript = _transcript_text(messages)
        if not transcript.strip():
            return []
        known = "\n".join(f"- {f}" for f in self.facts) or "(none yet)"
        user_name = " ".join((user_name or "").split())
        if user_name:
            who = (
                f"The person is {user_name} and the assistant is {assistant_name}; both names are "
                f"already known. Never record the person's name or a spelling of it, and never treat "
                f'"{user_name}" or "{assistant_name}" in the transcript as somebody else '
                "(the assistant addresses the person by name). "
            )
        else:
            who = f"The assistant is {assistant_name}; never record facts about the assistant. "
        prompt = [
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{who}The transcript comes from speech recognition, so names and rare words "
                    f"may be misspelt.\n\nKnown facts:\n{known}\n\nConversation transcript:\n"
                    f"{transcript}\n\nJSON array of NEW durable facts about the person:"
                ),
            },
        ]
        raw = await _complete(llm, prompt)
        facts = parse_fact_array(raw)
        if user_name:
            probe = Memory(path=self.path, facts=list(facts))
            for bad in probe.drop_name_facts(user_name):
                facts.remove(bad)
        new = self.merge(facts)
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
