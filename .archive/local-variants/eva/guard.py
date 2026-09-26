"""What a small brain must not say out loud: a check on every sentence before it is spoken.

A 1B brain says fluent things that aren't true or aren't hers. From the owner's session of
2026-09-26 (MiniCPM5 1B on Chatterbox): "Hello, I'm Doston" (she took the user's name),
"I'm still learning about you, Doston. It's been a while since we talked." repeated in every
reply and once a whole earlier reply again (18.7 s), "I need to call the get_weather function",
and a tool call written as markup, spoken: ``<function name="set_timer">...``. In earlier
sessions: "I'm also working on ... the Skynet project" (the user's project, from her memory).

:class:`SpeechGuard` drops such sentences before the voice gets them; the pipeline also keeps
them out of the history (a small brain copies its own history) and, if a whole reply is
dropped, asks once more with a nudge. Every drop prints a dim line saying why. Rules, not a
model: they cost microseconds, and each one is a failure seen in a real session.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

# An action announced in words ("let me check", "I'll look it up"): with no tool offered for
# this line nothing is behind it.
PROMISE_RE = re.compile(
    r"\b(one sec|hang on|let me (check|look|find|pull|see if|get)|i'?ll (check|look|find|pull|set|remind|note|get|"
    r"search|have to look)|i'?m (looking|checking|setting|pulling|searching)|setting (that|a|the|your) |look (it|that) up)"
)
_MARKUP_RE = re.compile(r"</?[a-z_][^<>]{0,80}>|```|\{\s*\"(name|arguments)\"", re.IGNORECASE)
_TOOL_TALK_RE = re.compile(r"\b(call|calling|use|using|run|running) (the |a |my )?[\w ]{0,20}(function|tool)s?\b")
# Eva has no job and no projects; a small brain that says "I'm working on" is reading the user's facts as hers.
_SELF_WORK_RE = re.compile(r"\b(i'?m|i am|i'?ve been|i have been|i was) (also )?(working on|building|developing|creating)\b|\bmy (project|job|work)\b")
# Planning out loud: "I need to ask Doston for the city ... so I will request the city information",
# "I'll ask the user" (replays, 2026-09-26). Eva never says "the user".
_THINKING_ALOUD_RE = re.compile(
    r"\bthe user\b|\bi need to (ask|call|check|request|find|get)\b|\bso i('?ll| will) (ask|request|check|call)\b|"
    r"\bi will request\b|\bi('?ll| will) ask (him|her|them|doston)\b"
)
# A shared past she can only claim if the thing is in her memory or said in this conversation.
_SHARED_PAST_RE = re.compile(
    r"\b(it'?s been a while since we|last time we|you told me|you mentioned|like you said|as you said|"
    r"i remember (when|you|that)|we talked about|since we (last )?(talked|spoke))\b"
)
_WORD_RE = re.compile(r"[a-z][a-z']{4,}")
_STOP = frozenset("about their there these those thing things think would could should really going doing "
                  "since while since talked spoke remember mentioned said told something".split())
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _norm(text: str) -> str:
    return text.lower().replace("’", "'")


@dataclass
class SpeechGuard:
    """Per-session filter for what a small brain says. ``known`` is what she may refer to as
    shared history (her memory; the user's lines are added as the conversation goes)."""

    user_name: str = ""
    tool_names: tuple[str, ...] = ()
    known: str = ""
    said: list[str] = field(default_factory=list)  # sentences already spoken this session
    repeat_ratio: float = 0.85

    def heard(self, user_text: str) -> None:
        """What the user said is shared history from now on."""
        self.known += "\n" + _norm(user_text)

    def _why(self, sentence: str, tools_offered: bool) -> str | None:
        low = _norm(sentence)
        if _MARKUP_RE.search(sentence):
            return "markup"
        if any(n in low for n in self.tool_names) or _TOOL_TALK_RE.search(low):
            return "tool talk"
        if _THINKING_ALOUD_RE.search(low):
            return "thinking out loud"
        name = _norm(self.user_name).strip()
        if name and re.search(rf"\b(i'?m|i am|my name is|this is|it'?s)\s+{re.escape(name)}\b", low):
            return "took the user's name"
        if _SELF_WORK_RE.search(low):
            return "claims the user's work as hers"
        if not tools_offered and PROMISE_RE.search(low):
            return "promises an action with no tool"
        if _SHARED_PAST_RE.search(low):
            words = {w for w in _WORD_RE.findall(low) if w not in _STOP}
            if not any(w in self.known for w in words):
                return "claims a shared past that isn't in her memory"
        if len(low.split()) >= 4 and any(SequenceMatcher(None, low, s).ratio() >= self.repeat_ratio for s in self.said):
            return "repeats herself"
        return None

    def filter(self, text: str, *, tools_offered: bool) -> tuple[str, list[tuple[str, str]]]:
        """The part of ``text`` she may say, and ``(sentence, reason)`` for every sentence dropped."""
        kept: list[str] = []
        dropped: list[tuple[str, str]] = []
        for sentence in (s.strip() for s in _SENT_SPLIT_RE.split(text)):
            if not sentence:
                continue
            why = self._why(sentence, tools_offered)
            if why:
                dropped.append((sentence, why))
            else:
                kept.append(sentence)
                self.said.append(_norm(sentence))
        return " ".join(kept), dropped


__all__ = ["PROMISE_RE", "SpeechGuard"]
