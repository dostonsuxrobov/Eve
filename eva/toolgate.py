"""Offer a small brain only the tools the user's words point at, and fix what it gets wrong.

A 1-2B brain with every tool in reach calls them at random: MiniCPM5 1B checked the clock
after "how's it going?" and ended the call in the middle of "today was rough" (end-to-end run,
2026-09-25). A tool whose trigger is not in the user's line is simply not offered for that
turn; a tool without a trigger here is always offered. English only, like the session.

It also asked for the weather in New York for a user whose memory says "Lives in Philadelphia"
(owner's session, 2026-09-26): when the user didn't name a city, :class:`ToolGate` uses the
home city from memory.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .interfaces import Tool

TRIGGERS: dict[str, re.Pattern[str]] = {
    name: re.compile(pattern, re.IGNORECASE)
    for name, pattern in {
        "end_conversation": r"\b(bye|goodbye|good night|gotta go|got to go|have to go|need to go|going to go|"
                            r"talk (to you )?later|see you|ttyl|hang up|stop (talking|now)|that'?s all|i'?m off)\b",
        "get_current_time": r"\b(what time|the time|time is it|what day|which day|the date|what date|clock)\b",
        "set_timer": r"\b(timer|alarm|countdown|remind me in|in (a|an|one|two|three|four|five|six|seven|eight|nine|ten|"
                     r"fifteen|twenty|thirty|\d+) (minutes?|seconds?|hours?))\b",
        "remember_note": r"\b(remind|reminder|remember|note|write (that|this|it) down|don'?t let me forget|jot)\b",
        "recall_notes": r"\b(notes?|reminders?|what did i (ask|tell) you|what was i supposed to)\b",
        "get_weather": r"\b(weather|rain|raining|snow|snowing|sunny|forecast|temperature|umbrella|cold out|hot out)\b",
        "open_url": r"\b(open|website|youtube|browser|site|link|google)\b",
    }.items()
}
_CITY_RE = re.compile(r"\b(?:in|for|at|over in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)")  # "the weather in New York"
_HOME_RE = re.compile(r"\b(?i:lives in) ([A-Z][A-Za-z'. -]*?)(?:,|\.|$| and | with )")  # facts start "Lives in"


def gate_tools(text: str, tools: list[Tool]) -> list[Tool]:
    """The tools to offer for a turn that began with ``text``."""
    return [t for t in tools if t.name not in TRIGGERS or TRIGGERS[t.name].search(text or "")]


def home_city(facts: list[str]) -> str | None:
    """"Lives in Philadelphia, Pennsylvania." -> "Philadelphia"."""
    for fact in facts:
        m = _HOME_RE.search(fact)
        if m:
            return m.group(1).strip()
    return None


@dataclass
class ToolGate:
    """The gate for one session: remembers the line it was asked about, so the arguments of the
    call that follows can be checked against it."""

    home_city: str | None = None
    last_line: str = ""

    def filter(self, text: str, tools: list[Tool]) -> list[Tool]:
        self.last_line = text or ""
        return gate_tools(text, tools)

    def route(self, text: str) -> list[tuple[str, dict[str, Any]]]:
        """Calls the loop makes itself for a plain question, before the brain speaks: with the
        weather or the time asked for, MiniCPM5 1B called the tool in 2 of 8 replayed sessions
        once the prompt stopped insisting, and made the weather up in others ("it's a sunny
        Saturday morning" against drizzle and 81 % rain, 2026-09-26)."""
        calls: list[tuple[str, dict[str, Any]]] = []
        if TRIGGERS["get_weather"].search(text or ""):
            m = _CITY_RE.search(text or "")
            city = m.group(1).strip() if m else self.home_city
            if city:
                calls.append(("get_weather", {"city": city}))
        if TRIGGERS["get_current_time"].search(text or ""):
            calls.append(("get_current_time", {}))
        return calls

    def fix_args(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "get_weather" and self.home_city:
            city = str(args.get("city") or "")
            if not city or city.lower() not in self.last_line.lower():
                return {**args, "city": self.home_city}  # the user didn't name it: they mean home
        return args


__all__ = ["TRIGGERS", "ToolGate", "gate_tools", "home_city"]
