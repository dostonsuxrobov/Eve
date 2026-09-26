"""Offer a small brain only the tools the user's words point at.

A 1-2B brain with every tool in reach calls them at random: MiniCPM5 1B checked the clock
after "how's it going?" and ended the call in the middle of "today was rough" (end-to-end run,
2026-09-25). A tool whose trigger is not in the user's line is simply not offered for that
turn; a tool without a trigger here is always offered. English only, like the session.
"""
from __future__ import annotations

import re

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


def gate_tools(text: str, tools: list[Tool]) -> list[Tool]:
    """The tools to offer for a turn that began with ``text``."""
    return [t for t in tools if t.name not in TRIGGERS or TRIGGERS[t.name].search(text or "")]


__all__ = ["TRIGGERS", "gate_tools"]
