"""ElevenLabs credits: what each session spends, and the billing month's running total.

The owner's plan (eva/config.py): 121k credits a month, the month turning on the 28th. The API
key can't read the account balance (no ``user_read``), so the meter adds up the
``character-cost`` header ElevenLabs puts on every text-to-speech response (the credits that
request cost; measured 2026-09-26: 35 for a 64-character line on v3, 17 on v3 Conversational
and Flash v2.5) and keeps the month's total in ``usage.json``. When a response carries no
header, the characters sent count, which over-estimates v3.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .config import CREDITS_WARN_AT, PLAN_MONTHLY_CREDITS, PLAN_RENEWS_ON_DAY, USAGE_FILE


def billing_month(today: date | None = None, renews_on: int = PLAN_RENEWS_ON_DAY) -> str:
    """The billing month ``today`` falls in, named by the date it started: "2026-09-28"."""
    d = today or date.today()
    if d.day >= renews_on:
        return date(d.year, d.month, renews_on).isoformat()
    year, month = (d.year - 1, 12) if d.month == 1 else (d.year, d.month - 1)
    return date(year, month, renews_on).isoformat()


@dataclass
class CreditMeter:
    """Adds up credits per session and per billing month; ``save()`` writes the month to disk."""

    path: Path = USAGE_FILE
    monthly: int = PLAN_MONTHLY_CREDITS
    month: str = field(default_factory=billing_month)
    month_credits: int = 0  # this billing month, earlier sessions included
    session_credits: int = 0
    session_chars: int = 0
    requests: int = 0
    by_model: dict[str, int] = field(default_factory=dict)
    _turn_mark: int = 0

    def __post_init__(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        self.month_credits = int(data.get("months", {}).get(self.month, {}).get("credits", 0))

    def add(self, model_id: str, chars: int, credits: int | None) -> None:
        """One text-to-speech request: its characters and the credits it reported (or ``None``)."""
        spent = chars if credits is None else int(credits)
        self.requests += 1
        self.session_chars += chars
        self.session_credits += spent
        self.month_credits += spent
        self.by_model[model_id] = self.by_model.get(model_id, 0) + spent

    def turn_spent(self) -> int:
        """Credits since the last call (one reply's worth, when called once per turn)."""
        spent, self._turn_mark = self.session_credits - self._turn_mark, self.session_credits
        return spent

    @property
    def month_share(self) -> float:
        return self.month_credits / self.monthly if self.monthly else 0.0

    @property
    def warn(self) -> bool:
        return self.month_share >= CREDITS_WARN_AT

    def line(self) -> str:
        """For the console: this session and the month against the allowance."""
        return (f"credits: {self.session_credits:,} this session, {self.month_credits:,} of "
                f"{self.monthly:,} this month ({self.month_share:.0%}; month started {self.month})")

    def save(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        months = data.setdefault("months", {})
        m = months.setdefault(self.month, {"credits": 0, "sessions": 0})
        m["credits"] = self.month_credits
        m["sessions"] = int(m.get("sessions", 0)) + (1 if self.requests else 0)
        m["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        tmp.replace(self.path)


__all__ = ["CreditMeter", "billing_month"]
