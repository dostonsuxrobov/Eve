"""Small tool registry for Eva: time, timers, notes, weather, open URL.

Every tool returns a short *spoken-style* string so the LLM can relay it verbatim.
Errors are caught in :func:`execute` and turned into a short error string; the model
must never claim a task is done unless the tool result says so (see DESIGN.md).

Timers
------
:func:`set_timer` returns immediately.  When the timer fires, an event dict
``{"type": "timer", "label": ..., "seconds": ..., "set_at": ...}`` is put on the
module-level :data:`pending_events` :class:`asyncio.Queue`, which the pipeline drains
between turns (e.g. to say "your tea timer just went off").

The queue is created lazily on the running loop; call :func:`set_event_loop` if the
pipeline runs the tools from a different loop than the one that will drain the queue
(rare; a single ``asyncio.run`` never needs it).
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
import os
import re
import time
from typing import Any

import httpx

from .config import NOTES_FILE, USER_AGENT
from .interfaces import LLMToolCall, Tool

log = logging.getLogger("eva.tools")

TOOL_TIMEOUT_S = 15.0

# --------------------------------------------------------------------- events
pending_events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
_loop: asyncio.AbstractEventLoop | None = None
_timers: list[dict[str, Any]] = []


def set_event_loop(loop: asyncio.AbstractEventLoop | None = None) -> asyncio.Queue[dict[str, Any]]:
    """Bind timer scheduling (and a fresh :data:`pending_events`) to ``loop``.

    Only needed if tools are executed from one event loop and the queue is drained from
    another, or when several ``asyncio.run`` calls happen in one process (tests).
    Returns the queue now in use (also available as ``eva.tools.pending_events``).
    """
    global _loop, pending_events
    _loop = loop or asyncio.get_running_loop()
    pending_events = asyncio.Queue()
    _timers.clear()
    return pending_events


def _get_loop() -> asyncio.AbstractEventLoop:
    if _loop is not None and not _loop.is_closed():
        return _loop
    return asyncio.get_running_loop()


def active_timers() -> list[dict[str, Any]]:
    """Timers that have not fired yet (label, seconds, fires_at)."""
    now = time.monotonic()
    return [dict(t, remaining_s=round(t["fires_at"] - now, 1)) for t in _timers if t["fires_at"] > now]


# ---------------------------------------------------------------- formatting
def _spoken_duration(seconds: int) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    m, s = divmod(seconds, 60)
    if m < 60:
        base = f"{m} minute{'s' if m != 1 else ''}"
        return base if not s else f"{base} and {s} second{'s' if s != 1 else ''}"
    h, m = divmod(m, 60)
    base = f"{h} hour{'s' if h != 1 else ''}"
    return base if not m else f"{base} and {m} minute{'s' if m != 1 else ''}"


def _spoken_clock(dt: _dt.datetime) -> str:
    hour = dt.strftime("%I").lstrip("0") or "12"
    return f"{hour}:{dt.strftime('%M %p')}"


# -------------------------------------------------------------------- tools
def get_current_time() -> str:
    """Current local date and time, phrased for speech."""
    now = _dt.datetime.now().astimezone()
    return f"It's {_spoken_clock(now)} on {now.strftime('%A, %B')} {now.day}, {now.year} ({now.tzname()})."


async def set_timer(seconds: int, label: str = "timer") -> str:
    """Schedule a timer; returns at once. Fires an event on :data:`pending_events`."""
    seconds = int(seconds)
    if seconds <= 0:
        return "Error: the timer needs a positive number of seconds."
    if seconds > 24 * 3600:
        return "Error: timers longer than a day aren't supported."
    label = (label or "timer").strip() or "timer"
    loop = _get_loop()
    set_at = time.time()
    entry = {"label": label, "seconds": seconds, "fires_at": time.monotonic() + seconds, "set_at": set_at}
    _timers.append(entry)

    def _fire() -> None:
        try:
            pending_events.put_nowait(
                {"type": "timer", "label": label, "seconds": seconds, "set_at": set_at, "fired_at": time.time()}
            )
        except Exception as e:  # pragma: no cover
            log.warning("timer %r could not post its event: %s", label, e)
        try:
            _timers.remove(entry)
        except ValueError:
            pass

    entry["handle"] = loop.call_later(seconds, _fire)
    fires = _dt.datetime.now().astimezone() + _dt.timedelta(seconds=seconds)
    return f"Timer '{label}' set for {_spoken_duration(seconds)}. It will go off at {_spoken_clock(fires)}."


def _load_notes() -> list[dict[str, Any]]:
    if not NOTES_FILE.exists():
        return []
    try:
        data = json.loads(NOTES_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _save_notes(notes: list[dict[str, Any]]) -> None:
    tmp = NOTES_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(notes, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, NOTES_FILE)


def remember_note(text: str) -> str:
    """Append a note to notes.json."""
    text = (text or "").strip()
    if not text:
        return "Error: nothing to remember."
    notes = _load_notes()
    notes.append({"text": text, "ts": _dt.datetime.now().astimezone().isoformat(timespec="seconds")})
    _save_notes(notes)
    return f"Noted: {text}"


def recall_notes(limit: int = 10) -> str:
    """Read back the most recent notes."""
    notes = _load_notes()
    if not notes:
        return "There are no saved notes yet."
    recent = notes[-int(limit) :]
    lines = []
    for n in recent:
        try:
            when = _dt.datetime.fromisoformat(n["ts"]).strftime("%b %d")
        except (KeyError, ValueError):
            when = "?"
        lines.append(f"({when}) {n.get('text', '')}")
    return f"{len(notes)} note{'s' if len(notes) != 1 else ''} saved. Most recent: " + "; ".join(lines)


_WMO: dict[int, str] = {
    0: "clear skies", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "foggy", 48: "icy fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "rain showers",
    81: "rain showers", 82: "violent rain showers", 85: "snow showers", 86: "heavy snow showers",
    95: "a thunderstorm", 96: "a thunderstorm with hail", 99: "a thunderstorm with heavy hail",
}


async def get_weather(city: str, units: str = "auto") -> str:
    """Current conditions + today's range via open-meteo (no API key)."""
    city = (city or "").strip()
    if not city:
        return "Error: which city?"
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0), headers={"User-Agent": USER_AGENT}) as client:
        g = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
        )
        g.raise_for_status()
        results = g.json().get("results") or []
        if not results:
            return f"Error: I couldn't find a place called {city}."
        place = results[0]
        name = place.get("name", city)
        country = place.get("country_code", "")
        use_f = units == "fahrenheit" or (units == "auto" and country in ("US", "BS", "BZ", "KY", "PW", "LR"))
        params: dict[str, Any] = {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m,relative_humidity_2m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": 1,
        }
        if use_f:
            params["temperature_unit"] = "fahrenheit"
            params["wind_speed_unit"] = "mph"
        f = await client.get("https://api.open-meteo.com/v1/forecast", params=params)
        f.raise_for_status()
        data = f.json()
    cur = data.get("current", {})
    daily = data.get("daily", {})
    unit = "Fahrenheit" if use_f else "Celsius"
    temp = round(cur.get("temperature_2m", 0))
    feels = round(cur.get("apparent_temperature", temp))
    desc = _WMO.get(int(cur.get("weather_code", -1)), "mixed conditions")
    first = f"In {name}, it's {temp} degrees {unit} and {desc}"
    if abs(feels - temp) >= 3:
        first += f", feels like {feels}"
    rest: list[str] = []
    try:
        hi = round(daily["temperature_2m_max"][0])
        lo = round(daily["temperature_2m_min"][0])
        rest.append(f"Today's high is {hi} with a low of {lo}")
        pp = daily.get("precipitation_probability_max", [None])[0]
        if pp is not None and pp >= 20:
            rest.append(f"and there's a {int(pp)} percent chance of rain")
    except (KeyError, IndexError, TypeError):
        pass
    wind = cur.get("wind_speed_10m")
    if wind is not None and wind >= (15 if use_f else 25):
        rest.append(f"It's windy too, around {round(wind)} {'miles' if use_f else 'kilometers'} an hour")
    summary = first + "."
    if rest:
        summary += " " + ", ".join(rest) + "."
    return summary.replace(", It's windy", ". It's windy")


_URL_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def open_url(url: str) -> str:
    """Open a web page in the default browser (Windows ``os.startfile``)."""
    url = (url or "").strip()
    if not url:
        return "Error: no URL given."
    if not _URL_SCHEME_RE.match(url):
        url = "https://" + url
    scheme = url.split(":", 1)[0].lower()
    if scheme not in ("http", "https", "mailto"):
        return f"Error: I can only open web links, not {scheme} links."
    if hasattr(os, "startfile"):
        os.startfile(url)  # type: ignore[attr-defined]
    else:  # pragma: no cover - non-Windows fallback
        import webbrowser

        webbrowser.open(url)
    return f"Opened {url} in the browser."


# ----------------------------------------------------------------- registry
def end_conversation(reason: str = "") -> str:
    """Ask the pipeline to end the session after the current reply (the goodbye)."""
    pending_events.put_nowait({"type": "end_session", "reason": reason})
    # Only reached by the model when it called the tool without saying goodbye: the
    # pipeline skips the round after a final tool whose call already carried the goodbye.
    return "OK: the session ends right after this reply. Say one short, warm goodbye now, nothing else."


_TOOLS: list[Tool] = [
    Tool(
        name="end_conversation",
        description=(
            "End the conversation. Call this when the user says goodbye, says they have to go, "
            "or asks you to stop; say your goodbye in the same reply."
        ),
        parameters={
            "type": "object",
            "properties": {"reason": {"type": "string", "description": "Why the conversation ends, a few words."}},
            "required": [],
        },
        fn=end_conversation,
        spoken_hint=None,
        final=True,
    ),
    Tool(
        name="get_current_time",
        description="Get the current local date and time. Use when the user asks what time or day it is.",
        parameters={"type": "object", "properties": {}, "required": []},
        fn=get_current_time,
        spoken_hint=None,
    ),
    Tool(
        name="set_timer",
        description=(
            "Set a countdown timer that will notify the user when it ends. "
            "Convert the requested duration to seconds (two minutes = 120)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "seconds": {"type": "integer", "description": "Duration in seconds (1 to 86400)."},
                "label": {"type": "string", "description": "Short name for the timer, e.g. 'tea'."},
            },
            "required": ["seconds", "label"],
        },
        fn=set_timer,
        spoken_hint=None,
    ),
    Tool(
        name="remember_note",
        description="Save a short note or reminder text the user wants remembered for later.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string", "description": "The note to save, in the user's words."}},
            "required": ["text"],
        },
        fn=remember_note,
        spoken_hint=None,
    ),
    Tool(
        name="recall_notes",
        description="Read back the user's saved notes and reminders.",
        parameters={"type": "object", "properties": {}, "required": []},
        fn=recall_notes,
        spoken_hint="let me check my notes",
    ),
    Tool(
        name="get_weather",
        description="Get the current weather and today's forecast for a city.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name, e.g. 'Boston' or 'Tashkent'."}},
            "required": ["city"],
        },
        fn=get_weather,
        spoken_hint="let me check the weather",
    ),
    Tool(
        name="open_url",
        description="Open a website in the user's browser.",
        parameters={
            "type": "object",
            "properties": {"url": {"type": "string", "description": "Full URL, e.g. https://youtube.com"}},
            "required": ["url"],
        },
        fn=open_url,
        spoken_hint="opening that now",
    ),
]


def get_tools() -> list[Tool]:
    """All registered tools (a copy of the list)."""
    return list(_TOOLS)


def tool_notes(tools: list[Tool] | None = None) -> str:
    """The ``{tool_notes}`` text for the persona prompt.

    Small models like to *say* "timer set" instead of calling the function, so the
    note spells out that only a real call does anything.
    """
    tools = get_tools() if tools is None else tools
    if not tools:
        return ""
    lines = "\n".join(f"- {t.name}: {t.description}" for t in tools)
    # Measured on Cerebras qwen-3.8-27b (reasoning off, eva persona, the user_task
    # utterance, 8 tries each): a plain tool list -> 1-3/8 real calls, the rest were
    # "I've set the timer" without a call; this wording -> 8/8 (temp 0.7 and 0.8).
    return (
        "Tools, and this rule matters most: when they ask for a timer, a reminder or "
        "note, the weather, the time or to open a link, or when they say goodbye or that "
        "they have to go, your reply MUST include the function call (you may add a short "
        "aside like 'one sec', or the goodbye itself). You physically cannot do any of "
        "these by talking, so a reply without the call means it did not happen. "
        "The tools:\n" + lines
    )


def find_tool(name: str, tools: list[Tool] | None = None) -> Tool | None:
    for t in tools if tools is not None else _TOOLS:
        if t.name == name:
            return t
    return None


def schemas(tools: list[Tool]) -> list[dict[str, Any]]:
    """OpenAI ``tools=[...]`` payload for the given tools."""
    return [t.openai_schema() for t in tools]


def _coerce_args(tool: Tool, args: dict[str, Any]) -> dict[str, Any]:
    """Coerce argument types to the schema (small models send "120" for 120) and drop unknowns."""
    props: dict[str, Any] = tool.parameters.get("properties", {}) or {}
    out: dict[str, Any] = {}
    for key, val in (args or {}).items():
        if key not in props:
            continue
        typ = props[key].get("type")
        try:
            if typ == "integer" and not isinstance(val, bool):
                val = int(float(val))
            elif typ == "number":
                val = float(val)
            elif typ == "boolean" and isinstance(val, str):
                val = val.strip().lower() in ("true", "1", "yes", "on")
            elif typ == "string" and not isinstance(val, str):
                val = str(val)
        except (TypeError, ValueError):
            pass
        out[key] = val
    return out


async def execute(call: LLMToolCall, tools: list[Tool] | None = None) -> str:
    """Run one tool call and return its (spoken-style) result; never raises."""
    tool = find_tool(call.name, tools)
    if tool is None:
        return f"Error: there is no tool called {call.name}."
    args = _coerce_args(tool, call.arguments)
    missing = [k for k in tool.parameters.get("required", []) if k not in args]
    if missing:
        return f"Error: {call.name} is missing {', '.join(missing)}."
    t0 = time.perf_counter()
    try:
        if asyncio.iscoroutinefunction(tool.fn):
            result = await asyncio.wait_for(tool.fn(**args), timeout=TOOL_TIMEOUT_S)
        else:
            result = await asyncio.wait_for(asyncio.to_thread(tool.fn, **args), timeout=TOOL_TIMEOUT_S)
        result = str(result)
    except asyncio.TimeoutError:
        result = f"Error: {call.name} took too long and was cancelled."
    except Exception as e:  # noqa: BLE001 - tools must never crash the conversation
        log.warning("tool %s failed: %r", call.name, e)
        result = f"Error: {call.name} failed ({type(e).__name__}: {str(e)[:80]})."
    log.info("tool %s(%s) -> %r in %.3f s", call.name, args, result[:80], time.perf_counter() - t0)
    return result


__all__ = [
    "pending_events",
    "set_event_loop",
    "active_timers",
    "get_tools",
    "tool_notes",
    "find_tool",
    "schemas",
    "execute",
    "get_current_time",
    "set_timer",
    "remember_note",
    "recall_notes",
    "get_weather",
    "open_url",
    "NOTES_FILE",
]
