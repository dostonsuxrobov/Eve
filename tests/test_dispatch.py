"""The dispatch job offline: the world's planted cases, the desk's hard rules, and the
speech-to-speech plumbing (schemas, costs, the tool loop) without a network."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

import pytest

from eva.jobs.dispatch.desk import Desk, dispatch_tools
from eva.jobs.dispatch.world import PLANTED


@pytest.fixture(scope="module")
def desk(tmp_path_factory: pytest.TempPathFactory) -> Desk:
    path = tmp_path_factory.mktemp("world") / "dispatch.db"
    return Desk(path, rebuild=True, now=datetime.now())


def j(s: str) -> dict:
    return json.loads(s)


def test_world_is_big_and_joined(desk: Desk) -> None:
    n = {t: desk.q1(f"SELECT COUNT(*) AS n FROM {t}")["n"] for t in ("postings", "loads", "pings", "brokers", "trucks")}
    assert n["postings"] > 5000 and n["loads"] > 20000 and n["pings"] > 50000 and n["brokers"] >= 200 and n["trucks"] > 120


def test_lowball_load_is_below_floor_and_bookable_above(desk: Desk) -> None:
    board = j(desk.search_load_board("Dallas", "Atlanta", "reefer", "tomorrow"))
    assert PLANTED["lowball_posting"] in [x["id"] for x in board["loads"]]
    price = j(desk.price_load(PLANTED["lowball_posting"], truck=PLANTED["lowball_truck"]))
    assert price["posted_rate"] < price["floor_rate"] < price["target_ask"]
    below = j(desk.book_load(PLANTED["lowball_posting"], PLANTED["lowball_truck"], price["floor_rate"] - 50))
    assert below["booked"] is False and "floor" in below["error"]


def test_do_not_use_broker_is_refused_whatever_the_rate(desk: Desk) -> None:
    profile = j(desk.broker_profile("Rapid Eagle"))
    assert profile["status"] == "do not use" and profile["why"]
    out = j(desk.book_load(PLANTED["dnu_posting"], "330", 5000))
    assert out["booked"] is False and "do-not-use" in out["error"]


def test_hazmat_needs_an_endorsed_driver(desk: Desk) -> None:
    trucks = j(desk.find_available_trucks("Houston", "dry van", "tomorrow", hazmat=True))
    assert PLANTED["hazmat_truck"] in [t["truck"] for t in trucks["available"]]
    assert PLANTED["hazmat_decoy_truck"] in [t["truck"] for t in trucks["nearby_but_not_available"]]
    out = j(desk.book_load(PLANTED["hazmat_posting"], PLANTED["hazmat_decoy_truck"], 5000))
    assert out["booked"] is False and "hazmat" in out["error"]


def test_late_load_and_breakdown_tell_the_truth(desk: Desk) -> None:
    late = j(desk.load_status(PLANTED["late_ref"]))
    assert late["load"] == PLANTED["late_load"] and "late" in late["vs_appointment"] and "Strict" in late["receiver_rules"]
    broken = j(desk.load_status("PHX-55120"))
    assert broken["eta"].startswith("none") and "Turbo failure" in broken["incidents"][0]
    rescue = j(desk.find_available_trucks("Oklahoma City", "reefer", "today", radius_miles=200))
    assert PLANTED["rescue_truck"] in [t["truck"] for t in rescue["available"]]
    assert PLANTED["broken_truck"] in [t["truck"] for t in rescue["nearby_but_not_available"]]


def test_places_resolve_the_way_callers_say_them(desk: Desk) -> None:
    assert desk.resolve("DFW") == desk.resolve("Dallas, TX") == desk.resolve("dallas texas")
    assert len(desk.resolve("Ohio")) >= 5 and len(desk.resolve("the Midwest")) > 15
    assert desk.equipment("refrigerated") == "reefer" and desk.equipment("53 van") == "dry_van"


def test_tools_have_schemas_every_backend_takes() -> None:
    from eva.s2s.gemini_live import declarations
    from eva.s2s.openai_rt import tool_schema

    tools = dispatch_tools()
    names = [t.name for t in tools]
    assert {"search_load_board", "find_available_trucks", "price_load", "book_load", "load_status", "end_conversation"} <= set(names)
    oa = tool_schema(tools)
    assert all(t["type"] == "function" and t["parameters"]["type"] == "object" for t in oa)
    g = declarations(tools, "BLOCKING")
    book = next(d for d in g if d["name"] == "book_load")
    assert book["parameters"]["type"] == "OBJECT" and book["parameters"]["properties"]["rate"]["type"] == "INTEGER"
    assert set(book["parameters"]["required"]) == {"posting_id", "truck", "rate"}


def test_costs_follow_the_price_tables() -> None:
    from eva.s2s.gemini_live import cost_of as gcost
    from eva.s2s.openai_rt import cost_of as ocost

    usage = {"input_token_details": {"text_tokens": 1000, "audio_tokens": 100, "cached_tokens_details": {"text_tokens": 500}},
             "output_token_details": {"text_tokens": 50, "audio_tokens": 200}}
    # mini: 500 text * 0.60 + 500 cached * 0.06 + 100 audio * 10 + 50 * 2.40 + 200 * 20, per million
    assert ocost("gpt-realtime-2.1-mini", usage) == pytest.approx((500 * 0.6 + 500 * 0.06 + 1000 + 120 + 4000) / 1e6)
    g = {"promptTokensDetails": [{"modality": "TEXT", "tokenCount": 2000}, {"modality": "AUDIO", "tokenCount": 250}],
         "responseTokensDetails": [{"modality": "AUDIO", "tokenCount": 300}]}
    assert gcost("gemini-3.8-live", g) == pytest.approx((250 * 3 + 2000 * 0.75 + 300 * 12) / 1e6)


def test_call_runs_tools_and_waits_for_the_answer(desk: Desk, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fake speech-to-speech client that calls load_status, then speaks: the driver runs the real
    tool, sends the result back, and returns one turn with both."""
    import eva.jobs.dispatch.desk as desk_mod
    from eva.s2s.driver import Call

    monkeypatch.setattr(desk_mod, "_desk", desk)

    class Fake:
        out_rate = 24000
        cost = 0.0

        def __init__(self) -> None:
            self.events: asyncio.Queue = asyncio.Queue()
            self.results: list = []

        async def connect(self) -> None:
            pass

        async def send_text(self, text: str, role: str = "user") -> None:
            self.events.put_nowait({"type": "tool_calls", "calls": [("c1", "load_status", {"reference": PLANTED["late_ref"]})]})
            self.events.put_nowait({"type": "turn_done", "cost": 0.001, "more": True})

        async def send_tool_results(self, results: list) -> None:
            self.results = results
            self.events.put_nowait({"type": "audio", "pcm": b"\0\0" * 2400})
            self.events.put_nowait({"type": "text_out", "delta": "She's running about forty-five minutes late."})
            self.events.put_nowait({"type": "turn_done", "cost": 0.002, "more": False})

        async def truncate(self, ms: int) -> None:
            pass

        async def close(self) -> None:
            pass

    async def go() -> None:
        fake = Fake()
        call = Call(fake, dispatch_tools())
        await call.start()
        turn = await call.ask("Where's PO 7781234?", quiet_s=0.2, timeout_s=10)
        await call.close()
        assert [t["name"] for t in turn.tools] == ["load_status"] and PLANTED["late_load"] in fake.results[0][2]
        assert "late" in turn.said and turn.first_audio_s is not None and turn.cost == pytest.approx(0.003)

    asyncio.run(go())


def test_a_job_never_touches_the_owners_memory(tmp_path: Path) -> None:
    from eva.config import make_preset
    from eva.session import build_session

    s = build_session(make_preset("qwen4b", "kokoro"), user_name="Doston", job="dispatch")
    assert s.job == "dispatch" and "Red Oak Transport" in s.system_prompt and s.memory.facts == []
    assert "data" in str(s.memory.path) and "dispatch" in str(s.memory.path)
    assert {t.name for t in s.tools} >= {"book_load", "load_status"} and s.greeting_event("Doston")["type"] == "job_start"
