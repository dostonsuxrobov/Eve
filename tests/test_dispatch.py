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


def test_gpt_live_hands_backend_function_calls_to_the_driver() -> None:
    """GPT-Live's backend calls arrive nested in response.event: collected from output_item.done,
    handed over once the backend response completes, its tokens costed at the backend's price."""
    from eva.s2s import make_client

    live = make_client("openai-live", "gpt-live-1+gpt-5.6-sol", "prompt", dispatch_tools())
    assert live.model == "gpt-live-1" and live.backend == "gpt-5.6-sol"
    live._backend({"type": "response.created", "response": {"id": "r1"}})
    live._backend({"type": "response.output_item.done", "response_id": "r1",
                   "item": {"type": "function_call", "call_id": "c1", "name": "load_status", "arguments": '{"reference": "7781234"}'}})
    assert live.events.empty() and live._busy == {"r1"}
    live._backend({"type": "response.completed", "response": {"id": "r1", "usage": {"input_tokens": 1000, "output_tokens": 100}}})
    ev = live.events.get_nowait()
    assert ev == {"type": "tool_calls", "calls": [("c1", "load_status", {"reference": "7781234"})]}
    assert not live._busy and live.backend_cost == pytest.approx((1000 * 4.0 + 100 * 20.0) / 1e6)


def test_latency_counts_her_voice_not_her_silence() -> None:
    """A full-duplex model streams audio all the time: only chunks above -45 dBFS count as her
    speaking, so the time to her first word isn't the time to her first silent packet."""
    from eva.s2s.driver import Call

    class Quiet:
        out_rate = 24000
        cost = 0.0

        def __init__(self) -> None:
            self.events: asyncio.Queue = asyncio.Queue()

        async def connect(self) -> None:
            pass

        async def close(self) -> None:
            pass

    async def go() -> None:
        c = Quiet()
        call = Call(c, [])
        await call.start()
        c.events.put_nowait({"type": "audio", "pcm": bytes(960)})  # 20 ms of digital silence
        tone = (np.sin(np.arange(480) / 3) * 8000).astype(np.int16).tobytes()
        c.events.put_nowait({"type": "audio", "pcm": tone})
        await asyncio.sleep(0.1)
        assert len(call.turn.voiced) == 1 and call.turn.first_audio_s is not None
        await call.close()

    import numpy as np

    asyncio.run(go())


def test_the_dispatcher_speaks_in_its_own_elevenlabs_voice(tmp_path: Path) -> None:
    """ElevenLabs models in a dispatch session use the job's voice, warmed up at start (it took
    3.95 s to its first audio cold on v3, 0.66 s warm); the companion keeps the owner's voice."""
    from eva.config import ELEVENLABS_VOICE, make_preset
    from eva.jobs import JOB_VOICES
    from eva.session import build_session

    from eva.jobs import job_settings

    job = build_session(make_preset("qwen4b", "v3"), user_name="Doston", job="dispatch")
    assert job.stack.voice.voice_id == JOB_VOICES["dispatch"] and job.stack.voice.warm_voice
    # no companion extras on a work call (owner: "a lot of fillers ... overdosed with scaffolding")
    assert job.fillers == {} and job.backchannels == {} and "Never write sound tags" in job.system_prompt
    s = job_settings(job.preset.settings)
    assert s.filler_after_ms == 0 and not s.backchannels and s.hold_tool_narration and job.preset.settings.filler_after_ms > 0
    mine = build_session(make_preset("qwen4b", "v3"), user_name="Doston", memory_path=tmp_path / "m.json")
    assert mine.stack.voice.voice_id == ELEVENLABS_VOICE and not mine.stack.voice.warm_voice


def test_a_reference_is_found_however_it_was_heard(desk: Desk) -> None:
    """Spoken calls: models asked for "PHX55120", "PHX 55120" and "PAX 55,120" for PHX-55120, and
    the exact match said there was no such load; the lookup now ignores spacing, dashes and case
    and falls back to the digits. A wrong number still finds nothing."""
    for said in ("PHX-55120", "PHX55120", "phx 55120", "PAX 55,120", "55120"):
        assert j(desk.load_status(said))["load"] == PLANTED["broken_load"], said
    assert "error" in j(desk.load_status("PH-556120"))


def test_barge_in_with_a_player_keeps_the_call_alive() -> None:
    """The owner's first OpenAI realtime session (2026-09-27) never heard him: its speech_started, sent
    at his first word, called the Player's properties as methods, the event loop died, and nothing
    after the greeting was processed. Now: playback stops, her reply is truncated at what was played,
    and the call keeps handling events."""
    from eva.audio.player import Player
    from eva.s2s.driver import Call

    class Fake:
        out_rate = 24000
        cost = 0.0
        audio_item = "item_1"

        def __init__(self) -> None:
            self.events: asyncio.Queue = asyncio.Queue()
            self.truncated: list[int] = []

        async def connect(self) -> None:
            pass

        async def truncate(self, ms: int) -> None:
            self.truncated.append(ms)

        async def close(self) -> None:
            pass

    async def go() -> None:
        fake = Fake()
        player = Player(24000)  # not started: nothing is played, buffered audio just waits
        call = Call(fake, [], player=player)
        await call.start()
        fake.events.put_nowait({"type": "audio", "pcm": b"\x10\x00" * 24000})  # a second of her voice queued
        fake.events.put_nowait({"type": "speech_started"})
        fake.events.put_nowait({"type": "text_in", "text": "Where's my load?"})
        await asyncio.sleep(0.2)
        assert fake.truncated == [0] and player.buffered_samples == 0 and call.turn.interrupted
        assert call.turn.heard == "Where's my load?" and not call.errors  # still listening after the barge-in
        await call.close()

    asyncio.run(go())
