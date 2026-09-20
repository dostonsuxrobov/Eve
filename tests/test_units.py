"""Offline unit tests for the small pure modules: language assets, delivery gates, the
sentence chunker, the memory store, the loudness leveler and the persona loader."""
from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import numpy as np

from eva import lang
from eva.audio.leveler import ClipLeveler, Leveler, voiced_rms_dbfs
from eva.audio.echo import EchoDetector
from eva.delivery import echo_similarity, is_hesitation, looks_hallucinated, looks_incomplete, looks_like_echo
from eva.interfaces import Tool
from eva.llm.chunker import SentenceChunker
from eva.memory import Memory
from eva.personas import list_personas, load_persona, render
from eva.pipeline import _recover_tool_calls


# ------------------------------------------------------------------ language
def test_language_plans() -> None:
    langs = lang.load_languages()
    assert list(langs)[0] == "en" and "ru" in langs
    auto = lang.plan("auto")
    assert not auto.locked and auto.stt_language_code is None
    assert auto.voices_by_lang == {"en": langs["en"].voice, "ru": langs["ru"].voice}
    assert set(auto.by_lang("fillers")) == {"en", "ru"}
    ru = lang.plan("ru")
    assert ru.locked and ru.stt_language_code == "ru" and ru.voice == langs["ru"].voice
    assert ru.voices_by_lang == {} and list(ru.by_lang("fillers")) == ["ru"]
    assert lang.modes() == ["auto", "en", "ru"]


def test_personas_per_language() -> None:
    assert list_personas()[0] == "eva"
    en, ru = load_persona("eva", lang="en"), load_persona("eva", lang="ru")
    assert en.lang == "en" and ru.lang == "ru" and ru.template != en.template
    assert load_persona("calm_coach", lang="ru").lang == "en"  # no Russian file yet: falls back
    prompt = render(ru, supports_audio_tags=True, memory_text="", user_name="Достон", tool_notes="", locked_language="Russian")
    assert "{" not in prompt.replace("{{", "") or all(s not in prompt for s in ("{user_name}", "{now}", "{memory}", "{tool_notes}"))
    assert "Достон" in prompt and "always answer in Russian" in prompt
    auto_prompt = render(en, supports_audio_tags=False, memory_text="", user_name="Sam", tool_notes="")
    assert "language Sam just used" in auto_prompt


# ------------------------------------------------------------------- delivery
def test_phantom_gate_is_engine_aware() -> None:
    real = ["Yeah.", "No.", "Bye!", "Okay"]
    assert all(looks_hallucinated(t, {}, whisper_class=False) is None for t in real)
    assert all(looks_hallucinated(t, {}, whisper_class=True) for t in real)
    scores = {"segment_scores": [{"no_speech_prob": 0.1, "avg_logprob": -0.3}]}
    assert looks_hallucinated("Yeah.", scores) is not None  # scores present -> whisper-class
    assert looks_hallucinated("Thank you for watching.", {}, whisper_class=False)  # subtitle junk everywhere
    assert looks_hallucinated("...", {}) == "no letters"


def test_hesitations_and_completeness() -> None:
    assert is_hesitation("Uh.") and is_hesitation("Hmm, uh...") and is_hesitation("Ну...")
    assert not is_hesitation("Uh, not much.")
    assert looks_incomplete("Uh.") and looks_incomplete("So, uh") and looks_incomplete("I was thinking about the")
    assert not looks_incomplete("Not much.") and not looks_incomplete("yeah")


def test_self_echo() -> None:
    said = "Hey Doston, what's on your mind today?"
    assert looks_like_echo("What's on your mind today?", said)
    assert looks_like_echo("on your mind", said)
    assert not looks_like_echo("Not much, what about you?", said)
    assert not looks_like_echo("Not much.", "Not much, you?")  # two words are never judged
    assert not looks_like_echo("Hey", said)


# -------------------------------------------------------------------- chunker
def _chunks(deltas: list[str], **kw: int) -> list[str]:
    ch = SentenceChunker(first_chunk_min_chars=18, min_chunk_chars=10, **kw)
    out: list[str] = []
    for d in deltas:
        out += ch.feed(d)
    return out + ch.flush()


def test_chunker_keeps_leaked_tool_call_json_whole() -> None:
    tools = [Tool(name="end_conversation", description="", parameters={"type": "object", "properties": {"reason": {"type": "string"}}, "required": []}, fn=None, final=True)]
    chunks = _chunks(['[warm] Bye, take care.', '{ "name": "end_conversation", "arguments": { "reason": "bye" }\n', '}'])
    assert len(chunks) == 1
    text, calls = _recover_tool_calls(chunks[0], tools, [0])
    assert text.strip() == "[warm] Bye, take care." and [c.name for c in calls] == ["end_conversation"]
    assert calls[0].arguments == {"reason": "bye"}


def test_chunker_normal_text_and_stray_brace() -> None:
    assert _chunks(["Honestly, ", "that sounds rough.", " Want to talk about it?", " Or not, that is fine too.\n", "Either way."]) == [
        "Honestly, that sounds rough.", "Want to talk about it?", "Or not, that is fine too.", "Either way."
    ]
    lens = [len(c) for c in _chunks(["Well { " + "word " * 90 + ". And then more.", " And a bit more. Final one."])]
    assert lens[0] > 400 and len(lens) >= 3  # a stray "{" holds at most MAX_BRACE_HOLD chars


# --------------------------------------------------------------------- memory
def test_memory_dedupe_and_name_facts(tmp_path: Path) -> None:
    m = Memory(path=tmp_path / "m.json")
    assert m.add("Considers making a Caribbean, Italian, and Mexican-style meal.")
    assert not m.add("Considers making Caribbean, Italian, and Mexican-style meals.")
    m.facts += ["Has a friend named Doston.", "The user's name is Doster.", "Has a sister named Priya.", "Named their cat Doston."]
    assert m.drop_name_facts("Doston") == ["Has a friend named Doston.", "The user's name is Doster."]
    assert "Has a sister named Priya." in m.facts and "Named their cat Doston." in m.facts
    assert m.drop_name_facts("") == []


def test_memory_extractor_prompt_knows_the_user(tmp_path: Path) -> None:
    class FakeLLM:
        prompt = ""

        async def complete(self, messages, **kw):
            self.prompt = messages[1]["content"]
            return json.dumps(["The user's name is Doster.", "Has a cat named Miso."])

    llm = FakeLLM()
    m = Memory(path=tmp_path / "m.json")
    new = asyncio.run(m.update_from_transcript(llm, [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hey Doston"}], user_name="Doston", save=False))
    assert new == ["Has a cat named Miso."]
    assert "The person is Doston" in llm.prompt and "speech recognition" in llm.prompt


# -------------------------------------------------------------------- leveler
def _tone(dbfs: float, seconds: float = 1.0, sr: int = 24000) -> bytes:
    amp = 32767 * 10 ** (dbfs / 20) * math.sqrt(2)  # RMS -> peak for a sine
    t = np.arange(int(sr * seconds)) / sr
    return (np.clip(amp * np.sin(2 * np.pi * 220 * t), -32768, 32767)).astype(np.int16).tobytes()


def test_leveler_equalises_sources_without_pumping() -> None:
    lev = Leveler(24000, target_dbfs=-19.0, seeds_dbfs={"flash": -21.0})
    quiet_clip = lev.begin("flash/v")  # seeded 2 dB under target -> +2 dB from the first byte
    assert abs(20 * math.log10(quiet_clip.gain) - 2.0) < 0.05
    out = b"".join(quiet_clip.process(_tone(-21.0)[i : i + 4800]) for i in range(0, 48000, 4800))
    level, peak = voiced_rms_dbfs(out, 24000)
    assert abs(level - (-19.0)) < 0.3 and peak <= 1.0
    assert quiet_clip.finish() is not None and abs(lev.levels["flash/v"] - (-21.0)) < 0.3
    # an unseen loud source: unity gain on its first clip, learned for the next one
    loud = lev.begin("v3/x")
    assert loud.gain == 1.0
    loud.process(_tone(-13.0))
    loud.finish()
    assert abs(20 * math.log10(lev.gain_for("v3/x")) - (-6.0)) < 0.3
    # the gain is bounded (+7 dB) and a raised voice never hard-clips: soft knee above 0.8 FS
    lev.levels["ru"] = -40.0
    assert abs(20 * math.log10(lev.gain_for("ru")) - 7.0) < 0.05
    hot = ClipLeveler(lev, "k", 8.0)  # x8 on a -13 dBFS sine would peak at 2.5 FS without the knee
    a = np.frombuffer(hot.process(_tone(-13.0)), dtype=np.int16)
    assert 0.9 * 32767 < np.abs(a).max() < 32767


def test_fuzzy_echo_on_the_garbled_lines_from_the_live_log() -> None:
    # her sentence -> what Scribe made of it coming back through the speakers (2026-09-19 session)
    assert echo_similarity("А когда придёшь к ней, шей.", "Ладно, когда придёшь к решению") >= 0.6
    assert echo_similarity("Сделай агенту.", "К ней? К Евой-агенту?") >= 0.6
    assert echo_similarity("И это уже лучше, чем...", "у неё есть конкретная задача, и это уже лучше,") >= 0.6
    assert looks_like_echo("Вижу, у тебя", "А я с ней вижу. Вижу, у тебя", min_words=2, fuzzy=0.6)
    # a person talking over her
    assert echo_similarity("wait, stop, I have a question", "Yeah, those days happen. The ones where even making tea feels like a project.") < 0.6
    assert echo_similarity("what about your day", "Yeah, those days happen. The ones where even making tea feels like a project.") < 0.6


class _FakePlayer:
    sample_rate = 24000

    def __init__(self) -> None:
        self.hist: list[tuple[float, bytes]] = []

    def played_since(self, t: float) -> list[tuple[float, bytes]]:
        return [h for h in self.hist if h[0] >= t]


def _speechlike(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    x = np.convolve(rng.standard_normal(n), np.ones(8) / 8, mode="same")
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * np.arange(n) / sr) ** 2
    return (x * env / np.abs(x * env).max() * 0.5).astype(np.float32)


def _echo_scores(echo_gain: float, user_gain: float, lag_ms: int = 120) -> tuple[float, float, float]:
    """(median score, median lag ms, fraction flagged) over 1.5 s of mic audio."""
    rng = np.random.default_rng(7)
    sr_out, sr_mic = 24000, 16000
    player = _FakePlayer()
    det = EchoDetector(player, sr_mic)
    played = _speechlike(sr_out * 2, sr_out, rng)
    t0 = 1000.0
    block = sr_out // 50
    for i in range(0, played.size, block):
        player.hist.append((t0 + i / sr_out, (played[i : i + block] * 32767).astype(np.int16).tobytes()))
    played16 = np.interp(np.linspace(0, played.size - 1, played.size * 2 // 3), np.arange(played.size), played)
    lag = int(sr_mic * lag_ms / 1000)
    echo = np.convolve(np.concatenate([np.zeros(lag), played16])[: played16.size], np.ones(5) / 5, mode="same") * echo_gain
    mic = echo + _speechlike(played16.size, sr_mic, rng) * user_gain + rng.standard_normal(played16.size) * 0.02
    frame = sr_mic // 50
    res = []
    for i in range(0, mic.size - frame, frame):
        det.push_mic(mic[i : i + frame])
        if i > sr_mic // 2 and (i // frame) % 3 == 0:
            v = det.check(now=t0 + (i + frame) / sr_mic)
            res.append((v.score, v.lag_s * 1000, v.is_echo))
    return float(np.median([r[0] for r in res])), float(np.median([r[1] for r in res])), float(np.mean([r[2] for r in res]))


def test_echo_detector_separates_speaker_echo_from_the_user() -> None:
    score, lag, flagged = _echo_scores(0.1, 0.0)  # AEC residual, -20 dB
    assert flagged > 0.9 and abs(lag - 120) < 40, (score, lag, flagged)
    score, lag, flagged = _echo_scores(0.1, 0.0, lag_ms=300)
    assert flagged > 0.9 and abs(lag - 300) < 40, (score, lag, flagged)
    score, _, flagged = _echo_scores(0.0, 0.3)  # the user alone
    assert flagged < 0.1 and score < 0.25
    score, _, flagged = _echo_scores(0.1, 0.3)  # the user talking over the echo: counts as the user
    assert flagged < 0.15 and score < 0.3
    score, _, flagged = _echo_scores(0.0, 0.0)  # silence
    assert flagged == 0.0


# ------------------------------------------------------------------ phone page
def test_web_page_scripts_parse() -> None:
    """The page is one file with two AudioWorklets inlined as template strings; a stray
    backtick inside one once made the whole script fail to parse (the Start button did
    nothing). Check with node when it is installed; always check for backticks."""
    import re
    import shutil
    import subprocess
    import tempfile

    import pytest

    html = (Path(__file__).resolve().parent.parent / "eva" / "web" / "static" / "index.html").read_text(encoding="utf-8")
    script = re.search(r"<script>(.*)</script>", html, re.S).group(1)
    worklets = {name: re.search(name + r" = `(.*?)`;", script, re.S).group(1) for name in ("captureWorklet", "playWorklet")}
    for name, body in worklets.items():
        assert "`" not in body, f"backtick inside the {name} template string"
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed: syntax check skipped")
    d = Path(tempfile.mkdtemp())
    (d / "main.js").write_text(script, encoding="utf-8")
    assert subprocess.run([node, "--check", str(d / "main.js")], capture_output=True, text=True).returncode == 0
    shim = "const sampleRate=48000, currentTime=0; class AudioWorkletProcessor{constructor(){this.port={postMessage(){}}}}; function registerProcessor(){}\n"
    for name, body in worklets.items():
        (d / f"{name}.js").write_text(shim + body, encoding="utf-8")
        r = subprocess.run([node, "--check", str(d / f"{name}.js")], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[:300]
