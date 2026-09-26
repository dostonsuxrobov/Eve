"""eva.failover on doubles: a dead or hanging primary degrades to the backup within one
request, the primary is probed again after the cooldown, and nothing is retried once a
result has started."""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

import numpy as np

from eva.failover import FailoverLLM, FailoverSTT, FailoverTTS
from eva.interfaces import LLMDelta, LLMDone, Transcript


class Brain:
    def __init__(self, name: str, *, fail: bool = False, hang: bool = False, text: str = "hi") -> None:
        self.name, self.fail, self.hang, self.text = name, fail, hang, text
        self.calls = 0
        self.pings = 0

    async def warmup(self) -> None: ...

    async def close(self) -> None: ...

    async def ping(self) -> float | None:
        self.pings += 1
        return None if self.fail else 0.01

    async def complete(self, messages: list[dict[str, Any]], **kw: Any) -> str:
        self.calls += 1
        if self.fail:
            raise ConnectionError("down")
        return self.text

    async def stream(self, messages: list[dict[str, Any]], tools: Any = None) -> AsyncIterator[Any]:
        self.calls += 1
        if self.fail:
            raise ConnectionError("down")
        if self.hang:
            await asyncio.sleep(60)
        yield LLMDelta(self.text)
        yield LLMDone(finish_reason="stop", ttft_s=0.01, total_s=0.02)


async def _drain(agen: AsyncIterator[Any]) -> list[Any]:
    return [ev async for ev in agen]


def test_llm_dead_primary_falls_back_and_recovers() -> None:
    async def go() -> None:
        events: list[tuple[str, dict[str, Any]]] = []
        cloud, local = Brain("cloud", fail=True), Brain("local", text="local says hi")
        llm = FailoverLLM(cloud, local, cooldown_s=0.2, on_event=lambda n, d: events.append((n, d)))
        evs = await _drain(llm.stream([{"role": "user", "content": "hi"}]))
        assert [e.text for e in evs if isinstance(e, LLMDelta)] == ["local says hi"]
        assert events[0][0] == "failover" and events[0][1]["to"] == "local"
        assert "fallback" in llm.name
        # inside the cooldown the primary is not even tried
        await _drain(llm.stream([]))
        assert cloud.calls == 1 and local.calls == 2
        # after the cooldown it is probed again; still dead -> backup again, no crash
        await asyncio.sleep(0.25)
        await _drain(llm.stream([]))
        assert cloud.calls == 2 and local.calls == 3
        # it comes back: the next request after the cooldown goes to the cloud
        cloud.fail = False
        await asyncio.sleep(0.25)
        evs = await _drain(llm.stream([]))
        assert [e.text for e in evs if isinstance(e, LLMDelta)] == ["hi"]
        assert events[-1][0] == "recovered" and llm.name == "cloud"

    asyncio.run(go())


def test_llm_hanging_primary_times_out_to_backup() -> None:
    async def go() -> None:
        cloud, local = Brain("cloud", hang=True), Brain("local", text="local")
        llm = FailoverLLM(cloud, local, first_token_timeout_s=0.1)
        evs = await _drain(llm.stream([]))
        assert [e.text for e in evs if isinstance(e, LLMDelta)] == ["local"]
        assert (await llm.complete([])) == "local"  # complete() follows the same state

    asyncio.run(go())


def test_llm_ping_recovers_primary() -> None:
    async def go() -> None:
        cloud, local = Brain("cloud", fail=True), Brain("local")
        llm = FailoverLLM(cloud, local, cooldown_s=0.0)
        await _drain(llm.stream([]))
        assert llm.health.down_since is not None
        cloud.fail = False
        await llm.ping()
        assert llm.health.down_since is None and llm.active is cloud

    asyncio.run(go())


class CloudSTT:
    name = "cloud-stt"
    sample_rate = 16000

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.fed = 0

    async def warmup(self) -> None: ...

    async def close(self) -> None: ...

    async def feed(self, pcm: np.ndarray) -> None:
        self.fed += 1

    async def commit(self) -> Transcript:
        if self.fail:
            raise ConnectionError("down")
        return Transcript(text="cloud text", latency_s=0.1)

    async def discard(self, keep_audio: bool = False) -> None: ...

    async def transcribe_batch(self, pcm: np.ndarray, sample_rate: int = 16000) -> Transcript:
        if self.fail:
            raise ConnectionError("down")
        return Transcript(text="cloud batch", latency_s=0.2)

    async def transcribe(self, pcm: np.ndarray, sample_rate: int = 16000) -> Transcript:
        return await self.transcribe_batch(pcm, sample_rate)


class LocalSTT:
    name = "local-stt"

    async def warmup(self) -> None: ...

    async def close(self) -> None: ...

    async def transcribe(self, pcm: np.ndarray, sample_rate: int = 16000) -> Transcript:
        return Transcript(text="local text", latency_s=0.3)


def test_stt_commit_failure_then_batch_goes_local() -> None:
    async def go() -> None:
        cloud, local = CloudSTT(fail=True), LocalSTT()
        stt = FailoverSTT(cloud, local, cooldown_s=60)
        pcm = np.zeros(16000, dtype=np.int16)
        await stt.feed(pcm)  # primary still believed up: streamed
        try:
            await stt.commit()
        except ConnectionError:
            pass
        else:
            raise AssertionError("commit should raise so the pipeline retries in batch")
        tr = await stt.transcribe_batch(pcm)  # the batch retry lands on the local model
        assert tr.text == "local text" and tr.meta["fallback_provider"] == "local-stt"
        # while down, feed() refuses at once so the pipeline buffers for batch
        try:
            await stt.feed(pcm)
        except RuntimeError:
            pass
        else:
            raise AssertionError("feed must raise while the primary is down")
        assert (await stt.transcribe(pcm)).text == "local text"

    asyncio.run(go())


class Voice:
    sample_rate = 24000
    supports_audio_tags = False

    def __init__(self, name: str, *, fail: bool = False, hang: bool = False, tags: bool = False) -> None:
        self.name, self.fail, self.hang = name, fail, hang
        self.supports_audio_tags = tags
        self.supports_cues = tags
        self.turns = 0

    async def warmup(self) -> None: ...

    async def close(self) -> None: ...

    def begin_turn(self) -> None:
        self.turns += 1

    async def synthesize(self, text: str, cue: str | None = None) -> AsyncIterator[bytes]:
        if self.fail:
            raise ConnectionError("down")
        if self.hang:
            await asyncio.sleep(60)
        yield self.name.encode()
        yield b"!!"


def test_tts_falls_back_before_first_byte_and_reports_active_capabilities() -> None:
    async def go() -> None:
        cloud, local = Voice("cloud", fail=True, tags=True), Voice("local")
        tts = FailoverTTS(cloud, local, cooldown_s=60)
        assert tts.supports_audio_tags is True  # primary believed up
        tts.begin_turn()
        out = b"".join([b async for b in tts.synthesize("hello", cue="warm")])
        assert out == b"local!!"
        assert tts.supports_audio_tags is False and "fallback" in tts.name  # now the local voice decides
        assert cloud.turns == 1 and local.turns == 1

        slow = FailoverTTS(Voice("slow", hang=True), Voice("local2"), first_byte_timeout_s=0.1)
        assert b"".join([b async for b in slow.synthesize("x")]) == b"local2!!"

    asyncio.run(go())


def test_tts_sample_rate_mismatch_is_refused() -> None:
    a, b = Voice("a"), Voice("b")
    b.sample_rate = 22050
    try:
        FailoverTTS(a, b)
    except ValueError:
        return
    raise AssertionError("mismatched sample rates must be rejected")
