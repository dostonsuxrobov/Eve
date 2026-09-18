"""Real-run checks and measurements for the eva.audio layer.

Run:  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe bench/test_audio.py [--no-devices] [--json]

Sections
  1. UtteranceSegmenter over samples/*.wav via FileMic(realtime=False): one SpeechEnd per file
  2. SpeechEnd.pcm length vs an energy-based estimate of the spoken length
  3. Player: 0.6 s 440 Hz tone, stop() after 200 ms -> stop latency, played samples
  4. Mic: open the real microphone for 1 s -> frame count, peak amplitude
  5. SileroVAD per-window inference time
  6. Extras: LinearResampler accuracy, ScriptedMic (two utterances, realtime pacing),
     realtime FileMic endpoint latency (speech end -> SpeechEnd event)
Results are printed and, with --json, written to bench/out/audio_results.json.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eva.config import SAMPLES_DIR, PipelineSettings  # noqa: E402
from eva.audio.mic import FileMic, LinearResampler, Mic, ScriptedMic, load_pcm  # noqa: E402
from eva.audio.player import Player  # noqa: E402
from eva.audio.vad import SileroVAD, SpeechEnd, SpeechStart, UtteranceSegmenter  # noqa: E402

RESULTS: dict[str, object] = {}
FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        FAILURES.append(msg)


def spoken_span(pcm: np.ndarray, sr: int, rel_thresh: float = 0.03, win_ms: int = 20) -> tuple[float, float]:
    """(start_s, end_s) of the region whose 20 ms RMS exceeds rel_thresh * peak RMS."""
    n = int(sr * win_ms / 1000)
    frames = len(pcm) // n
    x = pcm[: frames * n].astype(np.float32).reshape(frames, n)
    rms = np.sqrt((x**2).mean(axis=1))
    active = np.where(rms > rel_thresh * rms.max())[0]
    if not len(active):
        return 0.0, 0.0
    return active[0] * n / sr, (active[-1] + 1) * n / sr


# ------------------------------------------------------------------- 1 + 2: segmenter
async def section_segmenter(settings: PipelineSettings) -> None:
    print("\n[1] UtteranceSegmenter over samples/*.wav (FileMic realtime=False)")
    vad = SileroVAD()
    seg = UtteranceSegmenter(settings, vad)
    out: dict[str, object] = {}
    for path in sorted(glob.glob(str(SAMPLES_DIR / "*.wav"))):
        name = Path(path).stem
        seg.reset()
        mic = FileMic(path, realtime=False)
        events: list[tuple[float, SpeechStart | SpeechEnd]] = []
        n_frames = 0
        t0 = time.perf_counter()
        async for frame in mic.frames():
            n_frames += 1
            assert frame.dtype == np.int16 and frame.shape == (320,), (frame.dtype, frame.shape)
            for ev in seg.feed(frame):
                events.append((n_frames * mic.frame_ms / 1000, ev))
        proc_s = time.perf_counter() - t0
        starts = [t for t, e in events if isinstance(e, SpeechStart)]
        ends = [(t, e) for t, e in events if isinstance(e, SpeechEnd)]
        sp_start, sp_end = spoken_span(mic.speech, mic.sample_rate)
        spoken_s = sp_end - sp_start
        print(f"  {name}: {n_frames} frames ({mic.duration_s:.2f} s incl. 0.5 s lead + 2.0 s tail), "
              f"file speech {len(mic.speech)/16000:.2f} s, energy-span {spoken_s:.2f} s, processed in {proc_s*1000:.0f} ms")
        for t in starts:
            print(f"     SpeechStart at mic t={t:.2f} s (file speech begins at {mic.speech_start_s + sp_start:.2f} s)")
        for t, e in ends:
            print(f"     SpeechEnd   at mic t={t:.2f} s, pcm={len(e.pcm)} samples = {e.duration_s:.3f} s "
                  f"(file speech ends at {mic.speech_start_s + sp_end:.2f} s)")
        check(len(starts) == 1 and len(ends) == 1, f"{name}: exactly one SpeechStart and one SpeechEnd (got {len(starts)}/{len(ends)})")
        rec: dict[str, object] = {"frames": n_frames, "starts": len(starts), "ends": len(ends), "proc_ms": round(proc_s * 1000, 1)}
        if ends:
            t_end, e = ends[0]
            expected = spoken_s + settings.prespeech_buffer_ms / 1000 + 0.150
            ratio = e.duration_s / expected if expected else 0.0
            print(f"     [2] pcm {e.duration_s:.3f} s vs expected ~{expected:.3f} s (spoken + 0.3 s prespeech + 0.15 s tail): ratio {ratio:.3f}")
            check(0.85 <= ratio <= 1.15, f"{name}: SpeechEnd.pcm length within 15% of spoken length")
            check(e.pcm.dtype == np.int16, f"{name}: SpeechEnd.pcm is int16")
            # trailing silence in the emitted pcm must be <= 150 ms (+ one VAD window slack)
            _, pe = spoken_span(e.pcm, 16000)
            tail_ms = (e.duration_s - pe) * 1000
            print(f"     trailing low-energy tail in pcm: {tail_ms:.0f} ms")
            check(tail_ms <= 150 + 32 + 20, f"{name}: trailing silence trimmed to <= ~150 ms (got {tail_ms:.0f} ms)")
            check(t_end <= mic.duration_s - 0.5, f"{name}: SpeechEnd produced by the 2 s trailing silence, well before the file ends")
            endpoint_delay = t_end - (mic.speech_start_s + sp_end)
            print(f"     endpoint delay (energy speech end -> SpeechEnd): {endpoint_delay*1000:.0f} ms (setting: {settings.endpoint_silence_ms} ms)")
            rec.update(pcm_s=round(e.duration_s, 3), expected_s=round(expected, 3), ratio=round(ratio, 3),
                       tail_ms=round(tail_ms, 1), endpoint_delay_ms=round(endpoint_delay * 1000, 1))
        out[name] = rec
    RESULTS["segmenter"] = out


# ------------------------------------------------------------------------- 3: player
async def section_player(device: int | None) -> None:
    print("\n[3] Player: 0.6 s 440 Hz tone at 24 kHz, stop() after 200 ms")
    sr = 24000
    t = np.arange(int(0.6 * sr)) / sr
    tone = (0.3 * np.sin(2 * math.pi * 440 * t) * 32767).astype(np.int16)
    player = Player(sample_rate=sr, device=device, block_ms=20)
    player.start()
    print(f"  device: {player.device_name}, reported output latency {player.output_latency_s*1000:.0f} ms")
    await asyncio.sleep(0.05)  # let the stream settle
    player.mark()
    t_write = time.perf_counter()
    player.write(tone.tobytes())
    await asyncio.sleep(0.200)
    t_stop = time.perf_counter()
    played = player.stop()
    t_ret = time.perf_counter()
    # wait for the first callback after stop(): that block is already silence
    while (player.last_callback_t or 0) <= t_stop:
        await asyncio.sleep(0.0005)
    t_cb = player.last_callback_t or t_ret
    await asyncio.sleep(0.1)
    played_after = player.played_samples
    first_audio = player.first_audio_t
    dev_lat = player.output_latency_s or 0.0
    print(f"  stop() call duration: {(t_ret - t_stop)*1e6:.0f} us")
    print(f"  first silent callback after stop(): +{(t_cb - t_stop)*1000:.1f} ms  (block = {player.block_ms} ms)")
    print(f"  device output latency (PortAudio): {dev_lat*1000:.1f} ms -> physical silence ~{((t_cb - t_stop) + dev_lat)*1000:.0f} ms after stop()")
    print(f"  write -> first real block in callback: {((first_audio or t_write) - t_write)*1000:.1f} ms")
    print(f"  played samples at stop(): {played} = {played/sr*1000:.0f} ms (wall time since write: {(t_stop - t_write)*1000:.0f} ms); after: {played_after}")
    print(f"  buffered after stop: {player.buffered_seconds:.3f} s, is_active={player.is_active}")
    check(0 < played <= int(0.26 * sr), "played samples at stop() ~ 200 ms of audio (bounded by wall time)")
    check(played_after == played, "played counter frozen after stop()")
    check(not player.is_active, "buffer empty after stop()")
    check((t_cb - t_stop) * 1000 <= player.block_ms + 15, "stop() takes effect within ~one block")
    # second utterance: full playback + wait_until_done
    player.mark()
    t0 = time.perf_counter()
    player.write(tone.tobytes())
    await player.wait_until_done()
    done_s = time.perf_counter() - t0
    print(f"  full 0.6 s tone: wait_until_done resolved after {done_s*1000:.0f} ms, played {player.played_samples} samples (expect {len(tone)})")
    check(player.played_samples == len(tone), "all samples played on a full utterance")
    check(0.55 <= done_s <= 0.80, "wait_until_done resolves ~ tone length + one block")
    player_name = player.device_name
    player.close()
    RESULTS["player"] = {
        "stop_call_us": round((t_ret - t_stop) * 1e6, 1),
        "stop_to_silent_callback_ms": round((t_cb - t_stop) * 1000, 2),
        "device_output_latency_ms": round(dev_lat * 1000, 1),
        "write_to_first_block_ms": round(((first_audio or t_write) - t_write) * 1000, 2),
        "played_at_stop": played,
        "played_at_stop_ms": round(played / sr * 1000, 1),
        "wall_ms_at_stop": round((t_stop - t_write) * 1000, 1),
        "full_tone_wait_ms": round(done_s * 1000, 1),
        "callback_errors": player.callback_errors,
        "device": player_name,
    }


# ---------------------------------------------------------------------------- 4: mic
async def section_mic(device: int | None) -> None:
    print("\n[4] Mic: real microphone for 1 s")
    mic = Mic(device=device)
    frames: list[np.ndarray] = []
    t0 = time.perf_counter()
    t_first: float | None = None
    try:
        async with mic:
            print(f"  opened at native rate {mic.native_rate} Hz (resampling={'yes' if mic.native_rate != 16000 else 'no'}), "
                  f"stream latency {mic._stream.latency*1000:.0f} ms")
            t0 = time.perf_counter()

            async def stopper() -> None:  # frames() only ends on stop(); never hang if no callbacks arrive
                await asyncio.sleep(1.0)
                mic.stop()

            asyncio.create_task(stopper())
            async for frame in mic.frames():
                if t_first is None:
                    t_first = time.perf_counter()
                frames.append(frame)
    except Exception as e:
        print(f"  FAIL opening the microphone: {e!r}")
        base = getattr(sys, "_base_executable", sys.executable)
        if "WindowsApps" in base:
            print("  DIAGNOSIS: this venv runs on the Microsoft Store Python (packaged app) whose manifest declares no")
            print("  'microphone' capability; Windows records Deny for it in the microphone consent store, so every host API")
            print("  (MME/DirectSound/WASAPI) fails to open an input stream while a plain CPython opens it fine.")
            print("  FIX A: Settings > Privacy & security > Microphone > turn on 'Python 3.13' (the packaged-app entry).")
            print("  FIX B: rebuild .venv on a non-Store CPython: uv python install 3.13; uv venv --python 3.13 .venv; reinstall requirements.")
            print(r"  Verify with:  C:\Python314\python.exe bench/test_audio.py --mic-only   (needs only numpy + sounddevice)")
        FAILURES.append(f"mic: {e!r}")
        RESULTS["mic"] = {"error": repr(e), "base_executable": base}
        return
    elapsed = time.perf_counter() - t0
    peak = int(max((int(np.abs(f).max()) for f in frames), default=0))
    rms = float(np.sqrt(np.mean(np.concatenate(frames).astype(np.float32) ** 2))) if frames else 0.0
    shapes_ok = all(f.dtype == np.int16 and f.shape == (320,) for f in frames)
    print(f"  frames in {elapsed*1000:.0f} ms: {len(frames)} (expect ~50), first frame after {((t_first or t0) - t0)*1000:.0f} ms")
    print(f"  peak amplitude {peak} ({peak/32768*100:.1f}% FS), rms {rms:.0f}; dropped={mic.dropped_frames} status_flags={mic.callback_errors}")
    check(shapes_ok, "every mic frame is int16 (320,)")
    check(40 <= len(frames) <= 60, f"~50 frames in 1 s (got {len(frames)})")
    check(peak > 0, "mic delivers non-zero audio (peak > 0)")
    check(mic.dropped_frames == 0, "no dropped frames")
    RESULTS["mic"] = {"native_rate": mic.native_rate, "frames_1s": len(frames), "elapsed_ms": round(elapsed * 1000, 1),
                      "first_frame_ms": round(((t_first or t0) - t0) * 1000, 1), "peak": peak, "rms": round(rms, 1),
                      "dropped": mic.dropped_frames}


# ------------------------------------------------------------------------- 5: vad speed
def section_vad_speed() -> None:
    print("\n[5] SileroVAD per-window inference time")
    vad = SileroVAD()
    pcm = load_pcm(SAMPLES_DIR / "user_rough_day.wav")
    windows = [pcm[i : i + 512] for i in range(0, len(pcm) - 512, 512)]
    for w in windows[:20]:  # warm up
        vad(w)
    vad.reset()
    times: list[float] = []
    probs: list[float] = []
    for w in windows:
        t0 = time.perf_counter()
        probs.append(vad(w))
        times.append(time.perf_counter() - t0)
    ms = [t * 1000 for t in times]
    ms_sorted = sorted(ms)
    p95 = ms_sorted[int(0.95 * (len(ms) - 1))]
    print(f"  {len(ms)} windows: mean {statistics.mean(ms):.3f} ms, median {statistics.median(ms):.3f} ms, "
          f"p95 {p95:.3f} ms, max {max(ms):.3f} ms  (window = 32 ms -> {32/statistics.mean(ms):.0f}x realtime)")
    speech_frac = sum(p > 0.5 for p in probs) / len(probs)
    print(f"  speech windows (p>0.5): {speech_frac*100:.0f}% of the rough_day clip; max prob {max(probs):.3f}")
    check(statistics.mean(ms) < 5.0, "mean VAD inference < 5 ms per window")
    check(max(probs) > 0.9, "VAD detects speech in the sample (max prob > 0.9)")
    # silence sanity
    vad.reset()
    sil = [vad(np.zeros(512, np.int16)) for _ in range(10)]
    noise = [vad((np.random.default_rng(0).standard_normal(512) * 200).astype(np.int16)) for _ in range(10)]
    print(f"  prob on digital silence: max {max(sil):.3f}; on low white noise (-44 dBFS): max {max(noise):.3f} (threshold 0.5)")
    check(max(sil) < 0.1 and max(noise) < 0.5, "VAD stays below threshold on silence / low noise")
    RESULTS["vad"] = {"windows": len(ms), "mean_ms": round(statistics.mean(ms), 3), "median_ms": round(statistics.median(ms), 3),
                      "p95_ms": round(p95, 3), "max_ms": round(max(ms), 3), "speech_fraction": round(speech_frac, 3)}


# -------------------------------------------------------------------------- 6: extras
def section_resampler() -> None:
    print("\n[6a] LinearResampler 48 kHz -> 16 kHz, streamed in uneven blocks")
    src_sr, dst_sr, f0 = 48000, 16000, 440.0
    n = src_sr  # 1 s
    x = 0.5 * np.sin(2 * math.pi * f0 * np.arange(n) / src_sr)
    rs = LinearResampler(src_sr, dst_sr)
    out: list[np.ndarray] = []
    pos = 0
    rng = np.random.default_rng(1)
    while pos < n:
        blk = int(rng.integers(700, 1300))
        out.append(rs.process(x[pos : pos + blk]))
        pos += blk
    y = np.concatenate(out)
    ref = 0.5 * np.sin(2 * math.pi * f0 * np.arange(len(y)) / dst_sr)
    err = float(np.sqrt(np.mean((y - ref) ** 2)))
    jumps = float(np.abs(np.diff(y)).max())
    print(f"  out {len(y)} samples (expect ~{n * dst_sr // src_sr}), rms error vs ideal {err:.5f} (signal rms 0.354), max sample step {jumps:.3f}")
    check(abs(len(y) - n * dst_sr // src_sr) <= 2, "resampled length correct")
    check(err < 0.01, "resampler error < 1% of full scale")
    RESULTS["resampler"] = {"len": len(y), "rms_err": round(err, 6)}


async def section_scripted(settings: PipelineSettings) -> None:
    print("\n[6b] ScriptedMic: hello, 1.0 s gap, task, 1.0 s gap (realtime)")
    items = [(str(SAMPLES_DIR / "user_hello.wav"), 1.0), (str(SAMPLES_DIR / "user_task.wav"), 1.0)]
    mic = ScriptedMic(items, leading_silence_s=0.3)
    seg = UtteranceSegmenter(settings)
    t0 = time.perf_counter()
    n = 0
    ends: list[tuple[float, SpeechEnd]] = []
    starts: list[float] = []
    async for frame in mic.frames():
        n += 1
        for ev in seg.feed(frame):
            if isinstance(ev, SpeechEnd):
                ends.append((ev.t - t0, ev))
            else:
                starts.append(ev.t - t0)
    wall = time.perf_counter() - t0
    print(f"  {n} frames in {wall:.2f} s wall (script {mic.duration_s:.2f} s) -> pacing error {(wall - mic.duration_s)*1000:+.0f} ms")
    for t in starts:
        print(f"     SpeechStart at {t:.2f} s")
    for t, e in ends:
        print(f"     SpeechEnd   at {t:.2f} s, pcm {e.duration_s:.2f} s")
    check(len(starts) == 2 and len(ends) == 2, f"two utterances detected (got {len(starts)}/{len(ends)})")
    check(abs(wall - mic.duration_s) < 0.1, "realtime pacing within 100 ms over the whole script")
    RESULTS["scripted"] = {"frames": n, "wall_s": round(wall, 3), "script_s": round(mic.duration_s, 3),
                           "starts": [round(t, 3) for t in starts], "ends": [round(t, 3) for t, _ in ends]}


async def section_realtime_endpoint(settings: PipelineSettings) -> None:
    print("\n[6c] Realtime FileMic(user_hello): wall-clock endpoint latency")
    mic = FileMic(SAMPLES_DIR / "user_hello.wav", realtime=True, trailing_silence_s=1.5)
    seg = UtteranceSegmenter(settings)
    _, sp_end = spoken_span(mic.speech, mic.sample_rate)
    t_speech_end_expected = mic.speech_start_s + sp_end
    t0 = time.perf_counter()
    end_t: float | None = None
    start_t: float | None = None
    feed_us: list[float] = []
    async for frame in mic.frames():
        a = time.perf_counter()
        evs = seg.feed(frame)
        feed_us.append((time.perf_counter() - a) * 1e6)
        for ev in evs:
            if isinstance(ev, SpeechStart):
                start_t = ev.t - t0
            elif isinstance(ev, SpeechEnd):
                end_t = ev.t - t0
    assert end_t is not None
    lat = end_t - t_speech_end_expected
    print(f"  SpeechStart at {start_t:.3f} s, SpeechEnd at {end_t:.3f} s; speech actually ended at ~{t_speech_end_expected:.3f} s")
    print(f"  endpoint latency (speech end -> SpeechEnd event): {lat*1000:.0f} ms  (endpoint_silence_ms={settings.endpoint_silence_ms})")
    print(f"  segmenter.feed() per 20 ms frame: mean {statistics.mean(feed_us):.0f} us, max {max(feed_us):.0f} us")
    check(settings.endpoint_silence_ms - 50 <= lat * 1000 <= settings.endpoint_silence_ms + 250, "endpoint latency ~ endpoint_silence_ms + tail windows")
    RESULTS["realtime_endpoint"] = {"endpoint_latency_ms": round(lat * 1000, 1), "feed_mean_us": round(statistics.mean(feed_us), 1),
                                    "feed_max_us": round(max(feed_us), 1)}


async def section_barge_in(settings: PipelineSettings, device: int | None) -> None:
    print("\n[6d] Barge-in sim: Player plays a 3 s tone, ScriptedMic user speaks after 0.8 s -> stop() at barge_in_min_speech_ms")
    sr = 24000
    tone = (0.2 * np.sin(2 * math.pi * 330 * np.arange(3 * sr) / sr) * 32767).astype(np.int16)
    player = Player(sample_rate=sr, device=device)
    player.start()
    mic = ScriptedMic([(str(SAMPLES_DIR / "user_hello.wav"), 1.0)], leading_silence_s=0.8)
    seg = UtteranceSegmenter(settings)
    raised = min(0.9, settings.vad_threshold + 0.2) if settings.echo_guard else settings.vad_threshold
    seg.threshold = raised  # what the pipeline does while the agent talks (echo guard)
    player.mark()
    t0 = time.perf_counter()
    player.write(tone.tobytes())
    t_start: float | None = None
    t_stop: float | None = None
    played = 0
    async for frame in mic.frames():
        for ev in seg.feed(frame):
            if isinstance(ev, SpeechStart):
                t_start = ev.t - t0
        if t_stop is None and seg.speaking and seg.speaking_ms >= settings.barge_in_min_speech_ms:
            played = player.stop()
            t_stop = time.perf_counter() - t0
            seg.threshold = settings.vad_threshold
        if t_stop is not None and not seg.speaking:
            break
    await asyncio.sleep(0.05)
    player.close()
    assert t_start is not None and t_stop is not None
    print(f"  user speech begins at 0.8 s; SpeechStart at {t_start:.3f} s; stop() at {t_stop:.3f} s "
          f"(barge_in_min_speech_ms={settings.barge_in_min_speech_ms}); agent audio heard: {played} samples = {played/sr:.3f} s "
          f"(VAD threshold while agent speaks: {raised:.2f})")
    check(0.8 + settings.barge_in_min_speech_ms / 1000 - 0.05 <= t_stop <= 0.8 + settings.barge_in_min_speech_ms / 1000 + 0.35,
          "barge-in stop fired ~barge_in_min_speech_ms after the user starts talking")
    check(abs(played / sr - t_stop) < 0.08, "played-sample accounting matches wall clock at stop()")
    RESULTS["barge_in"] = {"speech_start_s": round(t_start, 3), "stop_s": round(t_stop, 3), "played_s": round(played / sr, 3)}


async def section_max_utterance(settings: PipelineSettings) -> None:
    print("\n[6e] max_utterance_s: 40 s of continuous speech (looped rough_day) must be split at 30 s")
    speech = load_pcm(SAMPLES_DIR / "user_rough_day.wav")
    # remove the clip's own trailing/leading quiet so the loop has no >550 ms gaps
    a, b = spoken_span(speech, 16000)
    core = speech[int(a * 16000) : int(b * 16000)]
    reps = int(math.ceil(40 * 16000 / len(core)))
    mic = FileMic(np.tile(core, reps), realtime=False, trailing_silence_s=1.0)
    seg = UtteranceSegmenter(settings)
    n = 0
    ends: list[tuple[float, float]] = []
    async for frame in mic.frames():
        n += 1
        for ev in seg.feed(frame):
            if isinstance(ev, SpeechEnd):
                ends.append((n * 0.02, ev.duration_s))
    print(f"  {n} frames; SpeechEnd events: " + ", ".join(f"t={t:.2f}s dur={d:.2f}s" for t, d in ends))
    check(len(ends) >= 2 and abs(ends[0][1] - settings.max_utterance_s) < 0.5,
          f"first utterance force-ended at ~{settings.max_utterance_s} s (got {ends[0][1] if ends else None})")
    RESULTS["max_utterance"] = {"ends": [(round(t, 2), round(d, 2)) for t, d in ends]}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-devices", action="store_true", help="skip the real Mic / Player sections")
    ap.add_argument("--mic-only", action="store_true", help="only run the Mic section (works with any python that has numpy+sounddevice)")
    ap.add_argument("--json", action="store_true", help="write bench/out/audio_results.json")
    ap.add_argument("--input-device", type=int, default=None)
    ap.add_argument("--output-device", type=int, default=None)
    args = ap.parse_args()
    settings = PipelineSettings()
    if args.mic_only:
        print(f"python: {getattr(sys, '_base_executable', sys.executable)}")
        await section_mic(args.input_device)
        print(f"\nfailures: {len(FAILURES)}")
        return 1 if FAILURES else 0
    print(f"settings: vad_threshold={settings.vad_threshold} endpoint_silence_ms={settings.endpoint_silence_ms} "
          f"min_speech_ms={settings.min_speech_ms} prespeech_buffer_ms={settings.prespeech_buffer_ms}")

    await section_segmenter(settings)
    if not args.no_devices:
        try:
            await section_player(args.output_device)
        except Exception as e:  # keep going; report
            print(f"  FAIL player section raised {e!r}")
            FAILURES.append(f"player: {e!r}")
        try:
            await section_mic(args.input_device)
        except Exception as e:
            print(f"  FAIL mic section raised {e!r}")
            FAILURES.append(f"mic: {e!r}")
    section_vad_speed()
    section_resampler()
    await section_scripted(settings)
    await section_realtime_endpoint(settings)
    if not args.no_devices:
        try:
            await section_barge_in(settings, args.output_device)
        except Exception as e:
            print(f"  FAIL barge-in section raised {e!r}")
            FAILURES.append(f"barge-in: {e!r}")
    await section_max_utterance(settings)

    print("\n==== summary ====")
    print(f"failures: {len(FAILURES)}")
    for f in FAILURES:
        print("  - " + f)
    RESULTS["failures"] = FAILURES
    if args.json:
        out_dir = ROOT / "bench" / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "audio_results.json").write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
        print(f"wrote {out_dir / 'audio_results.json'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.exit(asyncio.run(main()))
