# Eva – design notes

Goal: a personal voice agent that feels like Sesame's Maya (smooth, warm, emotionally
aware, interruptible) and can do small tasks. Not built for scale. One pipeline, two
stacks (`maya` in the cloud, `local` on the laptop), and the cloud stack falls back to the
local providers when a cloud service stops answering. Conversation quality comes first;
tool use is built on the same loop and grows from there.

## Layout

```
eva/
  interfaces.py      Protocols every provider implements (STT, LLM, TTS, Tool, TurnMetrics)
  config.py          keys, constants, PipelineSettings, BRAINS, PRESETS (maya, local)
  factory.py         builds providers from configs; build_stack() wraps the fallbacks
  failover.py        FailoverLLM / FailoverSTT / FailoverTTS: cloud -> local within one request
  session.py         build_session(): providers + language plan + persona + memory (run.py and the sim)
  lang.py            assets/lang/*.toml and the auto / en / ru plans
  personas.py        assets/personas/<lang>/*.md loading + prompt rendering
  audio/mic.py       sounddevice input -> asyncio queue of 20 ms int16 frames @16 kHz
  audio/vad.py       Silero VAD (onnxruntime, no torch) + utterance segmenter
  audio/player.py    sounddevice output with instant stop() and played-sample accounting
  audio/envelope.py  lead-in / fade-in / fade-out / tail shaping of every spoken turn
  audio/leveler.py   loudness leveling per TTS source (model x voice)
  audio/echo.py      is the mic hearing the speakers? (cross-correlation with what the player played)
  stt/               elevenlabs_realtime.py (+ elevenlabs_scribe.py batch), sherpa_parakeet.py
  llm/openai_compat.py  streaming chat-completions client (Cerebras, Ollama, anything)
  llm/chunker.py     stream text -> TTS-sized sentence chunks (early first chunk, JSON kept whole)
  llm/sanitize.py    strip markdown / emoji / stage directions before TTS
  tts/elevenlabs.py  HTTP stream, first-chunk model, per-language voices, continuity, leveling
  tts/kokoro_local.py
  assets/lang/       en.toml, ru.toml: voice, STT hint, fillers, tool hints, backchannels
  assets/personas/   en/eva.md, en/calm_coach.md, ru/eva.md
  tools.py           small tool registry (time, timer, notes, weather, open url, end_conversation)
  memory.py          tiny persistent facts store + end-of-session summariser
  pipeline.py        the conversation loop: VAD -> STT -> LLM -> chunker -> TTS -> player
  delivery.py        delivery cues ([warm]...), EN/RU detection, phantom / hesitation / echo gates
  mocks.py           doubles + the harness helpers used by tests/ and bench/
  web/               the phone client: server.py (page + WebSocket, optional TLS), transport.py
                     (WebMic / WebPlayer behind the pipeline's frame and PlayerLike contracts), static/index.html
run.py               CLI entry: `python run.py --user-name Doston [--lang ru] [--preset local]`
tests/               pytest, offline (pipeline scenarios on doubles, failover, units)
bench/               e2e_sim.py (real providers, --outage), conversation_eval.py + scenarios.json, summarize_e2e.py
docs/                DESIGN.md (this), MEASUREMENTS.md (numbers, known issues), EVAL_REPORT.md
```

## Failover

Each cloud provider is wrapped with its local counterpart (`eva/failover.py`). A request
that raises before its first result, or gives none within the first-result timeout (LLM
5 s, STT 6 s, TTS 4 s), is served by the backup and the primary is marked down for 20 s;
the next request after the cooldown (or the LLM keep-alive ping) probes it. Failures
*after* the first token/byte are never retried across providers: the pipeline already
speaks what it got. Backups warm in the background after the primaries so startup and
the greeting are not delayed. `--no-fallback` skips them (saves ~3 GB of VRAM for Ollama
and a few seconds of CPU model loading). Sample rates must match across a TTS pair
(both 24 kHz here).

## Languages

`eva/lang.py` resolves `--lang` into a plan: `auto` loads every language's assets, lets
Scribe auto-detect and switches the voice per sentence by script; `en` / `ru` lock the
session (STT hint, pinned voice, that language's fillers and persona file). The code is
script-agnostic; everything that differs is data under `eva/assets/`.

## Hard-won environment facts

* Windows 11, Python 3.13 venv at `.venv` on a uv-managed CPython (`uv python install 3.13`).
  The Microsoft Store Python cannot open the microphone (packaged app without the capability),
  so never build the venv on it. Run with `.venv/Scripts/python.exe`.
  Set `PYTHONIOENCODING=utf-8` when printing model output (cp1252 console).
* GPU: RTX 4050 laptop, 6 GB. Ollama holds ~5.7 GB with qwen3:8b loaded (27 % of it on the CPU).
  onnxruntime is CPU-only in the venv (fine for VAD, Kokoro, Parakeet int8).
* Ollama MUST be reached at `http://127.0.0.1:11434`, never `localhost` (+2 s per call).
* Cerebras MUST get a custom `User-Agent` header or Cloudflare returns 403 / error 1010.
* Cerebras models on this key: `qwen-3.8-27b` (the default brain, `reasoning_effort: "low"`;
  `disable_reasoning` truncates short replies, see docs/EVAL_REPORT.md) and `gpt-oss-120b`
  (`reasoning_effort: "low"`; leaks tool calls as JSON text now and then, the chunker and
  `_recover_tool_calls` handle it). Measured TTFT from here is 0.75–2.2 s on a cold connection;
  keep a warm keep-alive client and pre-connect at startup.
* The laptop mic array has hardware echo cancellation: what it records of Eva's own speaker
  output is useless for measurements; measure at the bytes written to `Player.write` instead.
* ElevenLabs key: TTS and Scribe STT work; `voices_read` / `user_read` are NOT granted.
  Voice IDs are hardcoded in `config.EL_VOICES`. Rachel (`21m00Tcm4TlvDq8ikWAM`) verified.
* No ffmpeg on PATH. Use `soundfile` / numpy for wav IO; ask ElevenLabs for `pcm_*` output.

## Audio contract

* Mic frames: int16 mono 16 kHz, 320 samples (20 ms) per frame.
* VAD: Silero v5 ONNX consumes 512-sample (32 ms) windows; the segmenter re-buffers.
* TTS yields raw int16 PCM at `tts.sample_rate`; the player is opened at that rate.
* Barge-in: player.stop() must return within one output block (~20–40 ms). The pipeline
  records how many samples were actually played and truncates the assistant transcript
  proportionally, appending " [interrupted]" so the LLM knows what the user heard.
  The confirmation counts speech-positive VAD windows (`segmenter.speaking_ms` /
  `SpeechEnd.speech_ms`), never the utterance length (which includes the 300 ms pre-speech
  ring and a 150 ms tail). While she is audible that is not enough (speaker echo passes the
  VAD): `PipelineSettings.barge_in_confirm = "words"` also needs the streaming STT's partial
  transcript to carry real words that are not a fuzzy match of her current reply
  (`delivery.echo_similarity` >= 0.6 is echo) and the echo detector to say the mic is not
  hearing the speakers; an onset that ends unproven is decided on its final transcript
  (`_late_check`). Three echo classifications in 20 s open a 30 s "storm" in which only the
  final transcript can interrupt. The pipeline sets `turn.cancelled` before `stop()`, cancels and
  awaits the response task, then calls `stop()` again: a writer/filler wake-up that was
  already scheduled in the same loop iteration cannot leave audio behind.

## Latency budget (target ≤ 900 ms speech-end -> first agent audio on the cloud stack)

| stage | budget |
|---|---|
| endpoint silence | 550 ms (this is the floor; smarter turn detection could cut it) |
| STT (Scribe batch on utterance) | 300–500 ms |
| LLM TTFT (Cerebras qwen, warm) | 300–700 ms |
| TTS first audio (EL Flash, http) | 150–300 ms |

Status (measured, docs/MEASUREMENTS.md): not met. The best cloud runs (selection phase, `cloud-fast` = today's `maya` before v3) are 1.01 s median
from the endpoint (STT 0.26 + TTFT 0.54 + TTFA 0.16 + playback), and the endpoint itself fires
0.62 s after the last word (18 VAD windows of 32 ms), so last word -> first audio is 1.65 s median,
about 0.7 s over budget. Only smarter turn detection (cutting the 550 ms) or a lower TTFT can close
that; STT and TTS are already inside their lines when ElevenLabs is in its fast state.

Fillers ("hm", "let me think") are pre-synthesized at startup in the active voice and
played only if no real audio has started `filler_after_ms` after speech end.

## Persona rules that matter for TTS realism

* Spoken register: short sentences, contractions, no lists, no markdown, no emoji.
* Emotional attunement first, information second. Mirror energy, name feelings lightly.
* Backchannel words allowed ("mm", "yeah", "right"); ellipses for pauses.
* If `supports_audio_tags` is true (ElevenLabs v3) the model may use [laughs], [sighs],
  [whispers], [exhales] sparingly; otherwise sanitizer strips any bracketed tags.
* Never claim a task is done unless a tool result says so.
