# Eva – design notes

Goal: a personal voice agent that feels like Sesame's Maya (smooth, warm, emotionally
aware, interruptible) and can do small tasks. Not built for scale. Several "paths"
(presets) share one pipeline so components can be compared head to head.

## Layout

```
eva/
  interfaces.py      Protocols every provider implements (STT, LLM, TTS, Tool, TurnMetrics)
  config.py          keys, constants, PRESETS table, PipelineSettings
  factory.py         builds providers from a preset (constructor contract lives here)
  audio/mic.py       sounddevice input -> asyncio queue of 20 ms int16 frames @16 kHz
  audio/vad.py       Silero VAD (onnxruntime, no torch) + utterance segmenter
  audio/player.py    sounddevice output with instant stop() and played-sample accounting
  audio/envelope.py  lead-in / fade-in / fade-out / tail shaping of every spoken turn
  stt/…              elevenlabs_scribe.py, faster_whisper_local.py, sherpa_parakeet.py
  llm/openai_compat.py  streaming chat-completions client (Cerebras, Ollama, anything)
  llm/chunker.py     stream text -> TTS-sized sentence chunks (early first chunk)
  llm/sanitize.py    strip markdown / emoji / stage directions before TTS
  tts/elevenlabs.py  websocket stream-input + HTTP stream modes
  tts/kokoro_local.py
  personas/*.md      system prompts; personas.py loads + renders them
  tools.py           small tool registry (time, timer, notes, weather, open url/app)
  memory.py          tiny persistent facts store + end-of-session summariser
  pipeline.py        the conversation loop: VAD -> STT -> LLM -> chunker -> TTS -> player
  delivery.py        delivery cues ([warm]...), EN/RU detection, phantom-transcript and unfinished-turn heuristics
run.py               CLI entry: `python run.py --preset cloud-fast --persona eva`
bench/               latency.py, conversation_eval.py, e2e_sim.py
```

## Hard-won environment facts

* Windows 11, Python 3.13 venv at `.venv` on a uv-managed CPython (`uv python install 3.13`).
  The Microsoft Store Python cannot open the microphone (packaged app without the capability),
  so never build the venv on it. Run with `.venv/Scripts/python.exe`.
  Set `PYTHONIOENCODING=utf-8` when printing model output (cp1252 console).
* GPU: RTX 4050 laptop, 6 GB. Ollama already holds ~3 GB with qwen3:4b loaded.
  onnxruntime is CPU-only in the venv (fine for VAD, Kokoro, Parakeet int8).
* Ollama MUST be reached at `http://127.0.0.1:11434`, never `localhost` (+2 s per call).
* Cerebras MUST get a custom `User-Agent` header or Cloudflare returns 403 / error 1010.
* Cerebras models on this key: `gpt-oss-120b` (use `reasoning_effort: "low"`) and
  `qwen-3.8-27b` (use `disable_reasoning: true` for chat turns). Measured TTFT from here is
  0.75–2.2 s on a cold connection; keep a warm keep-alive client and pre-connect at startup.
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
  ring and a 150 ms tail). The pipeline sets `turn.cancelled` before `stop()`, cancels and
  awaits the response task, then calls `stop()` again: a writer/filler wake-up that was
  already scheduled in the same loop iteration cannot leave audio behind.

## Latency budget (target ≤ 900 ms speech-end -> first agent audio on cloud-fast)

| stage | budget |
|---|---|
| endpoint silence | 550 ms (this is the floor; smarter turn detection could cut it) |
| STT (Scribe batch on utterance) | 300–500 ms |
| LLM TTFT (Cerebras qwen, warm) | 300–700 ms |
| TTS first audio (EL Flash, ws) | 150–300 ms |

Status (measured, README "Measured latency"): not met. The best cloud-fast runs are 1.01 s median
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
