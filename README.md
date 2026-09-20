# Eva

Eva is a personal voice agent in the spirit of Sesame's Maya: you talk, she listens while you
speak, answers about a second after you stop, can be interrupted mid-sentence, hums "mm, hang on"
when a reply is slow, switches between English and Russian per sentence, and can do a few small
things (timers, notes, weather, open a URL, end the call). One asyncio pipeline
(`VAD -> STT -> LLM -> sentence chunker -> TTS -> player`) behind swappable providers, with a
local stack that takes over when a cloud service stops answering. Built and measured on one
Windows 11 laptop (RTX 4050 6 GB, Python 3.13). Not built for scale.

* `docs/DESIGN.md` — architecture, module map, hard-won environment facts.
* `docs/MEASUREMENTS.md` — every latency, loudness, cost and outage number, and the known issues.
* `docs/EVAL_REPORT.md` — the conversation-quality evaluation that picked the brain and the persona.

## Run

```
.venv\Scripts\python.exe run.py --user-name Doston             # preset maya: Cerebras qwen-3.8-27b brain, auto language
.venv\Scripts\python.exe run.py --preset maya-gpt               # same ears and voice, OpenAI gpt-5.4-mini brain
.venv\Scripts\python.exe run.py --preset maya-local             # same ears and voice, Ollama qwen3:8b brain (free)
.venv\Scripts\python.exe run.py --preset local                 # everything on this laptop, offline
.venv\Scripts\python.exe run.py --user-name Doston --lang ru   # locked to Russian
.venv\Scripts\python.exe run.py --text                         # type instead of talk (she still speaks)
.venv\Scripts\python.exe run.py --once "hey, how's it going"   # one typed turn, then exit
.venv\Scripts\python.exe run.py --list-devices
```

Ctrl-C ends the session (or say goodbye: she ends it herself). Headphones are recommended:
through speakers the echo guard makes her harder to interrupt, and the first seconds of a session
are when the mic's echo canceller is still converging. `--no-greeting`, `--mute-fillers`,
`--no-barge-in`, `--no-fallback` (don't load the local backups), `--persona calm_coach`,
`--voice`, `--input-device` / `--output-device`, `--debug`.

## The presets: three brains on one stack, plus offline

| preset | STT | brain | TTS | eval | first token | brain cost per exchange |
|---|---|---|---|---|---|---|
| `maya` (default) | ElevenLabs Scribe v2 realtime | Cerebras `qwen-3.8-27b`, reasoning low | ElevenLabs v3 + Flash first chunk, `eva_en` / `eva_ru` | 6.2/10, tools 6/6 | 0.30 s | ~$0.003 ($0.99 / $1.49 per M in / out) |
| `maya-gpt` | same | OpenAI `gpt-5.4-mini`, reasoning off | same | 8.0/10, tools 5/6 | 0.49 s | ~$0.002 uncached, ~$0.0005 once the prompt prefix is cached ($0.75 / $4.50 per M, cached input $0.075) |
| `maya-local` | same | Ollama `qwen3:8b`, thinking off (27 % on CPU) | same | 3.7/10, tools 6/6 | 0.18 s (1-3 s on a cold prompt or a tool call) | electricity |
| `local` | Parakeet TDT 0.6B int8 (sherpa-onnx) | Ollama `qwen3:8b` | Kokoro (ONNX, CPU) | 3.7/10 | 0.18 s | nothing, offline |

An exchange is 2,800-4,300 prompt tokens (persona + tools + memory + up to 30 history
messages) and 40-150 completion tokens, at the sample cadence of 12-15 s per exchange (about
270 an hour): roughly **$0.95 an hour** on `maya`, **$0.15-0.60 an hour** on `maya-gpt`
(caching decides), **$0** on the local brains. The brain is not the big line: ElevenLabs
(Scribe per minute of audio, v3/Flash per character) costs more per hour than any of them.
Prices from the providers' pages on 2026-09-19 (Cerebras: $0.99/$1.49 per M tokens; OpenAI:
$0.75/$4.50, cached input $0.075); check them before relying on the numbers.

The three `maya-*` presets carry the `local` providers as **fallbacks**: when a cloud request
cannot start or gives no first result in time, that turn is served locally, the cloud provider is
marked down for 20 s and probed again after (`eva/failover.py`; the console says so). The
backups are warmed in the background after the cloud providers, so startup is not delayed.

**Brains** (`--brain`, `eva/config.py: BRAINS`), the three that won their metric in
`docs/EVAL_REPORT.md`: `qwen` (Cerebras qwen-3.8-27b: 6.2/10, 6/6 on tool calls, 0.30 s to
first token; the default), `gpt` (OpenAI gpt-5.4-mini with reasoning off:
8.0/10, the tersest and most Maya-like replies, 0.49 s, costs money per turn), and `local`
(Ollama qwen3:8b with thinking off: 3.7/10 as a companion but 6/6 on tool calls; the offline
preset and the fallback). gpt-oss-120b, qwen3:4b, qwen2.5-coder:7b, gpt-5.4-nano and
gpt-5.6-luna/terra were measured and dropped (report sections 3 and 10).

## Languages

`--lang auto` (default) follows you: Scribe auto-detects, the voice switches per sentence by
script, fillers follow your last turn, and the persona is the English prompt with a bilingual
rule. `--lang en` / `--lang ru` lock the session: Scribe gets a language hint (more reliable on
one-word answers), the voice is pinned, and the persona is `eva/assets/personas/<lang>/eva.md`
when it exists (a prompt written in the language beats an English prompt with a "speak natural
Russian" rule). Everything language-specific is data:

```
eva/assets/lang/en.toml, ru.toml        voice, STT hint, fillers, tool hints, backchannels
eva/assets/personas/en/*.md, ru/*.md    system prompts (front matter may override the fillers)
bench/scenarios.json                    eval scenarios, "lang": "en" | "ru"
```

`eva/assets/personas/ru/eva.md` is a first draft; the register is yours to own.

## What makes it feel like a person

* **Turn taking**: 500 ms endpoint, a 600 ms grace for a transcript that trails off ("...and",
  "потому что,") which is then merged with what follows; bare hesitations ("uh", "hmm") are
  waited on, never answered; a transcript that is a run of what she just said (the mic hearing
  her through the speakers) is dropped; the phantom-transcript filter built for Whisper is applied
  only to Whisper-class STTs, so "yeah", "no", "bye" are real answers on Scribe.
* **Barge-in** on 300 ms of confirmed speech: the player stops within a block, only the words you
  actually heard stay in the history, marked `[interrupted]`.
* **Fillers and backchannels**: "mm, hang on" if nothing is audible 800 ms after you stop;
  "mm-hm" at a breath inside a long story (headphones).
* **Delivery**: the brain starts about every other sentence with a cue (`[warm] [soft] [teasing]`
  ...); v3 renders it, Flash maps it to voice settings; consecutive sentences keep one prosodic
  line; every reply has a lead-in, fades and a tail (`eva/audio/envelope.py`); and every clip is
  loudness-leveled per model x voice (`eva/audio/leveler.py`), because Flash, v3 and the Russian
  voice differ by up to 9 dB otherwise.
* **Endings**: say you have to go and she says goodbye and calls `end_conversation` in the same
  reply (a final tool: no second goodbye).
* **Memory**: durable facts about you are extracted at the end of each session into
  `memory.json` (the summariser is told who you and Eva are, so a misheard name is never stored).

## Setup

```
uv python install 3.13
uv venv .venv --python cpython-3.13.15-windows-x86_64-none
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

Do not build the venv on the Microsoft Store Python (`AppData\Local\Microsoft\WindowsApps`):
Windows denies audio capture to that packaged app. Keys: `cerebras_api_key.txt`,
`elevenlabs_key.txt` and, for `--brain gpt`, `openAI_api.txt` in the project root (or
`CEREBRAS_API_KEY` / `ELEVENLABS_API_KEY` / `OPENAI_API_KEY`); all gitignored.
Ollama must be running with `qwen3:8b` pulled for the `local` preset and
the fallback brain. Models for Parakeet, Kokoro and the Silero VAD live under `models/`.

## Tests and benches

```
.venv\Scripts\python.exe -m pytest tests -q                       # offline: pipeline scenarios on doubles, failover, units
.venv\Scripts\python.exe bench\e2e_sim.py --utterances samples/user_hello.wav,samples/user_ru_rough_day.wav --speakers
.venv\Scripts\python.exe bench\e2e_sim.py --outage llm,stt,tts   # the local stack takes over
.venv\Scripts\python.exe bench\conversation_eval.py --brain qwen --persona eva --lang ru
.venv\Scripts\python.exe bench\summarize_e2e.py
```

## Layout

```
eva/
  config.py        keys, PipelineSettings, BRAINS, PRESETS (maya, local)
  session.py       build_session(): providers + language plan + persona + memory, shared by run.py and the sim
  factory.py       build_stt / build_llm / build_tts / build_stack (wraps fallbacks)
  failover.py      FailoverLLM / FailoverSTT / FailoverTTS
  lang.py          language assets and the auto / en / ru plans
  personas.py      persona files and prompt rendering
  pipeline.py      the conversation loop and state machine
  delivery.py      delivery cues, language detection, phantom / hesitation / echo gates
  memory.py        facts store + end-of-session summariser
  tools.py         time, timer, notes, weather, open url, end_conversation
  audio/           mic, vad, player, envelope, leveler
  stt/ llm/ tts/   providers behind interfaces.py
  assets/          lang/*.toml, personas/<lang>/*.md
  mocks.py         doubles and the test harness helpers
tests/             pytest (offline)
bench/             e2e_sim.py, conversation_eval.py + scenarios.json, summarize_e2e.py
docs/              DESIGN.md, MEASUREMENTS.md, EVAL_REPORT.md
samples/           sample utterances for the simulator
run.py             CLI
```
