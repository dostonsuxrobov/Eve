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
.venv\Scripts\python.exe run.py --user-name Doston             # preset maya, auto language
.venv\Scripts\python.exe run.py --preset local                 # everything on this laptop, offline
.venv\Scripts\python.exe run.py --user-name Doston --lang ru   # locked to Russian
.venv\Scripts\python.exe run.py --web --tls --user-name Doston  # talk from your phone (see below)
.venv\Scripts\python.exe run.py --text                         # type instead of talk (she still speaks)
.venv\Scripts\python.exe run.py --once "hey, how's it going"   # one typed turn, then exit
.venv\Scripts\python.exe run.py --list-devices
```

Ctrl-C ends the session (or say goodbye: she ends it herself). Headphones are recommended:
through speakers the echo guard makes her harder to interrupt, and the first seconds of a session
are when the mic's echo canceller is still converging. `--no-greeting`, `--mute-fillers`,
`--no-barge-in`, `--no-fallback` (don't load the local backups), `--persona calm_coach`,
`--voice`, `--input-device` / `--output-device`, `--debug`.

## The stack

| preset | STT | brain | TTS |
|---|---|---|---|
| `maya` (default) | ElevenLabs Scribe v2 realtime (streams while you talk) | Cerebras `qwen-3.8-27b`, reasoning low | ElevenLabs v3 for every sentence, one delivery cue per reply; `eva_en` / `eva_ru` voices |
| `local` | Parakeet TDT 0.6B int8 (sherpa-onnx) | Ollama `qwen3:8b`, thinking off | Kokoro (ONNX, CPU) |

The brain is qwen-3.8-27b on Cerebras: 6.2/10 in the eval, 6/6 on tool calls, 0.30 s to the
first token, about $0.003 per exchange ($0.99 / $1.49 per M tokens in / out, 2026-09-19). An
exchange is 2,800-4,300 prompt tokens and 40-150 output tokens, so an hour of talking (about
270 exchanges) is roughly $0.95 of brain; ElevenLabs (Scribe per minute of audio, v3 / Flash per
character) costs more per hour than that. Every other brain measured, cloud and local, is in
`docs/EVAL_REPORT.md` (sections 3 and 10) and in git history; the live comparison on
2026-09-19 settled it ("much more natural by much larger margins").

`maya` carries the `local` providers as **fallbacks**: when a cloud request cannot start or
gives no first result in time, that turn is served locally, the cloud provider is marked down for
20 s and probed again after (`eva/failover.py`; the console says so). The backups are warmed in
the background after the cloud providers, so startup is not delayed.

## From your phone

```
.venv\Scripts\python.exe run.py --web --tls --user-name Doston
```

prints `open on your phone: https://192.168.0.154:8443` (your laptop's Wi-Fi address). The page
(`eva/web/static/index.html`) captures the phone's mic, streams it to the laptop over a
WebSocket, and plays her voice back; the brain, Scribe and ElevenLabs all still run on the
laptop, so it is the same Eva. Tap **Start** (the browser asks for the mic), talk; **Stop her**
interrupts by touch, **End** hangs up; she also hangs up when you say goodbye. The phone's own
echo canceller keeps her voice out of the mic (measured on an iPhone: none reached the STT), so
the laptop's echo defences are off for the web client: barge-in is the plain 300 ms VAD rule
and nothing you say is second-guessed as her echo (`--echo-gates on` forces them back).

Browsers allow the microphone only on `https://`, so `--tls` makes a self-signed certificate
for your laptop's address (once, with the openssl that ships with Git for Windows, under
`models/web/`). Accepting it:

* **iPhone / Safari**: open the URL, "Show Details" -> "visit this website", then Start.
* **Android / Chrome**: either download `https://<laptop-ip>:8443/cert.pem` and install it
  (Settings -> Security -> Encryption & credentials -> Install a certificate -> CA certificate),
  or open `chrome://flags/#unsafely-treat-insecure-origin-as-secure`, add
  `http://<laptop-ip>:8080`, and run without `--tls` on port 8080.
* **Anywhere, no certificate fuss** (also over LTE): a tunnel with a real certificate, e.g.
  `winget install Cloudflare.cloudflared` then `cloudflared tunnel --url http://localhost:8080`
  while `run.py --web` runs; open the `https://...trycloudflare.com` URL it prints. Adds the
  round trip to Cloudflare's edge (tens of milliseconds).

Both phone and laptop must be on the same Wi-Fi for the direct URL; Windows may ask once to
allow Python through the firewall on a private network. One phone at a time.

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
* **Barge-in** on 300 ms of confirmed speech while she is thinking; while she is *audible*,
  speech alone is not enough, because through speakers her own voice reaches the mic (a
  session looped twelve turns on that: she cut herself off, transcribed the garbled echo,
  answered it). Now the streaming transcript must show your words, they must not be a fuzzy
  copy of what she is saying, and the echo detector (`eva/audio/echo.py`, which correlates the
  mic with what the player just played) must not hear the speakers. A short interjection the
  partials missed is checked on its final transcript and interrupts her a few hundred
  milliseconds late. The player stops within a block; only the words you actually heard stay in
  the history, marked `[interrupted]`. Three echo classifications in twenty seconds make her
  ask for headphones and demand a full transcript before any interruption for thirty seconds.
* **Fillers and backchannels**: "mm, hang on" if nothing is audible 800 ms after you stop;
  "mm-hm" at a breath inside a long story (headphones).
* **One voice per reply**: the brain opens a reply with one cue (`[warm]`, `[teasing]`...) that
  sets the mood of the whole reply and keeps it across replies unless the mood shifts; every
  chunk inherits it (a chunk without a cue used to fall back to the default style mid-reply).
  All sentences go through v3: the Flash first chunk was 0.4 s faster to start but put a
  different timbre, pace and noise floor on the first sentence of every reply ("rushed start,
  then it settles"). Loudness is leveled per model x voice (`eva/audio/leveler.py`).
* **Edges and background**: v3 clips are trimmed hot (first 10 ms at -39 dBFS, last 10 ms at
  -31), so every reply gets a 120 ms fade in and a 280 ms fade out, every chunk boundary short
  edge fades, and the lead-in, the tail and idle time carry a faint room tone at v3's own
  in-speech floor (-62 dBFS, `PipelineSettings.room_tone_dbfs`) instead of digital silence, so
  the background never switches on with her first word and off after her last
  (`eva/audio/envelope.py`).
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
Windows denies audio capture to that packaged app. Keys: `cerebras_api_key.txt` and
`elevenlabs_key.txt` in the project root (or `CEREBRAS_API_KEY` / `ELEVENLABS_API_KEY`); gitignored.
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
  audio/           mic, vad, player, envelope, leveler, echo
  web/             the phone client: server.py (page + socket), transport.py (WebMic, WebPlayer), static/index.html
  stt/ llm/ tts/   providers behind interfaces.py
  assets/          lang/*.toml, personas/<lang>/*.md
  mocks.py         doubles and the test harness helpers
tests/             pytest (offline)
bench/             e2e_sim.py, conversation_eval.py + scenarios.json, summarize_e2e.py
docs/              DESIGN.md, MEASUREMENTS.md, EVAL_REPORT.md
samples/           sample utterances for the simulator
run.py             CLI
```
