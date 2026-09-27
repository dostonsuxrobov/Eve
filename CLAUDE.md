# Eva — project brief for Claude

Read this first in every session.

## What this is

Eva is a **realistic voice agent**: you talk, she listens while you speak, answers about a
second later, can be interrupted, remembers you, does small things. First a personal companion
(friend register, English), later jobs (customer support, truck dispatching) as personas and tool
sets on the same loop. It is a **serious, long-lived project**, not a demo. Two goals, in this
order: **feel real first, then be useful**. Nothing that makes her more useful may make her feel
less real.

The user is Doston (developer; native Russian speaker, English too). He owns every call on how
she *sounds and feels*. Measurements own what is *true*: latency, VRAM, credits, bugs.

## The setup (owner's decision, 2026-09-26): local brain and ears, ElevenLabs voice

* **Brain: Qwen3 4B instruct on Ollama, on this laptop** (`qwen3:4b-instruct-2507-q4_K_M`, 3.2 GB
  of the 6 GB GPU, 56 tok/s, first word 0.35-0.4 s once the persona is cached). It gets the
  small-brain scaffolding: `eva/toolgate.py` (only the tools the user's words point at; plain
  weather/time questions answered by the loop itself; a wrong city corrected from memory) and
  `eva/guard.py` (every sentence checked before it is spoken). **Ears: Parakeet** (sherpa-onnx, CPU).
* **Voice: ElevenLabs**, the owner's voice `bD9maNcCuQQS75DGuteM`, models `v3` (default),
  `v3conv` (v3 Conversational) and `flash` (Flash v2.5); `kokoro` is the local voice and the
  automatic fallback (`eva/failover.py`) when ElevenLabs can't answer (no internet, no credits).
  There is no "Flash v3".
* **Why:** the fully local build proved the voice was the bottleneck: Chatterbox took 2.3-3.4 s to
  its first sound on this GPU, while the brain costs 0.1-0.4 s at any size
  (`.archive/local-variants/docs/MEASUREMENTS.md`). With ElevenLabs the whole turn is 1.9-2.7 s
  after the endpoint (docs/MEASUREMENTS.md).
* **The plan:** Creator, billed annually, **paid through 2027-07-28**: 121k credits a month (the
  month turns on the 28th), unused credits roll over two months. v3 costs about 0.55 credits per
  character, v3 Conversational and Flash about half (the `character-cost` response header).
  At the rate measured on the first test line that is about 5-7 hours of conversation a month on
  v3 and 10-14 on the others (33-45k characters of her speech an hour); the meter will tell. `eva/credits.py` counts what
  every response reports into `usage.json`; the console shows it per turn and warns at 80 %.
* **English only** (since 2026-09-23); the Russian persona and data are in the first archive.

## The dispatch MVP (owner's direction, 2026-09-27)

The owner wants a useful product in front of real users before any GPU spend: build the job layer
(data, tools, behaviour, evals) on pay-per-use APIs, compare backends on quality first, cost later.
The first job is **a truck dispatcher** for a fictional carrier, Red Oak Transport: book loads,
negotiate rates, give updates.

* **The job is data + tools on the same loop** (`eva/jobs/dispatch/`): `world.py` builds a fake SQLite
  world (116 cities, 131 trucks, 140 drivers, 220 brokers, ~6k load-board postings, ~24k past loads,
  ~72k GPS pings, lane rates, fuel, policy) with planted scenario rows; `desk.py` is 12 tools that
  enforce the rules whatever the model says (cost floor, do-not-use brokers, hazmat endorsement,
  trailer type, pickup window). Persona `eva/assets/personas/en/dispatcher.md`. A job session never
  reads or writes the owner's personal memory.
* **Backends, same persona and tools:** Eva's loop (`run.py --job dispatch --brain qwen27b --voice v3conv`:
  Parakeet + Cerebras Qwen3.8-27B + ElevenLabs) and speech-to-speech (`run_s2s.py openai:gpt-realtime-2.1`,
  `openai:gpt-realtime-2.1-mini`, `gemini:gemini-3.8-live`; `eva/s2s/`, raw WebSockets, prices in `eva/s2s/__init__.py`).
  The owner allows closed models for these tests; the open-brain rule stands for Eva's own loop.
* **Measure:** `bench/dispatch_eval.py <backend>` (six scripted calls, fresh world each, fact checks;
  Cerebras ~50 completions a run, Gemini ~$0.25) and `bench/s2s_audio_check.py <backend>` (a spoken
  caller). Results in docs/MEASUREMENTS.md.
* **Keys** (gitignored, first key-looking line of the file): `cerebras_api_key.txt`, `google_api_key.txt`,
  `openai_key.txt`, `elevenlabs_key.txt`.
* **RunPod pilot scaffolding** (`deploy/runpod/`, `voice/server.py`, `qwen27b-pod`): written and tested
  offline, never run on a pod. The pod was stopped and the owner moved to API-first.

## Credits are money

* Anything that makes her speak through ElevenLabs spends credits: `run.py`, `bench/e2e_sim.py`
  (a few hundred per run). Debug on `--voice kokoro` or with `bench/replay.py` (silent voice, the
  real brain: free). Never run matrices of voices or long loops against ElevenLabs.
* She renders one sentence ahead (`tts_parallelism=1`), so an interruption wastes at most one
  unheard sentence. Fillers are rendered once and cached on disk.
* The key has only text_to_speech / speech_to_text (no models_read, voices_read, user_read): the
  balance can't be read from the API. Scribe (their STT) costs 330 credits a minute: the ears stay local.

## The archives

* `.archive/` — the cloud era up to 2026-09-25 (commit `d982caa`: Scribe + Cerebras 27B + ElevenLabs
  v3, failover, phone client). Brief `.archive/CLAUDE.old.md`; numbers `.archive/docs/MEASUREMENTS.md`,
  `.archive/docs/EVAL_REPORT.md`.
* `.archive/local-variants/` — 2026-09-25/26, fully local: seven small brains x Kokoro / Orpheus /
  Chatterbox (a PyTorch voice server in `.venv-voice`), the brain bench, the cloud 27B as a
  comparison. Brief `CLAUDE.local-variants.md`; numbers `docs/MEASUREMENTS.md` there.
* **Quarries, not dependencies:** bring a piece back on purpose with its tests; never import from them.
  What came back on 2026-09-26: the loop, guard, tool gate, memory, phone client and tests from
  local-variants; the ElevenLabs voice and `failover.py` from the cloud era.

## How to work here

**Measure before changing, and measure after.** Latency, VRAM, credits, RMS envelopes, tool-call
rates: find the signal-level evidence first. Record new numbers, with the date, in
`docs/MEASUREMENTS.md`. Before calling anything slower a regression, check the power state (on
battery Windows caps this GPU at 50 W; `run.py` warns).

**One pipeline, data for the differences.** Personas, voices, fillers and tool sets are data;
the loop is shared. New jobs are new personas + tool sets + scenarios, not new agents.

**Tests are the memory of bugs.** Every bug that reached the user gets a test that fails on the
old code. Run `pytest tests -q` before every commit (about 3.5 min).

**The user reads the console.** Every silent decision the pipeline takes (an ignored transcript,
a barge-in refused as echo, a sentence the guard dropped, a question the loop looked up itself,
a voice failover, the credits a turn cost) prints a dim line saying why.

**Commit when a step is verified**, with a message that says what was measured; push to
`origin/main`. Commit trailers as the session reminder specifies.

## Gotchas that cost hours

* Ollama at `127.0.0.1`, never `localhost` (+2 s per request). Hybrid Qwen3 models only stop
  thinking through the native `/api/chat` `think` field. Ollama keeps a model loaded for 30 min.
* ElevenLabs: `eleven_v3` and v3 Conversational take only the stability presets 0.0 / 0.5 / 1.0,
  no `optimize_streaming_latency` and no `previous_text` (HTTP 400); Flash takes all three.
* The laptop mic array has hardware echo cancelling: recording her own output through it is
  useless for measurement. Measure at the bytes written to the player.
* Windows Store Python cannot open the mic; use `.venv` (uv CPython 3.13). The console is
  cp1252: set `PYTHONIOENCODING=utf-8`. No ffmpeg on PATH; use soundfile/numpy.
* The Claude Code Bash tool mangles backslashes and some quoting inside heredocs: write scripts
  to the scratchpad with the Write tool and run them. Foreground `sleep` does not wait.
* Files that came from the archive may have CRLF endings (`eva/assets/personas/en/eva.md`):
  rewriting them with `\n` makes git show every line changed. Keep the file's own endings.
* The phone client (`run.py --web --tls`): browsers allow the mic only on https (self-signed
  certificate via Git's openssl); its AudioWorklets are template strings, so no backticks inside.

## Open with the owner

* Her memory holds facts that aren't true (speaks Dutch, Portuguese, Ukrainian: likely misheard
  languages from the cloud era; a sister named Priya: the eval's stand-in user). The owner decides
  which facts are true before they are removed.

## Next (the owner orders it)

0. **Dispatch MVP:** the owner talks to each backend and judges by ear; OpenAI runs once
   `openai_key.txt` exists; Eva's 27B talks too long (median 63-74 words a reply against Gemini's 25);
   then cost per call, a phone line, and a real design partner.
1. **The voice by ear:** v3 vs v3 Conversational vs Flash on the new voice, in real sessions.
2. **Latency:** end-of-turn prediction (SmartTurn) instead of the fixed 0.5 s wait; a streaming or
   faster STT for long utterances (Parakeet took 1.1 s on an 8 s one).
3. **Feel:** a local measure that agrees with the owner's ear; the 4B's behaviours as tests;
   fine-tuning the 4B on conversations he approves.
4. **Memory over days**, then useful tools, then the first job persona.
5. After 2027-07-28: renew, or move the voice to a rented GPU / a home GPU box (costed in the
   local-variants MEASUREMENTS).
