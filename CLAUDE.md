# Eva — project brief for Claude

Read this first in every session. Details live in `README.md` (how to run), `docs/DESIGN.md`
(architecture, environment facts), `docs/MEASUREMENTS.md` (every number, the known issues and
the bug hunts), `docs/EVAL_REPORT.md` (why the brain and the persona were chosen).

## What this is and where it is going

Eva is a **realistic voice agent**: you talk, she listens while you speak, answers about a
second later, can be interrupted, remembers you, does small things. Today it is an early
prototype of a *personal companion* (English and Russian, friend register) that one person
uses daily from a laptop or a phone. It is a **serious, long-lived project**, not a demo: the
same agent is meant to grow into work — **customer support, truck dispatching, basic friend
conversation** — through different personas, tool sets and deployments on one pipeline. Two
goals in that order: **feel real first, then be useful**. Nothing that makes her more useful
may make her feel less real.

The user is Doston (native Russian speaker, English too; developer). Treat him as the owner
of every judgement call about how she should *sound*; treat measurements as the owner of
every judgement about what is *true*.

## Current state (2026-09-20)

* Stack `maya`: ElevenLabs Scribe v2 realtime (STT) → Cerebras `qwen-3.8-27b` (brain, reasoning
  low) → ElevenLabs v3 (voice, one delivery cue per reply, `eva_en` / `eva_ru` voices).
  Falls back per provider to the local stack (Parakeet / Ollama `qwen3:8b` / Kokoro) when a
  cloud service stops answering (`eva/failover.py`). Preset `local` is that stack offline.
* Runs from the laptop (`run.py`) or from a phone in the browser (`run.py --web --tls`,
  `eva/web/`). Languages: `--lang auto|en|ru`; all language data is under `eva/assets/`.
* Measured: ~1.0–1.5 s from the user's last word to her first; 6.2/10 in the conversation
  eval; 6/6 on the tool-calling probe; ~$4.50 per hour of conversation, 80 % of it the voice.
* 44 offline tests (`pytest tests`), a real-audio simulator and a conversation eval in `bench/`.

## How to work here

**Measure before changing, and measure after.** Every tuning decision in this repo came from
a number (a latency table, an eval score, an RMS envelope, a ping). When something "sounds
wrong", find the signal-level evidence first; three of the last four audio bugs were in code
that had been written to fix a *different* audio complaint. Record new numbers in
`docs/MEASUREMENTS.md` with the date.

**One pipeline, data for the differences.** Personas, languages, voices, fillers and tool
sets are data (`eva/assets/`, `eva/config.py`); the loop (`eva/pipeline.py`) is shared. Do not
fork the pipeline for a use case. New jobs (support, dispatch) will be new personas + tool sets
+ eval scenarios, not new agents.

**Tests are the memory of bugs.** Every bug that reached the user gets a scenario in `tests/`
that fails on the old code (`y_continuous_audio_inside_a_chunk`, `u_echo_partial_does_not_interrupt`,
`test_web_page_scripts_parse` ...). Run `pytest tests -q` before every commit (about 3 min).

**API quotas cost real money and are small.** Debug on doubles (`eva/mocks.py`) or the local
stack; run the conversation eval only for a decision; never run benchmark matrices. Cerebras
and ElevenLabs keys are in gitignored `*_key.txt` files; never print or commit them.

**Commit when a step is verified**, with a message that says what was measured; push to
`origin/main`. Commit trailers as the session reminder specifies.

**The user reads the console.** Every silent decision the pipeline takes (an ignored
transcript, a barge-in refused as echo, a failover, the phone's buffer stats) prints a dim
line saying why. Keep that: it is how bugs get reported.

## Decisions and why (do not relitigate without new data)

* **Brain: qwen-3.8-27b on Cerebras.** Eval 6.2/10 vs gpt-oss-120b 3.5 (fakes tools, leaks
  JSON), local qwen3:4b 2.0, qwen2.5-coder:7b 2.3, gpt-5.4-nano 6.0; gpt-5.4-mini scored 8.0
  and gpt-5.6-luna 7.7 but OpenAI is not the brain (owner's constraint: open models). The
  live comparison settled it: "much more natural by much larger margins". `qwen3:8b` is the
  first local model that calls tools correctly (6/6) and is the fallback brain.
* **Voice: v3 for every sentence.** A Flash first chunk was 0.4 s faster but a different
  timbre, pace and noise floor on the first sentence of every reply. One delivery cue per
  reply, inherited by every chunk: a voice does not change colour every sentence.
* **Edges:** v3 clips are trimmed hot (first 10 ms −39 dBFS, last 10 ms −31); 120/280 ms
  reply fades, 40 ms fades and a 220 ms pause at chunk boundaries only. Room tone is off
  (audible on a phone speaker). Loudness leveled per model × voice (`eva/audio/leveler.py`).
* **Turn taking:** 500 ms endpoint; unfinished transcripts get a 600 ms grace and merge;
  bare hesitations wait; the Whisper phantom list applies only to Whisper-class STTs.
* **Speaker echo (laptop):** barge-in while she is audible needs real words in the partial
  transcript that are not a fuzzy copy of her reply, plus the echo detector (`eva/audio/echo.py`,
  correlation with what the player played). **Phone:** the browser cancels its own echo, so
  the gates are off there (`?aec=0` brings them back).
* **Failover, not retries:** a request that cannot start or gives no first result in 5/6/4 s
  goes local for that turn, cooldown 20 s, auto-recovery. Backups warm in the background.
* **Memory** is told who the user and the assistant are; name facts are never stored.

## Gotchas that cost hours

* The laptop mic array has hardware echo cancelling; recording her own output through it is
  useless for measurement. Measure at the bytes written to `Player.write` (monkeypatch).
* Ollama hybrid Qwen3 models only stop thinking through the native `/api/chat` `think` field
  (`eva/llm/ollama_native.py`); the OpenAI-compatible endpoint ignores it.
* Browsers allow the mic only on https; `--tls` makes a self-signed certificate with Git's
  openssl. The phone page is one file with two AudioWorklets as template strings: no
  backticks inside them (a test checks). Resample in the worklet from one ring buffer, never
  per message.
* The Claude Code Bash tool mangles backslashes inside heredocs (`\n` in Python source):
  write edit scripts to the scratchpad with the Write tool and run them. Foreground `sleep`
  does not wait; use a Python driver for paced stdin.
* Ollama at `127.0.0.1`, never `localhost` (+2 s). Cerebras needs a custom User-Agent.
* Windows Store Python cannot open the mic; the venv is a uv CPython 3.13.

## Roadmap (owner's order of value)

1. **Smart endpointing**: predict end-of-turn from Scribe's partials; cut the 500 ms wait to
   ~200 ms on a complete sentence, wait longer on a trailing one (~0.3 s off every reply).
2. **Her own voice**: an ElevenLabs voice design / clone instead of stock voices.
3. **Episodic memory**: what happened last time, moods over days, not a flat fact list.
4. **Tools that make her useful**: reminders that fire, calendar, messages, a web lookup;
   tool scenarios in the eval. Then the first *job* persona (customer support or dispatch)
   as data on the same loop, with its own scenarios and its own honesty checks.
5. **Always-on deployment** off the laptop (a small box or a host); the code does not care
   where the mic and the models live.
6. Russian persona (`eva/assets/personas/ru/eva.md`) is a draft: the owner rewrites the register.
7. Cost: the voice is 80 % of the bill; test the cheaper v3 tier, keep replies short.
