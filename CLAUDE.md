# Eva — project brief for Claude

Read this first in every session. Details live in `README.md` (how to run), `docs/DESIGN.md`
(architecture, environment facts), `docs/MEASUREMENTS.md` (every number, the known issues and
the bug hunts), `docs/EVAL_REPORT.md` (why the brain and the persona were chosen).

## What this is and where it is going

Eva is a **realistic voice agent**: you talk, she listens while you speak, answers about a
second later, can be interrupted, remembers you, does small things. Today it is an early
prototype of a *personal companion* (English and Russian, friend register) that one person
uses daily from a laptop or a phone, **in English only for now** (see "English first"). It is a **serious, long-lived project**, not a demo: the
same agent is meant to grow into work — **customer support, truck dispatching, basic friend
conversation** — through different personas, tool sets and deployments on one pipeline. Two
goals in that order: **feel real first, then be useful**. Nothing that makes her more useful
may make her feel less real.

The user is Doston (native Russian speaker, English too; developer). Treat him as the owner
of every judgement call about how she should *sound*; treat measurements as the owner of
every judgement about what is *true*.

## Current state (2026-09-23)

* Stack `maya`: ElevenLabs Scribe v2 realtime (STT, language hint `en`) → Cerebras
  `qwen-3.8-27b` (brain, reasoning low) → ElevenLabs v3 (voice, one delivery cue per reply,
  `eva_en` voice). `--preset maya-lite` is the same stack on v3 Conversational (under live A/B).
  Falls back per provider to the local stack (Parakeet / Ollama `qwen3:8b` / Kokoro) when a
  cloud service stops answering (`eva/failover.py`). Preset `local` is that stack offline.
* Runs from the laptop (`run.py`) or from a phone in the browser (`run.py --web --tls`,
  `eva/web/`). Language: English only by default (`lang.DEFAULT_MODE = "en"`); Russian is frozen
  but intact (`--lang auto|ru`, data under `eva/assets/`).
* Measured: ~1.0–1.5 s from the user's last word to her first on v3 (v3 Conversational is
  ~0.35 s sooner to first audio); 6.2/10 in the conversation eval; 6/6 on the tool-calling
  probe; ~$4.50 per hour on v3, ~$2.85 estimated on v3 Conversational (half the voice price;
  the voice is most of the bill).
* 49 offline tests (`pytest tests`), a real-audio simulator and a conversation eval in `bench/`.

## English first (owner's decision, 2026-09-23)

Eva is built **in English only** until English reaches the quality gates below; then other
languages come back one at a time. Russian is **frozen, not deleted**: `eva/assets/lang/ru.toml`,
`eva/assets/personas/ru/`, the language box and `--lang auto|ru` stay and their offline tests
keep passing, so bringing it back costs a one-line change (`eva/lang.py: DEFAULT_MODE = AUTO`) plus
fresh measurements.

* **Nothing Russian runs in a default session:** no Russian voice, fillers or hints are
  rendered, the STT gets `language_code=en` only, the prompt carries the English persona with
  "always answer in English", and a one-language session never switches language (a Cyrillic
  transcript is answered in English). `test_default_session_is_english_only` and
  `za_english_session_never_switches` fail if any of that creeps back.
* **No Russian work while frozen:** no Russian eval runs (`bench/conversation_eval.py --lang ru`),
  no Russian samples in benchmarks, no Russian voice tuning or voice design, no provider chosen
  or rejected on its Russian. English-only providers (e.g. Deepgram Flux STT, Chatterbox Turbo /
  Orpheus voices) are fair candidates. Keep new code language-agnostic (language data in
  `eva/assets/`), so the freeze stays cheap to lift.
* **The gates (all must hold, measured, before a second language):**

  | | 2026-09-23 | gate |
  |---|---|---|
  | last word → her first audio, median over a real session | ~1.0–1.5 s (before v3 Conv.'s −0.35 s) | ≤ 0.8 s |
  | conversation eval (`bench/conversation_eval.py`) | 6.2 / 10 | ≥ 7.5 |
  | cost per hour, measured from a real session | ~$4.50 (v3) / ~$2.85 (Conv.), estimated | ≤ $1.50 |
  | emotion: blind listening test (owner judges) | not measured yet | she wins or ties the best alternative |
  | stability: daily use with no new audio bug | — | 2 weeks |

  The owner may change a gate; record the change and the reason here.

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
* **Voice: one model for every sentence; v3 by default, v3 Conversational under A/B.** A Flash
  first chunk was 0.4 s faster but a different timbre, pace and noise floor on the first sentence
  of every reply. v3 Conversational (2026-09-23) is half v3's price, 0.35 s sooner, same tags,
  cleaner clip ends, ~20 % quicker pace, but the owner's first listen: "feels flat". So `maya`
  stays on v3 and `maya-lite` runs Conversational (`config.VOICE_MODELS`) until the live A/B
  decides; feel beats price.
  One delivery cue per reply, inherited by every chunk: a voice does not change colour every sentence.
* **Edges:** v3 clips are trimmed hot (first 10 ms −39 dBFS, last 10 ms −31); 120/280 ms
  reply fades, 40 ms fades and a 220 ms pause at chunk boundaries only. Room tone is off
  (audible on a phone speaker). Loudness leveled per model × voice (`eva/audio/leveler.py`).
* **Languages:** English only by default (above). For multi-language sessions (`--lang auto`,
  frozen): Scribe gets `language_code` + `secondary_languages` (the box: unboxed it heard English
  and Russian as Dutch, `ja`, `mk`); it is a bias, so the pipeline never switches her language on
  a label outside the session's (`stt_foreign`), and the persona says a line in another language
  is a mishearing.
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

## Roadmap (English phase, in order; each item moves a gate)

1. **Smart endpointing** (latency gate): predict end-of-turn from the partials; cut the 500 ms
   wait to ~200 ms on a complete sentence, wait longer on a trailing one (~0.3 s off every
   reply). Try an English turn-taking STT (Deepgram Flux) against Scribe + our own predictor.
2. **Emotion, measured** (emotion gate): a blind listening set (the same lines and cues across
   voices) and emotion scenarios in the conversation eval, so "sounds real" becomes a number.
3. **Voice provider** (cost + emotion gates): the live `maya` (v3) vs `maya-lite` (v3
   Conversational) A/B first, then a blind English test of the winner vs Inworld TTS-2 (~$5–25 / M chars) vs Cartesia; log characters sent vs heard (lookahead chunks
   billed on barge-in). Decide *before* item 4: a designed voice ties her to a provider.
4. **Her own voice**: a designed / cloned English voice on the chosen provider.
5. **Episodic memory**: what happened last time, moods over days, not a flat fact list.
6. **Tools that make her useful**: reminders that fire, calendar, messages, a web lookup; tool
   scenarios in the eval. Then the first *job* persona (customer support or dispatch) as data
   on the same loop, with its own scenarios and honesty checks.
7. **Always-on deployment** off the laptop (a small box or a host); a GPU box makes an open
   English voice (Chatterbox Turbo, Orpheus) a candidate main voice at $0 per hour, and an
   expressive fallback instead of Kokoro.

After the gates: other languages one at a time, starting with Russian (its persona is a draft
the owner rewrites; its voice and STT are re-measured, not assumed).
