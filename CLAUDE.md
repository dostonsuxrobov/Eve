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
she *sounds and feels*. Measurements own what is *true*: latency, VRAM, bugs.

## Fully local, on this laptop (owner's decision, 2026-09-25)

Everything runs on this laptop: no cloud API in the loop, no keys, $0 an hour. The brain is a
~4B open model on Ollama (`qwen3:4b-instruct-2507-q4_K_M` today). Reason: the owner's own
conversations with it through Ollama. It is fast and smart, with moments that have real "feel".
Quality comes from what we build around and into the model: scaffolding (turn-taking, tool
routing, guards, memory), fine-tuning on our own data, careful audio. It does not come from a
bigger model or a cloud service.

* **The machine:** RTX 4050 laptop GPU with **6 GB VRAM**. That is the budget STT, brain and
  voice share *at the same time*; record each component's VRAM when it is chosen. Windows 11;
  venv `.venv` (uv CPython 3.13).
* **Fine-tuning** happens on this laptop where it fits. If a step needs more than the laptop,
  ask the owner first.
* **English only** (since 2026-09-23). Other languages come back once English is good; the
  Russian persona and data are in the archive.

## The archive

`.archive/` holds everything built up to 2026-09-25 (snapshot of commit `d982caa`): the
cloud-stack pipeline with its local fallback, turn-taking, barge-in and echo handling, the phone
web client, tests, bench, docs, the model weights (`.archive/models/`: Parakeet TDT 0.6B int8,
Kokoro v1.0, Silero VAD) and Eva's memory files. Its brief is `.archive/CLAUDE.old.md`; its
numbers are `.archive/docs/MEASUREMENTS.md` and `.archive/docs/EVAL_REPORT.md`.

* **A quarry, not a dependency.** Bring a piece in on purpose, read it, keep what fits the local
  design and bring its tests with it. Never import from `.archive/`. Move weights out
  (`mv`, no download) when the new code needs them.
* **Its lessons still hold.** Before rebuilding a part (player, echo gate, endpointing, fades),
  read that part's bug hunt in the archived MEASUREMENTS.md. Three of its audio bugs were in code
  written to fix a *different* audio complaint.
* **Local baseline on record (2026-09-19/25), the numbers to beat:** fully local median 2.1 s
  from the user's last word to her first audio (4B first-sentence TTFT 1.1–1.4 s, 0.3 s with a
  warm prefix, 7.2 s on a cold load; Kokoro first audio 0.3–0.55 s; endpoint wait on top);
  Ollama keeps ~3 GB resident with the 4B.
* **4B behaviours on record: targets for scaffolding and fine-tuning.** Says an action ("One
  sec, setting that timer") without calling the tool (3/6 on `.archive/bench/tool_probe.py`); invents
  shared memories and physical presence; reuses example lines from the prompt verbatim; emoji
  and `*stage directions*` unless the prompt forbids them.

## How to work here

**Measure before changing, and measure after.** Latency, VRAM, RMS envelopes, tool-call rates:
find the signal-level evidence first. Record new numbers, with the date, in
`docs/MEASUREMENTS.md`.

**One pipeline, data for the differences.** Personas, voices, fillers and tool sets are data;
the loop is shared. New jobs are new personas + tool sets + scenarios, not new agents.

**Tests are the memory of bugs.** Every bug that reached the user gets a test that fails on the
old code. Run `pytest tests -q` before every commit.

**The user reads the console.** Every silent decision the pipeline takes (an ignored
transcript, a barge-in refused as echo, a guard that rewrote a reply) prints a dim line saying
why. That is how bugs get reported.

**Commit when a step is verified**, with a message that says what was measured; push to
`origin/main`. Commit trailers as the session reminder specifies.

## Gotchas that cost hours

* Ollama at `127.0.0.1`, never `localhost` (+2 s per request). Hybrid Qwen3 models only stop
  thinking through the native `/api/chat` `think` field; the OpenAI-compatible endpoint ignores it.
* The laptop mic array has hardware echo cancelling: recording her own output through it is
  useless for measurement. Measure at the bytes written to the player.
* Windows Store Python cannot open the mic; use `.venv`. The console is cp1252: set
  `PYTHONIOENCODING=utf-8`. No ffmpeg on PATH; use soundfile/numpy.
* The Claude Code Bash tool mangles backslashes inside heredocs: write scripts to the scratchpad
  with the Write tool and run them. Foreground `sleep` does not wait; use a Python driver for
  paced stdin.
* If the phone client comes back: browsers allow the mic only on https (self-signed certificate
  via Git's openssl); its AudioWorklets are template strings, so no backticks inside them.

## Where to start (proposed 2026-09-25; the owner orders it)

1. **The local loop:** Parakeet → Ollama 4B → Kokoro, with turn-taking, barge-in and echo
   handling brought over from the archive; measure latency and VRAM with all three loaded.
2. **Latency:** keep the model loaded and the prompt prefix warm, speak sentence by sentence,
   predict end-of-turn from the partials. Target ≤ 0.8 s from last word to first audio.
3. **Tools by scaffolding:** route plain intents (time, timer, goodbye) without the model, force
   the call when a reply promises an action, never let a promise stand without a call.
4. **Feel:** a local way to measure it that agrees with the owner's ear (rated live sessions,
   the 4B behaviours above as tests), then prompt, scaffold and fine-tune the 4B on
   conversations the owner approves.
5. **Voice:** the most expressive open TTS that fits next to the brain in 6 GB, blind-tested by
   the owner against Kokoro.
6. **Memory over days**, then useful tools, then the first job persona.
