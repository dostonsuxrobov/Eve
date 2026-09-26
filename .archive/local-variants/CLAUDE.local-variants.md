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

Everything runs on this laptop: no cloud API in the loop, no keys, $0 an hour. Quality comes
from what we build around and into open models: scaffolding (turn-taking, tool routing, guards,
memory), fine-tuning on our own data, careful audio. Not from a bigger model or a cloud service.

* **Variants, not one stack (owner, 2026-09-25):** a session is one **brain** x one **voice**
  (`eva/config.py` `BRAINS` / `VOICES`, `run.py --brain X --voice Y`, or pick by number). The
  owner talks to them and decides by ear. Brains: Ollama models from 0.8B to the 4B; the 1-2B ones
  get the short persona `eva_small` (a draft the owner rewrites) and the tool gate
  (`eva/toolgate.py`: only the tools the user's words point at). Voices: Kokoro (CPU), Orpheus 3B
  (5 voices), Chatterbox Turbo and Chatterbox (cloning `eva/assets/voices/eva.wav`, an Orpheus
  "tara" render).
* **One cloud brain, for comparison (owner, 2026-09-26):** `qwen27b` is the cloud era's Cerebras
  qwen-3.8-27b (`eva/llm/openai_compat.py`, key in the gitignored `cerebras_api_key.txt`), so the
  owner can hear what the brain alone changes with the same voice. It is a test, not the direction:
  the default stays local, and its quota is small (no benchmarks on it).
* **The voice server:** voices that need PyTorch run in `.venv-voice` behind `voice/server.py`
  (stdlib HTTP on 127.0.0.1:8765, one engine on the GPU at a time); `eva/tts/voice_server.py` is
  the loop's side and starts it on demand. It maps Eva's generic sounds (`[laughs]`) to each
  engine's spelling and her mood cue to Chatterbox's `exaggeration`. Separate environment because
  Chatterbox pins torch 2.6 / transformers 5.2 / gradio.
* **The machine:** RTX 4050 laptop GPU, **6 GB VRAM**, shared by brain and voice *at the same
  time* (STT and Kokoro run on the CPU). Measured: brains 1.2-3.3 GB; Orpheus 2.1 GB in Ollama;
  Chatterbox Turbo 3.1 GB and Chatterbox 3.6 GB of PyTorch, plus ~0.3 GB of CUDA context. Over
  ~5.7 GB, Windows does not fail: it spills into system RAM and the GPU runs ~20x slower. Sessions
  unload the Ollama models they don't use first (`eva/gpu.py`). Windows 11; venvs by uv, CPython 3.13.
* **Fine-tuning** happens on this laptop where it fits. If a step needs more than the laptop,
  ask the owner first.
* **English only** (since 2026-09-23). Other languages come back once English is good; the
  Russian persona and data are in the archive.

## The archive

`.archive/` holds everything built up to 2026-09-25 (snapshot of commit `d982caa`): the
cloud-stack pipeline with its local fallback, turn-taking, barge-in and echo handling, the phone
web client, tests, bench and docs. Its brief is `.archive/CLAUDE.old.md`; its numbers are
`.archive/docs/MEASUREMENTS.md` and `.archive/docs/EVAL_REPORT.md`. Already brought back out
(2026-09-25): the local loop and its tests, the model weights (`models/`), Eva's memory
(`memory.json`, `notes.json`) and the English sample recordings (`samples/`).

* **A quarry, not a dependency.** Bring a piece in on purpose, read it, keep what fits the local
  design and bring its tests with it. Never import from `.archive/`. Move weights out
  (`mv`, no download) when the new code needs them.
* **Its lessons still hold.** Before rebuilding a part (player, echo gate, endpointing, fades),
  read that part's bug hunt in the archived MEASUREMENTS.md. Three of its audio bugs were in code
  written to fix a *different* audio complaint.
* **Local baseline on record (2026-09-19/25), the numbers to beat:** fully local median 2.1 s
  from the user's last word to her first audio (4B first-sentence TTFT 1.1–1.4 s, 0.3 s with a
  warm prefix, 7.2 s on a cold load; Kokoro first audio 0.3–0.55 s; endpoint wait on top).
* **The 4B alone, measured 2026-09-25** (`docs/MEASUREMENTS.md`): 3255 MiB on the GPU at an 8k
  context, which leaves 2.8 GB for STT and voice; 56 tok/s; 0.07 s to the first word with the
  persona cached, 0.71 s on the first reply. Brains of 1–2B free up to 2 GB more, but none kept
  Eva's register on the same persona (same file).
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
* The phone client (`run.py --web --tls`): browsers allow the mic only on https (self-signed
  certificate via Git's openssl); its AudioWorklets are template strings, so no backticks inside.
* On battery, Windows caps the GPU at 50 W (101 W on AC) and the voice runs about a third slower:
  check the power state before calling anything a regression (`run.py` warns).
* Ollama keeps every model loaded for its keep-alive (30 min here): models from an earlier session
  still hold VRAM. `eva/gpu.py` unloads them; measure VRAM with `nvidia-smi` (it counts what
  `ollama ps` leaves out, like the Qwen 3.5 vision encoder).
* Orpheus through Ollama: raw `/api/generate`, prompt `<|audio|>{voice}: {text}<|eot_id|>`, stop at
  `<custom_token_2>` (without it the model runs on into a new utterance in another voice). Its
  SNAC decoder runs on the **CPU**: on the GPU it fights Ollama's process for the card and cut
  generation from 71 to 49 tok/s. Every clip opens with ~0.5 s of silence it must generate; the
  server prefills 5 silent frames instead (first voiced frame 0.22-0.36 s instead of 0.57-1.25 s).
* Hugging Face's xet downloads stalled at 0 bytes here: `HF_HUB_DISABLE_XET=1` (the voice server
  sets it). Chatterbox Turbo under NumPy 2 fails in `prepare_conditionals` ("expected scalar
  type Double"): the server casts its loudness-normalised reference back to float32.

## Where it stands and what's next (2026-09-25; the owner orders it)

1. **The local loop: done.** The archive's loop (turn-taking, barge-in, echo gates, fillers,
   memory, tools, phone client) without the cloud, 47 offline tests. Measured end to end on
   recorded speech (`bench/e2e_sim.py`): 1.5-1.7 s last word -> first audio with Kokoro,
   1.7-3.1 s with Orpheus.
2. **Latency:** Orpheus is the slow part (0.87x real time here, 1.1-1.8 s to its first sound);
   ideas in docs/MEASUREMENTS.md. Then end-of-turn prediction from the partials. Target <= 0.8 s.
3. **Scaffolding for small brains:** tool gate, speech guard (`eva/guard.py`), the loop answering
   plain weather/time questions itself, memory lines with the user's name: 18/56 -> 0/56 flagged
   replies on the owner's replayed lines (`bench/replay.py`). Next: help-desk lines, coherence.
4. **Feel:** the owner talks to the variants; a local measure of feel that agrees with his ear
   (rated live sessions, the brain behaviours on record as tests); prompt, scaffold and fine-tune
   the chosen brain on conversations he approves.
5. **Voice:** the listening set (`bench/voices.py`) for his ear; tune cues and sounds per voice.
6. **Memory over days**, then useful tools, then the first job persona.
