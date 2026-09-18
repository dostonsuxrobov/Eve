# Eva

Eva is a personal voice agent in the spirit of Sesame's Maya: you talk, she listens while you
speak, answers about a second after the endpoint fires on the fast cloud path (1.0 s median from
endpoint detection, 1.65 s from your last word, with ElevenLabs STT in its fast state; see
"Measured latency"), can be interrupted mid-sentence, hums "mm, hang on" when a reply is slow, and
can do a few small things (timers, notes, weather, open a URL). She is one asyncio pipeline (`VAD -> STT -> LLM -> sentence chunker -> TTS -> player`) with
swappable providers, so seven "presets" can be compared head to head on the same code. Built and
measured on one Windows 11 laptop (RTX 4050 6 GB, Python 3.13). Not built for scale.

`DESIGN.md` has the architecture notes and the hard-won environment facts; `docs/EVAL_REPORT.md`
is the conversation-quality evaluation that picked the default brain and persona.

## The seven presets

| preset | STT | LLM | TTS | what it tests |
|---|---|---|---|---|
| `cloud-fast` (default) | ElevenLabs Scribe v2 realtime (websocket, audio streamed while you talk) | Cerebras `qwen-3.8-27b`, reasoning low | ElevenLabs Flash v2.5 (HTTP stream) | the lowest-latency cloud stack |
| `cloud-smart` | same | Cerebras `gpt-oss-120b`, reasoning low | same | whether a bigger brain is worth its extra TTFT (it is not, see the eval) |
| `expressive` | same | Cerebras `qwen-3.8-27b`, reasoning low | ElevenLabs v3 with `[laughs]` / `[sighs]` audio tags | the most emotional voice against its higher TTS latency |
| `local-stt` | faster-whisper `base.en` (CPU, int8) | Cerebras `qwen-3.8-27b` | ElevenLabs Flash | removing the STT network hop |
| `local-brain` | ElevenLabs Scribe v2 (batch) | Ollama `qwen3:4b-instruct-2507-q4_K_M` | ElevenLabs Flash | how far a 4B local model gets with cloud audio |
| `fully-local` | faster-whisper `base.en` | Ollama `qwen3:4b` | Kokoro (ONNX, CPU) | free, private, offline |
| `parakeet-local` | NVIDIA Parakeet TDT 0.6B v2 int8 (sherpa-onnx) | Ollama `qwen3:4b` | Kokoro | a faster, more accurate local STT than whisper base |

The presets live in `eva/config.py` (`PRESETS`); `eva/factory.py` builds the providers from
them. The qwen presets send `reasoning_effort: "low"`; set `"reasoning": "none"` in a preset if
you want `disable_reasoning` (0.14 s faster to first content, but see "Known issues").

## Setup

```
uv venv .venv --python 3.13          # or python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
```

Keys: put them in `cerebras_api_key.txt` and `elevenlabs_key.txt` in the repo root (both
gitignored) or export `CEREBRAS_API_KEY` / `ELEVENLABS_API_KEY`. The ElevenLabs key needs
text-to-speech and speech-to-text permissions; `voices_read` is not required (voice ids are
hardcoded in `config.EL_VOICES`).

Models (the `models/` folder is gitignored):

* `models/silero_vad.onnx`: the Silero VAD v5 ONNX export from the silero-vad repository. Required
  by every preset.
* faster-whisper, Kokoro and Parakeet weights are downloaded on the first `warmup()` of the preset
  that needs them (139 MB, 311 MB + 27 MB of voices, and a 461 MB archive).
* Ollama presets: `ollama pull qwen3:4b-instruct-2507-q4_K_M`. Eva talks to
  `http://127.0.0.1:11434` (not `localhost`, which costs ~2 s per request on Windows).

## Run

```
.venv/Scripts/python.exe run.py                              # cloud-fast, persona eva, mic + speakers
.venv/Scripts/python.exe run.py --preset fully-local
.venv/Scripts/python.exe run.py --list-devices               # audio device ids, then exit
.venv/Scripts/python.exe run.py --input-device 1 --output-device 4
.venv/Scripts/python.exe run.py --text                       # type instead of talk (Eva still speaks)
.venv/Scripts/python.exe run.py --once "hey eva, how's it going"   # one typed turn, then exit
.venv/Scripts/python.exe run.py --persona calm_coach --voice lily --no-barge-in --mute-fillers
```

`--preset` one of the seven above; `--persona` one of `eva` (default), `calm_coach`, `maya_like`,
`witty_companion`; `--voice` an ElevenLabs name from `config.EL_VOICES` (or a raw voice id) or a
Kokoro voice; `--user-name` is rendered into the prompt; `--debug` prints every pipeline event.
Ctrl-C ends the session: the memory summariser makes one more LLM call to extract durable facts
into `memory.json`, then everything is closed. Set `PYTHONIOENCODING=utf-8` on a cp1252 console.

Bench and simulation (no microphone needed):

```
.venv/Scripts/python.exe bench/e2e_sim.py --mock                      # 19 pipeline tests on mocks, no network
.venv/Scripts/python.exe bench/e2e_sim.py --preset cloud-fast --silent # the three sample utterances end to end
.venv/Scripts/python.exe bench/summarize_e2e.py                        # the latency table below
```

The end-to-end simulator feeds `samples/user_*.wav` (a greeting, a rough-day story, a timer +
reminder request) into the real pipeline as if they came from the microphone and writes
`bench/out/e2e_<preset>.json` with per-turn text and timing. One cloud run costs five Cerebras
completions (warmup + three turns, the tool turn being two rounds) and three ElevenLabs STT plus
three to six TTS requests.

## Measured latency

Response latency is the number that matters: the user stops talking to the first agent audio.
The pipeline can only measure it from the moment the segmenter detects the endpoint, and that
moment is 0.6 s after the last word: 550 ms of endpoint silence is 18 VAD windows of 32 ms
(576 ms) plus the frame granularity, measured at 0.60-0.64 s (median 0.62 s) from the end of
speech in the sample wavs on every turn of the runs below. **Add 0.62 s to every response figure
in this table to get last-word-to-first-audio**: on `cloud-fast` that is 1.65 s median over the
nine turns of runs 1-3 (1.36-3.00 s). `DESIGN.md` budgets 900 ms for speech-end to first audio
with the endpoint silence inside the budget; the best measured runs miss that target by about
0.7 s, and the endpoint silence alone is two thirds of the budget.

Medians over the non-interrupted turns of the recorded runs (`cloud-fast` over three runs of the
three utterances, the others over one run each); every figure is measured, from
`bench/summarize_e2e.py`:

| preset | turns | median response (s) | max (s) | STT (s) | LLM TTFT (s) | TTS TTFA (s) | dominant stage |
|---|---|---|---|---|---|---|---|
| cloud-fast | 9 | **1.01** | 2.40 | 0.26 | 0.54 | 0.16 | llm_ttft |
| cloud-smart | 3 | **1.59** | 3.28 | 0.23 | 0.62 | 0.31 | llm_ttft |
| expressive | 3 | **4.18** | 4.33 | 2.53 | 0.64 | 0.54 | stt |
| local-stt | 3 | **1.29** | 1.29 | 0.43 | 0.66 | 0.15 | llm_ttft |
| local-brain | 3 | **2.60** | 2.93 | 1.20 | 1.24 | 0.28 | llm_ttft |
| fully-local | 3 | **2.10** | 2.18 | 0.47 | 1.16 | 0.46 | llm_ttft |
| parakeet-local | 3 | **1.84** | 1.95 | 0.28 | 1.08 | 0.51 | llm_ttft |

Which stage dominates: on every cloud preset it is the Cerebras time to first token (0.5-0.7 s
median, with a 2-3 s tail that a filler covers); `expressive` was hit by the realtime STT's
server-side tail in two of its three turns (3.1 and 2.5 s; its TTS is also 3x slower than Flash);
`local-brain` pays for batch Scribe (1.2 s) on top of the 4B model's 1.2 s TTFT; the two Kokoro
presets are bounded by the local LLM (1.1-1.4 s TTFT for the first sentence, 0.3 s when the
history is warm) and Kokoro's 0.3-0.55 s first-audio time. The endpoint silence (550 ms) is on
top of all of these and is the same for every preset.

The ElevenLabs rows are the account's *fast* STT state (all recorded between 20:36 and 21:00).
They are not reproducible while the account is in the slow state described under "Known issues"
(continuously since about 21:01 that evening): a `local-brain` rerun at 21:53
(`bench/out/verify_local_task/e2e_local-brain.json`) measured batch Scribe at 2.68 / 5.43 / 4.63 s
for 3 / 8 / 4.4 s of audio (4.53 / 6.06 / 6.35 s response, LLM and TTS unchanged), with nothing
else on the account for the first two turns; a `cloud-fast` run at 22:36 with the adaptive
commit deadline (`bench/out/fix_cloud-fast.json`) got 2.47 / 0.71 / 1.89 s (STT 1.59 / 0.21 /
1.21 s, no batch fallback) while a standalone batch probe took 3.0 s for 2.8 s of audio. The two
fully offline presets reproduced within 0.1 s the same evening.

Other cloud-fast measurements from the same files: the ElevenLabs stream-input websocket was not
faster than per-sentence HTTP streaming here (1.39 vs 0.88 s median, so HTTP is the default); a run
on the real speakers matched the silent runs (0.94 s); the original batch Scribe v1 configuration
was 2.02 s (STT 1.38 s), which is why the realtime websocket is used.

## Turn-taking: barge-in, fillers, echo guard

**Barge-in.** A speech onset while Eva is thinking or speaking is a candidate. It is confirmed
after 300 ms of speech-positive VAD windows (a cough or a "mm" never stops her). The count is the
segmenter's speech time both while you are still talking and when the blip ended before it
reached 300 ms: the utterance *length* is not usable for that because every utterance carries the
300 ms pre-speech buffer and a 150 ms tail (a 288 ms cough arrives as a 0.72 s utterance; before
this check every confirmed blip interrupted her at its endpoint). On confirmation the player
stops within one output block (measured 0.02 ms for the stop call, 20 ms from confirmation to
silence), the response task is cancelled (LLM stream, TTS requests, filler timer) and awaited,
then the player is stopped once more so a writer or filler that was already scheduled in the same
event-loop iteration can never leave audio behind, and the number of samples actually played is
mapped onto the per-sentence sample counts so only the words you heard go into the history,
followed by `[interrupted]` (e.g. `"Hey, Sam. Not bad, just [interrupted]"`). If nothing had been
heard yet, the interrupted question is taken back and merged with what you say next, so "Hey Eva
... how's it going" becomes one turn instead of two. A barge-in while a tool is still running
answers the pending tool call with a synthetic `cancelled` result before the `[interrupted]`
message, so the history stays a valid chat sequence. A reply uses at most three tool rounds; a
model that keeps calling tools after that has the extra call dropped instead of executed.

**Fillers.** Each persona lists a few backchannels ("mm", "hmm", "okay, so", "mm, hang on"). They are
pre-synthesized once at startup in the active voice, and one is played only if no real audio has
started 900 ms after you stop talking, with a 0.6 s grace period when the LLM is already producing
text (the real audio is then one TTS first-byte away and a 1 s filler would only add delay). On
`cloud-fast` the median turn beats the filler; on the local presets it plays most turns.

**Echo guard and headphones.** Speaker output leaks into the microphone. While Eva speaks the
pipeline raises the VAD threshold by 0.25 (0.5 to 0.75) so her own voice is not taken for a
barge-in. That also makes her a little harder to interrupt on speakers, and loud playback can still
get through the raised threshold. With headphones there is no bleed: barge-in triggers on the first 300 ms
of your speech and the transcript stays clean, so headphones are recommended. The guard is
`PipelineSettings.echo_guard` (on by default) and can be switched off for a headphone-only setup.

Streaming STT, which is what keeps `cloud-fast` near one second: with the realtime Scribe preset
every 20 ms microphone frame is sent to the websocket from speech onset (plus a 300 ms pre-speech
ring buffer), so at the endpoint a `commit` returns the text in 0.15-0.4 s instead of the 0.7-1.3 s a
batch upload needs. The socket is rotated after every commit (the fresh one opens in the background
while the reply plays) because the server stalls on repeated content within one session.

**LLM keep-alive.** After 15 s without an LLM request (`LLM_KEEPALIVE_S`), while nothing is in
flight and you are not talking, the pipeline sends `GET /models` on the LLM's pooled connection so
the next completion never starts on a stale socket; a turn waits for an in-flight ping rather
than opening a second, cold connection. See the Cerebras TTFT item under "Known issues".

## Conversation quality: the eval verdict

Full report: [`docs/EVAL_REPORT.md`](docs/EVAL_REPORT.md) (ten scripted scenarios, four personas,
three brains, three judge lenses; every number measured).

* **Default brain: `cerebras:qwen-3.8-27b`.** Judge mean 6.00/10 against 3.50 for `gpt-oss-120b` and
  1.71 for the local `qwen3:4b`. It is the only brain with real emotional reads, the only one that
  passed every honesty check in every persona (says "AI" plainly, never promises a timer it did not
  set, refuses to write the sick-pet lie three times and drafts an honest text instead), and it is
  fast (0.2-0.4 s TTFT).
* **Default persona: `eva`**, with `calm_coach` as the low-energy alternative (6.17 and 6.33 on the
  winning brain, within noise). Both prompts were rewritten during the eval: 12-36 % fewer words per
  reply, therapy phrases gone, the flat "I'm fine" handled in one to eight words.
* **Why `gpt-oss-120b` falls short:** it refuses the lie twice and then writes it, promises reminders
  it has no tool for, answers hurt with "Got it", and leaks "[End of conversation]" into speech. It
  Its median TTFT is close (0.62 vs 0.54 s) but one turn in three took 2.9 s. Not a companion brain.
* **Why the local 4B falls short:** writes the lie on first ask, invents memories and physical
  presence, fakes tool use in an emergency, emits emoji and stage directions. Keep it for the
  fully-local privacy path and expect a 2-3/10 experience.
* **The root cause of the truncated replies** ("That stings a", "Mm. You sure") that wiped out one
  scenario was `disable_reasoning: true`: with it qwen ends 25-40 % of very short replies before the
  final word, and the broken text cascades because it is fed back as history. With
  `reasoning_effort: "low"` the rate is 0/40 and 0/82 at +0.14 s median content TTFT. The qwen
  presets now send `low` with `max_tokens` 800 so a long think can never leave the reply empty.

## API cost per turn

Every exchange resends the system prompt (persona + tool notes + memory facts, about 2,700 tokens
after the persona rewrite) plus the whole history, so the LLM cost is dominated by prompt tokens.
Measured on Cerebras (`usage` from the recorded `cloud-fast` runs): 2,770-2,900 total tokens for
a normal exchange (2,700-2,850 prompt + 30-150 completion, of which 13-110 are reasoning tokens),
and about 5,900 for the timer exchange, which is two rounds (tool call + result); the run after the
fixes below measured 2,795 / 2,879 / 5,973. So 5.9 M tokens is roughly 2,000 fresh exchanges, or
1,400-1,500 in long sessions once the 30-message history cap is reached (about 3,850-4,300 tokens
per exchange then), i.e. about five to eight hours of back-and-forth at the sample cadence of
12-15 s per exchange. The keep-alive pings are `GET /models` requests and cost no tokens. Before the
persona rewrite the same exchanges cost 1,300-2,500 tokens. The prompt grew by 60-130 tokens per
exchange in those runs; the history is capped at 30 messages, so it stops growing after fifteen
exchanges. Cerebras reports 2,048 of the prompt tokens as cached from the second round on; they
still count as prompt tokens. Ending a session adds one summariser call.

ElevenLabs: one Scribe request per utterance (a realtime commit, or a batch upload on the
fallback and batch presets) and one Flash request per spoken sentence chunk (typically one to
three per reply; the persona's fillers are synthesized once per session). Ollama and
Kokoro presets cost nothing.

## Known issues

* **Never run two ElevenLabs Scribe requests at once.** A batch request sent while a realtime
  commit was still being served made both crawl (4.2 / 31.7 / 7.6 s STT per turn,
  `bench/out/e2e_cloud-fast_race.json`), presumably the account's concurrency limit. The pipeline
  used to race a batch request against a slow commit; it now waits for the commit, cancels it and
  drops its socket, and only then sends one batch request. The wait is `STT_COMMIT_DEADLINE_S`
  (2.5 s) or `STT_COMMIT_DEADLINE_FACTOR` (2) x the slowest recent Scribe round trip (last
  commit, last batch request, the warmup probe), whichever is longer: the batch endpoint is slow
  whenever the commits are, so a fixed 2.5 s deadline made the slow state worse (2.5 + 14.0 s for
  the 8 s utterance and 2.5 + 5.2 s for the 4.4 s one in `bench/out/verify_cloud_run.json`, while
  every commit that was allowed to finish in that state returned in 1.6-2.0 s). Cancelling the
  client side cannot cancel the server's work on the committed segment, so the deadline is the
  last resort, not the plan. `ElevenLabsRealtimeSTT` serializes commits and batch calls with a
  lock, waits for a retired socket to close before opening the next one or posting a batch
  request, and warms the batch endpoint before opening the first session. The realtime commit
  still has a server-side tail of 2-4 s on some turns.
* **ElevenLabs STT is sometimes slow for this account regardless of concurrency.** In the
  verification run after the change above (`bench/out/e2e_cloud-fast_run5.json`) the very first
  commit, alone on the account, did not return within 2.5 s; the single sequential batch request
  that followed took 7.3 s for 3 s of audio, a websocket `feed()` blocked long enough on the 8 s
  utterance to break the stream, and that utterance's batch upload took 32 s (1.2 s for the same
  file in the `local-brain` run). LLM and TTS were normal in the same run (TTFT 0.6-1.0 s, TTFA
  0.2-0.3 s). The same state was seen once before (`run4`, warmup 6.3 s, STT 2.0 s per turn). The
  cause is unknown (account-level STT throttling or an incident); when it happens the
  `local-stt` preset (faster-whisper, 0.43 s) is the fast alternative.
* `qwen-3.8-27b` with reasoning on occasionally reasons for its whole budget (2/82 turns hit
  `finish=length` with 400 tokens; the presets now allow 800). An empty reply is retried once, then
  once more with a "Mm." prefill, then replaced by a fixed spoken line, so there is no dead air.
* Cerebras TTFT has a client-side tail: 2.0, 2.9 and 1.75 s were recorded on single turns in
  otherwise 0.5 s runs. It is not the model: on every tail turn Cerebras' own `time_info` shows the
  request processed in 0.13-0.18 s with 0.003 s of queue time, while normal turns carry only
  0.25-0.35 s of client/network overhead above server time, so the extra 1.5-3 s is spent before
  the request reaches Cerebras, matching the 0.75-2.2 s cold-connection TTFT in `DESIGN.md`. The
  pipeline now keeps the pooled connection warm with `GET /models` after 15 s of LLM idleness
  (`OpenAICompatLLM.ping()` was written for that and had no caller). Whether that removes the
  tail is not verified: it would take many more Cerebras turns than the quota allows; the one run
  after the change had TTFT 0.34-0.61 s on its three turns and pings of 0.47-0.75 s. The filler
  covers a tail when it happens, but it is audible.
* faster-whisper `base.en` writes "5 minutes" and "mum" for "five minutes" and "mom"; Parakeet is
  more accurate but split "tonight" into "to night" once. Both are CPU-only here.
* The echo guard is a threshold bump, not echo cancellation: loud speakers can still trigger a false
  barge-in or mask a real one. Use headphones.
* The memory summariser runs at Ctrl-C only; a crash loses the session's facts.
* Voice list and account usage cannot be read with this key (`voices_read` / `user_read` missing),
  so `--voice` accepts only the hardcoded names or a raw id.
* Windows only has been tested (sounddevice + WASAPI); nothing is platform-specific in the code
  except the Ctrl-C handling in `run.py`.

## Next steps

* **Smart turn detection.** The 550 ms endpoint silence is the floor of every preset's latency and
  still cuts people off mid-thought; a small end-of-turn model on the partial transcript (the
  realtime STT already streams partials) could cut the silence to ~200 ms when the sentence is
  complete and wait longer when it is not.
* **Persistent TTS websocket.** The stream-input websocket was slower than HTTP here because it is
  reopened per reply; keeping one connection per session with a keep-alive should bring the TTFA
  below the 0.14-0.17 s HTTP figure and remove the per-request TLS cost.
* **Orpheus or CSM on a bigger GPU.** Kokoro is fast but flat. A conversational speech model
  (Orpheus 3B, Sesame CSM 1B) needs more than the 6 GB this laptop has left after Ollama, but would
  give the local path the prosody the cloud path has.
* Repair terminal punctuation on the assistant text before it is spoken and stored (removes any
  remaining truncation cascade regardless of the reasoning setting), and the sanitizer additions
  listed in the eval report section 9.
* A `disable_reasoning` retry when a reasoning turn ends `finish=length` with empty content.

## Layout

```
eva/                the package (see DESIGN.md for every module)
  config.py         keys, PipelineSettings, PRESETS
  pipeline.py       the conversation loop and state machine
  stt/ llm/ tts/    providers behind eva/interfaces.py
  personas/*.md     system prompts (eva, calm_coach, maya_like, witty_companion)
run.py              CLI
bench/e2e_sim.py    mock test suite and the end-to-end simulator
bench/summarize_e2e.py   latency table from bench/out/e2e_*.json
bench/conversation_eval.py   the scripted-scenario bench behind docs/EVAL_REPORT.md
docs/EVAL_REPORT.md the conversation-quality report
samples/            the three sample utterances used by the simulator
```

`bench/out/` (raw results, transcripts, judged runs) is gitignored; the report in `docs/` is the
committed summary of it.
