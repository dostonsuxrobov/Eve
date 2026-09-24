# Measurements

Everything measured on one Windows 11 laptop (RTX 4050 6 GB, Python 3.13) from Philadelphia.
Every number here was produced by a real run; the raw files are under `bench/out/` (gitignored).
The first sections are the selection phase (2026-09-17/18), when seven presets were compared
head to head; the preset names in those tables map to today's two like this:
`cloud-fast` = the `maya` stack before the v3 voice, delivery cues and leveling were added,
`parakeet-local` = today's `local`; the others were dropped on 2026-09-19 (this file keeps their
numbers as the record of why).

## Response latency (selection phase)

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

## Loudness across TTS sources (2026-09-19)

Voiced RMS (50 ms hops above -41 dBFS) of one sentence per source, measured through
`ElevenLabsTTS` before leveling was added:

| source | level | peak |
|---|---|---|
| Flash v2.5, `eva_en` (the first chunk of every reply) | -21.0 dBFS | 0.86 FS |
| v3, `eva_en` | -16.6 dBFS | 0.94 FS |
| v3, `eva_en`, `[quiet]` cue | -22.0 dBFS | 0.51 FS |
| v3, `eva_en`, `[soft]` / `[bright]` | -17.7 / -17.8 dBFS | |
| v3, `eva_ru` | -25.4 dBFS | 0.67 FS |

So every reply opened 4.4 dB under its second sentence and Russian sat 8.8 dB under English,
heard as "the volume dials up from the start to the middle of every reply". With the leveler
(`eva/audio/leveler.py`, target -19 dBFS, one fixed gain per clip, soft knee above 0.8 FS) the
same three sources measured -19.7 / -20.5 / -18.6 dBFS. Eva's own output stream was checked for
underruns on an 11 s, four-chunk reply: none, so the remaining "ramp" on speakers, if any, is the
device (Windows loudness equalization / the laptop amp waking after quiet stretches).

## Clip edges and floors (2026-09-20)

One sentence rendered by each model (10 ms hops):

| | first 10 ms | last 10 ms | quietest hop | in-speech pauses p50 | voiced |
|---|---|---|---|---|---|
| v3, `eva_en` | -38.7 dBFS | **-31.3 dBFS** | -93.7 | -64.8 | -17.3 |
| Flash v2.5, `eva_en` | -80.7 | -210 (digital zero) | -210 | -51.7 | -17.1 |

v3 clips are trimmed hot at both ends: with a 50 / 100 ms envelope the reply started audibly and
stopped dead ("shuts off abruptly"); Flash ends clean but its in-speech floor is 13 dB noisier,
so a Flash first chunk under v3 also switched backgrounds a sentence in. Since this measurement:
v3 for every chunk, 120 / 280 ms reply fades, 40 ms edge fades at chunk boundaries, and room
tone at -62 dBFS in the lead-in, the tail and idle time. Cost: first audio 0.5-0.7 s instead of
0.2 (`--once` after the change: TTFA 0.67 s, 1.38 s to first audio in text mode). A reply
measured after the change: lead-in at -62 dBFS, quietest 10 % of hops at -64, tail at -62.

## The "bad connection" (2026-09-20)

Heard on the phone and then on the laptop: her voice choppy "like a person with a bad
connection". Ruled out in order, each with a measurement: the laptop's internet (68 Mbit/s,
31-46 ms to the APIs), v3 delivery (2.7x real time after the first burst, no mid-stream stall),
the phone's Wi-Fi (rtt 7-15 ms in session, 0 buffer drops), iOS voice processing (same with
echo cancelling off), local player starvation (18 s and 9 s replies, zero gaps), and v3's own
rendering (7-10 natural intra-speech dips per 12 s clip, same as Flash and Turbo). Two real
causes, both mine, both from the tone commit the day before: the phone page resampled each
network message on its own (a click ~12x/s at the boundaries), and the writer applied the new
40 ms "chunk edge" fade at every ~80 ms piece release instead of only at chunk boundaries: a
12 Hz tremolo on 20 % of her speech (the regression test on the old writer: 67 of 335 hops
inside one chunk dipped below half level; fixed: 0). Lesson recorded in the test suite:
`y_continuous_audio_inside_a_chunk` and `test_web_page_scripts_parse`.

## Voice: v3 Conversational (2026-09-23)

`eleven_v3_conversational` costs $0.05 / 1k characters against v3's $0.10 (ElevenLabs API
pricing page, same rate on every plan) and takes the same inline tags. The same four cued lines
from a live session, same voices, raw (no leveler), one render each:

| model | line | TTFA | audio | voiced | peak | first 10 ms | last 10 ms | pause p50 |
|---|---|---|---|---|---|---|---|---|
| v3 | en `[warm]` | 0.62 s | 2.56 s | -13.7 | 0.90 | -68.4 | -52.9 | -54.3 |
| v3 | en `[amused]` | 0.71 | 4.40 | -17.4 | 0.97 | -81.9 | **-29.5** | -75.3 |
| v3 | ru `[gentle]` | 0.64 | 5.92 | -23.3 | 0.54 | -56.5 | **-35.4** | -60.3 |
| v3 | ru `[playful]` | 0.72 | 3.92 | -21.7 | 0.55 | -69.1 | **-24.4** | -56.5 |
| conversational | en `[warm]` | **0.33** | 2.56 | -20.3 | 0.57 | -80.5 | -30.8 | -66.3 |
| conversational | en `[amused]` | **0.30** | 3.60 | -16.4 | 1.00 | -87.6 | -83.0 | -54.0 |
| conversational | ru `[gentle]` | **0.27** | 4.64 | -20.7 | 0.64 | -58.4 | -57.1 | -61.5 |
| conversational | ru `[playful]` | 0.78 | 3.76 | -20.3 | 0.56 | -77.1 | -61.7 | -56.7 |

So: about 0.35 s sooner to first audio, English and Russian 1.6 dB apart instead of 6.9 (the
leveler seeds are now -18.4 / -20.5), three of four clip ends clean (v3: none), and about 20 %
quicker delivery (ru `[gentle]` 4.6 s against 5.9 s): the pace is the listening question. Like v3
it refuses `optimize_streaming_latency` and `previous_text` (HTTP 400 `unsupported_model`).
`run.py --once` in Russian after the switch: TTFA 0.30 s, 0.79 s to first audio in text mode
(v3 on 2026-09-20: 0.67 / 1.38 s). The WAVs are in `samples/out/v3_vs_conversational/`.

The owner's first live listen on Conversational: "feels flat". Both now ship: `maya` (v3, the
default again) and `maya-lite` (Conversational) for a live A/B; the numbers above say nothing
about expressiveness, which is the owner's call.

## STT language box (2026-09-23)

Live log: Russian and English speech came back from Scribe as Dutch ("Nee, het is niet.",
"Dat is een groot deal.") and she answered in Dutch. Without `language_code` Scribe realtime picks
among ~90 languages. Repo samples through the realtime socket, one socket at a time:

* Clean samples (en hello, en task, ru rough day, and 1.0-1.6 s slices): identical and correct
  in auto, `en` + `secondary_languages=[ru]` and `ru` + `[en]`, including the code-switch.
* The same speech in 1.2 s windows at 0 dB SNR (17 clips): auto mislabelled "...call my mom
  later" as `ja` ("You call my mom later。") and returned nothing for "Can you set a timer for
  five-"; both boxes got them right ("I'll call my mum later.", "Can you set a timer for five-"),
  and `en`-first and `ru`-first gave the same results. Noise-only tails still came back `mk` "Да."
  in every mode: the box is a bias, not a wall.
* `secondary_languages=zz` is refused with the list of valid codes, so the parameter is parsed.
  An array is the repeated query key; `session_started` echoes it.
* With `include_language_detection=true` Scribe sends `committed_transcript_with_timestamps`
  (with `language_code`, no words) *before* `committed_transcript`, so the label costs no wait.
  Commits with the box and detection on, samples fed at mic pace: 0.12-0.21 s, the same as before.

Hence: the realtime STT is boxed into the session's languages in auto mode, the pipeline never
switches her language on a label outside them (`stt_foreign` dim line, test
`z_foreign_language_label_keeps_language`), and the bilingual persona rule says a line in another
language is a mishearing. `--once "Dat is een groot deal."` after the rule: "Hmm, didn't quite
catch that. What did you mean?" (before: "Mm, groot is het. Maar ik ben er toch").

## The `maya` stack after the 2026-09-19 fixes

Two-utterance simulator run (`samples/user_hello.wav`, `samples/user_ru_rough_day.wav`,
gpt-oss brain, real speakers): 0.63 s and 0.47 s from endpoint to first audio (STT 0.09,
TTFT 0.30 / 0.20, TTFA 0.23 / 0.18). Later the same day on qwen in locked Russian mode:
1.7-2.2 s with the account in its slow Scribe state (commit 0.8-1.1 s; the same utterance in
auto mode at that moment: 0.76 s), so the language hint is not what costs there.

## Cloud outage (2026-09-19)

`bench/e2e_sim.py --outage llm,stt,tts` (every cloud provider pointed at a dead host), first
turn: STT 4.5 s (Scribe's two batch attempts, then Parakeet 0.45 s), LLM 5.9 s (Cerebras retry,
then Ollama qwen3:4b first token), TTS 3.2 s (ElevenLabs connect failure, then Kokoro), 13.6 s
in all; the reply was spoken. Later turns skip the dead primaries for the 20 s cooldown
(`eva/failover.py`), so they cost the local stack's own 1.3-2.2 s. With a hanging rather than a
refusing network each provider is bounded by its first-result timeout (5 / 6 / 4 s).

## Turn-taking mechanics

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
fallback and batch presets) and one v3 (or v3 Conversational on `maya-lite`) request per spoken sentence chunk
(typically one to three per reply; the persona's fillers are synthesized once per session).
Ollama and Kokoro presets cost nothing. Prices on 2026-09-23: Scribe realtime $0.39 per hour
of audio; v3 $0.10 and v3 Conversational $0.05 per 1k characters. The ~$4.50 per hour estimate
(80 % voice) is on v3 (`maya`), i.e. roughly 33k characters per hour; on v3 Conversational
(`maya-lite`) the same hour is about $2.85 (estimated from the price, not a measured session).

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
  failover layer (`eva/failover.py`) moves transcription to the local Parakeet model after
  `STT_TIMEOUT_S` (6 s) on one request and keeps it there for the 20 s cooldown.
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
* The echo guard is a threshold bump, not echo cancellation. On 2026-09-19 a live session on
  speakers looped for twelve turns: her voice reached the mic, 300 ms of VAD "speech" cut her
  off, Scribe transcribed the garbled residue ("когда придёшь к решению" came back as "А
  когда придёшь к ней, шей"), and she answered herself. Since then a barge-in while she is
  audible needs real words in the partial transcript that are not a fuzzy copy of her reply,
  plus a clear verdict from the echo detector (`eva/audio/echo.py`: normalised
  cross-correlation of the mic against what the player just played; on synthetic echo at
  -20 dB it flags 96-100 % of frames with the lag recovered to within 20 ms, the user's own
  voice 0 %). Not yet measured on the real speakers-to-mic path: the laptop's mic array has
  its own echo canceller whose residual is what leaks, and how strongly that residual
  correlates is unknown until a live session prints `echo_ratio` values. Headphones remain
  the clean setup.
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
* **Persistent TTS websocket.** The stream-input websocket was slower than HTTP here because it was
  reopened per reply (removed 2026-09-19; in git history). One connection per session with a
  keep-alive could bring the TTFA below the 0.14-0.17 s HTTP figure.
* **Orpheus or CSM on a bigger GPU.** Kokoro is fast but flat. A conversational speech model
  (Orpheus 3B, Sesame CSM 1B) needs more than the 6 GB this laptop has left after Ollama, but would
  give the local path the prosody the cloud path has.
* Repair terminal punctuation on the assistant text before it is spoken and stored (removes any
  remaining truncation cascade regardless of the reasoning setting), and the sanitizer additions
  listed in the eval report section 9.
* A `disable_reasoning` retry when a reasoning turn ends `finish=length` with empty content.
