# Which voice stack for a useful Maya: the dispatcher study (2026-09-26 → 09-27)

**Decision (owner, 2026-09-27): build the MVP on OpenAI GPT-Live with the gpt-5.6-luna backend.** Of
everything tested it is the only stack that is fast, felt real, calls tools reliably and costs the
least per minute *at the same time*. Eva's own loop and the open models stay as the fallback, the
feel benchmark and the cost option at scale. Numbers: `docs/MEASUREMENTS.md`; cost write-up with
the 2x2: the Claude doc "Eva dispatcher: cost vs reliable tools".

## 1. How we got here

1. **Fully local laptop (2026-09-25/26).** Small brains (0.8–4B) and local voices on the 6 GB RTX 4050.
   The voice was the bottleneck (Chatterbox 2.3–3.4 s to first sound), not the brain (0.1–0.4 s).
   Archived in `.archive/local-variants/`.
2. **Laptop brain + ElevenLabs voice (2026-09-26).** Qwen3 4B + Parakeet + ElevenLabs on the owner's
   prepaid Creator plan: 1.9–2.7 s after the endpoint. The companion setup today.
3. **RunPod pilot (2026-09-27), shelved.** Scripts for Qwen 27B + open voices on an RTX PRO 6000
   (`deploy/runpod/`). The pod's disk didn't persist across Stop, it bills by the hour, and a pod is
   a fixed cost with no customers yet.
4. **API-first MVP (2026-09-27, owner).** Build the useful layer (data, tools, behaviour, evals) on
   pay-per-use APIs, find product-market fit, earn, then move to own GPUs at scale.

## 2. The job: a truck dispatcher on the same loop

* `eva/jobs/dispatch/world.py` builds a fake carrier (Red Oak Transport) in SQLite: 116 cities, 131
  trucks, 140 drivers with hours of service, 220 brokers (credit, pay speed, do-not-use flags),
  ~6,000 load-board postings, ~24,000 past loads, ~72,000 GPS pings, lane market rates, fuel, policy.
  Planted cases: a lowball reefer load, a late load at a strict receiver, a do-not-use broker with a
  rate too good to be true, a hazmat load whose nearest driver isn't endorsed, a breakdown with a
  reefer load due tomorrow, a truck looking for a load home.
* `eva/jobs/dispatch/desk.py`: 13 tools, the same for every backend. They enforce the rules whatever
  the model says: cost floor, do-not-use brokers, hazmat endorsement, trailer type, pickup window.
* Persona `eva/assets/personas/en/dispatcher.md`; for GPT-Live `dispatcher_live.md` (voice, with the
  delegation policy) + `dispatcher_backend.md` (procedures). A job never touches personal memory.

## 3. What was compared, and how

Six scripted calls (17 caller lines, 28 checks from the tool calls, the database and her words),
the caller **spoken** by a local voice and streamed at real-time pace; a fresh world per call
(`bench/dispatch_eval.py --audio`, `bench/dispatch_table.py`). Every failure was read in the transcripts.

| Stack | Checks | First word (median) | $ / minute | Calls fully right |
|---|---|---|---|---|
| **GPT-Live 1 + gpt-5.6-luna** | **28/28** | **1.12 s** | **0.049** | **6/6** |
| GPT-Live 1 + gpt-5.6-sol | 28/28 | 1.87 s | 0.073 | 6/6 |
| Gemini 3.8 Live, extended thinking | 28/28 | 1.72 s | 0.088 | 6/6 |
| gpt-realtime-1.5 | 28/28 | 1.16 s | 0.109 | 6/6 |
| gpt-realtime-2.1 | 28/28 | 1.20 s | 0.124 | 6/6 |
| gpt-realtime-2 | 28/28 | 1.81 s | 0.140 | 6/6 |
| gpt-realtime (Aug 2025) | 27/28 | 1.81 s | 0.108 | 5/6 |
| Gemini 3.8 Live | 26/28 | 2.57 s | 0.054 | 5/6 |
| gpt-realtime-2.1-mini | 26/28 | 1.34 s | 0.038 | 5/6 |
| Gemini 3.1 Flash Live | 25/28 | 1.92 s | 0.038 | 5/6 |
| gpt-realtime-mini | 24/28 | 1.07 s | 0.029 | 3/6 |
| Gemini 2.5 Flash native audio | 23/28 | 3.31 s | 0.018 | 3/6 |
| Eva's loop: Cerebras 27B + ElevenLabs v3 | 28/28 (typed) | ~2.4–3.2 s (estimate) | ~0.06–0.08 (estimate; Flash ~0.03–0.04) | 6/6 typed |

**The owner's ear (live sessions):** GPT-Live "performed unusually good in terms of reality of
conversations"; Gemini voices "robotic"; Eva's loop felt weird with the companion's fillers.

**What the cheap tiers got wrong** (why cost per minute isn't the whole story): gpt-realtime-mini
never checked the blacklisted broker and started negotiating with him; gpt-realtime-2.1-mini dropped
a digit of a PO number; Gemini 3.1 Flash Live read out "load number 10037" for a booking it never
made; Gemini 2.5 booked with a made-up posting id. One wrong booking costs more than a year of the
price gap.

## 4. Why GPT-Live is the choice

* **Architecture:** a full-duplex voice (listens while it speaks, backchannels, interruptions) that
  *delegates* reasoning and tools to a text backend. The backend can be upgraded or swapped per call
  (luna → sol) without touching the voice. Nobody else offers that combination off the shelf.
* **Cost:** $0.05 a minute plus backend tokens (< $0.01 a call); a 4-minute call ≈ $0.20; the pilot
  (50 calls a day) ≈ $294 a month, a small business (300 a day) ≈ $1,764, plus a phone line
  (~$0.0085/min inbound).
* **Scale:** OpenAI runs it; more calls = more sessions, up to the account's concurrency limit.
* **What it took:** two prompts in OpenAI's recommended layout, and a spoken "checking" when the
  backend is still busy 2 s after a hand-off (it had left 12.6 s of silence). Then 28/28.
* **What it costs us:** a closed model and voice set, per-minute billing *including silence and hold*,
  OpenAI in the data path, their prices and limits. The tools and data stay ours and backend-agnostic.

## 5. Open speech-to-speech models (survey, 2026-09-27)

* **The open way to match GPT-Live** is its own design: an open full-duplex voice in front, handing
  tools to an open text brain (e.g. Cerebras Qwen 27B). A paper (arXiv 2609.19334) reports 92–97 %
  tool-call recall that way; no code published. Buildable, not off the shelf.
* **NVIDIA NemotronLabs VoiceChat 11B** (2026-08): the only open full-duplex model with native tools,
  but weak tool scores (33 % pass@1 on Full-Duplex-Bench v3), one fixed voice, ≥ 80 GB GPU in BF16
  (≥ 16 GB at Q4). Served by vLLM-Omni `/v1/realtime?duplex=1`.
* **NVIDIA PersonaPlex 7B** (Moshi-based): full duplex, the best feel, **no tools**; ~17 GB FP16.
  Hosted by fal.ai (~$0.06/min) and personaplex.io ($0.08/min); self-host on vLLM-Omni or moshi.cpp.
* **Qwen3-Omni-30B-A3B**: turn-based, tools through its text "thinker", not yet in a realtime server.
  Alibaba's hosted realtime Omni models are closed Flash/Plus variants, not the open weights.
* **MiniCPM-o 4.5, Step-Audio 2 mini, Fun-Audio-Chat, Kimi-Audio, GLM-4-Voice**: fit one big GPU;
  few or no tools; several have no realtime server. **Ultravox** is speech-in, text-out + TTS.
* **On this laptop (6 GB):** nothing full duplex is confirmed. PersonaPlex q4 through `moshi.cpp`
  (Windows CUDA) needs ~8 GB by its author's account; LFM2.5-Audio-1.5B fits but has no Windows runner.
  A download was started and stopped by the owner's call (the connection ran at ~0.6 MB/s); not tested.

## 6. Bugs and lessons worth keeping

* Player properties called as methods killed the speech-to-speech event loop on OpenAI's first
  `speech_started`: the call "couldn't hear". Evals without a player never reached that code.
* A reference lookup that only matched exactly ("PHX-55120") missed "PHX 55120" / "PAX 55,120": 6 of
  12 models failed the breakdown call on our bug, not theirs. Lookups must forgive spacing and case.
* OpenAI's default semantic VAD waited up to 4 s: 5 s answers. A 500 ms silence endpoint fixed it.
* GPT-Live streams audio all the time: time to first word is to the first *voiced* chunk, and a
  full-duplex model can talk over the caller.
* The companion's extras (fillers, backchannels, narration between lookups) make a work call sound
  wrong; `eva.jobs.job_settings` turns them off.
* Eval checks on spoken transcripts must accept spoken numbers ("forty-eight three o eight") and
  curly apostrophes; three "failures" were our regexes.

## 7. Next

1. A phone number (Twilio → GPT-Live's telephony), the tools on a small always-on server.
2. A design partner's real dispatch system or load board behind the same tools.
3. Harder calls: noise, accents, corrections, long holds (GPT-Live bills them).
4. luna by default, sol for negotiation-heavy calls, if its slower first word is worth it.
5. Revisit the open stack (full-duplex voice + delegated open brain on own GPUs) when volume passes
   ~100–150k minutes a month, or if vendor independence becomes a requirement.

## How to run

```
.venv\Scripts\python.exe dispatch.py --list
.venv\Scripts\python.exe dispatch.py openai-live            # laptop mic, headphones
.venv\Scripts\python.exe dispatch.py openai-live --web      # the phone: https://<laptop-ip>:8443
.venv\Scripts\python.exe dispatch.py --eval openai-live --audio
.venv\Scripts\python.exe bench\dispatch_table.py
```
