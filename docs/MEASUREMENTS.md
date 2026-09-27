# Measurements

Every number of the current setup (local 4B brain and Parakeet ears, ElevenLabs voice), dated,
with how it was taken. Earlier eras: `.archive/docs/MEASUREMENTS.md` (the cloud stack, to
2026-09-25) and `.archive/local-variants/docs/MEASUREMENTS.md` (fully local: small brains,
Orpheus, Chatterbox, the voice server, the brain bench, costs of pods and cloud voices).

The machine: RTX 4050 laptop GPU, 6141 MiB, 101 W on AC and 50 W on battery.

## The owner's ElevenLabs voice on three models (2026-09-26)

Voice `bD9maNcCuQQS75DGuteM`, one 64-character line, `/v1/text-to-speech/{voice}/stream`,
`pcm_24000`, a fresh connection each (so first audio includes the TLS handshake):

| model | first audio | whole clip | audio | credits (`character-cost` header) |
|---|---|---|---|---|
| `eleven_v3` | 0.72 s | 2.19 s | 5.9 s | 35 |
| `eleven_v3_conversational` | 0.31 s | 1.34 s | 5.4 s | 17 |
| `eleven_flash_v2_5` | 0.47 s | 0.62 s | 3.3 s | 17 |

So v3 costs about 0.55 credits per character and the other two about half. There is no "Flash
v3" model (ElevenLabs model list, 2026-09-26). Clips: `samples/out/elevenlabs_test/`.

## End to end (2026-09-26)

`bench/e2e_sim.py`: three recorded utterances (hello, a rough day, a timer and a note) through
Parakeet, the 4B with its scaffolding and ElevenLabs into a silent player. "Total" is from the
endpoint (the loop deciding the user stopped) to her first audio; the ~0.5 s endpoint wait comes
on top. No greeting in the simulator, so turn 1 pays the brain's first read of the persona
(~2.8k tokens), which a real session's greeting absorbs.

| voice | total, 3 turns | STT | brain first token | voice first audio |
|---|---|---|---|---|
| v3 | 2.35 / 2.28 / 2.69 s | 0.47 / 1.09 / 0.62 | 1.13 / 0.39 / 1.49 | 0.74 / 0.80 / 0.58 |
| v3 Conversational | 1.99 / 2.04 / 2.22 s | 0.46 / 1.18 / 0.60 | 1.04 / 0.42 / 1.37 | 0.48 / 0.43 / 0.24 |
| Flash v2.5 | 1.88 / 1.90 / 1.92 s | 0.45 / 1.08 / 0.20 | 1.03 / 0.35 / 1.46 | 0.40 / 0.46 / 0.26 |

* The voice is no longer the bottleneck: 0.24-0.80 s against Chatterbox's 2.3-3.4 s on the laptop.
* Brain: 0.35-0.42 s with the persona cached (turn 2). Turn 3 is the tool turn: the 4B writes
  the timer and note calls (52 tokens) before it speaks, and "One sec" covers the wait; both
  calls were right (300 s "call mum", the note).
* Parakeet took 1.08-1.18 s on the 8 s utterance (0.2-0.6 s on the short ones): the next thing
  to speed up, with a streaming or segment-wise STT.
* Credits for the three runs: about 520 (from the characters spoken; the simulator saves the
  meter from now on). The voice test above and the fillers, rendered once and cached, about 130.

## The 4B on the owner's replayed lines (2026-09-26)

`bench/replay.py`, the owner's six lines plus the greeting, 4 sessions each, silent voice (no
credits), with and without the small-brain scaffolding (`--no-scaffold`):

| | flagged replies | weather looked up | help-desk lines | words per reply (median) |
|---|---|---|---|---|
| 4B without scaffolding | 0/28 | 4/4 (it called the tool itself) | 0 | 20.5 |
| 4B with scaffolding | 0/28 (guard dropped nothing) | 4/4 | 0 | 13 |

Against the 1B on the same lines (local-variants MEASUREMENTS): 18/56 flagged before
scaffolding, and help-desk lines 6-11 of 56 even after it. The 4B answers make sense and stay in
character ("A tree grows, a house stands. One lives, the other stays put."). The scaffolding stays
on as a net: it costs nothing when it doesn't fire, and in the brain bench the 4B did say "One
sec, setting that timer" without the call and wrote `end_conversation{...}` as text, which these
lines don't exercise.

## The dispatcher on two backends (2026-09-27)

`bench/dispatch_eval.py`: six scripted calls to Red Oak Transport's dispatcher (a lowball reefer
load, a load running late into a strict receiver, a do-not-use broker with a rate too good, a hazmat
load whose nearest driver has no endorsement, the owner asking for the best load home for a truck, a
truck broken down with a reefer load due tomorrow), 17 caller lines, 26 fact checks from the tool
calls, the database and her words. Typed caller lines; a fresh world per call.

| backend | checks | first reply, median (range) | words per reply, median | cost of the 6 calls |
|---|---|---|---|---|
| Eva's loop: Cerebras Qwen3.8-27B (reasoning low), silent voice | 26/26 | 0.63 s (0.31-1.25) to her first sentence, before the voice | 74 | Cerebras tokens only (no voice in the bench) |
| Gemini 3.8 Live, audio out | 26/26 | 1.9 s (0.7-6.9) to first audio, tool lookups included | 25 | $0.235 |

* Eva's number is before the voice: add ElevenLabs' first audio (0.24-0.80 s measured above), and on a
  live call the 0.5 s endpoint and Parakeet (0.2-1.1 s). Gemini's includes its own lookups.
* A spoken caller into Gemini 3.8 Live (`bench/s2s_audio_check.py`, Kokoro's voice streamed in real
  time): end of speech to first audio 2.49 s on a turn with a lookup, 1.50 s without; it heard
  "P O seven seven eight one two three four" as "PO7781234" and looked the load up; $0.019 for two turns.
* Both negotiate: they open at or above the target ($2,250-2,350 against a $1,900 offer, floor
  $2,100) and give true reasons (lane market, a truck with no empty miles), but both drop to
  $2,150-2,250 on the broker's first push. Both refuse the do-not-use broker with the real reason,
  pick the hazmat driver 92 miles out over the unendorsed one in Houston, and tell the truth about
  the breakdown with a recovery truck from Tulsa and a new time.
* Found and fixed on the way: the 27B said "I'm calling them right now" about a receiver (no such
  tool: `notify_facility` now sends a request whose answer is pending, and the persona says she
  can't phone); on a recovery plan it spent its 800 tokens reasoning and said nothing (now 2,000);
  Gemini tried to book a made-up posting id (the persona and the tool now say to search the board).
* Open: the 27B talks too long for a phone (still a median of 63 words after "under about forty
  words" in the persona: it narrates between lookup rounds); Gemini booked the owner's load
  without asking him. OpenAI not run yet (no key on this machine).

## The dispatcher on every speech-to-speech tier, with a spoken caller (2026-09-27)

`bench/dispatch_eval.py <backend> --audio` (`dispatch.py --eval <variant> --audio`): the same six calls,
the caller's 17 lines spoken by Kokoro and streamed at real-time pace in 20 ms chunks, the way a
phone line sends them; 28 checks. "First word" is from the end of the caller's speech to her first
audio above -45 dBFS (full-duplex GPT-Live streams while the caller talks, so the first packet is
not her answer). OpenAI's realtime tiers run with a 500 ms silence endpoint: their default semantic
VAD answered in 5.0-5.3 s on the first spoken run. A call here is 2-4 turns and 1-3 minutes.

| backend | checks | first word, median / p90 | words per reply | $ per call |
|---|---|---|---|---|
| OpenAI gpt-realtime-2.1 | 28/28 | 1.20 / 2.04 s | 56 | 0.125 |
| OpenAI gpt-realtime-2 | 28/28 | 1.81 / 3.36 s | 50 | 0.127 |
| OpenAI gpt-realtime-1.5 | 28/28 | 1.16 / 3.61 s | 40 | 0.092 |
| OpenAI gpt-realtime (Aug 2025) | 27/28 | 1.81 / 6.78 s | 26 | 0.089 |
| OpenAI gpt-realtime-2.1-mini | 26/28 | 1.34 / 3.67 s | 59 | 0.033 |
| OpenAI gpt-realtime-mini | 24/28 | 1.07 / 1.92 s | 41 | 0.025 |
| OpenAI GPT-Live 1 + gpt-5.6-luna backend | 27/28 | 1.36 / 2.26 s | 34 | 0.179 |
| OpenAI GPT-Live 1 + gpt-5.6-sol backend | 27/28 | 1.87 / 6.06 s | 40 | 0.276 |
| Gemini 3.8 Live, extended thinking | 28/28 | 1.72 / 2.42 s | 22 | 0.098 |
| Gemini 3.8 Live | 26/28 | 2.57 / 6.40 s | 22 | 0.052 |
| Gemini 3.1 Flash Live (preview) | 25/28 | 1.92 / 2.15 s | 36 | 0.045 |
| Gemini 2.5 Flash native audio | 23/28 | 3.31 / 4.51 s | 23 | 0.022 |
| *Eva's loop, Cerebras 27B, typed caller, silent voice* | 28/28 | 0.59 / 1.08 s to her first sentence | 68 | tokens only |

* Eva's number is not comparable: it starts after the caller's text and stops before the voice.
  Add the 0.5 s endpoint, Parakeet (0.2-1.1 s) and ElevenLabs v3 in the dispatcher's voice
  (`GZ4PpFJV8ikEGUtBrjK7`: 0.66 s warm, 3.95 s cold; warmed at session start now): ~1.9-2.8 s.
* What failed, word for word: gpt-realtime-mini didn't check the do-not-use broker and started
  negotiating with him; gpt-realtime-2.1-mini looked up "PO-771234" for 7781234 (a digit dropped)
  and so never said the load was late; Gemini 3.1 Flash Live said "our load number is 10037" with no
  booking (a new check catches a claimed booking without one); Gemini 2.5 booked with a made-up
  posting id, then didn't; Gemini 3.8 Live didn't tell the broker about the breakdown in its first
  answer; GPT-Live once gave no new delivery time.
* A tool bug found by the spoken runs: models asked for "PHX55120", "PHX 55120", "PAX 55,120" for
  PHX-55120 and the exact match found nothing (6 of 12 missed the breakdown load); the lookup now
  ignores spacing, dashes and case and falls back to the digits (`test_a_reference_is_found_however_it_was_heard`).
* Cost of this round, from the saved reports: about $5.90 OpenAI and $1.50 Google (a little more
  with the replaced reruns). GPT-Live bills $0.05 a minute of session plus the backend's tokens.

## The owner's first live sessions, and what they changed (2026-09-27)

* **gpt-realtime-2.1 "can't hear me at all"**: the driver called `Player.buffered_samples()` and
  `played_seconds()`, which are properties; the TypeError on OpenAI's first `speech_started` (his
  first word) killed the event loop, so nothing after her greeting was processed. The evals never
  had a player. Fixed; `test_barge_in_with_a_player_keeps_the_call_alive` fails on the old code.
  Checked first that it wasn't the audio: quiet synthetic speech at -26 and -38 dBFS through the live
  path (16 kHz mic -> 24 kHz) was heard and transcribed with near-field and far-field noise reduction
  alike. The live runner now prints the mic level after 3 s, "heard you start / you stopped", and
  writes every event to `bench/out/sessions/`; OpenAI's noise reduction is far-field (laptop mic).
  Re-run through `run_s2s.py`'s own code with a recorded caller: heard, looked up, answered in 1.1-1.2 s.
* **Gemini's voices "robotic"**: `bench/voices.py` rendered one dispatcher line in every voice: 30
  Gemini Live, 10 OpenAI realtime, 11 of 12 GPT-Live (`delta` gave only silence twice) for $0.45, in
  `samples/out/voices/`.
* **eva-v3 "a lot of fillers ... overdosed with scaffolding"**: work calls now get no 0.8 s filler,
  no backchannels and no sound tags, and what the 27B writes between lookups is held until the round
  ends and dropped when another lookup follows (`narration_dropped`). On three calls it still passes
  all checks, but its replies stay long: 44-108 words.
* **GPT-Live tools**: two prompts in OpenAI's recommended structure (`dispatcher_live.md` with the
  delegation policy, `dispatcher_backend.md` with the procedures) and a spoken "checking" when the
  backend is still busy 2 s after a hand-off (it had left a 12.6 s silence on a broker's counter-offer).
  Six spoken calls: 28/28, first word 1.16 s median, 3.38 s p90, $0.195 a call; the lowball call again
  after the check-in: 6/6, 0.87-2.03 s.
