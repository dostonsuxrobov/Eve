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
