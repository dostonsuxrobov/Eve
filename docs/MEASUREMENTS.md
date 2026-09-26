# Measurements (local build)

Every number of the local build, dated, with how it was taken. The cloud era's numbers and bug
hunts are in `.archive/docs/MEASUREMENTS.md`.

The machine: RTX 4050 laptop GPU, 6141 MiB; `nvidia-smi` reads 0 MiB used with nothing loaded.

## Small brains against the 4B (2026-09-25)

Question (owner): would a 1–2B brain leave more room on the GPU for an expressive voice without
losing the feel?

Method: `bench/brains.py`. Each model alone on the GPU through Ollama 0.34 at 127.0.0.1,
context 8192, the model's own sampling defaults, thinking off where the model has it. The
English Eva persona (about 2.5k tokens, written for the cloud-era 27B) with the stand-in user
Sam and his memory; the ten English example conversations (41 user turns, each model's own
replies fed back); then the six cloud-era tool cases with Eva's seven tools. One run each.
"On the GPU" is `nvidia-smi` with the model loaded: `ollama ps` leaves out the Qwen 3.5 vision
encoder (0.2–0.7 GB) and the ~0.1 GB CUDA context. "Flagged" is a turn with at least one rule
break a script can see (over 40 words, more than one question, emoji, `*action*`, `[tag]`,
list, a promised action with no tools, a phrase the persona bans, the same opening as the reply
before).

| model | params, quant | on the GPU | left for a voice | first reply | to first word, median | decode | words/reply, median | flagged | tool probe |
|---|---|---|---|---|---|---|---|---|---|
| `qwen3:4b-instruct-2507-q4_K_M` (today's) | 4.0B Q4_K_M | 3255 MiB | 2.8 GB | 0.71 s | 0.07 s | 56 tok/s | 18 | 7/41 | 4/6 |
| `qwen3.5:2b-q4_K_M` | 2.3B Q4_K_M | 2411 MiB | 3.6 GB | 0.44 s | 0.08 s | 99 tok/s | 51 | 34/41 | 3/6 |
| `openbmb/minicpm5-2b` | 2.5B Q4_K_M | 1737 MiB | 4.3 GB | 0.45 s | 0.06 s | 92 tok/s | 50 | 28/41 | 3/6 |
| `LiquidAI/lfm2.5-1.2b-instruct:q8_0` | 1.2B Q8_0 | 1465 MiB | 4.6 GB | 0.25 s | 0.04 s | 123 tok/s | 23 | 19/41 | 2/6 |
| `openbmb/minicpm5:q8_0` (the 1B) | 1.1B Q8_0 | 1183 MiB | 4.8 GB | 0.19 s | 0.04 s | 142 tok/s | 16 | 19/41 | 4/6 |
| `gemma3:1b-it-qat` | 1.0B Q4_0 | 1197 MiB | 4.8 GB | 0.30 s | 0.08 s | 123 tok/s | 17 | 15/41 | no tools |
| `qwen3.5:0.8b` | 0.9B Q8_0 | 1417 MiB | 4.6 GB | 0.30 s | 0.08 s | 132 tok/s | 52 | 29/41 | 2/6 |

"First reply" reads the whole persona; later turns reuse it from Ollama's prompt cache, so the
time to the first word is under a tenth of a second for every model. Speed is not what
separates them: even the 4B decodes about 17 times faster than she speaks. Load time 2.8–5.1 s
for all.

What the transcripts show (`bench/out/brains/`, not committed; quotes verbatim):

* **4B:** short and dry, reacts before it asks ("okay, so you do. That's enough." to "I guess I
  do care a bit"). Still invents shared memories ("Miso's been sleeping on the windowsill again,
  hasn't he?"), answers the late-night sick cat with "Call the vet again." and "One sec. I've got
  it.", and writes the stomach-bug lie. Tools: says "One sec, setting that timer" with no call;
  writes `end_conversation{...}` as text.
* **Qwen 3.5 2B:** fluent but long (27 of 41 replies over 40 words) and gets facts wrong ("your
  manager canceled a six-month project", "I know how you feel").
* **MiniCPM5 2B:** long, and loses the thread ("Never again leave Sam and Miso behind. Get into
  the car now"; `[small pause]` as a whole reply).
* **LFM2.5 1.2B:** help-desk register ("You're not alone in this", "you've got this"); invents
  the weather ("It's a bit cloudy today") and a quiet Miso; tells the sick-cat owner to try broth.
* **MiniCPM5 1B:** the best small tool caller (timer 480 s and the note, both right) but answers
  "mm" to every line of two whole scenarios, including "Are you a real person?".
* **Gemma 3 1B:** wraps replies in quotes, writes "(Pause, a slight shift in tone)", says
  "it's 911, let me dial" and offers "a few options" for the lie. No tool support in Ollama.
* **Qwen 3.5 0.8B:** mostly incoherent ("Miso is getting better soon; that's good news").

Failures every model shares, the 4B included: invented memories, writing the lie, and no clean
"call the emergency line now" for the sick cat. They don't go away with size in this range, so
they are scaffolding and fine-tuning targets whatever the brain.

Not tested: the persona is one long rule-heavy prompt written for a 27B. A short persona may suit
the 1–2B models better; that would be the next brain experiment if a small brain is wanted.

## Expressive local voices (2026-09-25)

Owner's direction: small brains (1-2B) to leave room on the GPU for expressive voices, and
variants to talk to. Every figure below is from this laptop, one run unless a range is given.

**Orpheus 3B** (Canopy Labs, `legraphista/Orpheus:3b-ft-q4_k_m` in Ollama, SNAC 24 kHz decoder):

* 2.12 GB in Ollama at a 2k context. Generates 70.6-71.0 tok/s alone on the GPU; real time needs
  82 (7 codes per 85.3 ms frame), so **0.86x real time**. The GPU runs flat out (2565 MHz, 74 W of
  93, memory controller 98-100 %): it is bandwidth-bound. Unsloth's dynamic 3-bit build
  (`UD-Q3_K_XL`, 1.87 GB) gave 73.4-73.9 tok/s, 0.90x: the model reads its 156k-token output layer
  at full size every token either way.
* SNAC decoding on the GPU takes 4.7 ms a window, but from the voice server's process it fights
  Ollama's for the card: generation fell to 49 tok/s decoding every frame, 63 decoding every 4
  frames. On the CPU (8 threads, 45 ms for a 7-frame window, 13 % of real time) generation stays at
  70-72 tok/s. Decoder on the CPU, every 4 frames.
* Every clip opens with ~0.55 s of silence at -52 dBFS and closes with ~0.5 s. Trimmed at both ends
  (`EdgeTrim`), and the opening silence is prefilled as 5 frames in the prompt instead of generated:
  first voiced frame 0.22-0.36 s after the request against 0.57-1.25 s (3 takes each).
* Streaming: playback starts after a pre-roll of about 14 % of the clip's estimated length, so it
  never runs dry at 0.86x. First audio 1.1 s for short lines, up to ~2.7 s for a long sentence.
* Without `stop: <custom_token_2>` the model ran on after its line into a new utterance in another
  voice ("julia: I thought a quiet day at home...").

**Chatterbox Turbo** (Resemble, 350M, cloning `eva/assets/voices/eva.wav`): 3.03 GB PyTorch peak,
3.07 GB reserved; 2.2-2.6x real time (0.83 s for 1.8 s of speech), whole sentences (no streaming).
With three stale Ollama models loaded it ran at **0.05x** real time: the card had spilled into
system RAM. Load 16 s.

**Chatterbox** (0.5B, `exaggeration` / `cfg_weight` per request): 3.44 GB peak, 3.59 GB reserved;
0.98-1.19x real time (2.37 s for 2.32 s of speech at exaggeration 0.5; 4.14 s for 4.92 s at 0.9).
Load 16.9 s.

**End to end** (`bench/e2e_sim.py`, three recorded utterances through the real STT, brain and
voice into a silent player; "total" is the user's last word to her first audio):

| variant | stt | brain first token | voice first audio | total |
|---|---|---|---|---|
| minicpm1b + kokoro | 0.24-0.55 s | 0.24-0.42 s | 0.68-1.05 s | 1.47-1.70 s |
| qwen2b + orpheus-tara (before prefill / CPU decoder) | 0.23-0.53 s | 0.35-0.83 s | 1.13-3.25 s | 2.26-4.02 s |
| minicpm1b + orpheus-tara (prefill, CPU decoder, tool gate) | 0.23-0.54 s | 0.23-0.28 s | 1.21-2.59 s | 1.71-3.13 s |

In the first minicpm1b run the brain, offered every tool, checked the clock after "how's it
going?" and ended the call in the middle of "today was rough"; the tool gate (`eva/toolgate.py`)
now offers only the tools the user's line points at (test `zc_tool_gate`, which fails on the first
version: a spoken line keeps its transcript on the turn's metrics).

Ideas not yet tried for Orpheus' speed: prune its output layer to the ~28k audio tokens (most of
the per-token read), a smaller Orpheus-style model, or llama.cpp's own server with CUDA graphs.

## Live sessions with Chatterbox, and the brain question (2026-09-26)

From `voice/server.log` of the owner's own sessions (one line per sentence she spoke):

| voice | sentences | first audio per sentence, median (min-max) | sentence length, median | x real time, median |
|---|---|---|---|---|
| chatterbox (0.5B) | 59 | 2.49 s (1.22-6.99) | 2.2 s | 0.93 |
| chatterbox-turbo | 25 | 0.70 s (0.40-1.66) | 1.6 s | 2.29 |
| orpheus | 55 | 1.48 s (0.68-3.05) | 3.2 s | 0.73 |

Chatterbox renders a whole sentence before any of it plays, so a long sentence waits: 6-7 s
before the three longest. The small brain wrote a mood cue in 4 of 62 sentences, so Chatterbox
ran at its default emotion strength almost all the time.

What the 1B said that wasn't true, in the owner's sessions: "I'm also working on creating an AI
agent named Eva", "I have been working on the Skynet project as well". Both are facts about the
*owner* in `memory.json`, stored without a subject ("Is working on a project called Skynet...");
the 1B read them as its own. The same file holds facts that are not true (speaks Dutch,
Portuguese, Ukrainian: likely misheard languages from the cloud era; a sister named Priya: the
eval's stand-in user). Plus help-desk lines ("I can help with tasks like setting timers...") and
"I'm glad you asked!" three times.

The cloud 27B on the Chatterbox prompt, one call: first token 0.72 s, opened with `[amused]`, and
used the memory as the owner's ("Which model did you end up going with, the smaller one or the
big one?").

## Scaffolding for a 1B brain (2026-09-26)

The owner's session with MiniCPM5 1B + Chatterbox went wrong where an earlier one hadn't: "Hello,
I'm Doston", the same two sentences in every reply (once a whole earlier reply again, 18.7 s),
"I need to call the get_weather function", a tool call spoken as markup, the weather for New York.
Nothing had changed in the code or the memory between the two sessions: the 1B varies from session
to session, and one bad reply poisons the rest because it copies its own history.

Method: `bench/replay.py`, the owner's six lines plus the greeting through the real loop and the
real 1B (a copy of the real memory, stand-in tool results, silent voice), 8 sessions per row.
Counted on what she would have *said*:

| version | flagged replies | identity | repeat | markup | tool talk | promise | wrong city | real weather calls | help-desk lines | words/reply |
|---|---|---|---|---|---|---|---|---|---|---|
| before | 18/56 | 1 | 11 | 1 | 1 | 6 | 3 | 6/8 | 6 | 19 |
| speech guard + prompt fixes | 1/56 | 0 | 0 | 0 | 0 | 1 | 0 | 2/8 | 10 | 13.5 |
| + the loop answers plain questions itself | 0/56 | 0 | 0 | 0 | 0 | 0 | 0 | 8/8 | 11 | 12 |

Sessions before the fixes ranged from 0/7 to 5/7 flagged: the owner's "fine last time, broken
this time" is that spread.

What changed: (1) `eva/guard.py` checks every sentence before it is spoken and drops markup, tool
names, "I'm <user>", claiming the user's work, promises with no tool offered, a shared past not in
memory, thinking out loud, and repeats; dropped sentences stay out of the history, and a reply that
loses everything is asked for once more with a nudge (11 sentences dropped, 2 retries, 1 fallback
"I lost my train of thought" in the last row). (2) Prompt: memory lines carry the user's name
("Doston: is working on..."), the greeting says "You are Eva", and small brains get a tool note
without "MUST include the function call". (3) `eva/toolgate.py`: a weather or time question is
answered by the loop calling the tool itself before the brain speaks (city from the line, else the
home city from memory); a wrong city is corrected. The second row shows why (3) is needed: with
the softer tool note the 1B called the weather tool in 2 of 8 sessions and invented weather in
others ("it's a sunny Saturday morning" against drizzle and 81 % rain).

What it does not fix, from the same transcripts: coherence ("The tree is bigger than the house.",
"It's really important to protect the tree, so I'm not sure if you'll be around."), help-desk
lines (6 -> 11 of 56, up with the new tool note), and flat replies ("I'm not sure.", "I don't know.").
The cloud 27B on the same voice was coherent throughout the owner's session. Brain size decides the
sense; scaffolding decides whether what she says is checked against what she knows.

## "Slower and dumber" (2026-09-26, afternoon)

The owner's next two sessions (1B and cloud 27B, both on Chatterbox) felt slower and dumber.
From `voice/server.log`: Chatterbox ran at **0.70-0.73x real time** (1.39-1.41 s of work per second
of speech) against 0.88-0.99x in the five earlier sessions; first audio median 3.2-3.4 s against
2.1-2.7 s. Same code, same voice. The laptop was **on battery**: Windows capped the GPU at 50 W
(101 W on AC). Two `tail -f | grep` pipelines left over from this session's monitoring had also
been spinning since the day before (about 227 s of CPU each); stopped. `run.py` now warns at
startup when on battery (`eva/gpu.py on_battery`).

Mine: the router looked up the weather for mentions, not requests ("Raining outside. I love it.",
"What do you know about the rain?", "the difference between the weather and the climate"), ~0.9 s
each and a forecast in the reply. It now fires only on requests (`ROUTES` in `eva/toolgate.py`;
the owner's lines are the test).

"Too short" (owner): the persona asked for one to three sentences under forty words and called a
two-word reply "often plenty"; the small one under thirty. Now two to four sentences, a real answer
to a real question up to about sixty words (small persona: two or three, under fifty). Replay, 4
sessions of the 1B: 0/28 flagged, weather looked up 4/4, words per reply median 25 (was 12).
