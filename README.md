# Eva

A realistic voice agent that runs entirely on one laptop: open models, no cloud services.
You talk, she listens while you speak, answers in about a second or two, can be interrupted,
remembers you and does small things (timers, notes, the time).

`CLAUDE.md` is the project brief, `docs/MEASUREMENTS.md` has every number. The earlier version
(cloud voice and brain) is kept in `.archive/`.

## Talk to her

```
.venv\Scripts\python.exe run.py --user-name Doston          # pick a brain and a voice by number
.venv\Scripts\python.exe run.py --brain minicpm1b --voice orpheus-tara --user-name Doston
.venv\Scripts\python.exe run.py --list                      # every brain and voice, with sizes
.venv\Scripts\python.exe run.py --text                      # type instead of talk (she still speaks)
```

A variant is one **brain** (an Ollama model) and one **voice**. Headphones make interrupting
her reliable; on the laptop speakers the echo guard asks for real words before she stops.
Ctrl-C ends the session (she updates her memory of you first).

* Brains: `qwen4b` (the 4B), `qwen2b`, `minicpm2b`, `lfm1b`, `minicpm1b`, `gemma1b`, `qwen08b`.
  The 1-2B ones get a short persona (`eva_small`) and only the tools your words point at.
* Voices: `kokoro` (fast, flat, CPU), `orpheus-tara` / `-leah` / `-jess` / `-mia` / `-zoe`
  (Orpheus 3B: laughs, sighs, gasps), `chatterbox-turbo` (cloned voice, laughs),
  `chatterbox` (cloned voice, emotion strength follows her mood cue).
* The GPU has 6 GB. Brain + voice must fit: `run.py` unloads what the variant doesn't use and
  warns when the card is nearly full. The Chatterbox voices fit only with the 1B brains.

## Hear the voices side by side

```
.venv\Scripts\python.exe bench\voices.py        # the same six emotional lines in every voice
```

Files and timings land in `bench\out\voices\` (`listening.md` is the index).

## Measure

```
.venv\Scripts\python.exe -m pytest tests -q                            # offline tests, ~3.5 min
.venv\Scripts\python.exe bench\e2e_sim.py --brain minicpm1b --voice orpheus-tara   # recorded voice through the real stack, silent
.venv\Scripts\python.exe bench\brains.py                               # every brain on the example conversations
.venv\Scripts\python.exe bench\chat.py 4 --name Doston                 # type to one brain, speed after each reply
```

## Set up on a new machine

1. Python venv (uv, CPython 3.13; the Windows Store Python can't open the mic):
   `uv venv .venv --python 3.13` then
   `uv pip install --python .venv\Scripts\python.exe numpy sounddevice soundfile httpx rich onnxruntime kokoro-onnx sherpa-onnx pytest`.
   Parakeet, Kokoro and the Silero VAD download themselves into `models\` on first use.
2. Ollama (call it at 127.0.0.1): `ollama pull` the brains in `eva/config.py` and
   `legraphista/Orpheus:3b-ft-q4_k_m`.
3. The voice environment for Orpheus' decoder and Chatterbox (PyTorch with CUDA):
   ```
   uv venv .venv-voice --python 3.13
   uv pip install --python .venv-voice\Scripts\python.exe torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
   uv pip install --python .venv-voice\Scripts\python.exe chatterbox-tts snac
   ```
   `run.py` starts `voice\server.py` in it when a voice needs it; its log is `voice\server.log`.
