# Eva

A realistic voice agent. The brain and the ears run on this laptop (Qwen3 4B on Ollama,
Parakeet); the voice is ElevenLabs, with a local voice (Kokoro) as the automatic fallback.
`CLAUDE.md` is the project brief, `docs/MEASUREMENTS.md` has the numbers. Earlier versions are
kept in `.archive/` (the cloud era) and `.archive/local-variants/` (the fully local experiments).

## Talk to her

```
.venv\Scripts\python.exe run.py --user-name Doston                  # pick a voice by number
.venv\Scripts\python.exe run.py --voice v3 --user-name Doston       # ElevenLabs v3 (the default)
.venv\Scripts\python.exe run.py --voice v3conv --user-name Doston   # v3 Conversational: faster, half the credits
.venv\Scripts\python.exe run.py --voice flash --user-name Doston    # Flash v2.5: the fastest, half the credits
.venv\Scripts\python.exe run.py --voice kokoro --user-name Doston   # local, no credits
.venv\Scripts\python.exe run.py --text                              # type instead of talk (she still speaks)
.venv\Scripts\python.exe run.py --web --tls                         # from your phone: https://<laptop-ip>:8443
```

After every reply a dim line shows the credits it cost and where the month stands (121k a month
on the Creator plan, renewing on the 28th); the session ends with the total, saved in
`usage.json`. Headphones make interrupting her reliable. Ctrl-C ends the session (she updates
her memory of you first).

## Measure

```
.venv\Scripts\python.exe -m pytest tests -q                        # offline tests, ~3.5 min
.venv\Scripts\python.exe bench\replay.py                           # your lines through the real brain, silent: no credits
.venv\Scripts\python.exe bench\e2e_sim.py --voice v3               # recorded speech through everything: a few hundred credits
```

## Set up on a new machine

1. `uv venv .venv --python 3.13`, then
   `uv pip install --python .venv\Scripts\python.exe numpy sounddevice soundfile httpx rich onnxruntime kokoro-onnx sherpa-onnx pytest`
   (the Windows Store Python can't open the mic). Parakeet, Kokoro and the Silero VAD download
   themselves into `models\` on first use.
2. Ollama, called at 127.0.0.1: `ollama pull qwen3:4b-instruct-2507-q4_K_M`.
3. The ElevenLabs API key in `elevenlabs_key.txt` (gitignored) or `ELEVENLABS_API_KEY`.
