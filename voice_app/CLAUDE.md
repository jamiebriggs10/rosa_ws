# Project context for Claude Code

Read this first in every session. It is loaded automatically.

## What this is

The **voice app** of Jamie Briggs's 4th-year dissertation at Strathclyde (project 6074, natural-language and speech control of a UR12e cobot). It lives inside the console's repository (`rosa_ws/voice_app/`, since 2026-10-28) but runs as its own process on port 8100, with its own venv and `settings.yaml`.

It listens for a wake word (or a push-to-talk press), captures what is said, works out when the speaker has finished, turns the speech into text and streams it to the console (`../voice.py` is the console's end). It also speaks the lines the console composes (Piper, `speech.py`) and plays the loop's sounds, muting its own microphone meanwhile.

It is also a **test bench**: every stage has several interchangeable options, chosen in a settings page, so they can be compared fairly for the dissertation (latency, accuracy, false triggers).

## Hard rules

1. **Two processes, one repo.** The console (`..`) never imports this app's libraries and this app never imports the console; they talk over HTTP only. Keep this venv separate from the console's; never add this app's packages to `../requirements.txt`.
2. **Local only.** No cloud speech or LLM APIs, no API keys.
3. **Portable.** Runs on Jamie's M5 MacBook (Apple silicon) now; must later run on a Linux PC with an NVIDIA GPU with only a settings change. No Mac-only code outside a plug-in that is clearly marked as Mac-only. See `docs/03-portability.md`.
4. **Small, readable code.** Jamie will explain every line in a dissertation. Prefer plain Python, short files, comments that say *why*. No frameworks beyond Flask.
5. **Ask before** adding a heavy dependency (anything that pulls in PyTorch, NeMo, a compiler toolchain) or changing the architecture in `docs/01-architecture.md`.
6. **Say clearly when Jamie must do something physical** (allow microphone access, speak a phrase, plug in a mic) and wait.
7. **Measure, don't guess.** Every stage reports timestamps. Never claim something is fast without a number from `runs/`.
8. **Write down what you learn.** End every working session with a file in `logs/` (template in `logs/README.md`). Record failures too.

## Where things are

| Path | What |
|---|---|
| `briefs/` | What to build next and how we'll know it worked. **Start with `briefs/2026-10-08-build-plan.md`.** |
| `docs/` | Durable knowledge: architecture, options, portability, settings/UI, benchmarking, later integration, glossary |
| `logs/` | What actually happened, one file per session |
| `settings.yaml` | The current choice for every stage, plus each option's knobs |
| `requirements/` | `base.txt` (everywhere), `mac.txt`, `nvidia.txt` |
| `audio/` | Audio sources: microphone and WAV file (same interface) |
| `stages/` | The four stages; `stages/base.py` is the contract every option implements |
| `pipeline.py` | The state machine that drives the stages |
| `device.py`, `config.py`, `models.py` | Device detection, settings files, model downloads into `models/` |
| `cli.py`, `mic_check.py` | Terminal runner (mic or WAV), phase 0 mic check |
| `tests/` | pytest suite: `./.venv/bin/python -m pytest` (`-m "not model"` skips the model-loading tests) |
| `app.py`, `static/` | Flask settings page and live view |
| `bench/` | Recording the test set and replaying it through any settings combination |
| `training/` | Wake-word training notebook for Google Colab (phase 7, "Truffles") |
| `runs/` | JSONL logs of every utterance, live or benchmark |
| `tests/` | pytest tests, including synthetic-speech WAVs made with `say` |

## Machines

- **Now:** M5 MacBook Pro, macOS, Apple silicon. Ollama also runs here (the robot console's model), so the GPU is shared. Run the app in a venv inside this folder (`.venv/`); `./.venv/bin/python -m pytest -m "not model"` is the fast test run.
- **The lab PC:** Linux, NVIDIA GPU, also runs ROS, the LLM and the console; `../start.sh` starts this app there (`voice_start`). Setup: `../../docs/06-lab-pc-setup.md`, section 5b.
- The ZenBook (ROS laptop) is not involved in this prototype.

## Working style

- Explain as you go, in plain language. Define a term the first time it appears (VAD, endpointing, WER...).
- Build all phases of the brief in one run, in order, testing each phase against its "done when" before moving on.
- Test without Jamie where possible: generate speech with macOS `say` (e.g. `say -v Daniel -o x.aiff "move up five centimetres"`, then convert to 16 kHz WAV) and push it through `FileSource`. Keep automated tests in `tests/` (pytest). Only stop for Jamie when a step truly needs him (mic permission, live speech, a heavy install).
- After any change to the running app, restart it and check it, before saying it works.
- Check what is installed before installing (`python3 --version`, `which ffmpeg`, `brew list`).
