# Voice Control Prototype

Wake word or push-to-talk → listen → detect end of speech → speech-to-text → show the command on screen.

Part of Jamie Briggs's dissertation (project 6074, speech control of a UR12e cobot). The robot console lives one level up (`rosa_ws/`); this folder moved into its repository on 2026-10-28 and is **not** touched from here. This folder is a self-contained test bench for the voice front end: every stage has swappable options so they can be compared on latency and accuracy, first on the Mac, later on an NVIDIA Linux PC.

## The pipeline

```
 mic / WAV file
      │  16 kHz mono, 32 ms frames
      ▼
 ┌──────────┐   ┌─────────────┐   ┌─────────────┐   ┌────────────────┐   ┌─────────┐
 │ Trigger  │──▶│ VAD         │──▶│ End of turn │──▶│ Speech-to-text │──▶│ Output  │
 │ wake/PTT │   │ speech?     │   │ finished?   │   │ words          │   │ screen  │
 └──────────┘   └─────────────┘   └─────────────┘   └────────────────┘   └─────────┘
```

| Stage | Options |
|---|---|
| Trigger | openWakeWord, push-to-talk, both |
| VAD | Silero, WebRTC |
| End of turn | silence timeout, adaptive timeout, Smart Turn v3, push-to-talk release |
| Speech-to-text | Parakeet (MLX / sherpa-onnx), whisper.cpp, faster-whisper, Moonshine; streaming or batch |

## Run

Setup (once):
```bash
brew install python@3.11                             # if needed
python3.11 -m venv .venv
./.venv/bin/pip install -r requirements/mac.txt      # or requirements/nvidia.txt on the lab PC
```

Optional: `brew install whisper-cpp` for the whisper_cpp `server` driver (the default `bindings` driver needs nothing extra). Models download into `models/` on first use (~4.3 GB for every option).

Check the mic (records 3 s, saves `runs/mic_check.wav`, plays it back; allow microphone access when macOS asks):
```bash
./.venv/bin/python mic_check.py
```

Terminal (Enter to start/stop):
```bash
./.venv/bin/python cli.py                            # microphone
./.venv/bin/python cli.py --file clip.wav            # or play a WAV file
./.venv/bin/python cli.py --settings presets/baseline.yaml
```

Web page (then open http://localhost:8100):
```bash
./.venv/bin/python app.py                            # microphone
./.venv/bin/python app.py --file x.wav               # or loop a WAV at real speed
```

Tests (all ~1 min and loads models; `-m "not model"` ~10 s):
```bash
./.venv/bin/python -m pytest                         # all
./.venv/bin/python -m pytest -m "not model"          # fast
```

Benchmark:
```bash
./.venv/bin/python bench/make_synthetic.py           # synthetic set with macOS `say` (Mac-only, ~5 min)
./.venv/bin/python bench/run_bench.py --settings presets/baseline.yaml --set synthetic
./.venv/bin/python bench/summarise.py runs/*-synthetic.jsonl --out results/baseline
```

Recorded clips: `bench/record_set.py` creates `bench/testset/clips/`; then `run_bench.py --set recorded`.

## Speaking back

The app can also talk and beep, for the robot console's hands-free mode. The console
composes the words (it knows the facts) and posts them; the app only plays them:

| Route | Does |
|---|---|
| `POST /say {"text": ...}` | speaks the line with Piper; the microphone is muted meanwhile so the app never hears itself; `{"cancel": true}` hushes |
| `POST /play {"name": ...}` | one of the state sounds: `listen`, `heard`, `ask`, `go`, `done`, `cancel`, `refused`, `think` |
| `POST /followup {"seconds": N}` | for N seconds, speech counts without the wake word (the console has just asked something) |

`/status` reports `speaking`, `followup_s` and the `speech` block; `/events` carries
`speaking` and `followup`. Settings: the `speech:` and `sounds:` blocks in `settings.yaml`.
Fixed phrases are cached under `cache/say/`.

Voices: `python -m piper.download_voices en_GB-alan-medium --data-dir models/piper`
(the British voices are alan, alba, aru, cori, jenny_dioco, northern_english_male, semaine,
southern_english_female, vctk). Piper is `piper-tts` from OHF-Voice/piper1-gpl, GPL-3.0; the
original rhasspy/piper (MIT) was archived in October 2025. It works on the Mac in this
Python 3.11 venv (checked 2026-10-28; an earlier try with the system Python 3.10 could not
find its espeak data). Fetch a voice once: `./.venv/bin/python -m piper.download_voices
en_GB-alan-medium --data-dir models/piper`; without one the app stays silent and says why
in `/status`.

## Docs

- `docs/01-architecture.md`: how it fits together, the state machine, the stage contract
- `docs/02-options.md`: every option for every stage, its knobs and how to install it
- `docs/03-portability.md`: Mac now, NVIDIA Linux later
- `docs/04-settings-and-ui.md`: `settings.yaml` and the web page
- `docs/05-benchmarking.md`: test set, metrics, log format
- `docs/06-integration-later.md`: how this will connect to the robot console (not now)
- `docs/glossary.md`: the jargon in plain English

## Way of working

Same as `fyp-cobot`: `briefs/` before, `docs/` for things that stay true, `logs/` after. Start with `briefs/2026-10-08-build-plan.md`.
