# Brief 2026-10-08: build the voice control prototype

**Goal:** say a command (after a wake word or holding a button), see the text on screen with as little delay as possible, with every stage swappable from a settings page and every utterance timed, so options can be compared for the dissertation.

**Read first:** `CLAUDE.md`, then `docs/01-architecture.md` and `docs/02-options.md`. The others when their phase comes up.

**Not in scope:** anything in `../fyp-cobot/` (don't touch it), sending to the robot, text-to-speech, the stop shortcut.

Build all phases in one run, in order. Test each one against its "done when" before moving on (automated with synthetic speech from macOS `say` where possible; see CLAUDE.md). Each ends with something Jamie can run and see. Write a log at the end of each working session. Ask before heavy dependencies.

---

## Phase 0: environment and microphone
- Check Python (`python3.11`; install with Homebrew if missing, ask first), create `.venv/`, install `requirements/base.txt`.
- `python -m sounddevice` to list devices. Jamie allows microphone access when macOS asks.
- Tiny script: record 3 s, save a WAV, play it back.
- `device.py`: detect cuda / metal / cpu.

**Done when:** Jamie hears his own recording back, and `device.py` prints `metal`.

## Phase 1: the thinnest working pipeline (terminal only)
- `audio/`: `MicSource` and `FileSource` (16 kHz, 512-sample frames, resampling, level).
- `pipeline.py`: the state machine, pre-roll buffer, timestamps, JSONL logging to `runs/`.
- Options: trigger `push_to_talk` (Enter key in the terminal for now), VAD `silero` (ONNX), endpoint `ptt_release`, STT `parakeet` (`mlx`, batch).
- `settings.yaml` loaded; plug-ins looked up by name.

**Done when:** press Enter, say "move up five centimetres", press Enter, the text prints with its timings, a line appears in `runs/`. The same WAV through `FileSource` gives the same text.

## Phase 2: the web page
- `app.py` (Flask, port 8100) with the endpoints in `docs/04-settings-and-ui.md`; SSE events.
- `static/index.html`: state badge, level meter, hold-to-talk button (and space bar), live transcript, timing strip, settings panel driven by `/options`, session stats.
- Settings changes saved to `settings.yaml`, applied between utterances, models pre-loaded.

**Done when:** everything in phase 1 works from the browser, and switching an option in the page takes effect on the next utterance.

## Phase 3: hands-free
- Trigger `openwakeword` (pre-trained `hey_jarvis` for now) and `both`.
- Endpoint `silence`.
- VAD `webrtc` as the alternative.
- `armed_timeout_s` behaviour.

**Done when:** "hey Jarvis, go home" with no hands shows "go home"; Jamie can switch silence_ms in the page and feel the difference.

## Phase 4: streaming and smarter endpointing
- Streaming mode for Parakeet (native) with partial text on the page.
- Pseudo-streaming for batch engines (`partial_every_ms`).
- Endpoint `adaptive` (needs partial text) and `smart_turn` (ONNX; ask before adding its dependencies).

**Done when:** partial text appears while speaking; all four endpoint options selectable and logged.

## Phase 5: more speech-to-text engines
- `whisper_cpp` (decide bindings vs server, record why), `faster_whisper` (CPU on Mac), `moonshine`.
- `parakeet` `sherpa_onnx` backend (the one that will also run on NVIDIA).
- Domain prompt / hotwords where the engine supports it.

**Done when:** every engine runs from the page on the Mac; options it can't run are greyed with a reason.

## Phase 6: benchmark
- Before Jamie records: build a **synthetic test set** from `commands.csv` with several `say` voices (`bench/make_synthetic.py`, clips in `bench/testset/synthetic/`), so the whole benchmark can be built and tested now. Results on synthetic speech are labelled as such and never reported as real-speech results.
- `bench/record_set.py`: Jamie records `bench/testset/commands.csv` (Jamie must speak: plan a 30-minute session).
- `bench/run_bench.py` and `bench/summarise.py` with the metrics in `docs/05-benchmarking.md`, including command accuracy with number normalisation.
- Presets: `baseline`, plus a few for the experiments listed in `05`.

**Done when:** one command produces a comparison table across at least three presets, and the numbers are written up in a log.

## Phase 7: custom wake word
- Train "hey Rosa" with openWakeWord's training process (synthetic clips). Will likely need more disk and time: plan it, ask first.
- Benchmark against `hey_jarvis`: detection rate vs false wake-ups per hour.

**Done when:** "hey Rosa" works live and its numbers are in the results.

---

## How we'll know the whole thing worked
- Felt delay (end of speech → final text) median under ~1 s with the best settings on the Mac, measured not guessed.
- Command accuracy over 95 % on the test set with the best engine.
- A results table per experiment in `05`, reproducible from a preset and the recorded clips.
- Moving to the lab PC needs only `requirements/nvidia.txt` and `device: auto`.
