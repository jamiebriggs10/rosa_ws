# 02 Options for each stage

Researched 8 Oct 2026. Package names and APIs move quickly: **check each one's current README before building its plug-in**, and correct this file if something has changed.

**Built 8 Oct 2026** (logs/2026-10-08-build.md). Every option below now exists and runs on the M5 Mac. Notes marked *Built:* record what turned out different from the plan. Versions are pinned in `requirements/`. No option pulls in PyTorch.

General finding: every engine writes numbers as digits ("5", "90") and often uses US spelling or abbreviations ("centimeters", "cm"); Whisper glues coordinates ("x10y-60z70"). Anything that reads the text (the benchmark scorer now, the robot console later) must normalise; see `bench/score.py`.

Names in `code` are the plug-in names used in `settings.yaml`.

---

## 1. Trigger: when to start listening

### `openwakeword`
- Open source (Apache 2.0), small ONNX models, runs on CPU on any machine.
- Comes with pre-trained phrases ("hey jarvis", "alexa", "hey mycroft"...). Start with `hey_jarvis` to get the plumbing working.
- **Custom "hey Rosa"**: trained from synthetic speech (Piper TTS clips + openWakeWord's background audio set + a small classifier). Roughly 1–2 hours. Phase 7 of the build plan.
- Knobs: `model` (name or path), `threshold` (0–1, default 0.5), `patience` (frames above threshold before firing), `cooldown_s`.
- Wants 80 ms frames (1280 samples): buffer internally.
- On macOS select the ONNX inference backend (the default tflite runtime is a Linux thing).
- *Built:* `stages/trigger/wake_word.py` (file not named `openwakeword.py`, so it can't shadow the library). openwakeword 0.6.0; `download_models()` into `models/openwakeword/` (8.8 MB, also fetches tflite copies it doesn't need). Patience and cooldown are our own code, counted in audio time. ~1.4 ms per 80 ms prediction. Synthetic "hey Jarvis" fired for all four `say` voices (peak score ≥ 0.996); no fires on commands, silence or noise.
- *Training (phase 7)* needs PyTorch, TensorFlow 2.8, Piper TTS and large downloads: see the phase 7 plan in the build log.

### `push_to_talk`
- Hold a button in the web page (mouse or space bar while the page has focus). Exact and free.
- A global hotkey (works when the page isn't focused) needs macOS Accessibility permission (e.g. `pynput`). Optional, later.
- A USB foot pedal or a cheap USB button usually shows up as a keyboard key, so it works through the same path.

### `both`
- Either fires. The sensible default for real use.

### Ruled out
- **Picovoice Porcupine**: free tier ended 30 June 2026; the access key is checked even offline; no non-commercial tier.
- **Always-on speech-to-text looking for the phrase**: works but burns GPU constantly and fights Ollama.

---

## 2. VAD: is this frame speech?

### `silero`
- The standard. Small ONNX model, CPU, ~1 ms per 32 ms frame. Wants exactly 512 samples at 16 kHz, which is why that is the pipeline frame size.
- Knobs: `threshold` (default 0.5), `min_speech_ms` (ignore blips shorter than this).
- Install: `silero-vad` package, or load the ONNX file with `onnxruntime` directly (lighter: avoids PyTorch). Prefer the ONNX route.
- *Built:* ONNX route. `silero_vad.onnx` pinned to tag v5.1.2 (2.2 MB, `models/silero/`). The v5 model needs the last **64 samples of the previous frame prepended** (input 576 samples) plus a (2,1,128) state. 0.08 ms per frame on the M5.

### `webrtc`
- Google's classic VAD (`webrtcvad-wheels`). Tiny, rule-based, no model. More false positives on fans and lab noise. Useful as the "simple baseline" in the comparison.
- Knobs: `aggressiveness` 0–3. Frames of 10/20/30 ms.
- *Built:* `speech_prob` = fraction of the 10/20/30 ms pieces in the frame that were speech (so 0, 0.5 or 1 with 30 ms pieces); knobs `frame_ms`, `threshold`, `min_speech_ms` added. `reset()` must make a new `Vad` (set_mode doesn't clear its noise estimate). 0.005 ms per frame. On low white noise it flagged 5 % of frames as speech; Silero 0 %.

---

## 3. End of turn: has the speaker finished?

This is the biggest single source of felt delay. Every option is measured on **endpoint delay** and **early cut-offs** (see `05-benchmarking.md`).

### `silence`
- Finished after `silence_ms` of no speech (VAD). Default 700 ms. Also `max_utterance_s` (default 15) as a safety cap.
- Simple, predictable; the baseline.
- *Built:* counts silence only after the first speech frame (leading silence never ends a turn). `max_utterance_s` is enforced by the pipeline, not the endpoint.

### `adaptive`
- Like `silence`, but looks at the partial transcript. If it ends in a way that suggests more is coming ("by", "to", "and", "then", a bare number with no unit, "point"), it waits `long_silence_ms` (e.g. 1500); otherwise `short_silence_ms` (e.g. 400).
- Needs a streaming STT engine for the partial text; with a batch engine it falls back to `silence`.
- Our own small rule set: a nice, explainable contribution for the dissertation.
- *Built:* rules in small named functions in `stages/endpoint/adaptive.py`; also waits long after a **bare number** ("5", "minus", "sixty" with no unit after it). With no partial text it uses `fallback_silence_ms` (700). Logs `last_reason`. First observation: "Move left." counts as complete, so a pause after a direction word is cut at the short timeout; a candidate extra rule.

### `smart_turn`
- Pipecat's Smart Turn v3: an open model that listens to the last up-to-8 s of audio (intonation as well as words) and says "complete" or "incomplete". Triggered when the VAD sees a short pause (~200 ms). Falls back to "complete" after `max_wait_ms` (default 3000) of silence.
- ONNX, CPU, under ~100 ms per check. Can be run with `onnxruntime` and a Whisper feature extractor, without installing all of Pipecat. Weights on Hugging Face (`pipecat-ai/smart-turn-v3`, check the exact name).
- Knobs: `pause_ms` (when to ask), `threshold`, `max_wait_ms`.
- *Built:* `smart-turn-v3.2-cpu.onnx` (8.7 MB int8, newest CPU file in `pipecat-ai/smart-turn-v3`). Input `input_features` (1, 80, 800), output already a probability. **No transformers needed at runtime:** `stages/endpoint/whisper_mel.py` is a numpy Whisper log-mel that matches `WhisperFeatureExtractor` to ~1e-7 (note `do_normalize` normalises the *waveform*, after zero-padding at the start to 8 s). ~9 ms per check. On synthetic `say` speech the model did **not** separate complete from incomplete phrases (TTS ends everything with a falling pitch): needs real speech before drawing conclusions.

### `ptt_release`
- Finished when the push-to-talk button is let go. Only valid with the `push_to_talk` trigger. Zero guessing; the reference point for the others.
- *Built:* also works with trigger `both`: an utterance the wake word started (button never pressed) falls back to `fallback_silence_ms` (700).

---

## 4. Speech-to-text

Every engine can run in `streaming` (partial text while you speak) or `batch` (transcribe once at the end) mode. Engines without native streaming do **pseudo-streaming**: re-transcribe the growing buffer every `partial_every_ms`. Compare modes as well as engines.

Common knobs: `model`, `mode`, `partial_every_ms`, `prompt` / `hotwords` (domain words: "tool, home, centimetres, millimetres, x, y, z, Rosa, joint, gripper"), `language: en`.

### `parakeet`
- NVIDIA Parakeet TDT 0.6B v3: very fast, strong English, punctuation and capitals included.
- Backends (same model):
  - `mlx`: `parakeet-mlx` on Apple silicon. Has a streaming API (check current README). Reported ~10–20 ms per 0.5 s chunk on M1/M2 Pro, final text ~0.2 s after a 5 s phrase.
  - `sherpa_onnx`: `sherpa-onnx` on CPU or CUDA; runs on both machines. Check that a v3 export exists; a v2 export is known.
  - `nemo`: NVIDIA NeMo on CUDA. Heavy (PyTorch): ask before installing. Lab PC only.
- First choice for phase 1.
- *Built:* `mlx` (parakeet-mlx 0.5.3, weights 2.3 GB in `models/parakeet-mlx/`) and `sherpa_onnx` (sherpa-onnx 1.13.8, **v3 int8 export exists**: `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8`, 642 MB unpacked). `nemo` gives a clear error (not installed).
  - API: `get_logmel(mx.array(float32 audio), model.preprocessor_config)` then cast to bfloat16 (bfloat16 audio breaks its FFT); `model.generate(mel)[0].text`. Native streaming: `model.transcribe_stream(context_size=(left, right))`, `add_audio`, `.result.text`.
  - Speed (synthetic, warm): mlx batch finish ~30 ms for a 2 s phrase; sherpa CPU ~80–120 ms.
  - **Native streaming was unreliable**: with small look-ahead it dropped or garbled words ("Move the the inspection plate", one setting returned nothing). Defaults now `context_right` 256, `stream_chunk_ms` 640, and `final_pass: true` (partials from the stream, final text from one batch pass, ~30–40 ms). `final_pass: false` gives pure streaming for the comparison.
  - Ignores `prompt` (sherpa hotwords need modified beam search + a BPE vocab; not tried). Heard "halt" as "Hold." for all four synthetic voices: matters for the stop shortcut.
  - The first utterance after loading can take ~300–400 ms (new audio length); later ones ~30 ms.

### `whisper_cpp`
- whisper.cpp: the C/C++ Whisper. Metal on Mac, CUDA on NVIDIA, same code, different build flag. Models `tiny.en` → `large-v3-turbo`.
- Two ways to drive it; pick one and record why:
  - Python bindings (`pywhispercpp`): in-process. Check the wheel was built with Metal.
  - `whisper-server` (ships with whisper.cpp) over local HTTP: build once per machine with the right flag, the plug-in just posts audio. Very portable.
- No native streaming: pseudo-streaming, or the `stream` example's chunked approach. Expect 0.5–2 s behind when streaming.
- Supports an initial `prompt` for domain words.
- *Built:* **bindings** chosen as default: `pywhispercpp` 1.5.1 wheel is already built with Metal (`MTL : EMBED_LIBRARY = 1`), 5–10 ms faster than the server (no HTTP hop), no process to manage. The `server` driver also works (Homebrew `whisper-cpp` 1.9.2, Metal; the plug-in starts `whisper-server` itself if nothing answers, log in `runs/whisper-server.log`); use it on the lab PC if building with CUDA once is easier. Models from `ggerganov/whisper.cpp` on HF into `models/whisper.cpp/`: tiny.en 78 MB (14 ms, misheard "go home"), base.en 148 MB (25 ms), small.en 488 MB (68 ms), times for a 1.1 s clip on Metal. Emits `[BLANK_AUDIO]` on silence: stripped. pywhispercpp rejects `initial_prompt=None`.

### `faster_whisper`
- CTranslate2 Whisper. CPU int8 on Mac (slower, ~3× real time with large-v3), CUDA float16 on NVIDIA (fast). Best on the lab PC.
- Built-in Silero VAD filter (turn it off: our pipeline does VAD).
- Same Whisper models, so same accuracy as whisper.cpp: the comparison is speed per platform.
- *Built:* `stages/stt/faster_whisper_engine.py`; faster-whisper 1.2.1, ctranslate2 4.8.2 (no Metal: runs on the CPU on the Mac). small.en (464 MB) int8, 4 threads: ~0.9 s for a 3.7 s clip. `hotwords` **does** exist in 1.2.1; we pass the prompt as `hotwords` *or* `initial_prompt` (knob `use_hotwords`), never both (they land in the same place). beam 5 was ~14 % slower than beam 1 with identical output on one clip; default beam 1.

### `moonshine`
- Moonshine Streaming (Useful Sensors / Moonshine AI, 2026): built for streaming on small devices, sliding-window attention so delay doesn't grow with phrase length. Tiny / small / medium sizes. ONNX, runs anywhere.
- Check the current package name (Moonshine Voice library, or the ONNX package) before building.
- *Built:* package **`moonshine-voice` 0.1.5** (`import moonshine_voice`), no torch. `get_model_for_language("en", ModelArch.SMALL_STREAMING, cache_root=models/moonshine)` → `Transcriber(model_path, model_arch, update_interval)`; batch `transcribe_without_streaming(list, 16000)`; streaming `create_stream()`, `add_listener`, `add_audio`, `stop()`. It splits speech into "lines" with its own detector; we join them. small-streaming 138 MB; batch finish ~60–130 ms; streaming ~1–2 ms per frame, finish 1–3 ms. English only; `prompt` ignored (it has `set_keyterms()`, untried).

### Possible later, not in scope now
- `apple_speech`: macOS on-device speech framework. Fast and free on the Mac but Swift-only and Mac-only: breaks the portability rule. Only as a reference number, if at all.
- WhisperKit (Swift, CoreML): same problem.
- Cloud engines (Deepgram, Wispr Flow...): against the local-only rule.

---

## Suggested first combination (phase 1)

`push_to_talk` + `silero` + `ptt_release` → `parakeet` (`mlx`, `batch`), then switch on streaming. Fewest moving parts, and it gives the reference numbers everything else is compared to.

*Built:* this is `presets/baseline.yaml`. Synthetic benchmark results for it and the other presets are in `logs/2026-10-08-build.md` (synthetic speech only: not real-speech results).
