# 03 Portability: Mac now, NVIDIA Linux later

## The rule

Each plug-in hides its hardware. The rest of the code never asks "am I on a Mac?". Settings pick a **device**:

| `device` | Means |
|---|---|
| `auto` | pick the best available: `cuda` if an NVIDIA GPU is found, else `mps`/`metal` on Apple silicon, else `cpu` |
| `cuda` | NVIDIA GPU |
| `metal` | Apple GPU (Metal / MLX) |
| `cpu` | anywhere |

`device.py` (small, one function) detects what's available. Each plug-in's `supports(device)` says whether it can run there, and its `load()` chooses its backend from the device. The settings page greys out options the current machine can't run.

## Option by option

| Option | Mac (now) | NVIDIA Linux (later) | Code change |
|---|---|---|---|
| Mic capture (`sounddevice`/PortAudio) | yes | yes | none |
| `openwakeword` | ONNX, CPU | ONNX, CPU | none |
| `push_to_talk` (page button) | yes | yes | none |
| `silero`, `webrtc` | CPU | CPU | none |
| `silence`, `adaptive` | pure Python | pure Python | none |
| `smart_turn` | ONNX, CPU | ONNX, CPU or CUDA | none |
| `parakeet` | `mlx` backend | `sherpa_onnx` (CUDA) or `nemo` | backend chosen inside the plug-in |
| `whisper_cpp` | built with Metal | built with CUDA | build flag only |
| `faster_whisper` | CPU int8 | CUDA float16 | `compute_type` from device |
| `moonshine` | ONNX | ONNX (CUDA provider) | none |

Only Parakeet needs two genuinely different libraries. Keep both behind the one `parakeet` plug-in.

## Requirements files

- `requirements/base.txt`: everything that runs everywhere (Flask, numpy, sounddevice, onnxruntime, openwakeword, PyYAML, jiwer...).
- `requirements/mac.txt`: `-r base.txt` plus Apple-only packages (`parakeet-mlx`, `mlx`).
- `requirements/nvidia.txt`: `-r base.txt` plus CUDA packages (`onnxruntime-gpu` instead of `onnxruntime`, `faster-whisper` with CUDA, `sherpa-onnx` CUDA build).

Pin versions once each plug-in works, and note the versions in that session's log.

## Things to watch

- **GPU sharing.** On both machines speech-to-text shares the GPU with Ollama. Measure STT timings with Ollama idle and with Ollama busy (generating), and record both. On the lab PC this will matter for the final system.
- **Python version.** Use Python 3.11 in a venv: the speech libraries lag behind the newest Python. Record the exact version in the log.
- **Microphone.** macOS asks for microphone permission for the terminal / VS Code the first time. On Linux, check the input device with `python -m sounddevice`. Pick the device by name in `settings.yaml` (`audio.input_device`), not by index (indexes change when things are plugged in).
- **Sample rate.** Many mics run at 48 kHz. Capture at the device rate and resample to 16 kHz in `audio/`, once, so every stage sees 16 kHz.
- **Model files.** Downloaded into `models/` (git-ignored). Record what was downloaded and from where in the log, so the lab PC can be set up the same way.

## Moving day checklist (for the lab PC)

1. Clone/copy this folder; `python3.11 -m venv .venv`; `pip install -r requirements/nvidia.txt`.
2. Build whisper.cpp with CUDA if that option is in use.
3. Set `device: auto`, start the app, check the page shows the CUDA options enabled.
4. Run the benchmark (`bench/run_bench.py`) with the same settings files as on the Mac: same WAVs, so the results tables compare directly.
