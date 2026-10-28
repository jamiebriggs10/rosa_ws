# Presets

Named copies of `settings.yaml` for repeatable comparisons. Loadable from the page and passed to `bench/run_bench.py --settings`. **Once a preset has been used for reported results, don't change it: make a new one.**

- `baseline.yaml`: Button press, Silero VAD, endpoint = button release, Parakeet batch. Reference run; lower bound on felt delay.
- `silence-400.yaml`, `silence-700.yaml`, `silence-1000.yaml`: Fixed-duration silence endpoints (400–1000 ms).
- `adaptive.yaml`: Adaptive silence endpoint (short wait, longer after words like 'by', 'to') with Parakeet streaming.
- `smart-turn.yaml`: Smart Turn model decides when the speaker has finished.
- `streaming.yaml`: Baseline but Parakeet in streaming mode (partial text as it arrives).
- `whisper-cpp.yaml`, `faster-whisper.yaml`, `moonshine.yaml`, `moonshine-streaming.yaml`, `parakeet-sherpa.yaml`: Different STT engines or backends.
- `no-prompt.yaml`: Baseline with the domain prompt / hotword list switched off.
- `wake-jarvis.yaml`, `wake-truffles.yaml`: Wake word 'hey Jarvis' (pre-trained) or 'hey Rosa' (custom; needs training), with button as fallback.
