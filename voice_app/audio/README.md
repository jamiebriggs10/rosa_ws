# audio/

Two sources with the same interface, so live and benchmark runs go through identical code:

- `MicSource`: captures from `sounddevice` (PortAudio), resamples to 16 kHz, yields 512-sample float32 frames (32 ms). The callback queues raw chunks; all processing happens on the main thread.
- `FileSource`: reads a WAV, resamples if needed, yields the same frames. `realtime=True` paces to real time; `pad_end_s` adds silence after the clip (for silence-based endpoints to close the turn).

Both report mic level (RMS) for the page's meter. `Framer` (used by both) turns any chunk size and rate into frames.
