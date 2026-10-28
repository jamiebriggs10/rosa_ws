"""Live microphone input (sounddevice / PortAudio, works on macOS and Linux)."""

import queue

import numpy as np
import sounddevice as sd

from audio.framer import FRAME, Framer


def find_device(name: str | None) -> int | None:
    """Input device index whose name contains `name` (case-insensitive).
    By name, not index, because indexes change when things are plugged in."""
    if not name:
        return None   # system default
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and name.lower() in dev["name"].lower():
            return i
    raise ValueError(f"no input device matching {name!r}; see `python -m sounddevice`")


class MicSource:
    """Captures at the device's own rate; frames() hands out 16 kHz 512-sample frames."""

    kind = "mic"

    def __init__(self, input_device: str | None = None, frame_samples: int = FRAME):
        self.device = find_device(input_device)
        self.rate = int(sd.query_devices(self.device, "input")["default_samplerate"])
        self.framer = Framer(self.rate, frame_samples)
        self._q: queue.Queue = queue.Queue()
        self._stream = None
        self._running = False

    def _callback(self, indata, frames, time_info, status):
        # Runs on PortAudio's audio thread: copy and queue, nothing else,
        # so a slow stage can never cause dropped audio.
        self._q.put(indata[:, 0].copy())

    def start(self):
        self._running = True
        self._stream = sd.InputStream(device=self.device, samplerate=self.rate, channels=1,
                                      dtype="float32", blocksize=0, callback=self._callback)
        self._stream.start()

    def stop(self):
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def backlog(self) -> int:
        """Chunks waiting in the queue: grows if the pipeline can't keep up."""
        return self._q.qsize()

    def drain(self):
        """Throw away queued audio (e.g. what arrived while a model was loading),
        so a wake word isn't detected seconds late in stale audio."""
        while not self._q.empty():
            self._q.get_nowait()

    def frames(self):
        """Yield frames until stop() is called."""
        while self._running:
            try:
                chunk = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            yield from self.framer.push(chunk)


def record(seconds: float, input_device: str | None = None) -> tuple[np.ndarray, int]:
    """Record a fixed length at the device rate (for the phase 0 mic check)."""
    dev = find_device(input_device)
    rate = int(sd.query_devices(dev, "input")["default_samplerate"])
    audio = sd.rec(int(seconds * rate), samplerate=rate, channels=1, dtype="float32", device=dev)
    sd.wait()
    return audio[:, 0], rate
