"""WAV file input: the same frames as the mic, for repeatable tests and benchmarks."""

import time

import numpy as np
import soundfile as sf
import soxr

from audio.framer import FRAME, RATE, Framer


def load_wav(path) -> np.ndarray:
    """Read any WAV as mono float32 at 16 kHz."""
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)                     # mix to mono
    if rate != RATE:
        audio = soxr.resample(audio, rate, RATE)   # once, here, so stages only ever see 16 kHz
    return audio.astype(np.float32)


class FileSource:
    """Yields 512-sample frames from a WAV, as fast as possible or at real speed.

    `pad_end_s` adds silence after the clip so a silence-based endpoint has
    room to close the turn (recorded clips already have a 1.5 s tail).
    """

    kind = "file"

    def __init__(self, path, realtime: bool = False, pad_end_s: float = 0.0, frame_samples: int = FRAME):
        self.path = str(path)
        self.realtime = realtime
        self.frame_samples = frame_samples
        audio = load_wav(path)
        self.audio = np.concatenate([audio, np.zeros(int(pad_end_s * RATE), np.float32)])
        self.duration_s = len(self.audio) / RATE
        self._running = False

    def start(self):
        self._running = True

    def stop(self):
        self._running = False

    def backlog(self) -> int:
        return 0

    def frames(self):
        framer = Framer(RATE, self.frame_samples)
        t0 = time.perf_counter()
        frames = framer.push(self.audio) + framer.flush()
        for i, frame in enumerate(frames):
            if not self._running:
                return
            if self.realtime:
                # Wait until this frame would have arrived from a real mic.
                due = t0 + (i + 1) * self.frame_samples / RATE
                delay = due - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            yield frame
