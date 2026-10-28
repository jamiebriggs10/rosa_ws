"""Turn audio chunks of any size and rate into 512-sample, 16 kHz frames.

Both audio sources use this, so the mic and a WAV file give the pipeline
exactly the same kind of frames.
"""

import numpy as np
import soxr

RATE = 16000          # what every stage sees
FRAME = 512           # samples per frame = 32 ms at 16 kHz


class Framer:
    def __init__(self, in_rate: int, frame_samples: int = FRAME):
        self.frame_samples = frame_samples
        # A streaming resampler keeps its filter state between chunks, so there
        # are no clicks at chunk boundaries. Skipped when the rate is already right.
        self._resampler = None if in_rate == RATE else soxr.ResampleStream(in_rate, RATE, 1, dtype="float32")
        self._buf = np.zeros(0, dtype=np.float32)

    def push(self, chunk: np.ndarray) -> list[np.ndarray]:
        """Add a mono float32 chunk; return the whole frames now available."""
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if self._resampler is not None:
            chunk = self._resampler.resample_chunk(chunk)
        self._buf = np.concatenate([self._buf, chunk])
        n = len(self._buf) // self.frame_samples
        frames = [self._buf[i * self.frame_samples:(i + 1) * self.frame_samples] for i in range(n)]
        self._buf = self._buf[n * self.frame_samples:]
        return frames

    def flush(self) -> list[np.ndarray]:
        """End of audio: return what's left, zero-padded to a whole frame."""
        if self._resampler is not None:
            self._buf = np.concatenate([self._buf, self._resampler.resample_chunk(np.zeros(0, np.float32), last=True)])
        if len(self._buf) == 0:
            return []
        last = np.zeros(self.frame_samples, dtype=np.float32)
        last[:len(self._buf)] = self._buf
        self._buf = np.zeros(0, dtype=np.float32)
        return [last]


def rms(frame: np.ndarray) -> float:
    """Loudness of a frame (root mean square), 0..1. Drives the page's level meter."""
    return float(np.sqrt(np.mean(np.square(frame)))) if len(frame) else 0.0
