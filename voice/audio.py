"""Audio in: the microphone or a WAV file, as 512-sample frames (32 ms) at 16 kHz, mono, float32
in -1..1. Both sources have the same five methods (start, stop, frames, backlog, kind), so the
pipeline cannot tell them apart; a WAV through FileSource gives exactly the frames the microphone
would, which is what makes the tests repeatable.

The sound libraries (soxr, soundfile, sounddevice) are imported inside the functions that need
them, so this module imports on a machine without them and the console can say why it is silent.
"""

import queue
import time

import numpy as np

RATE = 16000                 # what every stage hears
FRAME = 512                  # samples per frame: 32 ms at 16 kHz
FRAME_S = FRAME / RATE


class Framer:
    """Turns chunks of any size and rate into whole 16 kHz frames; keeps its resampler's state
    between chunks so there are no clicks at chunk boundaries."""

    def __init__(self, in_rate, frame_samples=FRAME):
        self.frame_samples = frame_samples
        self._resampler = None
        if in_rate != RATE:
            import soxr
            self._resampler = soxr.ResampleStream(in_rate, RATE, 1, dtype="float32")
        self._buf = np.zeros(0, dtype=np.float32)

    def push(self, chunk):
        """Add a mono float32 chunk; return the whole frames now available."""
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if self._resampler is not None:
            chunk = self._resampler.resample_chunk(chunk)
        return self._frames_from(chunk)

    def _frames_from(self, chunk):
        self._buf = np.concatenate([self._buf, chunk])
        n = len(self._buf) // self.frame_samples
        frames = [self._buf[i * self.frame_samples:(i + 1) * self.frame_samples] for i in range(n)]
        self._buf = self._buf[n * self.frame_samples:]
        return frames

    def flush(self):
        """End of audio: what is left, zero-padded to a whole frame."""
        frames = []
        if self._resampler is not None:              # the resampler's tail (its filter delay) can be a frame or more
            frames = self._frames_from(self._resampler.resample_chunk(np.zeros(0, np.float32), last=True))
        if len(self._buf):
            last = np.zeros(self.frame_samples, dtype=np.float32)
            last[:len(self._buf)] = self._buf
            frames.append(last)
        self._buf = np.zeros(0, dtype=np.float32)
        return frames


def rms(frame):
    """Loudness of a frame (root mean square), 0..1: the page's level meter."""
    return float(np.sqrt(np.mean(np.square(frame)))) if len(frame) else 0.0


def load_wav(path):
    """Any WAV as mono float32 at 16 kHz."""
    import soundfile as sf
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != RATE:
        import soxr
        audio = soxr.resample(audio, rate, RATE)
    return audio.astype(np.float32)


class FileSource:
    """Frames from a WAV, as fast as possible or at real speed. `pad_end_s` adds silence after
    the clip so a silence-based end of turn has room to close it."""

    kind = "file"

    def __init__(self, path, realtime=False, pad_end_s=0.0):
        self.path, self.realtime = str(path), realtime
        audio = load_wav(path)
        self.audio = np.concatenate([audio, np.zeros(int(pad_end_s * RATE), np.float32)])
        self.duration_s = len(self.audio) / RATE
        self._running = False

    def start(self):
        self._running = True

    def stop(self):
        self._running = False

    def backlog(self):
        return 0

    def frames(self):
        framer = Framer(RATE)
        t0 = time.perf_counter()
        frames = framer.push(self.audio) + framer.flush()
        for i, frame in enumerate(frames):
            if not self._running:
                return
            if self.realtime:                        # wait until this frame would have come from a microphone
                delay = t0 + (i + 1) * FRAME_S - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            yield frame


def find_device(name):
    """The input device whose name contains `name` (case-insensitive), or None for the system
    default. By name, not index, because indexes change when things are plugged in."""
    if not name:
        return None
    import sounddevice as sd
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and name.lower() in dev["name"].lower():
            return i
    raise ValueError(f"no input device matching {name!r}")


def list_devices():
    """{"input": [names], "output": [names]} for the settings pop-up."""
    try:
        import sounddevice as sd
        devices = list(sd.query_devices())
    except Exception:
        return {"input": [], "output": []}
    return {"input": [d["name"] for d in devices if d["max_input_channels"] > 0],
            "output": [d["name"] for d in devices if d["max_output_channels"] > 0]}


class MicSource:
    """Captures at the device's own rate; frames() hands out 16 kHz frames."""

    kind = "mic"

    def __init__(self, input_device=None):
        import sounddevice as sd
        self._sd = sd
        self.device = find_device(input_device)
        self.rate = int(sd.query_devices(self.device, "input")["default_samplerate"])
        self.framer = Framer(self.rate)
        self._q = queue.Queue()
        self._stream = None
        self._running = False

    def _callback(self, indata, frames, time_info, status):
        self._q.put(indata[:, 0].copy())            # PortAudio's thread: copy and queue, nothing else

    def start(self):
        self._running = True
        self._stream = self._sd.InputStream(device=self.device, samplerate=self.rate, channels=1,
                                            dtype="float32", blocksize=0, callback=self._callback)
        self._stream.start()

    def stop(self):
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def backlog(self):
        return self._q.qsize()

    def drain(self):
        """Throw away queued audio (what arrived while a model was loading)."""
        while not self._q.empty():
            self._q.get_nowait()

    def frames(self):
        while self._running:
            try:
                chunk = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            yield from self.framer.push(chunk)
