"""The fixed stages of the voice pipeline, from the prototype's benchmark winner (8 October 2026):

    Trigger    the wake word "truffles" (openWakeWord) or a push-to-talk press, whichever first
    Silero     voice activity detection: is this frame speech?
    SilenceEnd the turn is over after `silence_ms` of silence following speech (wake-word turns)
    ReleaseEnd the turn is over when the button is let go (push-to-talk turns)
    Parakeet   NVIDIA Parakeet TDT 0.6B v3 through sherpa-onnx, int8, CPU or CUDA; batch only

Frames are numpy float32 arrays of 512 samples (32 ms) at 16 kHz. Time is counted in frames,
never the clock, so a WAV replayed through the pipeline behaves exactly as the microphone did.
The model libraries are imported inside load(), so the console starts without them.
"""

import shutil
import subprocess

import numpy as np

from voice import models

RATE = 16000
FRAME_MS = 512 / RATE * 1000          # 32 ms


def device():
    """"cuda" when an NVIDIA GPU with a driver is present, else "cpu"."""
    if shutil.which("nvidia-smi") is None:
        return "cpu"
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
        return "cuda" if out.returncode == 0 and "GPU" in out.stdout else "cpu"
    except (OSError, subprocess.TimeoutExpired):
        return "cpu"


# ------------------------------------------------------------------ triggers

class WakeWord:
    """Fires when the wake word is heard. openWakeWord wants 80 ms chunks (1280 samples, int16);
    the pipeline sends 32 ms frames, so samples are collected until a chunk is whole. Patience
    and cooldown count audio time (one prediction = 80 ms)."""

    CHUNK = 1280
    CHUNK_S = CHUNK / RATE

    def __init__(self, threshold=0.5, patience=1, cooldown_s=1.5, model_path=models.WAKE_MODEL):
        self.threshold, self.patience, self.cooldown_s, self.model_path = threshold, patience, cooldown_s, model_path
        self.model = None
        self.enabled = True
        self.last_score = 0.0
        self._pending = np.zeros(0, dtype=np.int16)
        self._hits = 0
        self._cooldown_left = 0.0

    def load(self):
        from openwakeword.model import Model
        folder = models.openwakeword_dir()
        self.model = Model(wakeword_models=[str(self.model_path)], inference_framework="onnx",
                           melspec_model_path=str(folder / "melspectrogram.onnx"),
                           embedding_model_path=str(folder / "embedding_model.onnx"))

    def reset(self):
        # The cooldown is kept on purpose: reset() runs after every utterance, which is exactly
        # when the wake word must not fire again on the tail of what was said.
        self._pending = np.zeros(0, dtype=np.int16)
        self._hits = 0
        if self.model is not None:
            self.model.reset()

    def process(self, frame):
        if not self.enabled:
            return False
        samples = (np.clip(frame, -1, 1) * 32767).astype(np.int16)
        self._pending = np.concatenate([self._pending, samples])
        fired = False
        while len(self._pending) >= self.CHUNK:
            chunk, self._pending = self._pending[:self.CHUNK], self._pending[self.CHUNK:]
            fired = self._predict(chunk) or fired
        return fired

    def _predict(self, chunk):
        scores = self.model.predict(chunk)
        self.last_score = float(max(scores.values())) if scores else 0.0
        if self._cooldown_left > 0:
            self._cooldown_left -= self.CHUNK_S
            self._hits = 0
            return False
        self._hits = self._hits + 1 if self.last_score >= self.threshold else 0
        if self._hits >= self.patience:
            self._hits, self._cooldown_left = 0, self.cooldown_s
            return True
        return False


class PushToTalk:
    """The button: press() is remembered and fires on the next frame, once."""

    def __init__(self):
        self._pressed = False

    def load(self):
        pass

    def press(self):
        self._pressed = True

    def reset(self):
        self._pressed = False                          # a press from before the last utterance is stale

    def process(self, frame):
        fired, self._pressed = self._pressed, False
        return fired


class Trigger:
    """Wake word or button, whichever comes first. `last_fire` says which: "wake" means wait
    for speech to start; "ptt" means the person is already talking."""

    def __init__(self, wake, ptt):
        self.wake, self.ptt = wake, ptt
        self.last_fire = "wake"

    def load(self):
        self.wake.load()
        self.ptt.load()

    def reset(self):
        self.wake.reset()
        self.ptt.reset()

    def press(self):
        self.ptt.press()

    def enable_wake(self, on):
        self.wake.enabled = on

    def process(self, frame):
        wake = self.wake.process(frame)                # always fed, so its audio buffer stays continuous
        if self.ptt.process(frame):
            self.last_fire = "ptt"
            return True
        if wake:
            self.last_fire = "wake"
            return True
        return False


# ----------------------------------------------------------------------- VAD

class Silero:
    """Silero VAD v5 straight from its ONNX file (about 1 ms a frame on the CPU). The model is
    stateful: each call needs the last 64 samples of the previous frame and a memory tensor."""

    CONTEXT = 64

    def __init__(self, threshold=0.5, min_speech_ms=100):
        self.threshold = threshold
        self.min_speech_frames = max(1, int(round(min_speech_ms / FRAME_MS)))
        self.session = None

    def load(self):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(models.silero()), opts, providers=["CPUExecutionProvider"])
        self.sr = np.array(RATE, dtype=np.int64)
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), np.float32)
        self.context = np.zeros(self.CONTEXT, np.float32)

    def speech_prob(self, frame):
        x = np.concatenate([self.context, frame]).astype(np.float32)[None, :]
        prob, self.state = self.session.run(None, {"input": x, "state": self.state, "sr": self.sr})
        self.context = frame[-self.CONTEXT:].copy()
        return float(prob[0, 0])


# --------------------------------------------------------------- end of turn

class SilenceCounter:
    """Consecutive non-speech frames, counted only once speech has been heard."""

    def reset(self):
        self.heard_speech, self.silent_frames = False, 0

    def update(self, is_speech):
        if is_speech:
            self.heard_speech, self.silent_frames = True, 0
        elif self.heard_speech:
            self.silent_frames += 1

    def reached(self, ms):
        return self.heard_speech and self.silent_frames * FRAME_MS >= ms


class SilenceEnd:
    """Finished after `silence_ms` of silence following speech."""

    def __init__(self, silence_ms=700):
        self.silence_ms = silence_ms
        self.counter = SilenceCounter()
        self.counter.reset()

    def reset(self):
        self.counter.reset()

    def update(self, is_speech):
        self.counter.update(is_speech)
        return self.counter.reached(self.silence_ms)


class ReleaseEnd:
    """Finished when the push-to-talk button is let go. The kind of utterance is decided in
    reset(): held at the start means a button utterance (ends on release); not held means a
    wake-word utterance (ends after `fallback_silence_ms`, and an old release cannot end it);
    a press mid-way through a wake-word utterance turns it into a button one."""

    def __init__(self, fallback_silence_ms=700):
        self.fallback_silence_ms = fallback_silence_ms
        self.counter = SilenceCounter()
        self.counter.reset()
        self.held = False
        self.ptt_utterance = False

    def press(self):
        self.held = self.ptt_utterance = True

    def release(self):
        self.held = False

    def reset(self):
        self.counter.reset()
        self.ptt_utterance = self.held

    def update(self, is_speech):
        self.counter.update(is_speech)
        if self.ptt_utterance:
            return not self.held
        return self.counter.reached(self.fallback_silence_ms)


# ------------------------------------------------------------ speech to text

class Parakeet:
    """Keeps the utterance's audio and transcribes it whole when the turn closes."""

    def __init__(self, threads=4, dev=None):
        self.threads, self.dev = threads, dev or device()
        self.recognizer = None
        self._chunks = []

    def load(self):
        import sherpa_onnx
        folder = models.parakeet()
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(folder / "encoder.int8.onnx"), decoder=str(folder / "decoder.int8.onnx"),
            joiner=str(folder / "joiner.int8.onnx"), tokens=str(folder / "tokens.txt"),
            model_type="nemo_transducer", num_threads=int(self.threads),
            provider="cuda" if self.dev == "cuda" else "cpu")
        self.transcribe(np.zeros(RATE, np.float32))    # warm up on silence, so the first real one is not slow

    def reset(self):
        self._chunks = []

    def start(self, preroll):
        self._chunks = [np.asarray(preroll, dtype=np.float32)]

    def feed(self, frame):
        self._chunks.append(frame)

    def finish(self):
        audio = np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32)
        self._chunks = []
        return self.transcribe(audio)

    def transcribe(self, audio):
        stream = self.recognizer.create_stream()
        stream.accept_waveform(RATE, np.asarray(audio, dtype=np.float32))
        self.recognizer.decode_stream(stream)
        return stream.result.text.strip()
