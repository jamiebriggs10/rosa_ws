"""Smart Turn v3 end-of-turn detector (Pipecat's open model, ONNX, CPU).

A plain silence timer cannot tell "move left..." (thinking) from "move left."
(done). Smart Turn listens to the last 8 s of audio, including intonation,
and says how likely it is that the speaker has finished.

How it is used here: wait for a short pause (pause_ms), ask the model once.
If it says "finished" we stop. If not, keep listening; a later pause asks
again. If the silence reaches max_wait_ms we stop anyway, so a speaker who
simply trailed off is never left hanging.

Runs on the CPU on every machine, so it needs no device-specific code.
Weights: https://huggingface.co/pipecat-ai/smart-turn-v3 (see MODEL_URL).
"""

import time
from collections import deque

import numpy as np

import models
from stages.base import RATE, EndOfTurn
from stages.endpoint.whisper_mel import log_mel

# The newest CPU file listed in the repo (v3.2 is newer than 3.0 / 3.1; the
# "cpu" file is the 8 MB int8 one, quick on a CPU. The "gpu" one is bigger).
MODEL_FILE = "smart-turn-v3.2-cpu.onnx"
MODEL_URL = f"https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/{MODEL_FILE}"

FRAME_MS = 512 / RATE * 1000     # 32 ms
WINDOW_S = 8                     # the model only looks at the last 8 s


class SmartTurn(EndOfTurn):
    name = "smart_turn"
    knobs = {
        "pause_ms": {"type": "int", "default": 200, "min": 50, "max": 1000, "step": 50,
                     "help": "Silence this long makes the model check whether you have finished."},
        "threshold": {"type": "float", "default": 0.5, "min": 0, "max": 1, "step": 0.05,
                      "help": "Model probability at or above this means finished."},
        "max_wait_ms": {"type": "int", "default": 3000, "min": 500, "max": 10000, "step": 250,
                        "help": "If the silence reaches this long, finish even if the model says not yet."},
    }

    def __init__(self, settings: dict, device: str):
        super().__init__(settings, device)
        self._session = None
        self.reset()

    @classmethod
    def supports(cls, device: str) -> tuple[bool, str]:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return False, "onnxruntime is not installed"
        return True, ""

    def load(self) -> None:
        import onnxruntime as ort
        path = models.fetch(MODEL_URL, models.path("smart_turn", MODEL_FILE))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1          # small model: one thread is enough
        self._session = ort.InferenceSession(str(path), sess_options=options,
                                             providers=["CPUExecutionProvider"])

    def reset(self) -> None:
        self._audio = deque(maxlen=int(WINDOW_S * RATE / 512))   # last 8 s of frames
        self._time_s = 0.0          # audio time since the utterance began
        self._seen_speech = False
        self._silence_ms = 0.0
        self._checked = False       # already asked the model during this pause?
        self.last_prob = None
        self.checks = []            # (audio_time_s, prob, ms taken), one per model call

    def probability(self) -> float:
        """Run the model on the audio kept so far; chance (0..1) the turn is over."""
        if self._session is None:
            self.load()
        features = log_mel(np.concatenate(self._audio), WINDOW_S)[None]    # (1, 80, 800)
        t0 = time.perf_counter()
        logits = self._session.run(None, {"input_features": features})[0]
        ms = (time.perf_counter() - t0) * 1000
        prob = float(logits.reshape(-1)[0])    # the file already ends in a sigmoid
        self.last_prob = prob
        self.checks.append((round(self._time_s, 3), prob, ms))
        return prob

    def update(self, frame, is_speech: bool, partial_text: str) -> bool:
        self._audio.append(frame)
        self._time_s += len(frame) / RATE

        if is_speech:
            self._seen_speech = True
            self._silence_ms = 0.0
            self._checked = False       # new speech: the next pause gets a fresh check
            return False
        if not self._seen_speech:
            return False                # leading silence never ends a turn

        self._silence_ms += FRAME_MS
        if self._silence_ms >= self.settings["max_wait_ms"]:
            return True                 # fallback: stop waiting for the model
        if self._silence_ms >= self.settings["pause_ms"] and not self._checked:
            self._checked = True        # once per pause
            return self.probability() >= self.settings["threshold"]
        return False
