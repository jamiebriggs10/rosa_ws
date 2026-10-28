"""Silero VAD (voice activity detection), run straight from its ONNX file.

Why: it is the standard small speech detector (about 1 ms per 32 ms frame, CPU
only). We call it through onnxruntime alone, so no PyTorch is needed.

The model is stateful. Each call needs (a) the last 64 samples of the previous
frame in front of the new 512 samples, and (b) a (2, 1, 128) memory tensor it
handed back last time. reset() clears both between utterances.
"""

import numpy as np

import models
from stages.base import VAD

# Pinned release so the experiment is repeatable.
URL = "https://raw.githubusercontent.com/snakers4/silero-vad/v5.1.2/src/silero_vad/data/silero_vad.onnx"
CONTEXT = 64


class Silero(VAD):
    name = "silero"
    knobs = {
        "threshold": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                      "help": "Probability at or above this counts as speech."},
        "min_speech_ms": {"type": "int", "default": 100, "min": 0, "max": 1000, "step": 32,
                          "help": "Speech must last this long to start a turn (ignores blips)."},
    }

    @classmethod
    def supports(cls, device):
        try:
            import onnxruntime  # noqa: F401
        except ImportError as e:
            return False, f"onnxruntime missing: {e}"
        return True, ""

    def load(self):
        import onnxruntime as ort
        path = models.fetch(URL, models.path("silero", "silero_vad.onnx"))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1      # tiny model: more threads only add overhead
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.sr = np.array(16000, dtype=np.int64)
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), np.float32)
        self.context = np.zeros(CONTEXT, np.float32)

    def speech_prob(self, frame):
        x = np.concatenate([self.context, frame]).astype(np.float32)[None, :]
        prob, self.state = self.session.run(None, {"input": x, "state": self.state, "sr": self.sr})
        self.context = frame[-CONTEXT:].copy()
        return float(prob[0, 0])
