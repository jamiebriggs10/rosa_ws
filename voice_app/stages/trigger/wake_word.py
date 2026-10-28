"""openWakeWord: fires when a wake phrase ("hey Jarvis", later "hey Rosa") is heard.

Small ONNX models that run on the CPU on any machine. The library wants
80 ms chunks (1280 samples, int16) but the pipeline sends 32 ms frames
(512 samples), so we collect samples until we have a whole chunk.

Patience and cooldown are done here, in plain code, counting audio time (one
prediction = 80 ms of audio) rather than the clock, so a benchmark replayed
faster than real time behaves the same as a live run.
"""

from pathlib import Path

import numpy as np

import models
from stages.base import Trigger

CHUNK = 1280                 # samples per prediction: 80 ms at 16 kHz
CHUNK_S = CHUNK / 16000
FOLDER = models.path("openwakeword")
ROOT = Path(__file__).resolve().parent.parent.parent


# Phrases openWakeWord ships ready-trained (openwakeword.MODELS in 0.6.0).
PRETRAINED = ["hey_jarvis", "alexa", "hey_mycroft", "hey_rhasspy", "timer", "weather"]


def custom_models() -> list[str]:
    """Our own trained models: any .onnx placed directly in models/ (e.g.
    models/truffles.onnx). Listed when the app starts, so restart after adding one."""
    return sorted(f"models/{p.name}" for p in models.MODELS.glob("*.onnx"))


class WakeWord(Trigger):
    name = "openwakeword"
    last_fire = "wake"
    knobs = {
        "model": {"type": "choice", "default": "hey_jarvis", "choices": PRETRAINED + custom_models(),
                  "help": "A pre-trained phrase, or a custom model trained by us (models/*.onnx)."},
        "threshold": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                      "help": "Score at or above this counts as a hit. Lower = more sensitive, more false triggers."},
        "patience": {"type": "int", "default": 1, "min": 1, "max": 10, "step": 1,
                     "help": "Hits needed in a row (each is 80 ms) before firing."},
        "cooldown_s": {"type": "float", "default": 1.5, "min": 0.0, "max": 10.0, "step": 0.1,
                       "help": "After firing, ignore the wake word for this many seconds of audio."},
    }

    def __init__(self, settings: dict, device: str):
        super().__init__(settings, device)
        self.model = None
        self.enabled = True
        self.last_score = 0.0
        self._pending = np.zeros(0, dtype=np.int16)   # samples waiting for a whole chunk
        self._hits = 0                                # consecutive predictions >= threshold
        self._cooldown_left = 0.0                     # seconds of audio still to ignore

    @classmethod
    def supports(cls, device: str) -> tuple[bool, str]:
        try:
            import openwakeword  # noqa: F401
        except ImportError as e:
            return False, f"openwakeword not installed: {e}"
        return True, ""            # ONNX on the CPU: works on every machine

    def _model_path(self) -> Path:
        name = self.settings["model"].strip()
        if name.endswith(".onnx"):
            return ROOT / name if not Path(name).is_absolute() else Path(name)
        # a pre-trained name: find the downloaded file, e.g. hey_jarvis_v0.1.onnx
        matches = sorted(FOLDER.glob(f"{name}*.onnx"))
        return matches[0] if matches else FOLDER / f"{name}.onnx"

    def load(self) -> None:
        import openwakeword
        from openwakeword.model import Model

        # Pre-trained names are lower case with underscores ("Hey Jarvis" -> "hey_jarvis").
        name = self.settings["model"].strip()
        if not name.endswith(".onnx"):
            name = name.lower().replace(" ", "_")
            if name not in PRETRAINED:
                raise ValueError(f"no pre-trained wake word {name!r}; choose one of {PRETRAINED}"
                                 " or train a custom model (see briefs/2026-10-08-phase7-wake-word.md)")
            self.settings["model"] = name
            # Also fetches the shared mel-spectrogram and embedding models.
            openwakeword.utils.download_models(model_names=[name], target_directory=str(FOLDER))
        self.model = Model(
            wakeword_models=[str(self._model_path())],
            inference_framework="onnx",     # the default tflite runtime is a Linux thing
            melspec_model_path=str(FOLDER / "melspectrogram.onnx"),
            embedding_model_path=str(FOLDER / "embedding_model.onnx"),
        )

    def on_event(self, event: str) -> None:
        if event == "wake_off":
            self.enabled = False
        elif event == "wake_on":
            self.enabled = True

    def reset(self) -> None:
        # The cooldown is kept on purpose: reset() runs after every utterance,
        # and the end of a turn is exactly when we must not re-fire.
        self._pending = np.zeros(0, dtype=np.int16)
        self._hits = 0
        if self.model is not None:
            self.model.reset()

    def process(self, frame) -> bool:
        if not self.enabled:
            return False
        # float -1..1 -> int16, which is what the library expects
        samples = (np.clip(frame, -1, 1) * 32767).astype(np.int16)
        self._pending = np.concatenate([self._pending, samples])

        fired = False
        while len(self._pending) >= CHUNK:
            chunk, self._pending = self._pending[:CHUNK], self._pending[CHUNK:]
            if self._predict(chunk):
                fired = True
        if fired:
            self.last_fire = "wake"
        return fired

    def _predict(self, chunk) -> bool:
        """One 80 ms prediction. True if this is the one that fires."""
        scores = self.model.predict(chunk)
        self.last_score = float(max(scores.values())) if scores else 0.0

        if self._cooldown_left > 0:
            self._cooldown_left -= CHUNK_S
            self._hits = 0
            return False

        if self.last_score >= self.settings["threshold"]:
            self._hits += 1
        else:
            self._hits = 0
        if self._hits >= self.settings["patience"]:
            self._hits = 0
            self._cooldown_left = self.settings["cooldown_s"]
            return True
        return False
