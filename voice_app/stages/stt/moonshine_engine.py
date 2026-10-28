"""Moonshine: small speech-to-text models that run on the CPU (ONNX), anywhere.

Why it is here: Moonshine's "streaming" models are built to transcribe while
the person is still talking, so they give a real partial transcript with little
delay. That makes them the natural contrast to the Whisper-style engines, which
re-read the whole buffer each time (see BufferedSTT).

File is called moonshine_engine.py so it does not shadow the library's own
`moonshine_voice` package name or confuse imports.

Batch mode:     one call, transcribe_without_streaming(), on the whole buffer.
Streaming mode: we feed frames into a library stream; it tells us (through a
                listener) when the text of a line changes. A "line" is a
                stretch of speech the library has cut out itself with its own
                built-in voice detector, so one utterance can have several.
English only; the `prompt` setting is ignored (no vocabulary biasing used here).
"""

import threading

import numpy as np

import models
from stages.base import RATE
from stages.stt.buffered import BufferedSTT

MODEL_DIR = models.path("moonshine")

# Our setting name -> name of the library's ModelArch member.
ARCHS = {
    "moonshine-tiny": "TINY",
    "moonshine-base": "BASE",
    "moonshine-streaming-tiny": "TINY_STREAMING",
    "moonshine-streaming-base": "BASE_STREAMING",
    "moonshine-streaming-small": "SMALL_STREAMING",
    "moonshine-streaming-medium": "MEDIUM_STREAMING",
}


class Moonshine(BufferedSTT):
    name = "moonshine"
    streaming_native = True    # in streaming mode it works inside feed()

    knobs = {
        "model": {"type": "choice", "default": "moonshine-streaming-small",
                  "choices": list(ARCHS),
                  "help": "Model size. Bigger is more accurate but slower."},
        "update_interval_s": {"type": "float", "default": 0.3, "min": 0.1, "max": 1.0,
                              "step": 0.1,
                              "help": "Streaming: least audio (seconds) between text updates."},
    }

    @classmethod
    def supports(cls, device):
        try:
            import moonshine_voice  # noqa: F401
        except ImportError as e:
            return False, f"moonshine-voice is not installed ({e})"
        return True, ""

    def load(self):
        from moonshine_voice import ModelArch, Transcriber
        from moonshine_voice.download import get_model_for_language

        arch = getattr(ModelArch, ARCHS[self.settings["model"]])
        # cache_root keeps the model files inside this project (models/moonshine/).
        path, arch = get_model_for_language("en", arch, cache_root=MODEL_DIR)
        self._transcriber = Transcriber(model_path=path, model_arch=arch,
                                        update_interval=self.settings["update_interval_s"])
        self._stream = None
        self._lock = threading.Lock()   # listener may run on another thread
        self._lines = {}                # line id -> latest text, in order of arrival

    def reset(self):
        self._close_stream()
        self._lines = {}

    # ---- batch ----------------------------------------------------------

    def transcribe(self, audio):
        result = self._transcriber.transcribe_without_streaming(
            audio.astype(np.float32).tolist(), RATE)
        return " ".join(line.text.strip() for line in result.lines if line.text.strip())

    # ---- native streaming -------------------------------------------------

    def start(self, preroll):
        super().start(preroll)           # batch mode keeps using the buffer
        if self.settings.get("mode") != "streaming":
            return
        self._close_stream()
        self._lines = {}
        self._changed = False
        self._stream = self._transcriber.create_stream(
            update_interval=self.settings["update_interval_s"])
        self._stream.add_listener(self._on_event)
        self._stream.start()
        if len(preroll):
            self._stream.add_audio(np.asarray(preroll, np.float32).tolist(), RATE)

    def _on_event(self, event):
        """Called by the library whenever a line starts, changes or completes."""
        with self._lock:
            self._lines[event.line.line_id] = event.line.text
            self._changed = True

    def _text(self):
        with self._lock:
            return " ".join(t.strip() for t in self._lines.values() if t.strip())

    def feed(self, frame):
        if self.settings.get("mode") != "streaming":
            return super().feed(frame)
        self._stream.add_audio(np.asarray(frame, np.float32).tolist(), RATE)
        if not self._changed:
            return None
        self._changed = False
        text = self._text()
        if text == self._last_partial:
            return None
        self._last_partial = text
        return text

    def finish(self):
        if self.settings.get("mode") != "streaming":
            return super().finish()
        # stop() flushes the audio still waiting and fires the last events.
        self._stream.stop()
        text = self._text()
        self._close_stream()
        return text

    def _close_stream(self):
        if getattr(self, "_stream", None) is not None:
            self._stream.close()
            self._stream = None
