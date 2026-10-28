"""Shared logic for engines that transcribe a whole buffer at once.

An engine only has to implement `transcribe(audio) -> str`. This class gives it:
  - batch mode: keep the audio, transcribe once in finish();
  - pseudo-streaming mode: also re-transcribe the growing buffer every
    `partial_every_ms` of new audio and return that as partial text.
Engines that stream natively (Parakeet MLX, Moonshine) override start/feed/finish
for streaming mode and use this for batch mode.
"""

import numpy as np

from stages.base import RATE, SpeechToText


class BufferedSTT(SpeechToText):

    def start(self, preroll: np.ndarray) -> None:
        self._chunks = [np.asarray(preroll, dtype=np.float32)]
        self._since_partial = 0          # samples added since the last partial
        self._last_partial = ""

    def feed(self, frame) -> str | None:
        self._chunks.append(frame)
        if self.settings.get("mode") != "streaming":
            return None
        self._since_partial += len(frame)
        if self._since_partial < self.settings.get("partial_every_ms", 500) * RATE / 1000:
            return None
        self._since_partial = 0
        text = self.transcribe(self.audio())
        if text == self._last_partial:
            return None
        self._last_partial = text
        return text

    def finish(self) -> str:
        return self.transcribe(self.audio())

    def audio(self) -> np.ndarray:
        """Everything heard so far in this utterance, one array."""
        return np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32)

    def transcribe(self, audio: np.ndarray) -> str:
        """Audio (float32, 16 kHz, mono) to text. Each engine implements this."""
        raise NotImplementedError
