"""WebRTC VAD: Google's old rule-based speech detector (no model at all).

Why: it is the simple baseline to compare Silero against. It is tiny and fast
but fooled more easily by fans and lab noise.

It only accepts 16-bit audio in 10, 20 or 30 ms pieces, while the pipeline
sends 32 ms frames. So we keep the audio in a buffer, cut off as many whole
pieces as it holds, and report what fraction of them were speech.
"""

import numpy as np

from stages.base import VAD, RATE


class WebRTC(VAD):
    name = "webrtc"
    knobs = {
        "aggressiveness": {"type": "int", "default": 2, "min": 0, "max": 3, "step": 1,
                           "help": "0 lets more through, 3 is strictest about calling something speech."},
        "frame_ms": {"type": "choice", "default": 30, "choices": [10, 20, 30],
                     "help": "Size of the pieces the detector judges."},
        "threshold": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                      "help": "Fraction of pieces that must be speech for the frame to count."},
        "min_speech_ms": {"type": "int", "default": 100, "min": 0, "max": 1000, "step": 32,
                          "help": "Speech must last this long to start a turn (ignores blips)."},
    }

    @classmethod
    def supports(cls, device):
        try:
            import webrtcvad  # noqa: F401
        except ImportError as e:
            return False, f"webrtcvad-wheels missing: {e}"
        return True, ""

    def load(self):
        self.piece = RATE * int(self.settings["frame_ms"]) // 1000   # samples per piece
        self.reset()

    def reset(self):
        self.buffer = np.zeros(0, np.int16)   # audio not yet judged
        self.last = 0.0
        # The detector adapts to the noise it has heard, so a fresh one is the
        # only way to truly forget the last utterance.
        import webrtcvad
        self.vad = webrtcvad.Vad(int(self.settings["aggressiveness"]))

    def speech_prob(self, frame):
        pcm = (np.clip(frame, -1, 1) * 32767).astype(np.int16)
        self.buffer = np.concatenate([self.buffer, pcm])
        votes = []
        while len(self.buffer) >= self.piece:
            piece, self.buffer = self.buffer[:self.piece], self.buffer[self.piece:]
            votes.append(self.vad.is_speech(piece.tobytes(), RATE))
        if votes:                              # otherwise keep the previous answer
            self.last = sum(votes) / len(votes)
        return self.last
