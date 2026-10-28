"""End of turn by silence: the baseline option.

The speaker has finished when the VAD has said "not speech" for `silence_ms`
in a row, after at least one speech frame in this utterance. Simple and
predictable, so the smarter options can be compared against it.

Time is counted in frames (audio time, 32 ms each), not wall-clock time, so
the same audio gives the same answer live and when replayed by the benchmark.
"""

from stages.base import RATE, EndOfTurn

FRAME_MS = 512 / RATE * 1000   # 32 ms


class SilenceCounter:
    """Counts consecutive non-speech frames, but only once speech has been heard.

    Shared by `silence`, `adaptive` and `ptt_release` so they all measure
    silence the same way.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.heard_speech = False
        self.silent_frames = 0

    def update(self, is_speech: bool) -> None:
        if is_speech:
            self.heard_speech = True
            self.silent_frames = 0
        elif self.heard_speech:
            self.silent_frames += 1   # silence before any speech is not counted

    def silent_ms(self) -> float:
        return self.silent_frames * FRAME_MS

    def reached(self, ms: float) -> bool:
        """True once at least `ms` of silence has followed speech."""
        return self.heard_speech and self.silent_ms() >= ms


class Silence(EndOfTurn):
    name = "silence"
    knobs = {
        "silence_ms": {"type": "int", "default": 700, "min": 200, "max": 3000,
                       "step": 50, "help": "Finished after this much silence following speech."},
    }

    def __init__(self, settings: dict, device: str):
        super().__init__(settings, device)
        self.counter = SilenceCounter()

    def reset(self) -> None:
        self.counter.reset()

    def update(self, frame, is_speech: bool, partial_text: str) -> bool:
        self.counter.update(is_speech)
        return self.counter.reached(self.settings["silence_ms"])
