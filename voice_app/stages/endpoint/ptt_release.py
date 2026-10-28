"""End of turn by letting go of the push-to-talk button.

No guessing: the utterance ends when the button is released.

Rule for "which kind of utterance is this?" (decided in reset()):
  * the button is held when the utterance starts -> a PTT utterance; it ends
    the moment the button is up. The pipeline sends ptt_down BEFORE reset(),
    so `held` is already True by then. reset() therefore must not clear it.
  * the button is NOT held at the start (e.g. the wake word fired while the
    trigger is "both") -> a hands-free utterance; behave like `silence` with
    fallback_silence_ms. A release that arrived before this reset (the button
    from the previous utterance) cannot end it, because held is already False.
  * a press that arrives mid-way through a hands-free utterance turns it into
    a PTT one: the speaker has taken over with the button.
"""

from stages.base import EndOfTurn
from stages.endpoint.silence import SilenceCounter


class PttRelease(EndOfTurn):
    name = "ptt_release"
    knobs = {
        "fallback_silence_ms": {"type": "int", "default": 700, "min": 200, "max": 3000,
                                "step": 50, "help": "Silence that ends an utterance not started by the button."},
    }

    def __init__(self, settings: dict, device: str):
        super().__init__(settings, device)
        self.counter = SilenceCounter()
        self.held = False          # is the button down right now (survives reset)
        self.ptt_utterance = False # is this utterance ended by the button

    def on_event(self, event: str) -> None:
        if event == "ptt_down":
            self.held = True
            self.ptt_utterance = True
        elif event == "ptt_up":
            self.held = False

    def reset(self) -> None:
        self.counter.reset()
        self.ptt_utterance = self.held   # see the module docstring

    def update(self, frame, is_speech: bool, partial_text: str) -> bool:
        self.counter.update(is_speech)
        if self.ptt_utterance:
            return not self.held
        return self.counter.reached(self.settings["fallback_silence_ms"])
