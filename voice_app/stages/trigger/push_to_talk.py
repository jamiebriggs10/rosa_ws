"""Push-to-talk: the trigger is a button press, nothing is guessed.

The pipeline broadcasts on_event("ptt_down") when the button goes down. We
remember it and fire on the next audio frame, because process() is only ever
called with a frame. Exact and free, so it is the reference trigger.
"""

from stages.base import Trigger


class PushToTalk(Trigger):
    name = "push_to_talk"
    knobs = {}
    last_fire = "ptt"      # a button means "listen now", no waiting for speech

    def __init__(self, settings: dict, device: str):
        super().__init__(settings, device)
        self._pressed = False

    def on_event(self, event: str) -> None:
        if event == "ptt_down":
            self._pressed = True      # ptt_up is not our business (the endpointer uses it)

    def reset(self) -> None:
        self._pressed = False         # a press from before the last utterance is stale

    def process(self, frame) -> bool:
        if self._pressed:
            self._pressed = False     # fire once per press
            self.last_fire = "ptt"
            return True
        return False
