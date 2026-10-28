"""Both: fires if the button is pressed or the wake word is heard.

The sensible default for real use. It just holds one of each and asks both.
The pipeline passes the wake word's knobs under the key "openwakeword".
"""

from stages.base import Trigger
from stages.trigger.push_to_talk import PushToTalk
from stages.trigger.wake_word import WakeWord


class Both(Trigger):
    name = "both"
    knobs = {}

    def __init__(self, settings: dict, device: str):
        super().__init__(settings, device)
        self.ptt = PushToTalk({}, device)
        self.wake = WakeWord(self.settings.get("openwakeword", {}), device)

    @property
    def last_score(self) -> float:
        return self.wake.last_score

    @classmethod
    def supports(cls, device: str) -> tuple[bool, str]:
        return WakeWord.supports(device)

    def load(self) -> None:
        self.ptt.load()
        self.wake.load()

    def reset(self) -> None:
        self.ptt.reset()
        self.wake.reset()

    def on_event(self, event: str) -> None:
        self.ptt.on_event(event)
        self.wake.on_event(event)

    def process(self, frame) -> bool:
        # Always feed the wake word so its audio buffer stays continuous,
        # even on the frame where the button fires.
        wake = self.wake.process(frame)
        if self.ptt.process(frame):       # button first: it is exact
            self.last_fire = "ptt"
            return True
        if wake:
            self.last_fire = "wake"
            return True
        return False
