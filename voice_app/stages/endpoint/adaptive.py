"""End of turn that reads the partial transcript: our own small rule set.

Plain `silence` waits the same time whatever was said. But "move left by" or
"move up five" is clearly unfinished, while "go home" is clearly done. So:

  * if the text ends in a word that expects more ("by", "to", "and"...) or in
    a number with no unit after it ("move five", "minus sixty"), wait long;
  * otherwise wait only a short time;
  * if there is no partial text (batch STT, or nothing recognised yet),
    use one middle value so the option still works.

Silence is counted in frames, exactly as in `silence`.
"""

import re

from stages.base import EndOfTurn
from stages.endpoint.silence import SilenceCounter

NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "minus", "point",
}


def normalise(text: str) -> list[str]:
    """Lower-case the text, drop punctuation (keeping '-' and '.' inside
    numbers like -60 and 0.5), and split into words."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9.\- ]", " ", text)
    words = (w.strip(".-") if not re.search(r"\d", w) else w.rstrip(".-")
             for w in text.split())
    return [w for w in words if w]


def last_continue_word(words: list[str], continue_words) -> str | None:
    """The last word if it is one that expects more to follow, else None."""
    if words and words[-1] in continue_words:
        return words[-1]
    return None


def ends_in_bare_number(words: list[str]) -> bool:
    """True if the last word is a number (digits or a number word). Because
    only the LAST word is looked at, a unit after the number ("five
    centimetres") means this is False: the number is no longer bare."""
    if not words:
        return False
    last = words[-1]
    return bool(re.fullmatch(r"-?\d+(\.\d+)?", last)) or last in NUMBER_WORDS


class Adaptive(EndOfTurn):
    name = "adaptive"
    knobs = {
        "short_silence_ms": {"type": "int", "default": 400, "min": 100, "max": 2000,
                             "step": 50, "help": "Wait this long when the text sounds finished."},
        "long_silence_ms": {"type": "int", "default": 1500, "min": 500, "max": 5000,
                            "step": 100, "help": "Wait this long when the text sounds unfinished."},
        "continue_words": {"type": "list",
                           "default": ["by", "to", "and", "then", "point", "about", "at", "of", "the", "a"],
                           "help": "If the text ends in one of these, more is coming."},
        "fallback_silence_ms": {"type": "int", "default": 700, "min": 200, "max": 3000,
                                "step": 50, "help": "Used when there is no partial text."},
    }

    def __init__(self, settings: dict, device: str):
        super().__init__(settings, device)
        self.counter = SilenceCounter()
        self.last_reason = "no partial text"   # for the run log

    def reset(self) -> None:
        self.counter.reset()
        self.last_reason = "no partial text"

    def wait_ms(self, partial_text: str) -> float:
        """How long to wait for this text; also sets last_reason."""
        words = normalise(partial_text or "")
        if not words:
            self.last_reason = "no partial text"
            return self.settings["fallback_silence_ms"]
        word = last_continue_word(words, self.settings["continue_words"])
        if word:
            self.last_reason = f"continue word '{word}'"
            return self.settings["long_silence_ms"]
        if ends_in_bare_number(words):
            self.last_reason = "bare number"
            return self.settings["long_silence_ms"]
        self.last_reason = "complete"
        return self.settings["short_silence_ms"]

    def update(self, frame, is_speech: bool, partial_text: str) -> bool:
        self.counter.update(is_speech)
        return self.counter.reached(self.wait_ms(partial_text))
