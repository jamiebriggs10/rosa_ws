"""What a spoken phrase means. Only a whole utterance of stop words is a stop, so "stop the
move" is a command for the model, not an emergency stop. Only "confirm" or "go ahead" move the
robot: "yes", "okay" and "go" are too easy to mishear or to say in passing."""

import re

STOP_WORDS = frozenset({"stop", "halt", "freeze"})
YES_WORDS = frozenset({"confirm", "go ahead"})
NO_WORDS = frozenset({"no", "cancel"})
WAKE_WORDS = ("rosa", "hey rosa", "truffles")     # stripped from the front of an utterance
_NO_START = re.compile(r"^(no|nope|cancel)\b[,.]?\s*(?:(?:actually|rather|instead)\b[,.]?\s*)?")


def normalise(text):
    """Lower case, letters and digits only, single spaces: "Stop." -> "stop"."""
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", (text or "").lower()).split())


def is_stop(text):
    """True when every word is a stop word (and there is at least one)."""
    words = normalise(text).split()
    return bool(words) and all(w in STOP_WORDS for w in words)


def says_yes(text):
    return normalise(text) in YES_WORDS


def says_no(text):
    return normalise(text) in NO_WORDS


def strip_wake(text, words=WAKE_WORDS):
    """The utterance without a leading wake word ("Truffles, confirm" -> "confirm"), normalised.
    A wake word in the middle of a sentence is left alone."""
    out = normalise(text)
    for word in sorted((normalise(w) for w in words if w), key=len, reverse=True):
        if out == word:
            return ""
        if out.startswith(word + " "):
            return out[len(word) + 1:].strip()
    return out


def split_no(text):
    """For an utterance that starts with a no-word, the rest of it: "" for a bare "no",
    "point eleven" for "no, point eleven" (a one-step correction). None if it does not start
    with one, so "point eleven" alone is just a command."""
    words = normalise(text)
    m = _NO_START.match(words)
    if not m:
        return None
    return words[m.end():].strip()
