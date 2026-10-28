"""Scoring: turn a transcript into something comparable, then measure it.

Two scores (docs/05-benchmarking.md):
  - wer(): word error rate, the usual speech-to-text number.
  - command_ok(): did the words that matter to a robot (numbers, units,
    directions, actions) come out identical, in the same order?
    "move up fifteen" vs "move up fifty" is 1 word of WER but a dangerous error.

Both first run the text through normalise(), so "Move up 5 cm." and
"move up five centimetres" compare equal.
"""

import re

# ---- number words ----------------------------------------------------------
UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {w: 10 * (i + 2) for i, w in enumerate(
    "twenty thirty forty fifty sixty seventy eighty ninety".split())}
MINUS_WORDS = ("minus", "negative")

# ---- units: every spelling maps to one canonical word ----------------------
UNIT_WORDS = {
    "cm": "centimetres", "centimeter": "centimetres", "centimeters": "centimetres",
    "centimetre": "centimetres", "centimetres": "centimetres",
    "mm": "millimetres", "millimeter": "millimetres", "millimeters": "millimetres",
    "millimetre": "millimetres", "millimetres": "millimetres",
    "degree": "degrees", "degrees": "degrees",
}

# ---- command-critical words: if any of these differ, the robot would do
# something different. Everything else ("the", "please", "a bit") is ignored.
CRITICAL_WORDS = {
    # directions and axes
    "up", "down", "left", "right", "forward", "forwards", "back", "backward", "backwards",
    "x", "y", "z",
    # stopping
    "stop", "halt", "freeze", "cancel",
    # named places / things the command refers to
    "home", "joint", "base", "gripper", "tool",
    # actions
    "move", "go", "rotate", "turn", "raise", "lift", "open", "close", "draw", "circle", "square",
    # sign (normally already folded into the number, e.g. -60)
    "minus",
    # units (after normalise)
    "centimetres", "millimetres", "degrees",
}
NUMBER = re.compile(r"-?\d+(\.\d+)?$")


def _parse_int_words(tokens, i):
    """Read one whole number from words starting at tokens[i]:
    "one hundred and twenty" -> (120, next index). None if tokens[i] isn't a number word."""
    total, last, j = None, None, i
    while j < len(tokens):
        w = tokens[j]
        if w in UNITS:
            v = UNITS[w]
            if last is None or last in ("hundred", "and"):
                total = (total or 0) + v
            elif last == "tens" and 1 <= v <= 9:       # "sixty" + "three"
                total += v
            else:
                break
            last = "units"
        elif w in TENS:
            if last not in (None, "hundred", "and"):
                break
            total = (total or 0) + TENS[w]
            last = "tens"
        elif w == "hundred" and last == "units" and total is not None and 1 <= total <= 9:
            total *= 100
            last = "hundred"
        elif (w == "and" and last == "hundred" and j + 1 < len(tokens)
              and (tokens[j + 1] in UNITS or tokens[j + 1] in TENS)):
            last = "and"                                 # "hundred AND twenty"
        else:
            break
        j += 1
    if total is None:
        return None
    return total, j


def _digit_of(tok):
    """A single digit (0-9) as a string, from a word or a digit token, else None."""
    if tok in UNITS and UNITS[tok] <= 9:
        return str(UNITS[tok])
    if tok.isdigit() and len(tok) == 1:
        return tok
    return None


def _number_at(tokens, i):
    """Try to read a (possibly negative, possibly decimal) number at tokens[i].
    Returns (text like "-6.5", next index) or None."""
    j, sign = i, ""
    if tokens[j] in MINUS_WORDS and j + 1 < len(tokens):
        sign, j = "-", j + 1
    # integer part: digits already, or number words, or nothing ("point five")
    if re.fullmatch(r"\d+", tokens[j]):
        whole, j = tokens[j], j + 1
    else:
        got = _parse_int_words(tokens, j)
        if got:
            whole, j = str(got[0]), got[1]
        else:
            whole = None
    # decimal part: "point" followed by single digits ("point two five" -> .25)
    if j + 1 < len(tokens) and tokens[j] == "point" and _digit_of(tokens[j + 1]):
        k, frac = j + 1, ""
        while k < len(tokens) and _digit_of(tokens[k]):
            frac += _digit_of(tokens[k])
            k += 1
        return f"{sign}{whole or '0'}.{frac}", k
    if whole is None:
        return None
    return f"{sign}{whole}", j


def normalise(text: str) -> str:
    t = text.lower().replace("°", " degrees ")
    t = t.replace("’", "").replace("'", "")
    t = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", t)         # forty-five -> forty five
    # Split glued tokens (Whisper writes "x10y-60z70", "5cm"): at every letter/digit
    # boundary, and before a minus sign that is followed by a digit.
    t = re.sub(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", " ", t)
    t = re.sub(r"(?<=[a-z])(?=-\d)", " ", t)
    t = re.sub(r"(?<=\d)-", " ", t)                      # 5-6 -> "5 6"
    t = re.sub(r"-(?!\d)", " ", t)                       # a minus not before a digit is just a dash
    t = re.sub(r"(?<![\d\s])-", " ", t)                  # ... and so is one stuck to a word
    t = re.sub(r"(?:(?<=\s)|^)\.(?=\d)", " 0.", t)       # ".5" -> "0.5"
    t = re.sub(r"\.(?!\d)|(?<!\d)\.", " ", t)            # keep a point only inside a number
    t = re.sub(r"[^a-z0-9.\- ]", " ", t)                 # all other punctuation
    tokens = t.split()

    out, i = [], 0
    while i < len(tokens):
        got = _number_at(tokens, i)
        if got:
            out.append(got[0])
            i = got[1]
        else:
            out.append(UNIT_WORDS.get(tokens[i], tokens[i]))
            i += 1
    return " ".join(out)


def wer(ref: str, hyp: str) -> float:
    """Word error rate between two transcripts, after normalising both."""
    import jiwer
    r, h = normalise(ref), normalise(hyp)
    if not r:
        return 0.0 if not h else 1.0
    if not h:
        return 1.0
    return float(jiwer.wer(r, h))


def critical_tokens(text: str) -> list[str]:
    return [w for w in normalise(text).split() if NUMBER.match(w) or w in CRITICAL_WORDS]


def command_ok(ref: str, hyp: str) -> bool:
    """True iff the command-critical tokens match exactly, in order."""
    return critical_tokens(ref) == critical_tokens(hyp)
