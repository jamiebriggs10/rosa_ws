"""Pure-logic tests for silence, adaptive and ptt_release (no audio, no models).

Each update() call stands for one 32 ms frame, so 22 silent frames = 704 ms.
"""

import numpy as np

from stages.endpoint.adaptive import Adaptive, ends_in_bare_number, normalise
from stages.endpoint.ptt_release import PttRelease
from stages.endpoint.silence import Silence

FRAME = np.zeros(512, np.float32)


def run(ep, pattern, text=""):
    """Feed 'S' (speech) and '.' (silence) frames; return the index where it
    first said finished, or None."""
    for i, c in enumerate(pattern):
        if ep.update(FRAME, c == "S", text):
            return i
    return None


def test_silence_finishes_after_700ms():
    ep = Silence({}, "cpu")
    assert run(ep, "SSS" + "." * 21) is None      # 672 ms: not yet
    ep.reset()
    assert run(ep, "SSS" + "." * 22) == 24        # 704 ms: done


def test_silence_never_finishes_before_speech():
    assert run(Silence({}, "cpu"), "." * 200) is None


def test_silence_speech_resets_count():
    ep = Silence({}, "cpu")
    assert run(ep, "S" + "." * 20 + "S" + "." * 20) is None


def test_silence_reset():
    ep = Silence({}, "cpu")
    run(ep, "S" + "." * 10)
    ep.reset()
    assert run(ep, "." * 100) is None             # speech heard before is forgotten


def test_adaptive_short_when_complete():
    ep = Adaptive({}, "cpu")
    assert run(ep, "S" + "." * 13, "go home") == 13   # 13 frames = 416 ms >= 400
    assert ep.last_reason == "complete"


def test_adaptive_long_after_continue_word():
    ep = Adaptive({}, "cpu")
    assert run(ep, "S" + "." * 40, "move left by") is None
    assert ep.last_reason == "continue word 'by'"
    assert run(ep, "." * 10, "move left by") == 6    # 47 silent frames = 1504 ms


def test_adaptive_bare_number_vs_unit():
    ep = Adaptive({}, "cpu")
    assert run(ep, "S" + "." * 30, "move up five") is None
    assert ep.last_reason == "bare number"
    ep.reset()
    assert run(ep, "S" + "." * 30, "move up five centimetres") is not None
    assert ep.last_reason == "complete"


def test_adaptive_fallback_without_text():
    ep = Adaptive({}, "cpu")
    assert run(ep, "S" + "." * 21, "") is None
    assert ep.last_reason == "no partial text"
    assert run(ep, ".", "") is not None               # 22 frames >= 700 ms


def test_adaptive_number_forms_and_punctuation():
    for text in ["-60", "0.5", "minus sixty", "Move, to 5."]:
        assert ends_in_bare_number(normalise(text)), text
    assert not ends_in_bare_number(normalise("go home."))
    assert normalise("Move to the, A!") == ["move", "to", "the", "a"]


def test_ptt_press_frames_release():
    ep = PttRelease({}, "cpu")
    ep.on_event("ptt_down")
    ep.reset()
    assert run(ep, "SS..S") is None               # held: nothing ends it
    ep.on_event("ptt_up")
    assert ep.update(FRAME, True, "") is True     # next update finishes


def test_ptt_release_before_any_speech_still_finishes():
    ep = PttRelease({}, "cpu")
    ep.on_event("ptt_down")
    ep.reset()
    ep.on_event("ptt_up")
    assert ep.update(FRAME, False, "") is True


def test_ptt_never_pressed_falls_back_to_silence():
    ep = PttRelease({}, "cpu")
    ep.reset()
    assert run(ep, "." * 100) is None
    assert run(ep, "S" + "." * 21) is None
    assert run(ep, ".") is not None


def test_ptt_old_release_does_not_end_wake_utterance():
    ep = PttRelease({}, "cpu")
    ep.on_event("ptt_down")
    ep.reset()
    ep.on_event("ptt_up")
    assert ep.update(FRAME, True, "") is True     # first utterance ends
    ep.reset()                                    # next one, by wake word
    assert ep.update(FRAME, True, "") is False
    assert run(ep, "." * 21) is None
    assert run(ep, ".") is not None


def test_ptt_press_during_wake_utterance_takes_over():
    ep = PttRelease({}, "cpu")
    ep.reset()
    run(ep, "SS")
    ep.on_event("ptt_down")
    assert run(ep, "." * 50) is None              # silence no longer ends it
    ep.on_event("ptt_up")
    assert ep.update(FRAME, False, "") is True
