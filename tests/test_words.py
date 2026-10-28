"""voice/words.py: what a spoken phrase means. Pure Python.
    python3 -m pytest tests/test_words.py"""
import pytest

from voice import words


@pytest.mark.parametrize("text,stop", [("Stop.", True), ("stop stop", True), ("HALT!", True), ("freeze", True),
                                       ("stop the move", False), ("", False), ("   ", False)])
def test_is_stop(text, stop):
    assert words.is_stop(text) is stop


def test_only_confirm_and_go_ahead_are_yes():
    assert words.says_yes("Confirm.") and words.says_yes("go ahead")
    assert not words.says_yes("yes") and not words.says_yes("okay") and not words.says_yes("confirm it now")
    assert words.says_no("Cancel") and words.says_no("no.")
    assert not words.says_no("cancel the second move")


@pytest.mark.parametrize("text,rest", [("no", ""), ("No, point eleven", "point eleven"), ("cancel", ""),
                                       ("no actually go home", "go home"), ("nope, rather up five", "up five"),
                                       ("No. Instead, point twelve.", "point twelve"),
                                       ("point eleven", None), ("nothing here", None), ("", None)])
def test_split_no(text, rest):
    assert words.split_no(text) == rest


@pytest.mark.parametrize("text,out", [("Truffles, confirm", "confirm"), ("rosa confirm", "confirm"), ("hey rosa, no point eleven", "no point eleven"),
                                      ("confirm", "confirm"), ("truffles", ""), ("the truffles are nice", "the truffles are nice")])
def test_a_leading_wake_word_is_stripped(text, out):
    assert words.strip_wake(text) == out


def test_truffles_is_a_wake_word():
    assert "truffles" in words.WAKE_WORDS
    assert words.strip_wake("truffles go home") == "go home"
