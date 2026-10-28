"""Tests for the trigger options. The wake word tests use SYNTHETIC speech (macOS `say`)."""

import time

import numpy as np
import pytest

from tests.conftest import frames
from stages.trigger.both import Both
from stages.trigger.push_to_talk import PushToTalk
from stages.trigger.wake_word import WakeWord

VOICES = ["Daniel", "Samantha", "Karen", "Moira"]
QUIET = np.zeros(512, np.float32)


def run(trigger, audio):
    """Feed audio; return (number of fires, peak score)."""
    fires, peak = 0, 0.0
    for f in frames(audio):
        if trigger.process(f):
            fires += 1
        peak = max(peak, getattr(trigger, "last_score", 0.0))
    return fires, peak


@pytest.fixture(scope="module")
def wake():
    w = WakeWord({}, "cpu")
    w.load()
    return w


def test_ptt_fires_once_after_down():
    t = PushToTalk({}, "cpu")
    assert not t.process(QUIET)
    t.on_event("ptt_down")
    assert t.process(QUIET) and t.last_fire == "ptt"
    assert not t.process(QUIET)
    t.on_event("ptt_down")
    t.reset()
    assert not t.process(QUIET)
    t.on_event("ptt_up")
    assert not t.process(QUIET)


def test_both_forwards_ptt():
    t = Both({"openwakeword": {"threshold": 0.7}}, "cpu")
    assert t.wake.settings["threshold"] == 0.7
    t.on_event("ptt_down")
    assert t.ptt._pressed
    assert t.process(QUIET) and t.last_fire == "ptt"
    assert not t.process(QUIET)


def test_knob_defaults_match_settings():
    assert {k: v["default"] for k, v in WakeWord.knobs.items()} == {
        "model": "hey_jarvis", "threshold": 0.5, "patience": 1, "cooldown_s": 1.5}


@pytest.mark.model
def test_wake_word_voices(wake, clip):
    results = {}
    for v in VOICES:
        wake.reset(); wake._cooldown_left = 0
        results[v] = run(wake, clip("hey Jarvis", voice=v))
    print("hey Jarvis (fires, peak):", results)
    assert sum(f > 0 for f, _ in results.values()) >= 2


@pytest.mark.model
def test_wake_word_with_command(wake, clip):
    results = {}
    for v in VOICES:
        wake.reset(); wake._cooldown_left = 0
        results[v] = run(wake, clip("hey Jarvis, go home", voice=v))
    print("hey Jarvis, go home (fires, peak):", results)
    assert sum(f > 0 for f, _ in results.values()) >= 2


@pytest.mark.model
def test_no_false_triggers(wake, clip):
    wake.reset(); wake._cooldown_left = 0
    for v in VOICES:
        assert run(wake, clip("move up five centimetres", voice=v))[0] == 0
    rng = np.random.default_rng(0)
    for audio in (np.zeros(160000, np.float32),
                  (rng.standard_normal(160000) * 0.003).astype(np.float32)):
        wake.reset()
        assert run(wake, audio)[0] == 0


@pytest.mark.model
def test_off_cooldown_and_timing(wake, clip):
    audio = clip("hey Jarvis", voice="Daniel")
    wake.reset(); wake._cooldown_left = 0
    wake.on_event("wake_off")
    assert run(wake, audio)[0] == 0
    wake.on_event("wake_on")
    wake.reset()
    assert run(wake, audio)[0] >= 1
    # cooldown survives reset(): the same phrase straight away must not fire again
    wake.reset()
    assert run(wake, audio[: 16000])[0] == 0
    # timing per 80 ms prediction
    wake._cooldown_left = 0
    chunk = np.zeros(1280, np.int16)
    t0 = time.perf_counter()
    for _ in range(100):
        wake.model.predict(chunk)
    print("ms per 80 ms prediction:", (time.perf_counter() - t0) * 10)
