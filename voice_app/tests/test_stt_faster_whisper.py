"""faster-whisper option: real model, synthetic `say` speech (labelled synthetic)."""

import time

import pytest

from stages.stt.faster_whisper_engine import FasterWhisper
from tests.conftest import frames, words

PROMPT = "Rosa, tool, home, joint, gripper, x, y, z, centimetres, millimetres, degrees, up, down, left, right."
COMMON = {"mode": "batch", "partial_every_ms": 500, "language": "en", "prompt": PROMPT}

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def engine():
    e = FasterWhisper(dict(COMMON), "metal")
    t = time.time()
    e.load()
    print(f"\nload + warm-up: {time.time() - t:.2f} s")
    return e


def run(engine, audio):
    engine.start(audio[:0])
    for f in frames(audio):
        engine.feed(f)
    return engine.finish()


def test_supports():
    assert FasterWhisper.supports("cpu") == (True, "")
    assert FasterWhisper.supports("cuda") == (True, "")


@pytest.mark.parametrize("voice", ["Daniel", "Samantha"])
def test_move_up(engine, clip, voice):
    audio = clip("move up five centimetres", voice)
    t = time.time()
    text = run(engine, audio)
    print(f"{voice}: {text!r}  ({(time.time() - t) * 1000:.0f} ms for {len(audio) / 16000:.1f} s)")
    w = words(text)
    assert "move" in w and "up" in w
    assert "five" in w or "5" in w.split()
    assert "centimetres" in w or "centimeters" in w


@pytest.mark.parametrize("voice", ["Daniel", "Samantha"])
def test_go_home(engine, clip, voice):
    text = run(engine, clip("go home", voice))
    print(f"{voice}: {text!r}")
    assert "home" in words(text)


@pytest.mark.parametrize("voice", ["Daniel", "Samantha"])
def test_coordinates(engine, clip, voice):
    text = run(engine, clip("move to x ten y minus sixty z seventy", voice))
    print(f"{voice}: {text!r}")
    # Whisper often glues these into one token ("x10y-60z70"), so look for
    # digits or number words anywhere in the text, not as separate words.
    w = words(text)
    assert "move" in w
    assert "10" in w or "ten" in w
    assert "60" in w or "sixty" in w
    assert "70" in w or "seventy" in w


def test_pseudo_streaming_partials(clip):
    e = FasterWhisper(dict(COMMON, mode="streaming"), "cpu")
    e.load()
    audio = clip(["move up", 0.3, "five centimetres", 0.3, "then go home"])
    print(f"clip length {len(audio) / 16000:.1f} s")
    e.start(audio[:0])
    partials = [p for p in (e.feed(f) for f in frames(audio)) if p]
    final = e.finish()
    print("partials:", partials, "final:", final)
    assert len(partials) >= 1
    assert "home" in words(final)


def test_beam_size_timing(clip):
    audio = clip("move to x ten y minus sixty z seventy")
    for beam in (1, 5):
        e = FasterWhisper(dict(COMMON, beam_size=beam), "cpu")
        e.load()
        t = time.time()
        for _ in range(3):
            text = run(e, audio)
        print(f"beam {beam}: {(time.time() - t) / 3 * 1000:.0f} ms per clip, {text!r}")
