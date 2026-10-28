"""Moonshine engine: batch and native streaming on synthetic speech."""

import time

import pytest

from stages.stt.moonshine_engine import Moonshine
from tests.conftest import frames, words

PHRASES = [
    ("move up five centimetres", ["move", "up", ["five", "5"], ["centimetres", "centimeters", "cm"]]),
    ("go home", ["go", "home"]),
    ("open the gripper", ["open", "gripper"]),
]
VOICES = ["Daniel", "Samantha"]
COMMON = {"partial_every_ms": 500, "language": "en", "prompt": ""}


def make(mode):
    stt = Moonshine({**COMMON, "mode": mode}, "cpu")
    t = time.time()
    stt.load()
    print(f"\nload ({mode}): {(time.time() - t) * 1000:.0f} ms")
    return stt


def check(text, keys):
    got = words(text)
    for k in keys:
        opts = k if isinstance(k, list) else [k]
        assert any(o in got for o in opts), f"{opts} not in {got!r}"


def run(stt, audio):
    stt.reset()
    stt.start(audio[:0])
    partials, feed_ms = [], []
    for f in frames(audio):
        t = time.time()
        p = stt.feed(f)
        feed_ms.append((time.time() - t) * 1000)
        if p:
            partials.append(p)
    t = time.time()
    final = stt.finish()
    return final, partials, feed_ms, (time.time() - t) * 1000


def test_supports():
    assert Moonshine.supports("cpu")[0]


@pytest.fixture(scope="module")
def batch():
    return make("batch")


@pytest.fixture(scope="module")
def streaming():
    return make("streaming")


@pytest.mark.model
@pytest.mark.parametrize("voice", VOICES)
@pytest.mark.parametrize("text,keys", PHRASES)
def test_batch(batch, clip, voice, text, keys):
    final, partials, _, fin_ms = run(batch, clip(text, voice))
    print(f"batch {voice}: {final!r} finish {fin_ms:.0f} ms")
    assert partials == []
    check(final, keys)


@pytest.mark.model
@pytest.mark.parametrize("voice", VOICES)
@pytest.mark.parametrize("text,keys", PHRASES)
def test_streaming(streaming, clip, voice, text, keys):
    final, partials, feed_ms, fin_ms = run(streaming, clip(text, voice))
    print(f"stream {voice}: {final!r} partials={len(partials)} finish {fin_ms:.0f} ms "
          f"feed mean {sum(feed_ms) / len(feed_ms):.2f} max {max(feed_ms):.0f} ms")
    check(final, keys)


@pytest.mark.model
def test_streaming_gives_partials_and_all_lines(streaming, clip):
    audio = clip(["move up five centimetres", 1.2, "then open the gripper"])
    final, partials, _, _ = run(streaming, audio)
    print(f"long: {final!r} partials={partials}")
    assert len(partials) >= 1
    check(final, ["move", "gripper"])
