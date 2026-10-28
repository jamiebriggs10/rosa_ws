"""Parakeet plug-in: both backends, batch and native streaming.
Clips are synthetic (macOS `say`), so accuracy here is only a smoke test."""

import time

import pytest

from stages.stt.parakeet import Parakeet
from tests.conftest import frames, words

CASES = [
    ("move up five centimetres", ["move", "up", "centimetres"], ["five", "5"]),
    ("go home", ["go", "home"], []),
    ("move left fifteen centimetres", ["move", "left", "centimetres"], ["fifteen", "15"]),
]
VOICES = ["Daniel", "Samantha"]
COMMON = {"mode": "batch", "partial_every_ms": 500, "language": "en", "prompt": ""}


def make(backend, device, **extra):
    stt = Parakeet(COMMON | {"backend": backend} | extra, device)
    stt.load()
    print(f"\n[{backend}] load {stt.load_s:.2f}s, warm-up {stt.warmup_s:.2f}s")
    return stt


def check(text, must, number):
    w = words(text).replace("centimeters", "centimetres")
    assert all(m in w.split() or m in w for m in must), text
    if number:
        assert any(n in w.split() for n in number), text


def run_batch(stt, clip):
    for phrase, must, number in CASES:
        for voice in VOICES:
            stt.start(clip(phrase, voice)[:0])
            for f in frames(clip(phrase, voice)):
                stt.feed(f)
            t0 = time.perf_counter()
            text = stt.finish()
            ms = (time.perf_counter() - t0) * 1000
            print(f"  {voice:9} {ms:6.0f} ms  {text!r}")
            check(text, must, number)


def test_supports():
    assert Parakeet.supports("cpu")[0] and Parakeet.supports("metal")[0]


def test_backend_errors():
    with pytest.raises(RuntimeError, match="nemo"):
        Parakeet(COMMON | {"backend": "nemo"}, "cpu").load()
    with pytest.raises(RuntimeError, match="mlx"):
        Parakeet(COMMON | {"backend": "mlx"}, "cpu").load()


@pytest.mark.model
def test_mlx_batch(clip):
    run_batch(make("mlx", "metal"), clip)


@pytest.mark.model
def test_sherpa_batch(clip):
    run_batch(make("sherpa_onnx", "cpu"), clip)


@pytest.mark.model
def test_mlx_streaming(clip):
    stt = make("mlx", "metal", mode="streaming")
    assert stt.streaming_native
    audio = clip("move up five centimetres", "Daniel")
    stt.start(audio[:0])
    partials, feed_ms = [], []
    for f in frames(audio):
        t0 = time.perf_counter()
        p = stt.feed(f)
        if p is not None:
            feed_ms.append((time.perf_counter() - t0) * 1000)
            partials.append(p)
    t0 = time.perf_counter()
    final = stt.finish()
    print(f"  {len(partials)} partials, feed with work avg {sum(feed_ms)/max(1,len(feed_ms)):.0f} ms, "
          f"finish {(time.perf_counter()-t0)*1000:.0f} ms")
    print("  partials:", partials)
    print("  final:", final)
    assert partials
    check(final, ["move", "up", "centimetres"], ["five", "5"])
    # Batch after streaming must still work (attention restored on close).
    stt.settings["mode"] = "batch"
    stt.start(audio[:0])
    for f in frames(audio):
        stt.feed(f)
    batch = stt.finish()
    # Same words apart from number formatting ("5" vs "five") and the odd stray full stop.
    assert words(batch).split()[:2] == words(final).split()[:2]
    assert words(batch).split()[-1] == words(final).split()[-1]
    check(batch, ["move", "up", "centimetres"], ["five", "5"])
