"""whisper.cpp plug-in: both drivers, on synthetic `say` speech.

Needs the model files (downloaded on first run) so it is marked `model`.
Synthetic voices: timings and accuracy here are indicative only.
"""

import time

import pytest

from stages.stt.whisper_cpp import WhisperCpp
from tests.conftest import frames, words

pytestmark = pytest.mark.model

PHRASES = {
    "move up five centimetres": ["move", "up", "centimeters|centimetres|cm"],
    "go home": ["go", "home"],
    "rotate joint six by ninety degrees": ["rotate", "joint", "degrees"],
}
NUMBERS = {"move up five centimetres": ["5|five"],
           "rotate joint six by ninety degrees": ["6|six", "90|ninety"]}
SETTINGS = {"mode": "batch", "partial_every_ms": 500, "language": "en", "prompt": "",
            "server_url": "http://127.0.0.1:8179"}    # own port: never clashes with a real server


def has(text, options):
    return any(o in words(text).split() or o in words(text) for o in options.split("|"))


@pytest.fixture(scope="module", params=["bindings", "server"])
def engine(request):
    stt = WhisperCpp(SETTINGS | {"driver": request.param}, "metal")
    t = time.perf_counter()
    stt.load()
    print(f"\n[{request.param}] load {time.perf_counter() - t:.2f} s")
    yield stt
    stt.close()


def run(stt, audio):
    stt.start(audio[:0])
    for f in frames(audio):
        stt.feed(f)
    return stt.finish()


def test_supports():
    assert WhisperCpp.supports("metal")[0]


@pytest.mark.parametrize("voice", ["Daniel", "Samantha"])
@pytest.mark.parametrize("phrase", list(PHRASES))
def test_phrases(engine, clip, phrase, voice):
    audio = clip(phrase, voice=voice)
    t = time.perf_counter()
    text = run(engine, audio)
    ms = (time.perf_counter() - t) * 1000
    print(f"[{engine.settings['driver']}] {voice} {len(audio) / 16000:.1f}s audio, {ms:.0f} ms: {text!r}")
    for key in PHRASES[phrase] + NUMBERS.get(phrase, []):
        assert has(text, key), f"{key!r} missing from {text!r}"


def test_pseudo_streaming(engine, clip):
    stt = WhisperCpp(engine.settings | {"mode": "streaming", "partial_every_ms": 500}, "metal")
    stt._model, stt._server = engine._model, None     # share the loaded model
    audio = clip(["move up", 0.3, "five centimetres"])
    assert len(audio) >= 3 * 16000
    stt.start(audio[:0])
    partials = [p for p in (stt.feed(f) for f in frames(audio)) if p]
    final = stt.finish()
    print(f"[{engine.settings['driver']}] partials: {partials}; final: {final!r}")
    assert partials
    assert "move" in words(final)
