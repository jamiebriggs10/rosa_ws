"""Tests for the two VAD options, on synthetic `say` speech (label as synthetic)."""

import time

import numpy as np
import pytest

from tests.conftest import frames
from stages.vad.silero import Silero
from stages.vad.webrtc import WebRTC

pytestmark = pytest.mark.model
LEAD_S = 0.5


@pytest.fixture(params=[Silero, WebRTC], ids=["silero", "webrtc"])
def vad(request):
    v = request.param({}, "cpu")
    v.load()
    return v


def probs(vad, audio):
    vad.reset()
    return np.array([vad.speech_prob(f) for f in frames(audio)])


def test_speech_and_lead_silence(vad, clip):
    p = probs(vad, clip("move up five centimetres"))
    lead = int(LEAD_S * 16000 / 512) - 1          # frames wholly inside the lead
    assert (p[:lead] < vad.threshold).all()
    # the speech sits between the 0.5 s lead and 1.5 s tail
    speech = p[lead + 2: len(p) - int(1.5 * 16000 / 512) - 2]
    frac = (speech >= vad.threshold).mean()
    print(f"{vad.name}: {frac:.0%} of speech frames above threshold")
    assert frac > 0.6


def test_digital_silence(vad):
    p = probs(vad, np.zeros(16000, np.float32))
    assert (p < vad.threshold).all()


def test_timing_and_noise(vad):
    noise = (np.random.default_rng(0).standard_normal(32000) * 0.01).astype(np.float32)
    fs = frames(noise)
    vad.reset()
    t = time.perf_counter()
    p = np.array([vad.speech_prob(f) for f in fs])
    ms = (time.perf_counter() - t) * 1000 / len(fs)
    print(f"{vad.name}: {ms:.3f} ms/frame; white noise 0.01: max prob {p.max():.2f}, "
          f"{(p >= vad.threshold).mean():.0%} frames above threshold")
    assert ms < 5


def test_reset_repeats(vad, clip):
    audio = clip("go home")
    assert np.array_equal(probs(vad, audio), probs(vad, audio))
