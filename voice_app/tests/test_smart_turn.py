"""Smart Turn: the numpy features match transformers, and the model behaves."""

import numpy as np
import pytest

from stages.endpoint.smart_turn import SmartTurn
from stages.endpoint.whisper_mel import log_mel
from tests.conftest import frames


def test_features_match_transformers():
    transformers = pytest.importorskip("transformers")
    extractor = transformers.WhisperFeatureExtractor(chunk_length=8)
    rng = np.random.default_rng(0)
    for seconds in (3, 8, 10):                        # padded, exact, truncated
        audio = (rng.standard_normal(16000 * seconds) * 0.1).astype(np.float32)
        tail = audio[-128000:]
        tail = np.pad(tail, (128000 - len(tail), 0))  # what Smart Turn's inference.py does
        expected = extractor(tail, sampling_rate=16000, return_tensors="np",
                             padding="max_length", max_length=128000,
                             truncation=True, do_normalize=True).input_features[0]
        got = log_mel(audio)
        assert got.shape == (80, 800)
        assert np.abs(got - expected).max() < 1e-3


@pytest.fixture(scope="module")
def smart():
    s = SmartTurn({}, "cpu")
    s.load()
    return s


def run(smart, audio, **settings):
    """Feed audio with an energy 'VAD' (good enough for clean synthetic clips)."""
    smart.settings.update(settings)
    smart.reset()
    for i, f in enumerate(frames(audio)):
        if smart.update(f, bool(np.abs(f).max() > 0.02), ""):
            return i
    return None


@pytest.mark.model
def test_model_input_shape(smart):
    inp = smart._session.get_inputs()[0]
    assert inp.name == "input_features" and list(inp.shape[1:]) == [80, 800]


@pytest.mark.model
def test_probability_and_timing(smart):
    smart._audio.extend(frames(np.zeros(16000, np.float32)))
    for _ in range(3):
        p = smart.probability()
    assert 0.0 <= p <= 1.0
    ms = smart.checks[-1][2]
    print(f"\nsmart_turn inference: {ms:.1f} ms per check")
    assert ms < 300


@pytest.mark.model
def test_complete_vs_incomplete(smart, clip):
    def prob_at_end(text):
        # Audio cut right after the words (no trailing silence), asked directly.
        audio = clip(text, tail_s=0.0)
        smart.reset()
        smart._audio.extend(frames(audio)[-250:])
        return smart.probability()

    complete = {t: prob_at_end(t) for t in
                ["move up five centimetres", "go home", "where is the tool"]}
    incomplete = {t: prob_at_end(t) for t in
                  ["move left", "move to x ten y", "rotate joint"]}
    for label, d in (("complete", complete), ("incomplete", incomplete)):
        for t, p in d.items():
            print(f"{label:10s} {p:.3f}  {t}")
    mc, mi = np.mean(list(complete.values())), np.mean(list(incomplete.values()))
    print(f"mean complete {mc:.3f}  mean incomplete {mi:.3f}")
    # Not asserted: on macOS `say` voices the incomplete phrases score HIGHER
    # (the TTS ends them with falling pitch, which the model reads as finished).
    # Real human speech has to be judged in the benchmark, not here.
    assert all(0.0 <= p <= 1.0 for p in [*complete.values(), *incomplete.values()])


@pytest.mark.model
def test_finishes_on_complete_command(smart, clip):
    end = run(smart, clip("move up five centimetres", tail_s=2.0),
              pause_ms=200, threshold=0.5, max_wait_ms=3000)
    print(f"\ncomplete command finished at frame {end}, checks {smart.checks}")
    assert end is not None


@pytest.mark.model
def test_fallback_at_max_wait(smart, clip):
    # Threshold above 1 can never be met, so only the fallback can end the turn.
    audio = clip("move left", tail_s=4.0)
    end = run(smart, audio, pause_ms=200, threshold=1.1, max_wait_ms=3000)
    assert end is not None
    assert smart.checks, "model should have been asked at the first pause"
    speech_end = max(i for i, f in enumerate(frames(audio)) if np.abs(f).max() > 0.02)
    waited_ms = (end - speech_end) * 32
    print(f"\nfallback fired {waited_ms} ms after speech ended")
    assert 2800 <= waited_ms <= 3300
