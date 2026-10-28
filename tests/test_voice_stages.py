"""voice/stages.py: the fixed stages. The pure tests need numpy only; the model tests
(`-m model`) download and load the real models."""
import numpy as np
import pytest

from voice import stages
from tests.conftest import frames, words


def test_silence_end_finishes_after_700ms_of_silence_following_speech():
    e = stages.SilenceEnd(700); e.reset()
    assert not e.update(False) and not e.update(True)
    assert not any(e.update(False) for _ in range(21))      # 21 x 32 ms = 672 ms
    assert e.update(False)                                   # 704 ms


def test_silence_end_never_finishes_before_speech():
    e = stages.SilenceEnd(700); e.reset()
    assert not any(e.update(False) for _ in range(100))


def test_release_end_ends_on_release_when_pressed_before_the_utterance():
    e = stages.ReleaseEnd(); e.press(); e.reset()
    assert not e.update(True)
    e.release()
    assert e.update(False)


def test_release_end_falls_back_to_silence_for_a_wake_utterance():
    e = stages.ReleaseEnd(700); e.reset()
    e.update(True)
    assert not any(e.update(False) for _ in range(21))
    assert e.update(False)


def test_release_end_an_old_release_does_not_end_a_wake_utterance():
    e = stages.ReleaseEnd(); e.press(); e.release(); e.reset()
    e.update(True)
    assert not e.update(False)


def test_push_to_talk_fires_once_per_press():
    p = stages.PushToTalk(); p.press()
    assert p.process(np.zeros(512)) and not p.process(np.zeros(512))


def test_trigger_prefers_the_button_and_names_the_fire():
    class NeverWake:                                         # stands in for WakeWord
        enabled = True
        def load(self): pass
        def reset(self): pass
        def process(self, frame): return False
    t = stages.Trigger(NeverWake(), stages.PushToTalk())
    t.ptt.press()
    assert t.process(np.zeros(512)) and t.last_fire == "ptt"
    assert not t.process(np.zeros(512))


@pytest.mark.model
def test_wake_word_fires_on_truffles(clip):
    pytest.importorskip("openwakeword")
    w = stages.WakeWord(threshold=0.5); w.load(); w.reset()
    fired_at = [i for i, f in enumerate(frames(clip("truffles-go-home"))) if w.process(f)]
    assert fired_at and fired_at[0] * 0.032 < 1.6, f"fired at {fired_at}, best score {w.last_score}"


@pytest.mark.model
def test_silero_hears_speech_and_lead_silence(clip):
    pytest.importorskip("onnxruntime")
    v = stages.Silero(); v.load(); v.reset()
    probs = [v.speech_prob(f) for f in frames(clip("move-up-five"))]
    assert max(probs[:10]) < 0.5 and max(probs) > 0.5


@pytest.mark.model
def test_parakeet_transcribes_the_clip(clip):
    pytest.importorskip("sherpa_onnx")
    p = stages.Parakeet(); p.load()
    assert "go home" in words(p.transcribe(clip("truffles-go-home")))
