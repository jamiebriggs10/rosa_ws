"""Keep the tests away from the real settings file: config reads ~/.config/rosa_ws/settings.json
at import, and an operator's saved limits or model must not change what the tests see."""
import os
import tempfile

os.environ["ROSA_SETTINGS"] = os.path.join(tempfile.mkdtemp(), "settings.json")   # does not exist: defaults apply


# -- speech clips for the voice tests (tests/clips/*.wav: 16 kHz mono, 0.5 s lead, 1.5 s tail) --
import pathlib  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

CLIPS = pathlib.Path(__file__).resolve().parent / "clips"


@pytest.fixture
def clip():
    """clip("stop") -> float32 16 kHz array of tests/clips/stop.wav."""
    pytest.importorskip("soundfile")
    from voice import audio
    return lambda name: audio.load_wav(CLIPS / f"{name}.wav")


def frames(audio, n=512):
    """Cut audio into whole 512-sample frames (the pipeline's frame size), the last one padded."""
    audio = np.concatenate([audio, np.zeros((-len(audio)) % n, np.float32)])
    return [audio[i:i + n] for i in range(0, len(audio), n)]


def words(text):
    """Lower case, letters and digits only: for loose transcript comparisons."""
    return " ".join("".join(c for c in text.lower() if c.isalnum() or c.isspace()).split())
