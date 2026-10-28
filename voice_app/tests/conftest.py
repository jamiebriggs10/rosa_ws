"""Shared test helpers: synthetic speech clips made with macOS `say`, cached.

    def test_x(clip):
        audio = clip("move up five centimetres")             # float32, 16 kHz, padded
        audio = clip(["move left", 1.2, "ten centimetres"])  # with a 1.2 s pause
        path = clip.path("go home", voice="Karen")           # a WAV file on disk
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench import say  # noqa: E402

CACHE = ROOT / "tests" / ".clips"


class Clips:
    def path(self, text, voice="Daniel", lead_s=0.5, tail_s=1.5) -> Path:
        parts = text if isinstance(text, list) else [text]
        key = hashlib.md5(repr((parts, voice, lead_s, tail_s)).encode()).hexdigest()[:12]
        out = CACHE / f"{voice}-{key}.wav"
        if not out.exists():
            say.save(say.pad(say.say_parts(parts, voice), lead_s, tail_s), out)
        return out

    def __call__(self, text, voice="Daniel", lead_s=0.5, tail_s=1.5) -> np.ndarray:
        audio, _ = sf.read(str(self.path(text, voice, lead_s, tail_s)), dtype="float32")
        return audio


@pytest.fixture(scope="session")
def clip():
    if not say.available():
        pytest.skip("macOS `say` not available")
    return Clips()


def frames(audio, n=512):
    """Cut audio into whole 512-sample frames (the pipeline's frame size)."""
    audio = np.concatenate([audio, np.zeros((-len(audio)) % n, np.float32)])
    return [audio[i:i + n] for i in range(0, len(audio), n)]


def words(text: str) -> str:
    """Lower case, letters/digits only: for loose transcript comparisons."""
    return " ".join("".join(c for c in text.lower() if c.isalnum() or c.isspace()).split())
