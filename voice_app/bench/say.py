"""Synthetic speech with macOS `say`, for tests and the synthetic test set.

Mac-only (uses the `say` command). Used only for testing, never by the app.
Results from these clips are SYNTHETIC and must be labelled as such.

    from bench.say import say_wav
    path = say_wav("move up five centimetres", "Daniel", "out.wav", lead_s=0.5, tail_s=1.5)
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

RATE = 16000
# Kate and Serena (named in the build instructions) are not installed on this Mac.
VOICES = ["Daniel", "Samantha", "Karen", "Moira"]


def available() -> bool:
    return shutil.which("say") is not None


def say_audio(text: str, voice: str = "Daniel", rate_wpm: int | None = None) -> np.ndarray:
    """Speak `text` and return it as float32, 16 kHz, mono. No padding."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "s.wav"
        cmd = ["say", "-v", voice, "--file-format=WAVE", "--data-format=LEI16@16000", "-o", str(out)]
        if rate_wpm:
            cmd += ["-r", str(rate_wpm)]
        subprocess.run(cmd + [text], check=True)
        audio, rate = sf.read(str(out), dtype="float32")
    assert rate == RATE
    return trim(audio)


def trim(audio: np.ndarray, floor: float = 0.003) -> np.ndarray:
    """Cut silence off both ends (say adds a little), so padding is exact."""
    loud = np.flatnonzero(np.abs(audio) > floor)
    return audio[loud[0]:loud[-1] + 1] if len(loud) else audio


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * RATE), np.float32)


def say_parts(parts: list, voice: str = "Daniel") -> np.ndarray:
    """Join spoken strings and pauses: ["move left", 1.2, "about ten centimetres"].
    A number is a pause in seconds."""
    pieces = [silence(p) if isinstance(p, (int, float)) else say_audio(p, voice) for p in parts]
    return np.concatenate(pieces)


def pad(audio: np.ndarray, lead_s: float = 0.5, tail_s: float = 1.5) -> np.ndarray:
    """Fixed lead and tail silence, as record_set.py does for real clips."""
    return np.concatenate([silence(lead_s), audio, silence(tail_s)])


def save(audio: np.ndarray, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, RATE, subtype="PCM_16")
    return path


def say_wav(text: str, voice: str, path, lead_s: float = 0.5, tail_s: float = 1.5) -> Path:
    """One padded clip on disk."""
    return save(pad(say_audio(text, voice), lead_s, tail_s), path)
