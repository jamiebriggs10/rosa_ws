"""Whisper-style log-mel features in plain numpy.

Smart Turn's model was trained on the features that Whisper's feature
extractor makes. The official code gets them from `transformers`, but that is
a big library, so this file redoes the same sums in about 50 lines.
tests/test_smart_turn.py checks that the numbers match transformers.

Recipe (for 8 s of 16 kHz audio -> an 80 x 800 picture of the sound):
  1. Normalise the waveform to zero mean and unit variance.
  2. Cut it into 25 ms windows (400 samples) every 10 ms (160 samples),
     multiply each by a Hann window, and take the FFT power.
  3. Add up the FFT bins into 80 mel bands (a scale close to how we hear pitch).
  4. log10, clamp to within 8 of the loudest value, then (x + 4) / 4.
"""

import numpy as np

RATE = 16000
N_FFT = 400
HOP = 160
N_MELS = 80


def _hz_to_mel(f):
    """Slaney mel scale: straight line below 1 kHz, logarithmic above."""
    f = np.asarray(f, dtype=np.float64)
    mel = f / (200.0 / 3)
    log_part = 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (np.log(6.4) / 27.0)
    return np.where(f >= 1000.0, log_part, mel)


def _mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    hz = m * (200.0 / 3)
    log_part = 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0))
    return np.where(m >= 15.0, log_part, hz)


def mel_filters() -> np.ndarray:
    """Triangular filters, shape (201, 80): FFT bin -> mel band (Slaney-normalised)."""
    fft_freqs = np.linspace(0, RATE / 2, 1 + N_FFT // 2)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(RATE / 2), N_MELS + 2))
    widths = np.diff(edges)
    ramps = edges[None, :] - fft_freqs[:, None]                  # (201, 82)
    down = -ramps[:, :-2] / widths[:-1]                          # rising slope
    up = ramps[:, 2:] / widths[1:]                               # falling slope
    filters = np.maximum(0.0, np.minimum(down, up))
    return filters * (2.0 / (edges[2:] - edges[:-2]))[None, :]   # equal area per band


_FILTERS = mel_filters()
_WINDOW = np.hanning(N_FFT + 1)[:-1]    # "periodic" Hann, as Whisper uses


def log_mel(audio: np.ndarray, seconds: float = 8.0) -> np.ndarray:
    """Features for the last `seconds` of `audio`: float32 array (80, 100*seconds).

    Shorter audio is padded with zeros at the START, so the speech sits at the
    right-hand end, as in Smart Turn's own inference.py.
    """
    n = int(seconds * RATE)
    audio = np.asarray(audio, dtype=np.float32)[-n:]
    audio = np.pad(audio, (n - len(audio), 0))

    # transformers normalises the waveform (padding included), not the features.
    audio = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)

    # Reflect-pad half a window each side so frames are centred on their time.
    padded = np.pad(audio, (N_FFT // 2, N_FFT // 2), mode="reflect")
    n_frames = 1 + (len(padded) - N_FFT) // HOP
    index = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    power = np.abs(np.fft.rfft(padded[index] * _WINDOW, axis=1)) ** 2    # (frames, 201)

    mel = _FILTERS.T @ power.T                                           # (80, frames)
    spec = np.log10(np.maximum(mel, 1e-10))
    spec = spec[:, :-1]                      # Whisper drops the last frame -> 800
    spec = np.maximum(spec, spec.max() - 8.0)
    return ((spec + 4.0) / 4.0).astype(np.float32)
