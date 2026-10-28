"""voice/audio.py: frames from a WAV or the microphone. Needs soxr and soundfile (the venv)."""
import numpy as np
import pytest

pytest.importorskip("soxr")
pytest.importorskip("soundfile")
from voice import audio  # noqa: E402


def test_framer_resamples_48k_to_512_sample_frames():
    f = audio.Framer(48000)
    frames = f.push(np.zeros(48000, np.float32)) + f.flush()       # 1 s -> 16000 samples -> 31.25 frames, padded
    assert len(frames) == 32 and all(len(x) == 512 for x in frames)


def test_file_source_yields_the_clip_in_frames():
    src = audio.FileSource("tests/clips/stop.wav", pad_end_s=0.5)
    src.start()
    frames = list(src.frames())
    assert abs(src.duration_s - (2.66 + 0.5)) < 0.02
    assert len(frames) == int(np.ceil(src.duration_s / audio.FRAME_S))
    assert audio.rms(frames[0]) < 0.01                               # the 0.5 s lead is silent


def test_mic_source_names_the_missing_device():
    pytest.importorskip("sounddevice")
    with pytest.raises(ValueError, match="no input device matching 'NoSuchMic'"):
        audio.MicSource("NoSuchMic")


def test_list_devices_has_input_and_output_names():
    pytest.importorskip("sounddevice")
    devices = audio.list_devices()
    assert set(devices) == {"input", "output"} and all(isinstance(n, str) for n in devices["input"] + devices["output"])
