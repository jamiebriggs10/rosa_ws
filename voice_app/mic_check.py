"""Phase 0 microphone check: record 3 s, save a WAV, play it back.

    ./.venv/bin/python mic_check.py            # system default mic
    ./.venv/bin/python mic_check.py "MacBook"  # a mic whose name contains "MacBook"

The first time, macOS asks whether the terminal (or VS Code) may use the
microphone: allow it. If the recording is silent, check System Settings >
Privacy & Security > Microphone.
"""

import sys

import numpy as np
import sounddevice as sd
import soundfile as sf

import config
from audio.mic import find_device, record

name = sys.argv[1] if len(sys.argv) > 1 else None
dev = find_device(name)
print("input device:", sd.query_devices(dev, "input")["name"])
print("Recording 3 seconds: say something...")
audio, rate = record(3.0, name)
peak = float(np.abs(audio).max())
out = config.ROOT / "runs" / "mic_check.wav"
out.parent.mkdir(exist_ok=True)
sf.write(out, audio, rate)
print(f"saved {out} ({rate} Hz), peak level {peak:.3f}" + ("  <-- silent? check mic permission" if peak < 0.01 else ""))
print("Playing it back...")
sd.play(audio, rate)
sd.wait()
