"""Audio sources. See audio/README.md and docs/01-architecture.md.

MicSource and FileSource have the same interface (start, stop, frames,
backlog, kind), so the pipeline can't tell them apart.
"""

from audio.file import FileSource, load_wav
from audio.framer import FRAME, RATE, Framer, rms

__all__ = ["FileSource", "load_wav", "Framer", "rms", "FRAME", "RATE", "MicSource"]


def __getattr__(name):
    # MicSource is imported lazily: sounddevice needs PortAudio, which a
    # headless benchmark machine might not have.
    if name == "MicSource":
        from audio.mic import MicSource
        return MicSource
    raise AttributeError(name)
