"""The model files, downloaded once into models/ (git-ignored, except Jamie's own wake-word model).

    python -m voice.models            # fetch everything the voice package needs

Everything the voice package reads from the internet comes through here, so all the models
live in one folder that can be copied to another machine, and the log can say exactly what
was downloaded from where.
"""

import sys
import tarfile
import urllib.request
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"
WAKE_MODEL = MODELS / "truffles.onnx"                 # trained in Colab on 2026-10-08 (openWakeWord); committed
SILERO_URL = "https://raw.githubusercontent.com/snakers4/silero-vad/v5.1.2/src/silero_vad/data/silero_vad.onnx"
PARAKEET_NAME = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
PARAKEET_URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{PARAKEET_NAME}.tar.bz2"
PIPER_VOICE = "en_GB-alan-medium"


def path(*parts):
    return MODELS.joinpath(*parts)


def fetch(url, dest):
    """Download `url` to `dest` unless it is already there."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)                                   # only a finished download gets the real name
    return dest


def fetch_tarball(url, folder):
    """Download and unpack a .tar.bz2 into `folder` (once)."""
    folder = Path(folder)
    marker = folder / ".unpacked"
    if marker.exists():
        return folder
    archive = fetch(url, folder.parent / Path(url).name)
    with tarfile.open(archive) as tar:
        tar.extractall(folder.parent, filter="data")
    marker.touch()
    archive.unlink()
    return folder


def silero():
    return fetch(SILERO_URL, path("silero", "silero_vad.onnx"))


def parakeet():
    return fetch_tarball(PARAKEET_URL, path(PARAKEET_NAME))


def openwakeword_dir():
    """openWakeWord's two shared models (mel spectrogram, embedding), beside our wake word."""
    folder = path("openwakeword")
    if not (folder / "melspectrogram.onnx").exists() or not (folder / "embedding_model.onnx").exists():
        import openwakeword
        folder.mkdir(parents=True, exist_ok=True)
        openwakeword.utils.download_models(model_names=[], target_directory=str(folder))
    return folder


def piper_voice(name=PIPER_VOICE):
    """A Piper voice (<name>.onnx + .json) in models/piper."""
    folder = path("piper")
    if not (folder / f"{name}.onnx").exists():
        from piper.download_voices import download_voice
        folder.mkdir(parents=True, exist_ok=True)
        download_voice(name, folder)
    return folder / f"{name}.onnx"


def download_all(voice=PIPER_VOICE):
    silero(); parakeet(); openwakeword_dir(); piper_voice(voice)
    if not WAKE_MODEL.exists():
        print(f"missing {WAKE_MODEL}: the wake word model is part of the repository", file=sys.stderr)
    print(f"models ready in {MODELS}")


if __name__ == "__main__":
    download_all(sys.argv[1] if len(sys.argv) > 1 else PIPER_VOICE)
