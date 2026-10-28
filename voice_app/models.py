"""Download model files into models/ (git-ignored), once.

Every plug-in that needs a file from the internet gets it through here, so
all models live in one folder that can be copied to the lab PC, and the log
can say exactly what was downloaded from where.
"""

import tarfile
import urllib.request
from pathlib import Path

MODELS = Path(__file__).resolve().parent / "models"


def path(*parts) -> Path:
    """A path inside models/, e.g. path("silero", "silero_vad.onnx")."""
    return MODELS.joinpath(*parts)


def fetch(url: str, dest: Path) -> Path:
    """Download `url` to `dest` unless it is already there."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)      # only a finished download gets the real name
    return dest


def fetch_tarball(url: str, folder: Path) -> Path:
    """Download and unpack a .tar.bz2 / .tar.gz into `folder` (once)."""
    folder = Path(folder)
    marker = folder / ".unpacked"
    if marker.exists():
        return folder
    archive = fetch(url, folder.parent / Path(url).name)
    with tarfile.open(archive) as tar:
        tar.extractall(folder.parent, filter="data")
    marker.touch()
    archive.unlink()      # keep the unpacked files only
    return folder
