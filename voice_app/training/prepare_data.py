"""Download everything wake-word training needs (runs on Google Colab, not the Mac).

Called by truffles_wake_word.ipynb:   python prepare_data.py [--quick]

  mit_rirs/          room echoes (impulse responses), to make clips sound like real rooms
  background_16k/    everyday background audio (speech, music, noise) from AudioSet
  acav_features.npy  ~2,000 h of "not the wake word" audio, already passed through
                     openWakeWord's frozen "ear" (17 GB; --quick takes only the start)
  validation_set_features.npy  ~11 h of other audio, used to count false wake-ups

Written 8 Oct 2026 because the official notebook's downloads had broken: its
AudioSet .tar link is gone (now Parquet files) and its music source is a
`datasets` loading script that new versions no longer run. This uses
huggingface_hub directly instead of `datasets`.
"""

import argparse
import io
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests
import soundfile as sf
from huggingface_hub import hf_hub_download, snapshot_download
from scipy.signal import resample_poly

FEATURES = "davidscripka/openwakeword_features"


def room_echoes():
    out = Path("mit_rirs")
    if out.exists() and any(out.glob("*.wav")):
        return
    folder = snapshot_download("davidscripka/MIT_environmental_impulse_responses", repo_type="dataset",
                               allow_patterns=["16khz/*"], local_dir="mit_rirs_download")
    out.mkdir(exist_ok=True)
    for f in Path(folder, "16khz").glob("*.wav"):
        shutil.copy(f, out / f.name)
    print("room echoes:", len(list(out.glob("*.wav"))))


def to_16k_mono(audio, rate):
    """Training expects 16 kHz mono, like the prototype."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != 16000:
        g = np.gcd(int(rate), 16000)
        audio = resample_poly(audio, 16000 // g, int(rate) // g)
    return np.clip(audio, -1, 1)


def background(n_files, max_clips):
    """AudioSet's balanced set: 10 s YouTube clips of every kind of sound, music included."""
    out = Path("background_16k")
    out.mkdir(exist_ok=True)
    count = len(list(out.glob("*.wav")))
    for i in range(n_files):
        if count >= max_clips:
            break
        path = hf_hub_download("agkphysics/AudioSet", f"data/bal_train/{i:02d}.parquet",
                               repo_type="dataset", local_dir="audioset")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=["video_id", "audio"]):
            for vid, audio in zip(batch.column("video_id").to_pylist(), batch.column("audio").to_pylist()):
                try:
                    data, rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
                except Exception:
                    continue                      # the odd undecodable clip is just skipped
                sf.write(out / f"{vid}.wav", to_16k_mono(data, rate), 16000, subtype="PCM_16")
                count += 1
                if count >= max_clips:
                    break
            if count >= max_clips:
                break
        os.remove(path)                           # keep the Colab disk tidy
    if count == 0:
        raise SystemExit("No background clips could be decoded: send this error to Claude.")
    print("background clips:", count)


def npy_head(url, out, rows):
    """Download only the first `rows` rows of a huge .npy file (for the quick test).
    A .npy file is a small header followed by the rows, so two HTTP range requests do it."""
    if Path(out).exists():
        return
    f = io.BytesIO(requests.get(url, headers={"Range": "bytes=0-4095"}, timeout=60).content)
    major, _ = np.lib.format.read_magic(f)
    read_header = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
    shape, fortran, dtype = read_header(f)
    start, row_bytes = f.tell(), int(np.prod(shape[1:])) * dtype.itemsize
    rows = min(rows, shape[0])
    data = requests.get(url, headers={"Range": f"bytes={start}-{start + rows * row_bytes - 1}"}, timeout=600).content
    np.save(out, np.frombuffer(data, dtype).reshape((rows,) + tuple(shape[1:])))
    print(f"{out}: first {rows} of {shape[0]} rows")


def features(quick):
    if not Path("validation_set_features.npy").exists():
        hf_hub_download(FEATURES, "validation_set_features.npy", repo_type="dataset", local_dir=".")
    if quick:
        url = f"https://huggingface.co/datasets/{FEATURES}/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
        npy_head(url, "acav_features.npy", 20000)
    elif not Path("acav_features.npy").exists() or Path("acav_features.npy").stat().st_size < 1e10:
        path = hf_hub_download(FEATURES, "openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
                               repo_type="dataset", local_dir=".")
        os.replace(path, "acav_features.npy")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny data for the rehearsal run")
    quick = ap.parse_args().quick
    room_echoes()
    background(n_files=1 if quick else 2, max_clips=100 if quick else 3000)
    features(quick)
    print("data ready")
