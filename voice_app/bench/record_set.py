"""Record the REAL test set with the microphone (Jamie speaks; run this in a terminal).

    ./.venv/bin/python bench/record_set.py [--speaker jamie] [--wake "hey Jarvis"] [--noise-minutes 5]
                                           [--only cmd_001,pause_002] [--device NAME]

For each row of commands.csv it shows what to say. Enter = start recording,
Enter again = stop. Then: Enter keeps it, r redoes it, s skips it, q quits.
Clips are trimmed to exactly 0.5 s of silence before the speech and 1.5 s after
(the fixed padding matters: endpoint delay is measured against it) and saved to
bench/testset/clips/<id>.wav with a manifest.csv in the same format as the
synthetic set. Clips that already exist are skipped, so you can stop and resume.

Noise rows are not trimmed: you record them for --noise-minutes each.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.make_synthetic import (COMMANDS, COLUMNS, LEAD_S, TAIL_S, clean_text,  # noqa: E402
                                  read_manifest, write_manifest)

RATE = 16000
OUT = ROOT / "bench" / "testset" / "clips"
FRAME = 512


def find_speech(audio: np.ndarray, vad=None, threshold: float = 0.5, min_frames: int = 3):
    """(start_s, end_s) of the speech in a clip, using the Silero VAD plug-in.
    Start = first run of `min_frames` speech frames (ignores clicks); end = last speech frame.
    Returns None if no speech is found."""
    if vad is None:
        from stages.vad.silero import Silero
        vad = Silero({}, "cpu")
        vad.load()
    vad.reset()
    flags = []
    for i in range(0, len(audio) - FRAME + 1, FRAME):
        flags.append(vad.speech_prob(audio[i:i + FRAME]) >= threshold)
    start = next((i for i in range(len(flags) - min_frames + 1) if all(flags[i:i + min_frames])), None)
    if start is None:
        return None
    end = max(i for i, f in enumerate(flags) if f) + 1
    return start * FRAME / RATE, end * FRAME / RATE


def trim_clip(audio: np.ndarray, start_s: float, end_s: float,
              lead_s: float = LEAD_S, tail_s: float = TAIL_S) -> np.ndarray:
    """Cut the clip so there are exactly lead_s before the speech and tail_s after it.
    If the recording has less silence than that, zeros are added."""
    a = int(round((start_s - lead_s) * RATE))
    b = int(round((end_s + tail_s) * RATE))
    before = np.zeros(max(0, -a), np.float32)                 # recording started too close to the speech
    after = np.zeros(max(0, b - len(audio)), np.float32)      # ... or stopped too soon
    return np.concatenate([before, audio[max(a, 0):b], after]).astype(np.float32)


# --------------------------------------------------------------- recording

def record_until_enter(device=None) -> np.ndarray:
    """Record from the mic until Enter is pressed. Returns mono float32 at 16 kHz."""
    import sounddevice as sd
    import soxr
    info = sd.query_devices(device, "input")
    rate = int(info["default_samplerate"])         # record at the device's own rate, resample once
    chunks = []
    with sd.InputStream(samplerate=rate, channels=1, dtype="float32", device=device,
                        callback=lambda data, *_: chunks.append(data[:, 0].copy())):
        input("  recording... press Enter to stop ")
    audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    return soxr.resample(audio, rate, RATE).astype(np.float32)


def record_minutes(minutes: float, device=None) -> np.ndarray:
    """Record for a fixed time (Ctrl-C stops early and keeps what was recorded)."""
    import sounddevice as sd
    import soxr
    rate = int(sd.query_devices(device, "input")["default_samplerate"])
    chunks = []
    with sd.InputStream(samplerate=rate, channels=1, dtype="float32", device=device,
                        callback=lambda data, *_: chunks.append(data[:, 0].copy())):
        t0 = time.time()
        try:
            while time.time() - t0 < minutes * 60:
                print(f"\r  {time.time() - t0:5.0f} s / {minutes * 60:.0f} s (Ctrl-C to stop early)", end="")
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    print()
    return soxr.resample(np.concatenate(chunks), rate, RATE).astype(np.float32)


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speaker", default="jamie", help="goes in the manifest's voice column")
    ap.add_argument("--wake", default="hey Jarvis")
    ap.add_argument("--noise-minutes", type=float, default=5)
    ap.add_argument("--only", help="comma-separated ids")
    ap.add_argument("--device", default=None, help="part of the input device name")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(out / "manifest.csv")
    only = a.only.split(",") if a.only else None
    with open(COMMANDS, newline="") as f:
        rows = [r for r in csv.DictReader(f) if not only or r["id"] in only]

    from stages.vad.silero import Silero
    vad = Silero({}, "cpu")
    vad.load()

    for row in rows:
        path = out / f"{row['id']}.wav"
        if path.exists() and row["id"] in manifest:
            continue
        kind, shown = row["kind"], row["text"].replace("<wake>", a.wake)
        print(f"\n[{row['id']}] ({kind}) {row['notes'] or ''}")
        if kind == "noise":
            print(f"  Noise to record: {row['notes']}  ({a.noise_minutes} min). No wake word!")
            if input("  Enter = start, s = skip, q = quit: ").strip() == "s":
                continue
            audio = record_minutes(a.noise_minutes, a.device)
            sf.write(str(path), audio, RATE, subtype="PCM_16")
            manifest[row["id"]] = dict(zip(COLUMNS, [row["id"], row["id"], kind, a.speaker, "", "", "", ""]))
            write_manifest(out / "manifest.csv", list(manifest.values()))
            continue
        if kind == "pause":
            print("  (pause where the ... is, about a second)")
        print(f'  SAY:  "{shown}"')
        while True:
            cmd = input("  Enter = record, s = skip, q = quit: ").strip().lower()
            if cmd in ("s", "q"):
                break
            raw = record_until_enter(a.device)
            bounds = find_speech(raw, vad)
            if bounds is None:
                print("  No speech found. Try again (check the microphone).")
                continue
            clip = trim_clip(raw, *bounds)
            start, end = find_speech(clip, vad) or (LEAD_S, LEAD_S + bounds[1] - bounds[0])
            print(f"  speech {start:.2f}-{end:.2f} s, clip {len(clip) / RATE:.2f} s")
            if kind == "wake":
                print("  (wake word end isn't detected automatically, so wake delay isn't scored for real clips)")
            ok = input("  Enter = keep, r = redo, s = skip: ").strip().lower()
            if ok == "r":
                continue
            if ok == "s":
                cmd = "s"
                break
            sf.write(str(path), clip, RATE, subtype="PCM_16")
            # For wake clips the VAD start is the wake word's start, not the command's: leave it blank.
            manifest[row["id"]] = dict(zip(COLUMNS, [
                row["id"], row["id"], kind, a.speaker, clean_text(row["text"]),
                "" if kind == "wake" else round(start, 3), round(end, 3), ""]))
            write_manifest(out / "manifest.csv", list(manifest.values()))
            break
        if cmd == "q":
            break
    print("Done. Clips and manifest.csv are in", out)


if __name__ == "__main__":
    main()
