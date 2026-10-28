"""Build the SYNTHETIC test set: computer-voiced clips from commands.csv.

Mac-only (uses `say` through bench/say.py). Everything made here is labelled
synthetic: a stand-in until Jamie records the real set with bench/record_set.py.
Because we build the clip ourselves, the true speech start/end are known
exactly (no VAD guess), which makes early cut-off measurable.

    ./.venv/bin/python bench/make_synthetic.py [--voices Daniel,Karen] [--wake "hey Jarvis"]
                                               [--only cmd_001,pause_002] [--noise-s 120] [--force]

Writes bench/testset/synthetic/<id>-<voice>.wav and manifest.csv.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench import say  # noqa: E402

COMMANDS = ROOT / "bench" / "testset" / "commands.csv"
OUT = ROOT / "bench" / "testset" / "synthetic"
RATE = say.RATE
LEAD_S, TAIL_S = 0.5, 1.5          # fixed padding, same as the real recordings
PAUSE_S = 1.0                      # gap where a pause clip has "..."
WAKE_GAP_S = {True: 0.8, False: 0.15}   # gap after the wake phrase: with "..." / without
COLUMNS = ["clip", "id", "kind", "voice", "text", "speech_start_s", "speech_end_s", "wake_end_s"]

# Everyday sentences that are NOT robot commands and have no wake word:
# the realistic "people talking nearby" false-trigger test.
CHATTER = [
    "Did you see the football last night",
    "I think the meeting has been moved to Thursday afternoon",
    "Can you pass me that screwdriver please",
    "The coffee machine on the second floor is broken again",
    "We should probably order some more cable ties",
    "My train was delayed by twenty minutes this morning",
    "Have you finished reading that paper yet",
    "It looks like it is going to rain later",
    "I will send you the slides after lunch",
    "The printer has run out of paper again",
    "She said the results were better than expected",
    "Let me know if you need a hand with the soldering",
    "I cannot remember where I left my keys",
    "The new students start on Monday",
    "That was a really interesting talk yesterday",
    "Shall we get some lunch in a bit",
    "The software update takes about an hour",
    "He is working on his dissertation this week",
    "Somebody left the window open in the lab",
    "I will be back in about five minutes",
]


# ---------------------------------------------------------------- manifest

def read_manifest(path) -> dict:
    """{clip: row} from an existing manifest.csv (empty dict if none)."""
    if not Path(path).exists():
        return {}
    with open(path, newline="") as f:
        return {r["clip"]: r for r in csv.DictReader(f)}


def write_manifest(path, rows) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        w.writerows(rows)


def clean_text(text: str) -> str:
    """The expected transcript: no "<wake>", no "...", single spaces."""
    return " ".join(text.replace("<wake>", " ").replace("...", " ").split())


# ------------------------------------------------------------ speech clips

def speech_clip(kind: str, text: str, voice: str, wake: str):
    """(audio, speech_start_s, speech_end_s, wake_end_s) for one command row.
    Timings follow directly from the padding we add, so they are exact."""
    wake_end = None
    if kind == "wake":
        rest = text.replace("<wake>", "", 1).strip()
        gap = WAKE_GAP_S[rest.startswith("...")]
        rest = clean_text(rest)
        wake_audio = say.say_audio(wake, voice)
        wake_end = LEAD_S + len(wake_audio) / RATE
        parts = [wake_audio, say.silence(gap), say.say_audio(rest, voice)]
        start = wake_end + gap                       # the command starts after the gap
        speech = np.concatenate(parts)
    elif kind == "pause":
        pieces = [p.strip() for p in text.split("...") if p.strip()]
        parts = []
        for k, piece in enumerate(pieces):
            if k:
                parts.append(say.silence(PAUSE_S))
            parts.append(say.say_audio(piece, voice))
        speech, start = np.concatenate(parts), LEAD_S
    else:
        speech, start = say.say_audio(text, voice), LEAD_S
    end = LEAD_S + len(speech) / RATE
    return say.pad(speech, LEAD_S, TAIL_S), start, end, wake_end


# ------------------------------------------------------------ noise clips
# No real recordings exist yet, so each noise row gets a made-up stand-in.

def pink_noise(n: int, rng) -> np.ndarray:
    """1/f noise (more bass than white noise, like room rumble), made in the frequency domain."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec), dtype=float)
    f[0] = 1
    return np.fft.irfft(spec / np.sqrt(f), n)


def scale_rms(x: np.ndarray, rms: float) -> np.ndarray:
    return (x * rms / (np.sqrt(np.mean(x ** 2)) + 1e-9)).astype(np.float32)


def synth_lab_noise(seconds: float, rng, voices) -> np.ndarray:
    return scale_rms(pink_noise(int(seconds * RATE), rng), 0.01)       # quiet: about -40 dBFS


def synth_chatter(seconds: float, rng, voices) -> np.ndarray:
    """Ordinary sentences with random 1-6 s gaps, voices rotating. No wake word."""
    pieces, total, k = [], 0.0, int(rng.integers(len(CHATTER)))
    while total < seconds:
        sentence = say.say_audio(CHATTER[k % len(CHATTER)], voices[k % len(voices)])
        gap = say.silence(float(rng.uniform(1.0, 6.0)))
        pieces += [sentence, gap]
        total += (len(sentence) + len(gap)) / RATE
        k += 1
    return np.concatenate(pieces)[:int(seconds * RATE)]


def synth_music(seconds: float, rng, voices) -> np.ndarray:
    """Chords of sine tones changing every ~1.5 s, over a little noise."""
    n = int(seconds * RATE)
    out = np.zeros(n, np.float32)
    notes = [220.0, 261.63, 293.66, 329.63, 392.0, 440.0]
    pos, chord_len = 0, int(1.5 * RATE)
    while pos < n:
        m = min(chord_len, n - pos)
        t = np.arange(m) / RATE
        envelope = np.minimum(1, np.minimum(t / 0.05, (m / RATE - t) / 0.1))   # fade in/out: no clicks
        for f in rng.choice(notes, 3, replace=False):
            out[pos:pos + m] += (0.08 * np.sin(2 * np.pi * f * t) * envelope).astype(np.float32)
        pos += m
    return out + scale_rms(rng.standard_normal(n), 0.005)


def synth_typing(seconds: float, rng, voices) -> np.ndarray:
    """Keyboard clicks: short bursts of decaying noise, in bursts of typing with pauses."""
    n = int(seconds * RATE)
    out = scale_rms(pink_noise(n, rng), 0.003)
    t = 0.0
    while t < seconds:
        for _ in range(int(rng.integers(5, 25))):        # one burst of typing
            t += float(rng.uniform(0.07, 0.25))
            i, m = int(t * RATE), int(0.02 * RATE)
            if i + m >= n:
                break
            out[i:i + m] += (rng.standard_normal(m) * np.exp(-np.arange(m) / (0.004 * RATE)) * 0.25)
        t += float(rng.uniform(1.0, 4.0))                # pause between bursts
    return out.astype(np.float32)


NOISE = {"noise_001": synth_lab_noise, "noise_002": synth_chatter,
         "noise_003": synth_music, "noise_004": synth_typing}


# -------------------------------------------------------------------- main

def build(out_dir=OUT, voices=("Daniel", "Samantha", "Karen", "Moira"), wake="hey Jarvis",
          only=None, noise_s=120, force=False, commands_csv=COMMANDS) -> list:
    """Make every missing clip and (re)write the manifest. Returns the manifest rows."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    existing = read_manifest(manifest_path)
    with open(commands_csv, newline="") as f:
        rows = [r for r in csv.DictReader(f) if not only or r["id"] in only]

    out_rows = []
    for row in rows:
        kind = row["kind"]
        for voice in (["synthetic"] if kind == "noise" else voices):
            clip = f"{row['id']}-{voice}"
            path = out_dir / f"{clip}.wav"
            if path.exists() and clip in existing and not force:
                out_rows.append(existing[clip])          # already built: keep its manifest row
                continue
            print("making", clip)
            if kind == "noise":
                rng = np.random.default_rng(int(row["id"].split("_")[1]))   # fixed seed: repeatable
                audio = NOISE[row["id"]](noise_s, rng, list(voices))
                say.save(audio, path)
                info = ["", "", ""]
            else:
                audio, start, end, wake_end = speech_clip(kind, row["text"], voice, wake)
                say.save(audio, path)
                info = [round(start, 3), round(end, 3), "" if wake_end is None else round(wake_end, 3)]
            out_rows.append(dict(zip(COLUMNS, [clip, row["id"], kind, voice,
                                               clean_text(row["text"]), *info])))
    write_manifest(manifest_path, out_rows)
    return out_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voices", default="Daniel,Samantha,Karen,Moira")
    ap.add_argument("--wake", default="hey Jarvis", help="phrase that replaces <wake>")
    ap.add_argument("--only", help="comma-separated ids to build, e.g. cmd_001,wake_002")
    ap.add_argument("--noise-s", type=float, default=120, help="seconds per noise clip")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--force", action="store_true", help="rebuild clips that already exist")
    a = ap.parse_args()
    if not say.available():
        sys.exit("macOS `say` not found: synthetic clips can only be made on the Mac")
    rows = build(a.out, a.voices.split(","), a.wake, a.only.split(",") if a.only else None,
                 a.noise_s, a.force)
    print(f"{len(rows)} clips in {a.out}")


if __name__ == "__main__":
    main()
