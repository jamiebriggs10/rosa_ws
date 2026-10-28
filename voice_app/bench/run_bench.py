"""Replay a test set through one settings preset and score every utterance.

    ./.venv/bin/python bench/run_bench.py --settings presets/baseline.yaml
        [--set synthetic|recorded] [--kinds command,pause,wake,noise] [--voices Daniel,Karen]
        [--limit N] [--realtime] [--out runs/NAME.jsonl]

--limit N keeps the first N clips of EACH kind (quick smoke tests).
Results go to runs/<date>-<preset>[-synthetic].jsonl: one line per utterance.

How a clip is triggered (docs/05-benchmarking.md):
  command / pause : a button press is simulated 0.2 s before the true speech start
                    (needs trigger push_to_talk or both). The release is simulated
                    only when the endpoint is ptt_release; otherwise the endpoint decides.
  wake            : no press; needs trigger openwakeword or both.
  noise           : no speech meant for the robot, so EVERY trigger is a false wake-up.
                    Run with a wake-word trigger (forced if the preset has only a button)
                    and, if the endpoint is ptt_release, the silence endpoint (nothing
                    would ever release the button).
"""

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from audio.file import FileSource  # noqa: E402
from bench import score, summarise  # noqa: E402
from pipeline import Pipeline  # noqa: E402

SETS = {"synthetic": ROOT / "bench" / "testset" / "synthetic",
        "recorded": ROOT / "bench" / "testset" / "clips"}
KIND_ORDER = ["command", "pause", "wake", "noise"]      # noise last: it may need a different trigger
PRESS_BEFORE_S = 0.2      # press the button this long before speech starts
RELEASE_AFTER_S = 0.1     # release it this long after speech ends
EARLY_MARGIN_S = 0.05     # closing the turn within 50 ms of the true end is not "early"


def number(x):
    """CSV cell -> float, or None if empty."""
    return float(x) if x not in (None, "") else None


def load_clips(set_dir: Path, kinds, voices, limit) -> list[dict]:
    with open(set_dir / "manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r["kind"] in kinds and (not voices or r["voice"] in voices or r["kind"] == "noise")]
    rows.sort(key=lambda r: KIND_ORDER.index(r["kind"]))      # stable: keeps file order inside a kind
    if limit:
        counts, kept = {}, []
        for r in rows:
            counts[r["kind"]] = counts.get(r["kind"], 0) + 1
            if counts[r["kind"]] <= limit:
                kept.append(r)
        rows = kept
    for r in rows:
        r["path"] = set_dir / f"{r['clip']}.wav"
        for k in ("speech_start_s", "speech_end_s", "wake_end_s"):
            r[k] = number(r[k])
    return rows


def noise_settings(settings: dict) -> dict:
    """Settings used for noise clips: wake-word trigger, and an endpoint that can close without a button."""
    s = copy.deepcopy(settings)
    if s["trigger"]["use"] == "push_to_talk":
        s["trigger"]["use"] = "openwakeword"
    if s["endpoint"]["use"] == "ptt_release":
        s["endpoint"]["use"] = "silence"
    return s


def events_for(clip: dict, settings: dict) -> list:
    """Simulated button presses for a command/pause clip (empty for wake/noise)."""
    if clip["kind"] not in ("command", "pause"):
        return []
    events = [(max(0.0, clip["speech_start_s"] - PRESS_BEFORE_S), "ptt_down")]
    if settings["endpoint"]["use"] == "ptt_release":
        events.append((clip["speech_end_s"] + RELEASE_AFTER_S, "ptt_up"))
    return events


def can_run(clip: dict, settings: dict) -> bool:
    use = settings["trigger"]["use"]
    if clip["kind"] in ("command", "pause"):
        return use in ("push_to_talk", "both")
    if clip["kind"] == "wake":
        return use in ("openwakeword", "both")
    return True


def score_clip(clip: dict, records: list[dict]) -> list[dict]:
    """Add "main" and "scores" to each record of one clip (see the module docstring of score.py)."""
    kind, expected = clip["kind"], clip["text"]
    for i, rec in enumerate(records):
        rec["main"] = i == 0
        if kind == "noise":
            rec["scores"] = {"false_wake": rec["outcome"] != "not_triggered"}
            continue
        if not rec["main"]:
            continue                                  # extra utterances are kept but not scored
        s = {}
        if kind == "wake":
            triggered = rec["outcome"] != "not_triggered"
            s["wake_detected"] = triggered
            s["wake_delay_ms"] = None
            if triggered and clip["wake_end_s"] is not None:
                s["wake_delay_ms"] = round((rec["marks_audio_s"]["trigger"] - clip["wake_end_s"]) * 1000, 1)
        if rec["outcome"] == "not_triggered":
            s.update(wer=None, command_ok=None, early_cutoff=None)
        else:
            final = rec.get("final", "")             # a no_speech record has no text: counts as wrong
            s["wer"] = round(score.wer(expected, final), 4)
            s["command_ok"] = score.command_ok(expected, final)
            closed = rec.get("marks_audio_s", {}).get("turn_closed")
            s["early_cutoff"] = (closed < clip["speech_end_s"] - EARLY_MARGIN_S
                                 if closed is not None and clip["speech_end_s"] is not None else None)
        rec["scores"] = s
    return records


def run(settings_path, set_name="synthetic", kinds=KIND_ORDER, voices=None, limit=None,
        realtime=False, out=None, set_dir=None, quiet=False) -> Path:
    settings = config.load(settings_path)
    synthetic = set_name == "synthetic"
    set_dir = Path(set_dir) if set_dir else SETS[set_name]
    clips = load_clips(set_dir, kinds, voices, limit)
    preset = Path(settings_path).stem
    run_name = f"{time.strftime('%Y-%m-%d')}-{preset}" + ("-synthetic" if synthetic else "")
    out = Path(out) if out else ROOT / "runs" / f"{run_name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("")

    pipe = Pipeline(settings, run_name=run_name)
    pipe.load()                                        # models load once, before the first clip
    base_settings, current = pipe.settings, "base"
    records = []
    pipe.on_record.append(records.append)
    skipped = {}
    t_start = time.time()

    for n, clip in enumerate(clips, 1):
        if not can_run(clip, base_settings):
            skipped[clip["kind"]] = skipped.get(clip["kind"], 0) + 1
            continue
        want = "noise" if clip["kind"] == "noise" else "base"
        if want != current:                            # swap trigger/endpoint only when the kind changes
            pipe.settings = noise_settings(base_settings) if want == "noise" else base_settings
            pipe.load()
            current = want
        for plugin in pipe.stages.values():
            plugin.reset()                             # nothing carries over from the previous clip
        source = FileSource(clip["path"], realtime=realtime, pad_end_s=0.5)
        pipe.context = {
            "clip": clip["clip"], "expected": clip["text"], "kind": clip["kind"],
            "voice": clip["voice"], "synthetic": synthetic,
            "true_speech_start_s": clip["speech_start_s"], "true_speech_end_s": clip["speech_end_s"],
            "wake_end_s": clip["wake_end_s"], "clip_duration_s": round(source.duration_s, 2),
        }
        if current == "noise" and base_settings is not pipe.settings:
            pipe.context["note"] = "noise clip: trigger forced to a wake word if the preset had only a button"
        records.clear()
        pipe.run(source, events_for(clip, base_settings))
        if not records:                                # never triggered
            records.append({"run": run_name, "outcome": "not_triggered", **pipe.context,
                            "settings_summary": pipe.summary()})
        score_clip(clip, records)
        with open(out, "a") as f:
            for rec in records:
                f.write(json.dumps(rec, default=str) + "\n")
        if not quiet:
            last = records[0]
            print(f"[{n}/{len(clips)}] {clip['clip']:<24} {last['outcome']:<13} {last.get('final', '')!r}")

    for kind, count in skipped.items():
        print(f"note: skipped {count} {kind} clips: trigger {base_settings['trigger']['use']!r} can't start them")
    print(f"\n{out} written in {time.time() - t_start:.0f} s")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", required=True, help="a preset, e.g. presets/baseline.yaml")
    ap.add_argument("--set", default="synthetic", choices=list(SETS))
    ap.add_argument("--dir", help="a different clip folder of that kind (e.g. bench/testset/synthetic-rosa)")
    ap.add_argument("--kinds", default=",".join(KIND_ORDER))
    ap.add_argument("--voices", help="comma-separated (synthetic set)")
    ap.add_argument("--limit", type=int, help="first N clips of each kind")
    ap.add_argument("--realtime", action="store_true", help="pace audio at real speed")
    ap.add_argument("--out")
    a = ap.parse_args()
    out = run(a.settings, a.set, a.kinds.split(","), a.voices.split(",") if a.voices else None,
              a.limit, a.realtime, a.out, a.dir)
    print()
    print(summarise.to_markdown([summarise.summarise_file(out)]))


if __name__ == "__main__":
    main()
