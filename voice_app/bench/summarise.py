"""Turn run files (runs/*.jsonl from run_bench.py) into one comparison table.

    ./.venv/bin/python bench/summarise.py runs/*.jsonl [--out results/summary]

Prints a Markdown table and writes <out>.md and <out>.csv, one row per run file.
Runs made from synthetic (computer-voiced) clips are marked "(synthetic)".

Per-clip numbers use only each clip's MAIN record (the first utterance it
produced); extra utterances from the same clip are early cut-off fallout and
are ignored except for false wake-ups on noise clips, which count every trigger.
"""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

COLUMNS = ["run", "settings", "clips", "felt_median_ms", "felt_p90_ms", "endpoint_median_ms",
           "stt_finish_median_ms", "early_cutoff_pause_pct", "early_cutoff_all_pct", "wer_mean_pct",
           "command_acc_pct", "wake_detect_pct", "wake_delay_median_ms", "false_wakeups_per_hour",
           "stt_finish_per_speech_s"]


def median(xs):
    return round(statistics.median(xs), 1) if xs else None


def p90(xs):
    """90th percentile (nearest-rank), so with few clips it is simply close to the maximum."""
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(0.9 * len(xs)))], 1)


def pct(xs):
    """Percentage of True in a list of bools (None if the list is empty)."""
    return round(100 * sum(xs) / len(xs), 1) if xs else None


def mean(xs):
    return round(statistics.mean(xs), 3) if xs else None


def summarise_records(records: list[dict], name: str = "") -> dict:
    main = [r for r in records if r.get("main") and r.get("kind") != "noise"]
    noise = [r for r in records if r.get("kind") == "noise"]
    derived = [r.get("derived", {}) for r in main]

    def scores(key, rows=main):
        return [r["scores"][key] for r in rows if r.get("scores", {}).get(key) is not None]

    pauses = [r for r in main if r.get("kind") == "pause"]
    # How long STT took to finish per second of speech (batch engines: ~ the real-time factor).
    ratios = [r["derived"]["stt_finish_ms"] / 1000 / (r["true_speech_end_s"] - r["true_speech_start_s"])
              for r in main if r.get("derived", {}).get("stt_finish_ms") is not None
              and r.get("true_speech_start_s") is not None and r.get("true_speech_end_s")]
    noise_hours = sum({r["clip"]: r.get("clip_duration_s", 0) for r in noise}.values()) / 3600
    false_wakes = sum(1 for r in noise if r["scores"]["false_wake"])
    first = records[0] if records else {}
    run = first.get("run", name)
    return {
        "run": run + (" (synthetic)" if first.get("synthetic") else ""),
        "settings": first.get("settings_summary", ""),
        "clips": len({r["clip"] for r in main}),
        "felt_median_ms": median([d["felt_delay_ms"] for d in derived if d.get("felt_delay_ms") is not None]),
        "felt_p90_ms": p90([d["felt_delay_ms"] for d in derived if d.get("felt_delay_ms") is not None]),
        "endpoint_median_ms": median([d["endpoint_delay_ms"] for d in derived
                                      if d.get("endpoint_delay_ms") is not None]),
        "stt_finish_median_ms": median([d["stt_finish_ms"] for d in derived if d.get("stt_finish_ms") is not None]),
        "early_cutoff_pause_pct": pct(scores("early_cutoff", pauses)),
        "early_cutoff_all_pct": pct(scores("early_cutoff")),
        "wer_mean_pct": None if not scores("wer") else round(100 * mean(scores("wer")), 1),
        "command_acc_pct": pct(scores("command_ok")),
        "wake_detect_pct": pct(scores("wake_detected")),
        "wake_delay_median_ms": median(scores("wake_delay_ms")),
        "false_wakeups_per_hour": round(false_wakes / noise_hours, 1) if noise_hours else None,
        "stt_finish_per_speech_s": mean(ratios),
    }


def summarise_file(path) -> dict:
    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    return summarise_records(records, Path(path).stem)


def to_markdown(rows: list[dict]) -> str:
    def cell(v):
        return "" if v is None else str(v)
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    lines += ["| " + " | ".join(cell(r[c]) for c in COLUMNS) + " |" for r in rows]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="results/summary", help="writes OUT.md and OUT.csv")
    a = ap.parse_args()
    rows = [summarise_file(p) for p in a.files]
    table = to_markdown(rows)
    print(table)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(table + "\n")
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwritten {out.with_suffix('.md')} and {out.with_suffix('.csv')}", file=sys.stderr)


if __name__ == "__main__":
    main()
