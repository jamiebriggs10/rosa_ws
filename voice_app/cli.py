"""Run the pipeline in the terminal (phase 1), from the mic or a WAV file.

    ./.venv/bin/python cli.py                      # mic: press Enter to start, Enter to stop
    ./.venv/bin/python cli.py --file clip.wav      # replay a WAV through the same pipeline
    ./.venv/bin/python cli.py --settings presets/baseline.yaml

With push-to-talk and a WAV file, the "button" is pressed at the start of
the file and released at its end.
"""

import argparse
import datetime
import sys
import threading

import config
from audio import FileSource
from pipeline import Pipeline


def show(kind, data):
    if kind == "state":
        print(f"[{data['state']}]", flush=True)
    elif kind == "partial":
        print(f"  ... {data['text']}", flush=True)
    elif kind == "final":
        print(f"\n>>> {data['text']}\n", flush=True)
    elif kind == "timings":
        d = data["derived"]
        print(f"    felt delay {d['felt_delay_ms']} ms = endpoint {d['endpoint_delay_ms']} ms"
              f" + speech-to-text {d['stt_finish_ms']} ms", flush=True)
    elif kind == "loading":
        print(f"loading {data['stage']}: {data['option']}" + (f" ({data['seconds']} s)" if data["done"] else "..."))
    elif kind == "error":
        print(f"ERROR {data}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", default=str(config.SETTINGS))
    ap.add_argument("--file", help="a WAV file instead of the microphone")
    ap.add_argument("--realtime", action="store_true", help="play the file at real speed")
    args = ap.parse_args()

    settings = config.load(args.settings)
    log = config.ROOT / "runs" / f"{datetime.date.today()}-{'file' if args.file else 'live'}.jsonl"
    pipe = Pipeline(settings, emit=show, log_path=log, run_name=f"{datetime.date.today()}-cli")
    pipe.load()

    if args.file:
        source = FileSource(args.file, realtime=args.realtime, pad_end_s=1.0)
        events = []
        if settings["trigger"]["use"] in ("push_to_talk", "both"):
            events = [(0.0, "ptt_down"), (source.duration_s - 1.0, "ptt_up")]
            if settings["endpoint"]["use"] == "ptt_release":
                print("(file mode: the button is released at the end of the file, so the endpoint"
                      " delay below is just the clip's silent tail; bench/run_bench.py releases"
                      " at the end of speech)")
        pipe.run(source, events)
        print(f"logged to {log}")
        return

    from audio import MicSource
    source = MicSource(settings["audio"].get("input_device"))

    def keyboard():
        # Enter toggles the "button": first press = down, second = up.
        held = False
        for _ in sys.stdin:
            held = not held
            (pipe.ptt_down if held else pipe.ptt_up)()
            print("  (button down: speak, then Enter)" if held else "  (button up)", flush=True)

    threading.Thread(target=keyboard, daemon=True).start()
    print("Ready. Press Enter, speak, press Enter. Ctrl+C to quit.")
    try:
        pipe.run(source)
    except KeyboardInterrupt:
        pipe.stop()
    print(f"logged to {log}")


if __name__ == "__main__":
    main()
