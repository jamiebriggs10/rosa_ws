# bench/

- `testset/commands.csv`: the script of what to record (id, kind, text). Edit freely before recording; once results are reported, only add rows, never change existing ones.
- `testset/clips/`: real recordings, `<id>.wav`, 16 kHz mono, 0.5 s silence lead + 1.5 s tail.
- `testset/synthetic/`: computer-voiced clips (Mac-only, using `say`), labelled as such in results.
- `say.py`: Generate synthetic speech with macOS `say` command (test-only, never used by the app).
- `score.py`: Word error rate (WER) and command accuracy (numbers, units, directions match exactly).
- `make_synthetic.py`: Build the synthetic test set from `commands.csv` with configurable voices and noise. Writes `testset/synthetic/*.wav` and `manifest.csv`.
- `record_set.py`: Interactive recording tool: walks through the CSV, records each clip, trims to 0.5 s lead / 1.5 s tail, shows each for approval (keep/redo/skip/quit).
- `run_bench.py`: Replay clips through one settings preset, score every utterance, write `runs/<date>-<preset>.jsonl`. Simulates button presses at 0.2 s before speech start.
- `summarise.py`: Comparison tables from run files: prints Markdown, writes `.md` and `.csv`. Marks synthetic runs as "(synthetic)".
- `results/`: summaries of runs, one row per run file.

Details in `docs/05-benchmarking.md`.
