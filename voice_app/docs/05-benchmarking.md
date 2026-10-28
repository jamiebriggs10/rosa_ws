# 05 Benchmarking: comparing options fairly

Live testing is useful for feel, but you never say a command the same way twice. For the dissertation's numbers, **record a fixed test set once and replay the same audio through every combination of settings.**

## The test set (`bench/testset/`)

- `commands.csv`: one row per clip: `id, kind, text` (the correct transcript), plus optional notes.
- `clips/<id>.wav`: 16 kHz mono WAV, recorded with `bench/record_set.py`.

Kinds:

| Kind | What | Used to measure |
|---|---|---|
| `command` | a robot command, said normally ("move up five centimetres") | accuracy, felt delay |
| `pause` | a command with a natural mid-sentence pause ("move left… about ten centimetres") | early cut-offs |
| `wake` | wake word followed by a command | wake detection, pre-roll |
| `noise` | lab background, talking that isn't to the robot, music: no wake word | false wake-ups |

Aim for ~40 commands, ~10 pause clips, ~10 wake clips, and at least 30 minutes of noise (can be long recordings). Record in a quiet room and again in the lab if possible (two conditions). A second speaker would strengthen the results.

`record_set.py` shows each line of `commands.csv`, records while you hold a key, trims silence to a fixed 0.5 s lead and 1.5 s tail, and saves the WAV. The fixed padding matters: endpoint delay is measured against it.

## Running a benchmark

```bash
./.venv/bin/python bench/run_bench.py --settings presets/baseline.yaml [--kinds command,pause] [--realtime]
```

- Feeds each WAV through `FileSource`, the same pipeline as live.
- Default: as fast as possible (audio-time marks are still exact; compute is measured by wall time per stage call).
- `--realtime`: paces frames at real speed, which is closer to live behaviour (queue build-up, GPU contention). Use for final numbers.
- For PTT runs, press/release is simulated at the clip's speech start/end (from the VAD on a clean pass), so `ptt_release` gives the reference.
  As built: press 0.2 s before the true speech start, release 0.1 s after the true speech end (from the manifest: exact for synthetic clips, Silero VAD for recorded ones). With any other endpoint only the press is simulated and the endpoint decides. Wake clips need trigger `openwakeword` or `both`; noise clips are always run with the wake word on (button-only presets switch to `openwakeword` for them, `ptt_release` switches to `silence`).
- `--limit N` takes the first N clips of each kind; `--voices` picks synthetic voices.

## Synthetic test set

`bench/make_synthetic.py` builds `bench/testset/synthetic/` from `commands.csv` with macOS `say` (voices Daniel, Samantha, Karen, Moira; Kate and Serena aren't installed on the Mac), with exact 0.5 s lead / 1.5 s tail, 1.0 s pauses at `...`, "hey Jarvis" for `<wake>`, and ~2 min synthesised noise clips (pink noise, `say` chatter without the wake word, tones, clicks). `manifest.csv` records each clip's true speech start/end. **Results on it are labelled synthetic and are never reported as real-speech results**: TTS is cleaner and more regular than people (e.g. Smart Turn can't separate complete from incomplete on it).
- Writes `runs/<date>-<preset>.jsonl` and prints a summary table.

`bench/summarise.py runs/*.jsonl` builds a comparison table (CSV and Markdown) across runs, ready for the dissertation.

## Metrics

| Metric | Definition | Good is |
|---|---|---|
| **Felt delay** | `t_final − t_speech_end` | low; report median and 90th percentile |
| Endpoint delay | `t_turn_closed − t_speech_end` | low |
| STT finish time | `t_final − t_turn_closed` (wall) | low |
| Early cut-off rate | turn closed before the true end of speech (`pause` clips mainly) | 0 |
| WER | word error rate vs `text` (`jiwer`, after normalising case and punctuation) | low |
| **Command accuracy** | after normalising numbers ("five" → 5) and units ("cm" → centimetres): do direction words, numbers and units all match? | 100 % |
| Wake detection rate | `wake` clips where the trigger fired | high |
| Wake delay | trigger time − end of wake phrase | low |
| False wake-ups / hour | triggers on `noise` audio | ~0 |
| Real-time factor | compute time ÷ audio duration, per stage | well under 1 |
| Resources | CPU %, memory, GPU (where measurable) | note it |

Command accuracy matters more than WER for a robot: "move up fifteen" vs "move up fifty" is one word of WER and a dangerous error.

## Log format (`runs/*.jsonl`)

One JSON object per utterance:

```json
{
  "run": "2026-10-20-baseline", "clip": "cmd_012", "source": "file",
  "machine": {"host": "jamie-mac", "device": "metal", "python": "3.11.9"},
  "settings": { "...": "full copy of the settings in force" },
  "expected": "move up five centimetres",
  "partials": [[0.62, "move"], [1.05, "move up five"]],
  "final": "Move up 5 centimetres.",
  "marks_audio_s": {"trigger": 0.00, "speech_start": 0.50, "speech_end": 2.10, "turn_closed": 2.80},
  "marks_wall_s":  {"turn_closed": 2.80, "final": 3.02},
  "derived": {"endpoint_delay_ms": 700, "stt_finish_ms": 220, "felt_delay_ms": 920},
  "scores": {"wer": 0.0, "command_ok": true},
  "stage_ms": {"vad_mean": 0.9, "stt_feed_mean": 14.2, "stt_finish": 220},
  "queue_max": 3
}
```

Live utterances are logged the same way with `"source": "mic"` and no `expected`/`scores`.

## Experiments this enables (for the dissertation)

1. Endpointing: `silence` (400 / 700 / 1000 ms) vs `adaptive` vs `smart_turn` vs `ptt_release`: felt delay against early cut-off rate. The trade-off curve is the result.
2. STT engines × model size: accuracy against finish time, Mac vs lab PC.
3. Streaming vs batch for the same engine.
4. With and without domain prompt / hotwords: command accuracy.
5. Wake word: pre-trained vs custom "hey Rosa", threshold sweep: detection rate against false wake-ups per hour.
6. GPU contention: STT timings with Ollama idle vs generating.
