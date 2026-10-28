# 04 Settings and the web page

## `settings.yaml`

One file holds the current choice for every stage and the knobs for every option (so switching back restores your last values). The page edits it; you can also edit it by hand. A benchmark run reads a settings file and copies it whole into each result line, so every number in `runs/` can be reproduced.

Changes take effect **between utterances**, never during one. Changing an engine or model triggers `load()` straight away (with a "loading…" badge) so the first utterance isn't slowed by model loading.

See the file itself for the full schema; it is commented.

Saved presets: `presets/<name>.yaml` (e.g. `fast.yaml`, `accurate.yaml`, `baseline.yaml`), loadable from the page and passable to the benchmark.

## The page (http://localhost:8100)

Plain HTML, CSS and a small script in `static/`; no build step (same approach as the robot console).

**Top: status**
- A big state badge: Idle / Armed / Listening / Finishing.
- A mic level meter (so you can see the mic is working).
- A push-to-talk button (hold it, or hold space bar with the page focused).
- On/off switch for the wake word.

**Middle: transcript**
- The current utterance: partial text in grey, updating live; final text in black when done.
- History of the last utterances, each with its timing strip.

**Timing strip (per utterance)**
- A small horizontal bar: trigger → speech start → speech end → turn closed → final text, with the numbers in ms. The **felt delay** (final − speech end) shown large.

**Side: settings**
- One section per stage: a dropdown of options (unavailable ones greyed, with the reason on hover) and that option's knobs.
- Device selector (auto / cuda / metal / cpu) and the detected hardware.
- Preset load / save.

**Bottom: session stats**
- Count, median and 90th percentile felt delay for this session and these settings.

## HTTP endpoints (`app.py`)

| Method | Path | Does |
|---|---|---|
| GET | `/` | the page |
| GET | `/events` | Server-Sent Events: `state`, `level`, `partial`, `final`, `timings`, `loading`, `error` |
| GET | `/settings` | current settings |
| POST | `/settings` | change settings (partial update is fine); saves to `settings.yaml` |
| GET | `/options` | every option, its knobs and defaults, and whether it is supported here |
| POST | `/ptt/down`, `/ptt/up` | push-to-talk |
| POST | `/wake` | `{"on": true/false}` wake word on or off |
| GET | `/presets`, POST `/presets/<name>` | list / save presets |
| POST | `/presets/<name>/load` | load a preset (becomes the current settings) |
| GET | `/status` | state, wake on/off, device, the options loaded |
| GET | `/history` | the last 50 utterances (so a page reload keeps its history) |

Saving a preset that already exists returns 409 unless `?overwrite=1` (presets used for reported results must not change). `/events` sends a `: connected` comment at once, so the browser's `open` fires straight away. The page rewrites `settings.yaml` without its comments: the commented original is `settings.example.yaml`.

Testing without a microphone: `./.venv/bin/python app.py --file some.wav` loops a WAV at real speed instead of the mic.

Port **8100** so it never clashes with the robot console (8000).

## Speaking back (added 2026-10-27)

`speech:` (`enabled`, `engine: piper`, `voice`, `output_device`) and `sounds:` (`enabled`) in
`settings.yaml`. Changing `speech.voice` through `POST /settings` rebuilds the speaker. The
routes `/say`, `/play` and `/followup` and the events `speaking` and `followup` are described
in the README; `speech.py` holds the Speaker (engine + player + cache) and the sounds.
