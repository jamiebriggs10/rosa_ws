# static/

The settings page and live view: `index.html` (plain HTML, CSS and a small script, no build step).

The page shows: a state badge (Idle/Armed/Listening/Finalising), mic level meter, push-to-talk button (hold or press space), wake word on/off switch; the current utterance (partial text in grey, final in black); history of recent utterances with timing strips (trigger → speech → turn closed → final text, with felt delay in ms); settings panel with stage options and knobs; device selector; preset load/save; and session stats (count, median, 90th-percentile felt delay). It talks to `app.py`'s endpoints for settings, events, options, push-to-talk, wake word, presets, and session history. Spec in `docs/04-settings-and-ui.md`.
