# 06 Integration with the robot console (later, not in this prototype)

> **2026-10-28:** this folder now lives inside the console's repository as `rosa_ws/voice_app/`, so `../fyp-cobot/rosa_ws` below means `..`. The two are still separate programs; the console also speaks and plays sounds through this app (`/say`, `/play`, `/followup`).

> **Done on 2026-10-22, on the console's side.** The console (`../fyp-cobot/rosa_ws`, `voice.py` and routes `/voice/*`) now subscribes to this app's `/events` stream, turns finals into commands, treats a bare "stop" as an immediate stop, and shows this app's settings in a pop-up through `GET /options`, `GET/POST /settings`, `/presets`, `/ptt/*` and `/wake`. Nothing in this folder changed for it; the `output.console` block in `settings.yaml` stays unused (`enabled: false`). The plan below is kept as the record of what was intended; items 1 to 3 exist (with the console as the client, not this app), 4 is partly there (listening stays on while the arm moves; a streaming engine lets "stop" act on partial text), 5 did not happen and is not planned: the two stay separate programs.

Recorded here so the prototype is built in a way that makes this easy. **Do not implement it, and do not edit `../fyp-cobot/`.**

## What the console offers

The robot console (`../fyp-cobot/rosa_ws/web_console.py`) is a Flask app on port 8000 (on the ZenBook for now). Relevant endpoints:

- `POST /command` with `{"text": "..."}`: sends a command to the agent. Returns 409 if it is still busy with the previous one.
- `POST /stop`: stops the robot immediately.
- `POST /confirm`: confirms a planned move (the console shows a preview card first).

## The plan

1. **A `console` output** next to the `screen` output. Settings: `console_url`, `enabled: false` by default. Posts the final text to `/command`. Shows the console's reply (or the 409) on the voice page.
2. **"Stop" shortcut.** If the final (or even partial) text is just "stop", "halt" or "freeze", call `/stop` directly, skipping the LLM (which takes ~7 s). Streaming STT makes it possible to act on the partial text, before the end of the turn. This is a safety feature and its own experiment (time from saying "stop" to the robot halting).
3. **Voice confirm.** "Yes" / "go" / "confirm" → `/confirm`, so a move can be approved hands-free.
4. **Barge-in.** Listening stays on while the robot moves, so "stop" works mid-motion.
5. Eventually the voice page and the console merge into one Flask app; keeping both in Flask with the same style is what makes that easy.

## Why the delay numbers matter

The console's LLM step currently takes ~6.5–8 s from command to motion (llama3.1:8b, 1 Oct log). A good voice front end adds ~0.3–1 s. So the voice stage won't dominate overall delay, but endpointing that's too slow or cuts people off will be felt on every command. The stop shortcut is where voice speed is critical.
