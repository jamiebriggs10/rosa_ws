# 01 Architecture

## Goal

Say a command, see the text on screen as fast as possible, and be able to swap every stage for another option without touching the rest. Delay matters most at two points: **how long after you stop talking the system decides you've finished**, and **how long speech-to-text takes after that**.

## Pieces

```
            ┌──────────── app.py (Flask, port 8100) ─────────────┐
 browser ◀──│  /  page    /events (SSE)   /settings   /ptt   /arm │
            └───────────────────────┬──────────────────────────────┘
                                    │ events ▲   commands ▼
            ┌───────────────────────┴──────────────────────────────┐
            │ pipeline.py  (one worker thread, the state machine)   │
            │   trigger ─ vad ─ endpoint ─ stt ─ output             │
            └───────────────────────┬──────────────────────────────┘
                                    │ frames
            ┌───────────────────────┴──────────────────────────────┐
            │ audio/  MicSource (sounddevice)  |  FileSource (WAV)  │
            └──────────────────────────────────────────────────────┘
```

- **Audio source**: produces frames of **512 samples, 16 kHz, mono, float32** (32 ms each). The mic and a WAV file look identical to the pipeline; that is what makes benchmarking fair. The mic callback only puts frames on a queue: no processing in the audio thread.
- **Pipeline**: one worker thread takes frames off the queue and passes them through the stages. It owns the state machine and timestamps.
- **Stages**: plug-ins chosen by name from `settings.yaml`. Each implements the small interface in `stages/base.py`. Stages buffer internally if they need a different frame size (openWakeWord wants 80 ms, WebRTC VAD 10/20/30 ms).
- **Output**: for now, "show on screen" (an event to the page) plus a line in `runs/`. Later, a second output posts to the robot console (see `06-integration-later.md`).
- **App**: Flask, same as the robot console, so it can be merged later. Live updates to the page via Server-Sent Events (one long HTTP response the server keeps writing to; simpler than WebSockets and fine in plain Flask).

## State machine

```
          trigger fires                  VAD: speech starts
 IDLE ─────────────────▶ ARMED ───────────────────────────▶ LISTENING
  ▲                       │  no speech within                │  frames → STT (streaming: partials)
  │                       │  `armed_timeout_s`               │  endpoint.update(...) each frame
  │                       ▼                                  ▼
  └──────────────────── (back to IDLE)            endpoint says "finished"
  │                                                          │
  │                                                          ▼
  └──────────── OUTPUT ◀──────────── FINALISING (stt.finish() → final text)
```

- **IDLE**: only the trigger runs. Cheap, always on.
- **ARMED**: trigger has fired, waiting for speech. Page shows "listening". Gives up after `armed_timeout_s` (default 4 s).
- **LISTENING**: every frame goes to the VAD, the end-of-turn detector and (in streaming mode) the speech-to-text engine. Partial text goes to the page.
- **FINALISING**: the endpoint has decided you've finished. `stt.finish()` returns the final text.
- **OUTPUT**: final text and timings to the page and to `runs/`. Back to IDLE.

Push-to-talk skips ARMED: press goes straight to LISTENING, release goes straight to FINALISING (when the endpoint option is `ptt_release`).

## Pre-roll buffer

People start talking straight after the wake word, often before the trigger has fired. The pipeline keeps a **ring buffer of the last ~0.5 s of audio** (`preroll_ms`) and hands it to the speech-to-text engine when LISTENING starts, so the first word isn't clipped. The wake word itself is trimmed by starting the pre-roll at the trigger time.

## The stage contract (`stages/base.py`)

| Stage | Called with | Returns |
|---|---|---|
| `Trigger` | `process(frame)` every frame while IDLE | `True` when it fires |
| `VAD` | `speech_prob(frame)` | probability 0–1 that this frame is speech |
| `EndOfTurn` | `update(frame, is_speech, partial_text)` every frame while LISTENING | `True` when the speaker has finished |
| `SpeechToText` | `start()`, `feed(frame)`, `finish()` | `feed` may return partial text; `finish` returns final text |

Every class has `name`, `supports(device) -> bool` (so the page can grey out options the machine can't run), a `load()` that does the slow model loading up front, and `reset()` between utterances.

A batch engine (records, then transcribes) just buffers in `feed` and does all the work in `finish`. A streaming engine does the work in `feed`. The pipeline doesn't care which.

## Timestamps

Each utterance records, in seconds:

| Mark | Meaning |
|---|---|
| `t_trigger` | wake word fired / PTT pressed |
| `t_speech_start` | first speech frame (VAD) |
| `t_speech_end` | last speech frame (VAD) |
| `t_turn_closed` | endpoint said "finished" / PTT released |
| `t_final` | final text available |

Derived: **endpoint delay** = `t_turn_closed − t_speech_end`; **STT finish time** = `t_final − t_turn_closed`; **felt delay** = `t_final − t_speech_end` (the number the user feels).

Two clocks: **audio time** (sample count, exact and identical for a WAV whatever the machine) for the speech marks, and **wall time** for compute. With a WAV file replayed faster than real time, compute cost is still measured by wall time per call. See `05-benchmarking.md`.

## Threads

1. Audio callback (sounddevice): puts frames on a queue. Nothing else.
2. Pipeline worker: everything else in order. If a stage is too slow the queue grows; the queue length is logged so this shows up.
3. Flask: serves the page, pushes events from an event queue, changes settings (which takes effect between utterances, never mid-utterance).

A streaming STT engine that is too slow to run per frame may run on its own thread and drop stale partial updates (as in the Parakeet live-transcription examples), but `finish()` is always run in full.

## What it deliberately doesn't do yet

- Send anything to the robot.
- Speak back (text-to-speech).
- Handle barge-in (talking over the robot) or a "stop" shortcut. Both are designed in `06-integration-later.md`.
