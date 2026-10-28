"""The voice front end of the console: the microphone in, the words out, Rosa's voice back.

    words.py     what a spoken phrase means: "stop", "confirm", "no, point eleven", the wake word
    audio.py     the microphone and WAV files as 32 ms frames at 16 kHz
    stages.py    the wake word, push-to-talk, Silero VAD, the two end-of-turn rules, Parakeet
    pipeline.py  the state machine that drives the stages, frame by frame, on its own thread
    speech.py    Piper for Rosa's lines and the loop's short sounds
    models.py    the model files, downloaded once into models/

`Voice`, below, is what the console owns: it builds the stages, runs the pipeline on a thread
and hands what it hears to the console through four callbacks. Everything heavy (the sound
libraries, the models) is imported and loaded inside start(), on that thread, so the console
starts on any machine; when something is missing, `latest["error"]` says what.
"""

import datetime
import json
import pathlib
import threading

from voice import audio, models

SESSIONS_DIR = pathlib.Path(__file__).resolve().parent.parent / "sessions"


class Voice:
    def __init__(self, settings, *, on_state, on_final, on_timings, on_idle, on_error=None,
                 make_stages=None, make_source=None, make_speaker=None, sessions_dir=SESSIONS_DIR):
        self.settings = dict(settings)              # the "voice" block of settings.json
        self.on_state, self.on_final, self.on_timings, self.on_idle = on_state, on_final, on_timings, on_idle
        self.on_error = on_error or (lambda message: None)
        self._make_stages = make_stages or self._real_stages
        self._make_source = make_source or (lambda s: audio.MicSource(s.get("input_device")))
        self._make_speaker = make_speaker or self._real_speaker
        self.sessions_dir = pathlib.Path(sessions_dir)
        self.latest = {"running": False, "state": "idle", "level": 0.0, "final": "", "action": "", "felt_ms": None,
                       "error": "", "loading": "", "speaking": False, "followup_s": 0, "wake_on": False,
                       "speech_available": False, "speech_reason": "", "devices": {"input": [], "output": []}}
        self.pipeline = None
        self.speaker = self.sounds = None
        self._thread = None
        self._source = None
        self._reopen = False
        self._stopping = False

    # -- starting and stopping ---------------------------------------------------------

    def start(self):
        self._thread = threading.Thread(target=self._run, name="voice", daemon=True)
        self._thread.start()

    def stop(self):
        self._stopping = True
        if self.speaker is not None:
            self.speaker.cancel()
        if self.pipeline is not None:
            self.pipeline.stop()

    def wait(self, timeout_s):
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)

    def _run(self):
        """The pipeline's thread: build, load, then run the microphone until stopped. A new
        microphone (update) stops the run and opens the next one here."""
        try:
            self.latest["loading"] = "loading the speech models…"
            from voice.pipeline import Pipeline
            trigger, vad, silence_end, release_end, stt = self._make_stages()
            self.pipeline = Pipeline(trigger, vad, silence_end, release_end, stt, emit=self._emit)
            self.pipeline.on_record.append(self._record)
            self.pipeline.load()
            self.pipeline.set_wake(bool(self.settings.get("on")))
            self._build_speaker()
            self.latest["devices"] = audio.list_devices()
            self.latest["loading"] = ""
            while not self._stopping:
                self._source = self._make_source(self.settings)
                self.latest["running"], self.latest["error"] = True, ""
                self.pipeline.run(self._source)
                if not self._reopen:
                    break
                self._reopen = False
        except Exception as exc:
            self.latest["error"] = str(exc)
            self.latest["loading"] = ""
            self._call(self.on_error, str(exc))
        finally:
            self.latest["running"] = False

    def _real_stages(self):
        from voice import stages
        wake = stages.WakeWord(threshold=float(self.settings.get("wake_threshold", 0.5)))
        silence_ms = int(self.settings.get("silence_ms", 700))
        return (stages.Trigger(wake, stages.PushToTalk()), stages.Silero(),
                stages.SilenceEnd(silence_ms), stages.ReleaseEnd(silence_ms), stages.Parakeet())

    def _real_speaker(self, voice, on_speaking, output_device):
        from voice import speech
        try:
            player = speech.sounddevice_player(output_device)
        except ImportError as exc:                  # no sound card library: silent, but the microphone may still work
            self.latest["speech_reason"] = f"no sound output: {exc}"
            player = lambda samples, rate, stop: None  # noqa: E731
        engine = speech.piper_engine(voice, models.path("piper"))
        speaker = speech.Speaker(engine or (lambda text: None), player, cache_dir=models.path("cache", "say"),
                                 on_speaking=on_speaking, voice_key=voice)
        speaker.available = engine is not None
        return speaker, speech.Sounds(player)

    def _build_speaker(self):
        old = self.speaker
        if old is not None:
            old.cancel()
        self.speaker, self.sounds = self._make_speaker(self.settings.get("speech_voice", models.PIPER_VOICE),
                                                       self._on_speaking, self.settings.get("output_device"))
        self.latest["speech_available"] = getattr(self.speaker, "available", True)

    def _on_speaking(self, on):
        self.latest["speaking"] = bool(on)
        if self.pipeline is not None:
            self.pipeline.set_muted(on)             # Rosa never hears herself

    # -- from the console ----------------------------------------------------------------

    def update(self, changes):
        """New values for the voice block. Applied between utterances; a new microphone or
        output device reopens it."""
        self.settings.update(changes)
        if self.pipeline is None:
            return
        if "on" in changes:
            self.set_wake(bool(changes["on"]))
        if "wake_threshold" in changes:
            self.pipeline.request(lambda: setattr(self.pipeline.trigger.wake, "threshold", float(changes["wake_threshold"])))
        if "silence_ms" in changes:
            def set_silence():
                self.pipeline.silence_end.silence_ms = int(changes["silence_ms"])
                self.pipeline.release_end.fallback_silence_ms = int(changes["silence_ms"])
            self.pipeline.request(set_silence)
        if "speech_voice" in changes or "output_device" in changes:
            self.pipeline.request(self._build_speaker)
        if "input_device" in changes and self.latest["running"]:
            self._reopen = True
            self.pipeline.stop()

    def ptt(self, down):
        if self.pipeline is not None:
            (self.pipeline.ptt_down if down else self.pipeline.ptt_up)()

    def set_wake(self, on):
        self.latest["wake_on"] = bool(on)
        if self.pipeline is not None:
            self.pipeline.set_wake(bool(on))

    def followup(self, seconds):
        if self.pipeline is not None:
            self.pipeline.followup(seconds)

    def say(self, text):
        """Speak a line. False, with latest["speech_reason"] saying why, when Rosa cannot."""
        if not self.settings.get("speech_enabled", True):
            self.latest["speech_reason"] = "speech is off in the settings"
            return False
        if self.speaker is None:
            self.latest["speech_reason"] = "voice is not running"
            return False
        spoken = self.speaker.say(text)
        self.latest["speech_reason"] = "" if spoken else self.speaker.reason
        return spoken

    def hush(self):
        if self.speaker is not None:
            self.speaker.cancel()

    def play(self, name):
        if not self.settings.get("sounds_enabled", True) or self.sounds is None:
            return False
        return self.sounds.play(name)

    def voices(self):
        """The Piper voices in models/piper, by name."""
        return sorted(p.name[:-5] for p in models.path("piper").glob("*.onnx"))

    # -- from the pipeline (its thread) ---------------------------------------------------

    def _emit(self, kind, data):
        latest = self.latest
        if kind == "state":
            latest["state"] = data["state"]
            if data["state"] == "idle":
                self._call(self.on_idle)
            elif data["state"] in ("armed", "listening"):
                self._call(self.on_state, data["state"])
        elif kind == "level":
            latest["level"] = data["rms"]
        elif kind == "final":
            latest["final"], latest["felt_ms"], latest["error"] = data["text"], None, ""
            latest["action"] = self._call(self.on_final, data["text"]) or ""
        elif kind == "timings":
            latest["felt_ms"] = data["derived"].get("felt_delay_ms")
            if latest["felt_ms"] is not None:
                self._call(self.on_timings, latest["felt_ms"])
        elif kind == "followup":
            latest["followup_s"] = data["until_s"]
        elif kind == "error":
            latest["error"] = data["message"]

    def _record(self, record):
        """One line per utterance, for the dissertation's latency numbers."""
        try:
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
            path = self.sessions_dir / f"voice-{datetime.date.today()}.jsonl"
            with open(path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError:
            pass

    def _call(self, callback, *args):
        """A fault in the console must never kill the pipeline thread."""
        if callback is None:
            return None
        try:
            result = callback(*args)
            return result if isinstance(result, str) else None
        except Exception as exc:
            self.latest["error"] = f"console: {exc}"
            return None
