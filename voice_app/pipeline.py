"""The pipeline: one worker thread that drives the four stages, frame by frame.

    IDLE --trigger--> ARMED --speech starts--> LISTENING --endpoint--> FINALISING --> OUTPUT --> IDLE
    (push-to-talk skips ARMED and goes straight to LISTENING)

Everything here runs on ONE thread (the worker): stages are loaded, used and
swapped on it, so models that care which thread they live on (MLX) are happy.
Other threads talk to it only through `self.commands` (a queue).

Two clocks (docs/01-architecture.md):
  - audio time: frames counted x 32 ms. Exact, and identical for a WAV on any machine.
  - wall time: time.perf_counter(), for how long the computing takes.
"""

import collections
import copy
import json
import queue
import time
from pathlib import Path

import numpy as np

import config
import device as device_mod
from audio.framer import FRAME, RATE, rms
from stages import registry

IDLE, ARMED, LISTENING, FINALISING = "idle", "armed", "listening", "finalising"
FRAME_S = FRAME / RATE          # 0.032 s
STAGES = ("trigger", "vad", "endpoint", "stt")


class StageTimer:
    """Wall-clock milliseconds spent in each kind of stage call."""

    def __init__(self):
        self.calls = collections.defaultdict(list)

    def run(self, key, fn, *args):
        t0 = time.perf_counter()
        result = fn(*args)
        self.calls[key].append((time.perf_counter() - t0) * 1000)
        return result

    def summary(self) -> dict:
        out = {}
        for key, ms in self.calls.items():
            out[f"{key}_mean"] = round(float(np.mean(ms)), 2)
            out[f"{key}_max"] = round(float(np.max(ms)), 2)
        return out


class Pipeline:
    def __init__(self, settings: dict, emit=None, log_path=None, run_name="live"):
        self.settings = copy.deepcopy(settings)
        self.emit = emit or (lambda kind, data: None)   # sends events to the page
        self.log_path = Path(log_path) if log_path else None
        self.run_name = run_name
        self.on_record = []          # callbacks given each finished utterance's record
        self.context = {}            # extra fields for the record (the benchmark sets clip, expected)
        self.commands = queue.Queue()
        self.stages = {}             # "trigger" -> plug-in instance, ...
        self._loaded_as = {}         # "trigger" -> what it was built from, to spot changes
        self._pending_settings = None
        self.device = device_mod.resolve(self.settings.get("device", "auto"))
        self.state = IDLE
        self.wake_on = True
        self.muted = False           # while the app speaks: frames are dropped before any stage sees them
        self.followup_until = 0.0    # wall clock (time.monotonic): until then speech needs no wake word
        self._follow_utt = None      # the scratch utterance the VAD fills while the window is open
        self.frame_count = 0
        self._stop = False
        self.source = None
        self.preroll = collections.deque(maxlen=self._preroll_frames())
        self.timer_idle = StageTimer()   # trigger cost while waiting (reset after each utterance)
        self.timer = StageTimer()        # stage costs during the current utterance
        self.queue_max = 0

    # ------------------------------------------------------------------ loading

    def _preroll_frames(self) -> int:
        return max(1, int(self.settings["audio"].get("preroll_ms", 500) / 1000 / FRAME_S))

    def load(self) -> None:
        """Build (and load the models of) any stage whose settings changed.
        Slow, so only ever called between utterances."""
        self.device = device_mod.resolve(self.settings.get("device", "auto"))
        for stage in STAGES:
            name, knobs = config.option_settings(self.settings, stage)
            key = json.dumps([name, knobs, self.device], sort_keys=True)
            if self._loaded_as.get(stage) == key:
                continue                       # unchanged: keep the loaded model
            try:
                cls = registry.get_class(stage, name)
                ok, reason = cls.supports(self.device)
                if not ok:
                    raise RuntimeError(f"{name} can't run on {self.device}: {reason}")
                self.emit("loading", {"stage": stage, "option": name, "done": False})
                t0 = time.perf_counter()
                plugin = cls(knobs, self.device)
                plugin.load()
                if not self.wake_on:
                    plugin.on_event("wake_off")
                old = self.stages.get(stage)
                if hasattr(old, "close"):
                    old.close()                # e.g. stop a whisper-server it started
                self.stages[stage] = plugin
                self._loaded_as[stage] = key
                self.emit("loading", {"stage": stage, "option": name, "done": True,
                                      "seconds": round(time.perf_counter() - t0, 2)})
            except Exception as e:
                self.emit("error", {"stage": stage, "option": name, "message": str(e)})
                if stage not in self.stages:
                    raise                      # nothing to fall back on
                # Otherwise keep the option that was already working.
        self.preroll = collections.deque(self.preroll, maxlen=self._preroll_frames())

    def request_settings(self, settings: dict) -> None:
        """New settings from another thread. Applied between utterances."""
        self.commands.put(("settings", copy.deepcopy(settings)))

    # ------------------------------------------------------------ outside events

    def ptt_down(self):
        self.commands.put(("event", "ptt_down"))

    def ptt_up(self):
        self.commands.put(("event", "ptt_up"))

    def set_wake(self, on: bool):
        self.commands.put(("event", "wake_on" if on else "wake_off"))

    def set_muted(self, on: bool):
        """The app is speaking (or has stopped): drop the microphone meanwhile, so it never hears itself."""
        self.commands.put(("event", "mute_on" if on else "mute_off"))

    def followup(self, seconds: float):
        """For `seconds`, speech counts as triggered without the wake word (the console has just
        asked the operator something). 0 closes the window."""
        until = time.monotonic() + float(seconds) if seconds > 0 else 0.0   # from now, not from when the thread gets to it
        self.followup_until = until          # readable at once (status); the thread resets its scratch below
        self.commands.put(("followup", until))

    @property
    def followup_s(self) -> float:
        """Seconds left in the follow-up window, 0 when closed."""
        return max(0.0, self.followup_until - time.monotonic())

    def stop(self):
        self._stop = True
        if self.source is not None:
            self.source.stop()

    def _handle_commands(self):
        while True:
            try:
                kind, value = self.commands.get_nowait()
            except queue.Empty:
                break
            if kind == "settings":
                self._pending_settings = value
            elif kind == "followup":
                self.followup_until, self._follow_utt = value, None
                self.emit("followup", {"until_s": round(self.followup_s, 2)})
            elif kind == "event":
                if value == "ptt_down" and self.settings["trigger"]["use"] == "openwakeword":
                    self.emit("error", {"stage": "trigger", "message":
                                        "push-to-talk is off: choose trigger push_to_talk or both"})
                if value in ("wake_on", "wake_off"):
                    self.wake_on = value == "wake_on"
                if value in ("mute_on", "mute_off"):
                    self.muted = value == "mute_on"
                    self.preroll.clear()           # nothing of the app's own voice leads the next transcript
                for plugin in self.stages.values():
                    plugin.on_event(value)
        # Settings never change in the middle of an utterance.
        if self._pending_settings is not None and self.state == IDLE:
            self.settings, self._pending_settings = self._pending_settings, None
            self.load()
            if self.source is not None and hasattr(self.source, "drain"):
                self.source.drain()            # skip audio that queued up while loading

    # ------------------------------------------------------------- main loop

    @property
    def t(self) -> float:
        """Audio time (s) at the END of the current frame."""
        return self.frame_count * FRAME_S

    def run(self, source, events=None) -> None:
        """Process a source until it ends (file) or stop() is called (mic).
        `events`: optional [(audio_time_s, "ptt_down"|"ptt_up"), ...] to simulate
        button presses at exact moments (benchmark of push-to-talk)."""
        if len(self.stages) < len(STAGES):
            self.load()
        events = sorted(events or [])
        self.source = source
        self._stop = False
        self.frame_count = 0           # audio time restarts at 0 for each source (each benchmark clip)
        self.preroll.clear()
        source.start()
        try:
            for frame in source.frames():
                # Deliver simulated presses that are due before this frame starts.
                while events and events[0][0] <= self.t:
                    self.commands.put(("event", events.pop(0)[1]))
                self.step(frame)
                if self._stop:
                    break
        finally:
            source.stop()
        if self.state in (ARMED, LISTENING):
            # The audio ran out mid-utterance: finish it rather than lose it.
            if self.state == LISTENING:
                self._finalise("end_of_audio")
            else:
                self._back_to_idle("no_speech")

    def step(self, frame: np.ndarray) -> None:
        """Everything that happens for one 32 ms frame."""
        self._handle_commands()
        self.frame_count += 1
        if self.frame_count % 3 == 0:          # ~10 level updates a second is plenty
            self.emit("level", {"rms": round(rms(frame), 4)})
        if self.muted:
            return                             # the app is speaking: no stage, not even the pre-roll, hears it
        self.preroll.append(frame)
        if self.source is not None:
            self.queue_max = max(self.queue_max, self.source.backlog())

        if self.state == IDLE:
            if self.followup_s > 0:
                self._follow(frame)
                return
            trigger = self.stages["trigger"]
            if self.timer_idle.run("trigger", trigger.process, frame):
                self._on_trigger(trigger.last_fire)
        elif self.state == ARMED:
            self._armed(frame)
        elif self.state == LISTENING:
            self._listening(frame)

    # ---------------------------------------------------------------- states

    def _set_state(self, state):
        self.state = state
        self.emit("state", {"state": state, "t": round(self.t, 3)})

    def _follow(self, frame):
        """The follow-up window is open: the VAD alone decides when a command starts."""
        if self._follow_utt is None:
            self._follow_utt = {"t_speech_start": None, "t_speech_end": None, "speech_run": 0}
            self.stages["vad"].reset()
        self.utt = self._follow_utt
        self._vad(frame)
        if self.utt["t_speech_start"] is not None:
            self.followup_until, self._follow_utt = 0.0, None
            self._on_trigger("followup")

    def _on_trigger(self, fired_by: str):
        self.utt = {
            "started_by": fired_by, "t_trigger": self.t, "t_speech_start": None,
            "t_speech_end": None, "partials": [], "partial": "", "speech_run": 0,
        }
        self.timer = StageTimer()
        self.queue_max = 0
        self.stages["vad"].reset()
        self.stages["endpoint"].reset()
        if fired_by in ("ptt", "followup"):
            # Button, or the follow-up window: the person is already talking (or about
            # to). Keep the audio from just before: that's the pre-roll.
            self._start_listening()
        else:
            # Wake word: drop what came before (that was the wake word itself)
            # and wait for the command to start.
            self.preroll.clear()
            self._set_state(ARMED)

    def _vad(self, frame) -> bool:
        """Run the VAD; track when speech starts and ends (audio time)."""
        vad, u = self.stages["vad"], self.utt
        is_speech = self.timer.run("vad", vad.speech_prob, frame) >= vad.threshold
        u["speech_run"] = u["speech_run"] + 1 if is_speech else 0
        if u["t_speech_start"] is None and u["speech_run"] >= vad.min_speech_frames:
            # Speech has lasted long enough to be real: it started at the run's first frame.
            u["t_speech_start"] = self.t - u["speech_run"] * FRAME_S
        if is_speech and u["t_speech_start"] is not None:
            u["t_speech_end"] = self.t
        return is_speech

    def _armed(self, frame):
        self._vad(frame)
        if self.utt["t_speech_start"] is not None:
            self._start_listening()
            # The endpoint missed the frames that proved speech had started; tell it.
            self.timer.run("endpoint", self.stages["endpoint"].update, frame, True, "")
        elif self.t - self.utt["t_trigger"] > self.settings["pipeline"]["armed_timeout_s"]:
            self._back_to_idle("no_speech")

    def _start_listening(self):
        self.utt["t_listen"] = self.t
        # The pre-roll already includes the current frame, so it isn't fed again.
        self.timer.run("stt_start", self.stages["stt"].start, np.concatenate(list(self.preroll)))
        self._set_state(LISTENING)

    def _listening(self, frame):
        u = self.utt
        is_speech = self._vad(frame)
        partial = self.timer.run("stt_feed", self.stages["stt"].feed, frame)
        if partial is not None:
            u["partial"] = partial
            u["partials"].append([round(self.t, 3), partial])
            self.emit("partial", {"text": partial})
        if self.timer.run("endpoint", self.stages["endpoint"].update, frame, is_speech, u["partial"]):
            self._finalise("endpoint")
        elif self.t - u["t_listen"] > self.settings["pipeline"]["max_utterance_s"]:
            self._finalise("max_length")

    def _finalise(self, reason: str):
        u = self.utt
        u["t_turn_closed"] = self.t
        self._set_state(FINALISING)
        t0 = time.perf_counter()
        try:
            text = self.stages["stt"].finish()
        except Exception as e:
            self.emit("error", {"stage": "stt", "message": str(e)})
            text = ""
        finish_s = time.perf_counter() - t0
        self.timer.calls["stt_finish"].append(finish_s * 1000)
        record = self._record(text.strip(), reason, finish_s)
        self.emit("final", {"text": record["final"], "outcome": reason})
        self.emit("timings", {"marks_audio_s": record["marks_audio_s"], "derived": record["derived"],
                              "stage_ms": record["stage_ms"], "settings_summary": self.summary()})
        self._output(record)
        self._back_to_idle(None)

    def _back_to_idle(self, reason):
        if reason == "no_speech":
            # The trigger fired but nobody spoke: log it (counts as a false wake-up).
            self._output({"run": self.run_name, "outcome": "no_speech", **self.context,
                          "started_by": self.utt["started_by"],
                          "marks_audio_s": {"trigger": round(self.utt["t_trigger"], 3)},
                          "settings_summary": self.summary()})
        for plugin in self.stages.values():
            plugin.reset()
        self.timer_idle = StageTimer()
        self._set_state(IDLE)

    # --------------------------------------------------------------- results

    def summary(self) -> str:
        """One line naming the options in force, e.g. ptt + silero + ptt_release + parakeet/batch."""
        s = self.settings
        return " + ".join([s["trigger"]["use"], s["vad"]["use"], s["endpoint"]["use"],
                           f'{s["stt"]["use"]}/{s["stt"]["mode"]}'])

    def _record(self, text: str, reason: str, finish_s: float) -> dict:
        u = self.utt
        marks = {"trigger": u["t_trigger"], "speech_start": u["t_speech_start"],
                 "speech_end": u["t_speech_end"], "turn_closed": u["t_turn_closed"]}
        derived = {"endpoint_delay_ms": None, "stt_finish_ms": round(finish_s * 1000, 1), "felt_delay_ms": None}
        if u["t_speech_end"] is not None:
            endpoint_ms = (u["t_turn_closed"] - u["t_speech_end"]) * 1000
            derived["endpoint_delay_ms"] = round(endpoint_ms, 1)
            derived["felt_delay_ms"] = round(endpoint_ms + finish_s * 1000, 1)
        endpoint = self.stages["endpoint"]
        return {
            "run": self.run_name,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source": getattr(self.source, "kind", "unknown"),
            **self.context,
            "machine": {k: v for k, v in device_mod.describe().items() if k in ("host", "python")}
                       | {"device": self.device},
            "settings": copy.deepcopy(self.settings),
            "settings_summary": self.summary(),
            "started_by": u["started_by"],
            "outcome": reason,
            "partials": u["partials"],
            "final": text,
            "marks_audio_s": {k: (round(v, 3) if v is not None else None) for k, v in marks.items()},
            "marks_wall_s": {"turn_closed": round(u["t_turn_closed"], 3),
                             "final": round(u["t_turn_closed"] + finish_s, 3)},
            "derived": derived,
            "stage_ms": self.timer.summary() | self.timer_idle.summary(),
            "endpoint_info": {k: getattr(endpoint, k) for k in ("last_reason", "last_prob", "checks")
                              if hasattr(endpoint, k)},
            "queue_max": self.queue_max,
        }

    def _output(self, record: dict):
        for callback in self.on_record:
            callback(record)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
