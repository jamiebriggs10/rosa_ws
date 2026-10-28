"""The pipeline: one thread drives the stages, frame by frame.

    IDLE --wake word--> ARMED --speech starts--> LISTENING --end of turn--> FINALISING --> IDLE
    (push-to-talk and the follow-up window skip ARMED: the person is already talking)

Everything runs on ONE thread (the console starts it in Voice.start()). Other threads talk to
it only through `commands`, a queue: a button press, the wake switch, mute while Rosa speaks,
the follow-up window, and `request(fn)` for anything that must happen between utterances
(a settings change, a new microphone).

Time is audio time: frames counted times 32 ms. It is exact, and identical for a WAV on any
machine, which is what makes the tests repeatable. Only the speech-to-text finish is clocked.
"""

import collections
import queue
import time

import numpy as np

from voice.audio import FRAME_S, rms

IDLE, ARMED, LISTENING, FINALISING = "idle", "armed", "listening", "finalising"


class Pipeline:
    def __init__(self, trigger, vad, silence_end, release_end, stt, *, emit=None,
                 armed_timeout_s=4.0, max_utterance_s=15.0, preroll_ms=500):
        self.trigger, self.vad, self.stt = trigger, vad, stt
        self.silence_end, self.release_end = silence_end, release_end
        self.endpoint = silence_end                 # the rule in force for the current utterance
        self.emit = emit or (lambda kind, data: None)
        self.on_record = []                         # callbacks given each finished utterance's record
        self.armed_timeout_s, self.max_utterance_s = armed_timeout_s, max_utterance_s
        self.preroll = collections.deque(maxlen=max(1, int(preroll_ms / 1000 / FRAME_S)))
        self.commands = queue.Queue()
        self._pending = []                          # requests waiting for the next idle moment
        self.state = IDLE
        self.wake_on = True
        self.muted = False                          # while Rosa speaks: frames reach only the trigger (barge-in)
        self.followup_until = 0.0                   # time.monotonic() until which no wake word is needed
        self._follow_utt = None
        self.frame_count = 0
        self._stop = False
        self.source = None
        self.utt = None

    def load(self):
        for stage in (self.trigger, self.vad, self.stt):
            stage.load()

    # -- from other threads ---------------------------------------------------------------

    def ptt_down(self):
        self.commands.put(("event", "ptt_down"))

    def ptt_up(self):
        self.commands.put(("event", "ptt_up"))

    def set_wake(self, on):
        self.commands.put(("event", "wake_on" if on else "wake_off"))

    def set_muted(self, on):
        self.commands.put(("event", "mute_on" if on else "mute_off"))

    def followup(self, seconds):
        """For `seconds`, speech counts as triggered without the wake word. 0 closes the window."""
        until = time.monotonic() + float(seconds) if seconds > 0 else 0.0
        self.followup_until = until                 # readable at once; the thread resets its scratch below
        self.commands.put(("followup", until))

    @property
    def followup_s(self):
        return max(0.0, self.followup_until - time.monotonic())

    def request(self, fn):
        """Run `fn` on the pipeline's thread, between utterances."""
        self.commands.put(("request", fn))

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
            if kind == "event":
                self._event(value)
            elif kind == "followup":
                self.followup_until, self._follow_utt = value, None
                self.emit("followup", {"until_s": round(self.followup_s, 2)})
            elif kind == "request":
                self._pending.append(value)
        if self._pending and self.state == IDLE:
            pending, self._pending = self._pending, []
            for fn in pending:
                fn()

    def _event(self, name):
        if name == "ptt_down":
            self.trigger.press()
            self.release_end.press()
        elif name == "ptt_up":
            self.release_end.release()
        elif name in ("wake_on", "wake_off"):
            self.wake_on = name == "wake_on"
            self.trigger.enable_wake(self.wake_on)
        elif name in ("mute_on", "mute_off"):
            self.muted = name == "mute_on"
            self.preroll.clear()                    # nothing of Rosa's own voice leads the next transcript

    # -- the loop --------------------------------------------------------------------------

    @property
    def t(self):
        """Audio time (s) at the end of the current frame."""
        return self.frame_count * FRAME_S

    def run(self, source, events=None):
        """Process a source until it ends (a file) or stop() is called (the microphone).
        `events`: [(audio_time_s, "ptt_down"|"ptt_up"|"mute_on"|"mute_off")] for the tests."""
        events = sorted(events or [])
        self.source = source
        self._stop = False
        self.frame_count = 0
        self.preroll.clear()
        source.start()
        try:
            for frame in source.frames():
                while events and events[0][0] <= self.t:
                    self.commands.put(("event", events.pop(0)[1]))
                self.step(frame)
                if self._stop:
                    break
        finally:
            source.stop()
        if self.state == LISTENING:                 # the audio ran out mid-utterance: finish it rather than lose it
            self._finalise("end_of_audio")
        elif self.state == ARMED:
            self._back_to_idle("no_speech")

    def step(self, frame):
        self._handle_commands()
        self.frame_count += 1
        if self.frame_count % 3 == 0:               # about ten level readings a second
            self.emit("level", {"rms": round(rms(frame), 4)})
        if self.muted:                              # Rosa is speaking: only the wake word and the button get through
            if self.state == IDLE and self.trigger.process(frame):
                self.emit("interrupt", {"by": self.trigger.last_fire})
                self._on_trigger(self.trigger.last_fire)
            return
        self.preroll.append(frame)
        if self.state == IDLE:
            if self.followup_s > 0:
                self._follow(frame)
            elif self.trigger.process(frame):
                self._on_trigger(self.trigger.last_fire)
        elif self.state == ARMED:
            self._armed(frame)
        elif self.state == LISTENING:
            self._listening(frame)

    def _set_state(self, state):
        self.state = state
        self.emit("state", {"state": state, "t": round(self.t, 3)})

    def _follow(self, frame):
        """The follow-up window is open: the VAD alone decides when a command starts."""
        if self._follow_utt is None:
            self._follow_utt = {"t_speech_start": None, "t_speech_end": None, "speech_run": 0}
            self.vad.reset()
        self.utt = self._follow_utt
        self._vad(frame)
        if self.utt["t_speech_start"] is not None:
            self.followup_until, self._follow_utt = 0.0, None
            self._on_trigger("followup")

    def _on_trigger(self, fired_by):
        self.utt = {"started_by": fired_by, "t_trigger": self.t, "t_speech_start": None, "t_speech_end": None,
                    "speech_run": 0}
        self.vad.reset()
        self.endpoint = self.release_end if fired_by == "ptt" else self.silence_end
        self.endpoint.reset()
        if fired_by in ("ptt", "followup"):         # already talking: keep the audio from just before
            self._start_listening()
        else:                                       # the wake word itself is dropped; wait for the command
            self.preroll.clear()
            self._set_state(ARMED)

    def _vad(self, frame):
        u = self.utt
        is_speech = self.vad.speech_prob(frame) >= self.vad.threshold
        u["speech_run"] = u["speech_run"] + 1 if is_speech else 0
        if u["t_speech_start"] is None and u["speech_run"] >= self.vad.min_speech_frames:
            u["t_speech_start"] = self.t - u["speech_run"] * FRAME_S     # it started at the run's first frame
        if is_speech and u["t_speech_start"] is not None:
            u["t_speech_end"] = self.t
        return is_speech

    def _armed(self, frame):
        self._vad(frame)
        if self.utt["t_speech_start"] is not None:
            self._start_listening()
            self.endpoint.update(True)              # the frames that proved speech had started
        elif self.t - self.utt["t_trigger"] > self.armed_timeout_s:
            self._back_to_idle("no_speech")

    def _start_listening(self):
        self.utt["t_listen"] = self.t
        self.stt.start(np.concatenate(list(self.preroll)))
        self._set_state(LISTENING)

    def _listening(self, frame):
        is_speech = self._vad(frame)
        self.stt.feed(frame)
        if self.endpoint.update(is_speech):
            self._finalise("endpoint")
        elif self.t - self.utt["t_listen"] > self.max_utterance_s:
            self._finalise("max_length")

    def _finalise(self, reason):
        u = self.utt
        u["t_turn_closed"] = self.t
        self._set_state(FINALISING)
        t0 = time.perf_counter()
        try:
            text = self.stt.finish()
        except Exception as exc:
            self.emit("error", {"stage": "stt", "message": str(exc)})
            text = ""
        finish_s = time.perf_counter() - t0
        record = self._record(text.strip(), reason, finish_s)
        self.emit("final", {"text": record["final"], "outcome": reason})
        self.emit("timings", {"marks_audio_s": record["marks_audio_s"], "derived": record["derived"]})
        for callback in self.on_record:
            callback(record)
        self._back_to_idle(None)

    def _back_to_idle(self, reason):
        if reason == "no_speech":                   # the trigger fired but nobody spoke (a false wake-up)
            record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "started_by": self.utt["started_by"],
                      "outcome": "no_speech", "final": "", "marks_audio_s": {"trigger": round(self.utt["t_trigger"], 3)},
                      "derived": {}}
            for callback in self.on_record:
                callback(record)
        for stage in (self.trigger, self.vad, self.endpoint, self.stt):
            stage.reset()
        self._set_state(IDLE)

    def _record(self, text, reason, finish_s):
        u = self.utt
        marks = {"trigger": u["t_trigger"], "speech_start": u["t_speech_start"], "speech_end": u["t_speech_end"],
                 "turn_closed": u["t_turn_closed"]}
        derived = {"endpoint_delay_ms": None, "stt_finish_ms": round(finish_s * 1000, 1), "felt_delay_ms": None}
        if u["t_speech_end"] is not None:
            endpoint_ms = (u["t_turn_closed"] - u["t_speech_end"]) * 1000
            derived["endpoint_delay_ms"] = round(endpoint_ms, 1)
            derived["felt_delay_ms"] = round(endpoint_ms + finish_s * 1000, 1)
        return {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "started_by": u["started_by"], "outcome": reason,
                "final": text, "marks_audio_s": {k: (round(v, 3) if v is not None else None) for k, v in marks.items()},
                "derived": derived}
