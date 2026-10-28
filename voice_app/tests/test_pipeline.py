"""The state machine, pre-roll and timestamps, with fake stages (no models).

The fakes make the timings exact: the fake VAD calls a frame speech if it is
loud, the fake STT returns how many samples it was given.
"""

import json

import numpy as np
import pytest

import config
from audio import FileSource
from bench import say
from pipeline import ARMED, FRAME_S, IDLE, LISTENING, Pipeline
from stages import registry
from stages.base import VAD, EndOfTurn, SpeechToText, Trigger


class LoudTrigger(Trigger):
    """Fires (as a wake word) on the first frame louder than 0.5."""
    name = "loud"

    def process(self, frame):
        self.last_fire = "wake"
        return float(np.abs(frame).max()) > 0.5


class ButtonTrigger(Trigger):
    name = "button"

    def __init__(self, settings, device):
        super().__init__(settings, device)
        self.pressed = False

    def on_event(self, event):
        self.pressed = self.pressed or event == "ptt_down"

    def process(self, frame):
        fired, self.pressed = self.pressed, False
        self.last_fire = "ptt"
        return fired


class LevelVAD(VAD):
    name = "level"
    knobs = {"threshold": {"type": "float", "default": 0.5},
             "min_speech_ms": {"type": "int", "default": 96}}   # 3 frames

    def speech_prob(self, frame):
        return 1.0 if float(np.abs(frame).max()) > 0.05 else 0.0


class QuietFor(EndOfTurn):
    """Finished after 10 non-speech frames following speech."""
    name = "quiet"

    def reset(self):
        self.heard, self.quiet = False, 0

    def update(self, frame, is_speech, partial_text):
        if not hasattr(self, "heard"):
            self.reset()
        self.heard |= is_speech
        self.quiet = 0 if is_speech else self.quiet + 1
        return self.heard and self.quiet >= 10


class CountSTT(SpeechToText):
    name = "count"

    def start(self, preroll):
        self.n = len(preroll)
        self.preroll_len = len(preroll)

    def feed(self, frame):
        self.n += len(frame)
        return f"{self.n} samples" if self.settings.get("mode") == "streaming" else None

    def finish(self):
        return f"{self.n} samples"


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setitem(registry.OPTIONS, "trigger", {"loud": f"{__name__}:LoudTrigger",
                                                      "button": f"{__name__}:ButtonTrigger"})
    monkeypatch.setitem(registry.OPTIONS, "vad", {"level": f"{__name__}:LevelVAD"})
    monkeypatch.setitem(registry.OPTIONS, "endpoint", {"quiet": f"{__name__}:QuietFor"})
    monkeypatch.setitem(registry.OPTIONS, "stt", {"count": f"{__name__}:CountSTT"})


def settings(trigger="loud", mode="batch"):
    s = config.load(config.ROOT / "settings.example.yaml")
    s["trigger"]["use"], s["vad"]["use"], s["endpoint"]["use"], s["stt"]["use"] = trigger, "level", "quiet", "count"
    s["stt"]["mode"] = mode
    return s


def frames_of(*parts):
    """Build audio from (kind, number of 32 ms frames): "silence", "click" (wake), "speech"."""
    out = []
    for kind, n in parts:
        value = {"silence": 0.0, "click": 0.9, "speech": 0.2}[kind]
        out += [np.full(512, value, np.float32) for _ in range(n)]
    return out


class ListSource:
    kind = "test"

    def __init__(self, frames):
        self._frames = frames

    def start(self): pass
    def stop(self): pass
    def backlog(self): return 0

    def frames(self):
        yield from self._frames


def run(pipe, frames, events=None):
    records = []
    pipe.on_record.append(records.append)
    pipe.run(ListSource(frames), events)
    return records


def test_wake_flow_and_timings(fakes):
    events = []
    pipe = Pipeline(settings(), emit=lambda k, d: events.append((k, d)))
    # 31 frames silence, 1 wake "click", 10 silence, 31 speech, 31 silence
    recs = run(pipe, frames_of(("silence", 31), ("click", 1), ("silence", 10),
                               ("speech", 31), ("silence", 31)))
    assert len(recs) == 1
    r = recs[0]
    m = r["marks_audio_s"]
    # trigger fires at the END of the click frame (frame 32)
    assert m["trigger"] == pytest.approx(32 * FRAME_S, abs=1e-6)
    # speech_start is the first speech frame's start, though it is confirmed 3 frames later
    assert m["speech_start"] == pytest.approx(42 * FRAME_S, abs=1e-6)
    assert m["speech_end"] == pytest.approx(73 * FRAME_S, abs=1e-6)
    # endpoint: 10 quiet frames after speech
    assert r["derived"]["endpoint_delay_ms"] == pytest.approx(320, abs=1)
    assert r["derived"]["felt_delay_ms"] >= r["derived"]["endpoint_delay_ms"]
    states = [d["state"] for k, d in events if k == "state"]
    assert states == [ARMED, LISTENING, "finalising", IDLE]
    assert any(k == "final" for k, _ in events) and any(k == "timings" for k, _ in events)


def test_preroll_starts_at_wake_trigger(fakes):
    pipe = Pipeline(settings())
    run(pipe, frames_of(("silence", 31), ("click", 1), ("silence", 5), ("speech", 31), ("silence", 31)))
    # Pre-roll is cut at the trigger (the wake word is dropped): it holds the
    # 5 silent frames after it plus the 3 speech frames that confirmed speech.
    assert pipe.stages["stt"].preroll_len == 8 * 512
    # A long gap is capped at preroll_ms (500 ms = 15 frames).
    pipe = Pipeline(settings())
    run(pipe, frames_of(("click", 1), ("silence", 40), ("speech", 31), ("silence", 31)))
    assert pipe.stages["stt"].preroll_len == 15 * 512


def test_armed_timeout_goes_back_to_idle(fakes):
    s = settings()
    s["pipeline"]["armed_timeout_s"] = 1.0
    pipe = Pipeline(s)
    recs = run(pipe, frames_of(("silence", 16), ("click", 1), ("silence", 62)))
    assert [r["outcome"] for r in recs] == ["no_speech"]
    assert pipe.state == IDLE


def test_push_to_talk_skips_armed_and_keeps_preroll(fakes):
    events = []
    pipe = Pipeline(settings("button"), emit=lambda k, d: events.append((k, d)))
    audio = frames_of(("silence", 31), ("speech", 31), ("silence", 31))
    recs = run(pipe, audio, events=[(0.96, "ptt_down")])
    states = [d["state"] for k, d in events if k == "state"]
    assert states[0] == LISTENING
    assert recs[0]["started_by"] == "ptt"
    # pressed just before speech: the pre-roll holds the last 0.5 s of audio
    assert pipe.stages["stt"].preroll_len == int(0.5 / FRAME_S) * 512


def test_streaming_partials_are_recorded(fakes):
    events = []
    pipe = Pipeline(settings(mode="streaming"), emit=lambda k, d: events.append((k, d)))
    recs = run(pipe, frames_of(("click", 1), ("speech", 16), ("silence", 31)))
    assert len(recs[0]["partials"]) > 5
    assert any(k == "partial" for k, _ in events)


def test_end_of_audio_finishes_the_utterance(fakes):
    pipe = Pipeline(settings())
    recs = run(pipe, frames_of(("click", 1), ("speech", 31)))
    assert recs[0]["outcome"] == "end_of_audio"


def test_max_utterance_cap(fakes):
    s = settings()
    s["pipeline"]["max_utterance_s"] = 1.0
    pipe = Pipeline(s)
    recs = run(pipe, frames_of(("click", 1), ("speech", 94), ("silence", 31)))
    assert recs[0]["outcome"] == "max_length"


def test_settings_change_waits_until_idle(fakes):
    pipe = Pipeline(settings())
    pipe.load()
    first = pipe.stages["stt"]
    new = settings()
    new["stt"]["prompt"] = "changed"
    # Arrives mid-utterance: must not swap the engine until the utterance is over.
    frames = frames_of(("click", 1), ("speech", 16))
    pipe.run(ListSource(frames))  # ends with end_of_audio
    pipe.request_settings(new)
    pipe.run(ListSource(frames_of(("silence", 3))))
    assert pipe.stages["stt"] is not first
    assert pipe.stages["stt"].settings["prompt"] == "changed"


def test_unchanged_stage_is_not_reloaded(fakes):
    pipe = Pipeline(settings())
    pipe.load()
    vad = pipe.stages["vad"]
    s = settings()
    s["endpoint"]["silence"]["silence_ms"] = 123   # a different option's knob
    pipe.request_settings(s)
    pipe.run(ListSource(frames_of(("silence", 3))))
    assert pipe.stages["vad"] is vad


def test_jsonl_log_written(fakes, tmp_path):
    log = tmp_path / "run.jsonl"
    pipe = Pipeline(settings(), log_path=log)
    run(pipe, frames_of(("click", 1), ("speech", 16), ("silence", 31)))
    line = json.loads(log.read_text().splitlines()[0])
    for key in ("settings", "marks_audio_s", "marks_wall_s", "derived", "stage_ms", "final", "queue_max"):
        assert key in line


def test_file_source_matches_frames(clip, tmp_path):
    audio = clip("go home")
    src = FileSource(clip.path("go home"))
    src.start()
    got = np.concatenate(list(src.frames()))
    assert np.allclose(got[:len(audio)], audio, atol=1e-4)
    assert len(got) % 512 == 0


def test_file_source_resamples_48k(tmp_path):
    import soundfile as sf
    t = np.arange(48000) / 48000
    sf.write(tmp_path / "a.wav", 0.5 * np.sin(2 * np.pi * 440 * t), 48000)
    src = FileSource(tmp_path / "a.wav")
    assert src.duration_s == pytest.approx(1.0, abs=0.01)


# ---- speaking back: mute while the app talks; a follow-up window without the wake word ----

def test_muted_frames_reach_no_stage(fakes):
    pipe = Pipeline(settings())
    pipe.set_muted(True)
    recs = run(pipe, frames_of(("click", 1), ("speech", 31), ("silence", 31)))
    assert recs == [] and pipe.state == IDLE
    pipe.set_muted(False)
    recs = run(pipe, frames_of(("click", 1), ("speech", 31), ("silence", 31)))
    assert len(recs) == 1


def test_followup_listens_without_the_wake_word(fakes):
    events = []
    pipe = Pipeline(settings(), emit=lambda k, d: events.append((k, d)))
    pipe.followup(5)
    recs = run(pipe, frames_of(("silence", 3), ("speech", 31), ("silence", 31)))
    assert len(recs) == 1 and recs[0]["started_by"] == "followup"
    states = [d["state"] for k, d in events if k == "state"]
    assert states[0] == LISTENING                     # straight in, no ARMED
    assert pipe.followup_s == 0                       # one utterance closes the window


def test_followup_keeps_preroll(fakes):
    pipe = Pipeline(settings())
    pipe.followup(5)
    run(pipe, frames_of(("silence", 3), ("speech", 31), ("silence", 31)))
    # 3 silent frames plus the 3 speech frames that confirmed speech: nothing of the first word is lost
    assert pipe.stages["stt"].preroll_len == 6 * 512


def test_followup_window_expires(fakes, monkeypatch):
    now = [0.0]
    monkeypatch.setattr("pipeline.time.monotonic", lambda: now[0])
    pipe = Pipeline(settings())
    pipe.followup(2)
    now[0] = 3.0
    recs = run(pipe, frames_of(("silence", 3), ("speech", 31), ("silence", 31)))
    assert recs == [] and pipe.state == IDLE and pipe.followup_s == 0


def test_muted_frames_do_not_enter_the_preroll(fakes):
    pipe = Pipeline(settings())
    pipe.followup(5)
    # 20 frames of the app's own voice while muted, then the operator speaks: the pre-roll must
    # hold only the 3 frames that confirmed speech, not Rosa's last words.
    run(pipe, frames_of(("speech", 20), ("speech", 31), ("silence", 31)),
        events=[(0.0, "mute_on"), (20 * FRAME_S, "mute_off")])
    assert pipe.stages["stt"].preroll_len == 3 * 512
