"""voice/pipeline.py: the state machine, pre-roll and timestamps, with fake stages (no models).
The fakes make the timings exact: the fake VAD calls a frame speech if it is loud, the fake
STT returns how many samples it was given."""
import numpy as np
import pytest

from voice.pipeline import ARMED, FRAME_S, IDLE, LISTENING, Pipeline


class LoudTrigger:
    """Fires (as a wake word) on the first frame louder than 0.5."""
    last_fire = "wake"
    def load(self): pass
    def reset(self): pass
    def press(self): pass
    def enable_wake(self, on): pass
    def process(self, frame): return float(np.abs(frame).max()) > 0.5


class ButtonTrigger:
    last_fire = "ptt"
    def __init__(self): self.pressed = False
    def load(self): pass
    def reset(self): self.pressed = False
    def press(self): self.pressed = True
    def enable_wake(self, on): pass
    def process(self, frame):
        fired, self.pressed = self.pressed, False
        return fired


class LevelVAD:
    threshold, min_speech_frames = 0.5, 3
    def load(self): pass
    def reset(self): pass
    def speech_prob(self, frame): return 1.0 if float(np.abs(frame).max()) > 0.05 else 0.0


class QuietFor:
    """Finished after 10 non-speech frames following speech."""
    def __init__(self): self.reset()
    def reset(self): self.heard, self.quiet = False, 0
    def update(self, is_speech):
        self.heard |= is_speech
        self.quiet = 0 if is_speech else self.quiet + 1
        return self.heard and self.quiet >= 10


class OnRelease:
    """Finished when released; never by silence."""
    def __init__(self): self.held = False
    def press(self): self.held = True
    def release(self): self.held = False
    def reset(self): pass
    def update(self, is_speech): return not self.held


class CountSTT:
    def load(self): pass
    def reset(self): pass
    def start(self, preroll): self.n = self.preroll_len = len(preroll)
    def feed(self, frame): self.n += len(frame)
    def finish(self): return f"{self.n} samples"


def frames_of(*parts):
    """Audio from (kind, number of 32 ms frames): "silence", "click" (wake), "speech"."""
    out = []
    for kind, n in parts:
        out += [np.full(512, {"silence": 0.0, "click": 0.9, "speech": 0.2}[kind], np.float32) for _ in range(n)]
    return out


class ListSource:
    kind = "test"
    def __init__(self, frames): self._frames = frames
    def start(self): pass
    def stop(self): pass
    def backlog(self): return 0
    def frames(self): yield from self._frames


def make(trigger=None, events=None, **kw):
    pipe = Pipeline(trigger or LoudTrigger(), LevelVAD(), QuietFor(), OnRelease(), CountSTT(),
                    emit=(lambda k, d: events.append((k, d))) if events is not None else None, **kw)
    return pipe


def run(pipe, frames, events=None):
    records = []
    pipe.on_record.append(records.append)
    pipe.run(ListSource(frames), events)
    return records


def test_wake_flow_and_timings():
    events = []
    pipe = make(events=events)
    recs = run(pipe, frames_of(("silence", 31), ("click", 1), ("silence", 10), ("speech", 31), ("silence", 31)))
    assert len(recs) == 1
    m = recs[0]["marks_audio_s"]
    assert m["trigger"] == pytest.approx(32 * FRAME_S, abs=1e-6)           # the end of the click frame
    assert m["speech_start"] == pytest.approx(42 * FRAME_S, abs=1e-6)      # confirmed 3 frames later, dated at its start
    assert m["speech_end"] == pytest.approx(73 * FRAME_S, abs=1e-6)
    assert recs[0]["derived"]["endpoint_delay_ms"] == pytest.approx(320, abs=1)
    assert recs[0]["derived"]["felt_delay_ms"] >= recs[0]["derived"]["endpoint_delay_ms"]
    assert recs[0]["started_by"] == "wake" and recs[0]["final"].endswith("samples")
    assert [d["state"] for k, d in events if k == "state"] == [ARMED, LISTENING, "finalising", IDLE]
    assert any(k == "final" for k, _ in events) and any(k == "timings" for k, _ in events)


def test_preroll_starts_at_wake_trigger():
    pipe = make()
    run(pipe, frames_of(("silence", 31), ("click", 1), ("silence", 5), ("speech", 31), ("silence", 31)))
    assert pipe.stt.preroll_len == 8 * 512          # the 5 silent frames after the wake word + 3 confirming speech
    pipe = make()
    run(pipe, frames_of(("click", 1), ("silence", 40), ("speech", 31), ("silence", 31)))
    assert pipe.stt.preroll_len == 15 * 512         # capped at preroll_ms (500 ms)


def test_armed_timeout_goes_back_to_idle():
    pipe = make(armed_timeout_s=1.0)
    recs = run(pipe, frames_of(("silence", 16), ("click", 1), ("silence", 62)))
    assert [r["outcome"] for r in recs] == ["no_speech"] and pipe.state == IDLE


def test_push_to_talk_skips_armed_and_keeps_preroll():
    events = []
    pipe = make(ButtonTrigger(), events=events)
    recs = run(pipe, frames_of(("silence", 31), ("speech", 31), ("silence", 31)), events=[(0.96, "ptt_down"), (1.98, "ptt_up")])
    assert [d["state"] for k, d in events if k == "state"][0] == LISTENING
    assert recs[0]["started_by"] == "ptt" and pipe.stt.preroll_len == int(0.5 / FRAME_S) * 512


def test_release_endpoint_is_used_for_a_ptt_utterance():
    pipe = make(ButtonTrigger())
    # the silence endpoint would close after 10 quiet frames; held, the button keeps the turn open for 40
    recs = run(pipe, frames_of(("speech", 10), ("silence", 40), ("speech", 5), ("silence", 5)),
               events=[(0.0, "ptt_down"), (60 * FRAME_S, "ptt_up")])
    assert len(recs) == 1 and recs[0]["marks_audio_s"]["turn_closed"] == pytest.approx(60 * FRAME_S, abs=FRAME_S)


def test_end_of_audio_finishes_the_utterance():
    recs = run(make(), frames_of(("click", 1), ("speech", 31)))
    assert recs[0]["outcome"] == "end_of_audio"


def test_max_utterance_cap():
    recs = run(make(max_utterance_s=1.0), frames_of(("click", 1), ("speech", 94), ("silence", 31)))
    assert recs[0]["outcome"] == "max_length"


def test_settings_change_waits_until_idle():
    pipe = make()
    done = []
    pipe.run(ListSource(frames_of(("click", 1), ("speech", 16))))      # ends mid-utterance (end_of_audio)
    pipe.request(lambda: done.append(pipe.state))                       # a change asked for while it was busy
    assert done == []
    pipe.run(ListSource(frames_of(("silence", 3))))
    assert done == [IDLE]                                               # applied between utterances, on the pipeline's thread


def test_muted_frames_reach_no_stage_but_the_trigger():
    pipe = make()
    pipe.set_muted(True)
    recs = run(pipe, frames_of(("click", 1), ("speech", 31), ("silence", 31)))
    # the wake word fired (barge-in), but the speech was never heard: the mute was never lifted
    assert [r["outcome"] for r in recs] == ["no_speech"] and pipe.state == IDLE
    pipe.set_muted(False)
    assert len(run(pipe, frames_of(("click", 1), ("speech", 31), ("silence", 31)))) == 1


def test_unmute_clears_the_preroll():
    pipe = make()
    pipe.followup(5)
    # 20 frames of Rosa's own voice while muted, then the operator speaks: the pre-roll must hold only
    # the 3 frames that confirmed speech, not Rosa's last words.
    run(pipe, frames_of(("speech", 20), ("speech", 31), ("silence", 31)), events=[(0.0, "mute_on"), (20 * FRAME_S, "mute_off")])
    assert pipe.stt.preroll_len == 3 * 512


def test_followup_listens_without_the_wake_word():
    events = []
    pipe = make(events=events)
    pipe.followup(5)
    recs = run(pipe, frames_of(("silence", 3), ("speech", 31), ("silence", 31)))
    assert len(recs) == 1 and recs[0]["started_by"] == "followup"
    assert [d["state"] for k, d in events if k == "state"][0] == LISTENING
    assert pipe.followup_s == 0                                         # one utterance closes the window


def test_followup_keeps_preroll():
    pipe = make()
    pipe.followup(5)
    run(pipe, frames_of(("silence", 3), ("speech", 31), ("silence", 31)))
    assert pipe.stt.preroll_len == 6 * 512


def test_followup_window_expires(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("voice.pipeline.time.monotonic", lambda: now[0])
    pipe = make()
    pipe.followup(2)
    now[0] = 3.0
    recs = run(pipe, frames_of(("silence", 3), ("speech", 31), ("silence", 31)))
    assert recs == [] and pipe.state == IDLE and pipe.followup_s == 0


def test_the_wake_word_interrupts_while_rosa_speaks():
    """Muted (Rosa speaking), the VAD and speech-to-text hear nothing, but the wake word still fires:
    the pipeline says "interrupt" and arms; once the mute lifts (Rosa hushed), the command is heard."""
    events = []
    pipe = make(events=events)
    pipe.set_muted(True)
    recs = run(pipe, frames_of(("speech", 5), ("click", 1), ("speech", 31), ("silence", 31)),
               events=[(6 * FRAME_S, "mute_off")])          # Voice lifts the mute as soon as Rosa is hushed
    assert [d for k, d in events if k == "interrupt"] == [{"by": "wake"}]
    assert len(recs) == 1 and recs[0]["started_by"] == "wake" and recs[0]["final"] == f"{(3 + 28 + 10) * 512} samples"
    assert [d["state"] for k, d in events if k == "state"][0] == ARMED


def test_speech_while_muted_without_the_wake_word_is_dropped():
    pipe = make()
    pipe.set_muted(True)
    assert run(pipe, frames_of(("speech", 31), ("silence", 31))) == [] and pipe.state == IDLE
