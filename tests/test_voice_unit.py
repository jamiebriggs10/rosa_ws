"""voice.Voice: the object the console owns. Fake stages, fake sources, a fake speaker: no models,
no microphone, no sound card. Needs numpy only."""
import json
import sys
import threading
import time

import numpy as np
import pytest

from voice import Voice
from tests.test_voice_pipeline import CountSTT, LevelVAD, ListSource, LoudTrigger, OnRelease, QuietFor, frames_of

SETTINGS = {"on": True, "confirm": True, "input_device": None, "output_device": None, "wake_threshold": 0.5,
            "silence_ms": 700, "speech_voice": "en_GB-alan-medium", "speech_enabled": True, "sounds_enabled": True}


def fakes():
    return LoudTrigger(), LevelVAD(), QuietFor(), OnRelease(), CountSTT()


class Seen:
    def __init__(self):
        self.states, self.finals, self.felt, self.idles, self.errors = [], [], [], [], []

    def voice(self, settings=SETTINGS, **kw):
        return Voice(settings, on_state=self.states.append, on_final=lambda t: self.finals.append(t) or "sent",
                     on_timings=self.felt.append, on_idle=lambda: self.idles.append(1), on_error=self.errors.append, **kw)


class SlowSource:
    """Silent frames for ever, slowly, until stopped (stands in for the microphone)."""
    kind = "test"
    def __init__(self): self._running = False
    def start(self): self._running = True
    def stop(self): self._running = False
    def backlog(self): return 0
    def frames(self):
        while self._running:
            time.sleep(0.005)
            yield np.zeros(512, np.float32)


def wait_until(predicate, timeout=5):
    deadline = time.time() + timeout
    while not predicate():
        assert time.time() < deadline, "timed out"
        time.sleep(0.01)


def test_voice_start_reports_missing_libraries(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "openwakeword", None)             # "import openwakeword" now fails
    seen = Seen()
    v = seen.voice(make_source=lambda s: ListSource([]), sessions_dir=tmp_path)
    v.start(); v.wait(5)
    assert "openwakeword" in v.latest["error"] and v.latest["running"] is False and seen.errors


def test_voice_start_reports_a_source_error(tmp_path):
    def no_mic(settings):
        raise ValueError("no input device matching 'NoSuchMic'")
    seen = Seen()
    v = seen.voice({**SETTINGS, "input_device": "NoSuchMic"}, make_stages=fakes, make_source=no_mic, sessions_dir=tmp_path)
    v.start(); v.wait(5)
    assert "NoSuchMic" in v.latest["error"] and v.latest["running"] is False


def test_a_final_reaches_on_final_and_latest(tmp_path):
    seen = Seen()
    v = seen.voice(make_stages=fakes, make_source=lambda s: ListSource(frames_of(("click", 1), ("speech", 31), ("silence", 31))),
                   sessions_dir=tmp_path)
    v.start(); v.wait(5)
    assert seen.finals and seen.finals[0].endswith("samples") and v.latest["final"] == seen.finals[0]
    assert v.latest["action"] == "sent" and v.latest["felt_ms"] is not None and seen.felt == [v.latest["felt_ms"]]
    assert "armed" in seen.states and seen.idles
    line = json.loads((tmp_path / f"voice-{time.strftime('%Y-%m-%d')}.jsonl").read_text().splitlines()[-1])
    assert line["final"] == seen.finals[0] and line["started_by"] == "wake"


def test_changing_the_input_device_reopens_the_source(tmp_path):
    opened = []
    def make_source(settings):
        opened.append(settings.get("input_device"))
        return SlowSource()
    seen = Seen()
    v = seen.voice(make_stages=fakes, make_source=make_source, sessions_dir=tmp_path)
    v.start()
    wait_until(lambda: v.latest["running"])
    v.update({"silence_ms": 900})
    time.sleep(0.05)
    assert opened == [None] and v.pipeline.silence_end.silence_ms == 900
    v.update({"input_device": "B"})
    wait_until(lambda: opened == [None, "B"])
    v.stop(); v.wait(5)
    assert v.latest["running"] is False


def test_say_mutes_the_pipeline_while_speaking(tmp_path):
    from voice.speech import Sounds, Speaker
    def engine(text): return np.zeros(int(22050 * 0.3), np.float32), 22050
    def player(samples, rate, stop): stop.wait(0.2)
    seen = Seen()
    v = seen.voice(make_stages=fakes, make_source=lambda s: SlowSource(), sessions_dir=tmp_path,
                   make_speaker=lambda voice, on_speaking, out: (Speaker(engine, player, on_speaking=on_speaking), Sounds(player)))
    v.start()
    wait_until(lambda: v.latest["running"])
    assert v.say("Point ten.") is True
    wait_until(lambda: v.latest["speaking"] and v.pipeline.muted)
    wait_until(lambda: not v.latest["speaking"] and not v.pipeline.muted)
    assert v.play("ask") is True
    v.update({"speech_enabled": False})
    assert v.say("x") is False and "off" in v.latest["speech_reason"]
    v.stop(); v.wait(5)


class HoldSource:
    """Frames on demand: yields `first`, then waits for go() before yielding `rest`, then ends."""
    kind = "test"
    def __init__(self, first, rest):
        self.first, self.rest, self._go, self._running = first, rest, threading.Event(), False
    def start(self): self._running = True
    def stop(self): self._running = False; self._go.set()
    def backlog(self): return 0
    def go(self): self._go.set()
    def frames(self):
        yield from self.first
        self._go.wait(5)
        if self._running:
            yield from self.rest


def test_changing_the_input_device_waits_for_the_utterance_to_end(tmp_path):
    opened, sources = [], []
    def make_source(settings):
        opened.append(settings.get("input_device"))
        src = HoldSource(frames_of(("click", 1), ("speech", 10)), frames_of(("speech", 21), ("silence", 31))) if len(opened) == 1 else SlowSource()
        sources.append(src)
        return src
    seen = Seen()
    v = seen.voice(make_stages=fakes, make_source=make_source, sessions_dir=tmp_path)
    v.start()
    wait_until(lambda: v.latest["state"] == "listening")
    v.update({"input_device": "B"})                      # mid-utterance: must wait
    time.sleep(0.1)
    assert opened == [None] and v.latest["state"] == "listening" and seen.finals == []
    sources[0].go()
    wait_until(lambda: opened == [None, "B"])
    # the whole utterance: 3 confirming frames in the pre-roll, 28 more speech, 10 quiet ones to the endpoint; nothing cut
    assert seen.finals == [f"{(3 + 28 + 10) * 512} samples"]
    v.stop(); v.wait(5)


def test_a_bad_microphone_can_be_corrected_from_the_settings(tmp_path):
    opened = []
    def make_source(settings):
        opened.append(settings.get("input_device"))
        if settings.get("input_device") == "NoSuchMic":
            raise ValueError("no input device matching 'NoSuchMic'")
        return SlowSource()
    seen = Seen()
    v = seen.voice({**SETTINGS, "input_device": "NoSuchMic"}, make_stages=fakes, make_source=make_source, sessions_dir=tmp_path)
    v.start()
    wait_until(lambda: "NoSuchMic" in v.latest["error"])
    assert v.latest["running"] is False
    v.update({"input_device": "Jabra"})
    wait_until(lambda: v.latest["running"] and v.latest["error"] == "")
    assert opened == ["NoSuchMic", "Jabra"]
    v.stop(); v.wait(5)
