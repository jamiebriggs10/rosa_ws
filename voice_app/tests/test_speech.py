"""speech.py: the Speaker (text to played audio, with a phrase cache) and the state sounds.
Pure Python with fakes for the engine and the player: no Piper, no sound card."""

import time

import numpy as np

from speech import SOUNDS, Sounds, Speaker, make_sounds


def fake_engine(text):
    return np.zeros(int(22050 * 0.3), dtype=np.float32), 22050


class Player:
    def __init__(self, raise_midway=False):
        self.calls, self.raise_midway = [], raise_midway

    def __call__(self, samples, rate, stop):
        self.calls.append(len(samples))
        if self.raise_midway:
            raise RuntimeError("device gone")
        stop.wait(0.2)


def test_say_plays_and_reports_speaking(tmp_path):
    seen, p = [], Player()
    s = Speaker(fake_engine, p, cache_dir=tmp_path, on_speaking=seen.append)
    assert s.say("Point ten.") is True
    s.wait(1)
    assert p.calls == [6615] and seen == [True, False] and s.speaking is False


def test_fixed_phrases_are_cached(tmp_path):
    n = {"calls": 0}

    def counting(text):
        n["calls"] += 1
        return fake_engine(text)

    s = Speaker(counting, Player(), cache_dir=tmp_path)
    s.say("Cancelled.")
    s.wait(1)
    s.say("Cancelled.")
    s.wait(1)
    assert n["calls"] == 1 and len(list(tmp_path.glob("*.npy"))) == 1


def test_second_say_cancels_the_first(tmp_path):
    p = Player()
    s = Speaker(fake_engine, p, cache_dir=tmp_path)
    s.say("one")
    time.sleep(0.05)
    s.say("two")
    s.wait(1)
    assert p.calls == [6615, 6615] and s.speaking is False


def test_say_without_an_engine_reports_why(tmp_path):
    s = Speaker(lambda text: None, Player(), cache_dir=tmp_path)
    assert s.say("x") is False and "engine" in s.reason


def test_mute_is_released_when_playback_raises(tmp_path):
    seen = []
    s = Speaker(fake_engine, Player(raise_midway=True), cache_dir=tmp_path, on_speaking=seen.append)
    s.say("x")
    s.wait(1)
    assert seen == [True, False] and s.speaking is False


def test_sounds_are_short_and_known():
    assert set(make_sounds()) == {"listen", "heard", "ask", "go", "done", "cancel", "refused", "think"}
    assert all(0.03 <= len(v) / 22050 <= 0.6 for v in SOUNDS.values())
    p = Player()
    snd = Sounds(p)
    assert snd.play("ask") is True and snd.play("nope") is False
    time.sleep(0.3)
    assert len(p.calls) == 1


class FakeSd:
    """Just enough of sounddevice: OutputStream objects that record their writes and closes."""

    def __init__(self):
        self.streams = []
        self.global_stops = 0

    class OutputStream:
        def __init__(self, outer, samplerate, channels, device, dtype):
            self.outer, self.samplerate, self.device = outer, samplerate, device
            self.writes, self.closed, self.started = [], False, False
            outer.streams.append(self)

        def start(self): self.started = True
        def write(self, chunk): self.writes.append(len(chunk))
        def stop(self): pass
        def close(self): self.closed = True

    def output_stream(self, **kw):
        return FakeSd.OutputStream(self, **kw)

    def stop(self):
        self.global_stops += 1


def test_sounddevice_player_uses_its_own_stream_and_never_the_global_stop():
    from speech import sounddevice_player
    import threading
    sd = FakeSd()
    sd.OutputStream = lambda **kw: FakeSd.OutputStream(sd, **kw)
    player = sounddevice_player("Jabra", sd_module=sd)
    player(np.zeros(22050, dtype=np.float32), 22050, threading.Event())
    assert len(sd.streams) == 1 and sd.streams[0].closed and sd.streams[0].device == "Jabra"
    assert sum(sd.streams[0].writes) == 22050 and sd.global_stops == 0


def test_sounddevice_player_stops_its_stream_when_asked():
    from speech import sounddevice_player
    import threading
    sd = FakeSd()
    sd.OutputStream = lambda **kw: FakeSd.OutputStream(sd, **kw)
    stop = threading.Event()
    stop.set()
    sounddevice_player(None, sd_module=sd)(np.zeros(22050 * 5, dtype=np.float32), 22050, stop)
    assert sum(sd.streams[0].writes) < 22050 * 5 and sd.streams[0].closed
