"""Every Flask endpoint, with the worker thread NOT started (no models load)."""

import shutil

import pytest
import yaml

import app as app_mod
import config


def make_client(tmp_path, monkeypatch, engine):
    from speech import Speaker, Sounds
    from tests.test_speech import Player
    settings = tmp_path / "settings.yaml"
    shutil.copy(config.ROOT / "settings.example.yaml", settings)
    monkeypatch.setattr(config, "PRESETS", tmp_path / "presets")
    (tmp_path / "presets").mkdir()
    player = Player()

    def make_speaker(voice, on_speaking, output_device=None):
        return Speaker(engine, player, cache_dir=tmp_path / "cache", on_speaking=on_speaking, voice_key=voice), Sounds(player)

    flask_app = app_mod.create_app(settings_path=settings, log_path=tmp_path / "live.jsonl", make_speaker=make_speaker)
    flask_app.testing = True
    c = flask_app.test_client()
    c.settings_path = settings
    c.pipe = flask_app.config["pipe"]
    c.speaker = flask_app.config["speaker"]
    c.player = player
    return c


@pytest.fixture
def client(tmp_path, monkeypatch):
    from tests.test_speech import fake_engine
    return make_client(tmp_path, monkeypatch, fake_engine)


@pytest.fixture
def client_no_engine(tmp_path, monkeypatch):
    return make_client(tmp_path, monkeypatch, lambda text: None)


def queued(client):
    out = []
    while True:
        try:
            out.append(client.pipe.commands.get_nowait())
        except Exception:
            return out


def test_page(client):
    r = client.get("/")
    assert r.status_code == 200 and b"<html" in r.data.lower()


def test_get_settings(client):
    assert client.get("/settings").get_json()["stt"]["use"] == "parakeet"


def test_post_partial_settings_saves_and_queues(client):
    r = client.post("/settings", json={"endpoint": {"use": "silence", "silence": {"silence_ms": 400}}})
    assert r.status_code == 200
    new = r.get_json()
    assert new["endpoint"]["use"] == "silence" and new["endpoint"]["silence"]["silence_ms"] == 400
    assert new["stt"]["use"] == "parakeet"                  # untouched parts kept
    saved = yaml.safe_load(client.settings_path.read_text())
    assert saved["endpoint"]["silence"]["silence_ms"] == 400
    kind, value = client.pipe.commands.get_nowait()          # handed to the pipeline
    assert kind == "settings" and value["endpoint"]["use"] == "silence"


@pytest.mark.parametrize("bad", [{"stt": {"use": "nope"}}, {"device": "tpu"}, {"stt": {"mode": "fast"}}, []])
def test_bad_settings_rejected(client, bad):
    assert client.post("/settings", json=bad).status_code == 400


def test_options(client):
    data = client.get("/options").get_json()
    assert set(data["options"]) == {"trigger", "vad", "endpoint", "stt"}
    assert data["device"] in ("metal", "cuda", "cpu")
    for stage in data["options"].values():
        for opt in stage.values():
            assert {"supported", "reason", "knobs"} <= set(opt)


def test_ptt_and_wake(client):
    assert client.post("/ptt/down").get_json()["ok"]
    assert client.post("/ptt/up").get_json()["ok"]
    assert client.post("/wake", json={"on": False}).get_json() == {"on": False}
    got = [client.pipe.commands.get_nowait() for _ in range(3)]
    assert got == [("event", "ptt_down"), ("event", "ptt_up"), ("event", "wake_off")]


def test_status_and_history(client):
    assert client.get("/status").get_json()["state"] == "idle"
    assert client.get("/history").get_json() == []


def test_presets_save_list_load(client):
    assert client.post("/presets/mine").status_code == 200
    assert client.post("/presets/mine").status_code == 409            # never silently overwritten
    assert client.post("/presets/mine?overwrite=1").status_code == 200
    assert client.get("/presets").get_json() == ["mine"]
    assert client.post("/presets/../etc").status_code in (400, 404)
    client.post("/settings", json={"vad": {"use": "webrtc"}})
    r = client.post("/presets/mine/load")
    assert r.status_code == 200 and r.get_json()["vad"]["use"] == "silero"
    assert client.post("/presets/missing/load").status_code == 404


def test_events_stream_starts(client):
    client.application.config["hub"].emit("state", {"state": "idle", "t": 0})
    r = client.get("/events", buffered=False)
    assert r.mimetype == "text/event-stream"
    first = next(r.response)                     # the stream never ends: read one event
    first = first.decode() if isinstance(first, bytes) else first
    assert first.startswith(": connected")
    r.close()


def test_unavailable_option_is_greyed_with_reason(client, monkeypatch):
    """An option whose library is missing shows as unsupported, with why, and can't be chosen."""
    from stages import registry
    monkeypatch.setitem(registry.OPTIONS["stt"], "missing_engine", "stages.stt.no_such_module:Nope")
    opt = client.get("/options").get_json()["options"]["stt"]["missing_engine"]
    assert opt["supported"] is False and "no_such_module" in opt["reason"]
    assert client.post("/settings", json={"stt": {"use": "missing_engine"}}).status_code == 400


def test_mlx_backend_refuses_without_metal():
    """Parakeet's mlx backend can only run on Apple silicon: a clear error elsewhere."""
    from stages.stt.parakeet import Parakeet
    with pytest.raises(Exception, match="(?i)metal|mlx"):
        Parakeet({"backend": "mlx"}, "cpu").load()


# ---- speaking back: /say, /play, /followup ----

def test_say_speaks_and_mutes(client):
    r = client.post("/say", json={"text": "Point ten. Confirm?"}).get_json()
    assert r["spoken"] is True
    client.speaker.wait(1)
    assert client.get("/status").get_json()["speaking"] is False
    assert client.player.calls == [6615]
    assert [c for c in queued(client) if c[0] == "event" and c[1].startswith("mute")] == [("event", "mute_on"), ("event", "mute_off")]


def test_say_without_an_engine_answers_but_does_not_speak(client_no_engine):
    r = client_no_engine.post("/say", json={"text": "x"}).get_json()
    assert r["spoken"] is False and "engine" in r["reason"]
    assert not [c for c in queued(client_no_engine) if c[0] == "event" and c[1] == "mute_on"]


def test_say_cancel_and_empty(client):
    assert client.post("/say", json={"text": ""}).status_code == 400
    assert client.post("/say", json={"cancel": True}).get_json() == {"spoken": False, "reason": "cancelled"}


def test_play_known_and_unknown(client):
    assert client.post("/play", json={"name": "ask"}).get_json()["played"] is True
    assert client.post("/play", json={"name": "zzz"}).status_code == 404


def test_followup_route_and_status(client):
    assert 7 <= client.post("/followup", json={"seconds": 8}).get_json()["until_s"] <= 8
    assert client.get("/status").get_json()["followup_s"] > 0
    assert client.post("/followup", json={"seconds": 0}).get_json()["until_s"] == 0
    assert client.get("/status").get_json()["followup_s"] == 0


def test_options_list_voices_and_status_describes_speech(client):
    assert "voices" in client.get("/options").get_json()
    speech = client.get("/status").get_json()["speech"]
    assert speech["voice"] == "en_GB-alan-medium" and speech["enabled"] is True and speech["available"] is True


def test_speaking_event_reaches_the_stream(client):
    client.post("/say", json={"text": "hello"})
    client.speaker.wait(1)
    assert client.application.config["hub"].last.get("speaking") == {"on": False}


def test_output_device_setting_reaches_the_speaker(tmp_path, monkeypatch):
    from tests.test_speech import fake_engine
    seen = []

    def make_speaker(voice, on_speaking, output_device=None):
        from speech import Speaker, Sounds
        from tests.test_speech import Player
        seen.append((voice, output_device))
        return Speaker(fake_engine, Player(), cache_dir=tmp_path / "c", on_speaking=on_speaking), Sounds(Player())
    settings = tmp_path / "settings.yaml"
    shutil.copy(config.ROOT / "settings.example.yaml", settings)
    monkeypatch.setattr(config, "PRESETS", tmp_path / "presets"); (tmp_path / "presets").mkdir()
    app = app_mod.create_app(settings_path=settings, log_path=tmp_path / "l.jsonl", make_speaker=make_speaker)
    app.testing = True
    c = app.test_client()
    assert seen[-1] == ("en_GB-alan-medium", None)
    assert c.post("/settings", json={"speech": {"output_device": "Jabra"}}).status_code == 200
    assert seen[-1] == ("en_GB-alan-medium", "Jabra")
