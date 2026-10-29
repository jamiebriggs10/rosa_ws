"""tests/mock_console.py: the scripted console for page work on the Mac. Pure Python + Flask.
    python3 -m pytest tests/test_mock_console.py"""
import sys
import pathlib
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
mock_console = pytest.importorskip("mock_console")


def wait_until(predicate, timeout=6):
    deadline = time.time() + timeout
    while not predicate():
        assert time.time() < deadline, "timed out"
        time.sleep(0.05)


@pytest.fixture
def client():
    app = mock_console.create_app(think_s=0.2, rows=12)
    app.testing = True
    return app.test_client()


def test_state_and_pages(client):
    s = client.get("/state").get_json()
    assert s["phase"] == "idle" and s["status"]["tool_cm"] and s["limits"]["MAX_STEP_CM"] == 20
    assert client.get("/").status_code == 200 and client.get("/stage").status_code == 404
    assert client.get("/static/robot.urdf").status_code == 200
    assert client.get("/pkg/ur_description/meshes/ur10e/visual/base.dae").status_code == 200
    caps = client.get("/capabilities").get_json()["interfaces"]
    assert sum(1 for r in caps if r["status"] == "undescribed") == 12 and any(r["node"] == "moveit" for r in caps)


def test_raster_point_and_confirm(client):
    assert client.post("/command", json={"text": "show me the raster and its points"}).status_code == 200
    wait_until(lambda: client.get("/state").get_json()["shown_path"])
    assert len(client.get("/state").get_json()["shown_path"]["points_cm"]) == 24
    client.post("/command", json={"text": "go to point 10"})
    wait_until(lambda: client.get("/state").get_json()["pending"])
    s = client.get("/state").get_json()
    assert s["pending"]["facts"]["label"] == "Point 10" and s["phase"] == "proposing" and s["pending"]["preview"]["path_deg"]
    client.post("/confirm", json={"id": s["pending"]["id"], "ok": True})
    wait_until(lambda: client.get("/state").get_json()["phase"] in ("done", "follow"), timeout=10)
    assert client.get("/state").get_json()["messages"][-1]["events"][-1]["outcome"] == "done"


def test_no_rest_and_tour(client):
    client.post("/command", json={"text": "show me the raster"})
    wait_until(lambda: client.get("/state").get_json()["shown_path"])
    client.post("/command", json={"text": "go to point 10"})
    wait_until(lambda: client.get("/state").get_json()["pending"])
    pid = client.get("/state").get_json()["pending"]["id"]
    client.post("/confirm", json={"id": pid, "ok": False})
    wait_until(lambda: client.get("/state").get_json()["phase"] in ("cancelled", "follow"))
    client.post("/command", json={"text": "what can I do with MoveIt"})
    wait_until(lambda: client.get("/state").get_json()["lit"], timeout=5)
    assert client.get("/state").get_json()["said"]
    assert client.post("/settings/voice", json={"verbosity": 2}).status_code == 200
    assert client.post("/settings/voice", json={"verbosity": 9}).status_code == 400


def test_heard_treats_spoken_words_as_the_console_does():
    m = mock_console.Mock(think_s=0.2)
    assert m.heard("truffles, go home") == "sent"           # the wake word is stripped, the command runs
    wait_until(lambda: m.pending)
    assert m.messages[0]["via"] == "voice" and m.phase == "proposing"
    assert m.heard("go ahead") == "confirmed"
    wait_until(lambda: m.phase in ("done", "follow"), timeout=10)
    assert m.records[-1]["via"] == "voice"


def test_heard_stop_no_and_a_held_correction():
    m = mock_console.Mock(think_s=0.2)
    assert m.heard("stop") == "stopped" and m.phase == "stopped"
    m.heard("up five")
    wait_until(lambda: m.pending)
    assert m.heard("no, down five") == "held"
    wait_until(lambda: m.pending and m.pending["facts"]["label"] == "Down 5 cm", timeout=10)
    assert m.heard("confirm") == "confirmed"
    wait_until(lambda: m.phase in ("done", "follow"), timeout=10)


def test_listening_lamp_goes_back_to_idle_when_nothing_was_said():
    m = mock_console.Mock(think_s=0.2)
    m.voice_state("armed")
    assert m.phase == "listening"
    m.voice_idle()
    assert m.phase == "idle"


def test_mock_voice_routes_without_a_voice_unit(client):
    assert client.post("/voice/ptt", json={"down": True}).status_code == 200
    assert client.get("/voice/devices").get_json() == {"input": [], "output": [], "voices": []}
    assert client.post("/voice/say", json={"text": "hi"}).get_json()["spoken"] is False
    r = client.post("/settings/voice", json={"silence_ms": 900})
    assert r.status_code == 200 and r.get_json()["voice"]["silence_ms"] == 900
    assert client.post("/settings/voice", json={"silence_ms": 5}).status_code == 400
    assert client.get("/state").get_json()["voice"]["silence_ms"] == 900
    assert client.post("/voice/url", json={"url": "http://x"}).status_code == 404


def test_mock_with_a_voice_unit_hears_a_clip(tmp_path):
    pytest.importorskip("soundfile")
    from voice import Voice
    from tests.test_voice_pipeline import CountSTT, LevelVAD, ListSource, LoudTrigger, OnRelease, QuietFor, frames_of
    app = mock_console.create_app(think_s=0.2)
    mock = app.config["mock"]
    unit = Voice(mock.voice, on_state=mock.voice_state, on_final=mock.heard, on_timings=mock.voice_timings, on_idle=mock.voice_idle,
                 make_stages=lambda: (LoudTrigger(), LevelVAD(), QuietFor(), OnRelease(), CountSTT()),
                 make_source=lambda s: ListSource(frames_of(("click", 1), ("speech", 31), ("silence", 31))),
                 make_speaker=lambda v, on_speaking, out: (type("S", (), {"say": lambda self, t: True, "cancel": lambda self: None, "reason": ""})(),
                                                            type("N", (), {"play": lambda self, n: True})()))
    mock.attach_voice(unit)
    unit.start(); unit.wait(5)
    assert mock.messages and mock.messages[0]["via"] == "voice" and mock.messages[0]["text"].endswith("samples")
    assert app.test_client().get("/state").get_json()["voice"]["final"].endswith("samples")


def test_mock_understands_loose_phrasings():
    m = mock_console.Mock()
    assert m.understand("raise the tool by three centimetres") == {"kind": "by", "dir": "up", "cm": 3}
    assert m.understand("move the z axis up five") == {"kind": "by", "dir": "up", "cm": 5}
    assert m.understand("lower it a bit")["dir"] == "down"
    assert m.understand("go to x forty, y minus ten, z thirty") == {"kind": "to", "x": 40, "y": -10, "z": 30}
    assert m.understand("move to x twenty y minus sixty") == {"kind": "to", "x": 20, "y": -60, "z": None}
    assert m.understand("back to home")["kind"] == "home"


def test_mock_asks_when_a_coordinate_is_missing_and_carries_on(client):
    client.post("/command", json={"text": "go to x twenty, y minus sixty"})
    wait_until(lambda: client.get("/state").get_json()["phase"] == "follow")
    s = client.get("/state").get_json()
    assert s["said"].endswith("?") and "z" in s["said"] and "78.5" in s["said"]
    client.post("/command", json={"text": "keep it"})
    wait_until(lambda: client.get("/state").get_json()["pending"])
    assert client.get("/state").get_json()["pending"]["facts"]["to_cm"] == [20, -60, 78.5]


def test_the_mock_has_the_engineer_view(client):
    s = client.get("/state").get_json()
    assert set(s["lichtblick"]) == {"url", "up"} and s["lichtblick"]["url"].startswith("http")
    assert client.post("/lichtblick", json={"url": ""}).get_json()["ok"] is True
