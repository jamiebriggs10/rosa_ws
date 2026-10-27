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
    assert client.get("/").status_code == 200 and client.get("/stage").status_code == 200
    assert client.get("/static/robot.urdf").status_code == 200
    if mock_console.MESHES.exists():                      # the meshes live in mockups/stage, outside rosa_ws
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
    assert client.post("/settings/voice", json={"voice_verbosity": 2}).status_code == 200
    assert client.post("/settings/voice", json={"voice_verbosity": 9}).status_code == 400
