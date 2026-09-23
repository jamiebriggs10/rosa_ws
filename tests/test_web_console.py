"""Tests for the web console with a fake agent, so no model or robot is needed.

Run on the ZenBook:  ./venv/bin/python -m pytest tests
"""
import threading
import time

import pytest

import robot_tools
import web_console


class FakeAgent:
    """Stands in for ROSA: calls the tool the test asks for, or just echoes."""

    def __init__(self):
        self.action = None

    def invoke(self, text):
        if self.action:
            return self.action()
        return f"echo: {text}"


@pytest.fixture
def client(monkeypatch, tmp_path):
    agent = FakeAgent()
    web_console.console = web_console.Console(lambda: agent)
    web_console.SESSIONS_DIR = tmp_path
    monkeypatch.setattr(web_console, "status", lambda: {"ollama": True, "ursim": False, "driver": False, "program": False})
    web_console.app.config["TESTING"] = True
    c = web_console.app.test_client()
    c.agent = agent
    yield c


def wait_until(predicate, timeout=5):
    deadline = time.time() + timeout
    while not predicate():
        assert time.time() < deadline, "timed out"
        time.sleep(0.02)


def test_state_has_the_shape_the_page_needs(client):
    body = client.get("/state").get_json()
    assert set(body) >= {"status", "busy", "pending", "messages", "model", "pendant"}
    assert body["status"]["ollama"] is True


def test_command_runs_and_reply_appears(client):
    assert client.post("/command", json={"text": "hello"}).status_code == 200
    wait_until(lambda: not web_console.console.busy)
    messages = client.get("/state").get_json()["messages"]
    assert [m["who"] for m in messages] == ["you", "robot"]
    assert messages[1]["text"] == "echo: hello"


def test_second_command_while_busy_is_rejected(client):
    started = threading.Event()
    def slow():
        started.set(); time.sleep(0.5); return "slow done"
    client.agent.action = slow
    client.post("/command", json={"text": "one"})
    started.wait(1)
    assert client.post("/command", json={"text": "two"}).status_code == 409


def test_confirm_card_gates_the_motion(client):
    outcome = {}
    def move():
        ok = robot_tools._confirm_handler("move joint 1 by +10 deg")
        outcome["confirmed"] = ok
        return "moved" if ok else "cancelled"
    client.agent.action = move
    client.post("/command", json={"text": "rotate joint 1 by 10"})
    wait_until(lambda: client.get("/state").get_json()["pending"] is not None)
    pending = client.get("/state").get_json()["pending"]
    assert "move joint 1" in pending["text"]
    assert client.post("/confirm", json={"id": pending["id"], "ok": True}).status_code == 200
    wait_until(lambda: not web_console.console.busy)
    assert outcome["confirmed"] is True
    assert client.get("/state").get_json()["messages"][-1]["text"] == "moved"


def test_cancel_and_stale_confirmation(client):
    def move():
        return "moved" if robot_tools._confirm_handler("go home") else "cancelled"
    client.agent.action = move
    client.post("/command", json={"text": "go home"})
    wait_until(lambda: client.get("/state").get_json()["pending"] is not None)
    pending = client.get("/state").get_json()["pending"]
    assert client.post("/confirm", json={"id": pending["id"], "ok": False}).status_code == 200
    wait_until(lambda: not web_console.console.busy)
    assert client.get("/state").get_json()["messages"][-1]["text"] == "cancelled"
    assert client.post("/confirm", json={"id": pending["id"], "ok": True}).status_code == 409


def test_finish_writes_a_session_table(client, tmp_path):
    client.post("/command", json={"text": "hello"})
    wait_until(lambda: not web_console.console.busy)
    body = client.post("/finish").get_json()
    text = open(body["path"]).read()
    assert text.startswith("# Session")
    assert "| hello |" in text
    assert len(body["records"]) == 1
