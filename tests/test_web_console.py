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
    web_console.SESSIONS_DIR = tmp_path
    monkeypatch.setattr(web_console, "SAVED_FILE", tmp_path / "saved.json")   # never touch the real saved moves
    monkeypatch.setattr(web_console, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(web_console.config, "MODEL", web_console.config.MODEL)
    monkeypatch.setattr(web_console.config, "THINK", web_console.config.THINK)
    web_console.console = web_console.Console(lambda: agent)
    monkeypatch.setattr(web_console, "status", lambda: {"ollama": True, "moveit": False, "joint_states": False, "joints": {}, "tool_cm": None})
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
        ok = robot_tools._confirm_handler("move the tool up 5 cm")
        outcome["confirmed"] = ok
        return "moved" if ok else "cancelled"
    client.agent.action = move
    client.post("/command", json={"text": "move up 5 cm"})
    wait_until(lambda: client.get("/state").get_json()["pending"] is not None)
    pending = client.get("/state").get_json()["pending"]
    assert "up 5 cm" in pending["text"]
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


def test_confirm_card_carries_the_preview(client):
    def move():
        preview = {"joints": ["a", "b"], "path_deg": [[0, 0], [10, -5]], "seconds": 4}
        ok = robot_tools._confirm_handler("move up 5 cm", preview)
        return "moved" if ok else "cancelled"
    client.agent.action = move
    client.post("/command", json={"text": "move up 5 cm"})
    wait_until(lambda: client.get("/state").get_json()["pending"] is not None)
    pending = client.get("/state").get_json()["pending"]
    assert pending["preview"]["path_deg"][-1] == [10, -5]
    assert pending["preview"]["seconds"] == 4
    client.post("/confirm", json={"id": pending["id"], "ok": False})
    wait_until(lambda: not web_console.console.busy)


def test_state_lists_the_joints_and_mesh_route_404s_for_unknown_package(client):
    joints = client.get("/state").get_json()["joint_list"]
    assert joints[2] == ["elbow_joint", "elbow"]
    assert client.get("/pkg/no_such_package/meshes/x.dae").status_code == 404
    assert client.get("/pkg/ur_description/../../etc/passwd").status_code in (404, 400)


def test_finish_writes_a_session_table(client, tmp_path):
    client.post("/command", json={"text": "hello"})
    wait_until(lambda: not web_console.console.busy)
    body = client.post("/finish").get_json()
    text = open(body["path"]).read()
    assert text.startswith("# Session")
    assert "| hello |" in text
    assert len(body["records"]) == 1


def test_switching_model_needs_a_tool_capable_model(client, monkeypatch):
    monkeypatch.setattr(web_console, "ollama_models",
                        lambda: [{"name": "llama3.1:8b", "tools": True}, {"name": "gemma2:2b", "tools": False}])
    monkeypatch.setattr(web_console.config, "MODEL", "llama3.1:8b")
    assert client.post("/model", json={"name": "gemma2:2b"}).status_code == 400
    assert client.post("/model", json={"name": "llama3.1:8b"}).status_code == 200
    assert client.get("/models").get_json()["current"] == "llama3.1:8b"
    assert "Now using llama3.1:8b" in client.get("/state").get_json()["messages"][-1]["text"]


def test_stop_thinking_frees_the_console_and_ignores_the_late_answer(client):
    started, release = threading.Event(), threading.Event()
    def slow():
        started.set(); release.wait(2); return "late answer"
    client.agent.action = slow
    client.post("/command", json={"text": "think hard"})
    started.wait(1)
    assert client.post("/cancel").status_code == 200
    assert client.get("/state").get_json()["busy"] is False
    release.set(); time.sleep(0.2)
    texts = [m["text"] for m in client.get("/state").get_json()["messages"]]
    assert "late answer" not in texts and any(t.startswith("Stopped after") for t in texts)
    assert client.post("/cancel").status_code == 409          # nothing left to stop


def test_think_switch_and_busy_timer(client):
    assert client.post("/think", json={"on": True}).get_json()["think"] is True
    assert client.get("/state").get_json()["think"] is True
    client.post("/think", json={"on": False})


def test_saved_moves_persist_and_rerun_skips_the_model(client, monkeypatch, tmp_path):
    client.post("/save", json={"name": "square", "tool": "move_path", "args": {"points": [[0, 5, 0]]}, "start_cm": [1, 2, 3]})
    assert client.get("/state").get_json()["saved"][0]["name"] == "square"
    assert "square" in (tmp_path / "saved.json").read_text()
    called = {}
    monkeypatch.setattr(web_console.Console, "_call_tool", lambda self, tool, args: called.update(tool=tool, args=args) or "ran")
    assert client.post("/rerun", json={"name": "square", "tool": "move_path", "args": {"points": [[0, 5, 0]]}}).status_code == 200
    wait_until(lambda: not web_console.console.busy)
    assert called == {"tool": "move_path", "args": {"points": [[0, 5, 0]]}}
    assert client.get("/state").get_json()["messages"][-2]["text"] == "Run again: square"
    assert client.post("/rerun", json={"tool": "stop"}).status_code == 400
    client.post("/unsave", json={"index": 0})
    assert client.get("/state").get_json()["saved"] == []


def test_new_conversation_forgets_the_agent(client):
    client.post("/command", json={"text": "hello"})
    wait_until(lambda: not web_console.console.busy)
    assert web_console.console._agent is not None
    assert client.post("/reset").status_code == 200
    assert web_console.console._agent is None
    assert "New conversation" in client.get("/state").get_json()["messages"][-1]["text"]


def test_an_empty_model_answer_is_explained(client):
    client.agent.action = lambda: ""
    client.post("/command", json={"text": "draw a circle"})
    wait_until(lambda: not web_console.console.busy)
    assert "gave no answer" in client.get("/state").get_json()["messages"][-1]["text"]


def test_a_stopped_command_does_not_block_later_ones(client, monkeypatch):
    started, release = threading.Event(), threading.Event()
    def slow():
        started.set(); release.wait(2); return "late"
    client.agent.action = slow
    client.post("/command", json={"text": "think"})
    started.wait(1)
    client.post("/cancel")
    release.set(); time.sleep(0.2)
    # many later re-runs, on fresh threads that may reuse the old id number, must still run
    monkeypatch.setattr(web_console.Console, "_call_tool",
                        lambda self, tool, args: robot_tools._reports_failures(lambda **kw: "ran")(**args))
    for _ in range(20):
        client.post("/rerun", json={"tool": "move_by", "args": {"dx": 0, "dy": 0, "dz": 1}})
        wait_until(lambda: not web_console.console.busy)
        assert client.get("/state").get_json()["messages"][-1]["text"] == "ran"


def test_the_model_remembers_exactly_what_was_done(client, monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage
    client.agent.chat_history = []
    def spring():
        robot_tools.events.append({"tool": "move_formula", "args": {"formula": "[[0, 0, t] for t in [1, 2]]"},
                                   "outcome": "done", "start_cm": [1, 2, 3], "message": "Done."})
        client.agent.chat_history += [HumanMessage(content="make a spring"), AIMessage(content="Done.")]
        return "Done."
    client.agent.action = spring
    client.post("/command", json={"text": "make a spring"})
    wait_until(lambda: not web_console.console.busy)
    memory = client.agent.chat_history[-1].content
    assert "move_formula" in memory and "[[0, 0, t] for t in [1, 2]]" in memory and "[1, 2, 3]" in memory
    # a re-run from the buttons is recorded too, as something the operator did
    monkeypatch.setattr(web_console.Console, "_call_tool", lambda self, tool, args: robot_tools.events.append(
        {"tool": tool, "args": args, "outcome": "done", "message": "Done."}) or "ran")
    client.post("/rerun", json={"tool": "move_by", "args": {"dx": 0, "dy": 0, "dz": 2}})
    wait_until(lambda: not web_console.console.busy)
    assert "operator ran a move directly" in client.agent.chat_history[-2].content
    assert "move_by" in client.agent.chat_history[-1].content


def test_limits_change_within_their_ceilings_and_are_announced(client, monkeypatch):
    for name in web_console.config.LIMIT_CEILINGS:
        monkeypatch.setattr(web_console.config, name, getattr(web_console.config, name))
    assert client.post("/limits", json={"MAX_PATH_REACH_CM": 50}).status_code == 200
    assert client.get("/state").get_json()["limits"]["MAX_PATH_REACH_CM"] == 50
    assert "path reach (cm) 30 → 50" in client.get("/state").get_json()["messages"][-1]["text"]
    assert client.post("/limits", json={"MAX_PATH_REACH_CM": 500}).status_code == 400      # past the ceiling
    assert client.post("/limits", json={"SPEED": 1.0}).status_code == 400                  # not changeable here
    assert client.post("/limits", json={"MAX_STEP_CM": 0}).status_code == 400
