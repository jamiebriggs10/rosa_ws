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
        self.calls = 0

    def invoke(self, text):
        self.calls += 1
        if self.action:
            return self.action()
        return f"echo: {text}"


@pytest.fixture
def client(monkeypatch, tmp_path):
    agent = FakeAgent()
    web_console.SESSIONS_DIR = tmp_path
    monkeypatch.setattr(web_console, "SAVED_FILE", tmp_path / "saved.json")   # never touch the real saved moves
    monkeypatch.setattr(web_console.config, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(web_console.config, "MODEL", web_console.config.MODEL)
    monkeypatch.setattr(web_console.config, "THINK", web_console.config.THINK)
    web_console.console = web_console.Console(lambda: agent)
    monkeypatch.setattr(web_console, "status", lambda: {"ollama": True, "moveit": False, "joint_states": False, "joints": {}, "tool_cm": None, "hint": ""})
    web_console.app.config["TESTING"] = True
    c = web_console.app.test_client()
    c.agent = agent
    yield c
    # Leave nothing behind: a card still waiting would time out later and speak through the
    # next test's recorder; a tour still running would light the next test's map.
    old = web_console.console
    if old.pending:
        old.answer(old.pending["id"], False)
    old._tour_id += 1
    if old._later:
        old._later.cancel()


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
    memory = client.agent.chat_history[-1]
    assert isinstance(memory, HumanMessage), "the record must not look like the model's own words"
    assert "move_formula" in memory.content and "[[0, 0, t] for t in [1, 2]]" in memory.content and "[1, 2, 3]" in memory.content
    # a re-run from the buttons is recorded too, as something the operator did
    monkeypatch.setattr(web_console.Console, "_call_tool", lambda self, tool, args: robot_tools.events.append(
        {"tool": tool, "args": args, "outcome": "done", "message": "Done."}) or "ran")
    client.post("/rerun", json={"tool": "move_by", "args": {"dx": 0, "dy": 0, "dz": 2}})
    wait_until(lambda: not web_console.console.busy)
    last = client.agent.chat_history[-1].content
    assert "operator ran a move directly" in last and "move_by" in last


def test_limits_change_within_their_ceilings_and_are_announced(client, monkeypatch):
    for name in web_console.config.LIMIT_CEILINGS:
        monkeypatch.setattr(web_console.config, name, getattr(web_console.config, name))
    assert client.post("/limits", json={"MAX_PATH_REACH_CM": 50}).status_code == 200
    assert client.get("/state").get_json()["limits"]["MAX_PATH_REACH_CM"] == 50
    assert "path reach (cm) 30 → 50" in client.get("/state").get_json()["messages"][-1]["text"]
    assert client.post("/limits", json={"MAX_PATH_REACH_CM": 500}).status_code == 400      # past the ceiling
    assert client.post("/limits", json={"SPEED": 1.0}).status_code == 400                  # not changeable here
    assert client.post("/limits", json={"MAX_STEP_CM": 0}).status_code == 400


# -- settings and capabilities -------------------------------------------------

def test_unreachable_ollama_is_reported_and_not_saved(client, monkeypatch):
    monkeypatch.setattr(web_console.config, "OLLAMA_URL", "http://localhost:11434")
    r = client.post("/ollama", json={"url": "http://127.0.0.1:9"})     # a closed port
    assert r.status_code == 502 and "Cannot reach Ollama" in r.get_json()["error"]
    assert web_console.config.OLLAMA_URL == "http://localhost:11434"
    r = client.post("/ollama", json={"url": "nonsense"})
    assert r.status_code == 400


def test_reachable_ollama_is_saved(client, monkeypatch):
    monkeypatch.setattr(web_console, "test_ollama", lambda url: ("http://10.0.0.5:11434", ["llama3.1:8b"]))
    r = client.post("/ollama", json={"url": "http://10.0.0.5:11434/"})
    assert r.status_code == 200 and r.get_json()["models"] == ["llama3.1:8b"]
    assert web_console.config.OLLAMA_URL == "http://10.0.0.5:11434"
    assert client.get("/state").get_json()["ollama_url"] == "http://10.0.0.5:11434"


def test_capabilities_have_the_shape_the_panel_needs(client, monkeypatch):
    rows = [{"name": "/multiview_scan/capture", "kind": "service", "type": "std_srvs/srv/Trigger",
             "means": "Take one view", "moves_robot": False, "status": "ready"}]
    monkeypatch.setattr(web_console.generic_tools, "capability_rows", lambda: rows)
    body = client.get("/capabilities").get_json()
    first = body["interfaces"][0]
    assert {k: first[k] for k in rows[0]} == rows[0] and first["node"] == "multiview_scan" and first["say"] == "take one view"
    assert body["moveit"] is False and "file" in body
    assert [r["node"] for r in body["interfaces"][1:]] == ["moveit"] * len(web_console.robot_tools.MOVEIT_SKILLS)


def test_capabilities_report_a_graph_problem(client, monkeypatch):
    def boom():
        raise RuntimeError("no ROS")
    monkeypatch.setattr(web_console.generic_tools, "capability_rows", boom)
    assert client.get("/capabilities").status_code == 502


def test_state_carries_the_last_path(client, monkeypatch):
    monkeypatch.setattr(web_console.generic_tools, "last_path", {"topic": "/p", "points_cm": [[0, 0, 0], [1, 2, 3]]})
    assert client.get("/state").get_json()["shown_path"]["topic"] == "/p"
    client.post("/reset")
    assert client.get("/state").get_json()["shown_path"] is None


def test_state_carries_tools_and_routes_refuse_unknown(client, monkeypatch):
    tools = client.get("/state").get_json()["tools"]
    assert [t["name"] for t in tools][:2] == ["rviz", "graph"]
    assert set(tools[0]) == {"name", "label", "available", "running"}
    r = client.post("/tools/nope")
    assert r.status_code == 404 and "nope" in r.get_json()["error"]
    monkeypatch.setattr(web_console.desktop_tools, "available", lambda name: False)
    r = client.post("/tools/rviz")
    assert r.status_code == 409 and r.get_json()["error"]
    assert client.post("/tools/nope/stop").status_code == 404
    assert client.post("/tools/rviz/stop").get_json() == {"ok": True, "stopped": False}   # it was never running


# -- the robot connection panel -------------------------------------------------

def test_state_carries_the_robot_ladder(client, monkeypatch):
    ladder = {"mode": "sim", "overall": {"lamp": "wait", "text": "Not checked yet"}, "alert": {}, "rungs": []}
    monkeypatch.setattr(web_console.robot_link, "latest", ladder)
    body = client.get("/state").get_json()
    assert body["robot"] == ladder and body["local"] is True            # the test client is 127.0.0.1
    body = client.get("/state", environ_base={"REMOTE_ADDR": "10.0.0.5"}).get_json()
    assert body["local"] is False


def test_robot_actions_are_refused_remotely_in_real_mode(client, monkeypatch):
    ran = []
    monkeypatch.setattr(web_console.robot_link, "run_action", lambda name, mode: ran.append((name, mode)) or "Driver starting.")
    monkeypatch.setattr(web_console.config, "ROBOT_MODE", "real")
    r = client.post("/robot/action", json={"name": "driver_start"}, environ_base={"REMOTE_ADDR": "10.0.0.5"})
    assert r.status_code == 403 and ran == []
    assert r.get_json()["error"] == "The real robot is only started from the robot's computer's own screen."
    r = client.post("/robot/action", json={"name": "driver_start"}, environ_base={"REMOTE_ADDR": "127.0.0.1"})
    assert r.status_code == 200 and ran == [("driver_start", "real")]
    assert r.get_json()["ok"] is True and r.get_json()["message"] == "Driver starting." and "robot" in r.get_json()
    monkeypatch.setattr(web_console.config, "ROBOT_MODE", "sim")        # the simulator may be started from anywhere
    r = client.post("/robot/action", json={"name": "driver_start"}, environ_base={"REMOTE_ADDR": "10.0.0.5"})
    assert r.status_code == 200 and ran[-1] == ("driver_start", "sim")
    assert client.post("/robot/action", json={"name": "reboot"}).status_code == 404
    def failing(name, mode):
        raise RuntimeError("URSim is not running")
    monkeypatch.setattr(web_console.robot_link, "run_action", failing)
    r = client.post("/robot/action", json={"name": "power_on"})
    assert r.status_code == 409 and "URSim" in r.get_json()["error"]


def test_robot_mode_is_saved(client, monkeypatch):
    monkeypatch.setattr(web_console.config, "ROBOT_MODE", "sim")
    refreshed = []
    monkeypatch.setattr(web_console.robot_link, "refresh", lambda: refreshed.append(1))
    monkeypatch.setattr(web_console.robot_link, "latest", {"mode": "real", "rungs": []})
    r = client.post("/robot/mode", json={"mode": "real"})
    assert r.status_code == 200 and r.get_json()["mode"] == "real" and refreshed == [1]
    assert web_console.config.ROBOT_MODE == "real"
    assert web_console.config.load_settings()["robot_mode"] == "real"
    assert client.post("/robot/mode", json={"mode": "moon"}).status_code == 400


# -- voice: what the console does with spoken text ---------------------------------

class FakeStop:
    """Stands in for the stop tool, which needs a robot."""

    def __init__(self):
        self.calls = 0

    def invoke(self, args):
        self.calls += 1
        return "Stop sent to the arm."


@pytest.fixture
def voice_on(monkeypatch):
    monkeypatch.setattr(web_console.config, "VOICE_ON", True)
    monkeypatch.setattr(web_console.config, "VOICE_CONFIRM", False)
    stop = FakeStop()
    monkeypatch.setattr(web_console.robot_tools, "stop", stop)
    return stop


def test_heard_submits_with_via_voice(client, voice_on):
    assert web_console.console.heard("move up 5 cm") == "sent"
    wait_until(lambda: not web_console.console.busy)
    messages = client.get("/state").get_json()["messages"]
    assert messages[0] == {"who": "you", "text": "move up 5 cm", "via": "voice"}
    assert messages[1]["text"] == "echo: move up 5 cm"                 # the model got it like typed text
    assert web_console.console.records[0]["via"] == "voice"


def test_heard_ignores_empty_text(client, voice_on):
    assert web_console.console.heard("  ") == "empty"
    assert web_console.console.messages == []


def test_heard_is_ignored_when_voice_is_off(client, voice_on, monkeypatch):
    monkeypatch.setattr(web_console.config, "VOICE_ON", False)
    assert web_console.console.heard("move up") == "off"
    assert web_console.console.messages == []


def test_heard_stop_halts_without_the_model(client, voice_on):
    invoked = []
    client.agent.action = lambda: invoked.append(1) or "moved"
    assert web_console.console.heard("Stop.") == "stopped"
    assert voice_on.calls == 1 and invoked == []
    assert web_console.console.messages[-1]["text"].startswith("Stopped by voice")


def test_partial_stop_swallows_the_final(client, voice_on):
    c = web_console.console
    assert c.heard("stop", partial=True) == "stopped"
    assert c.heard("stop") == "swallowed"                 # the final of the same utterance
    assert voice_on.calls == 1
    c.voice_idle()                                        # the voice app is back to idle: next utterance
    assert c.heard("stop") == "stopped" and voice_on.calls == 2


def test_partial_that_is_not_a_stop_does_nothing(client, voice_on):
    assert web_console.console.heard("move", partial=True) == "ignored"
    assert web_console.console.messages == []


def test_heard_while_busy_adds_a_note_and_does_not_queue(client, voice_on):
    started, release = threading.Event(), threading.Event()

    def slow():
        started.set()
        release.wait(2)
        return "slow done"
    client.agent.action = slow
    client.post("/command", json={"text": "one"})
    started.wait(1)
    assert web_console.console.heard("again") == "busy"
    assert web_console.console.messages[-1]["text"].startswith("Heard “again” but the model is still busy")
    release.set()
    wait_until(lambda: not web_console.console.busy)
    assert [m["text"] for m in web_console.console.messages if m["who"] == "you"] == ["one"]


def test_voice_confirm_answers_the_card_only_when_enabled(client, voice_on, monkeypatch):
    outcome = {}

    def move():
        outcome["confirmed"] = robot_tools._confirm_handler("move the tool up 5 cm")
        return "moved" if outcome["confirmed"] else "cancelled"
    client.agent.action = move
    c = web_console.console
    client.post("/command", json={"text": "move up 5 cm"})
    wait_until(lambda: c.pending is not None)
    assert c.heard("yes") == "busy"                       # voice confirm is off: the card waits for a click
    assert "confirm card" in c.messages[-1]["text"]
    monkeypatch.setattr(web_console.config, "VOICE_CONFIRM", True)
    assert c.heard("Yes.") == "held"                      # not a yes-word any more: cancels the card, runs as a command
    wait_until(lambda: outcome.get("confirmed") is False)
    wait_until(lambda: len(c.records) == 1 and c.pending, timeout=10)    # the held "Yes." ran as its own command
    c.answer(c.pending["id"], False)
    wait_until(lambda: len(c.records) == 2 and not c.busy, timeout=10)
    client.post("/command", json={"text": "move up 5 cm"})
    wait_until(lambda: c.pending is not None)
    assert c.heard("Confirm.") == "confirmed"
    wait_until(lambda: not c.busy)
    assert outcome["confirmed"] is True
    client.post("/command", json={"text": "move up 5 cm"})
    wait_until(lambda: c.pending is not None)
    assert c.heard("no") == "cancelled"
    wait_until(lambda: not c.busy)
    assert outcome["confirmed"] is False


def test_session_table_has_a_via_column(client, voice_on):
    client.post("/command", json={"text": "typed one"})
    wait_until(lambda: not web_console.console.busy)
    web_console.console.heard("spoken one")
    wait_until(lambda: not web_console.console.busy)
    text = open(client.post("/finish").get_json()["path"]).read()
    assert "| Via |" in text
    assert "| typed one | typed |" in text and "| spoken one | voice |" in text


# -- voice: the routes the pop-up uses -----------------------------------------------

def test_state_carries_voice(client, monkeypatch):
    monkeypatch.setattr(web_console.config, "VOICE_URL", "http://127.0.0.1:8100")
    body = client.get("/state").get_json()
    assert set(body["voice"]) >= {"running", "state", "partial", "final", "on", "confirm", "url"}
    assert body["voice"]["url"] == "http://127.0.0.1:8100" and body["voice"]["running"] is False


def test_voice_on_saves_and_tells_the_app(client, monkeypatch):
    told = []
    monkeypatch.setattr(web_console.voice_link, "set_on", lambda on: told.append(on))
    monkeypatch.setattr(web_console.config, "VOICE_ON", False)
    r = client.post("/voice/on", json={"on": True})
    assert r.status_code == 200 and r.get_json()["on"] is True and r.get_json()["note"] is None
    assert web_console.config.VOICE_ON is True and web_console.config.load_settings()["voice_on"] is True
    assert told == [True]
    assert web_console.console.messages[-1]["text"] == "Voice control is on."
    client.post("/voice/on", json={"on": False})
    assert web_console.console.messages[-1]["text"] == "Voice control is off." and told == [True, False]


def test_voice_on_saves_even_when_the_app_is_unreachable(client, monkeypatch):
    monkeypatch.setattr(web_console.voice_link, "set_on", lambda on: "The voice app is not answering at http://127.0.0.1:8100.")
    monkeypatch.setattr(web_console.config, "VOICE_ON", False)
    r = client.post("/voice/on", json={"on": True})
    assert r.status_code == 200 and "not answering" in r.get_json()["note"]
    assert web_console.config.VOICE_ON is True


def test_voice_confirm_is_saved(client, monkeypatch):
    monkeypatch.setattr(web_console.config, "VOICE_CONFIRM", False)
    assert client.post("/voice/confirm", json={"on": True}).status_code == 200
    assert web_console.config.VOICE_CONFIRM is True and web_console.config.load_settings()["voice_confirm"] is True


def test_bad_voice_url_is_refused(client, monkeypatch):
    monkeypatch.setattr(web_console.config, "VOICE_URL", "http://127.0.0.1:8100")

    def unreachable(method, path, body=None, timeout_s=3.0, base=None):
        raise web_console.voice.VoiceUnavailable(f"The voice app is not answering at {base}.")
    monkeypatch.setattr(web_console.voice_link, "request", unreachable)
    assert client.post("/voice/url", json={"url": "ten dot two"}).status_code == 400
    assert client.post("/voice/url", json={"url": "http"}).status_code == 400          # no host at all
    r = client.post("/voice/url", json={"url": "http://10.0.0.2:8100"})
    assert r.status_code == 502 and "not answering" in r.get_json()["error"]
    assert web_console.config.VOICE_URL == "http://127.0.0.1:8100"


def test_reachable_but_wrong_voice_url_is_refused(client, monkeypatch):
    """Ollama or the console itself answer on their ports, but they are not the voice app."""
    monkeypatch.setattr(web_console.config, "VOICE_URL", "http://127.0.0.1:8100")
    monkeypatch.setattr(web_console.voice_link, "request",
                        lambda method, path, body=None, timeout_s=3.0, base=None: (404, {"error": "no such page"}))
    r = client.post("/voice/url", json={"url": "http://127.0.0.1:11434"})
    assert r.status_code == 400 and "not the voice app" in r.get_json()["error"]
    assert web_console.config.VOICE_URL == "http://127.0.0.1:8100"


def test_cancelled_spoken_command_keeps_its_via(client, voice_on):
    started, release = threading.Event(), threading.Event()

    def slow():
        started.set()
        release.wait(2)
        return "late"
    client.agent.action = slow
    assert web_console.console.heard("move up") == "sent"
    started.wait(1)
    assert client.post("/cancel").status_code == 200
    assert web_console.console.records[-1]["via"] == "voice"
    release.set()


def test_good_voice_url_is_saved_and_reconnects(client, monkeypatch):
    monkeypatch.setattr(web_console.config, "VOICE_URL", "http://127.0.0.1:8100")
    asked, reconnected = [], []
    monkeypatch.setattr(web_console.voice_link, "request",
                        lambda method, path, body=None, timeout_s=3.0, base=None: asked.append((path, base)) or (200, {"state": "idle"}))
    monkeypatch.setattr(web_console.voice_link, "reconnect", lambda: reconnected.append(1))
    r = client.post("/voice/url", json={"url": "http://10.0.0.2:8100/"})
    assert r.status_code == 200 and r.get_json()["url"] == "http://10.0.0.2:8100"
    assert asked == [("/status", "http://10.0.0.2:8100")] and reconnected == [1]
    assert web_console.config.VOICE_URL == "http://10.0.0.2:8100"
    assert web_console.config.load_settings()["voice_url"] == "http://10.0.0.2:8100"


def test_voice_proxy_forwards_allowed_paths_only(client, monkeypatch):
    seen = []
    monkeypatch.setattr(web_console.voice_link, "request",
                        lambda method, path, body=None, timeout_s=3.0, base=None: seen.append((method, path, body)) or (200, {"ok": True}))
    assert client.get("/voice/options").get_json() == {"ok": True}
    assert client.post("/voice/presets/mine?overwrite=1").status_code == 200
    assert client.post("/voice/settings", json={"vad": {"use": "webrtc"}}).status_code == 200
    assert client.get("/voice/help").status_code == 200
    assert seen == [("GET", "/options", None), ("POST", "/presets/mine?overwrite=1", None),
                    ("POST", "/settings", {"vad": {"use": "webrtc"}}), ("GET", "/static/help.json", None)]
    assert client.post("/voice/wake", json={"on": True}).status_code == 404     # the switch is the console's
    assert client.get("/voice/../etc/passwd").status_code == 404
    assert client.get("/voice/presets/mine/delete").status_code == 404


def test_voice_proxy_passes_status_and_body_through(client, monkeypatch):
    monkeypatch.setattr(web_console.voice_link, "request",
                        lambda method, path, body=None, timeout_s=3.0, base=None: (409, {"error": "preset mine exists"}))
    r = client.post("/voice/presets/mine")
    assert r.status_code == 409 and r.get_json() == {"error": "preset mine exists"}

    def unreachable(method, path, body=None, timeout_s=3.0, base=None):
        raise web_console.voice.VoiceUnavailable("The voice app is not answering at http://127.0.0.1:8100.")
    monkeypatch.setattr(web_console.voice_link, "request", unreachable)
    r = client.get("/voice/status")
    assert r.status_code == 502 and "not answering" in r.get_json()["error"]


# ---- the loop: phases with timestamps, cached status, the new voice settings ----

def test_phases_follow_a_command(client):
    client.agent.action = lambda: (web_console.console.ask_operator("move", {"facts": {"kind": "line"}}), "ok")[1]
    client.post("/command", json={"text": "move"})
    wait_until(lambda: web_console.console.pending)
    assert client.get("/state").get_json()["phase"] == "proposing"
    pid = web_console.console.pending["id"]
    client.post("/confirm", json={"id": pid, "ok": True})
    wait_until(lambda: not web_console.console.busy)
    names = [p[0] for p in web_console.console.records[-1]["phases"]]
    assert names[:3] == ["thinking", "proposing", "moving"] and names[-1] in ("done", "idle")
    assert all(isinstance(p[1], float) for p in web_console.console.records[-1]["phases"])


def test_status_is_cached_for_a_second(client, monkeypatch):
    calls = {"n": 0}

    def counting():
        calls["n"] += 1
        return {"ollama": True, "moveit": False, "joint_states": False, "joints": {}, "tool_cm": None, "hint": ""}
    monkeypatch.setattr(web_console, "status", counting)
    web_console._status_cache.clear()
    for _ in range(5):
        client.get("/state")
    assert calls["n"] == 1


def test_new_settings_are_in_state(client):
    s = client.get("/state").get_json()
    assert s["verbosity"] == 1 and s["persona"] == "A" and s["follow_up_s"] == 8 and s["confirm_timeout_s"] == 20
    assert s["phase"] == "idle" and s["follow_until_s"] == 0 and s["trace"] is None and s["lit"] == [] and s["held"] is None


def test_finish_has_the_timing_columns(client, tmp_path):
    client.post("/command", json={"text": "hello"})
    wait_until(lambda: not web_console.console.busy)
    r = client.post("/finish").get_json()
    text = open(r["path"]).read()
    assert "| Felt | Think | To confirm | Motion |" in text


# ---- strict confirm, held corrections, the card's timeout ----

def card_agent(client, label="Point 10"):
    """An agent whose every call raises one confirm card with facts; records what the console sent it."""
    seen = []

    def invoke(text):
        seen.append(text)
        ok = web_console.console.ask_operator(f"move the tool in a straight line to {label}",
                                              {"facts": {"label": label, "kind": "line"}})
        return "moved" if ok else "cancelled"
    client.agent.invoke = invoke
    return seen


@pytest.fixture
def confirm_on(voice_on, monkeypatch):
    monkeypatch.setattr(web_console.config, "VOICE_CONFIRM", True)
    return voice_on


def test_no_without_a_card_is_a_command(client, confirm_on):
    assert web_console.console.heard("no") == "sent"
    wait_until(lambda: not web_console.console.busy)
    assert web_console.console.records[-1]["command"] == "no"


def test_no_plus_rest_cancels_and_runs_the_rest_with_context(client, confirm_on):
    c = web_console.console
    seen = card_agent(client)
    assert c.heard("go to point ten") == "sent"
    wait_until(lambda: c.pending)
    assert c.heard("no, point eleven") == "held"
    assert c.held == "point eleven"
    wait_until(lambda: len(seen) == 2 and c.pending, timeout=10)         # the held text is its own command
    assert seen[1] == '(The operator declined "Point 10" and said instead:) point eleven'
    assert len(c.records) == 1 and c.held is None
    c.answer(c.pending["id"], False)
    wait_until(lambda: len(c.records) == 2 and not c.busy, timeout=10)
    assert c.records[1]["command"] == "point eleven" and c.records[1]["via"] == "voice"


def test_second_card_after_no_is_declined_and_held_text_runs(client, confirm_on):
    c = web_console.console
    asked = []

    def invoke(text):
        asked.append(web_console.console.ask_operator("first card", {"facts": {"label": "Point 10"}}))
        if len(asked) == 1:
            asked.append(web_console.console.ask_operator("second card", {"facts": {"label": "Point 10"}}))
        return "done"
    client.agent.invoke = invoke
    c.heard("go to point ten")
    wait_until(lambda: c.pending)
    t0 = time.time()
    assert c.heard("no, point eleven") == "held"
    wait_until(lambda: len(asked) == 2 and c.pending, timeout=10)         # the held text's own card
    assert asked == [False, False] and time.time() - t0 < 5               # the second card never waited
    c.answer(c.pending["id"], False)
    wait_until(lambda: len(c.records) == 2 and not c.busy, timeout=10)
    assert c.records[1]["command"] == "point eleven"


def test_any_other_utterance_cancels_the_card_and_runs(client, confirm_on):
    c = web_console.console
    seen = card_agent(client)
    c.heard("go to point ten")
    wait_until(lambda: c.pending)
    assert c.heard("what is point ten") == "held"
    wait_until(lambda: len(seen) == 2 and c.pending, timeout=10)
    c.answer(c.pending["id"], False)
    wait_until(lambda: len(c.records) == 2 and not c.busy, timeout=10)
    assert c.records[1]["command"] == "what is point ten"


def test_card_timeout_is_cancelled_and_frees_the_console(client, confirm_on, monkeypatch):
    monkeypatch.setattr(web_console.config, "CONFIRM_TIMEOUT_S", 0.2)
    c = web_console.console
    card_agent(client)
    client.post("/command", json={"text": "go to point ten"})
    wait_until(lambda: c.pending)
    wait_until(lambda: not c.busy, timeout=5)
    assert c.pending is None and [p[0] for p in c.records[-1]["phases"]][:3] == ["thinking", "proposing", "cancelled"]
    assert c.timed_out == "Point 10"
    assert client.post("/command", json={"text": "hello"}).status_code == 200


def test_voice_confirm_default_is_on():
    assert web_console.config.VOICE_CONFIRM is True


def test_one_no_covers_the_run_with_a_click_too(client, voice_on):
    c = web_console.console
    asked = []

    def invoke(text):
        asked.append(c.ask_operator("first", {"facts": {"label": "A"}}))
        asked.append(c.ask_operator("second", {"facts": {"label": "A"}}))
        return "done"
    client.agent.invoke = invoke
    client.post("/command", json={"text": "two cards"})
    wait_until(lambda: c.pending)
    client.post("/confirm", json={"id": c.pending["id"], "ok": False})
    wait_until(lambda: not c.busy, timeout=5)
    assert asked == [False, False]


# ---- say, play, follow-up: every phase has a sound and, by verbosity, a line ----

class RecordingLink:
    """Stands in for voice.VoiceLink: keeps what the console asked the voice app to do."""

    def __init__(self):
        self.say_calls, self.play_calls, self.followup_calls, self.hushes = [], [], [], 0
        self.latest = {"running": True, "speaking": False, "state": "idle"}

    def say(self, text): self.say_calls.append(text)
    def play(self, name): self.play_calls.append(name)
    def followup(self, seconds): self.followup_calls.append(seconds)
    def hush(self): self.hushes += 1
    def set_on(self, on): return None
    def request(self, *a, **k): raise web_console.voice.VoiceUnavailable("fake")
    def reconnect(self): pass


@pytest.fixture
def recorder(monkeypatch):
    link = RecordingLink()
    monkeypatch.setattr(web_console, "voice_link", link)
    monkeypatch.setattr(web_console.config, "VOICE_VERBOSITY", 1)
    monkeypatch.setattr(web_console.config, "VOICE_PERSONA", "A")
    return link


def line_agent(client, facts):
    def invoke(text):
        ok = web_console.console.ask_operator("a card", {"facts": facts, "path_deg": [[0] * 6], "joints": [], "seconds": 6})
        return "moved" if ok else "cancelled"
    client.agent.invoke = invoke


def test_a_proposal_is_played_said_and_opens_a_follow_up(client, recorder):
    c = web_console.console
    line_agent(client, {"kind": "line", "label": "Point 10", "distance_cm": 14, "seconds": 6})
    client.post("/command", json={"text": "go to point ten"})
    wait_until(lambda: c.pending)
    wait_until(lambda: recorder.say_calls)
    assert recorder.play_calls[-1] == "ask"
    assert recorder.say_calls[-1] == "Point 10. 14 centimetres, 6 seconds, path clear. Confirm?"
    assert recorder.followup_calls[-1] == 20 and client.get("/state").get_json()["said"] == recorder.say_calls[-1]
    c.answer(c.pending["id"], False)
    wait_until(lambda: not c.busy)


def test_verbosity_zero_plays_but_never_says(client, recorder, monkeypatch):
    monkeypatch.setattr(web_console.config, "VOICE_VERBOSITY", 0)
    c = web_console.console
    line_agent(client, {"kind": "line", "label": "Point 10", "distance_cm": 14, "seconds": 6})
    client.post("/command", json={"text": "go to point ten"})
    wait_until(lambda: c.pending)
    assert "ask" in recorder.play_calls and recorder.say_calls == []
    c.answer(c.pending["id"], False)
    wait_until(lambda: not c.busy)
    assert recorder.say_calls == [] and "cancel" in recorder.play_calls


def test_done_move_opens_a_follow_up_window_and_state_counts_it_down(client, recorder, monkeypatch):
    c = web_console.console

    def invoke(text):
        ok = web_console.console.ask_operator("a card", {"facts": {"kind": "line", "label": "Point 10", "distance_cm": 14, "seconds": 6}})
        robot_tools.events.append({"tool": "move_to", "args": {}, "outcome": "done", "message": "Done."})
        return "moved" if ok else "cancelled"
    client.agent.invoke = invoke
    client.post("/command", json={"text": "go to point ten"})
    wait_until(lambda: c.pending)
    c.answer(c.pending["id"], True)
    wait_until(lambda: not c.busy)
    s = client.get("/state").get_json()
    assert s["phase"] == "follow" and 6 <= s["follow_until_s"] <= 8
    assert recorder.play_calls[-1] == "done" and recorder.say_calls[-1] == "At Point 10." and recorder.followup_calls[-1] == 8
    client.agent.invoke = lambda text: "echo"             # a plain command closes the window at once
    client.post("/command", json={"text": "hello"})
    assert client.get("/state").get_json()["follow_until_s"] == 0


def test_refusal_is_said_the_same_in_both_personas(client, recorder, monkeypatch):
    c = web_console.console
    said = {}
    for persona in ("A", "B"):
        monkeypatch.setattr(web_console.config, "VOICE_PERSONA", persona)

        def invoke(text):
            robot_tools.events.append({"tool": "move_by", "args": {}, "outcome": "refused", "message": "REFUSED by safety layer: too far. Nothing was moved."})
            return "refused"
        client.agent.invoke = invoke
        client.post("/command", json={"text": "up fifty"})
        wait_until(lambda: not c.busy)
        said[persona] = recorder.say_calls[-1]
    assert said["A"] == said["B"] and said["A"].startswith("REFUSED") and recorder.play_calls[-1] == "refused"


def test_felt_delay_lands_in_the_record(client, voice_on, recorder):
    c = web_console.console
    assert c.heard("move up") == "sent"
    c.voice_timings(412)
    wait_until(lambda: not c.busy)
    assert c.records[-1]["felt_ms"] == 412
    text = open(client.post("/finish").get_json()["path"]).read()
    assert "| 0.41 s |" in text


# ---- the map's data and the self-questions: answered by the console, never by the model ----

CAP_ROWS = [{"name": "/multiview_scan/capture", "kind": "service", "type": "std_srvs/srv/Trigger",
             "means": "Take one 3D view from here.", "moves_robot": False, "status": "ready", "node": "multiview_scan", "say": "take one 3d view from here"},
            {"name": "/touchdown_test/execute_raster", "kind": "service", "type": "std_srvs/srv/Trigger",
             "means": "Scan the raster with the probe down.", "moves_robot": True, "status": "ready", "node": "touchdown_test", "say": "run the raster"},
            {"name": "/joint_states", "kind": "topic", "type": "sensor_msgs/msg/JointState", "means": "", "moves_robot": None, "status": "undescribed"}]


@pytest.fixture
def caps(monkeypatch):
    monkeypatch.setattr(web_console.generic_tools, "capability_rows", lambda: [dict(r) for r in CAP_ROWS])
    return CAP_ROWS


def test_capabilities_include_moveit_rows_with_phrases(client, caps):
    rows = client.get("/capabilities").get_json()["interfaces"]
    home = next(r for r in rows if r["name"] == "go_home")
    assert home["node"] == "moveit" and home["say"] == "go home" and home["moves_robot"] is True and home["kind"] == "tool"
    assert next(r for r in rows if r["name"] == "/joint_states")["node"] == "root"      # old-style rows get a node too


def test_what_can_i_do_is_answered_without_the_model(client, caps, recorder):
    assert client.post("/command", json={"text": "what can I do with multiview_scan"}).status_code == 200
    wait_until(lambda: len(recorder.say_calls) >= 2, timeout=5)
    assert recorder.say_calls[0] == "Multiview scan offers 1 thing." and client.agent.calls == 0
    assert recorder.say_calls[1] == 'Take one 3D view from here: say "take one 3d view from here".'
    wait_until(lambda: client.get("/state").get_json()["lit"] == ["/multiview_scan/capture"], timeout=5)
    wait_until(lambda: client.get("/state").get_json()["lit"] == [], timeout=5)
    assert web_console.console.records[-1]["command"] == "what can I do with multiview_scan"
    assert client.get("/state").get_json()["follow_until_s"] > 0


def test_which_move_the_robot_lists_the_movers(client, caps, recorder):
    web_console.console.heard = web_console.console.heard   # (voice path shares submit)
    client.post("/command", json={"text": "Which of these move the robot?"})
    wait_until(lambda: recorder.say_calls, timeout=5)
    assert recorder.say_calls[0].startswith("These move the robot") and "run the raster" in recorder.say_calls[0]
    assert "go home" in recorder.say_calls[0] and client.agent.calls == 0


def test_show_ros_is_saved_and_in_state(client, caps, recorder, monkeypatch):
    monkeypatch.setattr(web_console.config, "MAP_ROS", False)
    client.post("/command", json={"text": "show ros"})
    wait_until(lambda: client.get("/state").get_json()["map_ros"] is True, timeout=5)
    client.post("/command", json={"text": "hide ROS."})
    wait_until(lambda: client.get("/state").get_json()["map_ros"] is False, timeout=5)
    assert client.agent.calls == 0


def test_a_plain_question_still_goes_to_the_model(client, caps, recorder):
    client.post("/command", json={"text": "what can the camera see"})
    wait_until(lambda: not web_console.console.busy)
    assert client.agent.calls == 1


# ---- the stage page and its settings pop-up ----

def test_stage_page_is_served(client):
    r = client.get("/stage")
    assert r.status_code == 200 and b'id="settings"' in r.data and b"panels.js" in r.data and b"map.js" in r.data


def test_voice_output_settings_route_validates(client, monkeypatch):
    monkeypatch.setattr(web_console.config, "VOICE_VERBOSITY", 1)
    monkeypatch.setattr(web_console.config, "VOICE_PERSONA", "A")
    r = client.post("/settings/voice", json={"voice_verbosity": 2, "voice_persona": "B", "follow_up_s": 5, "confirm_timeout_s": 30, "map_ros": True})
    assert r.status_code == 200 and r.get_json()["verbosity"] == 2 and web_console.config.VOICE_PERSONA == "B"
    assert client.post("/settings/voice", json={"voice_verbosity": 7}).status_code == 400
    assert client.post("/settings/voice", json={"confirm_timeout_s": 1}).status_code == 400
    assert client.post("/settings/voice", json={"ollama_url": "http://x"}).status_code == 400     # not one of these settings
    s = client.get("/state").get_json()
    assert s["verbosity"] == 2 and s["persona"] == "B" and s["follow_up_s"] == 5 and s["confirm_timeout_s"] == 30 and s["map_ros"] is True
