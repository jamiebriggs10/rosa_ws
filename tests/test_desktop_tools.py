"""Tests for the tool buttons' registry. Nothing real is started: shutil.which says every
command is installed, subprocess.Popen is a fake that only remembers how it was called,
and has_display is set by each test."""
import signal
import subprocess

import pytest

import desktop_tools
import robot_tools


class FakePopen:
    """Stands in for a started program: remembers its arguments and stays "running"."""
    calls = []

    def __init__(self, argv, **kwargs):
        self.argv, self.kwargs, self.pid = argv, kwargs, 4242
        self.signals, self.waits = [], 0
        FakePopen.calls.append(self)

    def poll(self):
        return None                       # still running

    def send_signal(self, sig):
        self.signals.append(("signal", sig))

    def terminate(self):
        self.signals.append(("terminate",))

    def wait(self, timeout=None):
        self.waits += 1
        if self.waits == 1:
            raise subprocess.TimeoutExpired(self.argv, timeout)   # ignored SIGINT the first time


@pytest.fixture
def tools(monkeypatch):
    FakePopen.calls = []
    monkeypatch.setattr(desktop_tools, "_running", {})
    monkeypatch.setattr(desktop_tools, "events", [])
    monkeypatch.setattr(desktop_tools.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")
    monkeypatch.setattr(desktop_tools.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(desktop_tools, "has_display", lambda: True)
    monkeypatch.setattr(desktop_tools.config, "RVIZ_CONFIG", "")
    return desktop_tools


def test_rows_list_every_registered_tool(tools):
    rows = tools.rows()
    assert [r["name"] for r in rows] == ["rviz", "graph", "logs", "rqt", "terminal", "record"]
    assert all(set(r) == {"name", "label", "available", "running"} for r in rows)
    assert all(r["available"] and not r["running"] for r in rows)


def test_tools_need_a_display_except_record(tools, monkeypatch):
    monkeypatch.setattr(tools, "has_display", lambda: False)
    assert [r["name"] for r in tools.rows() if r["available"]] == ["record"]
    with pytest.raises(RuntimeError, match="display"):
        tools.start("rviz")
    monkeypatch.setattr(tools.shutil, "which", lambda cmd: None)      # nothing installed at all
    with pytest.raises(RuntimeError, match="not installed"):
        tools.start("record")


def test_start_uses_popen_with_a_new_session(tools, monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert tools.start("graph") == "started"
    proc = FakePopen.calls[0]
    assert proc.argv[0] == "rqt_graph"
    assert proc.kwargs["start_new_session"] is True
    assert proc.kwargs["env"]["DISPLAY"] == ":0"                      # filled in when nothing names a screen
    assert tools.events[-1] == {"tool": "graph", "outcome": "started"}
    assert robot_tools.events[-1]["tool"] == "tool" and robot_tools.events[-1]["outcome"] == "started"
    assert [r["running"] for r in tools.rows() if r["name"] == "graph"] == [True]


def test_starting_a_running_tool_reports_running(tools):
    assert tools.start("record") == "started"
    assert tools.start("record") == "running"
    assert len(FakePopen.calls) == 1                                    # one recording, not two


def test_stop_sends_sigint_then_sigterm(tools):
    tools.start("record")
    proc = FakePopen.calls[0]
    assert tools.stop("record") is True
    assert proc.signals == [("signal", signal.SIGINT), ("terminate",)]
    assert tools.events[-1] == {"tool": "record", "outcome": "stopped"}
    assert tools.stop("record") is False                                # nothing left to stop


def test_unknown_tool_raises_keyerror(tools):
    with pytest.raises(KeyError):
        tools.start("nope")
    with pytest.raises(KeyError):
        tools.stop("nope")
    with pytest.raises(KeyError):
        tools.available("nope")


def test_record_command_names_the_sessions_dir(tools):
    argv = tools.command("record")
    assert argv[:4] == ["ros2", "bag", "record", "-a"]
    assert argv[argv.index("-x") + 1] == "(.*)(image|points|cloud|compressed)(.*)"
    folder = argv[argv.index("-o") + 1]
    assert folder.startswith(str(tools.SESSIONS_DIR / "bag-20"))      # bag-<YYYY-MM-DD-HHMM>
    assert tools.SESSIONS_DIR.name == "sessions"


def test_rviz_takes_a_settings_file(tools, monkeypatch):
    assert tools.command("rviz") == ["rviz2"]
    monkeypatch.setattr(tools.config, "RVIZ_CONFIG", "/home/lab/cell.rviz")
    assert tools.command("rviz") == ["rviz2", "-d", "/home/lab/cell.rviz"]
