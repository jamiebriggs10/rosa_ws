"""Tests for the robot-connection ladder. Every probe (sysfs, ip, ping, sockets, the dashboard,
tmux, the ROS topics) is replaced by a small fake, so nothing real is touched: the tests only
check that the ladder draws the right conclusion from what the probes say."""
import socket
import threading
import time

import pytest

import robot_link
import robot_tools

# What the dashboard answers when the arm is powered, braked off and running a program.
READY = {"robotmode": "Robotmode: RUNNING", "safetystatus": "Safetystatus: NORMAL",
         "programState": "PLAYING rosa.urp", "get loaded program": "Loaded program: /programs/rosa.urp",
         "is in remote control": "true"}


@pytest.fixture
def good(monkeypatch):
    """Every probe answers as it would with the robot fully ready; each test then breaks one."""
    for name, value in (("ROBOT_MODE", "real"), ("ROBOT_IP", "172.31.1.69"), ("COMPUTER_IP", "172.31.1.137"),
                        ("ROBOT_PORT", "enp3s0"), ("URSIM_IP", "192.168.56.101"), ("NETWORK_PROFILE", "robot")):
        monkeypatch.setattr(robot_link.config, name, value)
    answers = dict(READY)
    monkeypatch.setattr(robot_link, "carrier", lambda port: True)
    monkeypatch.setattr(robot_link, "port_speed", lambda port: 1000)
    monkeypatch.setattr(robot_link, "addresses", lambda port: ["172.31.1.137/24"])
    monkeypatch.setattr(robot_link, "ping", lambda ip: True)
    monkeypatch.setattr(robot_link, "tcp_open", lambda ip, port: True)
    monkeypatch.setattr(robot_link, "dashboard", lambda ip, command: answers[command])
    monkeypatch.setattr(robot_link, "tmux_session", lambda name: True)
    monkeypatch.setattr(robot_link, "driver_status", lambda: {"joint_states": True, "moveit": True})
    monkeypatch.setattr(robot_link, "program_running", lambda: True)
    monkeypatch.setattr(robot_link, "controller_active", lambda name: True)
    monkeypatch.setattr(robot_link, "speed_scaling", lambda: 10.0)     # percent, as the UR driver publishes it
    monkeypatch.setattr(robot_link, "run_command", lambda argv: "")
    monkeypatch.setattr(robot_link, "profile_exists", lambda name: True)
    return answers


def test_the_simulator_does_not_mind_the_speed_slider(good, monkeypatch):
    monkeypatch.setattr(robot_link, "speed_scaling", lambda: 100.0)
    assert robot_link.ladder("sim")["rungs"][4]["lamp"] == "ok"
    assert robot_link.ladder("real")["rungs"][6]["lamp"] == "gate"


def test_a_missing_network_profile_gets_the_one_time_command(good, monkeypatch):
    monkeypatch.setattr(robot_link, "addresses", lambda port: [])
    monkeypatch.setattr(robot_link, "profile_exists", lambda name: False)
    rung = robot_link.ladder("real")["rungs"][1]
    assert rung["lamp"] == "bad" and rung["action"] is None and rung["yours"] is True
    assert "nmcli con add" in rung["detail"] and "172.31.1.137/24" in rung["detail"]
    monkeypatch.setattr(robot_link, "profile_exists", lambda name: True)
    rung = robot_link.ladder("real")["rungs"][1]
    assert rung["action"]["name"] == "profile"


def lamps(ladder):
    return [r["lamp"] for r in ladder["rungs"]]


# -- the real-robot ladder ------------------------------------------------------

def test_unplugged_stops_at_rung_one(good, monkeypatch):
    monkeypatch.setattr(robot_link, "carrier", lambda port: False)
    ladder = robot_link.ladder("real")
    assert [r["name"] for r in ladder["rungs"]] == ["cable", "address", "reachable", "state", "driver", "play", "speed"]
    assert lamps(ladder) == ["bad"] + ["wait"] * 6
    assert ladder["rungs"][0]["yours"] is True and ladder["rungs"][0]["action"] is None
    assert "No cable on enp3s0" in ladder["rungs"][0]["detail"]
    assert ladder["rungs"][1]["detail"] == "Later."
    assert ladder["overall"]["lamp"] == "bad" and ladder["alert"]["text"] == ladder["rungs"][0]["detail"]


def test_plugged_in_with_the_driver_off_reaches_rung_five(good, monkeypatch):
    monkeypatch.setattr(robot_link, "tmux_session", lambda name: False)
    monkeypatch.setattr(robot_link, "driver_status", lambda: {"joint_states": False, "moveit": False})
    ladder = robot_link.ladder("real")
    assert lamps(ladder) == ["ok"] * 4 + ["bad", "wait", "wait"]
    driver = ladder["rungs"][4]
    assert driver["action"]["name"] == "driver_start" and driver["yours"] is False
    assert "joint states no" in driver["detail"] and "MoveIt no" in driver["detail"]
    assert ladder["overall"]["lamp"] == "bad"


def test_all_good_is_ready(good):
    ladder = robot_link.ladder("real")
    assert lamps(ladder) == ["ok"] * 7
    assert ladder["overall"]["lamp"] == "ok" and ladder["alert"]["text"] == "Ready"
    assert ladder["rungs"][4]["action"]["name"] == "driver_stop"       # a running driver can be stopped
    assert "1000 Mb/s" in ladder["rungs"][0]["detail"]
    assert all(set(r) == {"name", "title", "lamp", "detail", "action", "yours"} for r in ladder["rungs"])


def test_the_state_rung_offers_power_on_then_brake_release(good):
    good["robotmode"] = "Robotmode: POWER_OFF"
    rung = robot_link.ladder("real")["rungs"][3]
    assert rung["lamp"] == "gate" and rung["action"]["name"] == "power_on"
    assert "POWER_OFF" in rung["detail"] and "NORMAL" in rung["detail"] and "rosa.urp" in rung["detail"]
    good["robotmode"] = "Robotmode: IDLE"
    assert robot_link.ladder("real")["rungs"][3]["action"]["name"] == "brake_release"
    good["robotmode"], good["safetystatus"] = "Robotmode: RUNNING", "Safetystatus: PROTECTIVE_STOP"
    rung = robot_link.ladder("real")["rungs"][3]
    assert rung["lamp"] == "gate" and rung["action"] is None and rung["yours"] is True


def test_play_and_speed_are_the_operators(good, monkeypatch):
    monkeypatch.setattr(robot_link, "program_running", lambda: False)
    rung = robot_link.ladder("real")["rungs"][5]
    assert rung["lamp"] == "gate" and rung["yours"] is True and "Press Play" in rung["detail"]
    monkeypatch.setattr(robot_link, "program_running", lambda: True)
    monkeypatch.setattr(robot_link, "speed_scaling", lambda: 45.0)
    rung = robot_link.ladder("real")["rungs"][6]
    assert rung["lamp"] == "gate" and "45 %" in rung["detail"] and "10 to 20 %" in rung["detail"]


# -- the simulator ladder --------------------------------------------------------

def test_sim_ladder_with_ursim_down_is_bad_at_rung_one(good, monkeypatch):
    monkeypatch.setattr(robot_link, "tcp_open", lambda ip, port: False)
    ladder = robot_link.ladder("sim")
    assert [r["name"] for r in ladder["rungs"]] == ["ursim", "state", "driver", "play", "speed"]
    assert lamps(ladder) == ["bad", "wait", "wait", "wait", "wait"]
    assert ladder["rungs"][0]["action"]["name"] == "driver_start" and "URSim is not running" in ladder["rungs"][0]["detail"]


def test_sim_ladder_asks_the_simulator_not_the_robot(good, monkeypatch):
    asked = []
    monkeypatch.setattr(robot_link, "dashboard", lambda ip, command: asked.append(ip) or READY[command])
    monkeypatch.setattr(robot_link, "program_running", lambda: False)
    ladder = robot_link.ladder("sim")
    assert set(asked) == {"192.168.56.101"}
    play = ladder["rungs"][3]
    assert play["lamp"] == "gate" and play["yours"] is False and "driver's program" in play["detail"]


# -- the pieces --------------------------------------------------------------------

def test_the_dashboard_parser():
    assert robot_link.parse_reply("Robotmode: RUNNING") == "RUNNING"
    assert robot_link.parse_reply("Safetystatus: NORMAL") == "NORMAL"
    assert robot_link.parse_reply("STOPPED rosa.urp") == "STOPPED"
    assert robot_link.parse_reply("Loaded program: /programs/rosa.urp") == "/programs/rosa.urp"
    assert robot_link.parse_reply("true") == "true"


def test_the_port_is_picked_when_not_set(good, monkeypatch):
    monkeypatch.setattr(robot_link.config, "ROBOT_PORT", "")
    monkeypatch.setattr(robot_link, "wired_ports", lambda: ["enp1s0", "enp3s0", "enp4s0"])
    monkeypatch.setattr(robot_link, "default_route_port", lambda: "enp1s0")     # the internet is on this one
    monkeypatch.setattr(robot_link, "carrier", lambda port: port == "enp4s0")
    assert robot_link.robot_port() == "enp4s0"                                  # the one with a cable in it
    monkeypatch.setattr(robot_link, "carrier", lambda port: False)
    assert robot_link.robot_port() == "enp3s0"                                  # none plugged: the first spare
    monkeypatch.setattr(robot_link, "wired_ports", lambda: ["enp1s0"])
    assert robot_link.robot_port() == ""
    monkeypatch.setattr(robot_link.config, "ROBOT_PORT", "eth9")
    assert robot_link.robot_port() == "eth9"                                    # the setting always wins


def test_robot_status_names_the_first_problem(good, monkeypatch):
    monkeypatch.setattr(robot_link, "ping", lambda ip: False)
    monkeypatch.setattr(robot_link, "tcp_open", lambda ip, port: False)
    robot_link.refresh()
    text = robot_link.robot_status.invoke({})
    assert text.splitlines()[0].startswith("Mode: real")
    assert "172.31.1.69 does not answer" in text
    assert "3. Robot reachable: bad" in text
    assert robot_link.ALL_TOOLS == [robot_link.robot_status]


# -- review focus -----------------------------------------------------------------------

def test_a_failing_probe_is_a_rung_not_a_crash(good, monkeypatch):
    def no_sysfs(port):
        raise FileNotFoundError("/sys/class/net/enp3s0/carrier")
    monkeypatch.setattr(robot_link, "carrier", no_sysfs)
    ladder = robot_link.ladder("real")
    assert ladder["rungs"][0]["lamp"] == "gate" and "carrier" in ladder["rungs"][0]["detail"]
    assert lamps(ladder)[1:] == ["ok"] * 6                        # a gate does not stop the ladder
    def no_tmux(name):
        raise FileNotFoundError("tmux")
    monkeypatch.setattr(robot_link, "carrier", lambda port: True)
    monkeypatch.setattr(robot_link, "tmux_session", no_tmux)
    assert robot_link.ladder("real")["rungs"][4]["lamp"] == "gate"


def test_missing_addresses_are_explained(good, monkeypatch):
    pinged = []
    monkeypatch.setattr(robot_link, "ping", lambda ip: pinged.append(ip))
    monkeypatch.setattr(robot_link.config, "ROBOT_IP", "")
    rung = robot_link.ladder("real")["rungs"][2]
    assert rung["lamp"] == "gate" and "Set robot_ip in settings" in rung["detail"] and pinged == []
    monkeypatch.setattr(robot_link.config, "COMPUTER_IP", "")
    rung = robot_link.ladder("real")["rungs"][1]
    assert rung["lamp"] == "gate" and "Set computer_ip in settings" in rung["detail"]


def test_a_hanging_dashboard_times_out_into_a_gate(monkeypatch):
    """The real dashboard probe against a socket that never answers: it must give up after its
    1 s timeouts, and the ladder shows that as a gate rung rather than hanging. Only the probes
    before rung 4 are faked here, so the dashboard function itself is the one under test."""
    class Hung:
        def __init__(self):
            self.timeouts = []
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def settimeout(self, s): self.timeouts.append(s)
        def makefile(self, *a, **k): return self
        def readline(self):
            raise socket.timeout("timed out")
    sockets = []
    def connect(address, timeout=None):
        assert address == ("172.31.1.69", 29999) and timeout == 1.0
        sockets.append(Hung())
        return sockets[-1]
    monkeypatch.setattr(robot_link.socket, "create_connection", connect)
    for name, value in (("ROBOT_IP", "172.31.1.69"), ("COMPUTER_IP", "172.31.1.137"), ("ROBOT_PORT", "enp3s0")):
        monkeypatch.setattr(robot_link.config, name, value)
    monkeypatch.setattr(robot_link, "carrier", lambda port: True)
    monkeypatch.setattr(robot_link, "port_speed", lambda port: 1000)
    monkeypatch.setattr(robot_link, "addresses", lambda port: ["172.31.1.137/24"])
    monkeypatch.setattr(robot_link, "ping", lambda ip: True)
    monkeypatch.setattr(robot_link, "tcp_open", lambda ip, port: True)
    for name in ("tmux_session", "driver_status", "program_running", "controller_active", "speed_scaling"):
        monkeypatch.setattr(robot_link, name, lambda *a: None)         # the rungs after, not under test here
    rung = robot_link.ladder("real")["rungs"][3]
    assert rung["lamp"] == "gate" and "timed out" in rung["detail"]
    assert sockets and all(t == 1.0 for t in sockets[0].timeouts)


def test_latest_is_replaced_in_one_assignment(good, monkeypatch):
    """A mode change while the thread is mid-ladder must never leave a half-built ladder in
    latest: refresh builds the whole thing first, then swaps it in."""
    before = robot_link.latest
    seen = []
    real_ladder = robot_link.ladder
    def spy(mode):
        seen.append(robot_link.latest is before)        # still the old one while building
        return real_ladder(mode)
    monkeypatch.setattr(robot_link, "ladder", spy)
    monkeypatch.setattr(robot_link.config, "ROBOT_MODE", "sim")
    robot_link.refresh()
    assert seen == [True] and robot_link.latest is not before and robot_link.latest["mode"] == "sim"
    assert robot_link.latest["rungs"][0]["name"] == "ursim"


def test_the_placeholder_says_not_checked_yet():
    placeholder = robot_link.placeholder()
    assert placeholder["overall"] == {"lamp": "wait", "text": "Not checked yet"} and placeholder["rungs"] == []


def test_the_background_thread_survives_a_failing_refresh(monkeypatch):
    calls = []
    def flaky():
        calls.append(1)
        raise RuntimeError("no ip binary")
    monkeypatch.setattr(robot_link, "refresh", flaky)
    stop = robot_link.start_background(interval_s=0.01)
    deadline = time.time() + 2
    while len(calls) < 3 and time.time() < deadline:
        time.sleep(0.01)
    stop.set()                                              # so the thread does not outlive the test
    assert len(calls) >= 3


# -- actions -------------------------------------------------------------------------------

def test_actions_run_record_and_refresh(good, monkeypatch):
    ran = []
    monkeypatch.setattr(robot_link, "run_command", lambda argv: ran.append(argv) or "")
    refreshed = []
    monkeypatch.setattr(robot_link, "refresh", lambda: refreshed.append(1))
    robot_tools.events.clear()
    message = robot_link.run_action("driver_start", "sim")
    assert ran[-1][0].endswith("start.sh") and ran[-1][1:] == ["sim", "--no-console"]
    assert robot_tools.events[-1]["tool"] == "robot" and robot_tools.events[-1]["outcome"] == "done"
    assert robot_tools.events[-1]["args"] == {"name": "driver_start"} and message
    robot_link.run_action("driver_stop", "real")
    assert ran[-1][1:] == ["stop", "--no-console"]
    robot_link.run_action("profile", "real")
    assert ran[-1] == ["nmcli", "con", "up", "robot"]
    asked = []
    monkeypatch.setattr(robot_link, "dashboard", lambda ip, command: asked.append((ip, command)) or "Powering on")
    assert robot_link.run_action("power_on", "sim") == "Powering on"
    robot_link.run_action("brake_release", "real")
    assert asked == [("192.168.56.101", "power on"), ("172.31.1.69", "brake release")]
    assert len(refreshed) == 5


def test_a_failed_action_is_a_runtime_error_and_recorded(good, monkeypatch):
    def broken(argv):
        raise RuntimeError("Connection 'robot' does not exist")
    monkeypatch.setattr(robot_link, "run_command", broken)
    with pytest.raises(RuntimeError, match="does not exist"):
        robot_link.run_action("profile", "real")
    assert robot_tools.events[-1]["outcome"] == "failed"
    with pytest.raises(KeyError):
        robot_link.run_action("reboot", "real")


def test_real_mode_actions_need_this_computer():
    for name in robot_link.ACTIONS:
        assert robot_link.needs_local(name, "real") is True
        assert robot_link.needs_local(name, "sim") is False
