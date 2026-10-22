"""The tool buttons: programs the console may open on its own computer.

The page sends a tool *name*; this module turns it into a command. Only the commands
in TOOLS below can ever be started, so no command text crosses the network.

Most tools open a window (RViz, rqt, a terminal), so they need a screen: the console
service usually runs without one, and then those buttons are reported unavailable
instead of hanging. The bag recorder has no window and works anywhere.
"""
import collections
import datetime
import os
import pathlib
import shutil
import signal
import subprocess

import config
import robot_tools

SESSIONS_DIR = pathlib.Path(__file__).parent / "sessions"      # the same folder the session tables go in

# What each tool is: a label for the button, the command to run, and whether it needs a screen.
Tool = collections.namedtuple("Tool", "label command needs_display")

# Debian/Ubuntu name the default terminal x-terminal-emulator; GNOME's takes "-- bash" instead.
_TERMINAL = ["gnome-terminal", "--", "bash"] if shutil.which("gnome-terminal") else ["x-terminal-emulator", "-e", "bash"]

# Record every topic except the big camera ones (images, point clouds), which would fill the disk.
_RECORD = ["ros2", "bag", "record", "-a", "-x", "(.*)(image|points|cloud|compressed)(.*)"]

TOOLS = {
    "rviz": Tool("RViz", ["rviz2"], True),
    "graph": Tool("Node graph", ["rqt_graph"], True),
    "logs": Tool("ROS logs", ["rqt_console"], True),
    "rqt": Tool("rqt", ["rqt"], True),
    "terminal": Tool("Terminal", _TERMINAL, True),
    "record": Tool("Record session", _RECORD, False),
}

_running = {}     # name -> the Popen of a tool this console started and has not stopped
events = []       # one {"tool": name, "outcome": "started"|"stopped"} per button press


def has_display():
    """Is there a screen to open windows on? Either a display variable names one, or the
    usual socket of an X or Wayland session exists on this computer."""
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return True
    wayland = pathlib.Path(os.environ.get("XDG_RUNTIME_DIR", "/nonexistent")) / "wayland-0"
    return wayland.exists() or pathlib.Path("/tmp/.X11-unix/X0").exists()


def why_unavailable(name):
    """Why a tool cannot start here, in one sentence; empty when it can."""
    tool = TOOLS[name]
    if shutil.which(tool.command[0]) is None:
        return f"{tool.command[0]} is not installed on the console's computer."
    if tool.needs_display and not has_display():
        return "No display: open the console on the robot's computer to use this button."
    return ""


def available(name):
    return why_unavailable(name) == ""


def is_running(name):
    proc = _running.get(name)
    return proc is not None and proc.poll() is None      # poll() gives the exit code once it has quit


def rows():
    """One row per tool, for the page's button row."""
    return [{"name": name, "label": tool.label, "available": available(name), "running": is_running(name)}
            for name, tool in TOOLS.items()]


def command(name):
    """The exact command to run, built when the button is pressed: the bag folder is named
    after the time, and RViz may be given a settings file."""
    argv = list(TOOLS[name].command)
    if name == "record":
        argv += ["-o", str(SESSIONS_DIR / datetime.datetime.now().strftime("bag-%Y-%m-%d-%H%M"))]
    if name == "rviz" and config.RVIZ_CONFIG:
        argv += ["-d", config.RVIZ_CONFIG]
    return argv


def start(name):
    """Start a tool. Returns "started", or "running" if it already is. KeyError for a name
    not in TOOLS; RuntimeError, with the reason, when it cannot run here."""
    tool = TOOLS[name]
    if is_running(name):
        return "running"
    if not available(name):
        raise RuntimeError(why_unavailable(name) or f"{tool.label} cannot start here.")
    env = os.environ.copy()                               # ROS and the workspaces come with it
    if not env.get("DISPLAY") and not env.get("WAYLAND_DISPLAY"):
        env["DISPLAY"] = ":0"                             # the first screen, when nothing names one
    if name == "record":
        SESSIONS_DIR.mkdir(exist_ok=True)
    # Its own session, so stopping the console (or a Ctrl-C in its terminal) does not take it down.
    _running[name] = subprocess.Popen(command(name), env=env, start_new_session=True)
    _note(name, "started", f"{tool.label} started (pid {_running[name].pid})")
    return "started"


def stop(name):
    """Stop a tool this console started. Returns False if it was not running.
    SIGINT first, as Ctrl-C would: ros2 bag needs it to close the bag file properly.
    If the program has not quit after 5 s, SIGTERM."""
    tool = TOOLS[name]
    if not is_running(name):
        _running.pop(name, None)
        return False
    proc = _running.pop(name)
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.terminate()
    _note(name, "stopped", f"{tool.label} stopped")
    return True


def _note(name, outcome, message):
    """Remember a button press here, and in the robot's event list so it reaches the session table."""
    events.append({"tool": name, "outcome": outcome})
    if hasattr(robot_tools, "_record"):
        robot_tools._record("tool", {"name": name}, outcome, message)
