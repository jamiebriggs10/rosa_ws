"""The robot connection panel: is the robot ready to be driven, and if not, what is next?

The console shows a ladder of checks, in the order a person would do them: cable in,
address on the port, robot answering, robot powered and brakes off, driver running,
program playing, speed slider low. Each check is a *rung*: a lamp (ok, bad, gate, wait),
a sentence, and either a button the page can press for the operator (bring the network
profile up, start the driver, power on) or a note that it is the operator's job.

The checks are small *probe* functions at the top: each one asks one thing of the
operating system, the network, the robot's dashboard server or the ROS graph. The ladder
only combines their answers, so the tests replace the probes and never touch anything real.

The ladder is computed in a background thread every few seconds; the page reads `latest`.
Nothing here moves the robot: it only connects and reports.

Terms used below:
- *dashboard*: the UR control box's command line on TCP port 29999. One line in ("power on",
  "robotmode"), one line out ("Robotmode: RUNNING"). It is how a program presses the
  pendant's buttons.
- *carrier*: Linux's word for "a cable is plugged in and the other end is alive".
- *topic* and *service*: ROS 2's two ways of asking: a topic is a stream of messages anyone
  can listen to; a service is a question with one answer.
"""
import pathlib
import socket
import subprocess
import threading
import time

from langchain_core.tools import tool

import config
import robot_tools

DASHBOARD_PORT = 29999
ROBOT_PORTS = (29999, 30001, 30004)       # dashboard, primary interface, RTDE: all open on a live control box
FRESH_S = 2.0                             # a topic value older than this counts as absent
PROGRAM_TOPIC = "/io_and_status_controller/robot_program_running"
SPEED_TOPIC = "/speed_scaling_state_broadcaster/speed_scaling"
TRAJECTORY_CONTROLLER = "scaled_joint_trajectory_controller"
START_SH = pathlib.Path(__file__).parent / "start.sh"
TMUX_SESSION = "robot"                    # start.sh runs the driver and MoveIt in this tmux session

# The buttons the page may press, by name, with their labels. Only names cross the network.
ACTIONS = {
    "profile": "Use the robot profile",
    "driver_start": "Start the driver",
    "driver_stop": "Stop the driver",
    "power_on": "Power on",
    "brake_release": "Release the brakes",
}


# ---------------------------------------------------------------------------
# Probes: one question each. Any of them may raise; the ladder turns that into a rung.
# ---------------------------------------------------------------------------

def carrier(port):
    """Is a cable plugged into this port with something alive at the other end?"""
    return pathlib.Path(f"/sys/class/net/{port}/carrier").read_text().strip() == "1"


def port_speed(port):
    """The link speed in Mb/s, or None when the kernel does not say (no cable gives -1)."""
    try:
        speed = int(pathlib.Path(f"/sys/class/net/{port}/speed").read_text().strip())
    except (OSError, ValueError):
        return None
    return speed if speed > 0 else None


def run_command(argv, timeout_s=120):
    """Run a command and return what it printed; raise with its complaint if it failed."""
    done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
    if done.returncode != 0:
        raise RuntimeError((done.stderr or done.stdout).strip() or f"{argv[0]} failed (exit code {done.returncode})")
    return done.stdout


def addresses(port):
    """The IPv4 addresses on a port, as "a.b.c.d/nn" strings. `ip -o` prints one line per address:
    "2: enp3s0    inet 172.31.1.137/24 brd ... scope global enp3s0"."""
    found = []
    for line in run_command(["ip", "-o", "-4", "addr", "show", "dev", port]).splitlines():
        words = line.split()
        if "inet" in words:
            found.append(words[words.index("inet") + 1])
    return found


def default_route_port():
    """The port the internet goes out of ("default via 192.168.1.1 dev wlp2s0 ..."), or ""."""
    for line in run_command(["ip", "route", "show", "default"]).splitlines():
        words = line.split()
        if "dev" in words:
            return words[words.index("dev") + 1]
    return ""


def wired_ports():
    """Every wired network port on this computer: Linux names them en... (enp3s0, eno1)."""
    return sorted(p.name for p in pathlib.Path("/sys/class/net").iterdir() if p.name.startswith("en"))


def ping(ip):
    """One ping, one second: does anything answer at that address?"""
    return subprocess.run(["ping", "-c", "1", "-W", "1", ip], capture_output=True).returncode == 0


def tcp_open(ip, port):
    """Can we open a connection to that port? Closed within a second either way."""
    try:
        socket.create_connection((ip, port), timeout=1.0).close()
        return True
    except OSError:
        return False


def dashboard(ip, command):
    """Send one command to the robot's dashboard server and return its one-line reply.
    The server greets us first ("Connected: Universal Robots Dashboard Server"), so that line
    is read and dropped. Every step has a one-second timeout: a hung robot must not hang us."""
    with socket.create_connection((ip, DASHBOARD_PORT), timeout=1.0) as sock:
        sock.settimeout(1.0)                      # applies to every read and write that follows
        lines = sock.makefile("rw")
        if not lines.readline():
            raise RuntimeError(f"{ip}:{DASHBOARD_PORT} closed the connection without a welcome line.")
        lines.write(command + "\n")
        lines.flush()
        reply = lines.readline().strip()
    if not reply:
        raise RuntimeError(f"{ip}:{DASHBOARD_PORT} gave no answer to '{command}'.")
    return reply


def parse_reply(reply):
    """The value in a dashboard reply: "Robotmode: RUNNING" -> RUNNING, "Loaded program: x.urp"
    -> x.urp, and for programState's "STOPPED rosa.urp" (no colon) the first word."""
    return reply.split(":", 1)[1].strip() if ":" in reply else reply.split()[0]


def profile_exists(name):
    """Does NetworkManager know a connection of that name? (nmcli -t prints one name per line.)"""
    return name in run_command(["nmcli", "-t", "-f", "NAME", "con", "show"]).splitlines()


def tmux_session(name):
    """Does a tmux session of that name exist? (start.sh keeps the driver in one.)"""
    return subprocess.run(["tmux", "has-session", "-t", name], capture_output=True).returncode == 0


def driver_status():
    """Whether joint states are arriving and MoveIt answers, from the console's own ROS node."""
    s = robot_tools.status()
    return {"joint_states": s["joint_states"], "moveit": s["moveit"]}


# The two topics are watched by subscriptions made once on the console's existing node. Each
# keeps its latest message and when it arrived, so reading a value never waits on ROS.
_topics = {}            # topic name -> (value, time it arrived)
_subscribed = set()     # topics we already listen to


def _watch(topic, msg_type):
    """Start listening to a topic (once). The publisher's delivery settings (QoS) are copied, as
    read_topic does, because a "latched" publisher only reaches a subscriber that asks the same
    way; until the publisher exists there is nothing to copy, so we try again next time."""
    if topic in _subscribed:
        return
    from rclpy.qos import QoSProfile
    node = robot_tools._get_robot()
    publishers = node.get_publishers_info_by_topic(topic)
    if not publishers:
        return
    qos = QoSProfile(depth=1, durability=publishers[0].qos_profile.durability,
                     reliability=publishers[0].qos_profile.reliability)
    node.create_subscription(msg_type, topic, lambda m: _topics.update({topic: (m.data, time.time())}), qos)
    _subscribed.add(topic)


def _fresh(topic, max_age_s=FRESH_S):
    """The last value on a topic, or None when it is older than max_age_s (None: any age)."""
    value, when = _topics.get(topic, (None, 0.0))
    return value if max_age_s is None or time.time() - when < max_age_s else None


def program_running():
    """True/False from the driver's "a program is running" topic, or None when it is silent."""
    from std_msgs.msg import Bool
    _watch(PROGRAM_TOPIC, Bool)
    return _fresh(PROGRAM_TOPIC, max_age_s=None)       # the driver sends it only when it changes


def speed_scaling():
    """The pendant's speed slider in percent (the UR driver's broadcaster publishes 0 to 100),
    or None when the driver is silent."""
    from std_msgs.msg import Float64
    _watch(SPEED_TOPIC, Float64)
    return _fresh(SPEED_TOPIC)


_list_controllers = None


def controller_active(name):
    """Is that ros2_control controller active? One call to the controller manager's list service,
    waiting at most a second. None when the service (or its message package) is not there."""
    global _list_controllers
    try:
        from controller_manager_msgs.srv import ListControllers
    except ImportError:                          # the driver's packages are not installed here
        return None
    node = robot_tools._get_robot()
    if _list_controllers is None:
        _list_controllers = node.create_client(ListControllers, "/controller_manager/list_controllers")
    if not _list_controllers.wait_for_service(timeout_sec=1.0):
        return None
    result = robot_tools._wait(_list_controllers.call_async(ListControllers.Request()), timeout_s=1.0)
    return any(c.name == name and c.state == "active" for c in result.controller)


# ---------------------------------------------------------------------------
# The rungs. Each returns (lamp, detail, action name or None, yours).
# ---------------------------------------------------------------------------

def robot_port():
    """The wired port the robot cable is in: the setting, or else a guess: a wired port with a
    cable in it that is not carrying the internet; failing that, the first wired port that is
    not; failing that, nothing."""
    if config.ROBOT_PORT:
        return config.ROBOT_PORT
    spare = [p for p in wired_ports() if p != default_route_port()]
    plugged = [p for p in spare if carrier(p)]
    return (plugged or spare or [""])[0]


def robot_ip(mode):
    return config.URSIM_IP if mode == "sim" else config.ROBOT_IP


def _rung_cable(mode):
    port = robot_port()
    if not port:
        return "bad", "No spare wired port found on this computer for the robot cable.", None, True
    if not carrier(port):
        return "bad", f"No cable on {port}. Plug the cable from the control box into it.", None, True
    speed = port_speed(port)
    return "ok", f"Cable in on {port}" + (f", {speed} Mb/s." if speed else "."), None, False


def _rung_address(mode):
    if not config.COMPUTER_IP:
        return "gate", "Set computer_ip in settings: this computer's address on the robot cable.", None, True
    port = robot_port()
    have = addresses(port)
    if config.COMPUTER_IP in [a.split("/")[0] for a in have]:
        return "ok", f"{port} has {config.COMPUTER_IP}.", None, False
    has = f"{port} has {', '.join(have) or 'no address'}, not {config.COMPUTER_IP}"
    if not profile_exists(config.NETWORK_PROFILE):
        # Making a connection is a one-time job for a person at the machine (NetworkManager
        # asks the desktop session for permission); after that the button brings it up.
        return ("bad", f"{has}. No network profile '{config.NETWORK_PROFILE}' yet. Once, in a terminal on this computer:\n"
                f"nmcli con add type ethernet ifname {port} con-name {config.NETWORK_PROFILE} "
                f"ipv4.method manual ipv4.addresses {config.COMPUTER_IP}/24", None, True)
    return "bad", f"{has}: bring up the {config.NETWORK_PROFILE} network profile.", "profile", False


def _rung_reachable(mode):
    ip = config.ROBOT_IP
    if not ip:
        return "gate", "Set robot_ip in settings: the control box's address.", None, True
    answers = ping(ip)
    open_ports = [str(p) for p in ROBOT_PORTS if tcp_open(ip, p)]
    if open_ports:
        return "ok", f"{ip} answers (ports {', '.join(open_ports)} open).", None, False
    if answers:
        return "bad", f"{ip} answers ping but none of the robot's ports ({', '.join(map(str, ROBOT_PORTS))}) are open: is that the control box?", None, True
    return "bad", f"Cable is in but {ip} does not answer: control box off, or another computer holds the address?", None, True


def _rung_ursim(mode):
    if tcp_open(config.URSIM_IP, DASHBOARD_PORT):
        return "ok", f"URSim answers at {config.URSIM_IP}.", None, False
    return "bad", f"URSim is not running (nothing at {config.URSIM_IP}:{DASHBOARD_PORT}).", "driver_start", False


def _rung_state(mode):
    ip = robot_ip(mode)
    v = {name: parse_reply(dashboard(ip, command)) for name, command in (
        ("robotmode", "robotmode"), ("safety", "safetystatus"), ("program", "programState"),
        ("loaded", "get loaded program"), ("remote", "is in remote control"))}
    values = (f"robot mode {v['robotmode']}, safety {v['safety']}, program {v['program']}, "
              f"loaded {v['loaded']}, remote control {v['remote']}")
    if v["robotmode"] == "RUNNING" and v["safety"] == "NORMAL":
        return "ok", f"Powered, brakes released: {values}.", None, False
    action = {"POWER_OFF": "power_on", "IDLE": "brake_release"}.get(v["robotmode"])
    if v["safety"] != "NORMAL":
        return "gate", f"Safety is {v['safety']}: clear it on the pendant. {values}.", None, True
    return "gate", f"Robot is {v['robotmode']}: {values}.", action, action is None


def _word(value):
    """A yes/no value as a word for a sentence; None (nobody answered) is "unknown"."""
    return "unknown" if value is None else ("yes" if value else "no")


def _rung_driver(mode):
    session = tmux_session(TMUX_SESSION)
    s = driver_status()
    values = f"joint states {_word(s['joint_states'])}, MoveIt {_word(s['moveit'])}, tmux session {_word(session)}"
    if session and s["joint_states"]:
        return "ok", f"Driver running: {values}.", "driver_stop", False
    what = "the simulator and driver" if mode == "sim" else "the driver"
    return "bad", f"Start {what} ({values}).", "driver_start", False


def _rung_play(mode):
    running, active = program_running(), controller_active(TRAJECTORY_CONTROLLER)
    values = f"program running {_word(running)}, {TRAJECTORY_CONTROLLER} active {_word(active)}"
    if running and active:
        return "ok", f"The driver's program is running ({values}).", None, False
    if mode == "sim":
        return "gate", f"Waiting for the driver's program to run ({values}).", None, False
    return "gate", f"Press Play on the pendant (Real, not Simulation, program loaded). {values}.", None, True


def _rung_speed(mode):
    percent = speed_scaling()
    if percent is None:
        return "gate", "No speed value from the driver yet.", None, False
    percent = round(percent)
    if percent <= 20:
        return "ok", f"Speed slider at {percent} %.", None, False
    return "gate", f"Speed slider at {percent} %: set it to 10 to 20 % for a first run.", None, True


# name, title, the short words for the header widget when this rung is the first problem, the check
RUNGS = {
    "real": [("cable", "Cable in", "not plugged in", _rung_cable),
             ("address", "This computer's address", "no address", _rung_address),
             ("reachable", "Robot reachable", "not answering", _rung_reachable),
             ("state", "Robot powered", "not powered", _rung_state),
             ("driver", "Driver and MoveIt", "driver off", _rung_driver),
             ("play", "Program running", "waiting for Play", _rung_play),
             ("speed", "Speed slider", "speed too high", _rung_speed)],
    "sim": [("ursim", "URSim running", "URSim off", _rung_ursim),
            ("state", "Robot powered", "not powered", _rung_state),
            ("driver", "Driver and MoveIt", "driver off", _rung_driver),
            ("play", "Program running", "waiting for the program", _rung_play),
            ("speed", "Speed slider", "speed too high", _rung_speed)],
}


def ladder(mode):
    """Run the checks for a mode, in order, and return the ladder the page draws. Stops at the
    first bad rung (the rest wait). A probe that raises makes its rung a gate with the error,
    so a missing program or file is a sentence on the page, never a crash."""
    rungs, stopped, first = [], False, None
    for name, title, short, check in RUNGS[mode]:
        if stopped:
            lamp, detail, action, yours = "wait", "Later.", None, False
        else:
            try:
                lamp, detail, action, yours = check(mode)
            except Exception as exc:
                lamp, detail, action, yours = "gate", f"Could not check: {exc}", None, False
            stopped = lamp == "bad"
        rung = {"name": name, "title": title, "lamp": lamp, "detail": detail,
                "action": {"name": action, "label": ACTIONS[action]} if action else None, "yours": yours}
        rungs.append(rung)
        if first is None and lamp != "ok":
            first = (short, rung)
    if first is None:
        overall, alert = {"lamp": "ok", "text": "ready"}, {"lamp": "ok", "text": "Ready"}
    else:
        overall = {"lamp": first[1]["lamp"], "text": first[0]}
        alert = {"lamp": first[1]["lamp"], "text": first[1]["detail"]}
    return {"mode": mode, "overall": overall, "alert": alert, "rungs": rungs}


def placeholder():
    """What the page shows before the first check has run."""
    waiting = {"lamp": "wait", "text": "Not checked yet"}
    return {"mode": config.ROBOT_MODE, "overall": waiting, "alert": dict(waiting), "rungs": []}


latest = placeholder()
_watching = False        # True once start_background() keeps latest fresh


def refresh():
    """Compute the ladder for the current mode and publish it in one assignment, so a reader
    (the /state route, the model's tool) always sees a whole ladder, never a half-built one."""
    global latest
    latest = ladder(config.ROBOT_MODE)
    return latest


def start_background(interval_s=3.0):
    """Keep `latest` fresh from a daemon thread. Returns an Event: set it to stop the thread
    (the tests do; the console never does). A failing refresh is printed and tried again."""
    global _watching
    _watching = True
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            try:
                refresh()
            except Exception as exc:
                print(f"[robot] connection check failed: {exc}")
            stop.wait(interval_s)

    threading.Thread(target=loop, daemon=True, name="robot-ladder").start()
    return stop


# ---------------------------------------------------------------------------
# Actions: the steps a program can do for the operator
# ---------------------------------------------------------------------------

def needs_local(name, mode):
    """Must this action come from the robot's own computer? Everything in real mode: a real arm
    is only started by someone who can see it. The simulator may be started from anywhere."""
    return mode == "real"


def run_action(name, mode):
    """Do one action and return a sentence about it. KeyError for an unknown name, RuntimeError
    with the reason when it fails. Either way it is recorded for the session table and the
    ladder is checked again straight after, so the page shows the result."""
    label = ACTIONS[name]
    try:
        if name == "profile":
            run_command(["nmcli", "con", "up", config.NETWORK_PROFILE])
            message = f"Network profile {config.NETWORK_PROFILE} is up."
        elif name == "driver_start":
            run_command([str(START_SH), mode, "--no-console"])
            message = f"Driver starting ({mode}) in tmux session {TMUX_SESSION}."
        elif name == "driver_stop":
            run_command([str(START_SH), "stop", "--no-console"])
            message = "Driver stopped."
        else:                                           # power_on, brake_release: a dashboard command
            message = dashboard(robot_ip(mode), name.replace("_", " "))
    except Exception as exc:
        robot_tools._record("robot", {"name": name}, "failed", f"FAILED: {label}: {exc}")
        refresh()
        raise RuntimeError(f"{label}: {exc}")
    robot_tools._record("robot", {"name": name}, "done", message)
    refresh()
    return message


# ---------------------------------------------------------------------------
# The model's tool: read-only
# ---------------------------------------------------------------------------

def describe(lad):
    """The ladder as plain lines: mode, overall, then one line per rung."""
    mode = "real robot" if lad["mode"] == "real" else "simulator"
    lines = [f"Mode: {mode}. Overall: {lad['overall']['lamp']}, {lad['alert']['text']}"]
    for i, r in enumerate(lad["rungs"], 1):
        lines.append(f"{i}. {r['title']}: {r['lamp']}. {r['detail']}")
    return "\n".join(lines)


@tool
def robot_status() -> str:
    """Whether the robot can be driven right now, and if not what is in the way: the connection
    checks (cable, address, robot answering, powered, driver, program, speed) one line each.
    Reading only; nothing moves. Use it when a move fails or the operator asks if the robot is ready."""
    # The console's thread keeps latest fresh; without one (the terminal agent), check now.
    return describe(latest if _watching else refresh())


ALL_TOOLS = [robot_status]
