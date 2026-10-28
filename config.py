"""One place for everything that changes between machines, models and robots.

To use the agent with a different MoveIt robot, edit the MoveIt section below.
Nothing else in the code names a particular robot.
"""
import json
import os
import pathlib

# --- The settings file ---------------------------------------------------------
# Everything an operator changes from the console (Ollama address, model, safety
# limits) is kept here, so it survives restarts and the code never needs editing.
# The values below are the defaults; apply_settings() at the bottom of this file
# overrides them with whatever the file holds.

SETTINGS_FILE = pathlib.Path(os.environ.get("ROSA_SETTINGS", "~/.config/rosa_ws/settings.json")).expanduser()

# The optional description of the robot system's own interfaces (see interfaces.py).
INTERFACES_FILE = SETTINGS_FILE.with_name("interfaces.yaml")

# Whether the model may write to interfaces that the description file does not list.
# Off by default: an unknown service might move the robot.
ALLOW_UNLISTED_WRITES = False

# setup.bash files of the robot's own workspaces, sourced by the installed service so
# its packages (robot meshes, message types) can be found. Written by install.sh.
WORKSPACES = []

# --- The language model -------------------------------------------------------

# Where Ollama is running: the address of the machine with the model, port 11434 is
# Ollama's default. Entered on the console's settings panel; this is only the default.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# Which local model to use. Must support tool calling.
MODEL = os.environ.get("ROSA_MODEL", "llama3.1:8b")

# How much text the model can hold in its head at once (tokens). ROSA's system
# prompt plus the tool descriptions are about 6000 tokens, and a thinking model's
# reasoning comes on top, so Ollama's default of 2048 is far too small.
NUM_CTX = int(os.environ.get("ROSA_NUM_CTX", "16384"))

# The longest answer the model may write in one go (tokens). Without a cap a model can
# fall into repeating itself and run for many minutes, holding up every later command.
# Thinking models need far more room, because their reasoning counts too.
MAX_ANSWER_TOKENS = 1024
MAX_THINKING_TOKENS = 8192

# Whether models that can "think" (reason before answering, e.g. qwen3) are allowed to.
# Thinking can help with harder requests but makes each step minutes instead of seconds.
# The console has a switch for it.
THINK = os.environ.get("ROSA_THINK", "0") == "1"

# --- The robot, as MoveIt knows it ------------------------------------------------

# The MoveIt planning group: the set of joints MoveIt plans for.
PLANNING_GROUP = "ur_manipulator"

# Positions are measured from BASE_FRAME to TOOL_FRAME. For a UR, "base" matches
# the numbers the teach pendant shows.
BASE_FRAME = "base"
TOOL_FRAME = "tool0"

# MoveIt's standard interfaces. These names are the same on every MoveIt robot.
MOVE_ACTION = "/move_action"                     # plan a move (we only ask it to plan)
EXECUTE_ACTION = "/execute_trajectory"           # run a planned move
CARTESIAN_SERVICE = "/compute_cartesian_path"    # plan a straight line
STOP_TOPIC = "/trajectory_execution_event"       # publishing "stop" here halts the move
SCENE_SERVICE = "/get_planning_scene"            # MoveIt's model of the cell, for the 3D view

# The joints in order from the base, with the word an operator would use.
JOINTS = [
    ("shoulder_pan_joint", "base"),
    ("shoulder_lift_joint", "shoulder"),
    ("elbow_joint", "elbow"),
    ("wrist_1_joint", "wrist 1"),
    ("wrist_2_joint", "wrist 2"),
    ("wrist_3_joint", "wrist 3"),
]

# The home pose, in degrees per joint: "test_configuration" from the lab's MoveIt
# config, arm bent with the tool over the inspection table, clear of the cell walls.
HOME_DEG = {
    "shoulder_pan_joint": 88.2,
    "shoulder_lift_joint": -92.8,
    "elbow_joint": 80.2,
    "wrist_1_joint": -68.8,
    "wrist_2_joint": -91.7,
    "wrist_3_joint": -6.3,
}

# --- The safety layer --------------------------------------------------------------
# Where the arm may go comes from MoveIt's own model of the cell (walls, table,
# keep-out zones), so it is whatever the robot's owner has already set up.

SPEED = 0.1             # fraction of the robot's top speed and acceleration
MAX_STEP_CM = 20.0      # the furthest the tool may travel in one move
MAX_JOINT_DEG = 30.0    # the most one joint may turn in one move
MAX_PATH_STEPS = 200    # the most points in one path
MAX_PATH_REACH_CM = 30.0  # every point of a path stays this close to where it started
MAX_REFUSALS = 3        # after this many refused tool calls in one command, the model must stop

# The four limits above can be changed from the console, but never past these ceilings.
LIMIT_CEILINGS = {"MAX_STEP_CM": 50.0, "MAX_PATH_REACH_CM": 80.0, "MAX_JOINT_DEG": 90.0, "MAX_PATH_STEPS": 500}

# --- The web console ---------------------------------------------------------------

# --- The robot connection panel ----------------------------------------------------
# Which ladder of checks the console shows and which start.sh mode its Start button uses:
# "sim" (the simulator) or "real" (the arm on the cable). Switched on the page.
ROBOT_MODE = "sim"
# The real robot's control box and this computer's address on the cable to it, e.g.
# 172.31.1.69 and 172.31.1.137 in the Strathclyde cell. Empty until set in settings.
ROBOT_IP = ""
COMPUTER_IP = ""
# The wired port the robot cable goes into (e.g. enp130s0f0). Empty: the console picks a
# wired port that has a cable in it and does not carry this computer's internet route.
ROBOT_PORT = ""
# The NetworkManager connection that gives that port its address; "nmcli con up <name>".
NETWORK_PROFILE = "robot"
# Where URSim answers when the simulator is used.
URSIM_IP = "192.168.56.101"

# --- Voice ---------------------------------------------------------------------------
# One block, "voice", in settings.json. The voice package (voice/) reads the microphone, the wake
# word and the speech ones; the console reads the rest. apply_settings() checks every value.
VOICE_DEFAULTS = {
    "on": False,                     # the switch: whether spoken words become commands
    "confirm": True,                 # whether a spoken "confirm" / "no" may answer the confirm card
    "verbosity": 1,                  # 0 sounds only, 1 read-backs and results spoken, 2 everything
    "persona": "A",                  # A the technician (terse), B the colleague (warmer); see lines.py
    "follow_up_s": 8,                # after Rosa speaks, seconds during which no wake word is needed (0..30)
    "confirm_timeout_s": 20,         # how long a confirm card waits before it is cancelled (5..120)
    "map_ros": False,                # the map shows ROS names and the undescribed interfaces
    "input_device": None,            # part of the microphone's name ("Jabra"), or None for the default
    "output_device": None,           # part of the speaker's name, or None for the default
    "wake_threshold": 0.5,           # the wake word fires at or above this score (0.1..0.9)
    "silence_ms": 700,               # silence that ends a wake-word turn (300..1500)
    "speech_voice": "en_GB-alan-medium",
    "speech_enabled": True,
    "sounds_enabled": True,
}
VOICE = dict(VOICE_DEFAULTS)

_bool = lambda v: isinstance(v, bool)                                                   # noqa: E731
_num = lambda lo, hi: lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= v <= hi   # noqa: E731
_name = lambda v: v is None or (isinstance(v, str) and len(v) <= 80)                     # noqa: E731
VOICE_CHECKS = {
    "on": _bool, "confirm": _bool, "speech_enabled": _bool, "sounds_enabled": _bool, "map_ros": _bool,
    "verbosity": lambda v: v in (0, 1, 2) and not isinstance(v, bool),
    "persona": lambda v: v in ("A", "B"),
    "follow_up_s": _num(0, 30), "confirm_timeout_s": _num(5, 120), "wake_threshold": _num(0.1, 0.9),
    "silence_ms": lambda v: isinstance(v, int) and not isinstance(v, bool) and 300 <= v <= 1500,
    "input_device": _name, "output_device": _name,
    "speech_voice": lambda v: isinstance(v, str) and 0 < len(v) <= 80,
}
# The keys a settings.json from before 2026-10-28 used, and where they went.
_OLD_VOICE_KEYS = {"voice_on": "on", "voice_confirm": "confirm", "voice_verbosity": "verbosity", "voice_persona": "persona",
                   "follow_up_s": "follow_up_s", "confirm_timeout_s": "confirm_timeout_s", "map_ros": "map_ros"}
_GONE_KEYS = ("voice_url", "voice_start", "rviz_config")


def voice_settings_ok(changes):
    """None when every key is a voice setting with a value in range, else one sentence saying which."""
    for key, value in (changes or {}).items():
        if key not in VOICE_CHECKS:
            return f"{key} is not one of the voice settings."
        if not VOICE_CHECKS[key](value):
            return f"{key}: {value!r} is out of range."
    return None


def package_share(package):
    """The share folder of a ROS package (where meshes and config live), or None.
    ROS keeps an index of every package in the sourced workspaces; we ask it rather
    than guessing paths, so any robot's description package can be found."""
    try:
        from ament_index_python.packages import get_package_share_directory
        return pathlib.Path(get_package_share_directory(package))
    except Exception:
        return None


# --- Reading and writing the settings file ----------------------------------------

def load_settings(path=None):
    """The settings file as a dict, or {} if it is missing or unreadable. A file from before the
    voice block (flat voice_* keys) is rewritten into the new shape once."""
    path = path or SETTINGS_FILE
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            return {}
    except (OSError, ValueError):
        return {}
    if any(k in data for k in _OLD_VOICE_KEYS) or any(k in data for k in _GONE_KEYS):
        voice = dict(data.get("voice") or {})
        for old, new in _OLD_VOICE_KEYS.items():
            if old in data:
                voice.setdefault(new, data.pop(old))
        for key in _GONE_KEYS:
            data.pop(key, None)
        data["voice"] = voice
        try:
            path.write_text(json.dumps(data, indent=2) + "\n")
        except OSError:
            pass
    return data


def apply_settings(settings):
    """Copy valid values from a settings dict onto this module. Bad values are ignored,
    so a damaged file can never stop the console from starting."""
    global OLLAMA_URL, MODEL, THINK, ALLOW_UNLISTED_WRITES, INTERFACES_FILE, WORKSPACES
    global ROBOT_MODE, ROBOT_IP, COMPUTER_IP, ROBOT_PORT, NETWORK_PROFILE, URSIM_IP
    url = settings.get("ollama_url")
    if isinstance(url, str) and url.startswith("http"):
        OLLAMA_URL = url.rstrip("/")
    for key, value in (settings.get("voice") or {}).items():
        if key in VOICE_CHECKS and VOICE_CHECKS[key](value):
            VOICE[key] = value
    if isinstance(settings.get("model"), str) and settings["model"]:
        MODEL = settings["model"]
    if isinstance(settings.get("think"), bool):
        THINK = settings["think"]
    if isinstance(settings.get("allow_unlisted_writes"), bool):
        ALLOW_UNLISTED_WRITES = settings["allow_unlisted_writes"]
    if isinstance(settings.get("interfaces_file"), str) and settings["interfaces_file"]:
        INTERFACES_FILE = pathlib.Path(settings["interfaces_file"]).expanduser()
    if isinstance(settings.get("workspaces"), list):
        WORKSPACES = [w for w in settings["workspaces"] if isinstance(w, str)]
    if settings.get("robot_mode") in ("sim", "real"):
        ROBOT_MODE = settings["robot_mode"]
    for key, name in (("robot_ip", "ROBOT_IP"), ("computer_ip", "COMPUTER_IP"), ("robot_port", "ROBOT_PORT"),
                      ("network_profile", "NETWORK_PROFILE"), ("ursim_ip", "URSIM_IP")):
        if isinstance(settings.get(key), str):
            globals()[name] = settings[key].strip()
    for name, value in (settings.get("limits") or {}).items():
        if name in LIMIT_CEILINGS and isinstance(value, (int, float)) and not isinstance(value, bool) \
                and 0 < value <= LIMIT_CEILINGS[name]:
            globals()[name] = int(value) if name == "MAX_PATH_STEPS" else float(value)


def save_settings(changes, path=None):
    """Merge changes into the settings file and apply them. Returns the merged dict."""
    path = path or SETTINGS_FILE
    current = load_settings(path)
    merged = {**current, **changes}
    if isinstance(changes.get("voice"), dict):              # the voice block merges one level down
        merged["voice"] = {**(current.get("voice") or {}), **changes["voice"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2) + "\n")
    apply_settings(merged)
    return merged


apply_settings(load_settings())
