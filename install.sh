#!/usr/bin/env bash
# One-command install on a computer that runs ROS 2 Humble and MoveIt:
#
#   curl -fsSL https://raw.githubusercontent.com/jamiebriggs10/rosa_ws/main/install.sh | bash
#
# or, from a clone:  ./install.sh [--workspace ~/my_robot_ws/install/setup.bash]... [--no-service]
#
# It clones (or updates) ~/rosa_ws, makes a Python venv, installs the three packages,
# writes a settings file, and starts the console as a user service on port 8000.
# It never touches the robot's own workspace and never needs sudo.
set -euo pipefail

REPO="${ROSA_REPO:-https://github.com/jamiebriggs10/rosa_ws.git}"
DIR="${ROSA_DIR:-$HOME/rosa_ws}"
ROS="/opt/ros/humble/setup.bash"
CONF="$HOME/.config/rosa_ws"
UNIT_DIR="$HOME/.config/systemd/user"
WORKSPACES=()
SERVICE=1

while [ $# -gt 0 ]; do
  case "$1" in
    --workspace) WORKSPACES+=("$2"); shift 2 ;;
    --no-service) SERVICE=0; shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1 (try --help)"; exit 2 ;;
  esac
done

say() { printf '\n==> %s\n' "$*"; }
fail() { printf '\nInstall stopped: %s\n' "$*" >&2; exit 1; }

say "Checking this computer"
[ -f "$ROS" ] || fail "ROS 2 Humble was not found at $ROS. Install it first: https://docs.ros.org/en/humble/Installation.html"
command -v git >/dev/null || fail "git is not installed (sudo apt install git)"
python3 -c 'import venv' 2>/dev/null || fail "python3's venv module is missing (sudo apt install python3-venv)"
# Ubuntu ships venv without ensurepip unless python3-venv is installed. Without it we can
# still make a venv that borrows the system's pip, as long as python3-pip is there.
if python3 -c 'import ensurepip' 2>/dev/null; then VENV_OPTS=""
elif python3 -m pip --version >/dev/null 2>&1; then VENV_OPTS="--without-pip"
else fail "neither python3-venv nor python3-pip is installed (sudo apt install python3-venv)"; fi
if ! dpkg -s ros-humble-moveit >/dev/null 2>&1; then
  echo "Warning: ros-humble-moveit is not installed. The agent moves the arm through MoveIt, so install it"
  echo "         (sudo apt install ros-humble-moveit) or make sure your robot's workspace provides it."
fi

say "Getting the code into $DIR"
if [ -d "$DIR/.git" ]; then
  (cd "$DIR" && git pull --ff-only) || echo "Could not update $DIR (local changes?). Keeping what is there."
elif [ -e "$DIR" ]; then
  fail "$DIR exists but is not a git clone. Move it aside or set ROSA_DIR to another folder."
else
  git clone -q "$REPO" "$DIR"
fi

say "Python packages (in $DIR/venv, with the system's ROS packages visible)"
# ROS's setup script reads variables that may be unset, so relax -u while it runs.
set +u
# shellcheck disable=SC1090
source "$ROS"
set -u
# shellcheck disable=SC2086
[ -x "$DIR/venv/bin/python" ] || python3 -m venv --system-site-packages $VENV_OPTS "$DIR/venv"
"$DIR/venv/bin/python" -m pip install -q --disable-pip-version-check -r "$DIR/requirements.txt"

say "Settings in $CONF/settings.json"
mkdir -p "$CONF"
python3 - "$CONF/settings.json" "${WORKSPACES[@]}" <<'PY'
import json, pathlib, sys
path, workspaces = pathlib.Path(sys.argv[1]), sys.argv[2:]
try:
    settings = json.loads(path.read_text())
except (OSError, ValueError):
    settings = {}
settings.setdefault("ollama_url", "http://localhost:11434")
if workspaces:
    settings["workspaces"] = workspaces
settings.setdefault("workspaces", [])
path.write_text(json.dumps(settings, indent=2) + "\n")
print(f"  Ollama: {settings['ollama_url']}   (change it on the page under Settings)")
print(f"  Robot workspaces loaded at start: {settings['workspaces'] or 'none (add with --workspace)'}")
PY
if [ ! -f "$CONF/interfaces.yaml" ]; then
  echo "  No $CONF/interfaces.yaml yet: the system's own services and topics will be read-only."
  echo "  See $DIR/examples/ and the README for how to describe them."
fi

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
if [ "$SERVICE" = 1 ]; then
  say "Starting the console as a user service (rosa-console)"
  mkdir -p "$UNIT_DIR"
  cat > "$UNIT_DIR/rosa-console.service" <<UNIT
[Unit]
Description=Natural-language robot console (rosa_ws)
After=network.target

[Service]
ExecStart=$DIR/run.sh
Restart=on-failure
RestartSec=3
Environment=PYTHONUNBUFFERED=1
Environment=ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}
Environment=ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-0}

[Install]
WantedBy=default.target
UNIT
  systemctl --user daemon-reload
  systemctl --user enable -q rosa-console.service
  systemctl --user restart rosa-console.service
  sleep 2
  if systemctl --user is-active -q rosa-console.service; then
    echo "  Running. Logs: journalctl --user -u rosa-console -f"
  else
    echo "  The service did not start. See: journalctl --user -u rosa-console -n 50"
  fi
  echo "  It starts again at every login. To also run it before anyone logs in: sudo loginctl enable-linger $USER"
else
  say "Not installing a service (--no-service). Start the console with:  $DIR/run.sh"
fi

say "Done. Open http://${IP:-<ip-of-this-computer>}:8000 from the machine that runs Ollama."
echo "    On that page: Settings, enter the Ollama address, pick a model, and type a command."
echo "    Next time, to turn everything on:  $DIR/start.sh"
