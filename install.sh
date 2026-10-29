#!/usr/bin/env bash
# One-command install on a computer that runs ROS 2 Humble and MoveIt:
#
#   curl -fsSL https://raw.githubusercontent.com/jamiebriggs10/rosa_ws/main/install.sh | bash
#
# or, from a clone:  ./install.sh [--lab] [--workspace ~/my_robot_ws/install/setup.bash]...
#                                 [--no-service] [--with-ollama] [--model NAME]
#
# It checks what this computer has and says what is missing and how to get it, clones (or
# updates) ~/rosa_ws, makes a Python venv, installs the Python packages (the GPU build of
# onnxruntime when there is an NVIDIA card), fetches the voice models, writes a settings
# file, pulls Lichtblick's image when Docker is usable, and starts the console as a user
# service on port 8000. --lab is the Strathclyde UR12e cell: the example settings (the two
# bringup scripts, the robot addresses), the interfaces file and the UR_humble workspace.
# It never touches the robot's own workspace and never needs sudo, except that
# --with-ollama also installs Ollama on this computer (that one step asks for your
# password) and pulls the model named by --model (default llama3.1:8b). Run it again at
# any time: every step is skipped or repeated harmlessly.
set -euo pipefail

REPO="${ROSA_REPO:-https://github.com/jamiebriggs10/rosa_ws.git}"
DIR="${ROSA_DIR:-$HOME/rosa_ws}"
ROS="/opt/ros/humble/setup.bash"
CONF="$HOME/.config/rosa_ws"
UNIT_DIR="$HOME/.config/systemd/user"
WORKSPACES=()
SERVICE=1
LAB=0
OLLAMA=0
MODEL="llama3.1:8b"

while [ $# -gt 0 ]; do
  case "$1" in
    --workspace) WORKSPACES+=("$2"); shift 2 ;;
    --lab) LAB=1; shift ;;
    --no-service) SERVICE=0; shift ;;
    --with-ollama) OLLAMA=1; shift ;;
    --model) MODEL="$2"; shift 2 ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
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
if [ "$LAB" = 1 ] && [ -f "$HOME/UR_humble/install/setup.bash" ]; then WORKSPACES+=("$HOME/UR_humble/install/setup.bash"); fi

# Everything else is reported, not required: the console runs without it, with the thing it
# needs missing. Each line says how to get it; the sudo ones are what setup-machine.sh does.
MISSING=0
have() {
  # have "what" "how to get it" command...: one line, ok or missing.
  local what="$1" fix="$2"; shift 2
  if "$@" >/dev/null 2>&1; then printf '  ok       %s\n' "$what"
  else printf '  MISSING  %s\n           %s\n' "$what" "$fix"; MISSING=$((MISSING + 1)); fi
}
pkg() { dpkg -s "$1" >/dev/null 2>&1; }
ur_driver() { pkg ros-humble-ur-robot-driver || pkg ros-humble-ur || [ -f "$HOME/UR_humble/install/setup.bash" ]; }
sound_libs() { pkg libportaudio2 && pkg libsndfile1; }
have "MoveIt (the arm moves through it)"            "sudo apt install ros-humble-moveit, or your robot workspace provides it"  pkg ros-humble-moveit
have "the UR driver"                                 "sudo apt install ros-humble-ur (or the UR_humble workspace)"              ur_driver
have "foxglove_bridge (feeds the Lichtblick tab)"    "sudo apt install ros-humble-foxglove-bridge"                              pkg ros-humble-foxglove-bridge
have "tmux (keeps the robot software running)"       "sudo apt install tmux"                                                    command -v tmux
have "Docker, usable without sudo (URSim, Lichtblick)" "setup-machine.sh installs it; log out and in after it adds you to the docker group"  docker ps
have "sound libraries (the microphone, WAV files)"   "sudo apt install libportaudio2 libsndfile1"                               sound_libs
have "an NVIDIA GPU (the model and the voice run on it)" "setup-machine.sh installs the driver; without one everything runs on the CPU, slowly"  nvidia-smi
have "Ollama answering at http://localhost:11434"    "setup-machine.sh installs it, or --with-ollama here, or point Settings at another computer"  curl -fs -m 2 http://localhost:11434
GPU=0; nvidia-smi >/dev/null 2>&1 && GPU=1
for ws in "${WORKSPACES[@]}"; do
  have "robot workspace $ws" "build it first (the lab's: docs/06-lab-pc-setup.md section 2)" test -f "${ws/#\~/$HOME}"
done
[ "$MISSING" = 0 ] && echo "  Everything is here." || echo "  $MISSING missing: the console still installs; the lines above say what each one needs."

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
if [ "$GPU" = 1 ]; then
  # onnxruntime-gpu replaces onnxruntime (same import name, CUDA build) for Silero and the
  # wake word; the two cannot be installed together, so the CPU one is left out and removed.
  grep -v '^onnxruntime' "$DIR/requirements.txt" | "$DIR/venv/bin/python" -m pip install -q --disable-pip-version-check -r /dev/stdin
  "$DIR/venv/bin/python" -m pip uninstall -q -y onnxruntime 2>/dev/null || true
  "$DIR/venv/bin/python" -m pip install -q --disable-pip-version-check 'onnxruntime-gpu>=1.20'
  echo "  onnxruntime-gpu (NVIDIA card found)"
else
  "$DIR/venv/bin/python" -m pip install -q --disable-pip-version-check -r "$DIR/requirements.txt"
fi

say "Voice models (in $DIR/models; fetched once, about 700 MB the first time)"
(cd "$DIR" && "$DIR/venv/bin/python" -m voice.models) || echo "  Could not fetch them (no network?). Voice will say so on the page; run  $DIR/venv/bin/python -m voice.models  later."

if [ "$OLLAMA" = 1 ]; then
  say "Ollama on this computer"
  # Ollama serves the model at http://localhost:11434, which is the settings default.
  if command -v ollama >/dev/null; then
    echo "  already installed ($(ollama --version 2>/dev/null | head -1))"
  else
    echo "  Installing with Ollama's own installer. This step asks for your password (sudo)."
    curl -fsSL https://ollama.com/install.sh | sh
  fi
  echo "  Pulling the model $MODEL (several gigabytes the first time)"
  ollama pull "$MODEL"
fi

say "Settings in $CONF/settings.json"
mkdir -p "$CONF"
EXAMPLE=""; [ "$LAB" = 1 ] && EXAMPLE="$DIR/examples/settings.example.json"
python3 - "$CONF/settings.json" "$EXAMPLE" "${WORKSPACES[@]}" <<'PY'
import json, pathlib, sys
path, example, workspaces = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
try:
    settings = json.loads(path.read_text())
except (OSError, ValueError):
    settings = {}
if example:        # --lab: the cell's example settings fill in whatever is not set yet; nothing already set changes
    added = [k for k, v in json.loads(pathlib.Path(example).read_text()).items() if settings.setdefault(k, v) is v]
    print(f"  From the lab example: {', '.join(added) if added else 'nothing, all set already'}")
settings.setdefault("ollama_url", "http://localhost:11434")
if workspaces:
    settings["workspaces"] = sorted(set(settings.get("workspaces", [])) | set(workspaces))
settings.setdefault("workspaces", [])
path.write_text(json.dumps(settings, indent=2) + "\n")
print(f"  Ollama: {settings['ollama_url']}   (change it on the page under Settings)")
print(f"  Robot workspaces loaded at start: {settings['workspaces'] or 'none (add with --workspace)'}")
print(f"  Robot software: {settings.get('robot_start_sim') or 'no robot_start_sim'}; {settings.get('robot_start') or 'no robot_start'}")
PY
if [ ! -f "$CONF/interfaces.yaml" ]; then
  if [ "$LAB" = 1 ]; then
    cp "$DIR/examples/ur_humble.interfaces.yaml" "$CONF/interfaces.yaml"; echo "  interfaces.yaml: the lab cell's, copied from examples/"
  else
    echo "  No $CONF/interfaces.yaml yet: the system's own services and topics will be read-only."
    echo "  See $DIR/examples/ and the README for how to describe them."
  fi
fi

say "Lichtblick (the engineer's view) and foxglove_bridge"
if docker ps >/dev/null 2>&1; then
  if docker build -q -t rosa-lichtblick "$DIR/lichtblick" >/dev/null 2>&1; then echo "  image rosa-lichtblick built (start.sh runs it on port 8081)"
  else echo "  could not build the image (no network?); start.sh will try again"; fi
else
  echo "  Docker is not usable here, so the Lichtblick tab will have nothing to show until it is (setup-machine.sh)"
fi
pkg ros-humble-foxglove-bridge && echo "  foxglove_bridge is installed: the bringup scripts launch it on port 8765" \
  || echo "  foxglove_bridge is missing (sudo apt install ros-humble-foxglove-bridge): the tab will show no data"

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
TimeoutStopSec=10
RestartSec=3
Environment=PYTHONUNBUFFERED=1
Environment=ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}
Environment=ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-1}

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

if [ "$OLLAMA" = 1 ]; then
  say "Done. Open http://localhost:8000 on this computer (or http://${IP:-<ip-of-this-computer>}:8000 from another)."
  echo "    On that page: Settings, pick the model $MODEL, and type a command."
else
  say "Done. Open http://${IP:-<ip-of-this-computer>}:8000 from the machine that runs Ollama."
  echo "    On that page: Settings, enter the Ollama address, pick a model, and type a command."
fi
echo "    To turn everything on:  $DIR/start.sh sim   (the simulator)   or   $DIR/start.sh real   (the robot, at the machine)"
[ "$MISSING" = 0 ] || echo "    $MISSING thing(s) were missing at the top of this run; the lines there say how to get each."
