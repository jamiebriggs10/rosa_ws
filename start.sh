#!/usr/bin/env bash
# Turn everything on, once installed:   ~/rosa_ws/start.sh [sim|real|stop] [--no-console]
#
#   sim     start the simulator and the robot software against it, then the console
#   real    start the robot software against the real robot, then the console
#   stop    stop the simulator and the robot software (the console and Lichtblick keep running)
#   (none)  sim if the settings file has "robot_start_sim", else real if it has
#           "robot_start", else the console only
#
# The commands come from ~/.config/rosa_ws/settings.json:
#   "sim_start"        the simulator itself, e.g. URSim            -> tmux session "sim",   log ~/sim.log
#   "robot_start_sim"  driver and MoveIt against the simulator     -> tmux session "robot", log ~/robot.log
#   "robot_start"      driver and MoveIt against the real robot    -> tmux session "robot", log ~/robot.log
# Ready-made scripts for the UR_humble cell are in examples/ (see the README).
#
# The real robot is only started by a person in the room: "real" is refused from an SSH
# session unless ROSA_REMOTE_REAL=1 is set on purpose.
set -euo pipefail
SETTINGS="${ROSA_SETTINGS:-$HOME/.config/rosa_ws/settings.json}"
HERE="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-}"
CONSOLE=1; [ "${2:-}" = "--no-console" ] && CONSOLE=0   # the console itself calls this: do not restart it

fail() { printf '%s\n' "$*" >&2; exit 1; }

setting() {
  # One string from the settings file, with a leading ~ expanded; empty when it is missing.
  local value
  value="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2]) or "")' "$SETTINGS" "$1" 2>/dev/null || true)"
  echo "${value/#\~/$HOME}"
}

SIM_START="$(setting sim_start)"
ROBOT_START_SIM="$(setting robot_start_sim)"
ROBOT_START="$(setting robot_start)"

# No argument: pick the mode from what the settings file offers, simulator first.
if [ -z "$MODE" ]; then
  if [ -n "$ROBOT_START_SIM" ]; then MODE=sim
  elif [ -n "$ROBOT_START" ]; then MODE=real
  else MODE=console; fi
fi

refuse_remote_real() {
  # sshd sets SSH_CONNECTION in every remote shell. A real arm must only be started by
  # someone who can see it and reach its stop button, so a remote shell is refused.
  if [ -n "${SSH_CONNECTION:-}" ] && [ "${ROSA_REMOTE_REAL:-}" != 1 ]; then
    echo "Refusing to start the real robot from a remote (SSH) session." >&2
    echo "The real robot is only started by a person in the room, at the robot's computer." >&2
    echo "If you are in the room and this remote shell is on purpose:  ROSA_REMOTE_REAL=1 $0 real" >&2
    exit 3
  fi
}

run_in_tmux() {
  # Run a command in a named tmux session, so it survives this terminal closing, with its
  # output copied to a log file. Without tmux it runs in the background instead.
  local session="$1" cmd="$2" log="$3"
  if command -v tmux >/dev/null; then
    tmux kill-session -t "$session" 2>/dev/null || true
    tmux new-session -d -s "$session" "$cmd 2>&1 | tee $log"
    echo "  $session: '$cmd' in tmux session '$session'. Watch it: tmux attach -t $session  (log: $log)"
  else
    nohup bash -c "$cmd" > "$log" 2>&1 &
    echo "  $session: '$cmd' in the background, no tmux here (log: $log)"
  fi
}

stop_session() {
  # End a tmux session, or without tmux the background command it would have held.
  local session="$1" cmd="$2"
  if command -v tmux >/dev/null; then
    tmux kill-session -t "$session" 2>/dev/null && echo "  $session: stopped" || echo "  $session: was not running"
  elif [ -n "$cmd" ]; then
    pkill -f "$cmd" 2>/dev/null && echo "  $session: stopped" || echo "  $session: was not running"
  fi
}

start_lichtblick() {
  # The engineer's view: Lichtblick's web build (the official image) on port 8081, with our
  # layout baked in, so the console's Lichtblick tab has something to show. Left running by
  # "stop", like the console. No Docker, or no image and no network: a note, not a failure.
  command -v docker >/dev/null || { echo "  lichtblick: Docker is not installed, so the Lichtblick tab stays empty (setup-machine.sh installs it)"; return 0; }
  if [ -n "$(docker ps -q -f name='^rosa-lichtblick$' 2>/dev/null)" ]; then echo "  lichtblick: already running on port 8081"; return 0; fi
  docker rm -f rosa-lichtblick >/dev/null 2>&1 || true
  # lichtblick/Dockerfile: the official image plus our layout. Built here so a changed layout
  # is picked up at the next start (seconds when nothing changed).
  if docker build -q -t rosa-lichtblick "$HERE/lichtblick" >/dev/null 2>&1 \
     && docker run -d --name rosa-lichtblick --restart unless-stopped -p 8081:8080 rosa-lichtblick >/dev/null 2>&1; then
    echo "  lichtblick: started on port 8081 (container rosa-lichtblick)"
  else
    echo "  lichtblick: could not start (docker pull ghcr.io/lichtblick-suite/lichtblick:latest, or are you in the docker group?)"
  fi
}

import_display() {
  # The console runs as a systemd user service, which does not know which screen you are
  # logged in on. Handing it DISPLAY (X11), XAUTHORITY and WAYLAND_DISPLAY lets the page's
  # tool buttons open RViz and the other windows on this screen. Only the variables that are
  # set are passed; no screen, no service or no systemd (as on a Mac) is harmless.
  local names="" v
  for v in DISPLAY XAUTHORITY WAYLAND_DISPLAY; do [ -n "${!v:-}" ] && names="$names $v"; done
  # shellcheck disable=SC2086
  [ -z "$names" ] || systemctl --user import-environment $names 2>/dev/null || true
}

case "$MODE" in
  stop)
    echo "Stopping the robot software and the simulator (the console keeps running)."
    stop_session robot "${ROBOT_START_SIM:-$ROBOT_START}"
    stop_session sim "$SIM_START"
    exit 0 ;;
  sim)
    [ -n "$ROBOT_START_SIM" ] || fail "No \"robot_start_sim\" in $SETTINGS. Point it at a script that starts your driver and MoveIt against the simulator (see examples/)."
    echo "Simulator and robot software (sim):"
    if [ -n "$SIM_START" ]; then run_in_tmux sim "$SIM_START" "$HOME/sim.log"
    else echo "  sim: no \"sim_start\" set, so the robot_start_sim script is expected to start the simulator itself (the example does)"; fi
    run_in_tmux robot "$ROBOT_START_SIM" "$HOME/robot.log"
    start_lichtblick ;;
  real)
    [ -n "$ROBOT_START" ] || fail "No \"robot_start\" in $SETTINGS. Point it at a script that starts your driver and MoveIt against the real robot (see examples/)."
    refuse_remote_real
    echo "Robot software (real robot):"
    run_in_tmux robot "$ROBOT_START" "$HOME/robot.log"
    start_lichtblick ;;
  console)
    echo "Robot software: nothing to start. Start your robot's driver and MoveIt yourself, or put the command in $SETTINGS as \"robot_start_sim\" or \"robot_start\"." ;;
  -h|--help)
    sed -n '2,17p' "$0"; exit 0 ;;
  *)
    fail "Usage: $0 [sim|real|stop]   (try --help)" ;;
esac

if [ "$CONSOLE" = 1 ]; then
  import_display
  if systemctl --user restart rosa-console 2>/dev/null; then
    echo "Console: restarted (service rosa-console)."
    # From the box's own desktop (a screen, not SSH): open the page once it answers.
    if [ -n "${DISPLAY:-}" ] && [ -z "${SSH_CONNECTION:-}" ] && command -v xdg-open >/dev/null; then
      for i in $(seq 1 20); do curl -fs -m 1 http://localhost:8000/state >/dev/null 2>&1 && break; sleep 1; done
      xdg-open http://localhost:8000 >/dev/null 2>&1 &
    fi
  else
    echo "Console: no service installed; run $HERE/install.sh first, or start it with $HERE/run.sh"
  fi
fi

IP="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
echo
echo "Open http://localhost:8000 on this computer, or http://${IP:-<ip-of-this-computer>}:8000 from another one on the network."
