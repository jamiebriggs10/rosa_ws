#!/usr/bin/env bash
# Turn everything on, once installed:   ~/rosa_ws/start.sh [sim|real|stop]
#
#   sim     start the simulator and the robot software against it, then the console
#   real    start the robot software against the real robot, then the console
#   stop    stop the simulator and the robot software (the console keeps running)
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
    run_in_tmux robot "$ROBOT_START_SIM" "$HOME/robot.log" ;;
  real)
    [ -n "$ROBOT_START" ] || fail "No \"robot_start\" in $SETTINGS. Point it at a script that starts your driver and MoveIt against the real robot (see examples/)."
    refuse_remote_real
    echo "Robot software (real robot):"
    run_in_tmux robot "$ROBOT_START" "$HOME/robot.log" ;;
  console)
    echo "Robot software: nothing to start. Start your robot's driver and MoveIt yourself, or put the command in $SETTINGS as \"robot_start_sim\" or \"robot_start\"." ;;
  -h|--help)
    sed -n '2,17p' "$0"; exit 0 ;;
  *)
    fail "Usage: $0 [sim|real|stop]   (try --help)" ;;
esac

import_display
if systemctl --user restart rosa-console 2>/dev/null; then
  echo "Console: restarted (service rosa-console)."
else
  echo "Console: no service installed; run $HERE/install.sh first, or start it with $HERE/run.sh"
fi

IP="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
echo
echo "Open http://localhost:8000 on this computer, or http://${IP:-<ip-of-this-computer>}:8000 from another one on the network."
