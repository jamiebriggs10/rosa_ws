#!/usr/bin/env bash
# Turn everything on, once installed:   ~/rosa_ws/start.sh
#
# Restarts the console, starts the robot's own software if the settings file names
# a command for it, and prints the link to open. The robot command is whatever the
# robot's owner put in ~/.config/rosa_ws/settings.json as "robot_start", for example
# "~/lab-driver.sh": a script that sources their workspace and launches their driver.
# It runs in a tmux session called "robot", so it survives this terminal closing.
SETTINGS="${ROSA_SETTINGS:-$HOME/.config/rosa_ws/settings.json}"
HERE="$(cd "$(dirname "$0")" && pwd)"

if systemctl --user restart rosa-console 2>/dev/null; then
  echo "Console: restarted (service rosa-console)."
else
  echo "Console: no service installed; run $HERE/install.sh first, or start it with $HERE/run.sh"
fi

ROBOT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("robot_start", ""))' "$SETTINGS" 2>/dev/null)"
ROBOT="${ROBOT/#\~/$HOME}"
if [ -n "$ROBOT" ]; then
  if command -v tmux >/dev/null; then
    tmux kill-session -t robot 2>/dev/null
    tmux new-session -d -s robot "$ROBOT 2>&1 | tee ~/robot.log"
    echo "Robot software: started '$ROBOT' in tmux session 'robot'. Watch it with: tmux attach -t robot  (log: ~/robot.log)"
  else
    nohup bash -c "$ROBOT" > ~/robot.log 2>&1 &
    echo "Robot software: started '$ROBOT' in the background (log: ~/robot.log)"
  fi
else
  echo "Robot software: nothing to start. Start your robot's driver and MoveIt yourself, or put the command in $SETTINGS as \"robot_start\"."
fi

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo
echo "Open http://${IP:-<ip-of-this-computer>}:8000 from the machine that runs Ollama."
