#!/usr/bin/env bash
# Rosa on the lab PC: one file on the Desktop that updates and starts everything.
# Run it in a terminal on the box's own screen (the real robot is refused over SSH):
#
#   ~/Desktop/rosa.sh            update the code, then start the simulator and the console
#   ~/Desktop/rosa.sh real       update, then the real robot (someone on the stop; pendant ready)
#   ~/Desktop/rosa.sh sim        update, then the simulator
#   ~/Desktop/rosa.sh update     update only: the installer's checks, venv, models, settings, service
#   ~/Desktop/rosa.sh stop       stop the simulator and the robot software (console stays up)
#   ~/Desktop/rosa.sh status     what is running, and the addresses
#
# "update" is the one-line installer from GitHub (install.sh --lab), which reports what this
# computer has and what is missing, and then the one apt package it cannot add itself. When
# GitHub cannot be reached, the copy already in ~/rosa_ws is used. install.sh --lab puts a
# fresh copy of this file on the Desktop every time, so it keeps up with the repository.
set -uo pipefail
MODE="${1:-sim}"
ONE_LINER="https://raw.githubusercontent.com/jamiebriggs10/rosa_ws/main/install.sh"

update() {
  echo "== Updating (install.sh --lab)"
  if curl -fsSL -m 20 "$ONE_LINER" | bash -s -- --lab; then :
  else
    echo "== GitHub not reachable or the installer failed; trying the copy in ~/rosa_ws"
    bash ~/rosa_ws/install.sh --lab || { echo "Install failed: read the lines above."; exit 1; }
  fi
  # Packages the installer reports as missing need sudo, which it never uses. The new one
  # since 29 October is foxglove_bridge; add it here when a person is at the keyboard.
  if ! dpkg -s ros-humble-foxglove-bridge >/dev/null 2>&1 && [ -t 0 ]; then
    echo "== foxglove_bridge is missing (the Lichtblick tab's data): installing it needs your password"
    sudo apt-get install -y ros-humble-foxglove-bridge
  fi
}

status() {
  echo "== Console:  $(systemctl --user is-active rosa-console 2>/dev/null || echo 'not a service')   http://localhost:8000"
  echo "== Ollama:   $(curl -fs -m 2 http://localhost:11434 >/dev/null 2>&1 && echo answering || echo 'not answering')"
  echo "== Robot software (tmux):"; tmux ls 2>/dev/null | sed 's/^/   /' || echo "   none running"
  echo "== Lichtblick: $(docker ps --format '{{.Names}} {{.Status}}' 2>/dev/null | grep rosa-lichtblick || echo 'not running')   http://localhost:8081"
  echo "== Code: $(cd ~/rosa_ws 2>/dev/null && git log -1 --format='%h %s' | cut -c1-80)"
}

case "$MODE" in
  update) update ;;
  sim|real) update; echo; ~/rosa_ws/start.sh "$MODE" ;;
  stop) ~/rosa_ws/start.sh stop ;;
  status) status ;;
  -h|--help) sed -n '2,16p' "$0" ;;
  *) echo "Usage: $0 [sim|real|update|stop|status]"; exit 2 ;;
esac
