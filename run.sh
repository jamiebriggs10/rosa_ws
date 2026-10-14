#!/usr/bin/env bash
# Start the console with ROS and the robot's own workspaces loaded.
# Used by the installed service, and by hand:  ~/rosa_ws/run.sh
# The workspaces come from ~/.config/rosa_ws/settings.json ("workspaces": [setup.bash files]).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SETTINGS="${ROSA_SETTINGS:-$HOME/.config/rosa_ws/settings.json}"
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
if [ -f "$SETTINGS" ]; then
  while IFS= read -r ws; do
    ws="${ws/#\~/$HOME}"
    if [ -f "$ws" ]; then source "$ws"; else echo "run.sh: workspace file not found, skipping: $ws" >&2; fi
  done < <(python3 -c 'import json,sys; [print(w) for w in json.load(open(sys.argv[1])).get("workspaces", [])]' "$SETTINGS")
fi
cd "$HERE"
exec "$HERE/venv/bin/python" "$HERE/web_console.py"
