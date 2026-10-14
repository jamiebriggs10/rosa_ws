# rosa_ws: natural-language control of a robot arm through MoveIt

Jamie Briggs, Strathclyde 4th-year dissertation (project 6074).

Type a plain-English command ("move up 5 cm") and the arm moves. The agent is
one extra ROS 2 node that plugs into a robot already running **MoveIt**: it
plans and runs every move through MoveIt's standard interfaces, so it is not
tied to one robot. The language model runs locally through Ollama.

| File | Purpose |
|---|---|
| `config.py` | model and Ollama address; the robot as MoveIt knows it (planning group, frames, home pose); safety limits |
| `robot_tools.py` | the MoveIt adapter: the five tools the model may call, the safety checks, the operator confirm |
| `rosa_agent.py` | builds the ROSA agent and runs a text loop in the terminal |
| `web_console.py`, `static/` | the browser console: chat, confirm card with a preview of the plan, 3D view, stop |

## Run

The robot driver and MoveIt must already be running (simulator or robot).

```bash
source /opt/ros/humble/setup.bash
cd ~/rosa_ws && ./venv/bin/python web_console.py      # or rosa_agent.py for the terminal
```

To use another MoveIt robot, edit the MoveIt section of `config.py`.

## Attribution

Built on **ROSA** (Robot Operating System Agent) by NASA Jet Propulsion
Laboratory, https://github.com/nasa-jpl/rosa, licensed under the Apache
License 2.0. ROSA's licence and copyright notices are kept intact in the
installed package. See NOTICE.
