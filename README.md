# rosa_ws: natural-language control of a UR12e

Jamie Briggs, Strathclyde 4th-year dissertation (project 6074).

Type a plain-English command on the ZenBook and a simulated UR12e moves.
The language model runs locally on a Mac through Ollama; this code runs on
the ROS 2 machine.

| File | Purpose |
|---|---|
| `config.py` | model name, Ollama address, context size (all overridable by env vars) |
| `robot_tools.py` | the four actions the model may take, with the first safety checks |
| `rosa_agent.py` | builds the ROSA agent and runs the text loop |

## Run

```bash
source /opt/ros/humble/setup.bash
cd ~/rosa_ws && ./venv/bin/python rosa_agent.py
```

The UR driver must already be running against URSim or the robot.

## Attribution

Built on **ROSA** (Robot Operating System Agent) by NASA Jet Propulsion
Laboratory, https://github.com/nasa-jpl/rosa, licensed under the Apache
License 2.0. ROSA's licence and copyright notices are kept intact in the
installed package. See NOTICE.
