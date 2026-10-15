# rosa_ws: talk to a ROS 2 robot in plain English

Type "move up 5 cm" or "take a scan view" in a browser, and a robot arm running
ROS 2 and MoveIt does it, through a language model that runs on your own
machine. Nothing leaves your network.

Built by Jamie Briggs for a University of Strathclyde dissertation, on top of
[ROSA](https://github.com/nasa-jpl/rosa) (NASA JPL, Apache 2.0).

## What it is

- **One extra ROS 2 node** on the robot's computer. It changes nothing in the
  robot's own software.
- **Motion through MoveIt's standard interfaces** (plan, check, preview,
  confirm, execute, measure), so any robot that runs MoveIt 2 works. The
  robot-specific names (planning group, frames, home pose) sit in one file.
- **The robot system's own services and topics**, discovered at start and
  usable by the model without any code for them: call a Trigger service,
  send a point, read a result. A short description file gives them meaning
  and says which ones can move the robot.
- **A safety layer** that caps every move, refuses what MoveIt cannot plan
  cleanly, refuses writes to anything not described, and puts every motion
  behind a confirm card that shows the plan on a 3D view first.
- **A web console** with a chat, the 3D view, a stop button, the safety
  limits, the model picker and the settings.

## What you need

| Where | What |
|---|---|
| The robot's computer | Ubuntu 22.04 with ROS 2 Humble and MoveIt 2, the robot's own driver and MoveIt launch, `git`, `python3-pip` |
| Any machine on the same network | [Ollama](https://ollama.com) with a model that can call tools (`ollama pull llama3.1:8b`), set to listen on the network (Ollama app: Settings, "Expose Ollama to the network"; or `OLLAMA_HOST=0.0.0.0 ollama serve`) |

The two can be the same machine.

## Install, on the robot's computer

```bash
curl -fsSL https://raw.githubusercontent.com/jamiebriggs10/rosa_ws/main/install.sh | bash
```

If the robot's packages live in a workspace of their own (most do), tell the
installer where its `setup.bash` is, so the console can find the robot's
meshes and messages:

```bash
curl -fsSL https://raw.githubusercontent.com/jamiebriggs10/rosa_ws/main/install.sh | bash -s -- --workspace ~/my_robot_ws/install/setup.bash
```

The installer clones this repository to `~/rosa_ws`, makes a Python venv,
installs three packages, writes `~/.config/rosa_ws/settings.json`, and
starts the console as a user service (`rosa-console`) that comes back at
every login. It never needs `sudo` and never touches the robot's workspace.
It ends by printing the address to open. Run it again at any time to update.

Without a service: `install.sh --no-service`, then `~/rosa_ws/run.sh` by hand.

## Turning it on after that

```bash
~/rosa_ws/start.sh
```

Restarts the console, starts the robot's own software if you have told it how,
and prints the link. To have it start the robot's software too, put the command
in `~/.config/rosa_ws/settings.json`:

```json
"robot_start": "~/lab-driver.sh"
```

where `lab-driver.sh` is your own few lines: source your workspace, launch your
driver and MoveIt. It runs in a tmux session called `robot` (`tmux attach -t
robot` to watch it).

## First run

1. Start the robot's driver (or its simulator) and its MoveIt launch as you
   normally do, on the robot's computer.
2. From the machine with Ollama, open `http://<robot computer>:8000`.
3. Under **Settings**, enter the Ollama address (for example
   `http://192.168.1.10:11434`) and press *Save and test*. The models
   there appear in the **Model** menu; pick one that can call tools.
4. Check the three lights at the top: Model, MoveIt, Joint states. If one is
   off, the sentence under them says what to start.
5. Type `where is the arm right now?`, then `move up 5 cm`. A card shows the
   planned path on the 3D view; press *Confirm and move*.

If the robot is not a UR12e, first edit the MoveIt section of
`~/rosa_ws/config.py`: the planning group, the base and tool frame names, and
the home pose, as its MoveIt config defines them.

## Letting the model use the robot system's own nodes

The panel **What this system offers** lists every topic, service and action
on the robot's computer. Until you describe them they are read-only.

Describe them in `~/.config/rosa_ws/interfaces.yaml`:

```yaml
system: UR_humble inspection cell. A UR12e with a 3D camera and a probe.

interfaces:
  /multiview_scan/capture:
    means: Take one 3D view from where the camera is now and add it to the scan.
    moves_robot: false
  /clicked_point:
    means: Pick a point on the scanned surface. The system fits a flat patch there and lays a raster over it.
    moves_robot: false
  /inspection/raster_path:
    means: The raster path over the picked patch.
  /touchdown_test/execute:
    means: Run the previewed touchdown. The arm lowers the probe onto the surface.
    moves_robot: true
```

- `means` is what the model and the operator read.
- `moves_robot: true` (the default when the line is missing) puts that
  call behind the confirm card.
- Anything not listed can be read but not written. To allow writes to
  unlisted interfaces, with a confirm card each time, set
  `"allow_unlisted_writes": true` in `settings.json`.

Press *Rescan* on the panel after editing the file. A full example for the
Strathclyde/BAE inspection cell is in `examples/ur_humble.interfaces.yaml`.

The model gets four generic tools for these: `list_capabilities`,
`call_trigger(service)` for `std_srvs/Trigger` services,
`publish_point(topic, x, y, z)` for `geometry_msgs/PointStamped` topics
(centimetres from the robot's base), and `read_topic(topic)` for any topic.
A path read this way is drawn on the 3D view. Other interface types can be
added the same way, one tool per type, never per node.

## Safety

- Every move is planned by MoveIt against its planning scene, so the cell's
  walls, table and keep-out zones are respected if they are in that scene.
- Steps longer than 20 cm, paths that reach more than 30 cm from their start,
  and joint turns over 30 degrees are refused. The limits can be changed on
  the page, within fixed ceilings.
- Motion runs at a tenth of the robot's speed.
- Every motion and every `moves_robot` call waits for the operator's
  confirm. **Stop robot** halts the arm without waiting for the model.
- The result reported to the model is what was measured, not what was asked.
- This is research software. Keep the robot's own emergency stop within reach
  and the speed slider low, and use a simulator first.

## Files

| File | Purpose |
|---|---|
| `config.py` | defaults: Ollama, the robot as MoveIt knows it, safety limits; reads the settings file |
| `robot_tools.py` | the MoveIt adapter: the motion tools, the safety checks, the operator confirm |
| `discovery.py` | reads what the ROS graph offers |
| `interfaces.py` | the description file and the write policy |
| `generic_tools.py` | the tools for the system's own services and topics |
| `rosa_agent.py` | builds the ROSA agent; `python rosa_agent.py` is a terminal version |
| `web_console.py`, `static/` | the browser console |
| `install.sh`, `run.sh`, `start.sh` | install; run the console; turn everything on |
| `tests/` | `./venv/bin/python -m pytest` (no robot needed); `tests/sim/` has helpers for the simulator |

## Attribution and citing

Built on **ROSA** (Robot Operating System Agent) by NASA Jet Propulsion
Laboratory, https://github.com/nasa-jpl/rosa, Apache License 2.0, installed
unmodified from PyPI. See `NOTICE`. This repository is licensed under the
Apache License 2.0 as well (`LICENSE`).

If you use this in academic work, please cite: J. Briggs, *Natural-language
and speech control of a collaborative robot*, 4th-year dissertation,
University of Strathclyde, 2027, and ROSA (Royce et al., NASA JPL, 2024).
