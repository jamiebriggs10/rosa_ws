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

The two are usually the same machine: see `--with-ollama` under Install.

## Set up a fresh computer

A computer with nothing on it but Ubuntu 22.04 becomes the robot's computer
with one script. It is the only step that needs `sudo`:

```bash
git clone https://github.com/jamiebriggs10/rosa_ws.git ~/rosa_ws
bash ~/rosa_ws/setup-machine.sh
```

It installs, in this order, skipping whatever is already there: the base
tools (`git`, `curl`, `tmux`, Python's venv and pip, `build-essential`), the
HWE kernel for recent hardware, the NVIDIA open driver if an NVIDIA card is
found (570 or newer, from the graphics-drivers PPA if Ubuntu's own list stops
short), ROS 2 Humble with MoveIt and the Universal Robots driver, Docker (from Docker's own repository, as docker-ce) for
URSim, and Ollama with the model (`--model NAME`, default `llama3.1:8b`).
Flags turn steps off: `--no-kernel`, `--no-nvidia`, `--no-docker`,
`--no-ollama`; `--ssh` adds an SSH server. It refuses to run on anything but
Ubuntu 22.04, keeps a log in `~/setup-machine.log`, and ends by saying what it
changed. **If it says reboot, reboot** (a new kernel or graphics driver only
works after one), then continue with the install below. Read the script top to
bottom first: it is one function per step, each with a note saying why.

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

With the model on the same computer: `install.sh --with-ollama` also installs
Ollama if it is missing (that one step asks for your password) and pulls the
model (`--model NAME`, default `llama3.1:8b`). The settings default,
`http://localhost:11434`, then needs no change.

## Turning it on after that

```bash
~/rosa_ws/start.sh          # sim if the settings have robot_start_sim, else real, else console only
~/rosa_ws/start.sh sim      # the simulator and the robot software against it, then the console
~/rosa_ws/start.sh real     # the robot software against the real robot, then the console
~/rosa_ws/start.sh stop     # stop the simulator and the robot software; the console keeps running
```

Each mode restarts the console and prints the links: `http://localhost:8000`
on the robot's computer, and its network address for any other machine.
The robot software comes from three optional keys in
`~/.config/rosa_ws/settings.json`, each a command or script of your own:

| Key | Runs | tmux session, log |
|---|---|---|
| `sim_start` | the simulator itself, for example URSim | `sim`, `~/sim.log` |
| `robot_start_sim` | your driver and MoveIt against the simulator | `robot`, `~/robot.log` |
| `robot_start` | your driver and MoveIt against the real robot | `robot`, `~/robot.log` |
| `voice_start` | the voice app (see *Voice control*), started with the console, left running by `stop` | `voice`, `~/voice.log` |

`tmux attach -t robot` shows the robot software running (Ctrl-b then d to
leave it running). Without tmux the command runs in the background instead.

**The real robot is only started by a person in the room.** `start.sh real`
refuses to run from an SSH session; if you are in the room and the remote
shell is deliberate, `ROSA_REMOTE_REAL=1 ~/rosa_ws/start.sh real`.

Ready-made scripts for the Strathclyde UR12e cell (the `UR_humble` workspace)
are in `examples/`, with the settings that name them in
`examples/settings.example.json`:

- `ur_humble.sim.sh` starts URSim in Docker for a UR12e, powers it on and
  releases the brakes, launches the lab bringup in sim mode, and moves the
  simulated arm to the agent's home pose. Because it starts the simulator
  itself, `sim_start` is left unused; it goes in `robot_start_sim` alone.
- `ur_humble.real.sh` waits for the real control box, launches the lab
  bringup without the camera, and keeps the joint controller awake after
  every Play on the pendant. It goes in `robot_start`.

```bash
cp ~/rosa_ws/examples/settings.example.json ~/.config/rosa_ws/settings.json
~/rosa_ws/start.sh          # the simulator, since robot_start_sim is set
```

For another robot, copy one of them and change the sourced workspace and the
launch line.

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

## The robot-connection widget

The button next to the three lights says in a few words whether the robot can
be driven ("Real robot: not plugged in", "Simulator: ready"). Click it for the
full picture: a ladder of checks, run every few seconds on the robot's
computer, that stops at the first problem and says whose job it is.

| Real robot | Simulator |
|---|---|
| 1. a cable on the robot port | 1. URSim answering |
| 2. this computer's address on it (button: use the saved network profile) | 2. power, brakes, safety, program (buttons: power on, release brakes) |
| 3. the control box answering (ping, dashboard, primary, RTDE ports) | 3. the driver (button: start, stop) |
| 4. power, brakes, safety, loaded program, Local or Remote (buttons as the robot allows) | 4. the driver's program running |
| 5. the driver (button: start, stop) | 5. the speed slider |
| 6. Play on the pendant, and the joint controller active | |
| 7. the speed slider, with a warning above 20 % | |

The addresses come from `settings.json`: `robot_mode` (`sim` or `real`,
switched in the dialog), `robot_ip`, `computer_ip`, `robot_port` (empty: picked
automatically), `network_profile`, `ursim_ip`. The example settings carry the
Strathclyde cell's values.

In real mode the buttons work only when the page is opened on the robot's
computer's own screen, the same rule as `start.sh real`. Nothing on this panel
moves the arm; Play, the speed slider, protective stops and the e-stop stay on
the pendant on purpose. The model has a read-only `robot_status` tool with the
same ladder, so "is the robot ready?" gets a true answer.

## Voice control

Speech is handled by a separate program, the **voice app** (the Voice Control
Prototype: wake word or push-to-talk, then speech-to-text, with every stage
swappable for the dissertation's comparisons). It runs on the same computer
as the console, on port 8100, with its own settings file. The console listens
to it; the voice app never talks to the robot.

Next to the robot widget there is a **Voice** pill and a gear. The pill is the
switch: click it to turn voice control on or off (grey "Voice off"; green
"Voice on, idle"; amber while armed or listening). The gear opens the voice
pop-up: the same switch, a microphone level bar, the live text as it is
heard with the felt delay in milliseconds, a *Hold to talk* button, and then
every setting of the voice app (device; for each of trigger, VAD, end of turn
and speech-to-text the option in force and its knobs, each with a "What is
this?"), presets to load and save, the voice app's address, and the
voice-confirm checkbox. Settings changes are saved by the voice app in its
`settings.yaml` and apply to the next thing you say.

With voice on, what you say arrives in the chat as a command, marked with a
🎤, and the model treats it exactly like typed text: the same tools, the same
safety checks, the same confirm card. Two things are different:

- **"Stop"** (or "halt", "freeze", on its own) halts the arm at once, without
  the model. With a streaming speech-to-text engine this happens on the live
  text, before you have finished the word.
- **Voice confirm** is off by default: a confirm card is answered on screen.
  Tick the box in the pop-up and a spoken "yes" (or "confirm", "go",
  "go ahead", "do it", "okay") confirms the card and "no" or "cancel" cancels
  it. It is off by default because a misheard "yes" would move the arm.

If the model is still busy when you speak, nothing is queued: the chat says
what was heard and asks you to say it again. The session table and the
summary have a *Via* column (typed or voice).

The settings in `settings.json`: `voice_url` (default `http://127.0.0.1:8100`),
`voice_on`, `voice_confirm`, and `voice_start`, the command that starts the
voice app (`start.sh` runs it in tmux session `voice` alongside the console
and leaves it alone if it is already running; `tmux kill-session -t voice`
stops it; its output is in `~/voice.log`).

Getting the voice app onto the robot's computer: copy its folder (including
the git-ignored `models/`, the downloaded speech models) to, say, `~/voice`,
make a venv from its `requirements/nvidia.txt` (or `mac.txt`), and leave
`device: auto` in its `settings.yaml`; its `docs/03-portability.md` has the
moving-day checklist. The voice app is kept separate on purpose: its speech
libraries are heavy, its benchmark harness does not need ROS, and the console
only needs its HTTP side (`voice.py` is the whole link).

## The tool buttons

When the page is opened on the robot's computer itself, a row of buttons
under the safety limits opens the usual ROS tools on that screen, next to
the console:

| Button | Opens |
|---|---|
| RViz | `rviz2`, with the file named by `"rviz_config"` in `settings.json` if there is one |
| Node graph | `rqt_graph`: which nodes talk to which |
| ROS logs | `rqt_console`: every node's log lines |
| rqt | `rqt`: plots, topic monitor, the rest of the rqt plugins |
| Terminal | a terminal with ROS and your workspaces already loaded |
| Record session | `ros2 bag record` of every topic except images and point clouds, into `sessions/bag-<date>`; press again to stop |

The page only ever sends a button's name; the commands live in
`desktop_tools.py` and nothing else can be started from the page. Buttons
that need a screen are hidden when the console has none, for example when the
page is opened from another computer; `start.sh` hands the screen to the
console service, so run it from a terminal on the robot's computer's own
desktop. Each press is a line in the session table.

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
| `setup-machine.sh` | the once-only sudo steps for a fresh Ubuntu 22.04 computer |
| `install.sh`, `run.sh`, `start.sh` | install; run the console; turn everything on (`sim`, `real`, `stop`) |
| `desktop_tools.py` | the tool buttons: the fixed list of programs the console may open |
| `robot_link.py` | the robot-connection ladder: the checks, the actions, the `robot_status` tool |
| `voice.py` | the link to the voice app: its event stream, the word tests ("stop", "yes", "no"), the proxied requests |
| `examples/` | the description file and the start scripts for the Strathclyde UR12e cell |
| `tests/` | `./venv/bin/python -m pytest` (no robot needed); `tests/sim/` has helpers for the simulator |

## Attribution and citing

Built on **ROSA** (Robot Operating System Agent) by NASA Jet Propulsion
Laboratory, https://github.com/nasa-jpl/rosa, Apache License 2.0, installed
unmodified from PyPI. See `NOTICE`. This repository is licensed under the
Apache License 2.0 as well (`LICENSE`).

If you use this in academic work, please cite: J. Briggs, *Natural-language
and speech control of a collaborative robot*, 4th-year dissertation,
University of Strathclyde, 2027, and ROSA (Royce et al., NASA JPL, 2024).
