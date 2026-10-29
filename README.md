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
- **One page, the stage**, made to be used by ear from across the cell: one
  big panel that says what Rosa is doing, heard, proposes and measured; the
  3D view with the planned path; a map of what the connected nodes offer;
  every setting in one pop-up.
- **Voice, in the same program**: the wake word "truffles" or a Hold-to-talk
  button, speech-to-text on this computer, and Rosa speaking back. Only the
  word "confirm" moves the robot.

## What you need

| Where | What |
|---|---|
| The robot's computer | Ubuntu 22.04 with ROS 2 Humble and MoveIt 2, the robot's own driver and MoveIt launch, `git`, `python3-pip`; a microphone and a speaker (the Jabra headset in the lab) for voice |
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

The installer first reports what this computer has and what is missing
(MoveIt, the UR driver, foxglove_bridge, tmux, Docker, the sound libraries, an
NVIDIA GPU, Ollama, the robot workspace), with the command that gets each one;
only ROS 2, git and Python's venv are required to go on. Then it clones this
repository to `~/rosa_ws` (or updates it), makes a Python venv, installs the
Python packages (the GPU build of onnxruntime when there is an NVIDIA card),
fetches the voice models, writes `~/.config/rosa_ws/settings.json`, builds the
Lichtblick image when Docker is usable, and starts the console as a user
service (`rosa-console`) that comes back at every login. It never needs
`sudo` and never touches the robot's workspace. It ends by printing the
address to open. Run it again at any time to update: every step is skipped or
repeated harmlessly.

For the Strathclyde UR12e cell, add `--lab`: the example settings (the two
bringup scripts, the robot and computer addresses, the Jabra microphone), the
cell's `interfaces.yaml`, and the `UR_humble` workspace are filled in where
not already set.

```bash
curl -fsSL https://raw.githubusercontent.com/jamiebriggs10/rosa_ws/main/install.sh | bash -s -- --lab
```

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

## The robot connection

The chip in the header says in a few words whether the robot can be driven
("Real robot: not plugged in", "Simulator: ready"). Settings › Robot connection
has the full picture: a ladder of checks, run every few seconds on the robot's
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

## The stage

The page at `/` is the **stage**, made for working by ear and from across the cell as much
as for typing at the desk: one big panel that says what Rosa is doing, what she heard, what
she proposes and what she measured; the 3D view with the planned path; the **map**, a graph
of what the connected nodes offer; the line to type into; and Confirm, No and Stop buttons.

Everything on the stage comes from the same console. What it shows is driven by the
**loop**: one phase at a time, each with a lamp colour, a sound and (by verbosity) a
spoken line:

| Phase | Lamp | Sound | Rosa says (verbosity 1 unless noted) |
|---|---|---|---|
| listening, heard | amber, pulsing | rising two-tone, then a tick | |
| thinking | indigo | a quiet pulse | "On it." (verbosity 2) |
| proposing | orange | a two-note question | the read-back: "Point 10. 14 centimetres, 6 seconds, path clear. Confirm?" |
| moving | green | a falling pair | |
| done | green | a short resolve | "At Point 10." (what was measured) |
| refused | red | a low buzz | the reason, the same words in both personas |
| cancelled, stopped | red | a low note | "Cancelled." (2) / "Stopped at x, y, z." |
| follow | amber, dim | | (the follow-up window: no wake word needed) |

The **Settings** pop-up holds every setting: the model and the Ollama address, the safety
limits, the engineer's view (below), the robot-connection ladder, the Voice section (below),
what the system offers (with the map's "show ROS"), and the session buttons (new
conversation; finish, which writes the session table to `sessions/`).

**The World panel has two tabs.** *World* is Rosa's own view: the robot, the orange ghost
playing the plan, the purple route, the keep-out shapes. *Lichtblick* is the engineer's view:
[Lichtblick](https://github.com/lichtblick-suite/lichtblick) (the open fork of Foxglove
Studio, MPL-2.0) in an iframe, showing every topic, frame, message, image and plot of the
running system through `foxglove_bridge` (port 8765). `start.sh sim|real` starts both the
bridge (in the robot session, when `ros-humble-foxglove-bridge` is installed) and Lichtblick's
web build (the official Docker image, port 8081, with `lichtblick/default-layout.json` baked
in). The address is the `lichtblick_url` setting (default `http://localhost:8081`; empty hides
the tab), under Settings › Engineer's view. The iframe exists only while its tab is chosen,
since Lichtblick renders continuously and the browser shares the GPU with Ollama. Details in
`lichtblick/README.md`.

**The map** draws `/capabilities`: Rosa in the middle, one node per program (MoveIt, the UR
driver, and every node the description file names or whose interfaces it describes), one
small node per thing it offers, with the example phrase to say. Nothing on it is hard-coded:
describe a new node's services in `interfaces.yaml` (optionally with `say:` and `node:`),
press *Rescan*, and it appears, sayable. Amber rings mark what moves the robot; the running
command's path and a spoken tour light up. Tap a node to hear what it offers; tap an offer to
say its phrase. "Show ROS" switches to ROS names and adds the undescribed interfaces, the
rest collapsed into one node per group with a count. The questions the map answers are
answered by the console itself, never by the model: "what can I do (with X)", "what's
connected to X", "which of these move the robot", "where did that go", "show / hide ros".

For page work without ROS (on a Mac), `venv/bin/python tests/mock_console.py --rows 160`
serves the page with scripted answers; see the header of that file. `--voice` runs the real
voice package on the microphone in the same process (start it from your own Terminal so macOS
lets it use the mic), `--voice-file tests/clips/truffles-go-home.wav` loops a clip instead:
the wake word, the sounds, the read-backs and "confirm" then all work at home, with the
pretend robot.

## Voice control

Voice is part of the console: the package `voice/` runs the microphone on its own thread
inside the same program, and the console starts it with everything else. Nothing is
installed or started separately.

- `voice/stages.py`: the wake word **"truffles"** (openWakeWord, a model trained for this
  project, `models/truffles.onnx`) or the **Hold to talk** button; Silero to tell speech
  from silence; the turn ends after `silence_ms` of silence (a wake-word turn) or when the
  button is let go; **Parakeet** (NVIDIA, through sherpa-onnx, int8, CPU or CUDA) turns the
  whole turn into text at once. These are the winners of the voice prototype's benchmark of
  8 October 2026; the prototype itself (`Dissertation/Voice Control Prototype`) stays outside
  this repo as the test bench for comparing options.
- `voice/pipeline.py`: the state machine (idle, armed, listening, finalising), the pre-roll
  so the first word is never lost, the follow-up window, and muting while Rosa speaks so she
  never hears herself. One line per utterance goes to `sessions/voice-<date>.jsonl` with the
  felt delay (end of speech to text).
- `voice/speech.py`: Rosa's voice (Piper, `models/piper/`) and the loop's sounds.
- **Interrupting Rosa.** While she speaks the microphone is muted for speech-to-text, but the
  wake word and Hold to talk still get through: say "truffles" (or press) over her line and she
  stops at once and listens. **Questions.** When a request is incomplete ("go to x forty, y
  minus ten"), the model asks one short question naming the current value ("Keep z at 78.5?"),
  spoken and with the follow-up window open, so "keep it" or a number carries straight on;
  `move_to` keeps any coordinate that is left out.
- `voice/words.py`: what a phrase means. "stop" (alone) halts the arm without the model.
  Only **"confirm"** or **"go ahead"** answer a card; "no" or "cancel" declines it; "no,
  point eleven" declines it and runs the rest next, with one line of context; anything else
  said over a card does the same. One "no" covers the run. A card nobody answers is
  cancelled after `confirm_timeout_s`. After Rosa speaks, the follow-up window (`follow_up_s`)
  needs no wake word.

The Voice section of the Settings pop-up: the switch; the live part (level, what was heard
with its felt delay, what was done with it, Hold to talk); **Hearing** (the microphone, the
wake word's threshold, the end-of-turn silence, voice confirm); **Speaking** (Rosa on or off,
the Piper voice, the speaker, Test voice; the sounds; how much Rosa says, `verbosity` 0, 1 or
2; in whose words, `persona` A or B, see `lines.py`; the follow-up window; the card timeout).
Every one of them is a key of the `voice` block in `settings.json`
(`examples/settings.example.json` lists them with their defaults).

The models: `venv/bin/python -m voice.models` downloads Silero, the Parakeet bundle,
openWakeWord's shared files and the Piper voice into `models/` (git-ignored; the wake word
model is in the repo). On a computer without a sound card library the console still starts;
the Voice section says what is missing.

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
| `web_console.py` | the console: the loop (phases), the confirm gate, the held corrections, the self-questions and the spoken tour, every route |
| `lines.py` | what Rosa says at each moment, in two personas, from measured facts |
| `static/stage.html` | the page; it loads `view.js` (the 3D view), `panels.js` + `panels.css` (the settings pop-up) and `map.js` (the capability map); `static/meshes/` holds the UR meshes for the mock |
| `setup-machine.sh` | the once-only sudo steps for a fresh Ubuntu 22.04 computer |
| `install.sh`, `run.sh`, `start.sh` | install; run the console; turn everything on (`sim`, `real`, `stop`) |
| `lichtblick/` | the engineer's view: Lichtblick's layout, its Dockerfile and a README |
| `robot_link.py` | the robot-connection ladder: the checks, the actions, the `robot_status` tool |
| `voice/` | the voice front end: `words.py`, `audio.py`, `stages.py`, `pipeline.py`, `speech.py`, `models.py`, and the `Voice` object in `__init__.py` |
| `models/` | the speech models (git-ignored) and the wake word `truffles.onnx` (committed) |
| `examples/` | the description file and the start scripts for the Strathclyde UR12e cell |
| `tests/` | `./venv/bin/python -m pytest` (no robot needed; `tests/README.md` for the Mac's Docker run); `tests/sim/` has helpers for the simulator; `tests/clips/` three speech clips; `tests/mock_console.py` serves the page with scripted answers for page work on a Mac (`--voice` runs the voice package too) |

## Attribution and citing

Built on **ROSA** (Robot Operating System Agent) by NASA Jet Propulsion
Laboratory, https://github.com/nasa-jpl/rosa, Apache License 2.0, installed
unmodified from PyPI. See `NOTICE`. This repository is licensed under the
Apache License 2.0 as well (`LICENSE`).

If you use this in academic work, please cite: J. Briggs, *Natural-language
and speech control of a collaborative robot*, 4th-year dissertation,
University of Strathclyde, 2027, and ROSA (Royce et al., NASA JPL, 2024).
