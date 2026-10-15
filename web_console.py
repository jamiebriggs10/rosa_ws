"""Web console: a browser page that replaces the terminal for the ROSA agent.

Run on the ZenBook, open from the Mac at http://<zenbook-ip>:8000.

One Flask app, one page (static/index.html), and these JSON endpoints:
  GET  /state    everything the page shows, polled once a second
  GET  /models   the models Ollama has, and which of them can call tools
  GET  /cell     MoveIt's model of the cell (walls, table, keep-out zones) for the 3D view
  POST /model    {"name": ...}  switch to another model (fresh conversation)
  POST /think    {"on": true|false}  let thinking models reason first (slower)
  POST /command  {"text": ...}  send a command to the model
  POST /cancel   stop the model thinking about the current command
  POST /reset    start a new conversation: the model forgets earlier messages
  POST /limits   {"MAX_STEP_CM": ..., ...}  change the safety limits, within config's ceilings
  POST /rerun    {"tool", "args", "start_cm", "from_start"}  run a move again, without the model
  POST /save     {"name", "tool", "args", "start_cm"}  keep a move in the saved list (a file)
  POST /unsave   {"index": ...}  remove one from the saved list
  POST /confirm  {"id": ..., "ok": true|false}  answer the operator gate
  POST /stop     cancel the motion in progress, without the model
  POST /finish   write the session file and return the table
  GET  /capabilities         what the robot's system offers (discovery + the description file)
  POST /capabilities/rescan  read the graph and the description file again
  POST /ollama   {"url": ...}  test an Ollama address, save it, and list its models
"""
import datetime
import json
import pathlib
import threading
import time
import urllib.request

from flask import Flask, jsonify, request, send_from_directory
from langchain_core.messages import HumanMessage

import config
import generic_tools
import robot_tools

CONFIRM_TIMEOUT_S = 60
SESSIONS_DIR = pathlib.Path(__file__).parent / "sessions"
SAVED_FILE = SESSIONS_DIR / "saved_moves.json"
MOVE_TOOLS = {"move_by", "move_to", "move_path", "move_formula", "move_joint", "go_home"}   # the tools that can be re-run

app = Flask(__name__, static_folder="static")


class Console:
    """All the state behind the page. One instance, one operator, one command at a time."""

    def __init__(self, agent_factory, abort=lambda: None):
        self._agent_factory = agent_factory   # called once, on the first command
        self._abort = abort                   # hangs up on the model mid-answer
        self._agent = None
        self._lock = threading.Lock()         # guards starting a command
        self.busy = False
        self.busy_since = 0.0
        self._current = None                  # the thread of the command running, if any
        self.messages = []                    # what the chat shows, oldest first
        self.records = []                     # one per command, for the session table
        self.pending = None                   # the confirmation card, if one is showing
        self._answer = None
        self._answered = threading.Event()
        self.started = datetime.datetime.now()
        self.saved = self._load_saved()       # moves kept for re-running, survive restarts
        robot_tools.set_confirm_handler(self.ask_operator)

    # -- the operator gate ---------------------------------------------------
    def ask_operator(self, description, preview=None):
        """Called from the agent thread by a tool. Blocks until the page answers."""
        self._answered.clear()
        self.pending = {"id": str(time.time()), "text": description, "preview": preview}
        answered = self._answered.wait(timeout=CONFIRM_TIMEOUT_S)
        self.pending = None
        return bool(answered and self._answer)

    def answer(self, confirmation_id, ok):
        if self.pending and self.pending["id"] == confirmation_id:
            self._answer = ok
            self._answered.set()
            return True
        return False

    # -- commands ------------------------------------------------------------
    def submit(self, text):
        """Start a command for the model. Returns False if one is already running."""
        return self._start(text, self._ask_model, text)

    def rerun(self, label, tool, args):
        """Run a move again straight through its tool: no model, but every safety check."""
        return self._start(f"Run again: {label}", lambda: self._call_tool(tool, args), None)

    def _start(self, shown_text, work, command):
        with self._lock:
            if self.busy:
                return False
            self.busy, self.busy_since = True, time.time()
        self.messages.append({"who": "you", "text": shown_text})
        thread = threading.Thread(target=self._run, args=(shown_text, work, command), daemon=True)
        self._current = thread
        thread.start()
        return True

    def _ask_model(self, text):
        if self._agent is None:
            self._agent = self._agent_factory()
        return self._agent.invoke(text)

    def _call_tool(self, tool, args):
        return getattr(robot_tools, tool).invoke(args)

    def _run(self, shown_text, work, command):
        t0 = time.time()
        robot_tools.mark_command_time()
        robot_tools.events.clear()
        try:
            reply = work(command) if command is not None else work()
        except Exception as exc:
            reply = f"Error: {exc}"
        if not str(reply).strip():
            reply = "The model gave no answer. Try again, or press New conversation."
        if threading.current_thread() is not self._current:
            return                            # the operator stopped this one; its answer is not wanted
        events = list(robot_tools.events)
        self._remember_moves(events, by_model=command is not None)
        total = round(time.time() - t0, 1)
        self.messages.append({"who": "robot", "text": reply, "events": events, "total_s": total,
                              "command": command})
        self.records.append({"time": datetime.datetime.now().strftime("%H:%M:%S"), "model": config.MODEL,
                             "think": config.THINK, "command": shown_text, "events": events,
                             "total_s": total, "reply": reply})
        self._current = None
        self.busy = False

    def _remember_moves(self, events, by_model):
        """Add an exact record of what was done (tools, numbers or formulas, outcomes, positions)
        to the model's memory. Its own reply rarely says this, and without it the model cannot
        refer back to an earlier move, for example to retrace it.

        The record goes in as a note from the console, never as part of the model's own
        words: small models copy whatever they appear to have said before, and were seen
        writing fake "record" lines claiming moves that never happened."""
        history = getattr(self._agent, "chat_history", None)
        moves = [e for e in events if e["tool"] != "stop"]
        if history is None or not moves:
            return
        lines = []
        for e in moves:
            line = f"{e['tool']} {json.dumps(e['args'])[:800]} -> {e['outcome']}"
            if e.get("start_cm"):
                line += f", starting from {e['start_cm']} cm"
            lines.append(line + f". Result: {e['message']}")
        who = "you called" if by_model else "the operator ran a move directly from the console, not through you:"
        history.append(HumanMessage(content=(
            f"(Console note, not from the operator. For your memory only; never repeat or imitate it. "
            f"What actually happened, {who} " + " | ".join(lines) + ")")))

    def cancel(self):
        """The Stop thinking button: give up on the current command. The model may still be
        working on it for a while, but nothing it asks for afterwards will happen."""
        if not self.busy or self._current is None:
            return False
        robot_tools.abandoned_threads.add(self._current)
        try:
            self._abort()                     # stop the model generating, so the next command is not stuck behind it
        except Exception:
            pass
        waited = round(time.time() - self.busy_since, 1)
        self.records.append({"time": datetime.datetime.now().strftime("%H:%M:%S"), "model": config.MODEL,
                             "think": config.THINK, "command": self.messages[-1]["text"],
                             "events": list(robot_tools.events), "total_s": waited, "reply": "(stopped by the operator)"})
        self._current = None
        self._agent = None                    # the old one is still busy; start the next command afresh
        if self.pending:
            self.answer(self.pending["id"], False)
        self.messages.append({"who": "robot", "text": f"Stopped after {waited} s. Nothing more will be done "
                              "for that command. Earlier messages are forgotten.", "events": [], "total_s": waited})
        self.busy = False
        return True

    def _remember_settings(self):
        config.save_settings({"model": config.MODEL, "think": config.THINK, "limits": limits()})

    def set_limits(self, changes):
        """Change safety limits, each within its ceiling. Returns an error message, or None."""
        for name, value in changes.items():
            if name not in config.LIMIT_CEILINGS:
                return f"{name} is not a limit that can be changed."
            if not isinstance(value, (int, float)) or not 0 < value <= config.LIMIT_CEILINGS[name]:
                return f"{name} must be above 0 and at most {config.LIMIT_CEILINGS[name]:g}."
        with self._lock:
            if self.busy:
                return "Wait for the current command to finish first."
            before = limits()
            for name, value in changes.items():
                setattr(config, name, int(value) if name == "MAX_PATH_STEPS" else float(value))
        said = ", ".join(f"{LIMIT_LABELS[n]} {before[n]:g} → {getattr(config, n):g}"
                         for n in changes if before[n] != getattr(config, n)) or "no change"
        self.messages.append({"who": "robot", "text": f"Safety limits changed: {said}.", "events": [], "total_s": 0})
        self.records.append({"time": datetime.datetime.now().strftime("%H:%M:%S"), "model": config.MODEL,
                             "think": config.THINK, "command": f"(safety limits changed: {said})", "events": [],
                             "total_s": 0, "reply": ""})
        self._remember_settings()
        return None

    def new_conversation(self):
        """Forget the conversation so far, e.g. after a string of failures has confused the model."""
        with self._lock:
            if self.busy:
                return False
            self._agent = None
        generic_tools.clear_path()
        self.messages.append({"who": "robot", "text": "New conversation: earlier messages are forgotten.",
                              "events": [], "total_s": 0})
        return True

    def set_ollama(self, url):
        """Use another Ollama from the next command on. Returns False while a command runs."""
        with self._lock:
            if self.busy:
                return False
            config.save_settings({"ollama_url": url})
            self._agent = None
        return True

    def set_think(self, on):
        """Let thinking models reason before answering, from the next command on."""
        with self._lock:
            if self.busy:
                return False
            config.THINK = bool(on)
            self._agent = None
        self._remember_settings()
        self.messages.append({"who": "robot", "text": ("Thinking is on: thinking models will reason before "
                              "answering, which is slower." if on else "Thinking is off.") + " Earlier messages are forgotten.",
                              "events": [], "total_s": 0})
        return True

    # -- saved moves -----------------------------------------------------------
    def _load_saved(self):
        try:
            return json.loads(SAVED_FILE.read_text())
        except (OSError, ValueError):
            return []

    def _write_saved(self):
        SESSIONS_DIR.mkdir(exist_ok=True)
        SAVED_FILE.write_text(json.dumps(self.saved, indent=1))

    def save(self, name, tool, args, start_cm):
        self.saved.append({"name": name, "tool": tool, "args": args, "start_cm": start_cm,
                           "model": config.MODEL, "saved": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")})
        self._write_saved()

    def unsave(self, index):
        if 0 <= index < len(self.saved):
            del self.saved[index]
            self._write_saved()

    def switch_model(self, name):
        """Use another model from the next command on. Returns False while a command is running."""
        with self._lock:
            if self.busy:
                return False
            config.MODEL = name
            self._agent = None                # rebuilt with the new model, with a fresh conversation
        self._remember_settings()
        self.messages.append({"who": "robot", "text": f"Now using {name}. Earlier messages are forgotten.",
                              "events": [], "total_s": 0})
        return True

    def stop(self):
        """The red button: cancel a pending card and the motion, bypassing the model."""
        if self.pending:
            self.answer(self.pending["id"], False)
        message = robot_tools.stop.invoke({})
        self.messages.append({"who": "robot", "text": message, "events": [], "total_s": 0})
        return message

    # -- session file --------------------------------------------------------
    def finish(self):
        SESSIONS_DIR.mkdir(exist_ok=True)
        path = SESSIONS_DIR / self.started.strftime("%Y-%m-%d-%H%M.md")
        lines = [f"# Session {self.started:%Y-%m-%d %H:%M}", "",
                 "| Time | Model | Thinking | Command | Tool calls | First tool after | To motion | Total | Reply |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for r in self.records:
            calls = "; ".join(f"{e['tool']} {e['args']} ({e['outcome']})" for e in r["events"]) or "none"
            motion = ", ".join(f"{e['to_motion_s']} s" for e in r["events"] if "to_motion_s" in e) or ""
            reply = r["reply"].replace("\n", " ").replace("|", "/")
            first = f"{r['events'][0]['after_s']} s" if r["events"] and "after_s" in r["events"][0] else ""
            lines.append(f"| {r['time']} | {r['model']} | {'on' if r.get('think') else 'off'} | {r['command']} | {calls} | {first} | {motion} | {r['total_s']} s | {reply} |")
        path.write_text("\n".join(lines) + "\n")
        return {"path": str(path), "records": self.records}


# -- status checks, each answering in under a second ----------------------------

def ollama_up():
    try:
        urllib.request.urlopen(config.OLLAMA_URL + "/api/tags", timeout=1)
        return True
    except Exception:
        return False


LIMIT_LABELS = {"MAX_STEP_CM": "one move (cm)", "MAX_PATH_REACH_CM": "path reach (cm)",
                "MAX_JOINT_DEG": "joint turn (deg)", "MAX_PATH_STEPS": "points in a path"}


def limits():
    return {name: getattr(config, name) for name in config.LIMIT_CEILINGS}


def ollama_models():
    """Every model Ollama has, and whether it can call tools (the agent needs that)."""
    def get(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        with urllib.request.urlopen(urllib.request.Request(config.OLLAMA_URL + path, data=data), timeout=3) as r:
            return json.load(r)
    names = sorted(m["name"] for m in get("/api/tags")["models"])
    caps = {n: get("/api/show", {"model": n}).get("capabilities", []) for n in names}
    return [{"name": n, "tools": "tools" in caps[n], "thinks": "thinking" in caps[n]} for n in names]


def status():
    s = {"ollama": ollama_up(), "moveit": False, "joint_states": False, "joints": {}, "tool_cm": None}
    try:
        s.update(robot_tools.status())
    except Exception:
        pass
    s["hint"] = hint(s)
    return s


def hint(s):
    """One plain sentence about what is missing, for the first run. Empty when all is well."""
    if not s["ollama"]:
        return (f"Cannot reach Ollama at {config.OLLAMA_URL}. Open Settings below and enter the address of the "
                "machine that runs Ollama (it must be set to listen on the network).")
    if not s["joint_states"]:
        return "No joint states yet: start the robot's driver (or simulator) and MoveIt on this computer."
    if not s["moveit"]:
        return "MoveIt is not answering: start the robot's MoveIt launch on this computer."
    return ""


def test_ollama(url):
    """The models at an Ollama address, or raise with a plain reason."""
    if not isinstance(url, str) or not url.startswith("http"):
        raise ValueError("The address must start with http://, e.g. http://192.168.1.10:11434")
    url = url.rstrip("/")
    with urllib.request.urlopen(url + "/api/tags", timeout=3) as r:
        return url, [m["name"] for m in json.load(r)["models"]]


def package_file(package, rel):
    """Find a file inside a ROS package's share folder, for the robot meshes."""
    if ".." in pathlib.PurePosixPath(rel).parts:      # no climbing out of the package
        return None
    share = config.package_share(package)
    if share is None:
        return None
    candidate = share / rel
    return candidate if candidate.is_file() else None   # may be a symlink into the source tree, fine


# -- routes ------------------------------------------------------------------------

console = None   # set in main() or by the tests


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/state")
def state():
    return jsonify(status=status(), busy=console.busy, pending=console.pending,
                   busy_for_s=round(time.time() - console.busy_since, 1) if console.busy else 0,
                   messages=console.messages, model=config.MODEL, think=config.THINK,
                   pendant=config.PENDANT_URL, joint_list=config.JOINTS, saved=console.saved,
                   limits=limits(), ceilings=config.LIMIT_CEILINGS, labels=LIMIT_LABELS, speed=config.SPEED,
                   shown_path=generic_tools.last_path, ollama_url=config.OLLAMA_URL,
                   frames={"base": config.BASE_FRAME, "tool": config.TOOL_FRAME})


@app.get("/pkg/<package>/<path:rel>")
def package_asset(package, rel):
    """Serve robot mesh files referenced by the URDF as package://<package>/<path>."""
    found = package_file(package, rel)
    if found is None:
        return jsonify(error=f"{package}/{rel} not found"), 404
    return send_from_directory(found.parent, found.name)


@app.post("/command")
def command():
    text = (request.json or {}).get("text", "").strip()
    if not text:
        return jsonify(error="Type a command first."), 400
    if not console.submit(text):
        return jsonify(error="Wait for the current command to finish, or press Stop thinking."), 409
    return jsonify(ok=True)


@app.post("/confirm")
def confirm():
    body = request.json or {}
    if not console.answer(body.get("id"), bool(body.get("ok"))):
        return jsonify(error="That confirmation has expired."), 409
    return jsonify(ok=True)


@app.get("/models")
def models():
    try:
        return jsonify(models=ollama_models(), current=config.MODEL)
    except Exception as exc:
        return jsonify(error=f"Cannot reach Ollama at {config.OLLAMA_URL}: {exc}"), 502


@app.post("/model")
def model():
    name = (request.json or {}).get("name", "")
    try:
        usable = {m["name"] for m in ollama_models() if m["tools"]}
    except Exception as exc:
        return jsonify(error=f"Cannot reach Ollama at {config.OLLAMA_URL}: {exc}"), 502
    if name not in usable:
        return jsonify(error=f"{name} cannot call tools, so it cannot drive the robot."), 400
    if not console.switch_model(name):
        return jsonify(error="Wait for the current command to finish first."), 409
    return jsonify(ok=True, current=config.MODEL)


@app.get("/capabilities")
def capabilities():
    """What the robot's system offers, for the panel on the page."""
    try:
        rows = generic_tools.capability_rows()
    except Exception as exc:
        return jsonify(error=f"Cannot read the ROS graph: {exc}"), 502
    names = {r["name"] for r in rows}
    return jsonify(system=generic_tools.description().system, interfaces=rows,
                   moveit=config.MOVE_ACTION in names and config.CARTESIAN_SERVICE in names,
                   file=str(config.INTERFACES_FILE), allow_unlisted_writes=config.ALLOW_UNLISTED_WRITES)


@app.post("/capabilities/rescan")
def rescan():
    try:
        generic_tools.reload_description()
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    return capabilities()


@app.post("/ollama")
def set_ollama():
    try:
        url, models = test_ollama((request.json or {}).get("url", ""))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception as exc:
        return jsonify(error=f"Cannot reach Ollama at {(request.json or {}).get('url', '')}: {exc}"), 502
    if not console.set_ollama(url):
        return jsonify(error="Wait for the current command to finish first."), 409
    return jsonify(ok=True, url=url, models=models)


@app.get("/cell")
def cell():
    """MoveIt's model of the cell, so the 3D view can show where the arm may not go."""
    try:
        return jsonify(shapes=robot_tools._get_robot().cell_objects())
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 502


@app.post("/think")
def think():
    if not console.set_think((request.json or {}).get("on", False)):
        return jsonify(error="Wait for the current command to finish first."), 409
    return jsonify(ok=True, think=config.THINK)


@app.post("/limits")
def change_limits():
    error = console.set_limits(request.json or {})
    if error:
        return jsonify(error=error), 400
    return jsonify(ok=True, limits=limits())


@app.post("/reset")
def reset():
    if not console.new_conversation():
        return jsonify(error="Wait for the current command to finish first."), 409
    return jsonify(ok=True)


@app.post("/cancel")
def cancel():
    if not console.cancel():
        return jsonify(error="The model is not working on anything."), 409
    return jsonify(ok=True)


def _shift_to_start(tool, args, start_cm):
    """A move re-run "from its start": first go back to where it started, then do it,
    all as one path. Only straight-line moves remember their start."""
    here = robot_tools._cm(robot_tools._get_robot().tool_pose())
    back = [round(s - h, 1) for s, h in zip(start_cm, here)]
    if tool == "move_by":
        points = [back, [b + d for b, d in zip(back, (args["dx"], args["dy"], args["dz"]))]]
    elif tool in ("move_path", "move_formula"):
        path = args["points"] if tool == "move_path" else robot_tools.evaluate_formula(args["formula"])
        points = [back] + [[b + p for b, p in zip(back, point)] for point in path]
    else:
        return tool, args
    return "move_path", {"points": points}


@app.post("/rerun")
def rerun():
    body = request.json or {}
    tool, args = body.get("tool"), body.get("args") or {}
    if tool not in MOVE_TOOLS:
        return jsonify(error="Only moves can be run again."), 400
    label = body.get("name") or tool
    if body.get("from_start") and body.get("start_cm"):
        try:
            tool, args = _shift_to_start(tool, args, body["start_cm"])
        except (RuntimeError, ValueError) as exc:
            return jsonify(error=str(exc)), 502
        label += " (from its start)"
    if not console.rerun(label, tool, args):
        return jsonify(error="Wait for the current command to finish first."), 409
    return jsonify(ok=True)


@app.post("/save")
def save():
    body = request.json or {}
    if body.get("tool") not in MOVE_TOOLS:
        return jsonify(error="Only moves can be saved."), 400
    console.save(body.get("name") or body["tool"], body["tool"], body.get("args") or {}, body.get("start_cm"))
    return jsonify(ok=True)


@app.post("/unsave")
def unsave():
    console.unsave(int((request.json or {}).get("index", -1)))
    return jsonify(ok=True)


@app.post("/stop")
def stop():
    return jsonify(message=console.stop())


@app.post("/finish")
def finish():
    return jsonify(console.finish())


def main():
    global console
    from rosa_agent import abort, build_agent
    console = Console(build_agent, abort)
    print(f"Web console on http://0.0.0.0:8000, model {config.MODEL} at {config.OLLAMA_URL}")
    app.run(host="0.0.0.0", port=8000, threaded=True)


if __name__ == "__main__":
    main()
