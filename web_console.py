"""Web console: a browser page that replaces the terminal for the ROSA agent.

Run on the ZenBook, open from the Mac at http://192.168.1.145:8000.

One Flask app, one page (static/index.html), four JSON endpoints:
  GET  /state    everything the page shows, polled once a second
  POST /command  {"text": ...}  send a command to the model
  POST /confirm  {"id": ..., "ok": true|false}  answer the operator gate
  POST /stop     cancel the motion in progress, without the model
  POST /finish   write the session file and return the table
"""
import datetime
import pathlib
import socket
import threading
import time
import urllib.request

from flask import Flask, jsonify, request, send_from_directory

import config
import robot_tools

CONFIRM_TIMEOUT_S = 60
SESSIONS_DIR = pathlib.Path(__file__).parent / "sessions"

app = Flask(__name__, static_folder="static")


class Console:
    """All the state behind the page. One instance, one operator, one command at a time."""

    def __init__(self, agent_factory):
        self._agent_factory = agent_factory   # called once, on the first command
        self._agent = None
        self._lock = threading.Lock()         # held while a command is running
        self.busy = False
        self.messages = []                    # what the chat shows, oldest first
        self.records = []                     # one per command, for the session table
        self.pending = None                   # the confirmation card, if one is showing
        self._answer = None
        self._answered = threading.Event()
        self.started = datetime.datetime.now()
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
        """Start a command in the background. Returns False if one is already running."""
        if not self._lock.acquire(blocking=False):
            return False
        self.busy = True
        self.messages.append({"who": "you", "text": text})
        threading.Thread(target=self._run, args=(text,), daemon=True).start()
        return True

    def _run(self, text):
        t0 = time.time()
        robot_tools.mark_command_time()
        robot_tools.events.clear()
        try:
            if self._agent is None:
                self._agent = self._agent_factory()
            reply = self._agent.invoke(text)
        except Exception as exc:
            reply = f"Error: {exc}"
        events = list(robot_tools.events)
        total = round(time.time() - t0, 1)
        self.messages.append({"who": "robot", "text": reply, "events": events, "total_s": total})
        self.records.append({"time": datetime.datetime.now().strftime("%H:%M:%S"),
                             "command": text, "events": events, "total_s": total, "reply": reply})
        self.busy = False
        self._lock.release()

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
        lines = [f"# Session {self.started:%Y-%m-%d %H:%M}, model {config.MODEL}", "",
                 "| Time | Command | Tool calls | To motion | Total | Reply |",
                 "|---|---|---|---|---|---|"]
        for r in self.records:
            calls = "; ".join(f"{e['tool']} {e['args']} ({e['outcome']})" for e in r["events"]) or "none"
            motion = ", ".join(f"{e['to_motion_s']} s" for e in r["events"] if "to_motion_s" in e) or ""
            reply = r["reply"].replace("\n", " ").replace("|", "/")
            lines.append(f"| {r['time']} | {r['command']} | {calls} | {motion} | {r['total_s']} s | {reply} |")
        path.write_text("\n".join(lines) + "\n")
        return {"path": str(path), "records": self.records}


# -- status checks, each answering in under a second ----------------------------

def ollama_up():
    try:
        urllib.request.urlopen(config.OLLAMA_URL + "/api/tags", timeout=1)
        return True
    except Exception:
        return False


def ursim_up():
    try:
        with socket.create_connection(("192.168.56.101", 29999), timeout=0.5):
            return True
    except OSError:
        return False


def status():
    s = {"ollama": ollama_up(), "ursim": ursim_up(), "driver": False, "program": False, "joints": None}
    try:
        s.update(robot_tools.status())
    except Exception:
        pass
    return s


def package_file(package, rel):
    """Find a file inside a ROS package's share folder, for the robot meshes."""
    if ".." in pathlib.PurePosixPath(rel).parts:      # no climbing out of the package
        return None
    for root in config.PACKAGE_ROOTS:
        share = pathlib.Path(root.format(pkg=package)) / package
        candidate = share / rel
        if candidate.is_file():                     # may be a symlink into the source tree, fine
            return candidate
    return None


# -- routes ------------------------------------------------------------------------

console = None   # set in main() or by the tests


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/state")
def state():
    return jsonify(status=status(), busy=console.busy, pending=console.pending,
                   messages=console.messages, model=config.MODEL, pendant="http://localhost:6080/vnc.html",
                   joint_names=robot_tools.JOINT_NAMES)


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
        return jsonify(error="Wait for the current command to finish, or press Stop."), 409
    return jsonify(ok=True)


@app.post("/confirm")
def confirm():
    body = request.json or {}
    if not console.answer(body.get("id"), bool(body.get("ok"))):
        return jsonify(error="That confirmation has expired."), 409
    return jsonify(ok=True)


@app.post("/stop")
def stop():
    return jsonify(message=console.stop())


@app.post("/finish")
def finish():
    return jsonify(console.finish())


def main():
    global console
    from rosa_agent import build_agent
    console = Console(build_agent)
    print(f"Web console on http://0.0.0.0:8000, model {config.MODEL} at {config.OLLAMA_URL}")
    app.run(host="0.0.0.0", port=8000, threaded=True)


if __name__ == "__main__":
    main()
