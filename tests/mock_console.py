#!/usr/bin/env python3
"""A scripted console for page work on the Mac: serves the real page and scripts from static/ with
pretend answers, so the stage can be worked on without ROS, a model or a robot. From the console it
imports only the voice package and config.py, which are pure Python (the Mac has no rclpy); it
never runs on the lab PC.

    python3 tests/mock_console.py [--port 8002] [--rows 160] [--think 3.5] [--voice | --voice-file clip.wav]

--voice runs the real voice package in this process on the microphone (start it from your own
Terminal so macOS lets it use the mic); --voice-file loops a WAV instead. Either way spoken words
arrive as commands, the sounds and the read-backs play, exactly as on the lab PC. Without them the
voice routes answer as if the voice package were not running.

Then open http://localhost:8002/. What it understands: "show me the raster (and its points)",
"go to point N", "go home", "up/down/left/right/forward/back N", "joint N by D degrees", "take a
scan view", "run the raster", "where is the tool", "what can I do (with X)", "which of these move
the robot", "where did that go", "show/hide ros", "stop", and on a card: "confirm", "no", "no,
point N". Everything else is refused with the list of nodes. Phases advance on timers with the
plan's facts, so the page sees the same /state shape as from the real console.
--rows pads the capability list with that many undescribed interfaces (the map's hard case).
"""
import argparse
import math
import pathlib
import re
import sys
import threading
import time

from flask import Flask, jsonify, request, send_from_directory

HERE = pathlib.Path(__file__).resolve().parent
STATIC = HERE.parent / "static"
sys.path.insert(0, str(HERE.parent))
import config                                 # noqa: E402  the voice block's defaults and checks
from voice import words                      # noqa: E402  what a spoken phrase means

# The sound each phase makes (the same table as the console's _announce).
PHASE_SOUNDS = {"listening": "listen", "heard": "heard", "thinking": "think", "proposing": "ask", "moving": "go",
                "done": "done", "refused": "refused", "cancelled": "cancel", "stopped": "cancel"}
MESHES = STATIC / "meshes"                 # the UR meshes, kept in the repo for page work without ROS

JOINTS = [["shoulder_pan_joint", "base"], ["shoulder_lift_joint", "shoulder"], ["elbow_joint", "elbow"],
          ["wrist_1_joint", "wrist 1"], ["wrist_2_joint", "wrist 2"], ["wrist_3_joint", "wrist 3"]]
HOME = {"shoulder_pan_joint": 88.2, "shoulder_lift_joint": -92.8, "elbow_joint": 80.2,
        "wrist_1_joint": -68.8, "wrist_2_joint": -91.7, "wrist_3_joint": -6.3}
HOME_TOOL = [15.0, -63.4, 78.5]         # where the saved UR12e description puts tool0 at the home pose, cm
LIMITS = {"MAX_STEP_CM": 20.0, "MAX_PATH_REACH_CM": 30.0, "MAX_JOINT_DEG": 30.0, "MAX_PATH_STEPS": 200}
CEILINGS = {"MAX_STEP_CM": 50.0, "MAX_PATH_REACH_CM": 80.0, "MAX_JOINT_DEG": 90.0, "MAX_PATH_STEPS": 500}
LABELS = {"MAX_STEP_CM": "One move at most (cm)", "MAX_PATH_REACH_CM": "A path stays within (cm) of its start",
          "MAX_JOINT_DEG": "One joint turn at most (degrees)", "MAX_PATH_STEPS": "Points in one path at most"}
SPEED_CM_S, JOINT_DEG_S = 2.5, 3.5
NUM = {w: i for i, w in enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                                   "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                                   "eighteen", "nineteen", "twenty"])}
NUM.update(thirty=30, forty=40, fifty=50)
N = r"(\d+(?:\.\d+)?|" + "|".join(NUM) + ")"

BASE_ROWS = [
    {"name": "/multiview_scan/capture", "kind": "service", "type": "std_srvs/srv/Trigger", "node": "multiview_scan",
     "means": "Take one 3D view with the camera from where the arm is now and add it to the scan.", "say": "take a scan view", "moves_robot": False, "status": "ready"},
    {"name": "/clicked_point", "kind": "topic", "type": "geometry_msgs/msg/PointStamped", "node": "inspection",
     "means": "Pick a point on the scanned surface. The system fits a flat patch around it.", "say": "pick the point at x thirty, y minus sixty, z five", "moves_robot": False, "status": "ready"},
    {"name": "/inspection/raster_path", "kind": "topic", "type": "nav_msgs/msg/Path", "node": "inspection",
     "means": "The raster path over the picked patch, in the world frame.", "say": "show me the raster and its points", "moves_robot": False, "status": "ready"},
    {"name": "/inspection/raster_plan_valid", "kind": "topic", "type": "std_msgs/msg/Bool", "node": "inspection",
     "means": "Whether MoveIt could plan the approach and raster.", "say": "is the raster plan valid", "moves_robot": False, "status": "ready"},
    {"name": "/touchdown_test/execute_raster", "kind": "service", "type": "std_srvs/srv/Trigger", "node": "touchdown_test",
     "means": "Touch down at the start of the raster and scan the whole raster with the probe pressed on the surface.", "say": "run the raster", "moves_robot": True, "status": "ready"},
    {"name": "/zivid/points", "kind": "topic", "type": "sensor_msgs/msg/PointCloud2", "node": "zivid",
     "means": "The camera's point cloud.", "say": "show the camera", "moves_robot": False, "status": "missing"},
]
MOVEIT = [("where_is_the_tool", "Where the tool is, in centimetres from the base.", "where is the tool", False, "tf2 lookup"),
          ("move_by", "Move the tool in a straight line by an amount.", "up five centimetres", True, "GetCartesianPath + ExecuteTrajectory"),
          ("move_to", "Move the tool in a straight line to a position.", "go to x forty, y minus ten, z thirty", True, "GetCartesianPath + ExecuteTrajectory"),
          ("move_joint", "Turn one joint by some degrees.", "joint two by ten degrees", True, "MoveGroup action"),
          ("go_home", "Move the whole arm to its home pose.", "go home", True, "MoveGroup action"),
          ("stop", "Halt the motion in progress.", "stop", False, "ExecuteTrajectory cancel")]


def num(s, default=None):
    if s is None:
        return default
    s = s.lower()
    if s in NUM:
        return NUM[s]
    try:
        return float(s)
    except ValueError:
        return default


class Mock:
    def __init__(self, think_s=3.5, rows=0):
        self.think_s = think_s
        self.lock = threading.Lock()
        self.phase, self.phase_since = "idle", time.time()
        self.busy = self.pending = self.held = None
        self.busy = False
        self.messages, self.records = [], []
        self.pose = dict(HOME)
        self.tool = list(HOME_TOOL)
        self.raster = None
        self.shown_path = None
        self.said, self.lit, self.trace = "", [], None
        self.follow_until = 0.0
        self.voice = {**config.VOICE_DEFAULTS, "on": True}   # the voice block, as the console keeps it
        self.model, self.think = "mock-model", False
        self.rows = [dict(r) for r in BASE_ROWS]
        for i in range(rows):
            self.rows.append({"name": f"/controller_{i % 8}/thing_{i}", "kind": "topic", "type": "std_msgs/msg/Float64", "node": f"controller_{i % 8}",
                              "means": "", "say": f"/controller_{i % 8}/thing_{i}", "moves_robot": None, "status": "undescribed"})
        for name, means, say, moves, typ in MOVEIT:
            self.rows.append({"name": name, "kind": "tool", "type": typ, "node": "moveit", "means": means, "say": say, "moves_robot": moves, "status": "ready"})
        self.last_event = None
        self._answer = threading.Event()
        self._ok = False
        self._run_id = 0
        self.voice_unit = None                # the real voice package, when --voice is given
        self.felt_ms = None

    # -- the voice package, when there is one -------------------------------------
    def attach_voice(self, voice_unit):
        self.voice_unit = voice_unit

    def voice_timings(self, felt_ms):
        self.felt_ms = felt_ms

    def heard(self, text):
        """Spoken text, handled as the console's heard() does: only "stop" skips the model."""
        text = words.strip_wake(text)
        if not self.voice["on"]:
            return "off"
        if not text:
            return "empty"
        if words.is_stop(text):
            self.stop("Stopped by voice.")
            return "stopped"
        if not self.busy:
            self.set_phase("heard")
        else:
            self._play("heard")
        if self.pending and self.voice["confirm"]:
            if words.says_yes(text) and self.answer(self.pending["id"], True):
                return "confirmed"
            rest = words.split_no(text)
            rest = text if rest is None else rest
            self.held = rest or None          # set before the answer wakes the run, which reads it
            if self.answer(self.pending["id"], False):
                return "held" if rest else "cancelled"
            self.held = None
        if self.command(text, via="voice"):
            return "sent"
        self.messages.append({"who": "robot", "text": f"Heard “{text}” but a command is running. Say it again when it has finished.",
                              "events": [], "total_s": 0})
        return "busy"

    def voice_state(self, state):
        if state in ("armed", "listening") and not self.busy and self.phase != "listening":
            self.set_phase("listening")

    def voice_idle(self):
        if self.phase == "listening" and not self.busy:    # the window closed with nothing said
            self.set_phase("idle")

    def _play(self, name):
        if self.voice_unit:
            self.voice_unit.play(name)

    def stop(self, note="Stop sent to the arm (mock)."):
        if self.pending:
            self.answer(self.pending["id"], False)
        self.stop_flag = self.phase == "moving"
        self.set_phase("stopped")
        self.say("Stopped.")
        self.messages.append({"who": "robot", "text": note, "events": [], "total_s": 0})
        return note

    # -- the loop --------------------------------------------------------------
    def set_phase(self, p):
        if p != self.phase and p in PHASE_SOUNDS:
            self._play(PHASE_SOUNDS[p])
        self.phase, self.phase_since = p, time.time()

    def say(self, text):
        self.said = text
        if self.voice_unit:
            self.voice_unit.say(text)

    def state(self):
        left = max(0.0, self.follow_until - time.time())
        if self.phase == "follow" and left == 0 and not self.busy:
            self.set_phase("idle")
        return {"status": {"ollama": True, "moveit": True, "joint_states": True, "joints": dict(self.pose),
                           "tool_cm": [round(v, 1) for v in self.tool], "hint": ""},
                "busy": self.busy, "pending": self.pending, "busy_for_s": 0, "messages": self.messages, "model": self.model,
                "think": self.think, "joint_list": JOINTS, "limits": LIMITS, "ceilings": CEILINGS,
                "labels": LABELS, "speed": 0.1, "shown_path": self.shown_path, "ollama_url": "http://localhost:11434",
                "frames": {"base": "base", "tool": "tool0"}, "local": True,
                "voice": {**(self.voice_unit.latest if self.voice_unit else {"running": False, "state": "idle", "final": "", "error": "", "felt_ms": None}), **self.voice},
                "robot": {"mode": "sim", "overall": {"lamp": "ok", "text": "ready (mock)"}, "alert": None,
                          "rungs": [{"title": "URSim answering", "detail": "mock", "lamp": "ok"}, {"title": "the driver", "detail": "mock", "lamp": "ok"}]},
                "phase": self.phase, "phase_since_s": round(time.time() - self.phase_since, 1), "follow_until_s": round(left, 1),
                "trace": self.trace, "lit": self.lit, "held": self.held, "said": self.said,
                "verbosity": self.voice["verbosity"], "persona": self.voice["persona"], "follow_up_s": self.voice["follow_up_s"],
                "confirm_timeout_s": self.voice["confirm_timeout_s"], "map_ros": self.voice["map_ros"]}

    # -- commands --------------------------------------------------------------
    def command(self, text, via=None):
        text = text.strip()
        if self.pending:
            return self.answer_text(text)
        if self.busy:
            return False
        self.messages.append({"who": "you", "text": text, **({"via": via} if via else {})})
        if self.self_question(text):
            return True
        self.busy = True
        self.follow_until, self.lit, self.trace = 0.0, [], None
        if self.voice_unit:
            self.voice_unit.followup(0)
        self._run_id += 1
        threading.Thread(target=self._run, args=(text, self._run_id, via), daemon=True).start()
        return True

    def answer_text(self, text):
        t = text.lower().strip(" .!?")
        if t in ("confirm", "go ahead"):
            return self.answer(self.pending["id"], True)
        m = re.match(r"^(no|nope|cancel)\b[,.]?\s*(?:(?:actually|rather|instead)\b[,.]?\s*)?(.*)$", t)
        rest = m.group(2).strip() if m else t
        self.answer(self.pending["id"], False)
        self.held = rest or None
        return True

    def answer(self, pid, ok):
        if self.pending and self.pending["id"] == pid:
            self._ok = ok
            self._answer.set()
            if self.voice_unit:
                self.voice_unit.followup(0)
            return True
        return False

    def self_question(self, text):
        t = text.lower().strip(" .!?")
        m = re.match(r"^what can i (?:do|say)(?: with (.+))?$", t) or re.match(r"^what(?:'s| is) connected to (.+)$", t)
        if m:
            group = (m.group(1) or "").strip()
            node = next((n for n in {r["node"] for r in self.rows} if n == group or n.replace("_", " ") == group or self.group_name(n).lower() == group), None) if group else None
            if group and not node:
                self._say_now(f"I don't know a node called {group}.")
                return True
            if node:
                offers = [r for r in self.rows if r["node"] == node and r["status"] != "undescribed"]
                sentences = [(f"{self.group_name(node)} offers {len(offers)} thing{'s' if len(offers) != 1 else ''}.", [node])]
                for r in offers:
                    sentences.append((f"{(r['means'] or r['name']).split('. ')[0].rstrip('.')}: say \"{r['say']}\"." + (" It will ask you to confirm." if r["moves_robot"] else ""), [r["name"]]))
            else:
                nodes = []
                for r in self.rows:
                    if r["status"] != "undescribed" and r["node"] not in nodes:
                        nodes.append(r["node"])
                sentences = [(f"{len(nodes)} nodes are connected: {', '.join(self.group_name(n) for n in nodes)}. Say \"what can I do with\" and a name.", nodes)]
            threading.Thread(target=self._tour, args=(sentences,), daemon=True).start()
            return True
        if re.match(r"^(?:which (?:of these )?moves? the robot|what moves the robot)$", t):
            movers = [r for r in self.rows if r["moves_robot"]]
            threading.Thread(target=self._tour, args=([(f"These move the robot and will ask you to confirm: {', '.join(r['say'] for r in movers)}.", [r["name"] for r in movers])],), daemon=True).start()
            return True
        if re.match(r"^where did (?:that|it) go$", t):
            if not self.last_event:
                self._say_now("Nothing has run yet.")
            else:
                row = next((r for r in self.rows if r["name"] == self.last_event), None)
                threading.Thread(target=self._tour, args=([(f"The last command went to {self.group_name(row['node'])}, {row['type']}.", [row["node"], row["name"]])],), daemon=True).start()
            return True
        m = re.match(r"^(show|hide) ros$", t)
        if m:
            self.voice["map_ros"] = m.group(1) == "show"
            self._say_now("ROS names shown." if self.voice["map_ros"] else "ROS names hidden.")
            return True
        return False

    def group_name(self, node):
        return {"moveit": "MoveIt"}.get(node) or node.replace("_", " ").capitalize()

    def _say_now(self, line):
        self.say(line)
        self.messages.append({"who": "robot", "text": line, "events": [], "total_s": 0})

    def _tour(self, sentences):
        for line, ids in sentences:
            self.lit = list(ids)
            self.say(line)
            time.sleep(1.8)
        self.lit = []
        self.messages.append({"who": "robot", "text": " ".join(l for l, _ in sentences), "events": [], "total_s": 0})
        self._open_follow_up()

    def _open_follow_up(self, seconds=None):
        seconds = self.voice["follow_up_s"] if seconds is None else seconds
        self.follow_until = time.time() + seconds
        if self.voice_unit and seconds > 0:
            self.voice_unit.followup(seconds)

    # -- the pretend model and robot ---------------------------------------------
    def understand(self, t):
        t = t.lower().replace(",", " ")
        m = re.search(r"\bpoint\s+" + N, t)
        if re.search(r"\b(show|see|read|display)\b.*\braster\b", t) or re.search(r"\braster\b.*\bpoints?\b", t):
            return {"kind": "raster"}
        if m:                                   # "go to point 10", or just "point 11" as a correction
            return {"kind": "point", "n": int(num(m.group(1), 0))}
        if re.search(r"\b(home|rest)\b", t) and "where" not in t:
            return {"kind": "home"}
        m = re.search(r"\b(up|down|left|right|forward|forwards|back|backwards)\b(?:\s+(?:by\s+)?" + N + ")?", t)
        if m:
            return {"kind": "by", "dir": m.group(1), "cm": num(m.group(2), 5)}
        m = re.search(r"\bjoint\s+" + N + r"\b.*?" + N + r"\s*(?:degrees?|deg)?", t)
        if m:
            return {"kind": "joint", "j": int(num(m.group(1), 1)), "deg": num(m.group(2), 10)}
        if re.search(r"\bwhere\b.*\b(tool|arm|robot|are you)\b", t):
            return {"kind": "pose"}
        if re.search(r"\bscan view\b|\btake a (scan|view)\b", t):
            return {"kind": "trigger", "name": "/multiview_scan/capture", "label": "Take a scan view", "moves": False}
        if re.search(r"\brun the raster\b|\btouch ?down\b", t):
            return {"kind": "trigger", "name": "/touchdown_test/execute_raster", "label": "Run the raster", "moves": True}
        if re.search(r"\bvalid\b", t):
            return {"kind": "valid"}
        if re.search(r"^\s*(stop|halt|freeze)\s*$", t):
            return {"kind": "stop"}
        return None

    def _run(self, text, run_id, via=None):
        self._via = via
        it = self.understand(text)
        self.set_phase("thinking")
        self.say("On it." if it and it["kind"] in ("point", "home", "by", "joint", "trigger") else "Let me check.")
        time.sleep(self.think_s)
        events, reply = [], ""
        if not it:
            nodes = sorted({self.group_name(r["node"]) for r in self.rows if r["status"] == "ready"})
            reply = f"I don't know how to do that yet. The connected nodes are {', '.join(nodes)}."
            self._finish(text, events, reply, "refused", say=reply)
            return
        kind = it["kind"]
        if kind == "raster":
            self.raster = self._make_raster()
            self.shown_path = {"topic": "/inspection/raster_path", "points_cm": [p for p in self.raster]}
            self.trace = {"tool": "/inspection/raster_path"}
            events.append(self._event("read_topic", {"topic": "/inspection/raster_path"}, "done", f"Read /inspection/raster_path: a path of {len(self.raster)} points; drawn on the view."))
            reply = f"{len(self.raster)} points, the plan is valid. They're on the screen. Say a point number, or \"run the raster\"."
            self._finish(text, events, reply, "done", say=reply, follow=True)
            return
        if kind == "pose":
            reply = f"The tool is at x={self.tool[0]:.1f}, y={self.tool[1]:.1f}, z={self.tool[2]:.1f} cm."
            events.append(self._event("where_is_the_tool", {}, "done", reply))
            self._finish(text, events, reply, "done", say=f"Tool at x {round(self.tool[0])}, y {round(self.tool[1])}, z {round(self.tool[2])} centimetres.")
            return
        if kind == "valid":
            reply = "The raster plan is valid." if self.raster else "No raster has been read yet."
            events.append(self._event("read_topic", {"topic": "/inspection/raster_plan_valid"}, "done", reply))
            self._finish(text, events, reply, "done", say=reply)
            return
        if kind == "stop":
            self._finish(text, [], "Nothing to stop: the arm was not moving.", "stopped", say="Nothing to stop.")
            return
        if kind == "trigger":
            if it["moves"]:
                ok = self._ask(f"call {it['name']}: {it['label']}", {"facts": {"kind": "trigger", "label": it["label"], "name": it["name"]}}, f"{it['label']}. This moves the robot. Confirm?")
                if not ok:
                    events.append(self._event("call_trigger", {"service": it["name"]}, "cancelled", "Cancelled by the operator. Nothing was done."))
                    self._finish(text, events, "Cancelled.", "cancelled")
                    return
                self.set_phase("moving"); time.sleep(2)
            events.append(self._event("call_trigger", {"service": it["name"]}, "done", f"Called {it['name']}: success."))
            self.last_event = it["name"]
            self._finish(text, events, f"Done: {it['label'].lower()}.", "done", say=f"Done: {it['label']}.", follow=it["moves"])
            return
        # a move: plan, propose, confirm, move
        if kind == "point":
            if not self.raster or not 1 <= it["n"] <= len(self.raster):
                reply = f"The raster has {len(self.raster) if self.raster else 0} points; there is no point {it['n']}." if self.raster else "I haven't read a raster yet. Say \"show me the raster\" first."
                self._finish(text, [], reply, "refused", say=reply)
                return
            plan = self._plan_line(self.raster[it["n"] - 1], f"Point {it['n']}", "move_to")
        elif kind == "home":
            plan = self._plan_joints(dict(HOME), "Home position", "go_home")
        elif kind == "by":
            v = {"up": (0, 0, 1), "down": (0, 0, -1), "left": (0, 1, 0), "right": (0, -1, 0), "forward": (1, 0, 0), "forwards": (1, 0, 0), "back": (-1, 0, 0), "backwards": (-1, 0, 0)}[it["dir"]]
            plan = self._plan_line([self.tool[i] + v[i] * it["cm"] for i in range(3)], f"{it['dir'].capitalize()} {it['cm']:g} cm", "move_by")
        else:
            if abs(it["deg"]) > LIMITS["MAX_JOINT_DEG"] or not 1 <= it["j"] <= 6:
                reply = f"REFUSED by safety layer: {it['deg']:g} deg is more than the {LIMITS['MAX_JOINT_DEG']:g} deg limit for one move. Nothing was moved."
                self._finish(text, [self._event("move_joint", {"joint": it["j"], "degrees": it["deg"]}, "refused", reply)], reply, "refused", say=reply, follow=True)
                return
            target = dict(self.pose); name = JOINTS[it["j"] - 1][0]; target[name] += it["deg"]
            plan = self._plan_joints(target, f"Joint {it['j']} by {it['deg']:g}°", "move_joint")
        if plan.get("refused"):
            self._finish(text, [self._event(plan["tool"], {}, "refused", plan["refused"])], plan["refused"], "refused", say=plan["refused"], follow=True)
            return
        self.trace = {"tool": plan["tool"]}
        facts = plan["facts"]
        line = (f"{facts['label']}. {facts['distance_cm']:g} centimetres, {facts['seconds']:g} seconds, path clear. Confirm?" if facts["kind"] == "line"
                else f"{facts['label']}. {facts['largest_turn_deg']:g} degrees, {facts['seconds']:g} seconds, clear. Confirm?")
        if self.held_changed:
            line = f"{facts['label']} instead. " + line.split(". ", 1)[1]
            self.held_changed = False
        ok = self._ask(plan["description"], plan["preview"], line)
        if not ok:
            events.append(self._event(plan["tool"], {}, "cancelled", "Cancelled by the operator. Nothing was moved."))
            self._finish(text, events, "Cancelled.", "cancelled")
            return
        self.set_phase("moving")
        t0, T = time.time(), facts["seconds"]
        path = plan["path"]
        stopped = False
        while time.time() - t0 < T:
            if self.stop_flag:
                stopped = True
                break
            k = (time.time() - t0) / T
            frame = path[min(len(path) - 1, int(k * (len(path) - 1)))]
            self.pose = {n: frame[i] for i, (n, _) in enumerate(JOINTS)}
            if facts["kind"] == "line":
                self.tool = [facts["from_cm"][i] + (facts["to_cm"][i] - facts["from_cm"][i]) * k for i in range(3)]
            time.sleep(0.05)
        self.stop_flag = False
        if stopped:
            reply = f"Stopped part-way by the operator. The tool is at {[round(v, 1) for v in self.tool]} cm."
            events.append(self._event(plan["tool"], {}, "stopped", reply))
            self._finish(text, events, reply, "stopped", say=f"Stopped at x {round(self.tool[0])}, y {round(self.tool[1])}, z {round(self.tool[2])}.", follow=True)
            return
        self.pose = {n: path[-1][i] for i, (n, _) in enumerate(JOINTS)}
        if facts["kind"] == "line":
            self.tool = list(facts["to_cm"])
        reply = f"Done. The tool is now at x={self.tool[0]:.1f}, y={self.tool[1]:.1f}, z={self.tool[2]:.1f} cm."
        events.append(self._event(plan["tool"], {}, "done", reply))
        self.last_event = plan["tool"]
        self._finish(text, events, reply, "done", say=f"At {facts['label']}.", follow=True)

    stop_flag = False
    held_changed = False
    _via = None

    def _ask(self, description, preview, line):
        self._answer.clear()
        self.pending = {"id": str(time.time()), "text": description, "preview": preview, "facts": preview.get("facts")}
        self.set_phase("proposing")
        self.say(line)
        self._open_follow_up(self.voice["confirm_timeout_s"])
        answered = self._answer.wait(timeout=self.voice["confirm_timeout_s"])
        self.pending = None
        if not answered:
            self.set_phase("cancelled")
            self.say(f"{preview.get('facts', {}).get('label', 'That')} cancelled.")
            return False
        if self._ok:
            self.set_phase("moving")
        else:
            self.set_phase("cancelled")
            self.say("Cancelled.")
        return self._ok

    def _finish(self, text, events, reply, end, say=None, follow=False):
        self.set_phase(end)
        if say:
            self.say(say)
        self.messages.append({"who": "robot", "text": reply, "events": events, "total_s": round(self.think_s, 1), "command": text})
        self.records.append({"time": time.strftime("%H:%M:%S"), "model": self.model, "think": False, "command": text, "via": self._via or "typed",
                             "events": events, "total_s": round(self.think_s, 1), "reply": reply, "phases": [], "timings": {}})
        self.busy = False
        self.trace = None
        if self.held:
            held, self.held = self.held, None
            self.held_changed = True
            self.command(held, via=self._via)
            return
        if follow or end == "cancelled":
            self.set_phase("follow")
            self._open_follow_up()

    def _event(self, tool, args, outcome, message):
        return {"tool": tool, "args": args, "outcome": outcome, "message": message, "after_s": round(self.think_s, 1)}

    def _make_raster(self):
        c = [HOME_TOOL[0], HOME_TOOL[1], HOME_TOOL[2] - 14]
        pts = []
        for row in range(4):
            for i in range(6):
                col = 5 - i if row % 2 else i
                pts.append([round(c[0] - 7.5 + col * 3, 1), round(c[1] - 4.5 + row * 3, 1), round(c[2], 1)])
        return pts

    def _plan_line(self, target, label, tool):
        d = math.dist(self.tool, target)
        if d > LIMITS["MAX_STEP_CM"]:
            return {"refused": f"REFUSED by safety layer: that move is {d:.1f} cm, more than the {LIMITS['MAX_STEP_CM']:g} cm limit for one move. Nothing was moved.", "tool": tool}
        # The ghost: a small bend of the shoulder and elbow in proportion to the move (no inverse kinematics here).
        end = dict(self.pose)
        end["shoulder_lift_joint"] += (target[2] - self.tool[2]) * 0.8
        end["elbow_joint"] -= (target[2] - self.tool[2]) * 0.4 - (target[0] - self.tool[0]) * 0.3
        end["shoulder_pan_joint"] += (target[1] - self.tool[1]) * 0.3
        return self._plan(end, label, tool, {"kind": "line", "label": label, "from_cm": [round(v, 1) for v in self.tool], "to_cm": [round(v, 1) for v in target],
                                              "distance_cm": round(d, 1), "seconds": max(1, round(d / SPEED_CM_S))}, [list(self.tool), list(target)])

    def _plan_joints(self, target, label, tool):
        turn = max(abs(target[n] - self.pose[n]) for n, _ in JOINTS)
        return self._plan(target, label, tool, {"kind": "joint", "label": label, "largest_turn_deg": round(turn, 1), "seconds": max(1, round(turn / JOINT_DEG_S))}, None)

    def _plan(self, end, label, tool, facts, points_cm):
        names = [n for n, _ in JOINTS]
        path = [[self.pose[n] + (end[n] - self.pose[n]) * k / 30 for n in names] for k in range(31)]
        return {"tool": tool, "facts": facts, "path": path, "description": f"{label} ({facts['seconds']} s)",
                "preview": {"joints": names, "path_deg": path, "seconds": facts["seconds"], "points_cm": points_cm, "facts": facts}}


def create_app(think_s=3.5, rows=0):
    app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")
    mock = Mock(think_s=think_s, rows=rows)
    app.config["mock"] = mock

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "stage.html")

    @app.get("/pkg/<package>/<path:rel>")
    def mesh(package, rel):
        return send_from_directory(MESHES / package, rel)

    @app.get("/state")
    def state():
        return jsonify(mock.state())

    @app.get("/capabilities")
    def capabilities():
        return jsonify(system="A mock inspection cell (tests/mock_console.py).", interfaces=mock.rows, moveit=True,
                       file="~/.config/rosa_ws/interfaces.yaml (mock)", allow_unlisted_writes=False)

    @app.post("/capabilities/rescan")
    def rescan():
        return capabilities()

    @app.get("/cell")
    def cell():
        return jsonify(shapes=[])

    @app.post("/command")
    def command():
        text = (request.json or {}).get("text", "").strip()
        if not text:
            return jsonify(error="Type a command first."), 400
        if not mock.command(text):
            return jsonify(error="Wait for the current command to finish."), 409
        return jsonify(ok=True)

    @app.post("/confirm")
    def confirm():
        body = request.json or {}
        return jsonify(ok=mock.answer(body.get("id"), bool(body.get("ok"))))

    @app.post("/stop")
    def stop():
        return jsonify(ok=True, message=mock.stop())

    @app.post("/cancel")
    def cancel():
        return jsonify(ok=True)

    @app.post("/reset")
    def reset():
        mock.messages.clear(); mock.shown_path = None; mock.raster = None
        return jsonify(ok=True)

    @app.post("/finish")
    def finish():
        return jsonify(path="(mock: nothing written)", records=mock.records)

    @app.get("/models")
    def models():
        return jsonify(models=[{"name": "mock-model", "tools": True, "thinks": False}, {"name": "mock-thinker", "tools": True, "thinks": True}], current=mock.model)

    @app.post("/model")
    def model():
        mock.model = (request.json or {}).get("name", mock.model)
        return jsonify(ok=True)

    @app.post("/think")
    def think():
        mock.think = bool((request.json or {}).get("on"))
        return jsonify(ok=True)

    @app.post("/limits")
    def limits():
        for k, v in (request.json or {}).items():
            if k in LIMITS and isinstance(v, (int, float)) and 0 < v <= CEILINGS[k]:
                LIMITS[k] = v
        return jsonify(ok=True)

    @app.post("/ollama")
    def ollama():
        return jsonify(ok=True, url=(request.json or {}).get("url", ""), models=["mock-model"])

    @app.post("/robot/mode")
    def robot_mode():
        return jsonify(mock.state()["robot"])

    @app.post("/robot/action")
    def robot_action():
        return jsonify(ok=True, message="mock", robot=mock.state()["robot"])

    @app.post("/voice/on")
    def voice_on():
        mock.voice["on"] = bool((request.json or {}).get("on"))
        if mock.voice_unit:
            mock.voice_unit.set_wake(mock.voice["on"])
        return jsonify(ok=True, on=mock.voice["on"])

    @app.post("/voice/confirm")
    def voice_confirm():
        mock.voice["confirm"] = bool((request.json or {}).get("on"))
        return jsonify(ok=True, confirm=mock.voice["confirm"])

    @app.post("/voice/ptt")
    def voice_ptt():
        if mock.voice_unit:
            mock.voice_unit.ptt(bool((request.json or {}).get("down")))
        return jsonify(ok=True)

    @app.post("/voice/say")
    def voice_say():
        if not mock.voice_unit:
            return jsonify(spoken=False, reason="voice is not running (mock: start me with --voice)")
        spoken = mock.voice_unit.say(str((request.json or {}).get("text", "")))
        return jsonify(spoken=spoken, reason=mock.voice_unit.latest.get("speech_reason", ""))

    @app.get("/voice/devices")
    def voice_devices():
        if not mock.voice_unit:
            return jsonify(input=[], output=[], voices=[])
        return jsonify(**mock.voice_unit.latest.get("devices", {"input": [], "output": []}), voices=mock.voice_unit.voices())

    @app.post("/settings/voice")
    def settings_voice():
        body = request.json or {}
        problem = config.voice_settings_ok(body)
        if problem:
            return jsonify(error=problem), 400
        mock.voice.update(body)
        if mock.voice_unit:
            mock.voice_unit.update(body)
        return jsonify(ok=True, voice=mock.voice)

    return app


class LoopingFile:
    """A WAV played at real speed, over and over: the microphone for page work without one."""

    kind = "file"

    def __init__(self, path):
        self.path, self._running, self._src = path, False, None

    def start(self):
        self._running = True

    def stop(self):
        self._running = False
        if self._src:
            self._src.stop()

    def backlog(self):
        return 0

    def frames(self):
        from voice.audio import FileSource
        while self._running:
            self._src = FileSource(self.path, realtime=True, pad_end_s=1.0)
            self._src.start()
            yield from self._src.frames()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="A scripted console for page work on the Mac.")
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--rows", type=int, default=0, help="pad the capability list with this many undescribed interfaces")
    ap.add_argument("--think", type=float, default=3.5, help="the pretend model's think time, seconds")
    ap.add_argument("--voice", action="store_true", help="run the voice package on the microphone")
    ap.add_argument("--voice-file", metavar="WAV", help="run the voice package on this clip, looped")
    args = ap.parse_args()
    app = create_app(think_s=args.think, rows=args.rows)
    mock = app.config["mock"]
    if args.voice or args.voice_file:
        from voice import Voice
        unit = Voice(mock.voice, on_state=mock.voice_state, on_final=mock.heard, on_timings=mock.voice_timings,
                     on_idle=mock.voice_idle, on_error=lambda m: print("voice:", m),
                     make_source=(lambda s: LoopingFile(args.voice_file)) if args.voice_file else None)
        mock.attach_voice(unit)
        unit.start()
    print(f"Mock console on http://localhost:{args.port}/. Nothing here is real"
          + (", except the voice package (microphone)." if args.voice else ", except the voice package on a clip." if args.voice_file else "."))
    app.run(host="127.0.0.1", port=args.port, threaded=True)
