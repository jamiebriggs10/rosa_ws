"""Web console: a browser page that replaces the terminal for the ROSA agent.

Runs on the robot's computer; open http://localhost:8000 there, or its address from any machine.

One Flask app, one page (static/stage.html), and these JSON endpoints:
  GET  /state    everything the page shows, polled once a second
  GET  /models   the models Ollama has, and which of them can call tools
  GET  /cell     MoveIt's model of the cell (walls, table, keep-out zones) for the 3D view
  POST /model    {"name": ...}  switch to another model (fresh conversation)
  POST /think    {"on": true|false}  let thinking models reason first (slower)
  POST /command  {"text": ...}  send a command to the model
  POST /cancel   stop the model thinking about the current command
  POST /reset    start a new conversation: the model forgets earlier messages
  POST /limits   {"MAX_STEP_CM": ..., ...}  change the safety limits, within config's ceilings
  POST /confirm  {"id": ..., "ok": true|false}  answer the operator gate
  POST /stop     cancel the motion in progress, without the model
  POST /finish   write the session file and return the table
  GET  /capabilities         what the robot's system offers (discovery + the description file)
  POST /capabilities/rescan  read the graph and the description file again
  POST /ollama   {"url": ...}  test an Ollama address, save it, and list its models
  POST /robot/mode    {"mode": "sim"|"real"}  which robot the connection panel checks; saved, returns the ladder
  POST /robot/action  {"name": ...}  one step of the connection ladder (bring the profile up, start the
                      driver, power on; 404 unknown, 403 real-robot steps from another computer, 409 failed)
  POST /voice/on       {"on": true|false}  the voice switch: spoken words become commands (saved)
  POST /voice/confirm  {"on": true|false}  let a spoken "confirm" / "no" answer the confirm card (saved)
  POST /voice/ptt      {"down": true|false}  Hold to talk
  POST /voice/say      {"text": ...}  Rosa says it (the Test voice button)
  GET  /voice/devices  microphones, speakers and Piper voices for the Settings menus
  POST /settings/voice {key: value, ...}  any key of the voice block, checked and saved
"""
import datetime
import json
import pathlib
import re
import threading
import time
import urllib.request

from flask import Flask, jsonify, request, send_from_directory
from langchain_core.messages import HumanMessage

import config
import generic_tools
import lines
import robot_link
import robot_tools
import voice
from voice import words

# The loop: one phase at a time, each shown by the page as a lamp and (later) a sound and a line.
PHASES = ("idle", "listening", "heard", "thinking", "proposing", "moving", "done", "refused", "cancelled",
          "stopped", "follow")
END_PHASE = {"done": "done", "no_motion": "done", "refused": "refused", "stopped": "stopped",
             "cancelled": "cancelled", "failed": "refused"}
SESSIONS_DIR = pathlib.Path(__file__).parent / "sessions"
MOVE_TOOLS = {"move_by", "move_to", "move_path", "move_formula", "move_joint", "go_home"}   # the tools that move the arm

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
        self._via = None                      # where the running command came from ("voice"), for its record
        # The loop (see PHASES). Each change is timestamped into the running command's record.
        self.phase, self.phase_since = "idle", time.time()
        self._phases = []
        self.follow_until = 0.0               # until when voice listens without the wake word (0: closed)
        self.held = None                      # a correction kept for after the current command ("no, point eleven")
        self.declined_run = False             # the operator said no once in this run: no more cards for it
        self.lit = []                         # map nodes a spoken tour is pointing at right now
        self.said = ""                        # the last line Rosa was asked to say
        self.timed_out = None                 # the label of a card nobody answered, for the spoken line
        self._last_label = None               # what the last card proposed, for the held correction's context
        self._changed = False                 # this command is a held correction: the proposal says "instead"
        self._felt_ms = None                  # the felt delay of the spoken command running now
        self._tour_id = 0                     # a new question or command interrupts a running tour
        self._last_event = None               # the last tool event, for "where did that go"
        self._stop_said = False               # stop() has already announced this stop
        self._later = None                    # the timer that returns a finished phase to idle
        robot_tools.set_confirm_handler(self.ask_operator)

    # -- the loop --------------------------------------------------------------
    def set_phase(self, name):
        """Move the loop on. The record of the running command keeps every change with its time."""
        self.phase, self.phase_since = name, time.time()
        self._phases.append([name, self.phase_since])

    def _announce(self, phase, **facts):
        """Every phase has a sound; by verbosity, a spoken line too. The words come from lines.py and
        the facts the console measured. All of it goes to the voice package and never blocks."""
        v, persona = config.VOICE["verbosity"], config.VOICE["persona"]
        sound, kind, level = {
            "listening": ("listen", None, 9), "heard": ("heard", None, 9), "thinking": ("think", "preface", 2),
            "proposing": ("ask", "propose", 1), "moving": ("go", None, 9), "done": ("done", "done", 1),
            "refused": ("refused", "refused", 1), "cancelled": ("cancel", "cancel", 2), "stopped": ("cancel", "stopped", 1),
        }.get(phase, (None, None, 9))
        if sound:
            voice_unit.play(sound)
        kind = facts.pop("kind", kind)
        if not kind or v < facts.pop("level", level):
            return
        try:
            text = lines.line(kind, persona, **facts)
        except (KeyError, IndexError):
            return
        self.said = text
        voice_unit.say(text)

    def _open_follow_up(self, seconds=None):
        """Rosa has just addressed the operator: for a while, no wake word is needed."""
        seconds = config.VOICE["follow_up_s"] if seconds is None else seconds
        if seconds <= 0:
            return
        self.follow_until = time.time() + seconds
        voice_unit.followup(seconds)

    def voice_timings(self, felt_ms):
        """The felt delay of the utterance voice just finished: kept with that command's record."""
        self._felt_ms = felt_ms
        if self.records and self.records[-1].get("via") == "voice" and "felt_ms" not in self.records[-1] \
                and not self.busy:
            self.records[-1]["felt_ms"] = felt_ms

    def _rest(self, after_s=None):
        """A finished phase (done, refused, cancelled, stopped) returns to idle after a while,
        unless something else has started by then."""
        if self._later:
            self._later.cancel()
        ending = self.phase

        def back():
            if self.phase == ending and not self.busy and not self.pending:
                self.set_phase("idle")
        self._later = threading.Timer(config.VOICE["follow_up_s"] if after_s is None else after_s, back)
        self._later.daemon = True
        self._later.start()

    # -- the operator gate ---------------------------------------------------
    def ask_operator(self, description, preview=None):
        """Called from the agent thread by a tool. Blocks until the page answers, or the card
        times out (config.VOICE["confirm_timeout_s"]), which counts as no."""
        facts = (preview or {}).get("facts") if isinstance(preview, dict) else None
        self._last_label = (facts or {}).get("label") or description[:40]
        if self.declined_run:                 # one "no" covers the whole run: no second card for the same command
            return False
        self._answered.clear()
        self.pending = {"id": str(time.time()), "text": description, "preview": preview, "facts": facts}
        self.set_phase("proposing")
        self._announce("proposing", **self._proposal_facts(facts, description))
        self._open_follow_up(config.VOICE["confirm_timeout_s"])
        answered = self._answered.wait(timeout=config.VOICE["confirm_timeout_s"])
        self.pending = None
        if not answered:
            self.timed_out = self._last_label
            self.declined_run = True
            self.set_phase("cancelled")
            self._announce("cancelled", kind="timeout", level=1, label=self._last_label)
        return bool(answered and self._answer)

    def _proposal_facts(self, facts, description):
        """The spoken proposal's kind and facts from a card's facts dict."""
        f = dict(facts or {})
        kind = f.pop("kind", None)
        label = f.get("label") or description[:40]
        if kind == "trigger":
            return {"kind": "propose_trigger", "label": label}
        if kind == "joint":
            return {"kind": "propose_joint", "label": label, "largest_turn_deg": f.get("largest_turn_deg", 0),
                    "seconds": f.get("seconds", 0)}
        return {"kind": "changed" if self._changed else "propose", "label": label,
                "distance_cm": f.get("distance_cm", 0), "seconds": f.get("seconds", 0)}

    def answer(self, confirmation_id, ok):
        if self.pending and self.pending["id"] == confirmation_id:
            self._answer = ok
            self.set_phase("moving" if ok else "cancelled")
            self._announce("moving" if ok else "cancelled")
            voice_unit.followup(0)            # answered (by a tap or a word): the card's window closes
            if not ok:
                self.declined_run = True
            self._answered.set()
            return True
        return False

    # -- commands ------------------------------------------------------------
    def submit(self, text, via=None, shown=None):
        """Start a command for the model. Returns False if one is already running.
        `via` says where the words came from ("voice"); typed text has none. `shown` is what
        the chat and the session table show when it differs from what the model is sent."""
        if not self.busy and self.self_question(text, via):
            return True
        if via != "voice" and words.is_stop(text) and not self.busy:
            self.messages.append({"who": "you", "text": text})     # a typed "stop" is the red button, not a question for the model
            self.stop()
            return True
        return self._start(shown if shown is not None else text, self._ask_model, text, via)

    # -- questions about the console itself: the map's tour, answered without the model --------
    def self_question(self, text, via=None):
        """Answer "what can I do with X", "which of these move the robot", "where did that go" and
        "show/hide ros" here: spoken, with the map lit. True when handled."""
        for kind, pattern in SELF_QUESTIONS:
            m = pattern.match((text or "").strip())
            if m:
                break
        else:
            return False
        self._tour_id += 1
        try:
            rows, _ = capability_rows_with_moveit()
        except Exception as exc:
            self._say_one(f"I cannot read what the system offers: {exc}", text, via)
            return True
        persona = config.VOICE["persona"]
        sentences = []
        if kind == "tour":
            group = find_group(rows, m.group("g")) if m.group("g") else None
            if m.group("g") and group is None:
                self._say_one(f"I don't know a node called {m.group('g')}. Say \"what can I do\" for the list.", text, via)
                return True
            if group is None:
                nodes = []
                for r in rows:
                    if r["status"] != "undescribed" and r["node"] not in nodes:
                        nodes.append(r["node"])
                sentences.append((lines.line("tour_all", persona, count=len(nodes), groups=[group_name(n) for n in nodes]), nodes))
            else:
                offers = [r for r in rows if r["node"] == group and r["status"] != "undescribed"] or \
                         [r for r in rows if r["node"] == group]
                sentences.append((lines.line("tour_intro", persona, group=group_name(group), count=len(offers)), [group]))
                for r in offers:
                    name = (r.get("means") or r["name"]).split(". ")[0].rstrip(".")
                    sentences.append((lines.line("tour_item", persona, name=name, say=r["say"], moves=bool(r.get("moves_robot"))), [r["name"]]))
        elif kind == "movers":
            movers = [r for r in rows if r.get("moves_robot")]
            sentences.append((lines.line("movers", persona, names=[r["say"] for r in movers]), [r["name"] for r in movers]))
        elif kind == "trace":
            last = self._last_event
            if not last:
                sentences.append(("Nothing has run yet.", []))
            else:
                row = next((r for r in rows if r["name"] == last.get("tool") or r["name"] == last.get("args", {}).get("service")
                            or r["name"] == last.get("args", {}).get("topic")), None)
                if row:
                    sentences.append((lines.line("trace", persona, group=group_name(row["node"]), ros=row["type"] or row["name"]), [row["node"], row["name"]]))
                else:
                    sentences.append((f"The last command used {last.get('tool')}.", []))
        elif kind == "ros":
            on = m.group("on").lower() == "show"
            config.save_settings({"voice": {"map_ros": on}})
            sentences.append(("ROS names shown." if on else "ROS names hidden.", []))
        self._record_question(text, via, sentences[0][0] if sentences else "")
        threading.Thread(target=self._tour, args=(sentences, self._tour_id), daemon=True).start()
        return True

    def _say_one(self, line, text, via):
        self._record_question(text, via, line)
        self.said = line
        voice_unit.say(line)
        self.messages.append({"who": "robot", "text": line, "events": [], "total_s": 0})

    def _record_question(self, text, via, reply):
        message = {"who": "you", "text": text}
        if via:
            message["via"] = via
        self.messages.append(message)
        self.records.append({"time": datetime.datetime.now().strftime("%H:%M:%S"), "model": config.MODEL,
                             "think": config.THINK, "command": text, "via": via or "typed", "events": [],
                             "total_s": 0, "reply": reply, "phases": []})

    def _tour(self, sentences, tour_id):
        """Speak the sentences one by one, lighting the map nodes each is about. Stops when a
        command starts or another question is asked."""
        said = []
        for line, ids in sentences:
            if self.busy or tour_id != self._tour_id:
                break
            self.lit, self.said = list(ids), line
            voice_unit.say(line)
            said.append(line)
            self._wait_said()
        if tour_id == self._tour_id:
            self.lit = []
            self.messages.append({"who": "robot", "text": " ".join(said), "events": [], "total_s": 0})
            self._open_follow_up()

    def _wait_said(self):
        """Wait for Rosa to finish the line: voice reports speaking on and off; without it,
        a fixed 1.8 s per sentence."""
        latest = voice_unit.latest
        t0 = time.time()
        if not latest.get("running"):
            time.sleep(1.8)
            return
        while time.time() - t0 < 0.5 and not latest.get("speaking"):
            time.sleep(0.05)
        if not latest.get("speaking"):
            time.sleep(max(0.0, 1.8 - (time.time() - t0)))
            return
        while latest.get("speaking") and time.time() - t0 < 20:
            time.sleep(0.05)

    def _start(self, shown_text, work, command, via=None):
        with self._lock:
            if self.busy:
                return False
            self.busy, self.busy_since = True, time.time()
        self._phases, self.declined_run, self.follow_until, self.lit, self.timed_out = [], False, 0.0, [], None
        self._tour_id += 1                    # a tour still talking stops here
        voice_unit.followup(0)                # and voice's follow-up window, if one was open, closes
        self.set_phase("thinking")
        self._announce("thinking")
        message = {"who": "you", "text": shown_text}
        if via:
            message["via"] = via
        self._via = via                       # so a command stopped half-way is still recorded as spoken
        self.messages.append(message)
        thread = threading.Thread(target=self._run, args=(shown_text, work, command, via), daemon=True)
        self._current = thread
        thread.start()
        return True

    def _ask_model(self, text):
        if self._agent is None:
            self._agent = self._agent_factory()
        return self._agent.invoke(text)

    def _call_tool(self, tool, args):
        return getattr(robot_tools, tool).invoke(args)

    def _run(self, shown_text, work, command, via=None):
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
        self._remember_moves(events)
        total = round(time.time() - t0, 1)
        shown_reply = tidy_reply(reply, events)
        self.messages.append({"who": "robot", "text": shown_reply, "events": events, "total_s": total,
                              "command": command})
        end = END_PHASE.get(events[-1]["outcome"], "done") if events else "done"
        if events:
            self._last_event = events[-1]
        self.set_phase(end)
        self._announce_end(end, events, shown_reply)
        record = {"time": datetime.datetime.now().strftime("%H:%M:%S"), "model": config.MODEL,
                  "think": config.THINK, "command": shown_text, "via": via or "typed", "events": events,
                  "total_s": total, "reply": reply, "phases": list(self._phases)}
        if via == "voice" and self._felt_ms is not None:
            record["felt_ms"], self._felt_ms = self._felt_ms, None
        self.records.append(record)
        self._current = None
        self.busy = False
        self._changed = False
        moved = any(e["tool"] in MOVE_TOOLS or e["outcome"] in ("stopped", "refused") for e in events)
        asked = str(reply).strip().endswith("?")   # the model wants something cleared up
        if (end in ("done", "refused", "stopped") and moved) or asked:
            self.set_phase("follow")          # Rosa has just reported or asked: the operator can answer without the wake word
            self._open_follow_up()
        elif end == "cancelled":
            self.set_phase("follow")
            self._open_follow_up()
        else:
            self._rest(3)
        if self.held:
            # A correction said over a card ("no, point eleven"): now that the model has finished
            # with the declined move, the rest runs as its own command, with one line of context.
            text, self.held = self.held, None
            label = self._last_label or "that"
            if self.submit(f'(The operator declined "{label}" and said instead:) {text}', via=via, shown=text):
                self._changed = True

    def _announce_end(self, end, events, reply):
        """The sound and the line for how a command ended, from what the tools recorded."""
        last = events[-1] if events else None
        if end == "cancelled":
            return                            # already announced when the card was answered or timed out
        if end == "stopped" and self._stop_said:
            self._stop_said = False
            return                            # the stop button or the spoken stop already said it
        if end == "refused":
            self._announce("refused", reason=(last or {}).get("message", "Refused."))
        elif end == "stopped":
            self._announce("stopped", kind="result", text=(last or {}).get("message", "Stopped."))
        elif last and last["tool"] in MOVE_TOOLS:
            self._announce("done", label=self._last_label or "there")
        elif last and last["tool"] == "call_trigger":
            self._announce("done", kind="done_trigger", label=self._last_label or last["args"].get("service", "done"))
        else:
            self._announce("done", kind="result", text=reply)

    def _remember_moves(self, events):
        """Add an exact record of what was done (tools, numbers or formulas, outcomes, positions)
        to the model's memory. Its own reply rarely says this, and without it the model cannot
        refer back to an earlier move, for example to retrace it.

        The record goes in as a note from the console, never as part of the model's own
        words: small models copy whatever they appear to have said before, and were seen
        writing fake "record" lines claiming moves that never happened."""
        history = getattr(self._agent, "chat_history", None)
        moves = [e for e in events if e["tool"] != "stop"]                 # not the stop button
        if history is None or not moves:
            return
        lines = []
        for e in moves:
            line = f"{e['tool']} {json.dumps(e['args'])[:800]} -> {e['outcome']}"
            if e.get("start_cm"):
                line += f", starting from {e['start_cm']} cm"
            lines.append(line + f". Result: {e['message']}")
        history.append(HumanMessage(content=(
            f"(Console note, not from the operator. For your memory only; never repeat or imitate it. "
            f"What actually happened, you called " + " | ".join(lines) + ")")))

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
        self.set_phase("cancelled")
        self.records.append({"time": datetime.datetime.now().strftime("%H:%M:%S"), "model": config.MODEL,
                             "think": config.THINK, "command": self.messages[-1]["text"], "via": self._via or "typed",
                             "events": list(robot_tools.events), "total_s": waited, "reply": "(stopped by the operator)",
                             "phases": list(self._phases)})
        self._current = None
        self._agent = None                    # the old one is still busy; start the next command afresh
        self._stop_said = False               # the abandoned command's ending never comes to clear it
        if self.pending:
            self.answer(self.pending["id"], False)
        self._rest(3)
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

    def stop(self, note=None):
        """The red button: cancel a pending card and the motion, bypassing the model.
        `note` is put in front of the message when something other than the button asked."""
        if self.pending:                      # a card still showing is declined quietly: the stop is the news
            self._answer, self.declined_run = False, True
            self._answered.set()
            voice_unit.followup(0)
        message = robot_tools.stop.invoke({})
        self.set_phase("stopped")
        tool = (_cached_status() or {}).get("tool_cm")
        if tool:
            self._announce("stopped", x=round(tool[0]), y=round(tool[1]), z=round(tool[2]))
        else:
            self._announce("stopped", kind="result", text="Stopped.")
        self._stop_said = self.busy           # the command's own ending must not say it again
        if not self.busy:
            self._rest(3)
        self.messages.append({"who": "robot", "text": f"{note} {message}" if note else message, "events": [], "total_s": 0})
        return message

    # -- voice -----------------------------------------------------------------
    def heard(self, text):
        """Spoken text from the voice package. Works out what it means and returns one word saying
        what was done, which the pop-up shows: off, empty, stopped, confirmed, cancelled, held, sent
        or busy. Only "stop" bypasses the model; every other command goes the way typed text goes,
        through submit() and the tools' checks."""
        text = words.strip_wake(text)                          # "Truffles, confirm" is a confirm
        if not config.VOICE["on"]:
            return "off"
        if not text:
            return "empty"
        if words.is_stop(text):
            try:
                self.stop(note="Stopped by voice.")
            except Exception as exc:           # no robot, or the stop topic is missing
                self.messages.append({"who": "robot", "text": f"Stopped by voice, but the stop failed: {exc}",
                                      "events": [], "total_s": 0})
            return "stopped"
        if not self.busy:
            self.set_phase("heard")
        self._announce("heard")
        pending = self.pending
        if pending and config.VOICE["confirm"]:
            if words.says_yes(text) and self.answer(pending["id"], True):
                return "confirmed"
            rest = words.split_no(text)
            if rest is None:
                rest = text                   # anything that is not a yes cancels the card and is a new command
            self.held = rest or None          # set before the answer wakes the run, which reads it
            if self.answer(pending["id"], False):
                return "held" if rest else "cancelled"
            self.held = None
        if self.submit(text, via="voice"):
            return "sent"
        if self.pending:
            note = (f"Heard “{text}”, but a confirm card is waiting. Answer it on screen, or turn voice "
                    "confirm on in the settings.")
        else:
            note = f"Heard “{text}” but the model is still busy. Say it again when it has finished."
        self.messages.append({"who": "robot", "text": note, "events": [], "total_s": 0})
        return "busy"

    def voice_state(self, state):
        """The voice package is armed or listening: the lamp and the listen tone, once per utterance."""
        if state in ("armed", "listening") and not self.busy and self.phase != "listening":
            self.set_phase("listening")
            self._announce("listening")

    def voice_idle(self):
        """The voice package is back to idle: a window that closed with nothing said releases the lamp."""
        if self.phase in ("listening", "heard") and not self.busy:
            self.set_phase("idle")

    # -- session file --------------------------------------------------------
    def finish(self):
        SESSIONS_DIR.mkdir(exist_ok=True)
        path = SESSIONS_DIR / self.started.strftime("%Y-%m-%d-%H%M.md")
        rows = [f"# Session {self.started:%Y-%m-%d %H:%M}", "",
                "| Time | Model | Thinking | Command | Via | Felt | Think | To confirm | Motion | Tool calls | "
                "First tool after | To motion | Total | Reply |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in self.records:
            calls = "; ".join(f"{e['tool']} {e['args']} ({e['outcome']})" for e in r["events"]) or "none"
            motion = ", ".join(f"{e['to_motion_s']} s" for e in r["events"] if "to_motion_s" in e) or ""
            reply = r["reply"].replace("\n", " ").replace("|", "/")
            first = f"{r['events'][0]['after_s']} s" if r["events"] and "after_s" in r["events"][0] else ""
            t = timings(r)
            rows.append(f"| {r['time']} | {r['model']} | {'on' if r.get('think') else 'off'} | {r['command']} | "
                        f"{r.get('via', 'typed')} | {t['felt']} | {t['think']} | {t['to_confirm']} | {t['motion']} | "
                        f"{calls} | {first} | {motion} | {r['total_s']} s | {reply} |")
        path.write_text("\n".join(rows) + "\n")
        return {"path": str(path), "records": self.records}


_NARRATED = re.compile(r"^\s*I (?:called|used|invoked|ran) the \w+ tool\b", re.I)
_CLAIMED = re.compile(r"\b(?:is now at|are now at|now at x|has been (?:stopped|moved)|have moved|I (?:have )?moved|the move was refused)\b", re.I)


def tidy_reply(reply, events):
    """What the page shows for the model's reply. Small models often narrate ("I called the move_by
    tool ... and it returned the result") instead of answering; then the tool's own message is
    the better reply. Worse, they sometimes claim a move or a stop without having called any tool;
    then the reply is shown with a plain warning, so nobody believes the arm moved. The record
    keeps the model's own words for the evaluation."""
    reply = str(reply)
    if events and _NARRATED.match(reply) and events[-1].get("message"):
        return events[-1]["message"]
    if not events and _CLAIMED.search(reply) and not reply.rstrip().endswith("?"):
        return f"Nothing was done: the model answered without calling a tool. It said: {reply}"
    return reply


def timings(record):
    """The loop's timings for one record, as table cells: felt delay (from voice),
    think (thinking to the proposal or the end), to confirm (proposing to moving), motion."""
    phases = record.get("phases") or []

    def span(start, ends):
        t0 = next((t for name, t in phases if name == start), None)
        if t0 is None:
            return ""
        t1 = next((t for name, t in phases if name in ends and t >= t0), None)
        return f"{t1 - t0:.1f} s" if t1 is not None else ""
    felt = record.get("felt_ms")
    return {"felt": f"{felt / 1000:.2f} s" if isinstance(felt, (int, float)) else "",
            "think": span("thinking", ("proposing", "done", "refused", "cancelled", "stopped")),
            "to_confirm": span("proposing", ("moving",)),
            "motion": span("moving", ("done", "refused", "stopped", "cancelled"))}


# -- status checks, each answering in under a second ----------------------------

_status_cache = {}        # {"at": time, "value": status()}: the stage polls four times a second


def _cached_status():
    """status() at most once a second, so a fast poll never hits Ollama repeatedly."""
    now = time.time()
    if _status_cache.get("value") is None or now - _status_cache.get("at", 0) > 1.0:
        _status_cache.update(at=now, value=status())
    return _status_cache["value"]


def capability_rows_with_moveit():
    """The capability rows as the map and the tour want them: every row with a node and an example
    phrase (older rows get them here), plus MoveIt's own tools as rows of node "moveit"."""
    from interfaces import node_of, say_of
    rows = []
    for r in generic_tools.capability_rows():
        r = dict(r)
        r.setdefault("node", node_of(r["name"]))
        r.setdefault("say", say_of(r["name"], {"means": r.get("means", "")}))
        rows.append(r)
    names = {r["name"] for r in rows}
    moveit = config.MOVE_ACTION in names and config.CARTESIAN_SERVICE in names
    for skill in robot_tools.MOVEIT_SKILLS:
        rows.append({"name": skill["name"], "kind": "tool", "type": skill["type"], "means": skill["means"],
                     "moves_robot": skill["moves_robot"], "status": "ready" if moveit else "missing",
                     "node": "moveit", "say": skill["say"]})
    return rows, moveit


def group_name(node):
    """A node id as the map shows it: multiview_scan -> Multiview scan, moveit -> MoveIt."""
    return {"moveit": "MoveIt", "root": "Root", "ur_driver": "UR driver"}.get(node) or node.replace("_", " ").capitalize()


def find_group(rows, words):
    """The node id the operator means by `words`, or None."""
    want = (words or "").strip().lower().strip("?.!")
    if not want:
        return None
    nodes = []
    for r in rows:
        if r["node"] not in nodes:
            nodes.append(r["node"])
    for node in nodes:
        if want in (node, node.replace("_", " "), group_name(node).lower()):
            return node
    for node in nodes:
        if node.startswith(want.replace(" ", "_")) or group_name(node).lower().startswith(want):
            return node
    return None


# Questions about the console itself, answered here (spoken, with the map lit) and never sent to
# the model. Whole utterances; the wake word, if any, is trimmed by words.strip_wake.
SELF_QUESTIONS = [
    ("tour", re.compile(r"^(?:what can i (?:do|say))(?: with (?P<g>.+?))?\s*[?.!]*$", re.I)),
    ("tour", re.compile(r"^what(?:'s| is) connected to (?P<g>.+?)\s*[?.!]*$", re.I)),
    ("tour", re.compile(r"^(?:tell me about|what does) (?P<g>.+?)(?: do)?\s*[?.!]*$", re.I)),
    ("movers", re.compile(r"^(?:which (?:of these |ones? )?moves? the robot|what moves the robot)\s*[?.!]*$", re.I)),
    ("trace", re.compile(r"^where did (?:that|it) go\s*[?.!]*$", re.I)),
    ("ros", re.compile(r"^(?P<on>show|hide) ros\s*[?.!]*$", re.I)),
]


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

# The voice front end (voice/): the microphone, the wake word, speech-to-text and Rosa's voice, on
# its own thread. Its callbacks look `console` up when called, because the tests replace it.
voice_unit = voice.Voice(config.VOICE, on_final=lambda text: console.heard(text),
                         on_timings=lambda ms: console.voice_timings(ms),
                         on_state=lambda state: console.voice_state(state),
                         on_idle=lambda: console.voice_idle())


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "stage.html")


@app.post("/settings/voice")
def voice_settings():
    """Any key of the voice block, checked, saved, and handed to the voice package."""
    body = request.json or {}
    problem = config.voice_settings_ok(body)
    if problem:
        return jsonify(error=problem), 400
    config.save_settings({"voice": body})
    voice_unit.update(body)
    return jsonify(ok=True, voice=config.VOICE)


@app.get("/state")
def state():
    follow_left = max(0.0, console.follow_until - time.time())
    if console.phase == "follow" and follow_left == 0 and not console.busy:
        console.set_phase("idle")
    return jsonify(status=_cached_status(), busy=console.busy, pending=console.pending,
                   phase=console.phase, phase_since_s=round(time.time() - console.phase_since, 1),
                   follow_until_s=round(follow_left, 1), trace=console_trace(), lit=console.lit,
                   held=console.held, said=console.said, verbosity=config.VOICE["verbosity"],
                   persona=config.VOICE["persona"], follow_up_s=config.VOICE["follow_up_s"],
                   confirm_timeout_s=config.VOICE["confirm_timeout_s"], map_ros=config.VOICE["map_ros"],
                   busy_for_s=round(time.time() - console.busy_since, 1) if console.busy else 0,
                   messages=console.messages, model=config.MODEL, think=config.THINK,
                   joint_list=config.JOINTS,
                   limits=limits(), ceilings=config.LIMIT_CEILINGS, labels=LIMIT_LABELS, speed=config.SPEED,
                   shown_path=generic_tools.last_path, ollama_url=config.OLLAMA_URL,
                   frames={"base": config.BASE_FRAME, "tool": config.TOOL_FRAME},
                   robot=robot_link.latest, local=_local(), voice=voice_state())


def console_trace():
    """Which tool is running right now, for the map's lit path; None when none is."""
    tool = robot_tools._current_event.get("tool") if console.busy else None
    return {"tool": tool} if tool else None


def voice_state():
    """What the page shows about voice: the voice package's latest news plus the settings."""
    return {**voice_unit.latest, **config.VOICE}


def _local():
    """Is this request from the robot's own computer (the browser on its screen)?"""
    return request.remote_addr in ("127.0.0.1", "::1")


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
    """What the robot's system offers, for the panel on the page and the map: every interface
    with its node and an example phrase, plus MoveIt's tools as rows of node "moveit"."""
    try:
        rows, moveit = capability_rows_with_moveit()
    except Exception as exc:
        return jsonify(error=f"Cannot read the ROS graph: {exc}"), 502
    return jsonify(system=generic_tools.description().system, interfaces=rows, moveit=moveit,
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


@app.post("/stop")
def stop():
    return jsonify(message=console.stop())


# -- the robot connection panel: the ladder lives in robot_link, only names come from the page --

@app.post("/robot/mode")
def robot_mode():
    mode = (request.json or {}).get("mode")
    if mode not in ("sim", "real"):
        return jsonify(error="The mode is sim or real."), 400
    config.save_settings({"robot_mode": mode})
    robot_link.refresh()
    return jsonify(robot_link.latest)


@app.post("/robot/action")
def robot_action():
    name = (request.json or {}).get("name", "")
    if name not in robot_link.ACTIONS:
        return jsonify(error=f"There is no action called {name}."), 404
    if robot_link.needs_local(name, config.ROBOT_MODE) and not _local():
        return jsonify(error="The real robot is only started from the robot's computer's own screen."), 403
    try:
        message = robot_link.run_action(name, config.ROBOT_MODE)
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 409
    return jsonify(ok=True, message=message, robot=robot_link.latest)


@app.post("/finish")
def finish():
    out = console.finish()
    return jsonify(path=out["path"], records=[{**r, "timings": timings(r)} for r in out["records"]])


# -- voice: the switch, voice confirm, Hold to talk, Test voice, the device menus ------

@app.post("/voice/on")
def voice_on():
    on = bool((request.json or {}).get("on", False))
    config.save_settings({"voice": {"on": on}})
    voice_unit.set_wake(on)
    console.messages.append({"who": "robot", "text": "Voice control is on." if on else "Voice control is off.",
                             "events": [], "total_s": 0})
    return jsonify(ok=True, on=on)


@app.post("/voice/confirm")
def voice_confirm():
    config.save_settings({"voice": {"confirm": bool((request.json or {}).get("on", False))}})
    return jsonify(ok=True, confirm=config.VOICE["confirm"])


@app.post("/voice/ptt")
def voice_ptt():
    """Hold to talk: {"down": true} when the button goes down, false when it is let go."""
    voice_unit.ptt(bool((request.json or {}).get("down", False)))
    return jsonify(ok=True)


@app.post("/voice/say")
def voice_say():
    """The Test voice button."""
    spoken = voice_unit.say(str((request.json or {}).get("text", "")).strip())
    return jsonify(spoken=spoken, reason=voice_unit.latest.get("speech_reason", ""))


@app.get("/voice/devices")
def voice_devices():
    """Microphones, speakers and Piper voices, for the pop-up's menus."""
    return jsonify(**voice_unit.latest.get("devices", {"input": [], "output": []}), voices=voice_unit.voices())


def main():
    global console
    from rosa_agent import abort, build_agent
    console = Console(build_agent, abort)
    print(f"Web console on http://0.0.0.0:8000, model {config.MODEL} at {config.OLLAMA_URL}")
    robot_link.start_background()             # the connection ladder, checked every few seconds
    voice_unit.start()                        # the microphone, on its own thread
    app.run(host="0.0.0.0", port=8000, threaded=True)


if __name__ == "__main__":
    main()
