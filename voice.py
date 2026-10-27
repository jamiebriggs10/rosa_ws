"""The console's link to the voice app.

Speech is handled by a separate program, the Voice Control Prototype ("the voice app"): it
listens to the microphone, waits for the wake word or a push-to-talk press, works out when
the speaker has finished, and turns the words into text. It has its own page on port 8100
and its own settings file (settings.yaml). This module is the console's end of the wire.

  - One background thread keeps a connection open to the voice app's /events. That is a
    stream of small messages (Server-Sent Events: the server keeps one HTTP response open
    and writes "event: final / data: {...}" blocks into it as things happen).
  - What arrives is kept in `latest` for the page (the state lamp, the live text, the mic
    level) and handed to the console through three callbacks: a finished utterance
    (`on_final`), live text while someone is still speaking (`on_partial`), and "back to
    idle" (`on_idle`).
  - `request` carries the pop-up's settings changes on to the voice app.

If the voice app is not running, the thread tries again every RETRY_S seconds and
latest["running"] is False. Nothing else in the console depends on it.

The word tests at the top decide what a spoken phrase means. Only a whole utterance of
stop words is a stop, so "stop the move" is a command for the model, not an emergency stop.
"""
import http.client
import json
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import config

STOP_WORDS = frozenset({"stop", "halt", "freeze"})
# Only these move the robot. "yes", "okay" and "go" are too easy to mishear or to say in passing.
YES_WORDS = frozenset({"confirm", "go ahead"})
NO_WORDS = frozenset({"no", "cancel"})
_NO_START = re.compile(r"^(no|nope|cancel)\b[,.]?\s*(?:(?:actually|rather|instead)\b[,.]?\s*)?")

RETRY_S = 2.0             # how long to wait before trying the voice app again
STREAM_TIMEOUT_S = 30.0   # the voice app writes a keep-alive every 15 s; silence this long means the line is dead
REQUEST_TIMEOUT_S = 3.0   # for ordinary requests (settings, push-to-talk)


def normalise(text):
    """Lower case, letters and digits only, single spaces: "Stop." -> "stop"."""
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", (text or "").lower()).split())


def is_stop(text):
    """True when every word is a stop word (and there is at least one)."""
    words = normalise(text).split()
    return bool(words) and all(w in STOP_WORDS for w in words)


def says_yes(text):
    return normalise(text) in YES_WORDS


def says_no(text):
    return normalise(text) in NO_WORDS


def split_no(text):
    """For an utterance that starts with a no-word, the rest of it: "" for a bare "no",
    "point eleven" for "no, point eleven" (a one-step correction). None if it does not start
    with one, so "point eleven" alone is just a command."""
    words = normalise(text)
    m = _NO_START.match(words)
    if not m:
        return None
    return words[m.end():].strip()


class VoiceUnavailable(Exception):
    """The voice app did not answer (not running, wrong address, or too slow)."""


def _fresh():
    return {"running": False, "state": "", "wake_on": None, "level": None, "partial": "", "final": "",
            "felt_ms": None, "outcome": "", "loading": "", "error": "", "summary": "", "device": "", "action": "",
            "speaking": False, "followup_s": 0}


def _json(response):
    try:
        return json.load(response)
    except ValueError:
        return {}


class VoiceLink:
    def __init__(self, url=None, on_final=None, on_partial=None, on_idle=None, on_timings=None):
        self._url = url or (lambda: config.VOICE_URL)      # a function, so a changed setting is picked up
        self.on_final, self.on_partial, self.on_idle, self.on_timings = on_final, on_partial, on_idle, on_timings
        self.latest = _fresh()
        self._sock = None                                   # the open stream's socket, so reconnect() can cut it
        self._lock = threading.Lock()

    @property
    def url(self):
        return self._url().rstrip("/")

    # -- requests ------------------------------------------------------------------

    def request(self, method, path, body=None, timeout_s=REQUEST_TIMEOUT_S, base=None):
        """One HTTP request to the voice app: (status, json). HTTP errors come back as their
        status and body (the voice app explains itself in them); no answer at all raises
        VoiceUnavailable. `base` tries another address, e.g. one the operator has just typed."""
        where = (base or self.url).rstrip("/")
        data = json.dumps(body if body is not None else {}).encode() if method == "POST" else None
        headers = {"Content-Type": "application/json"} if data is not None else {}
        try:
            req = urllib.request.Request(where + path, data=data, method=method, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                return r.status, _json(r)
        except urllib.error.HTTPError as e:
            return e.code, _json(e)
        except (OSError, ValueError) as e:     # connection refused, timeout, no such host, or not an address at all
            if base is None:
                self.latest["running"] = False
            raise VoiceUnavailable(f"The voice app is not answering at {where}.") from e

    def set_on(self, on):
        """Tell the voice app whether its wake word is on. Returns an error sentence, or None."""
        try:
            self.request("POST", "/wake", {"on": bool(on)})
            return None
        except VoiceUnavailable as e:
            return str(e)

    # -- speaking back: fire-and-forget, so a phase change never waits for the voice app ------

    def _post_later(self, path, body):
        def go():
            try:
                self.request("POST", path, body)
            except VoiceUnavailable:
                pass                           # no voice app: the console simply makes no sound
        threading.Thread(target=go, daemon=True, name="voice-say").start()

    def say(self, text):
        """Ask the voice app to speak a line (the console composes it, lines.py)."""
        if text:
            self._post_later("/say", {"text": text})

    def hush(self):
        self._post_later("/say", {"cancel": True})

    def play(self, name):
        """One of the state sounds: listen, heard, ask, go, done, cancel, refused, think."""
        self._post_later("/play", {"name": name})

    def followup(self, seconds):
        """For `seconds`, the voice app listens without the wake word (Rosa has just asked something)."""
        self._post_later("/followup", {"seconds": seconds})

    # -- the stream -----------------------------------------------------------------

    def start_background(self):
        def loop():
            while True:
                try:
                    self.listen_once()
                except Exception:              # not running, or the line dropped: try again shortly
                    pass
                self.latest["running"] = False
                time.sleep(RETRY_S)
        threading.Thread(target=loop, daemon=True, name="voice-link").start()

    def listen_once(self):
        """One connection: open the stream, bring the two programs into step, read until it ends."""
        parts = urllib.parse.urlsplit(self.url)
        # The socket is made here rather than inside http.client, so reconnect() can reach it:
        # http.client lets go of its socket once the response owns it.
        sock = socket.create_connection((parts.hostname, parts.port or 80), timeout=STREAM_TIMEOUT_S)
        conn = http.client.HTTPConnection(parts.hostname, parts.port or 80)
        conn.sock = sock
        try:
            conn.request("GET", "/events", headers={"Accept": "text/event-stream"})
            resp = conn.getresponse()
            if resp.status != 200:
                raise VoiceUnavailable(f"The voice app at {self.url} answered {resp.status} for /events.")
            with self._lock:
                self._sock = sock
            self.latest["running"] = True
            self.latest["error"] = self.latest["loading"] = ""   # a fresh start: the old run's problems are gone
            self._sync()
            self._read(resp)
        finally:
            with self._lock:
                self._sock = None
            conn.close()
            sock.close()
            self.latest["running"] = False

    def reconnect(self):
        """Cut the current stream, so the thread connects again (to a new address, say) at once."""
        with self._lock:
            sock = self._sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)      # wakes the blocked read with "end of stream"
            except OSError:
                pass

    def _sync(self):
        """Facts no event carries (wake word on?, which options, which device) and the console's
        switch position, so both programs agree after either one restarts."""
        try:
            _, status = self.request("GET", "/status")
            for key in ("wake_on", "summary", "device", "state"):
                if key in status:
                    self.latest[key] = status[key]
            self.request("POST", "/wake", {"on": config.VOICE_ON})
        except VoiceUnavailable:
            pass                               # the stream itself is open; the next sync will catch up

    def _read(self, resp):
        """Parse the stream: "event:" names the kind, "data:" lines carry JSON, a blank line
        ends one message. Lines starting with ":" are the server's keep-alive comments."""
        kind, data = None, []
        while True:
            raw = resp.readline()              # b"" when the server has closed the stream
            if not raw:
                return
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if not line:
                if kind and data:
                    try:
                        payload = json.loads("\n".join(data))
                    except ValueError:
                        payload = None         # a damaged message is dropped, the stream goes on
                    if isinstance(payload, dict):
                        self.handle(kind, payload)
                kind, data = None, []
            elif line.startswith(":"):
                continue
            elif line.startswith("event:"):
                kind = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:"):].strip())

    def handle(self, kind, data):
        """One event from the voice app into `latest`, and on to the console where it matters."""
        latest = self.latest
        latest["running"] = True
        if kind == "state":
            latest["state"] = data.get("state", "")
            if latest["state"] == "idle":
                latest["partial"] = ""
                self._call(self.on_idle)
        elif kind == "level":
            latest["level"] = data.get("rms")
        elif kind == "partial":
            latest["partial"] = data.get("text", "")
            self._call(self.on_partial, latest["partial"])
        elif kind == "final":
            latest["final"], latest["outcome"] = data.get("text", ""), data.get("outcome", "")
            latest["partial"], latest["felt_ms"], latest["error"] = "", None, ""
            latest["action"] = self._call(self.on_final, latest["final"]) or ""
        elif kind == "timings":
            latest["felt_ms"] = (data.get("derived") or {}).get("felt_delay_ms")
            if latest["felt_ms"] is not None:
                self._call(self.on_timings, latest["felt_ms"])
        elif kind == "speaking":
            latest["speaking"] = bool(data.get("on"))
        elif kind == "followup":
            latest["followup_s"] = data.get("until_s", 0)
        elif kind == "loading":
            latest["loading"] = "" if data.get("done") else f"loading {data.get('stage')}: {data.get('option')}…"
        elif kind == "error":
            latest["error"] = data.get("message", "")

    def _call(self, callback, *args):
        """Run a console callback; a fault in the console must never kill the link."""
        if callback is None:
            return None
        try:
            result = callback(*args)
            return result if isinstance(result, str) else None
        except Exception as exc:
            self.latest["error"] = f"console: {exc}"
            return None
