"""Tests for voice.py, the console's link to the voice app, against a fake voice app.

These need no ROS, so they also run on the Mac:  python3 -m pytest tests/test_voice.py
"""
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import config
import voice


class FakeVoiceApp:
    """Just enough of the voice app's HTTP side: /events streams scripted events then closes,
    /status answers a fixed line, every POST is recorded in `posts`."""

    def __init__(self, events, raw_lines=(), hold_s=0.0):
        self.events, self.raw_lines, self.hold_s = events, raw_lines, hold_s
        self.posts = []            # (path, body) of every POST, in order
        self.connections = 0       # how many times /events was opened
        self.post_status = {}      # path -> (status, body) to answer with; default 200 {"ok": true}
        app = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == "/events":
                    app.connections += 1
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    try:
                        self.wfile.write(b": connected\n\n")
                        for kind, data in app.events:
                            self.wfile.write(f"event: {kind}\ndata: {json.dumps(data)}\n\n".encode())
                        for line in app.raw_lines:
                            self.wfile.write(line.encode())
                        time.sleep(app.hold_s)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                elif self.path == "/status":
                    self.reply(200, {"state": "idle", "wake_on": True, "device": "cpu",
                                     "summary": "ptt + silero + ptt_release + parakeet/batch"})
                else:
                    self.reply(404, {"error": "no such page"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n)) if n else None
                app.posts.append((self.path, body))
                status, reply = app.post_status.get(self.path, (200, {"ok": True}))
                self.reply(status, reply)

            def reply(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def fake():
    apps = []

    def make(events, **kw):
        app = FakeVoiceApp(events, **kw)
        apps.append(app)
        return app
    yield make
    for app in apps:
        app.close()


def closed_port_url():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}"


def wait_until(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while not predicate():
        assert time.time() < deadline, "timed out"
        time.sleep(0.02)


# -- the word tests ----------------------------------------------------------------

@pytest.mark.parametrize("text,stop", [("Stop.", True), ("stop stop", True), ("HALT!", True), ("freeze", True),
                                       ("stop the move", False), ("", False), ("   ", False)])
def test_is_stop(text, stop):
    assert voice.is_stop(text) is stop


def test_only_confirm_and_go_ahead_are_yes():
    assert voice.says_yes("Confirm.") and voice.says_yes("go ahead")
    assert not voice.says_yes("yes") and not voice.says_yes("okay") and not voice.says_yes("confirm it now")
    assert voice.says_no("Cancel") and voice.says_no("no.")
    assert not voice.says_no("cancel the second move")


@pytest.mark.parametrize("text,rest", [("no", ""), ("No, point eleven", "point eleven"), ("cancel", ""),
                                       ("no actually go home", "go home"), ("nope, rather up five", "up five"),
                                       ("No. Instead, point twelve.", "point twelve"),
                                       ("point eleven", None), ("nothing here", None), ("", None)])
def test_split_no(text, rest):
    assert voice.split_no(text) == rest


# -- the stream -----------------------------------------------------------------------

def test_events_fill_latest_and_call_back(fake, monkeypatch):
    monkeypatch.setattr(config, "VOICE_ON", False)
    app = fake([("state", {"state": "listening", "t": 1.0}), ("partial", {"text": "move"}),
                ("final", {"text": "move up", "outcome": "silence"}),
                ("timings", {"derived": {"felt_delay_ms": 412}}),
                ("loading", {"stage": "stt", "option": "parakeet", "done": False}),
                ("error", {"stage": "audio", "message": "no mic"}),
                ("state", {"state": "idle", "t": 2.0})])
    finals, partials, idles = [], [], []
    link = voice.VoiceLink(url=lambda: app.url, on_final=lambda t: finals.append(t) or "sent",
                           on_partial=partials.append, on_idle=lambda: idles.append(1))
    link.listen_once()                                  # returns when the fake closes the stream
    latest = link.latest
    assert latest["state"] == "idle" and latest["partial"] == ""
    assert latest["final"] == "move up" and latest["outcome"] == "silence" and latest["felt_ms"] == 412
    assert latest["loading"] == "loading stt: parakeet…" and latest["error"] == "no mic"
    assert latest["action"] == "sent"                    # what the console did with the final
    assert latest["wake_on"] is True and latest["device"] == "cpu" and latest["summary"].startswith("ptt")
    assert finals == ["move up"] and partials == ["move"] and idles == [1]
    assert ("/wake", {"on": False}) in app.posts         # the console's switch, sent on connect
    assert latest["running"] is False                    # the stream has ended


def test_a_fresh_connection_clears_an_old_error(fake):
    """The voice app was restarted (say, after a microphone fix): its old error must not linger."""
    app = fake([("state", {"state": "idle", "t": 0.0})])
    link = voice.VoiceLink(url=lambda: app.url)
    link.latest["error"], link.latest["loading"] = "no input device matching 'Jabra'", "loading stt: parakeet…"
    link.listen_once()
    assert link.latest["error"] == "" and link.latest["loading"] == ""


def test_stream_ignores_comments_and_bad_json(fake):
    app = fake([], raw_lines=[": keep-alive\n\n", "event: final\ndata: {nope\n\n",
                              "event: final\ndata: {\"text\": \"ok\", \"outcome\": \"silence\"}\n\n"])
    finals = []
    link = voice.VoiceLink(url=lambda: app.url, on_final=finals.append)
    link.listen_once()
    assert finals == ["ok"]


def test_a_failing_callback_does_not_kill_the_stream(fake):
    app = fake([("final", {"text": "one", "outcome": "silence"}), ("final", {"text": "two", "outcome": "silence"})])
    seen = []

    def boom(text):
        seen.append(text)
        raise RuntimeError("console fell over")
    link = voice.VoiceLink(url=lambda: app.url, on_final=boom)
    link.listen_once()
    assert seen == ["one", "two"] and "console fell over" in link.latest["error"]


def test_reconnects_after_the_stream_drops(fake, monkeypatch):
    monkeypatch.setattr(voice, "RETRY_S", 0.05)
    app = fake([("state", {"state": "idle", "t": 0.0})])
    link = voice.VoiceLink(url=lambda: app.url)
    link.start_background()
    wait_until(lambda: app.connections >= 2)


def test_running_is_true_while_connected(fake):
    app = fake([("state", {"state": "idle", "t": 0.0})], hold_s=3.0)
    link = voice.VoiceLink(url=lambda: app.url)
    thread = threading.Thread(target=link.listen_once, daemon=True)
    thread.start()
    wait_until(lambda: link.latest["running"])
    link.reconnect()                                     # leaves the stream at once, not after hold_s
    thread.join(1.0)
    assert not thread.is_alive() and link.latest["running"] is False


# -- requests ----------------------------------------------------------------------------

def test_set_on_posts_wake(fake):
    app = fake([])
    link = voice.VoiceLink(url=lambda: app.url)
    assert link.set_on(True) is None
    assert ("/wake", {"on": True}) in app.posts


def test_set_on_reports_an_unreachable_app():
    link = voice.VoiceLink(url=closed_port_url)
    assert "not answering" in link.set_on(True)


def test_request_raises_when_unreachable():
    link = voice.VoiceLink(url=closed_port_url)
    with pytest.raises(voice.VoiceUnavailable):
        link.request("GET", "/status")
    assert link.latest["running"] is False


def test_request_treats_a_broken_address_as_unavailable():
    link = voice.VoiceLink(url=lambda: "http")          # passes a "starts with http" check, but is no address
    with pytest.raises(voice.VoiceUnavailable):
        link.request("GET", "/status")


def test_request_returns_http_errors_not_raises(fake):
    app = fake([])
    app.post_status["/presets/x"] = (409, {"error": "preset x exists"})
    link = voice.VoiceLink(url=lambda: app.url)
    assert link.request("POST", "/presets/x") == (409, {"error": "preset x exists"})
    assert link.request("GET", "/status")[0] == 200


def test_request_can_try_another_address(fake):
    app = fake([])
    link = voice.VoiceLink(url=closed_port_url)
    status, body = link.request("GET", "/status", base=app.url + "/")
    assert status == 200 and body["state"] == "idle"


# -- speaking back: say, play, follow-up, and the speaking event ---------------------------

def test_say_play_followup_post_and_swallow_errors(fake):
    app = fake([])
    link = voice.VoiceLink(url=lambda: app.url)
    link.say("Hello"); link.play("ask"); link.followup(8); link.hush()
    wait_until(lambda: len(app.posts) == 4)
    assert ("/say", {"text": "Hello"}) in app.posts and ("/play", {"name": "ask"}) in app.posts
    assert ("/followup", {"seconds": 8}) in app.posts and ("/say", {"cancel": True}) in app.posts
    voice.VoiceLink(url=closed_port_url).say("nobody")           # must not raise, must not block
    voice.VoiceLink(url=closed_port_url).play("ask")


def test_speaking_and_followup_events_fill_latest_and_timings_call_back(fake, monkeypatch):
    monkeypatch.setattr(config, "VOICE_ON", False)
    app = fake([("speaking", {"on": True}), ("followup", {"until_s": 7.5}), ("speaking", {"on": False}),
                ("timings", {"derived": {"felt_delay_ms": 333}})])
    felt = []
    link = voice.VoiceLink(url=lambda: app.url, on_timings=felt.append)
    link.listen_once()
    assert link.latest["speaking"] is False and link.latest["followup_s"] == 7.5 and felt == [333]
