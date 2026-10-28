"""The web page and its HTTP endpoints (Flask, port 8100). docs/04-settings-and-ui.md

    ./.venv/bin/python app.py                 # microphone, then open http://localhost:8100
    ./.venv/bin/python app.py --file x.wav    # loop a WAV at real speed instead of the mic (testing)

Threads: Flask serves requests; ONE worker thread runs the pipeline (and loads
models). They meet at two queues: commands go to the pipeline, events come back.
"""

import argparse
import datetime
import json
import queue
import signal
import sys
import threading
from collections import deque

from flask import Flask, Response, jsonify, request, send_from_directory

import config
import device as device_mod
import speech
from pipeline import Pipeline
from stages import registry

SPEECH_DEFAULTS = {"enabled": True, "engine": "piper", "voice": "en_GB-alan-medium", "output_device": None}
SOUNDS_DEFAULTS = {"enabled": True}
VOICES_DIR = config.ROOT / "models" / "piper"


def default_speaker(voice, on_speaking, output_device=None):
    """The real thing: Piper for the words, the sound card for playback. Either may be missing;
    then the Speaker says why and the app stays silent."""
    player = speech.sounddevice_player(output_device)
    engine = speech.piper_engine(voice, VOICES_DIR)
    speaker = speech.Speaker(engine or (lambda text: None), player, cache_dir=config.ROOT / "cache" / "say",
                             on_speaking=on_speaking, voice_key=voice)
    speaker.available = engine is not None
    return speaker, speech.Sounds(player)


class EventHub:
    """Fans pipeline events out to every open page (one queue per page)."""

    def __init__(self):
        self.clients = []
        self.lock = threading.Lock()
        self.last = {}            # the latest event of each kind, sent to a page when it connects

    def emit(self, kind, data):
        self.last[kind] = data
        with self.lock:
            for q in self.clients:
                try:
                    q.put_nowait((kind, data))
                except queue.Full:
                    pass          # a stalled page misses events rather than slowing the pipeline

    def subscribe(self):
        q = queue.Queue(maxsize=500)
        with self.lock:
            self.clients.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            self.clients.remove(q)


class LoopingFile:
    """A WAV played at real speed, over and over: lets the app run without a mic (tests)."""

    kind = "file"

    def __init__(self, path):
        from audio import FileSource
        self.path, self._running, self._src = path, False, None
        self.FileSource = FileSource

    def start(self):
        self._running = True

    def stop(self):
        self._running = False
        if self._src:
            self._src.stop()

    def backlog(self):
        return 0

    def frames(self):
        while self._running:
            self._src = self.FileSource(self.path, realtime=True, pad_end_s=1.0)
            self._src.start()
            yield from self._src.frames()


def create_app(settings_path=config.SETTINGS, make_source=None, log_path=None, make_speaker=None):
    """Build the Flask app. `make_source()` returns the audio source (mic by default);
    `make_speaker(voice, on_speaking)` returns (Speaker, Sounds) (Piper and the sound card by default)."""
    app = Flask(__name__, static_folder="static")
    hub = EventHub()
    history = deque(maxlen=50)                        # recent utterances, for a page reload
    state = {"settings": config.load(settings_path)}
    state["settings"].setdefault("speech", {})
    state["settings"].setdefault("sounds", {})
    for key, value in SPEECH_DEFAULTS.items():
        state["settings"]["speech"].setdefault(key, value)
    for key, value in SOUNDS_DEFAULTS.items():
        state["settings"]["sounds"].setdefault(key, value)

    if log_path is None:
        log_path = config.ROOT / "runs" / f"{datetime.date.today()}-live.jsonl"
    pipe = Pipeline(state["settings"], emit=hub.emit, log_path=log_path,
                    run_name=f"{datetime.date.today()}-live")
    pipe.on_record.append(history.append)

    def worker():
        try:
            if make_source is None:
                from audio import MicSource
                source = MicSource(state["settings"]["audio"].get("input_device"))
            else:
                source = make_source()
            pipe.run(source)           # loads the models first, on this thread
        except Exception as e:         # e.g. no microphone
            hub.emit("error", {"stage": "audio", "message": str(e)})

    thread = threading.Thread(target=worker, name="pipeline", daemon=True)

    # -- speaking back: the console sends the words; the app mutes its own microphone meanwhile
    def on_speaking(on):
        pipe.set_muted(on)
        hub.emit("speaking", {"on": bool(on)})

    def build_speaker():
        voice = state["settings"]["speech"]["voice"]
        make = make_speaker or default_speaker
        speaker, sounds = make(voice, on_speaking, state["settings"]["speech"].get("output_device"))
        if not hasattr(speaker, "available"):
            speaker.available = True
        app.config.update(speaker=speaker, sounds=sounds)

    build_speaker()
    app.config.update(pipe=pipe, hub=hub, worker=thread)

    # ------------------------------------------------------------- the page

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/events")
    def events():
        """Server-Sent Events: one long response the server keeps writing to."""
        q = hub.subscribe()

        def stream():
            try:
                # Send something at once: the browser's "open" (which makes the page
                # load its data) only fires when the first bytes arrive.
                yield ": connected\n\n"
                # A page that connects mid-way gets the current state straight away.
                for kind in ("state", "loading", "error"):
                    if kind in hub.last:
                        yield f"event: {kind}\ndata: {json.dumps(hub.last[kind])}\n\n"
                while True:
                    try:
                        kind, data = q.get(timeout=15)
                        yield f"event: {kind}\ndata: {json.dumps(data, default=str)}\n\n"
                    except queue.Empty:
                        yield ": keep-alive\n\n"      # stops proxies/browsers timing out
            finally:
                hub.unsubscribe(q)

        return Response(stream(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ------------------------------------------------------------- settings

    @app.get("/settings")
    def get_settings():
        return jsonify(state["settings"])

    @app.post("/settings")
    def post_settings():
        update = request.get_json(force=True, silent=True)
        if not isinstance(update, dict):
            return jsonify(error="send a JSON object"), 400
        new = config.merge(state["settings"], update)
        problem = check(new)
        if problem:
            return jsonify(error=problem), 400
        voice_changed = (new["speech"]["voice"], new["speech"].get("output_device")) != \
                        (state["settings"]["speech"]["voice"], state["settings"]["speech"].get("output_device"))
        state["settings"] = new
        config.save(new, settings_path)
        pipe.request_settings(new)     # applied by the worker between utterances
        if voice_changed:
            build_speaker()
        return jsonify(new)

    def check(settings) -> str | None:
        """Reject settings the pipeline can't run, with a reason for the page."""
        try:
            dev = device_mod.resolve(settings.get("device", "auto"))
        except ValueError as e:
            return str(e)
        for stage in ("trigger", "vad", "endpoint", "stt"):
            name = settings.get(stage, {}).get("use")
            try:
                ok, reason = registry.get_class(stage, name).supports(dev)
            except Exception as e:
                return f"{stage}: {e}"
            if not ok:
                return f"{stage} {name} can't run on {dev}: {reason}"
        if settings["stt"].get("mode") not in ("batch", "streaming"):
            return "stt mode must be batch or streaming"
        if not isinstance(settings.get("speech", {}).get("voice", "x"), str):
            return "speech.voice must be a voice name"
        return None

    @app.get("/options")
    def options():
        dev = device_mod.resolve(state["settings"].get("device", "auto"))
        voices = sorted(p.name[:-5] for p in VOICES_DIR.glob("*.onnx")) if VOICES_DIR.is_dir() else []
        return jsonify(device=dev, machine=device_mod.describe(), options=registry.describe(dev), voices=voices)

    @app.get("/status")
    def status():
        sp, cfg = app.config["speaker"], state["settings"]["speech"]
        return jsonify(state=pipe.state, wake_on=pipe.wake_on, device=pipe.device,
                       summary=pipe.summary(), loaded={k: type(v).name for k, v in pipe.stages.items()},
                       speaking=sp.speaking, followup_s=round(pipe.followup_s, 2),
                       speech={"enabled": cfg["enabled"], "engine": cfg["engine"], "voice": cfg["voice"],
                               "available": bool(getattr(sp, "available", True)), "reason": sp.reason})

    # --------------------------------------------------------- speaking back

    @app.post("/say")
    def say():
        """Speak a line (the console composes it). {"cancel": true} hushes."""
        body = request.get_json(force=True, silent=True) or {}
        sp = app.config["speaker"]
        if body.get("cancel"):
            sp.cancel()
            return jsonify(spoken=False, reason="cancelled")
        text = str(body.get("text", "")).strip()
        if not text:
            return jsonify(error="send {\"text\": ...}"), 400
        if not state["settings"]["speech"]["enabled"]:
            return jsonify(spoken=False, reason="speech is off in the settings")
        spoken = sp.say(text)
        return jsonify(spoken=spoken, reason="" if spoken else sp.reason)

    @app.post("/play")
    def play():
        """One of the state sounds: listen, heard, ask, go, done, cancel, refused, think."""
        name = str((request.get_json(force=True, silent=True) or {}).get("name", ""))
        if name not in speech.SOUNDS:
            return jsonify(error=f"no sound called {name}"), 404
        if not state["settings"]["sounds"]["enabled"]:
            return jsonify(played=False)
        return jsonify(played=app.config["sounds"].play(name))

    @app.post("/followup")
    def followup():
        """For N seconds, speech needs no wake word (the console has just asked something)."""
        body = request.get_json(force=True, silent=True) or {}
        try:
            seconds = max(0.0, min(120.0, float(body.get("seconds", 0))))
        except (TypeError, ValueError):
            return jsonify(error="seconds must be a number"), 400
        pipe.followup(seconds)
        return jsonify(until_s=round(pipe.followup_s, 2))

    @app.get("/history")
    def get_history():
        keep = ("final", "outcome", "started_by", "settings_summary", "marks_audio_s", "derived", "time")
        return jsonify([{k: r.get(k) for k in keep} for r in history])

    # --------------------------------------------------------------- controls

    @app.post("/ptt/down")
    def ptt_down():
        pipe.ptt_down()
        return jsonify(ok=True)

    @app.post("/ptt/up")
    def ptt_up():
        pipe.ptt_up()
        return jsonify(ok=True)

    @app.post("/wake")
    def wake():
        on = bool((request.get_json(force=True, silent=True) or {}).get("on", True))
        pipe.set_wake(on)
        return jsonify(on=on)

    # ---------------------------------------------------------------- presets

    @app.get("/presets")
    def presets():
        return jsonify(config.list_presets())

    @app.post("/presets/<name>")
    def save_preset(name):
        """Save the current settings as presets/<name>.yaml. Won't overwrite
        (presets used for reported results must not change) unless ?overwrite=1."""
        try:
            path = config.preset_path(name)
        except ValueError as e:
            return jsonify(error=str(e)), 400
        if path.exists() and request.args.get("overwrite") != "1":
            return jsonify(error=f"preset {name} exists; add ?overwrite=1 to replace it"), 409
        config.save(state["settings"], path)
        return jsonify(saved=name)

    @app.post("/presets/<name>/load")
    def load_preset(name):
        try:
            path = config.preset_path(name)
        except ValueError as e:
            return jsonify(error=str(e)), 400
        if not path.exists():
            return jsonify(error=f"no preset {name}"), 404
        new = config.load(path)
        problem = check(new)
        if problem:
            return jsonify(error=problem), 400
        state["settings"] = new
        config.save(new, settings_path)
        pipe.request_settings(new)
        return jsonify(new)

    return app


def main():
    # `kill` sends SIGTERM, which normally skips Python's exit handlers; turn it
    # into a normal exit so plug-ins can clean up (e.g. stop a whisper-server).
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="loop this WAV instead of using the microphone")
    ap.add_argument("--port", type=int, default=8100)
    args = ap.parse_args()
    app = create_app(make_source=(lambda: LoopingFile(args.file)) if args.file else None)
    app.config["worker"].start()
    print(f"open http://localhost:{args.port}")
    app.run(host="127.0.0.1", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
