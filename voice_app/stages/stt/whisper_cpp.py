"""whisper.cpp speech-to-text: the C/C++ version of OpenAI's Whisper.

Same code runs on Metal (Mac), CUDA (NVIDIA) or CPU; only the build differs.
Whisper cannot stream by itself, so BufferedSTT gives batch and
pseudo-streaming (re-transcribe the growing buffer) for free.

Two drivers, chosen by the `driver` knob, so they can be timed against each other:
  bindings: pywhispercpp, in-process. Fewest moving parts, no network hop.
  server:   whisper-server over local HTTP. The model stays loaded in a
            separate process; build it once per machine with the right flag
            (e.g. CUDA on the lab PC) and this plug-in just posts audio.

Whisper tends to invent text ("Thank you.") from silence. We do not guard
against that here: the pipeline only transcribes audio after the VAD heard speech.
"""

import atexit
import io
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
import soundfile as sf

import models
from stages.base import RATE
from stages.stt.buffered import BufferedSTT

MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{}.bin"
LOG_FILE = Path(__file__).resolve().parents[2] / "runs" / "whisper-server.log"


def clean(text: str) -> str:
    """Tidy spaces, and drop whisper's [BLANK_AUDIO]-style tags (it means "no speech")."""
    text = " ".join(text.split())
    return "" if text.startswith("[") and text.endswith("]") else text


class WhisperCpp(BufferedSTT):
    name = "whisper_cpp"
    knobs = {
        "model": {"type": "choice", "default": "base.en",
                  "choices": ["tiny.en", "base.en", "small.en", "large-v3-turbo"],
                  "help": "Whisper size. Bigger is more accurate but slower."},
        "driver": {"type": "choice", "default": "bindings", "choices": ["bindings", "server"],
                   "help": "bindings = pywhispercpp in this process; server = whisper-server over HTTP."},
        "server_url": {"type": "str", "default": "http://127.0.0.1:8178",
                       "help": "Where whisper-server listens (server driver)."},
        "server_bin": {"type": "str", "default": "whisper-server",
                       "help": "whisper-server program to start if nothing answers at server_url."},
        "threads": {"type": "int", "default": 4, "min": 1, "max": 16, "step": 1,
                    "help": "CPU threads (the GPU does most of the work when there is one)."},
    }

    def __init__(self, settings: dict, device: str):
        super().__init__(settings, device)
        self._model = None       # pywhispercpp model (bindings driver)
        self._server = None      # subprocess, only if WE started the server

    @classmethod
    def supports(cls, device: str) -> tuple[bool, str]:
        if device not in ("metal", "cuda", "cpu"):
            return False, f"unknown device {device!r}"
        if shutil.which("whisper-server"):
            return True, ""
        try:
            import pywhispercpp  # noqa: F401
        except ImportError:
            return False, "needs pywhispercpp (pip) or the whisper-server program (brew install whisper-cpp)"
        return True, ""

    # ---- loading ----

    def load(self) -> None:
        path = models.fetch(MODEL_URL.format(self.settings["model"]),
                            models.path("whisper.cpp", f"ggml-{self.settings['model']}.bin"))
        if self.settings["driver"] == "server":
            self._start_server(path)
        else:
            from pywhispercpp.model import Model     # heavy, so imported here
            # redirect_whispercpp_logs_to=None sends whisper.cpp's chatter to the bin
            self._model = Model(str(path), n_threads=self.settings["threads"],
                                print_progress=False, print_realtime=False,
                                print_timestamps=False, redirect_whispercpp_logs_to=None)
        # The first call is slow (GPU kernels compile); do it now, not mid-sentence.
        self.transcribe(np.zeros(RATE, np.float32))

    def reset(self) -> None:
        pass     # nothing is remembered between utterances (no_context)

    def close(self) -> None:
        """Stop the server, but only one we started ourselves."""
        if self._server is not None:
            self._server.terminate()
            try:
                self._server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._server.kill()
            self._server = None

    # ---- server driver ----

    def _server_up(self) -> bool:
        try:
            requests.get(self.settings["server_url"], timeout=1)
            return True
        except requests.RequestException:
            return False

    def _start_server(self, model_path: Path) -> None:
        if self._server_up():
            return       # someone else's server: use it, never stop it
        binary = shutil.which(self.settings["server_bin"])
        if binary is None:
            raise RuntimeError(f"{self.settings['server_bin']} not found; install whisper-cpp or use driver: bindings")
        url = urlparse(self.settings["server_url"])
        LOG_FILE.parent.mkdir(exist_ok=True)
        log = open(LOG_FILE, "w")
        self._server = subprocess.Popen(
            [binary, "-m", str(model_path), "--host", url.hostname, "--port", str(url.port or 8178),
             "-l", self.settings["language"], "-t", str(self.settings["threads"])],
            stdout=log, stderr=log)
        atexit.register(self.close)          # do not leave it running if Python exits
        for _ in range(60):                  # wait up to 30 s for the model to load
            if self._server.poll() is not None:
                raise RuntimeError(f"whisper-server exited early, see {LOG_FILE}")
            if self._server_up():
                return
            time.sleep(0.5)
        self.close()
        raise RuntimeError(f"whisper-server did not answer within 30 s, see {LOG_FILE}")

    def _transcribe_server(self, audio: np.ndarray) -> str:
        wav = io.BytesIO()
        sf.write(wav, audio, RATE, format="WAV", subtype="PCM_16")   # what the server expects
        reply = requests.post(
            self.settings["server_url"].rstrip("/") + "/inference",
            files={"file": ("audio.wav", wav.getvalue(), "audio/wav")},
            data={"response_format": "json", "temperature": "0",
                  "prompt": self.settings.get("prompt") or ""},
            timeout=60)
        reply.raise_for_status()
        return reply.json().get("text", "").strip()

    # ---- the one method BufferedSTT needs ----

    def transcribe(self, audio: np.ndarray) -> str:
        audio = np.asarray(audio, dtype=np.float32)
        if self.settings["driver"] == "server":
            return clean(self._transcribe_server(audio))
        options = dict(language=self.settings["language"], no_timestamps=True,
                       print_progress=False, print_realtime=False,
                       print_timestamps=False, temperature=0.0)
        if self.settings.get("prompt"):      # the bindings reject initial_prompt=None
            options["initial_prompt"] = self.settings["prompt"]
        segments = self._model.transcribe(audio, **options)
        return clean(" ".join(s.text for s in segments))
