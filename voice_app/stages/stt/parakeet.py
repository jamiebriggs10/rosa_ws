"""Speech-to-text option: NVIDIA Parakeet TDT 0.6B v3 (25 European languages).

One plug-in, two backends hidden inside it, so settings.yaml only ever says
"parakeet" and the machine decides how it runs:

  mlx          parakeet-mlx, runs on the Apple GPU (Metal). Mac only.
               Can stream natively: it keeps encoder state between chunks, so
               each new chunk costs a little work instead of re-reading the
               whole utterance (which is what pseudo-streaming does).
  sherpa_onnx  sherpa-onnx with an int8 ONNX export. Runs on any CPU, and on
               an NVIDIA GPU with provider "cuda". Batch only; "streaming"
               mode falls back to pseudo-streaming from BufferedSTT.
  nemo         the original NVIDIA toolkit. Needs PyTorch, which is not
               installed (ask Jamie first), so choosing it gives a clear error.

backend "auto" picks mlx on metal and sherpa_onnx everywhere else.
"""

import time

import numpy as np

import models
from stages.base import RATE
from stages.stt.buffered import BufferedSTT

SHERPA_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
              "{}.tar.bz2")


class Parakeet(BufferedSTT):
    name = "parakeet"
    knobs = {
        "model": {"type": "str", "default": "mlx-community/parakeet-tdt-0.6b-v3",
                  "help": "Hugging Face repo of the MLX weights (mlx backend)."},
        "backend": {"type": "choice", "default": "auto",
                    "choices": ["auto", "mlx", "sherpa_onnx", "nemo"],
                    "help": "auto = mlx on Mac, sherpa_onnx elsewhere. nemo needs PyTorch (not installed)."},
        "sherpa_model": {"type": "str", "default": "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
                         "help": "Folder name of the sherpa-onnx model in models/."},
        "threads": {"type": "int", "default": 4, "min": 1, "max": 32, "step": 1,
                    "help": "CPU threads (sherpa_onnx backend)."},
        "stream_chunk_ms": {"type": "int", "default": 640, "min": 80, "max": 2000, "step": 80,
                            "help": "Native streaming (mlx): audio fed to the model per step."},
        "context_left": {"type": "int", "default": 256, "min": 32, "max": 512, "step": 32,
                         "help": "Native streaming (mlx): encoder frames of past context kept (1 frame = 80 ms)."},
        "context_right": {"type": "int", "default": 256, "min": 2, "max": 256, "step": 2,
                          "help": "Native streaming (mlx): look-ahead frames. Smaller = faster text, less accurate."},
        # Measured 8 Oct on synthetic clips: native streaming dropped or garbled
        # words with small look-ahead/chunks (e.g. "Move the the inspection plate").
        # So by default the partials come from the stream (for the page and the
        # adaptive endpoint) and the FINAL text from one batch pass (~30-40 ms here).
        "final_pass": {"type": "bool", "default": True,
                       "help": "Streaming (mlx): final text from a batch pass over the whole utterance."},
    }

    @property
    def streaming_native(self):
        # Only the mlx backend streams by itself, and only in streaming mode.
        return self.backend == "mlx" and self.settings.get("mode") == "streaming"

    @classmethod
    def supports(cls, device):
        return True, ""     # sherpa_onnx runs anywhere; backend checks are in load()

    # ---- loading -------------------------------------------------------

    def load(self):
        backend = self.settings["backend"]
        if backend == "auto":
            backend = "mlx" if self.device == "metal" else "sherpa_onnx"
        if backend == "nemo":
            raise RuntimeError("parakeet backend 'nemo' needs PyTorch/NeMo, which is not installed. "
                               "Use 'mlx' (Mac) or 'sherpa_onnx'.")
        if backend == "mlx" and self.device != "metal":
            raise RuntimeError(f"parakeet backend 'mlx' needs the Apple GPU (device is {self.device!r}). "
                               "Use 'sherpa_onnx' or 'auto'.")
        self.backend = backend
        t0 = time.perf_counter()
        if backend == "mlx":
            self._load_mlx()
        else:
            self._load_sherpa()
        self.load_s = time.perf_counter() - t0
        # The first run compiles GPU kernels / allocates buffers; do it now
        # on silence so the first real utterance is not slowed.
        t0 = time.perf_counter()
        self.transcribe(np.zeros(RATE, np.float32))
        self.warmup_s = time.perf_counter() - t0

    def _load_mlx(self):
        import parakeet_mlx
        from huggingface_hub import snapshot_download
        # Keep the weights in models/ (copyable to another machine); from_pretrained
        # accepts a local folder containing config.json + model.safetensors.
        # One folder per model, and no network call once it's there (works offline).
        folder = models.path("parakeet-mlx", self.settings["model"].split("/")[-1])
        if not (folder / "config.json").exists():
            snapshot_download(self.settings["model"], local_dir=str(folder))
        self.model = parakeet_mlx.from_pretrained(str(folder))

    def _load_sherpa(self):
        import sherpa_onnx
        name = self.settings["sherpa_model"]
        folder = models.fetch_tarball(SHERPA_URL.format(name), models.path(name))
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(folder / "encoder.int8.onnx"),
            decoder=str(folder / "decoder.int8.onnx"),
            joiner=str(folder / "joiner.int8.onnx"),
            tokens=str(folder / "tokens.txt"),
            model_type="nemo_transducer",
            num_threads=int(self.settings["threads"]),
            provider="cuda" if self.device == "cuda" else "cpu",
        )

    def reset(self):
        self._close_stream()

    # ---- batch ---------------------------------------------------------

    def transcribe(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        if self.backend == "mlx":
            import mlx.core as mx
            from parakeet_mlx.audio import get_logmel
            # The mel step needs float32 input (bfloat16 breaks its FFT trick);
            # the model weights are bfloat16, so cast the mel after.
            mel = get_logmel(mx.array(audio), self.model.preprocessor_config).astype(mx.bfloat16)
            return self.model.generate(mel)[0].text
        stream = self.recognizer.create_stream()
        stream.accept_waveform(RATE, audio)
        self.recognizer.decode_stream(stream)
        return stream.result.text.strip()

    # ---- native streaming (mlx only) -----------------------------------

    def start(self, preroll):
        self._close_stream()
        if not self.streaming_native:
            return super().start(preroll)
        ctx = (int(self.settings["context_left"]), int(self.settings["context_right"]))
        self._stream = self.model.transcribe_stream(context_size=ctx)
        self._stream.__enter__()
        self._pending = np.asarray(preroll, dtype=np.float32)
        self._everything = [self._pending]     # the whole utterance, for the final pass
        self._last_partial = ""
        self._native_chunk = int(self.settings["stream_chunk_ms"] * RATE / 1000)

    def feed(self, frame):
        if self._stream is None:
            return super().feed(frame)
        # Frames are 32 ms; the model is fed bigger chunks to keep per-call overhead low.
        self._pending = np.concatenate([self._pending, frame])
        self._everything.append(frame)
        if len(self._pending) < self._native_chunk:
            return None
        self._push()
        text = self._stream.result.text
        if text == self._last_partial:
            return None
        self._last_partial = text
        return text

    def finish(self):
        if self._stream is None:
            return super().finish()
        if self.settings["final_pass"]:
            self._close_stream()             # back to normal attention for the batch pass
            return self.transcribe(np.concatenate(self._everything))
        self._push()
        text = self._stream.result.text      # finalized + draft tokens
        self._close_stream()
        return text

    def _push(self):
        if len(self._pending):
            import mlx.core as mx
            self._stream.add_audio(mx.array(self._pending))
            self._pending = np.zeros(0, np.float32)

    def _close_stream(self):
        # Leaving the context restores the model's normal attention, which batch mode needs.
        stream = getattr(self, "_stream", None)
        self._stream = None
        if stream is not None:
            stream.__exit__(None, None, None)

    _stream = None
