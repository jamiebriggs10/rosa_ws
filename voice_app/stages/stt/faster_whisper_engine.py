"""faster-whisper: Whisper run by CTranslate2.

Same Whisper models as whisper.cpp, so accuracy should match; what differs is
speed per platform. CTranslate2 has no Metal backend, so on the Mac it runs on
the CPU (int8); on the lab PC it runs on the NVIDIA GPU (float16).

The file is called faster_whisper_engine.py so it does not shadow the
`faster_whisper` library it imports.
"""

import numpy as np

from stages.base import RATE
from stages.stt.buffered import BufferedSTT


class FasterWhisper(BufferedSTT):
    name = "faster_whisper"
    knobs = {
        "model": {"type": "choice", "default": "small.en",
                  "choices": ["tiny.en", "base.en", "small.en", "medium.en", "large-v3", "large-v3-turbo"],
                  "help": "Whisper size. Bigger is more accurate and slower."},
        "compute_type": {"type": "choice", "default": "auto",
                         "choices": ["auto", "int8", "float16", "float32"],
                         "help": "Number format. auto = int8 on cpu/metal, float16 on cuda."},
        "beam_size": {"type": "int", "default": 1, "min": 1, "max": 10, "step": 1,
                      "help": "Candidates kept while decoding. 1 = greedy (fastest)."},
        "threads": {"type": "int", "default": 4, "min": 1, "max": 16, "step": 1,
                    "help": "CPU threads (ignored on cuda)."},
        "use_hotwords": {"type": "bool", "default": True,
                         "help": "Give the domain words as hotwords (else as an initial prompt)."},
    }

    @classmethod
    def supports(cls, device: str) -> tuple[bool, str]:
        return True, ""     # CPU works everywhere

    def load(self) -> None:
        from faster_whisper import WhisperModel     # heavy import, so only here

        # No Metal backend in CTranslate2: Mac ("metal") falls back to the CPU.
        self._device = "cuda" if self.device == "cuda" else "cpu"
        compute = self.settings["compute_type"]
        if compute == "auto":
            compute = "float16" if self._device == "cuda" else "int8"
        self._model = WhisperModel(
            self.settings["model"], device=self._device, compute_type=compute,
            cpu_threads=int(self.settings["threads"]), download_root="models/faster-whisper",
        )
        self.transcribe(np.zeros(RATE, np.float32))     # warm-up: first call is slow

    def reset(self) -> None:
        pass    # no state is kept between utterances

    def transcribe(self, audio: np.ndarray) -> str:
        prompt = self.settings.get("prompt") or None
        # Hotwords and initial_prompt end up in the same place in the model's
        # input, so we use one or the other, never both (that would repeat the words).
        hotwords = prompt if self.settings["use_hotwords"] else None
        initial_prompt = None if self.settings["use_hotwords"] else prompt
        segments, _info = self._model.transcribe(
            np.asarray(audio, dtype=np.float32),
            language=self.settings.get("language") or None,
            beam_size=int(self.settings["beam_size"]),
            vad_filter=False,                   # our pipeline already did VAD
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            condition_on_previous_text=False,   # each utterance stands alone
            without_timestamps=True,
        )
        # segments is a lazy generator: the decoding happens as we join it.
        return " ".join(s.text.strip() for s in segments).strip()
