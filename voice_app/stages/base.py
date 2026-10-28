"""The contract every stage option implements.

The pipeline only ever talks to these four interfaces, so any option can be
swapped for another from settings.yaml without touching the pipeline.
See docs/01-architecture.md.

Audio frames are numpy float32 arrays of 512 samples (32 ms) at 16 kHz, mono,
values in -1..1. A stage that needs a different frame size buffers internally.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

Frame = np.ndarray  # shape (512,), float32, 16 kHz
RATE = 16000


class Stage(ABC):
    """What every option has in common."""

    name: str = "unnamed"   # the key used in settings.yaml

    # What the settings page shows for this option, one entry per knob:
    #   {"threshold": {"type": "float", "default": 0.5, "min": 0, "max": 1,
    #                  "step": 0.05, "help": "..."}}
    # type is one of float | int | str | bool | choice (with "choices": [...]) | list
    knobs: dict = {}

    def __init__(self, settings: dict, device: str):
        # This option's own knobs from settings.yaml, with defaults filled in.
        self.settings = {k: v.get("default") for k, v in self.knobs.items()} | dict(settings or {})
        self.device = device       # "cuda" | "metal" | "cpu" (already resolved from "auto")

    @classmethod
    def supports(cls, device: str) -> tuple[bool, str]:
        """Can this option run on this machine? Returns (ok, reason if not)."""
        return True, ""

    def load(self) -> None:
        """Slow set-up (downloading / loading a model). Called once, before use."""

    def reset(self) -> None:
        """Forget everything about the last utterance."""

    def on_event(self, event: str) -> None:
        """Something happened outside the audio: "ptt_down", "ptt_up",
        "wake_on", "wake_off". Most options ignore these."""


class Trigger(Stage):
    """Decides when to start listening (wake word, push-to-talk...)."""

    # After process() returns True: "wake" (go to ARMED and wait for speech)
    # or "ptt" (a button: go straight to LISTENING).
    last_fire: str = "wake"

    @abstractmethod
    def process(self, frame: Frame) -> bool:
        """Called for every frame while idle. True = fire now."""


class VAD(Stage):
    """Voice activity detection: is this frame speech?"""

    @abstractmethod
    def speech_prob(self, frame: Frame) -> float:
        """Probability 0..1 that this frame contains speech."""

    @property
    def threshold(self) -> float:
        """speech_prob at or above this counts as speech."""
        return float(self.settings.get("threshold", 0.5))

    @property
    def min_speech_frames(self) -> int:
        """Speech must last this many frames in a row to count as the start of
        speech (stops a cough or a click from starting a turn)."""
        ms = float(self.settings.get("min_speech_ms", 0))
        return max(1, int(round(ms / (512 / RATE * 1000))))


class EndOfTurn(Stage):
    """Decides when the speaker has finished."""

    @abstractmethod
    def update(self, frame: Frame, is_speech: bool, partial_text: str) -> bool:
        """Called for every frame while listening. True = speaker has finished."""


class SpeechToText(Stage):
    """Turns the audio of one utterance into text.

    A batch engine buffers in feed() and does the work in finish().
    A streaming engine does the work in feed() and may return partial text.

    Besides its own knobs, every engine's settings include the common ones
    from the `stt:` block: mode (batch | streaming), partial_every_ms,
    language, prompt.
    """

    streaming_native: bool = False   # True if the engine streams by itself

    @abstractmethod
    def start(self, preroll: np.ndarray) -> None:
        """A new utterance begins. `preroll` is the audio just before it."""

    @abstractmethod
    def feed(self, frame: Frame) -> str | None:
        """Add a frame. Return new partial text, or None if nothing new."""

    @abstractmethod
    def finish(self) -> str:
        """The utterance is over. Return the final text."""
