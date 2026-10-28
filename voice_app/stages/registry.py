"""Where each option named in settings.yaml lives.

Modules are imported only when an option is used (or listed), so a missing
library for one engine doesn't stop the others from working.
"""

import importlib

OPTIONS = {
    "trigger": {
        "push_to_talk": "stages.trigger.push_to_talk:PushToTalk",
        "openwakeword": "stages.trigger.wake_word:WakeWord",
        "both": "stages.trigger.both:Both",
    },
    "vad": {
        "silero": "stages.vad.silero:Silero",
        "webrtc": "stages.vad.webrtc:WebRTC",
    },
    "endpoint": {
        "silence": "stages.endpoint.silence:Silence",
        "adaptive": "stages.endpoint.adaptive:Adaptive",
        "smart_turn": "stages.endpoint.smart_turn:SmartTurn",
        "ptt_release": "stages.endpoint.ptt_release:PttRelease",
    },
    "stt": {
        "parakeet": "stages.stt.parakeet:Parakeet",
        "whisper_cpp": "stages.stt.whisper_cpp:WhisperCpp",
        "faster_whisper": "stages.stt.faster_whisper_engine:FasterWhisper",
        "moonshine": "stages.stt.moonshine_engine:Moonshine",
    },
}


def get_class(stage: str, name: str):
    """The class for option `name` of `stage` ("trigger", "vad", "endpoint", "stt")."""
    try:
        target = OPTIONS[stage][name]
    except KeyError:
        raise ValueError(f"unknown {stage} option {name!r}; known: {list(OPTIONS.get(stage, {}))}")
    module, cls = target.split(":")
    return getattr(importlib.import_module(module), cls)


def describe(device: str) -> dict:
    """Every option, its knobs, and whether it can run on `device`. Feeds GET /options."""
    out = {}
    for stage, names in OPTIONS.items():
        out[stage] = {}
        for name in names:
            try:
                cls = get_class(stage, name)
                ok, reason = cls.supports(device)
                knobs = cls.knobs
            except Exception as e:      # library not installed, import error...
                ok, reason, knobs = False, f"cannot import: {e}", {}
            out[stage][name] = {"supported": ok, "reason": reason, "knobs": knobs}
    return out
