"""Reading, merging and saving settings.yaml (and presets/*.yaml)."""

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
SETTINGS = ROOT / "settings.yaml"
PRESETS = ROOT / "presets"

# Knobs in the `stt:` block that apply to every engine, not just one.
STT_COMMON = ("mode", "partial_every_ms", "language", "prompt")

HEADER = ("# Voice Control Prototype: current settings (the web page rewrites this file).\n"
          "# Each stage has `use:` (the option in force) and one block of knobs per option.\n"
          "# What each knob does: docs/02-options.md. Commented original: settings.example.yaml\n\n")


def load(path=SETTINGS) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def save(settings: dict, path=SETTINGS) -> None:
    with open(path, "w") as f:
        f.write(HEADER)
        yaml.safe_dump(settings, f, sort_keys=False, default_flow_style=False)


def merge(base: dict, update: dict) -> dict:
    """A copy of `base` with `update` laid over it, nested dicts merged key by key.
    This is what lets the page send just {"endpoint": {"silence": {"silence_ms": 400}}}."""
    out = copy.deepcopy(base)
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def option_settings(settings: dict, stage: str) -> tuple[str, dict]:
    """(option name, its knobs) for one stage, ready to hand to the plug-in."""
    block = settings[stage]
    name = block["use"]
    knobs = dict(block.get(name) or {})
    if stage == "stt":
        # Engines also need the shared knobs (mode, prompt...).
        knobs = {k: block[k] for k in STT_COMMON if k in block} | knobs
    if stage == "trigger" and name == "both":
        # `both` contains a wake-word detector, so it needs the wake word's knobs.
        knobs.setdefault("openwakeword", dict(block.get("openwakeword") or {}))
    return name, knobs


def preset_path(name: str) -> Path:
    if not name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("preset names may only use letters, digits, - and _")
    return PRESETS / f"{name}.yaml"


def list_presets() -> list[str]:
    return sorted(p.stem for p in PRESETS.glob("*.yaml"))
