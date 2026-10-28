"""Every stage, option and knob on the settings page has a plain-English explanation."""

import json

import config
from stages import registry

HELP = json.load(open(config.ROOT / "static" / "help.json"))
STT_COMMON = ["mode", "partial_every_ms", "language", "prompt"]   # shown by the page for every engine


def test_every_stage_has_help():
    for stage in list(registry.OPTIONS) + ["device", "presets"]:
        assert HELP["stages"].get(stage, {}).get("what"), stage


def test_every_option_and_knob_has_help():
    missing = []
    for stage, names in registry.OPTIONS.items():
        for name in names:
            if not HELP["options"].get(f"{stage}.{name}", {}).get("what"):
                missing.append(f"{stage}.{name}")
            for knob in registry.get_class(stage, name).knobs:
                if not HELP["knobs"].get(f"{stage}.{name}.{knob}", {}).get("what"):
                    missing.append(f"{stage}.{name}.{knob}")
    missing += [f"stt.{k}" for k in STT_COMMON if not HELP["knobs"].get(f"stt.{k}", {}).get("what")]
    assert not missing, f"no help text for: {missing}"


def test_no_help_for_settings_that_dont_exist():
    """Catches a renamed knob leaving stale help behind."""
    known = {f"stt.{k}" for k in STT_COMMON}
    for stage, names in registry.OPTIONS.items():
        for name in names:
            known |= {f"{stage}.{name}.{k}" for k in registry.get_class(stage, name).knobs}
    assert set(HELP["knobs"]) <= known, set(HELP["knobs"]) - known
