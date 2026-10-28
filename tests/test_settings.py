"""Tests for the settings file: valid values are applied, bad ones ignored, saves merge.

Run on the ZenBook:  ./venv/bin/python -m pytest tests
"""
import json

import config


def test_valid_settings_are_applied(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_URL", "http://localhost:11434")
    monkeypatch.setattr(config, "MAX_STEP_CM", 20.0)
    monkeypatch.setattr(config, "ALLOW_UNLISTED_WRITES", False)
    f = tmp_path / "settings.json"
    f.write_text(json.dumps({"ollama_url": "http://192.168.1.148:11434/", "model": "qwen3:8b",
                             "allow_unlisted_writes": True, "limits": {"MAX_STEP_CM": 12}}))
    config.apply_settings(config.load_settings(f))
    assert config.OLLAMA_URL == "http://192.168.1.148:11434"     # trailing slash dropped
    assert config.MODEL == "qwen3:8b"
    assert config.ALLOW_UNLISTED_WRITES is True
    assert config.MAX_STEP_CM == 12.0


def test_bad_values_are_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_URL", "http://localhost:11434")
    monkeypatch.setattr(config, "MAX_STEP_CM", 20.0)
    monkeypatch.setattr(config, "MAX_JOINT_DEG", 30.0)
    f = tmp_path / "settings.json"
    f.write_text(json.dumps({"ollama_url": "not a url", "limits": {"MAX_STEP_CM": "big", "MAX_JOINT_DEG": 999,
                                                                 "NOT_A_LIMIT": 5}}))
    config.apply_settings(config.load_settings(f))
    assert config.OLLAMA_URL == "http://localhost:11434"
    assert config.MAX_STEP_CM == 20.0 and config.MAX_JOINT_DEG == 30.0


def test_missing_or_broken_file_is_fine(tmp_path):
    assert config.load_settings(tmp_path / "nope.json") == {}
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert config.load_settings(broken) == {}


def test_save_merges_and_keeps_unknown_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL", "llama3.1:8b")
    f = tmp_path / "deep" / "settings.json"          # the folder does not exist yet
    config.save_settings({"model": "llama3.1:8b", "someone_elses_key": 1}, f)
    merged = config.save_settings({"model": "qwen3:8b"}, f)
    assert merged == {"model": "qwen3:8b", "someone_elses_key": 1}
    assert json.loads(f.read_text())["someone_elses_key"] == 1
    assert config.MODEL == "qwen3:8b"


def test_old_flat_voice_keys_are_migrated(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"voice_on": True, "voice_confirm": False, "voice_verbosity": 2, "voice_persona": "B",
                                "follow_up_s": 5, "confirm_timeout_s": 30, "map_ros": True,
                                "voice_url": "http://x", "voice_start": "y", "model": "m"}))
    settings = config.load_settings(path)                     # migrates and rewrites the file
    assert settings["voice"]["on"] is True and settings["voice"]["persona"] == "B" and settings["voice"]["follow_up_s"] == 5
    assert "voice_on" not in settings and "voice_url" not in settings and "voice_start" not in settings
    assert settings["model"] == "m" and json.loads(path.read_text()) == settings


def test_voice_block_is_applied_and_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VOICE", dict(config.VOICE_DEFAULTS))
    f = tmp_path / "settings.json"
    f.write_text(json.dumps({"voice": {"on": True, "confirm": "yes", "verbosity": 2, "silence_ms": 900, "wake_threshold": 0.95,
                                       "input_device": "Jabra", "follow_up_s": 31}}))
    config.apply_settings(config.load_settings(f))
    assert config.VOICE["on"] is True and config.VOICE["verbosity"] == 2 and config.VOICE["silence_ms"] == 900
    assert config.VOICE["input_device"] == "Jabra"
    assert config.VOICE["confirm"] is True                    # a string is not a bool: ignored
    assert config.VOICE["wake_threshold"] == 0.5 and config.VOICE["follow_up_s"] == 8   # out of range: ignored


def test_voice_settings_are_validated():
    assert config.voice_settings_ok({"wake_threshold": 0.3, "silence_ms": 900, "persona": "B", "input_device": None}) is None
    assert "silence_ms" in config.voice_settings_ok({"silence_ms": 100})
    assert "verbosity" in config.voice_settings_ok({"verbosity": True})
    assert "ollama_url" in config.voice_settings_ok({"ollama_url": "http://x"})


def test_save_merges_the_voice_block_one_level_down(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VOICE", dict(config.VOICE_DEFAULTS))
    f = tmp_path / "settings.json"
    config.save_settings({"voice": {"on": True}}, f)
    merged = config.save_settings({"voice": {"silence_ms": 900}}, f)
    assert merged["voice"] == {"on": True, "silence_ms": 900} and config.VOICE["on"] is True and config.VOICE["silence_ms"] == 900


def test_voice_confirm_defaults_on():
    assert config.VOICE_DEFAULTS["confirm"] is True
