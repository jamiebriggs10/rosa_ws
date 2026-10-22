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


def test_voice_settings_are_applied_and_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VOICE_URL", "http://127.0.0.1:8100", raising=False)
    monkeypatch.setattr(config, "VOICE_ON", False, raising=False)
    monkeypatch.setattr(config, "VOICE_CONFIRM", False, raising=False)
    f = tmp_path / "settings.json"
    f.write_text(json.dumps({"voice_url": "http://10.0.0.2:8100/", "voice_on": True, "voice_confirm": "yes"}))
    config.apply_settings(config.load_settings(f))
    assert config.VOICE_URL == "http://10.0.0.2:8100"     # trailing slash dropped
    assert config.VOICE_ON is True
    assert config.VOICE_CONFIRM is False                    # a string is not a bool: ignored
    f.write_text(json.dumps({"voice_url": "ten dot two"}))
    config.apply_settings(config.load_settings(f))
    assert config.VOICE_URL == "http://10.0.0.2:8100"      # not a web address: ignored
