"""Benchmark tools: scoring, clip trimming, synthetic set, and one small end-to-end run."""

import csv
import json

import numpy as np
import pytest
import yaml

import config
from bench import make_synthetic, record_set, run_bench, say, score, summarise


# ------------------------------------------------------------------ scoring

@pytest.mark.parametrize("raw, expected", [
    ("five", "5"),
    ("fifteen", "15"),
    ("one hundred and twenty", "120"),
    ("six point five", "6.5"),
    ("point five", "0.5"),
    ("minus sixty", "-60"),
    ("sixty three", "63"),
    ("forty-five degrees", "45 degrees"),
    ("x10y-60z70", "x 10 y -60 z 70"),
    ("5cm", "5 centimetres"),
    ("Move up 5 Centimeters.", "move up 5 centimetres"),
    ("20 mm", "20 millimetres"),
    ("rotate by 90°", "rotate by 90 degrees"),
    ("move to x ten y minus sixty z seventy", "move to x 10 y -60 z 70"),
])
def test_normalise(raw, expected):
    assert score.normalise(raw) == expected


def test_wer():
    assert score.wer("Move up five centimetres", "move up 5 cm.") == 0.0
    assert score.wer("go home", "go") == pytest.approx(0.5)
    assert score.wer("stop", "") == 1.0


def test_command_ok():
    assert not score.command_ok("move up fifteen", "move up fifty")
    assert score.command_ok("move up five centimetres", "Move up 5 cm.")
    assert score.command_ok("move to x ten y minus sixty z seventy", "Move to x10y-60z70.")
    assert not score.command_ok("move up five centimetres", "move down five centimetres")
    assert score.command_ok("move up a bit", "move up a little bit")      # filler words don't matter


# ----------------------------------------------------------------- trimming

def test_trim_clip_gives_exact_padding():
    rate = record_set.RATE
    raw = np.zeros(6 * rate, np.float32)
    raw[int(2.0 * rate):int(3.0 * rate)] = 0.3             # "speech" from 2.0 s to 3.0 s
    clip = record_set.trim_clip(raw, 2.0, 3.0)
    assert len(clip) == int(round((0.5 + 1.0 + 1.5) * rate))
    assert np.all(clip[:int(0.5 * rate)] == 0) and clip[int(0.5 * rate) + 10] == pytest.approx(0.3)
    # Too little silence in the recording: zeros are added instead.
    short = record_set.trim_clip(raw[int(1.9 * rate):int(3.2 * rate)], 0.1, 1.1)
    assert len(short) == len(clip)


def test_find_speech_on_synthetic_clip(clip):
    audio = clip("move up five centimetres")
    start, end = record_set.find_speech(audio)
    assert 0.3 < start < 0.8                               # padding is 0.5 s
    trimmed = record_set.trim_clip(np.concatenate([np.zeros(16000, np.float32), audio]), start + 1.0, end + 1.0)
    assert len(trimmed) / 16000 == pytest.approx(0.5 + (end - start) + 1.5, abs=0.001)


# ------------------------------------------------------------- synthetic set

@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    if not say.available():
        pytest.skip("macOS `say` not available")
    out = tmp_path_factory.mktemp("synthetic")
    rows = make_synthetic.build(out, voices=["Daniel"], only=["cmd_001", "pause_001", "wake_002"])
    return out, rows


def test_make_synthetic(synth):
    out, rows = synth
    by_id = {r["id"]: r for r in rows}
    assert (out / "cmd_001-Daniel.wav").exists() and (out / "manifest.csv").exists()
    cmd, pause, wake = by_id["cmd_001"], by_id["pause_001"], by_id["wake_002"]
    assert cmd["text"] == "move up five centimetres" and float(cmd["speech_start_s"]) == 0.5
    assert pause["text"] == "move left about ten centimetres"
    assert float(pause["speech_end_s"]) - float(pause["speech_start_s"]) > 2.5      # includes the 1 s pause
    assert wake["text"] == "go home"
    assert float(wake["speech_start_s"]) == pytest.approx(float(wake["wake_end_s"]) + 0.8, abs=0.001)
    # A second call skips existing clips and keeps the same manifest.
    before = (out / "cmd_001-Daniel.wav").stat().st_mtime_ns
    again = make_synthetic.build(out, voices=["Daniel"], only=["cmd_001", "pause_001", "wake_002"])
    assert (out / "cmd_001-Daniel.wav").stat().st_mtime_ns == before
    assert [r["clip"] for r in again] == [r["clip"] for r in rows]
    assert str(again[2]["wake_end_s"]) == str(rows[2]["wake_end_s"])


def test_noise_clips_are_made(tmp_path):
    rows = make_synthetic.build(tmp_path, voices=["Daniel"], only=["noise_001", "noise_003", "noise_004"],
                                noise_s=3)
    assert len(rows) == 3
    for r in rows:
        assert (tmp_path / f"{r['clip']}.wav").stat().st_size > 90000          # ~3 s of 16-bit audio


# ----------------------------------------------------------------- end to end

@pytest.mark.model
def test_run_bench_end_to_end(synth, tmp_path):
    out, rows = synth
    settings = config.load(config.ROOT / "settings.example.yaml")
    settings["stt"]["use"] = "moonshine"                  # small and fast
    settings["stt"]["mode"] = "batch"
    settings["endpoint"]["use"] = "silence"               # the endpoint decides, no release is sent
    preset = tmp_path / "fast.yaml"
    preset.write_text(yaml.safe_dump(settings))

    result = run_bench.run(preset, "synthetic", kinds=["command", "pause", "wake"], out=tmp_path / "run.jsonl",
                           set_dir=out, quiet=True)
    records = [json.loads(line) for line in result.read_text().splitlines()]
    by_clip = {}
    for r in records:
        by_clip.setdefault(r["clip"], []).append(r)
    # Button preset: command and pause clips run, the wake clip can't be triggered and is skipped.
    assert set(by_clip) == {"cmd_001-Daniel", "pause_001-Daniel"}
    main = by_clip["cmd_001-Daniel"][0]
    assert main["synthetic"] is True and main["main"] is True
    assert set(main["scores"]) >= {"wer", "command_ok", "early_cutoff"}
    assert main["scores"]["command_ok"] is True
    assert main["derived"]["felt_delay_ms"] is not None

    row = summarise.summarise_file(result)
    assert row["run"].endswith("(synthetic)") and row["clips"] == 2
    assert row["felt_median_ms"] is not None
    assert "| run |" in summarise.to_markdown([row])
