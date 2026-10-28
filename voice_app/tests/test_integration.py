"""The whole pipeline with real models, fed synthetic speech through FileSource."""

import json

import pytest

import config
from audio import FileSource
from pipeline import Pipeline
from tests.conftest import words


def baseline():
    """push_to_talk + silero + ptt_release + parakeet (batch): the phase 1 combination."""
    s = config.load(config.ROOT / "settings.example.yaml")
    assert (s["trigger"]["use"], s["vad"]["use"], s["endpoint"]["use"], s["stt"]["use"]) == \
        ("push_to_talk", "silero", "ptt_release", "parakeet")
    return s


def say_through(pipe, path, events):
    records = []
    pipe.on_record.append(records.append)
    pipe.run(FileSource(path, pad_end_s=0.5), events)
    pipe.on_record.remove(records.append)
    return records


@pytest.mark.model
def test_phase1_file_gives_text_timings_and_log(clip, tmp_path):
    log = tmp_path / "run.jsonl"
    pipe = Pipeline(baseline(), log_path=log)
    path = clip.path("move up five centimetres")
    # 0.5 s lead, so press at 0.3 s; speech ends ~2.4 s, release 0.1 s later.
    speech_end = FileSource(path).duration_s - 1.5
    recs = say_through(pipe, path, [(0.3, "ptt_down"), (speech_end + 0.1, "ptt_up")])
    assert len(recs) == 1
    r = recs[0]
    assert "move up" in words(r["final"]) and ("5" in r["final"] or "five" in words(r["final"]))
    assert r["started_by"] == "ptt"
    assert r["marks_audio_s"]["speech_start"] == pytest.approx(0.5, abs=0.15)
    assert r["marks_audio_s"]["speech_end"] == pytest.approx(speech_end, abs=0.2)
    assert r["derived"]["felt_delay_ms"] is not None
    print("\nfinal:", r["final"], "| derived:", r["derived"])
    line = json.loads(log.read_text().splitlines()[0])
    assert line["final"] == r["final"]


@pytest.mark.model
def test_same_wav_same_text(clip):
    pipe = Pipeline(baseline())
    path = clip.path("move left fifteen centimetres", voice="Samantha")
    end = FileSource(path).duration_s - 1.5
    events = [(0.3, "ptt_down"), (end + 0.1, "ptt_up")]
    a = say_through(pipe, path, events)[0]["final"]
    b = say_through(pipe, path, events)[0]["final"]
    assert a == b and "15" in a


@pytest.mark.model
def test_hands_free_wake_word_then_silence(clip):
    """Phase 3: "hey Jarvis, go home" with no hands -> "go home"."""
    s = baseline()
    s["trigger"]["use"], s["endpoint"]["use"] = "openwakeword", "silence"
    pipe = Pipeline(s)
    recs = say_through(pipe, clip.path(["hey Jarvis", 0.4, "go home"]), [])
    main = [r for r in recs if r["outcome"] != "no_speech"]
    assert len(main) == 1, recs
    assert main[0]["started_by"] == "wake"
    assert words(main[0]["final"]).endswith("go home")
    assert "jarvis" not in words(main[0]["final"])     # wake word trimmed by the pre-roll
    print("\nwake ->", main[0]["final"], main[0]["derived"])


@pytest.mark.model
def test_streaming_partials_through_pipeline(clip):
    """Phase 4: Parakeet MLX native streaming shows partial text while speaking."""
    s = baseline()
    s["stt"]["mode"] = "streaming"
    pipe = Pipeline(s)
    path = clip.path("move the tool over the inspection plate")
    end = FileSource(path).duration_s - 1.5
    r = say_through(pipe, path, [(0.3, "ptt_down"), (end + 0.1, "ptt_up")])[0]
    assert len(r["partials"]) >= 2, r["partials"]
    assert r["partials"][0][0] < r["marks_audio_s"]["turn_closed"]     # partials came before the end
    assert "inspection plate" in words(r["final"])
    print("\npartials:", r["partials"], "\nfinal:", r["final"], r["derived"])


@pytest.mark.model
@pytest.mark.parametrize("endpoint", ["silence", "adaptive", "smart_turn", "ptt_release"])
def test_every_endpoint_selectable_and_logged(clip, endpoint):
    """Phase 4: all four end-of-turn options run in the pipeline and are logged.
    A pause clip: "move left" ... 1.0 s ... "about ten centimetres"."""
    s = baseline()
    s["endpoint"]["use"] = endpoint
    s["stt"]["mode"] = "streaming"          # adaptive needs partial text
    pipe = Pipeline(s)
    path = clip.path(["move left", 1.0, "about ten centimetres"])
    end = FileSource(path).duration_s - 1.5
    events = [(0.3, "ptt_down")] + ([(end + 0.1, "ptt_up")] if endpoint == "ptt_release" else [])
    recs = say_through(pipe, path, events)
    r = recs[0]
    assert r["settings"]["endpoint"]["use"] == endpoint
    print(f"\n{endpoint}: {[x['final'] for x in recs]} endpoint_delay={r['derived']['endpoint_delay_ms']}"
          f" info={r['endpoint_info']}")
    if endpoint == "ptt_release":
        assert len(recs) == 1 and "ten" in words(r["final"]) or "10" in r["final"]


@pytest.mark.model
def test_moonshine_streaming_through_pipeline(clip):
    s = baseline()
    s["stt"]["use"], s["stt"]["mode"] = "moonshine", "streaming"
    pipe = Pipeline(s)
    path = clip.path("open the gripper and then close it")
    end = FileSource(path).duration_s - 1.5
    r = say_through(pipe, path, [(0.3, "ptt_down"), (end + 0.1, "ptt_up")])[0]
    assert r["partials"] and "gripper" in words(r["final"])
