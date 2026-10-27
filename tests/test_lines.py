"""lines.py: what Rosa says at each moment, in two personas, with the facts always present.
Pure Python: runs on the Mac.  python3 -m pytest tests/test_lines.py"""
import pytest

import lines


def test_propose_lines_differ_by_persona_but_carry_the_facts():
    a = lines.line("propose", "A", label="Point 10", distance_cm=14, seconds=6)
    b = lines.line("propose", "B", label="Point 10", distance_cm=14, seconds=6)
    assert a == "Point 10. 14 centimetres, 6 seconds, path clear. Confirm?"
    assert a != b and "14" in b and "6" in b and b.endswith("Confirm?")


def test_refusal_has_no_personality():
    assert lines.line("refused", "A", reason="Too far") == lines.line("refused", "B", reason="Too far")


def test_result_is_trimmed_at_a_sentence():
    text = "Done. " + "Words " * 60 + ". The end."
    out = lines.line("result", "A", text=text)
    assert len(out) <= 200 and out.endswith(".")


def test_every_kind_renders_in_both_personas():
    facts = {"label": "Home position", "distance_cm": 3, "seconds": 2, "largest_turn_deg": 28, "reason": "x",
             "text": "ok", "x": 1, "y": 2, "z": 3, "group": "MoveIt", "count": 4, "name": "home position",
             "say": "go home", "moves": True, "groups": ["MoveIt", "UR driver"], "names": ["go home"], "ros": "/x"}
    for kind in lines.KINDS:
        for persona in lines.PERSONAS:
            assert lines.line(kind, persona, **facts)


def test_unknown_kind_raises():
    with pytest.raises(KeyError):
        lines.line("nope")
