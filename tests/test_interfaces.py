"""Tests for the description file and the write policy."""
import pathlib

import pytest

from discovery import Interface
from interfaces import Description, policy, report

EXAMPLE = pathlib.Path(__file__).parent.parent / "examples" / "ur_humble.interfaces.yaml"


def test_example_file_loads():
    d = Description.load(EXAMPLE)
    assert d.system.startswith("UR_humble")
    assert d.get("/touchdown_test/execute")["moves_robot"] is True
    assert d.get("/multiview_scan/capture")["moves_robot"] is False
    assert "3D view" in d.get("/multiview_scan/capture")["means"]


def test_moves_robot_defaults_to_true(tmp_path):
    f = tmp_path / "interfaces.yaml"
    f.write_text("system: test\ninterfaces:\n  /thing:\n    means: does a thing\n")
    assert Description.load(f).get("/thing")["moves_robot"] is True


def test_missing_file_is_empty():
    d = Description.load("/nowhere/interfaces.yaml")
    assert d.system == "" and d.entries == {}


def test_bad_file_is_an_error(tmp_path):
    f = tmp_path / "interfaces.yaml"
    f.write_text("- just\n- a list\n")
    with pytest.raises(ValueError):
        Description.load(f)


def test_policy():
    d = Description.load(EXAMPLE)
    assert policy(d, "/somebody/else", allow_unlisted=False)[0] == "refused"
    assert policy(d, "/somebody/else", allow_unlisted=True)[0] == "confirm"
    assert policy(d, "/touchdown_test/execute", allow_unlisted=False)[0] == "confirm"
    assert policy(d, "/multiview_scan/capture", allow_unlisted=False) == ("ok", "")


def test_report_marks_missing_and_undescribed():
    d = Description.load(EXAMPLE)
    graph = [Interface("/multiview_scan/capture", "service", "std_srvs/srv/Trigger"),
             Interface("/joint_states", "topic", "sensor_msgs/msg/JointState")]
    rows = {r["name"]: r for r in report(d, graph)}
    assert rows["/multiview_scan/capture"]["status"] == "ready"
    assert rows["/joint_states"]["status"] == "undescribed" and rows["/joint_states"]["moves_robot"] is None
    assert rows["/touchdown_test/execute"]["status"] == "missing" and rows["/touchdown_test/execute"]["kind"] == ""
