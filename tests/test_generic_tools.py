"""Tests for the generic tools: summaries of real messages, and the write policy with a fake node."""
import pathlib

import pytest
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, Int32, String
from std_msgs.msg import Header

import config
import generic_tools
import robot_tools
from discovery import Interface
from interfaces import Description

EXAMPLE = pathlib.Path(__file__).parent.parent / "examples" / "ur_humble.interfaces.yaml"


# -- summaries ------------------------------------------------------------------

def test_bool_and_string_and_unknown():
    assert generic_tools.summarise(Bool(data=True), "std_msgs/msg/Bool") == "true"
    assert generic_tools.summarise(String(data="hello"), "std_msgs/msg/String") == "hello"
    assert generic_tools.summarise(Int32(data=7), "std_msgs/msg/Int32") == '{"data": 7}'


def test_path_summary_counts_and_measures():
    path = Path()
    path.header.frame_id = "world"
    for x in (0.0, 0.1, 0.2):
        p = PoseStamped()
        p.pose.position.x = x
        p.pose.position.z = 0.05
        path.poses.append(p)
    text = generic_tools.summarise(path, "nav_msgs/msg/Path")
    assert "3 poses" in text and "'world'" in text and "20.0 cm long" in text
    assert "[0.0, 0.0, 5.0]" in text and "[20.0, 0.0, 5.0]" in text
    assert generic_tools.summarise(Path(), "nav_msgs/msg/Path") == "an empty path (0 poses)"


def test_point_cloud_summary():
    from sensor_msgs_py import point_cloud2
    cloud = point_cloud2.create_cloud_xyz32(Header(frame_id="world"), [[0.0, 0.0, 0.0], [0.4, -0.2, 0.1]])
    text = generic_tools.summarise(cloud, "sensor_msgs/msg/PointCloud2")
    assert "2 points" in text and "x 0.0 to 40.0" in text and "y -20.0 to 0.0" in text and "z 0.0 to 10.0" in text


# -- the policy, with a fake node so no ROS is needed --------------------------------

class FakePublisher:
    def __init__(self, listeners):
        self.listeners, self.sent = listeners, []
    def get_subscription_count(self):
        return self.listeners
    def publish(self, msg):
        self.sent.append(msg)


class FakeClock:
    def now(self):
        class T:
            def to_msg(self):
                from builtin_interfaces.msg import Time
                return Time()
        return T()


class FakeNode:
    def __init__(self, listeners=1):
        self.clients, self.publisher = [], FakePublisher(listeners)
    def create_client(self, srv, name):
        self.clients.append(name)
        raise AssertionError("the test should never reach the real call")
    def create_publisher(self, msg_type, topic, depth):
        return self.publisher
    def get_clock(self):
        return FakeClock()


GRAPH = [Interface("/multiview_scan/capture", "service", "std_srvs/srv/Trigger"),
         Interface("/touchdown_test/execute", "service", "std_srvs/srv/Trigger"),
         Interface("/somebody/else", "service", "std_srvs/srv/Trigger"),
         Interface("/compute_cartesian_path", "service", "moveit_msgs/srv/GetCartesianPath"),
         Interface("/clicked_point", "topic", "geometry_msgs/msg/PointStamped"),
         Interface("/joint_states", "topic", "sensor_msgs/msg/JointState")]


@pytest.fixture
def fake(monkeypatch):
    node = FakeNode()
    monkeypatch.setattr(generic_tools, "_graph", lambda: GRAPH)
    monkeypatch.setattr(generic_tools, "_description", Description.load(EXAMPLE))
    monkeypatch.setattr(generic_tools, "_publishers", {})
    monkeypatch.setattr(generic_tools, "SUBSCRIBER_WAIT_S", 0.05)
    monkeypatch.setattr(robot_tools, "_get_robot", lambda: node)
    monkeypatch.setattr(config, "ALLOW_UNLISTED_WRITES", False)
    robot_tools.events.clear()
    robot_tools.mark_command_time()
    return node


def test_unlisted_service_is_refused(fake):
    out = generic_tools.call_trigger.invoke({"service": "/somebody/else"})
    assert out.startswith("REFUSED") and "interfaces.yaml" in out
    assert robot_tools.events[-1]["outcome"] == "refused" and fake.clients == []


def test_unknown_service_is_reported_not_waited_for(fake):
    out = generic_tools.call_trigger.invoke({"service": "/not/running"})
    assert out.startswith("FAILED") and "not a service" in out and fake.clients == []


def test_wrong_type_is_reported(fake):
    out = generic_tools.call_trigger.invoke({"service": "/compute_cartesian_path"})
    assert "not a Trigger" in out


def test_moves_robot_service_asks_the_operator_and_declines(fake, monkeypatch):
    asked = {}
    monkeypatch.setattr(robot_tools, "_confirm_handler", lambda what, preview=None: asked.update(what=what) or False)
    out = generic_tools.call_trigger.invoke({"service": "/touchdown_test/execute"})
    assert "touchdown" in asked["what"] and out.startswith("Cancelled")
    assert robot_tools.events[-1]["outcome"] == "cancelled" and fake.clients == []


def test_safe_service_does_not_ask(fake, monkeypatch):
    monkeypatch.setattr(robot_tools, "_confirm_handler", lambda *a: pytest.fail("should not ask"))
    with pytest.raises(AssertionError, match="never reach the real call"):
        generic_tools.call_trigger.invoke({"service": "/multiview_scan/capture"})
    assert fake.clients == ["/multiview_scan/capture"]


def test_publish_point_converts_cm_and_reports_listeners(fake, monkeypatch):
    monkeypatch.setattr(robot_tools, "_confirm_handler", lambda *a: pytest.fail("clicked_point is marked safe"))
    out = generic_tools.publish_point.invoke({"topic": "/clicked_point", "x": 30, "y": -60, "z": 5})
    assert out.startswith("Sent") and "1 listening" in out
    msg = fake.publisher.sent[0]
    assert (msg.point.x, msg.point.y, msg.point.z) == (0.3, -0.6, 0.05) and msg.header.frame_id == config.BASE_FRAME


def test_publish_point_with_nobody_listening_fails(fake):
    fake.publisher.listeners = 0
    out = generic_tools.publish_point.invoke({"topic": "/clicked_point", "x": 1, "y": 2, "z": 3})
    assert out.startswith("FAILED") and "nothing is listening" in out


def test_publish_point_wrong_type(fake):
    out = generic_tools.publish_point.invoke({"topic": "/joint_states", "x": 1, "y": 2, "z": 3})
    assert "not a PointStamped" in out and fake.publisher.sent == []


def test_capabilities_text_lists_meanings_and_missing(fake):
    text = generic_tools.capabilities_text()
    assert text.startswith("This system: UR_humble")
    assert "/multiview_scan/capture (std_srvs/srv/Trigger): Take one 3D view" in text
    assert "/touchdown_test/execute" in text and "[moves the robot]" in text
    assert "MISSING /inspection/raster_path" in text
    assert "Read-only (not described):" in text and "/compute_cartesian_path (moveit_msgs/srv/GetCartesianPath)" in text


def test_gate_passes_trigger_facts_to_the_operator(fake, monkeypatch):
    got = {}
    monkeypatch.setattr(robot_tools, "_confirm_handler", lambda what, preview=None: got.update(preview=preview) or False)
    generic_tools.call_trigger.invoke({"service": "/touchdown_test/execute"})
    facts = got["preview"]["facts"]
    assert facts["kind"] == "trigger" and facts["name"] == "/touchdown_test/execute"
    assert facts["label"] and "Run the previewed touchdown" in facts["label"]
