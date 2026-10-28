"""robot_tools with a fake robot: the facts a confirm card carries, and the running tool's name.

Needs rclpy's message types, so it runs in the ros:humble Docker image like the console tests.
"""
import pytest
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

import config
import robot_tools


def trajectory(seconds=4.0):
    t = RobotTrajectory()
    t.joint_trajectory.joint_names = [n for n, _ in config.JOINTS]
    for k, frac in enumerate((0.0, 1.0)):
        p = JointTrajectoryPoint()
        p.positions = [0.1 * k] * len(config.JOINTS)
        p.time_from_start = Duration(sec=int(seconds * frac), nanosec=0)
        t.joint_trajectory.points.append(p)
    return t


class FakeRobot:
    def __init__(self):
        self.joints_deg = {n: 10.0 for n, _ in config.JOINTS}
        self.pose = Pose()
        self.pose.position.x, self.pose.position.y, self.pose.position.z = 0.30, -0.12, 0.45

    def tool_pose(self, timeout_s=2.0):
        return self.pose

    def plan_line(self, targets):
        self.last_target_cm = [round(getattr(targets[-1].position, a) * 100, 1) for a in "xyz"]
        return 1.0, trajectory()

    def plan_joints(self, target_deg):
        return trajectory(2.0)

    def execute(self, trajectory):
        return True


@pytest.fixture
def fake_robot(monkeypatch):
    robot = FakeRobot()
    monkeypatch.setattr(robot_tools, "_get_robot", lambda: robot)
    robot_tools.events.clear()
    robot_tools.mark_command_time()
    return robot


def declining_handler(got):
    def handler(what, preview=None):
        got.update(what=what, preview=preview, tool=robot_tools._current_event.get("tool"))
        return False
    return handler


def test_move_line_facts_carry_distance_and_label(fake_robot, monkeypatch):
    got = {}
    monkeypatch.setattr(robot_tools, "_confirm_handler", declining_handler(got))
    robot_tools.move_by.invoke({"dx": 0, "dy": 0, "dz": 5})
    f = got["preview"]["facts"]
    assert f["kind"] == "line" and f["label"] == "Up 5 cm"
    assert f["from_cm"] == [30.0, -12.0, 45.0] and f["to_cm"] == [30.0, -12.0, 50.0]
    assert f["distance_cm"] == 5.0 and f["seconds"] == 4.0
    assert got["preview"]["path_deg"]                     # the preview itself is unchanged


def test_move_to_label_names_the_target(fake_robot, monkeypatch):
    got = {}
    monkeypatch.setattr(robot_tools, "_confirm_handler", declining_handler(got))
    robot_tools.move_to.invoke({"x": 36, "y": -4, "z": 34})
    assert got["preview"]["facts"]["label"] == "x 36, y -4, z 34"


def test_joint_and_home_facts(fake_robot, monkeypatch):
    got = {}
    monkeypatch.setattr(robot_tools, "_confirm_handler", declining_handler(got))
    robot_tools.move_joint.invoke({"joint": 2, "degrees": 10})
    f = got["preview"]["facts"]
    assert f["kind"] == "joint" and f["label"] == "Joint 2 by 10°" and f["largest_turn_deg"] == 10.0 and f["seconds"] == 2.0
    robot_tools.go_home.invoke({})
    f = got["preview"]["facts"]
    assert f["kind"] == "joint" and f["label"] == "Home position" and f["largest_turn_deg"] > 0


def test_reports_failures_names_the_running_tool(fake_robot, monkeypatch):
    got = {}
    monkeypatch.setattr(robot_tools, "_confirm_handler", declining_handler(got))
    robot_tools.move_by.invoke({"dx": 1, "dy": 0, "dz": 0})
    assert got["tool"] == "move_by"
    assert robot_tools.events[-1]["tool"] == "move_by" and robot_tools.events[-1]["outcome"] == "cancelled"


def test_move_to_keeps_a_coordinate_that_is_left_out(fake_robot, monkeypatch):
    monkeypatch.setattr(robot_tools, "_confirm_handler", lambda what, preview=None: True)
    robot_tools.move_to.invoke({"x": 20.0, "y": -10.0})          # no z: it stays where it is
    assert fake_robot.last_target_cm == [20.0, -10.0, round(robot_tools._cm(fake_robot.tool_pose())[2], 1)]
    assert "z" in robot_tools.move_to.description and "keep" in robot_tools.move_to.description.lower()
