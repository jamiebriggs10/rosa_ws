"""The actions the language model is allowed to take on the UR12e.

Each function decorated with @tool becomes something the LLM can call.
The docstring of each tool is what the model reads to decide when to use it,
so the docstrings are written for the model, not for people.

ROS concepts used here, briefly:
- A *node* is one running program on the ROS network.
- A *topic* is a named stream of messages. /joint_states carries the arm's
  joint angles many times a second; we subscribe to read them.
- An *action* is a long-running request with feedback and a result. Moving
  the arm along a trajectory is an action: we send a goal, the controller
  executes it, and we can cancel it part way.

Safety layer, version one (deliberately small and obvious):
- no single relative move larger than MAX_RELATIVE_DEG,
- the resulting target must stay inside the UR12e's joint limits,
- motion always uses a slow fixed duration,
- the intended move is printed before it is sent.
"""
import math
import threading
import time

import rclpy
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Robot facts
# ---------------------------------------------------------------------------

# The order the scaled_joint_trajectory_controller expects, from the driver's
# ur_controllers.yaml. This is NOT the order /joint_states publishes in, so
# we always look joints up by name and never by position in a list.
JOINT_NAMES = [
    "shoulder_pan_joint",   # joint 1, the base
    "shoulder_lift_joint",  # joint 2, the shoulder
    "elbow_joint",          # joint 3, the elbow
    "wrist_1_joint",        # joint 4
    "wrist_2_joint",        # joint 5
    "wrist_3_joint",        # joint 6, the tool flange
]

# From ur_description/config/ur12e/joint_limits.yaml (degrees).
JOINT_LIMITS_DEG = {
    "shoulder_pan_joint": (-360.0, 360.0),
    "shoulder_lift_joint": (-360.0, 360.0),
    "elbow_joint": (-180.0, 180.0),
    "wrist_1_joint": (-360.0, 360.0),
    "wrist_2_joint": (-360.0, 360.0),
    "wrist_3_joint": (-360.0, 360.0),
}

# A safe, known pose: arm pointing straight up, which is URSim's start pose.
HOME_DEG = [0.0, -90.0, 0.0, -90.0, 0.0, 0.0]

# Safety settings.
MAX_RELATIVE_DEG = 30.0   # biggest single relative move we will accept
MOVE_DURATION_S = 4.0     # every move takes at least this long (slow)
MAX_SPEED_DEG_S = 15.0    # go_home stretches its duration to stay under this

ACTION_NAME = "/scaled_joint_trajectory_controller/follow_joint_trajectory"


# ---------------------------------------------------------------------------
# The ROS side: one node, one subscriber, one action client
# ---------------------------------------------------------------------------

class UR12eInterface(Node):
    """Small ROS 2 node that reads joint angles and sends trajectories."""

    def __init__(self):
        super().__init__("rosa_ur12e_tools")
        self._latest = None                       # the last JointState message
        self.last_joint_state_time = 0.0          # when it arrived, for the status display
        self.program_running = None               # True when the pendant program is playing
        self._goal_handle = None                  # the trajectory in progress
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
        # The driver publishes this only when it changes, "latched" so late joiners
        # still get the last value. Our subscription must ask for that (transient local).
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Bool, "/io_and_status_controller/robot_program_running",
                                 self._on_program_running, latched)
        self._client = ActionClient(self, FollowJointTrajectory, ACTION_NAME)

    def _on_joint_state(self, msg):
        self._latest = msg
        self.last_joint_state_time = time.time()

    def _on_program_running(self, msg):
        self.program_running = msg.data

    def joint_positions_deg(self, timeout_s=3.0):
        """Return {joint name: degrees} in controller order, or raise if no data."""
        deadline = time.time() + timeout_s
        while self._latest is None and time.time() < deadline:
            time.sleep(0.05)
        if self._latest is None:
            raise RuntimeError("No /joint_states received. Is the UR driver running and the program playing?")
        by_name = dict(zip(self._latest.name, self._latest.position))
        return {name: math.degrees(by_name[name]) for name in JOINT_NAMES}

    def send_trajectory(self, target_deg, duration_s):
        """Send a one-point trajectory to the controller and wait for it.

        Returns True if the motion completed, False if stop() cancelled it."""
        if not self._client.wait_for_server(timeout_sec=3.0):
            raise RuntimeError(f"Action server {ACTION_NAME} not available. Is the driver running?")

        point = JointTrajectoryPoint()
        point.positions = [math.radians(d) for d in target_deg]
        point.time_from_start = Duration(sec=int(duration_s), nanosec=int((duration_s % 1) * 1e9))

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory(joint_names=JOINT_NAMES, points=[point])

        goal_handle = _wait(self._client.send_goal_async(goal))
        if not goal_handle.accepted:
            raise RuntimeError("Controller rejected the trajectory goal.")
        self._goal_handle = goal_handle
        to_motion = time.time() - _command_time - _current_event.get("confirm_wait_s", 0.0)
        _current_event["to_motion_s"] = round(to_motion, 2)
        print(f"[robot] motion started {to_motion:.2f} s after the command was entered (excluding confirmation)")

        wrapped = _wait(goal_handle.get_result_async(), timeout_s=duration_s + 10.0)
        self._goal_handle = None
        if wrapped.status == GoalStatus.STATUS_CANCELED:
            return False
        result = wrapped.result
        if result.error_code != 0:
            raise RuntimeError(
                f"Trajectory failed: {result.error_string} (code {result.error_code}). "
                "If the arm did not move at all, check the pendant is in Real mode, not Simulation, "
                "and that the External Control program is playing.")
        return True

    def cancel(self):
        if self._goal_handle is None:
            return False
        _wait(self._goal_handle.cancel_goal_async())
        self._goal_handle = None
        return True


def _wait(future, timeout_s=10.0):
    """Block until an rclpy future completes. The executor thread does the work."""
    deadline = time.time() + timeout_s
    while not future.done():
        if time.time() > deadline:
            raise RuntimeError("Timed out waiting for the robot to respond.")
        time.sleep(0.02)
    return future.result()


_robot = None
_executor = None
_spin_thread = None
_command_time = time.time()


def _get_robot():
    """Create the node on first use and spin it in a background thread."""
    global _robot, _executor, _spin_thread
    if _robot is None:
        rclpy.init()
        _robot = UR12eInterface()
        _executor = SingleThreadedExecutor()
        _executor.add_node(_robot)
        _spin_thread = threading.Thread(target=_executor.spin, daemon=True)
        _spin_thread.start()
    return _robot


def shutdown():
    """Stop the background thread cleanly so Python can exit without complaint."""
    if _executor is not None:
        _executor.shutdown()
        _spin_thread.join(timeout=2.0)
        _robot.destroy_node()
        rclpy.shutdown()


def mark_command_time():
    """Called by the agent when the user presses Enter, for the timing results."""
    global _command_time
    _command_time = time.time()


# ---------------------------------------------------------------------------
# The operator gate and the event record
# ---------------------------------------------------------------------------
# Before any motion the tool describes what it is about to do and asks the
# operator. The default asks y/n in the terminal; the web console swaps in
# its own handler that shows a card with Confirm and Cancel.

def _terminal_confirm(description):
    answer = input(f"[confirm] {description}\nType y to go ahead: ").strip().lower()
    return answer == "y"


_confirm_handler = _terminal_confirm


def set_confirm_handler(handler):
    """handler(description: str) -> bool. True means go ahead."""
    global _confirm_handler
    _confirm_handler = handler


def _confirm(description):
    """Ask the operator, and note how long they took so timings stay honest."""
    t0 = time.time()
    ok = _confirm_handler(description)
    _current_event["confirm_wait_s"] = round(time.time() - t0, 2)
    return ok


# Every tool call appends one record here, so the agent or the web console can
# show what actually happened, independent of what the model says happened.
events = []
_current_event = {}


def _record(tool, args, outcome, message):
    _current_event.update(tool=tool, args=args, outcome=outcome, message=message)
    events.append(dict(_current_event))
    _current_event.clear()
    return message


def status():
    """What the status strip needs: is the driver alive, is the program playing."""
    robot = _get_robot()
    return {
        "driver": time.time() - robot.last_joint_state_time < 1.0,
        "program": bool(robot.program_running),
    }


def _joint_name(joint):
    if not 1 <= joint <= 6:
        raise ValueError("joint must be 1 to 6")
    return JOINT_NAMES[joint - 1]


# ---------------------------------------------------------------------------
# The tools the LLM can call
# ---------------------------------------------------------------------------

@tool
def get_joint_positions() -> str:
    """Read the current angle of each of the six joints of the UR12e arm, in degrees.
    Use this to answer questions like "where is the arm" or "what is joint 3 at"."""
    positions = _get_robot().joint_positions_deg()
    lines = [f"joint {i + 1} ({name}): {deg:.1f} deg" for i, (name, deg) in enumerate(positions.items())]
    return _record("get_joint_positions", {}, "done", "\n".join(lines))


@tool
def move_joint(joint: int, degrees: float) -> str:
    """Rotate ONE joint of the UR12e arm by a RELATIVE amount, in degrees, from where it is now.
    joint: 1 = base (shoulder pan), 2 = shoulder lift, 3 = elbow, 4 = wrist 1, 5 = wrist 2, 6 = wrist 3.
    "the wrist" on its own means joint 6. Positive degrees rotate one way, negative the other.
    Moves bigger than 30 degrees are refused by the safety layer, so ask for smaller steps."""
    args = {"joint": joint, "degrees": degrees}
    name = _joint_name(joint)

    # Safety check 1: cap the size of any single move.
    if abs(degrees) > MAX_RELATIVE_DEG:
        return _record("move_joint", args, "refused",
                       f"REFUSED by safety layer: a relative move of {degrees} deg exceeds the "
                       f"limit of +/-{MAX_RELATIVE_DEG} deg. Nothing was moved.")

    robot = _get_robot()
    current = robot.joint_positions_deg()
    target = dict(current)
    target[name] = current[name] + degrees

    # Safety check 2: the target must stay inside the joint's limits.
    lo, hi = JOINT_LIMITS_DEG[name]
    if not lo <= target[name] <= hi:
        return _record("move_joint", args, "refused",
                       f"REFUSED by safety layer: joint {joint} would end at {target[name]:.1f} deg, "
                       f"outside its limits of {lo} to {hi} deg. Nothing was moved.")

    # Safety check 3 and 4: fixed slow duration, and say what we are about to do.
    description = (f"move joint {joint} ({name}) by {degrees:+.1f} deg: "
                   f"{current[name]:.1f} -> {target[name]:.1f} deg over {MOVE_DURATION_S} s")
    print(f"[robot] {description}")

    # The operator gate.
    if not _confirm(description):
        return _record("move_joint", args, "cancelled", "Cancelled by the operator. Nothing was moved.")

    completed = robot.send_trajectory([target[n] for n in JOINT_NAMES], MOVE_DURATION_S)
    now = robot.joint_positions_deg()[name]      # report what happened, not what was asked
    moved = now - current[name]
    if not completed:
        return _record("move_joint", args, "stopped",
                       f"Stopped part-way by the operator. Joint {joint} is at {now:.1f} deg.")
    # The controller's own goal tolerance is about 5.7 deg, so for small moves it
    # can report success without the arm moving. Trust the measurement, not the controller.
    if abs(moved) < 0.5 * abs(degrees):
        return _record("move_joint", args, "no_motion",
                       f"NO MOTION: the controller reported success but joint {joint} moved only "
                       f"{moved:+.1f} of {degrees:+.1f} deg and is at {now:.1f} deg. On a real robot check the "
                       f"pendant: enabling switch held, program running, no protective stop.")
    return _record("move_joint", args, "done",
                   f"Done. Joint {joint} moved by {moved:+.1f} deg and is now at {now:.1f} deg.")


@tool
def go_home() -> str:
    """Move the whole UR12e arm slowly to its safe home pose (arm pointing straight up)."""
    robot = _get_robot()
    current = robot.joint_positions_deg()
    largest = max(abs(current[n] - h) for n, h in zip(JOINT_NAMES, HOME_DEG))
    duration = max(MOVE_DURATION_S, largest / MAX_SPEED_DEG_S)
    description = f"go home: largest joint change {largest:.1f} deg over {duration:.1f} s"
    print(f"[robot] {description}")

    if not _confirm(description):
        return _record("go_home", {}, "cancelled", "Cancelled by the operator. Nothing was moved.")

    if not robot.send_trajectory(HOME_DEG, duration):
        return _record("go_home", {}, "stopped", "Stopped part-way by the operator. The arm is NOT at its home pose.")
    return _record("go_home", {}, "done", "Done. The arm is at its home pose.")


@tool
def stop() -> str:
    """Immediately stop the arm by cancelling the motion that is in progress."""
    if _get_robot().cancel():
        return _record("stop", {}, "done", "Stopped: the active motion was cancelled.")
    return _record("stop", {}, "done", "Nothing to stop: the arm was not moving.")


ALL_TOOLS = [get_joint_positions, move_joint, go_home, stop]
