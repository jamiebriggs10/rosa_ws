"""The actions the language model is allowed to take, all through MoveIt.

Each function decorated with @tool becomes something the LLM can call.
The docstring of each tool is what the model reads to decide when to use it,
so the docstrings are written for the model, not for people.

This file is an *adapter*: it talks to MoveIt's standard interfaces, so it
works with any robot that runs MoveIt. The robot-specific names (planning
group, frames, home pose) live in config.py.

ROS concepts used here, briefly:
- A *node* is one running program on the ROS network. This file makes one.
- A *topic* is a named stream of messages. /joint_states carries the joint
  angles many times a second; we read it for the 3D view.
- A *service* is a question with one answer. /compute_cartesian_path answers
  "plan a straight line to here".
- An *action* is a long request with feedback and a result. /move_action
  plans a move; /execute_trajectory runs one.
- Publishing "stop" on the /trajectory_execution_event topic makes MoveIt
  halt the move it is running.
- *tf* keeps track of where every part of the robot is. We ask it where the
  tool is relative to the base.

Every move takes the same route:
  plan -> safety checks -> preview and operator confirm -> execute that exact
  plan -> measure where the tool really ended up.
"""
import ast
import functools
import math
import threading
import time
import weakref

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Pose
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes, PlanningSceneComponents
from moveit_msgs.srv import GetCartesianPath, GetPlanningScene
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from langchain_core.tools import tool

import config

SUCCESS = MoveItErrorCodes.SUCCESS


# ---------------------------------------------------------------------------
# The ROS side: one node that talks to MoveIt
# ---------------------------------------------------------------------------

class MoveItClient(Node):
    """Plans with MoveIt, runs the plans, and reads where the arm is."""

    def __init__(self):
        super().__init__("rosa_moveit_client")
        self.joints_deg = {}                      # latest joint angles by name, for the 3D view
        self.last_joint_state_time = 0.0
        self._goal_handle = None                  # the move in progress, if any
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
        self._tf = Buffer()
        self._tf_listener = TransformListener(self._tf, self)
        self._planner = ActionClient(self, MoveGroup, config.MOVE_ACTION)
        self._executor = ActionClient(self, ExecuteTrajectory, config.EXECUTE_ACTION)
        self._cartesian = self.create_client(GetCartesianPath, config.CARTESIAN_SERVICE)
        self._stop = self.create_publisher(String, config.STOP_TOPIC, 10)
        self._scene = self.create_client(GetPlanningScene, config.SCENE_SERVICE)

    def _on_joint_state(self, msg):
        self.joints_deg.update({n: math.degrees(p) for n, p in zip(msg.name, msg.position)})
        self.last_joint_state_time = time.time()

    def moveit_ready(self):
        return self._planner.server_is_ready() and self._executor.server_is_ready()

    def tool_pose(self, timeout_s=2.0):
        """Where the tool is now, as a Pose in BASE_FRAME (metres)."""
        try:
            t = self._tf.lookup_transform(config.BASE_FRAME, config.TOOL_FRAME, Time(),
                                          timeout=Duration(seconds=timeout_s))
        except Exception as exc:
            raise RuntimeError(f"Cannot see where {config.TOOL_FRAME} is: {exc}. Is MoveIt running?")
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = (
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z)
        pose.orientation = t.transform.rotation
        return pose

    def plan_line(self, targets):
        """Plan straight lines from here through each target Pose in turn, as one move.
        Returns (fraction of the route planned, trajectory)."""
        if not self._cartesian.wait_for_service(timeout_sec=3.0):
            raise RuntimeError(f"{config.CARTESIAN_SERVICE} is not available. Is MoveIt running?")
        req = GetCartesianPath.Request()
        req.header.frame_id = config.BASE_FRAME
        req.start_state.is_diff = True            # start from where the arm is now
        req.group_name = config.PLANNING_GROUP
        req.link_name = config.TOOL_FRAME
        req.waypoints = targets
        req.max_step = 0.01                       # check the line every centimetre
        req.avoid_collisions = True
        req.max_velocity_scaling_factor = config.SPEED
        req.max_acceleration_scaling_factor = config.SPEED
        # On ROS 2 Humble the very first reply on a new connection is sometimes lost,
        # so ask a second time before giving up.
        try:
            res = _wait(self._cartesian.call_async(req), timeout_s=5.0)
        except RuntimeError:
            res = _wait(self._cartesian.call_async(req), timeout_s=5.0)
        if res.error_code.val != SUCCESS:
            raise RuntimeError(f"MoveIt could not plan the line (error code {res.error_code.val}).")
        return res.fraction, res.solution

    def plan_joints(self, target_deg):
        """Plan a move to the given joint angles (by name, degrees). Returns the trajectory."""
        if not self._planner.wait_for_server(timeout_sec=3.0):
            raise RuntimeError(f"{config.MOVE_ACTION} is not available. Is MoveIt running?")
        goal = MoveGroup.Goal()
        goal.request.group_name = config.PLANNING_GROUP
        goal.request.start_state.is_diff = True
        goal.request.num_planning_attempts = 5
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = config.SPEED
        goal.request.max_acceleration_scaling_factor = config.SPEED
        goal.request.goal_constraints = [Constraints(joint_constraints=[
            JointConstraint(joint_name=name, position=math.radians(deg),
                            tolerance_above=0.01, tolerance_below=0.01, weight=1.0)
            for name, deg in target_deg.items()])]
        goal.planning_options.plan_only = True    # we run the plan ourselves, after the operator confirms
        handle = _wait(self._planner.send_goal_async(goal))
        if not handle.accepted:
            raise RuntimeError("MoveIt refused the planning request.")
        result = _wait(handle.get_result_async(), timeout_s=30.0).result
        if result.error_code.val != SUCCESS:
            raise RuntimeError(f"MoveIt could not plan the move (error code {result.error_code.val}).")
        return result.planned_trajectory

    def execute(self, trajectory):
        """Run a planned trajectory and wait. Returns True if it finished, False if stopped."""
        if not self._executor.wait_for_server(timeout_sec=3.0):
            raise RuntimeError(f"{config.EXECUTE_ACTION} is not available. Is MoveIt running?")
        handle = _wait(self._executor.send_goal_async(ExecuteTrajectory.Goal(trajectory=trajectory)))
        if not handle.accepted:
            raise RuntimeError("MoveIt refused to run the move.")
        self._goal_handle = handle
        to_motion = time.time() - _command_time - _current_event.get("confirm_wait_s", 0.0)
        _current_event["to_motion_s"] = round(to_motion, 2)
        print(f"[robot] motion started {to_motion:.2f} s after the command (excluding confirmation)")

        # Generous wait: a real robot's speed slider can slow the move a lot.
        wrapped = _wait(handle.get_result_async(), timeout_s=60.0 + 20 * _seconds(trajectory))
        self._goal_handle = None
        code = wrapped.result.error_code.val
        if code == MoveItErrorCodes.PREEMPTED:           # halted by stop
            return False
        if code == MoveItErrorCodes.CONTROL_FAILED:
            raise RuntimeError("the robot stopped the move part-way (MoveIt error -4). Look at the pendant: "
                               "a protective stop, or the program paused. Clear it and press Play again.")
        if code != SUCCESS:
            raise RuntimeError(f"the move failed (MoveIt error code {code}). "
                               "Check the robot driver and its controller are running.")
        return True

    def cell_objects(self):
        """The fixed things in MoveIt's model of the cell (walls, table, keep-out zones):
        the places the arm may not go. Each shape's pose is given in its object's frame."""
        if not self._scene.wait_for_service(timeout_sec=3.0):
            raise RuntimeError(f"{config.SCENE_SERVICE} is not available. Is MoveIt running?")
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
        scene = _wait(self._scene.call_async(req)).scene
        kinds = {1: "box", 2: "sphere", 3: "cylinder"}
        shapes = []
        for obj in scene.world.collision_objects:
            for shape, pose in zip(obj.primitives, obj.primitive_poses):
                position, rotation = _combine(obj.pose, pose)
                shapes.append({"name": obj.id, "frame": obj.header.frame_id, "kind": kinds.get(shape.type, "other"),
                               "size": list(shape.dimensions), "position": position, "rotation": rotation})
        return shapes

    def cancel(self):
        if self._goal_handle is None:
            return False
        self._stop.publish(String(data="stop"))
        return True


def _wait(future, timeout_s=10.0):
    """Block until an rclpy future completes. The spin thread does the work."""
    deadline = time.time() + timeout_s
    while not future.done():
        if time.time() > deadline:
            raise RuntimeError("Timed out waiting for MoveIt to respond.")
        time.sleep(0.02)
    return future.result()


def _combine(outer, inner):
    """A shape's pose inside an object, as a position and quaternion [x, y, z, w] in the object's frame."""
    q, p = outer.orientation, inner.position
    # rotate the inner position by the outer rotation (q p q*), then add the outer position
    tx, ty, tz = 2 * (q.y * p.z - q.z * p.y), 2 * (q.z * p.x - q.x * p.z), 2 * (q.x * p.y - q.y * p.x)
    position = [outer.position.x + p.x + q.w * tx + (q.y * tz - q.z * ty),
                outer.position.y + p.y + q.w * ty + (q.z * tx - q.x * tz),
                outer.position.z + p.z + q.w * tz + (q.x * ty - q.y * tx)]
    r = inner.orientation
    rotation = [q.w * r.x + q.x * r.w + q.y * r.z - q.z * r.y,
                q.w * r.y - q.x * r.z + q.y * r.w + q.z * r.x,
                q.w * r.z + q.x * r.y - q.y * r.x + q.z * r.w,
                q.w * r.w - q.x * r.x - q.y * r.y - q.z * r.z]
    return position, rotation


def _seconds(trajectory):
    """How long a planned trajectory takes, in seconds."""
    points = trajectory.joint_trajectory.points
    if not points:
        return 0.0
    t = points[-1].time_from_start
    return t.sec + t.nanosec * 1e-9


_refusals = 0                                     # refused tool calls in the current command
_robot = None
_spinner = None
_spin_thread = None
_command_time = time.time()


def _get_robot():
    """Create the node on first use and spin it in a background thread."""
    global _robot, _spinner, _spin_thread
    if _robot is None:
        rclpy.init()
        _robot = MoveItClient()
        _spinner = SingleThreadedExecutor()
        _spinner.add_node(_robot)
        _spin_thread = threading.Thread(target=_spinner.spin, daemon=True)
        _spin_thread.start()
    return _robot


def shutdown():
    """Stop the background thread cleanly so Python can exit without complaint."""
    if _spinner is not None:
        _spinner.shutdown()
        _spin_thread.join(timeout=2.0)
        _robot.destroy_node()
        rclpy.shutdown()


def mark_command_time():
    """Called when the operator sends a command: starts the timing and the refusal count."""
    global _command_time, _refusals
    _command_time = time.time()
    _refusals = 0


# ---------------------------------------------------------------------------
# The operator gate and the event record
# ---------------------------------------------------------------------------
# Before any motion the tool describes what it is about to do and asks the
# operator. The default asks y/n in the terminal; the web console swaps in
# its own handler that shows a card with Confirm, Cancel and a preview.

def _terminal_confirm(description, preview=None):
    answer = input(f"[confirm] {description}\nType y to go ahead: ").strip().lower()
    return answer == "y"


_confirm_handler = _terminal_confirm


def set_confirm_handler(handler):
    """handler(description: str, preview: dict | None) -> bool. True means go ahead.
    preview is {"joints": [names], "path_deg": [[angles], ...], "seconds": duration,
    "points_cm": [[x, y, z], ...] or None}: the planned path, so a console can play it
    before it happens, and for straight-line moves the points the tool will pass through."""
    global _confirm_handler
    _confirm_handler = handler


def _confirm(description, trajectory, points_cm=None, facts=None):
    """Ask the operator, and note how long they took so timings stay honest. `facts` is the
    structured version of the description (label, distances, seconds) for the spoken read-back
    and the stage; it rides along in the preview."""
    jt = trajectory.joint_trajectory
    path = [[round(math.degrees(p), 2) for p in point.positions] for point in jt.points]
    step = max(1, len(path) // 50)                # 50 frames is plenty for the preview
    preview = {"joints": list(jt.joint_names), "path_deg": path[::step] + path[-1:],
               "seconds": round(_seconds(trajectory), 1), "points_cm": points_cm, "facts": facts}
    t0 = time.time()
    ok = _confirm_handler(description, preview)
    _current_event["confirm_wait_s"] = round(time.time() - t0, 2)
    return ok


# Every tool call appends one record here, so the agent or the web console can
# show what actually happened, independent of what the model says happened.
events = []
_current_event = {}


def _record(tool, args, outcome, message):
    global _refusals
    if outcome == "refused":
        _refusals += 1
    _current_event.update(tool=tool, args=args, outcome=outcome, message=message)
    events.append(dict(_current_event))
    _current_event.clear()
    return message


def _cm(pose):
    p = pose.position
    return [round(p.x * 100, 1), round(p.y * 100, 1), round(p.z * 100, 1)]


def status():
    """What the console's status strip and 3D view need."""
    robot = _get_robot()
    try:
        tool_cm = _cm(robot.tool_pose(timeout_s=0.1))
    except RuntimeError:
        tool_cm = None
    return {
        "moveit": robot.moveit_ready(),
        "joint_states": time.time() - robot.last_joint_state_time < 1.0,
        "joints": dict(robot.joints_deg),             # {joint name: degrees}
        "tool_cm": tool_cm,                       # [x, y, z] in BASE_FRAME
    }


# ---------------------------------------------------------------------------
# The tools the LLM can call
# ---------------------------------------------------------------------------

# Commands the operator has stopped while the model was still thinking. Their threads run on
# until the model answers, but any tool they call afterwards does nothing. The threads
# themselves are kept, not their id numbers: the system reuses id numbers for new threads.
abandoned_threads = weakref.WeakSet()


def _reports_failures(func):
    """If a tool hits a problem (MoveIt not answering, the robot stopping a move), report it
    as a FAILED result the operator and the model can read, instead of an error.
    Also notes how long after the command the tool was called: the model's thinking time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if threading.current_thread() in abandoned_threads:
            return "The operator stopped this command. Do nothing more."
        # Small models can get stuck calling a refused tool again and again: stop that.
        if _refusals >= config.MAX_REFUSALS:
            return (f"STOP. {_refusals} moves in this command were already refused. Do not call any more "
                    "tools; tell the operator what was refused and why.")
        _current_event["after_s"] = round(time.time() - _command_time, 1)
        _current_event["tool"] = func.__name__          # so the console can say which tool is running
        try:
            return func(*args, **kwargs)
        except RuntimeError as exc:
            return _record(func.__name__, kwargs, "failed", f"FAILED: {exc}")
    return wrapper


def _line_label(name, args, target_cm):
    """What the card and the spoken read-back call a straight move: "Up 5 cm" for move_by (the
    largest axis of the offset names it), "x 36, y -4, z 34" for move_to."""
    if name == "move_by":
        dx, dy, dz = (float(args.get(k, 0)) for k in ("dx", "dy", "dz"))
        axis, value = max((("z", dz), ("y", dy), ("x", dx)), key=lambda kv: abs(kv[1]))
        word = {"z": ("Up", "Down"), "y": ("Left", "Right"), "x": ("Forward", "Back")}[axis][0 if value >= 0 else 1]
        return f"{word} {abs(value):g} cm"
    return f"x {target_cm[0]:g}, y {target_cm[1]:g}, z {target_cm[2]:g}"


def _move_line(name, args, target_cm):
    """Shared by move_to and move_by: a straight-line move of the tool to target_cm."""
    robot = _get_robot()
    start = robot.tool_pose()
    start_cm = _cm(start)

    # Safety check 1: no single move longer than MAX_STEP_CM, and no empty moves.
    distance = math.dist(start_cm, target_cm)
    if distance < 0.1:
        return _record(name, args, "refused",
                       "REFUSED: that move is 0 cm, so there is nothing to do. Nothing was moved.")
    if distance > config.MAX_STEP_CM:
        return _record(name, args, "refused",
                       f"REFUSED by safety layer: that move is {distance:.1f} cm, more than the "
                       f"{config.MAX_STEP_CM} cm limit for one move. Nothing was moved.")

    # Safety check 2: MoveIt must find a straight line the whole way that hits nothing in its model of the cell.
    target = Pose()
    target.position.x, target.position.y, target.position.z = (v / 100 for v in target_cm)
    target.orientation = start.orientation       # keep the tool pointing the same way
    fraction, trajectory = robot.plan_line([target])
    if fraction < 0.999:
        return _record(name, args, "refused",
                       f"REFUSED: MoveIt could plan only {fraction:.0%} of the straight line: the target is "
                       "out of reach, something is in the way, or the arm is already touching something in "
                       "MoveIt's model of the cell. Nothing was moved.")

    # Safety check 3: the operator sees the plan and confirms it.
    _current_event["start_cm"] = start_cm
    description = (f"move the tool in a straight line from {start_cm} to {target_cm} cm "
                   f"({distance:.1f} cm, about {_seconds(trajectory):.0f} s)")
    print(f"[robot] {description}")
    facts = {"kind": "line", "label": _line_label(name, args, target_cm), "from_cm": start_cm,
             "to_cm": [round(float(v), 1) for v in target_cm], "distance_cm": round(distance, 1),
             "seconds": round(_seconds(trajectory), 1)}
    if not _confirm(description, trajectory, [start_cm, target_cm], facts):
        return _record(name, args, "cancelled", "Cancelled by the operator. Nothing was moved.")

    completed = robot.execute(trajectory)
    now_cm = _cm(robot.tool_pose())               # report what happened, not what was asked
    if not completed:
        return _record(name, args, "stopped", f"Stopped part-way by the operator. The tool is at {now_cm} cm.")
    error = math.dist(now_cm, target_cm)
    if error > 1.0:
        return _record(name, args, "no_motion",
                       f"NOT THERE: MoveIt reported success but the tool is at {now_cm} cm, "
                       f"{error:.1f} cm from the target.")
    return _record(name, args, "done", f"Done. The tool is now at x={now_cm[0]}, y={now_cm[1]}, z={now_cm[2]} cm.")


@tool
@_reports_failures
def where_is_the_tool() -> str:
    """Report where the robot's tool is, as x, y, z in centimetres measured from the robot's base.
    z is height. Use this to answer questions like "where is the arm"."""
    x, y, z = _cm(_get_robot().tool_pose())
    return _record("where_is_the_tool", {}, "done", f"The tool is at x={x}, y={y}, z={z} cm.")


@tool
@_reports_failures
def move_by(dx: float, dy: float, dz: float) -> str:
    """Move the robot's tool in a straight line by a RELATIVE amount, in centimetres, from where it is now.
    dz is up (+) and down (-). dx and dy are the horizontal directions of the robot base.
    Example: "move up 5 cm" is move_by(dx=0, dy=0, dz=5). Use 0 for directions not mentioned."""
    target = [a + b for a, b in zip(_cm(_get_robot().tool_pose()), (dx, dy, dz))]
    return _move_line("move_by", {"dx": dx, "dy": dy, "dz": dz}, target)


@tool
@_reports_failures
def move_to(x: float, y: float, z: float) -> str:
    """Move the robot's tool in a straight line to an ABSOLUTE position, in centimetres from the robot's base.
    Use this only when the operator gives all three coordinates. z is height."""
    return _move_line("move_to", {"x": x, "y": y, "z": z}, [x, y, z])


@tool
@_reports_failures
def move_path(points: list[list[float]]) -> str:
    """Move the tool through a list of points in straight lines, planned, previewed and confirmed
    as ONE move. Each point is [x, y, z] in centimetres MEASURED FROM WHERE THE TOOL IS NOW
    (so [0, 0, 0] is the start); z is up. Use this whenever a request needs more than one
    straight move. Example: "forward 5 then up 3" is move_path(points=[[5, 0, 0], [5, 0, 3]]).
    To finish where you started, end with [0, 0, 0]."""
    return _follow_points("move_path", {"points": points}, points)


@tool
@_reports_failures
def move_formula(formula: str) -> str:
    """Move the tool through points that YOU DESCRIBE WITH A FORMULA; the computer works out every
    point exactly. Use this for circles, spirals, curves, or many evenly spaced points, instead of
    calculating numbers yourself. The formula is a Python list comprehension giving [x, y, z] points
    in centimetres measured from where the tool is now ([0, 0, 0] is the start; z is up). Allowed:
    numbers, + - * / **, sin cos tan sqrt atan2 radians pi abs min max round, range, and
    "for t in [expression]" to name a value. x, y and z on their own are 0: the tool's position.
    Take every number (sizes, turns, point counts) from the operator's request. Two examples of
    the form only, with made-up numbers:
    Ellipse 6 cm by 2 cm in the x-z plane, 20 points, starting and ending at the tool:
      [[3*cos(2*pi*i/20) - 3, 0, 1*sin(2*pi*i/20)] for i in range(21)]
    Helix of radius 2 cm around a centre 2 cm along -x, 4 turns, rising 8 cm, 12 points per turn:
      [[2*cos(2*pi*4*t) - 2, 2*sin(2*pi*4*t), 8*t] for i in range(49) for t in [i/48]]"""
    args = {"formula": formula}
    try:
        points = evaluate_formula(formula)
    except ValueError as exc:
        return _record("move_formula", args, "refused",
                       f"FORMULA ERROR, nothing was moved: {exc} Fix the formula and call move_formula again.")
    return _follow_points("move_formula", args, points)


# What a formula may use: plain maths, nothing that can touch the computer.
_FORMULA_NAMES = {"sin": math.sin, "cos": math.cos, "tan": math.tan, "sqrt": math.sqrt, "atan2": math.atan2,
                  "radians": math.radians, "pi": math.pi, "abs": abs, "min": min, "max": max, "round": round}
_FORMULA_NODES = (ast.Expression, ast.ListComp, ast.List, ast.Tuple, ast.comprehension, ast.Name, ast.Load,
                  ast.Store, ast.Constant, ast.BinOp, ast.UnaryOp, ast.Call, ast.Add, ast.Sub, ast.Mult, ast.Div,
                  ast.Pow, ast.Mod, ast.FloorDiv, ast.USub, ast.UAdd, ast.Compare, ast.Lt, ast.LtE, ast.Gt,
                  ast.GtE, ast.Eq, ast.NotEq, ast.IfExp, ast.BoolOp, ast.And, ast.Or)


def _formula_range(*args):
    if len(args) > 3 or any(float(a) != int(a) for a in args):
        raise ValueError("range takes whole numbers")
    r = range(*(int(a) for a in args))
    if len(r) > 1000:
        raise ValueError("range is too long")
    return r


def evaluate_formula(formula):
    """Work out the points a formula describes. Only the maths above is allowed: anything else
    (attributes, imports, files, names it does not know) is refused before it runs."""
    if len(formula) > 500:
        raise ValueError("the formula is too long.")
    try:
        tree = ast.parse(formula.strip(), mode="eval")
    except SyntaxError:
        raise ValueError("it is not a valid list comprehension.")
    if sum(isinstance(n, ast.Call) and getattr(n.func, "id", "") == "range" for n in ast.walk(tree)) > 1:
        raise ValueError("use range only once (name extra values with 'for t in [...]').")
    for node in ast.walk(tree):
        if not isinstance(node, _FORMULA_NODES):
            raise ValueError(f"{type(node).__name__} is not allowed.")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError("only numbers are allowed.")
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            node.value = float(node.value)    # float maths fails fast instead of building huge numbers
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.keywords
                                           or node.func.id not in (*_FORMULA_NAMES, "range")):
            raise ValueError("only the listed maths functions can be called.")
    try:
        allowed = {"__builtins__": {}, **_FORMULA_NAMES, "range": _formula_range,   # globals: comprehensions look names up there
                   "x": 0.0, "y": 0.0, "z": 0.0}       # the tool's own position, the origin of every point
        points = eval(compile(tree, "<formula>", "eval"), allowed)
        return [[float(v) for v in point] for point in points]
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"working it out failed ({type(exc).__name__}: {exc}).")


ABSOLUTE_SNAP_CM = 10.0   # a path passing this close to the tool, read as absolute coordinates, is taken as absolute


def _as_offsets(points, start_cm):
    """Read the model's points as offsets from the tool, or, when that cannot be right, as absolute
    coordinates. Models often write a shape around the tool's current position instead of around
    [0, 0, 0]. Returns (offsets, note); note says when the absolute reading was used."""
    reach = max(math.hypot(*p) for p in points)
    if reach <= config.MAX_PATH_REACH_CM:
        return points, ""
    absolute = [[p[i] - start_cm[i] for i in range(3)] for p in points]
    if min(math.hypot(*p) for p in absolute) <= ABSOLUTE_SNAP_CM \
            and max(math.hypot(*p) for p in absolute) <= config.MAX_PATH_REACH_CM:
        return absolute, " (the points were read as absolute coordinates, since they lie around where the tool is)"
    return points, ""


def _follow_points(name, args, points):
    """Shared by move_path and move_formula: move through points measured from the start, as one move."""
    # Safety check 1: a sensible route: real, finite numbers and not too many points.
    try:
        offsets = [[float(v) for v in point] for point in points]
    except (TypeError, ValueError):
        offsets = None
    if not offsets or any(len(point) != 3 or not all(math.isfinite(v) for v in point) for point in offsets):
        return _record(name, args, "refused",
                       "REFUSED: points must be a list of [x, y, z] in cm from the start. Nothing was moved.")
    if len(offsets) > config.MAX_PATH_STEPS:
        return _record(name, args, "refused",
                       f"REFUSED by safety layer: {len(offsets)} points is more than the {config.MAX_PATH_STEPS} "
                       "allowed in one path. Nothing was moved.")

    # Safety check 2: no point far from the start, and no single leg too long.
    robot = _get_robot()
    start = robot.tool_pose()
    start_cm = _cm(start)
    offsets, note = _as_offsets(offsets, start_cm)
    reach = max(math.hypot(*point) for point in offsets)
    if reach > config.MAX_PATH_REACH_CM:
        return _record(name, args, "refused",
                       f"REFUSED by safety layer: a point is {reach:.1f} cm from where the tool is now, "
                       f"more than the {config.MAX_PATH_REACH_CM} cm limit. Nothing was moved.")
    legs = [math.dist(a, b) for a, b in zip([[0.0, 0.0, 0.0]] + offsets, offsets)]
    if sum(legs) < 0.1:
        return _record(name, args, "refused",
                       "REFUSED: every point is where the tool already is, so there is nothing to do. Nothing was moved.")
    if max(legs) > config.MAX_STEP_CM:
        return _record(name, args, "refused",
                       f"REFUSED by safety layer: one leg of the path is {max(legs):.1f} cm, more than the "
                       f"{config.MAX_STEP_CM} cm limit for one straight line. Nothing was moved.")

    # Safety check 3: MoveIt plans the whole route at once, and every part must hit nothing.
    points_cm = [[round(s + o, 1) for s, o in zip(start_cm, point)] for point in offsets]
    targets = []
    for p in points_cm:
        target = Pose()
        target.position.x, target.position.y, target.position.z = (v / 100 for v in p)
        target.orientation = start.orientation   # keep the tool pointing the same way
        targets.append(target)
    fraction, trajectory = robot.plan_line(targets)
    if fraction < 0.999:
        return _record(name, args, "refused",
                       f"REFUSED: MoveIt could plan only {fraction:.0%} of the path: part of it is out of "
                       "reach or would hit something in the cell. Nothing was moved.")

    # Safety check 4: the operator sees the whole route and confirms it once.
    _current_event["start_cm"] = start_cm
    description = (f"move the tool through {len(offsets)} points, {sum(legs):.1f} cm in total, "
                   f"from {start_cm} to {points_cm[-1]} cm (about {_seconds(trajectory):.0f} s){note}")
    print(f"[robot] {description}")
    facts = {"kind": "path", "label": f"Path of {len(offsets)} points", "from_cm": start_cm, "to_cm": points_cm[-1],
             "distance_cm": round(sum(legs), 1), "seconds": round(_seconds(trajectory), 1)}
    if not _confirm(description, trajectory, [start_cm] + points_cm, facts):
        return _record(name, args, "cancelled", "Cancelled by the operator. Nothing was moved.")

    completed = robot.execute(trajectory)
    now_cm = _cm(robot.tool_pose())
    if not completed:
        return _record(name, args, "stopped", f"Stopped part-way by the operator. The tool is at {now_cm} cm.")
    return _record(name, args, "done",
                   f"Done. Went through all {len(offsets)} points; the tool is now at x={now_cm[0]}, y={now_cm[1]}, z={now_cm[2]} cm.{note}")


@tool
@_reports_failures
def move_joint(joint: int, degrees: float) -> str:
    """Turn ONE joint of the arm by a RELATIVE amount, in degrees, from where it is now.
    Use this whenever the operator names a joint, e.g. "turn the base 30 degrees" or
    "bend the elbow 10 degrees". Positive and negative turn opposite ways."""
    args = {"joint": joint, "degrees": degrees}

    # Safety check 1: a real joint, and not too big a turn.
    if not 1 <= joint <= len(config.JOINTS):
        return _record("move_joint", args, "refused",
                       f"REFUSED: there is no joint {joint}; joints are 1 to {len(config.JOINTS)}. Nothing was moved.")
    if abs(degrees) > config.MAX_JOINT_DEG:
        return _record("move_joint", args, "refused",
                       f"REFUSED by safety layer: {degrees} deg is more than the {config.MAX_JOINT_DEG} deg "
                       "limit for one move. Nothing was moved.")

    # Safety check 2: MoveIt plans it, checking the joint's limits and the cell.
    robot = _get_robot()
    name, word = config.JOINTS[joint - 1]
    target = {n: robot.joints_deg[n] for n, _ in config.JOINTS}
    start = target[name]
    target[name] = start + degrees
    try:
        trajectory = robot.plan_joints(target)
    except RuntimeError as exc:
        return _record("move_joint", args, "refused",
                       f"REFUSED: MoveIt could not plan that turn ({exc}): it would pass a joint limit or "
                       "hit something in the cell. Nothing was moved.")

    # Safety check 3: the operator sees the plan and confirms it.
    description = (f"turn joint {joint} ({word}) by {degrees:+.1f} deg: {start:.1f} -> {target[name]:.1f} deg "
                   f"(about {_seconds(trajectory):.0f} s)")
    print(f"[robot] {description}")
    facts = {"kind": "joint", "label": f"Joint {joint} by {degrees:g}°", "largest_turn_deg": round(abs(degrees), 1),
             "seconds": round(_seconds(trajectory), 1)}
    if not _confirm(description, trajectory, None, facts):
        return _record("move_joint", args, "cancelled", "Cancelled by the operator. Nothing was moved.")

    completed = robot.execute(trajectory)
    now = robot.joints_deg[name]                  # report what happened, not what was asked
    if not completed:
        return _record("move_joint", args, "stopped",
                       f"Stopped part-way by the operator. Joint {joint} ({word}) is at {now:.1f} deg.")
    return _record("move_joint", args, "done",
                   f"Done. Joint {joint} ({word}) turned {now - start:+.1f} deg and is now at {now:.1f} deg.")


# The model reads this list to match "the elbow" to a joint number.
move_joint.description += " Joints: " + ", ".join(f"{i} = {word}" for i, (_, word) in enumerate(config.JOINTS, 1)) + "."


@tool
@_reports_failures
def go_home() -> str:
    """Move the whole arm slowly to its safe home pose."""
    robot = _get_robot()
    trajectory = robot.plan_joints(config.HOME_DEG)
    description = f"go to the home pose (about {_seconds(trajectory):.0f} s)"
    print(f"[robot] {description}")
    turn = max(abs(config.HOME_DEG[n] - robot.joints_deg.get(n, config.HOME_DEG[n])) for n in config.HOME_DEG)
    facts = {"kind": "joint", "label": "Home position", "largest_turn_deg": round(turn, 1),
             "seconds": round(_seconds(trajectory), 1)}
    if not _confirm(description, trajectory, None, facts):
        return _record("go_home", {}, "cancelled", "Cancelled by the operator. Nothing was moved.")
    if not robot.execute(trajectory):
        return _record("go_home", {}, "stopped", "Stopped part-way by the operator. The arm is NOT at its home pose.")
    return _record("go_home", {}, "done", "Done. The arm is at its home pose.")


@tool
def stop() -> str:
    """Immediately stop the arm by halting the motion that is in progress."""
    if _get_robot().cancel():
        return _record("stop", {}, "done", "Stopped: the active motion was halted.")
    return _record("stop", {}, "done", "Nothing to stop: the arm was not moving.")


ALL_TOOLS = [where_is_the_tool, move_by, move_to, move_path, move_formula, move_joint, go_home, stop]

# What MoveIt (the move_group node) offers through these tools, for the console's map and the spoken
# tour. The same shape as a described interface: a meaning, an example phrase, and whether it moves.
MOVEIT_SKILLS = [
    {"name": "where_is_the_tool", "means": "Where the tool is, in centimetres from the base.", "say": "where is the tool",
     "moves_robot": False, "type": "tf2 lookup"},
    {"name": "move_by", "means": "Move the tool in a straight line by an amount.", "say": "up five centimetres",
     "moves_robot": True, "type": "GetCartesianPath + ExecuteTrajectory"},
    {"name": "move_to", "means": "Move the tool in a straight line to a position.", "say": "go to x forty, y minus ten, z thirty",
     "moves_robot": True, "type": "GetCartesianPath + ExecuteTrajectory"},
    {"name": "move_path", "means": "Move the tool through a list of points.", "say": "move through the points zero zero five and five zero five",
     "moves_robot": True, "type": "GetCartesianPath + ExecuteTrajectory"},
    {"name": "move_formula", "means": "Move the tool along a shape given as a formula.", "say": "draw a ten centimetre circle",
     "moves_robot": True, "type": "GetCartesianPath + ExecuteTrajectory"},
    {"name": "move_joint", "means": "Turn one joint by some degrees.", "say": "joint two by ten degrees",
     "moves_robot": True, "type": "MoveGroup action"},
    {"name": "go_home", "means": "Move the whole arm to its home pose.", "say": "go home",
     "moves_robot": True, "type": "MoveGroup action"},
    {"name": "stop", "means": "Halt the motion in progress.", "say": "stop",
     "moves_robot": False, "type": "ExecuteTrajectory cancel"},
]
