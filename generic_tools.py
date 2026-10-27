"""Tools that work with any ROS node, keyed on the *type* of interface, never its name.

Motion goes through MoveIt (robot_tools.py). Everything else a robot system offers
(take a scan, pick a point, read a result) is some node's own service or topic. These
four tools let the model use them without our code knowing the node:

  list_capabilities()           what discovery found, with the meanings from interfaces.yaml
  call_trigger(service)         call a std_srvs/Trigger service ("do it, tell me if it worked")
  publish_point(topic, x, y, z) send one geometry_msgs/PointStamped ("this point here")
  read_topic(topic)             wait for one message on any topic and summarise it

Every write passes the policy in interfaces.py: refused if the description file does
not list it, through the operator's confirm card if it may move the robot. Reads are
always allowed. Every call is recorded in robot_tools.events, like a move.

ROS concepts used here, briefly:
- *QoS* (quality of service): how a topic is delivered. A "latched" (transient local)
  publisher keeps its last message for late subscribers; to receive it, we must ask
  with the same setting, so read_topic copies the publisher's QoS.
- *tf* transforms: a path published in the "world" frame is moved into the robot's
  base frame before the console draws it, so it lines up with the arm.
"""
import json
import math
import threading
import time

from langchain_core.tools import tool
from rclpy.qos import QoSProfile
from rclpy.time import Time
from rclpy.duration import Duration
from rosidl_runtime_py import message_to_ordereddict
from rosidl_runtime_py.utilities import get_message
from geometry_msgs.msg import PointStamped
from std_srvs.srv import Trigger

import config
import discovery
import robot_tools
from interfaces import Description, policy, report

SERVICE_WAIT_S = 3.0        # for a service to appear
CALL_TIMEOUT_S = 120.0      # for a Trigger to answer (a touchdown can take a while)
SUBSCRIBER_WAIT_S = 2.0     # for someone to be listening before we publish
READ_TIMEOUT_S = 4.0        # for one message to arrive
MAX_DRAWN_POINTS = 300      # a path is thinned to this many points for the console

_description = None
_publishers = {}            # topic -> publisher, kept so the second publish needs no wait
last_path = None            # {"topic", "points_cm"}: the last path read, drawn by the console


def description():
    global _description
    if _description is None:
        _description = Description.load(config.INTERFACES_FILE)
    return _description


def reload_description():
    global _description
    _description = None
    return description()


# -- the graph and the policy --------------------------------------------------------

def _graph():
    return discovery.scan(robot_tools._get_robot())


def _find(name):
    for i in _graph():
        if i.name == name:
            return i
    return None


def capability_rows():
    """The console's table: everything on the graph, with meanings and status."""
    return report(description(), _graph())


def capabilities_text(limit=40):
    """A short listing for the model: described interfaces first, then the rest."""
    rows = capability_rows()
    lines = []
    if description().system:
        lines.append(f"This system: {description().system}")
    described = [r for r in rows if r["status"] == "ready"]
    missing = [r for r in rows if r["status"] == "missing"]
    other = [r for r in rows if r["status"] == "undescribed"]
    for r in described:
        flag = " [moves the robot]" if r["moves_robot"] else ""
        lines.append(f"{r['kind']} {r['name']} ({r['type']}): {r['means']}{flag}")
    for r in missing:
        lines.append(f"MISSING {r['name']}: described but not running now")
    if other:
        shown = other[: max(0, limit - len(lines))]
        lines.append("Read-only (not described): " + ", ".join(f"{r['name']} ({r['type']})" for r in shown)
                     + (f" and {len(other) - len(shown)} more" if len(other) > len(shown) else ""))
    return "\n".join(lines) if lines else "Nothing found on the ROS graph. Is the robot's system running?"


def _gate(tool_name, args, name, what):
    """Apply the write policy. Returns None to go ahead, or the message to return instead."""
    verdict, reason = policy(description(), name, config.ALLOW_UNLISTED_WRITES)
    if verdict == "refused":
        return robot_tools._record(tool_name, args, "refused", f"REFUSED: {reason}")
    if verdict == "confirm":
        print(f"[robot] {what}")
        entry = description().get(name) or {}
        label = (entry.get("means") or name).split(". ")[0].rstrip(".")
        t0 = time.time()
        ok = robot_tools._confirm_handler(what, {"facts": {"kind": "trigger", "label": label, "name": name}})
        robot_tools._current_event["confirm_wait_s"] = round(time.time() - t0, 2)
        if not ok:
            return robot_tools._record(tool_name, args, "cancelled", "Cancelled by the operator. Nothing was done.")
    return None


# -- summaries ----------------------------------------------------------------------

def _cm(x, y, z):
    return [round(x * 100, 1), round(y * 100, 1), round(z * 100, 1)]


def summarise(msg, type_name):
    """One readable line about a message, for the model and the chat."""
    if type_name == "std_msgs/msg/Bool":
        return "true" if msg.data else "false"
    if type_name == "std_msgs/msg/String":
        return msg.data
    if type_name == "nav_msgs/msg/Path":
        n = len(msg.poses)
        if n == 0:
            return "an empty path (0 poses)"
        pts = [p.pose.position for p in msg.poses]
        length = sum(math.dist((a.x, a.y, a.z), (b.x, b.y, b.z)) for a, b in zip(pts, pts[1:]))
        first, last = _cm(pts[0].x, pts[0].y, pts[0].z), _cm(pts[-1].x, pts[-1].y, pts[-1].z)
        return (f"a path of {n} poses in frame '{msg.header.frame_id}', from {first} to {last} cm, "
                f"{length * 100:.1f} cm long")
    if type_name == "sensor_msgs/msg/PointCloud2":
        from sensor_msgs_py import point_cloud2
        n = msg.width * msg.height
        if n == 0:
            return "an empty point cloud (0 points)"
        xyz = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        lo = [float(xyz[f].min()) * 100 for f in ("x", "y", "z")]
        hi = [float(xyz[f].max()) * 100 for f in ("x", "y", "z")]
        return (f"a point cloud of {n} points in frame '{msg.header.frame_id}', x {lo[0]:.1f} to {hi[0]:.1f}, "
                f"y {lo[1]:.1f} to {hi[1]:.1f}, z {lo[2]:.1f} to {hi[2]:.1f} cm")
    text = json.dumps(message_to_ordereddict(msg), default=str)
    return text if len(text) <= 400 else text[:400] + "…"


def path_points_cm(msg, node):
    """The poses of a nav_msgs/Path in the robot's base frame, in cm, thinned for drawing."""
    from tf2_geometry_msgs import do_transform_point
    frame = msg.header.frame_id or config.BASE_FRAME
    step = max(1, len(msg.poses) // MAX_DRAWN_POINTS)
    poses = msg.poses[::step] + ([msg.poses[-1]] if len(msg.poses) % step else [])
    if frame == config.BASE_FRAME:
        return [_cm(p.pose.position.x, p.pose.position.y, p.pose.position.z) for p in poses]
    t = node._tf.lookup_transform(config.BASE_FRAME, frame, Time(), timeout=Duration(seconds=1.0))
    out = []
    for p in poses:
        ps = PointStamped()
        ps.header.frame_id = frame
        ps.point = p.pose.position
        q = do_transform_point(ps, t).point
        out.append(_cm(q.x, q.y, q.z))
    return out


# -- the tools -------------------------------------------------------------------------

@tool
@robot_tools._reports_failures
def list_capabilities() -> str:
    """List what this robot system offers besides moving the arm: its services and topics,
    what each means, and which can move the robot. Call this first for any request that
    is not a plain move, then use the exact names it lists."""
    return robot_tools._record("list_capabilities", {}, "done", capabilities_text())


@tool
@robot_tools._reports_failures
def call_trigger(service: str) -> str:
    """Call one of the system's Trigger services by its exact name from list_capabilities,
    e.g. call_trigger(service="/multiview_scan/capture"). It does the thing and reports
    whether it worked. Services that move the robot are confirmed by the operator first."""
    args = {"service": service}
    found = _find(service)
    if found is None or found.kind != "service":
        return robot_tools._record("call_trigger", args, "failed",
                                   f"FAILED: {service} is not a service on this system right now. Call list_capabilities.")
    if found.type != "std_srvs/srv/Trigger":
        return robot_tools._record("call_trigger", args, "failed",
                                   f"FAILED: {service} is a {found.type}, not a Trigger, so this tool cannot call it.")
    entry = description().get(service)
    what = f"call {service}" + (f": {entry['means']}" if entry and entry["means"] else "")
    blocked = _gate("call_trigger", args, service, what)
    if blocked:
        return blocked
    node = robot_tools._get_robot()
    client = node.create_client(Trigger, service)
    try:
        if not client.wait_for_service(timeout_sec=SERVICE_WAIT_S):
            raise RuntimeError(f"{service} did not answer within {SERVICE_WAIT_S:.0f} s.")
        t0 = time.time()
        res = robot_tools._wait(client.call_async(Trigger.Request()), timeout_s=CALL_TIMEOUT_S)
    finally:
        node.destroy_client(client)
    took = round(time.time() - t0, 1)
    if res.success:
        return robot_tools._record("call_trigger", args, "done", f"Done ({took} s). {service} says: {res.message or 'ok'}")
    return robot_tools._record("call_trigger", args, "failed", f"FAILED: {service} reported failure: {res.message or 'no reason given'}")


@tool
@robot_tools._reports_failures
def publish_point(topic: str, x: float, y: float, z: float) -> str:
    """Send a point to one of the system's PointStamped topics, by its exact name from
    list_capabilities, e.g. publish_point(topic="/clicked_point", x=30, y=-60, z=5).
    x, y, z are in centimetres from the robot's base, the same frame as move_to."""
    args = {"topic": topic, "x": x, "y": y, "z": z}
    found = _find(topic)
    if found is None or found.kind != "topic":
        return robot_tools._record("publish_point", args, "failed",
                                   f"FAILED: {topic} is not a topic on this system right now. Call list_capabilities.")
    if found.type != "geometry_msgs/msg/PointStamped":
        return robot_tools._record("publish_point", args, "failed",
                                   f"FAILED: {topic} carries {found.type}, not a PointStamped, so this tool cannot publish to it.")
    entry = description().get(topic)
    what = f"send the point ({x}, {y}, {z}) cm to {topic}" + (f": {entry['means']}" if entry and entry["means"] else "")
    blocked = _gate("publish_point", args, topic, what)
    if blocked:
        return blocked
    node = robot_tools._get_robot()
    pub = _publishers.get(topic)
    if pub is None:
        pub = _publishers[topic] = node.create_publisher(PointStamped, topic, 10)
    deadline = time.time() + SUBSCRIBER_WAIT_S
    while pub.get_subscription_count() == 0 and time.time() < deadline:
        time.sleep(0.05)
    msg = PointStamped()
    msg.header.frame_id = config.BASE_FRAME
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.point.x, msg.point.y, msg.point.z = x / 100, y / 100, z / 100
    pub.publish(msg)
    listeners = pub.get_subscription_count()
    if listeners == 0:
        return robot_tools._record("publish_point", args, "failed",
                                   f"FAILED: nothing is listening on {topic}, so the point went nowhere. Is that node running?")
    return robot_tools._record("publish_point", args, "done",
                               f"Sent the point ({x}, {y}, {z}) cm in frame '{config.BASE_FRAME}' to {topic} ({listeners} listening).")


@tool
@robot_tools._reports_failures
def read_topic(topic: str) -> str:
    """Read the latest message on one of the system's topics, by its exact name from
    list_capabilities, e.g. read_topic(topic="/inspection/raster_plan_valid"). Returns a
    short description of what the message says. Reading never changes anything."""
    global last_path
    args = {"topic": topic}
    found = _find(topic)
    if found is None or found.kind != "topic":
        return robot_tools._record("read_topic", args, "failed",
                                   f"FAILED: {topic} is not a topic on this system right now. Call list_capabilities.")
    node = robot_tools._get_robot()
    msg_type = get_message(found.type)
    qos = QoSProfile(depth=1)
    publishers = node.get_publishers_info_by_topic(topic)
    if publishers:                                  # match the publisher, or a latched message never arrives
        qos.durability = publishers[0].qos_profile.durability
        qos.reliability = publishers[0].qos_profile.reliability
    got, arrived = [], threading.Event()

    def on_message(m):
        got.append(m)
        arrived.set()

    sub = node.create_subscription(msg_type, topic, on_message, qos)
    try:
        arrived.wait(READ_TIMEOUT_S)
    finally:
        node.destroy_subscription(sub)
    if not got:
        why = "nobody is publishing on it" if not publishers else f"nothing arrived in {READ_TIMEOUT_S:.0f} s"
        return robot_tools._record("read_topic", args, "failed", f"FAILED: no message on {topic}: {why}.")
    msg = got[0]
    if found.type == "nav_msgs/msg/Path" and msg.poses:
        try:
            last_path = {"topic": topic, "points_cm": path_points_cm(msg, node)}
        except Exception as exc:                      # no tf for that frame: still report, just do not draw
            print(f"[robot] could not draw {topic}: {exc}")
    return robot_tools._record("read_topic", args, "done", f"{topic} says: {summarise(msg, found.type)}")


def clear_path():
    global last_path
    last_path = None


ALL_TOOLS = [list_capabilities, call_trigger, publish_point, read_topic]
