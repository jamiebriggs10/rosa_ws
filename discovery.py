"""Find out what the robot's system offers: its topics, services and actions.

ROS keeps a live directory ("the graph") of every topic, service and action that
any running node provides, with the message type of each. Reading it is how the
agent learns what *this* system can do, without anyone naming a robot in our code.

Only the type of each interface matters to us: a std_srvs/Trigger service can be
called the same way whatever it is for. What it is *for* comes from the optional
description file (interfaces.py).
"""
import re
from collections import namedtuple

import config

# One entry of the graph: its name, whether it is a topic, service or action, and
# its type written the ROS way ("std_srvs/srv/Trigger").
Interface = namedtuple("Interface", "name kind type")

# ROS plumbing that every node has and no operator wants to see.
_PLUMBING = re.compile(
    r"^/(rosout|parameter_events)$"
    r"|/(get|set|list|describe)_parameters(_atomically)?$"
    r"|/get_parameter_types$"
    r"|/get_type_description$"
    r"|/_action/"                        # the five hidden topics and services behind each action
)


def filter_graph(topics, services, actions, own_names=()):
    """Turn the raw graph lists into Interfaces worth showing.

    Each list holds (name, [types]) pairs as rclpy returns them. own_names are the
    names of our own node(s), whose parameter services are dropped like any other.
    """
    found = []
    for kind, pairs in (("topic", topics), ("service", services), ("action", actions)):
        for name, types in pairs:
            if _PLUMBING.search(name) or any(name.startswith(f"/{own}/") for own in own_names):
                continue
            found.append(Interface(name, kind, types[0] if types else ""))
    return sorted(found)


def scan(node):
    """Read the live graph through our node. Blocks for nothing: the graph is cached locally.

    Actions are listed from their *servers* only. The plain action listing would also
    count our own clients, so MoveIt would look present when only we are running."""
    from rclpy.action import get_action_server_names_and_types_by_node
    actions = {}
    for name, namespace in node.get_node_names_and_namespaces():
        if name == node.get_name():
            continue
        try:
            for action, types in get_action_server_names_and_types_by_node(node, name, namespace):
                actions.setdefault(action, types)
        except Exception:                      # a node that went away between the two calls
            pass
    return filter_graph(node.get_topic_names_and_types(), node.get_service_names_and_types(),
                        sorted(actions.items()), own_names=(node.get_name(),))


def moveit_present(interfaces):
    """True if the MoveIt interfaces named in config are on the graph."""
    names = {i.name for i in interfaces}
    return config.MOVE_ACTION in names and config.CARTESIAN_SERVICE in names
