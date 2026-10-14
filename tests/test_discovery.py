"""Tests for discovery: plumbing is dropped, user interfaces are kept, MoveIt is noticed."""
import config
from discovery import Interface, filter_graph, moveit_present

TOPICS = [("/rosout", ["rcl_interfaces/msg/Log"]), ("/parameter_events", ["rcl_interfaces/msg/ParameterEvent"]),
          ("/clicked_point", ["geometry_msgs/msg/PointStamped"]),
          ("/move_action/_action/feedback", ["moveit_msgs/action/MoveGroup_FeedbackMessage"]),
          ("/inspection/raster_path", ["nav_msgs/msg/Path"])]
SERVICES = [("/multiview_scan/capture", ["std_srvs/srv/Trigger"]),
            ("/move_group/get_parameters", ["rcl_interfaces/srv/GetParameters"]),
            ("/move_group/describe_parameters", ["rcl_interfaces/srv/DescribeParameters"]),
            ("/move_group/set_parameters_atomically", ["rcl_interfaces/srv/SetParametersAtomically"]),
            ("/move_group/get_type_description", ["type_description_interfaces/srv/GetTypeDescription"]),
            ("/rosa_moveit_client/list_parameters", ["rcl_interfaces/srv/ListParameters"]),
            ("/move_action/_action/cancel_goal", ["action_msgs/srv/CancelGoal"]),
            ("/compute_cartesian_path", ["moveit_msgs/srv/GetCartesianPath"])]
ACTIONS = [("/move_action", ["moveit_msgs/action/MoveGroup"])]


def test_plumbing_is_dropped_and_user_interfaces_kept():
    found = filter_graph(TOPICS, SERVICES, ACTIONS, own_names=("rosa_moveit_client",))
    names = [i.name for i in found]
    assert "/rosout" not in names and "/parameter_events" not in names
    assert not any("/_action/" in n or "parameters" in n or "type_description" in n for n in names)
    assert Interface("/multiview_scan/capture", "service", "std_srvs/srv/Trigger") in found
    assert Interface("/clicked_point", "topic", "geometry_msgs/msg/PointStamped") in found
    assert Interface("/move_action", "action", "moveit_msgs/action/MoveGroup") in found
    assert names == sorted(names)


def test_moveit_present_needs_both_names():
    found = filter_graph(TOPICS, SERVICES, ACTIONS)
    assert moveit_present(found)
    without = [i for i in found if i.name != config.CARTESIAN_SERVICE]
    assert not moveit_present(without)
