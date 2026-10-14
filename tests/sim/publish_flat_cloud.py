#!/usr/bin/env python3
"""Test helper, not part of the product: publish a flat square of points as a PointCloud2,
standing in for a scanned part so the lab's surface selector can be tried without a camera.

    ./venv/bin/python tests/sim/publish_flat_cloud.py [--topic /inspection/workspace_cloud]
        [--frame base] [--x 0.30] [--y -0.60] [--z 0.05] [--size 0.40] [--spacing 0.005]

Publishes once a second until stopped. Positions are metres in --frame.
"""
import argparse

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/inspection/workspace_cloud")
    ap.add_argument("--frame", default="base")
    ap.add_argument("--x", type=float, default=0.30)
    ap.add_argument("--y", type=float, default=-0.60)
    ap.add_argument("--z", type=float, default=0.05)
    ap.add_argument("--size", type=float, default=0.40)
    ap.add_argument("--spacing", type=float, default=0.005)
    a = ap.parse_args()
    n = int(a.size / a.spacing) + 1
    points = [[a.x - a.size / 2 + i * a.spacing, a.y - a.size / 2 + j * a.spacing, a.z]
              for i in range(n) for j in range(n)]
    rclpy.init()
    node = Node("flat_cloud_for_testing")
    pub = node.create_publisher(PointCloud2, a.topic, qos_profile_sensor_data)

    def tick():
        header = Header(frame_id=a.frame, stamp=node.get_clock().now().to_msg())
        pub.publish(point_cloud2.create_cloud_xyz32(header, points))

    node.create_timer(1.0, tick)
    print(f"Publishing {len(points)} points, a {a.size} m square at z={a.z} in '{a.frame}', on {a.topic}. Ctrl-C to stop.")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
