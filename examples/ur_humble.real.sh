#!/usr/bin/env bash
# The lab UR12e: start the UR_humble driver and MoveIt for the real cell, without the
# Zivid camera, and keep the scaled joint controller awake. Lives in rosa_ws/examples as
# ur_humble.real.sh and is named in ~/.config/rosa_ws/settings.json as "robot_start",
# so ~/rosa_ws/start.sh real runs it (tmux session "robot", log ~/robot.log). Its simulator
# twin is ur_humble.sim.sh. Lab-specific: edit ROBOT_IP and the launch line for another cell.
#
# Why the controller watcher: after every Play on the pendant the driver's watchdog
# leaves scaled_joint_trajectory_controller inactive, and MoveIt cannot move the arm
# until it is active again. The loop below switches it on whenever it is found inactive.
export ROS_LOCALHOST_ONLY=1       # keep our ROS graph separate from other laptops on the cell network
source /opt/ros/humble/setup.bash
source "$HOME/UR_humble/install/setup.bash"

# Wait for the control box before launching. If the driver starts first, its
# controller spawners give up after 10 s and the arm never publishes joint states.
ROBOT_IP=172.31.1.69
until ping -c 1 -W 1 "$ROBOT_IP" >/dev/null 2>&1; do
  echo "[lab-driver] waiting for the robot at $ROBOT_IP (cable in? control box on?)"
  sleep 3
done
echo "[lab-driver] robot reachable, launching the driver"

(
  sleep 20
  while true; do
    state="$(timeout 10 ros2 service call /controller_manager/list_controllers controller_manager_msgs/srv/ListControllers 2>/dev/null \
             | grep -o "name='scaled_joint_trajectory_controller', state='[a-z]*'" | grep -o "state='[a-z]*'")"
    if [ "$state" = "state='inactive'" ]; then
      echo "[lab-driver] scaled_joint_trajectory_controller is inactive: switching it on"
      timeout 15 ros2 service call /controller_manager/switch_controller controller_manager_msgs/srv/SwitchController \
        "{activate_controllers: [scaled_joint_trajectory_controller], strictness: 2, activate_asap: true, timeout: {sec: 5}}" >/dev/null 2>&1 \
        && echo "[lab-driver] controller active" || echo "[lab-driver] could not switch the controller on"
    fi
    sleep 5
  done
) &
WATCHER=$!
trap 'kill $WATCHER 2>/dev/null' EXIT

# Hardware is the launch file's default. No camera: the Zivid package is not built here.
# enable_legacy_live_surface_selector: the click-to-raster selector, which needs only a point cloud
# (a real scan, or tests/sim/publish_flat_cloud.py when the camera is off). Touchdown stays off here.
exec ros2 launch bae_ur_bringup inspection_workflow.launch.py launch_camera:=false enable_legacy_live_surface_selector:=true
