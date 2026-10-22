#!/usr/bin/env bash
# The simulated lab UR12e: URSim (Universal Robots' own simulator, in Docker) for a UR12e,
# powered on with the brakes released, then the UR_humble driver and MoveIt against it,
# then the simulated arm moved to the agent's home pose. The simulator twin of
# ur_humble.real.sh. Named in ~/.config/rosa_ws/settings.json as "robot_start_sim", so
# ~/rosa_ws/start.sh (or start.sh sim) runs it in the tmux session "robot", log ~/robot.log.
# It starts the simulator itself, so no "sim_start" is needed next to it.
#
# Everything runs in the foreground so the tmux session keeps the log; the script ends when
# the bringup ends (start.sh stop). Written for the UR_humble cell: edit the launch line for
# another robot.
set -o pipefail           # not -e (a wait that times out is reported, not fatal) and not -u (ROS's setup.bash trips on it)
export ROS_LOCALHOST_ONLY=1       # keep our ROS graph on this computer (other laptops on the lab network run ROS too)
source /opt/ros/humble/setup.bash
source "$HOME/UR_humble/install/setup.bash"

URSIM_IP=192.168.56.101   # the address start_ursim.sh gives the container
dash() {
  # One command to URSim's dashboard server (port 29999), the same one the pendant uses.
  printf '%s\nquit\n' "$1" | timeout 8 nc "$URSIM_IP" 29999 | tr -d '\r' | grep -v -E '^(Connected|Disconnected)' || true
}

echo "[sim] 1/3 URSim"
# A leftover container from last time would hold the address, so remove it first. When this
# script is stopped the container is removed too.
docker rm -f ursim >/dev/null 2>&1 || true
trap 'docker rm -f ursim >/dev/null 2>&1' EXIT
ros2 run ur_client_library start_ursim.sh -m ur12e > "$HOME/ursim.log" 2>&1 &
for i in $(seq 1 60); do timeout 2 bash -c "echo > /dev/tcp/$URSIM_IP/29999" 2>/dev/null && break; sleep 3; done
sleep 15                  # the dashboard answers before the robot program is ready to be powered
dash "power on"; sleep 8
dash "brake release"
for i in $(seq 1 20); do dash robotmode | grep -q RUNNING && break; sleep 2; done
dash robotmode            # should print "Robotmode: RUNNING"

echo "[sim] 2/3 driver and MoveIt (lab bringup, sim mode)"
# sim:=true points the driver at URSim. enable_legacy_live_surface_selector: the click-to-raster
# selector, which needs only a point cloud (tests/sim/publish_flat_cloud.py when there is no camera).
ros2 launch bae_ur_bringup inspection_workflow.launch.py sim:=true enable_legacy_live_surface_selector:=true &
LAUNCH=$!

echo "[sim] 3/3 home pose"
# Wait until the driver has the joint controller active, which means it is talking to URSim.
for i in $(seq 1 40); do
  timeout 10 ros2 service call /controller_manager/list_controllers controller_manager_msgs/srv/ListControllers 2>/dev/null \
    | grep -q "name='scaled_joint_trajectory_controller', state='active'" && break
  sleep 3
done
sleep 5
# URSim can start in a pose that MoveIt's cell model says is inside a wall, and MoveIt then
# refuses every plan. Put the arm at the agent's home pose (config.py HOME_DEG, in radians
# here) straight through the joint controller. Tried three times: the first goal can arrive
# before the driver's connection to URSim has settled.
for i in 1 2 3; do
  if timeout 60 ros2 action send_goal /scaled_joint_trajectory_controller/follow_joint_trajectory control_msgs/action/FollowJointTrajectory \
      "{trajectory: {joint_names: [shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint], points: [{positions: [1.54, -1.62, 1.4, -1.2, -1.6, -0.11], time_from_start: {sec: 10}}]}}" \
      | grep -q SUCCEEDED; then
    echo "[sim] arm at the home pose, clear of the cell"; break
  fi
  [ "$i" = 3 ] && echo "[sim] WARNING: could not move the simulated arm to its home pose; MoveIt may refuse plans until it is moved"
  sleep 10
done

echo "[sim] running; the console's lights go green within about 30 s"
wait "$LAUNCH"
