# The console's tests need rclpy, which only Ubuntu 22.04 + ROS 2 Humble has. This image is that
# plus the console's Python packages (including the voice package's), so one `pytest` in it runs
# the whole program's suite on a Mac. Build: docker build -t rosa-test -f tests/Dockerfile .
FROM ros:humble
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip libportaudio2 libsndfile1 \
    ros-humble-moveit-msgs ros-humble-tf2-ros-py ros-humble-sensor-msgs-py ros-humble-nav-msgs ros-humble-trajectory-msgs \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
