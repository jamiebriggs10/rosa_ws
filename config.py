"""One place for the settings that change between machines and experiments.

Everything can be overridden with an environment variable, so swapping the
model is a one-line change:  ROSA_MODEL=qwen3:8b python rosa_agent.py
"""
import os

# Where Ollama is running. The Mac's LAN address, port 11434 is Ollama's default.
MAC_IP = os.environ.get("MAC_IP", "192.168.1.148")
OLLAMA_URL = os.environ.get("OLLAMA_URL", f"http://{MAC_IP}:11434")

# Which local model to use. Must support tool calling.
MODEL = os.environ.get("ROSA_MODEL", "llama3.1:8b")

# Some setups prefix every joint name, e.g. "UR20265102792_shoulder_pan_joint"
# when talking to the robot controller's own ROS 2 interface. Empty for the
# plain UR driver.
JOINT_PREFIX = os.environ.get("JOINT_PREFIX", "")

# The trajectory action server to send moves to.
ACTION_NAME = os.environ.get("ACTION_NAME", "/scaled_joint_trajectory_controller/follow_joint_trajectory")

# How much text the model can hold in its head at once (tokens). ROSA's system
# prompt plus the tool descriptions are long, so Ollama's default of 2048 is
# too small and the model would silently forget its instructions.
NUM_CTX = int(os.environ.get("ROSA_NUM_CTX", "8192"))
