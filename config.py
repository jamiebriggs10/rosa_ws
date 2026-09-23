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

# How much text the model can hold in its head at once (tokens). ROSA's system
# prompt plus the tool descriptions are long, so Ollama's default of 2048 is
# too small and the model would silently forget its instructions.
NUM_CTX = int(os.environ.get("ROSA_NUM_CTX", "8192"))
