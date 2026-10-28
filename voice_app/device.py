"""Work out which compute device this machine has: cuda, metal or cpu.

This is the ONLY place that asks "what hardware am I on?". Plug-ins are told
the answer and pick their own backend from it (docs/03-portability.md).

Run it directly to see what was found:  ./.venv/bin/python device.py
"""

import platform
import shutil
import subprocess


def has_cuda() -> bool:
    """An NVIDIA GPU with a working driver (nvidia-smi lists at least one GPU)."""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
        return out.returncode == 0 and "GPU" in out.stdout
    except (OSError, subprocess.TimeoutExpired):
        return False


def has_metal() -> bool:
    """Apple silicon (M-series) Mac: the GPU is reachable through Metal / MLX."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def detect() -> str:
    """The best device available: cuda, then metal, then cpu."""
    if has_cuda():
        return "cuda"
    if has_metal():
        return "metal"
    return "cpu"


def resolve(choice: str) -> str:
    """Turn the `device:` setting into a real device. "auto" means detect()."""
    choice = (choice or "auto").lower()
    if choice == "auto":
        return detect()
    if choice not in ("cuda", "metal", "cpu"):
        raise ValueError(f"unknown device {choice!r} (use auto, cuda, metal or cpu)")
    return choice


def describe() -> dict:
    """Facts about this machine for the settings page and the benchmark logs."""
    return {
        "host": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "detected": detect(),
        "available": [d for d, ok in (("cuda", has_cuda()), ("metal", has_metal()), ("cpu", True)) if ok],
    }


if __name__ == "__main__":
    print(detect())
