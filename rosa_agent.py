"""Type a plain-English command, and the simulated UR12e moves.

The language model runs on the Mac (Ollama). This script runs on the ZenBook,
next to ROS, and uses ROSA (NASA JPL, Apache 2.0) to turn the model's
decisions into calls to the tools in robot_tools.py.
"""
import sys
import time

from langchain_ollama import ChatOllama
from rosa import ROSA, RobotSystemPrompts

import config
import robot_tools


def build_agent():
    llm = ChatOllama(
        model=config.MODEL,
        base_url=config.OLLAMA_URL,
        temperature=0,           # deterministic: same command, same decision
        num_ctx=config.NUM_CTX,
    )

    prompts = RobotSystemPrompts(
        embodiment_and_persona=(
            "You are the control agent for a Universal Robots UR12e, a six-axis collaborative "
            "robot arm. Joints are numbered 1 to 6 from the base: 1 shoulder pan (base), "
            "2 shoulder lift, 3 elbow, 4 wrist 1, 5 wrist 2, 6 wrist 3."
        ),
        critical_instructions=(
            "When the operator asks for a motion you MUST call move_joint, go_home or stop. "
            "Never describe what you would do instead of doing it. All moves are RELATIVE to "
            "the current pose: 'rotate joint 1 by 10 degrees' means move_joint(joint=1, degrees=10). "
            "'Move it back' means the same joint by the opposite sign of the previous move. "
            "'The wrist' on its own means joint 6. To answer where the arm is, call get_joint_positions."
        ),
        constraints_and_guardrails=(
            "Moves are limited to 30 degrees per call by a safety layer. If a tool replies REFUSED, "
            "report that refusal to the operator plainly and do not retry with a different value "
            "unless the operator asks. Do not chain several moves to get round the limit."
        ),
        about_your_capabilities=(
            "You can only move one joint at a time by a relative angle, go to the home pose, stop, "
            "and read the joint angles. Reply in one or two short sentences."
        ),
    )

    return ROSA(
        ros_version=2,
        llm=llm,
        tools=robot_tools.ALL_TOOLS,
        prompts=prompts,
        verbose=True,            # prints each tool call, useful evidence for the write-up
        streaming=False,
    )


def main():
    print(f"Model {config.MODEL} at {config.OLLAMA_URL}. Type a command, or 'quit'.")
    agent = build_agent()
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line.lower() in ("quit", "exit"):
            break
        robot_tools.mark_command_time()
        t0 = time.time()
        try:
            reply = agent.invoke(line)
        except Exception as exc:          # keep the loop alive on model or ROS errors
            reply = f"Error: {exc}"
        print(f"\n{reply}\n[{time.time() - t0:.1f} s total]\n")


if __name__ == "__main__":
    main()
