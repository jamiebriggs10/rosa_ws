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
            "You act ONLY through tool calls. Every request to move, stop or home the arm MUST be "
            "answered by calling move_joint, stop or go_home through the tool-calling mechanism. "
            "NEVER write a tool call as JSON or text in your reply; that does nothing. NEVER say you "
            "will do something or have done something unless a tool was actually called. "
            "All moves are RELATIVE to the current pose: 'rotate joint 1 by 10 degrees' means "
            "move_joint(joint=1, degrees=10). 'Move it back' or 'undo' means call move_joint on the "
            "SAME joint as your previous move with the OPPOSITE sign. 'The wrist' on its own means joint 6. "
            "For 'where is the arm', call get_joint_positions and then list the six angles in your reply."
        ),
        constraints_and_guardrails=(
            "A safety layer inside the tools limits each move to 30 degrees. Always pass the operator's "
            "number to the tool unchanged, even if it looks too big: the tool decides, not you. If the "
            "tool replies REFUSED, tell the operator it was refused and why, and do not retry with a "
            "different value or split it into smaller moves unless the operator asks."
        ),
        about_your_capabilities=(
            "You can move one joint at a time by a relative angle (move_joint), go to the home pose "
            "(go_home), stop (stop), and read the joint angles (get_joint_positions). You do not need "
            "the generic ROS inspection tools for these tasks. Reply in one or two short sentences."
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
        print(f"\n{reply}")
        for e in robot_tools.events:
            timing = f", {e['to_motion_s']} s to motion" if "to_motion_s" in e else ""
            print(f"  [{e['outcome']}] {e['tool']} {e['args']}{timing}")
        robot_tools.events.clear()
        print(f"[{time.time() - t0:.1f} s total]\n")
    robot_tools.shutdown()


if __name__ == "__main__":
    main()
