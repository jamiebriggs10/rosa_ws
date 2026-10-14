"""Type a plain-English command, and the arm moves through MoveIt.

The language model runs on the Mac (Ollama). This script runs on the ZenBook,
next to ROS, and uses ROSA (NASA JPL, Apache 2.0) to turn the model's
decisions into calls to the tools in robot_tools.py.
"""
import json
import time
import urllib.request

from langchain_ollama import ChatOllama
from rosa import ROSA, RobotSystemPrompts

import config
import robot_tools


def can_think(model):
    """True if the model can reason step by step before answering (e.g. qwen3)."""
    req = urllib.request.Request(config.OLLAMA_URL + "/api/show", data=json.dumps({"model": model}).encode())
    with urllib.request.urlopen(req, timeout=3) as r:
        return "thinking" in json.load(r).get("capabilities", [])


_llm = None   # the model connection in use, so abort() can hang up on it


def abort():
    """Hang up on Ollama mid-answer. Ollama then stops generating, instead of finishing
    an answer nobody wants while every later command waits behind it."""
    if _llm is not None:
        _llm._client._client.close()


def build_agent():
    global _llm
    thinking = config.THINK and can_think(config.MODEL)
    _llm = llm = ChatOllama(
        model=config.MODEL,
        base_url=config.OLLAMA_URL,
        temperature=0,           # deterministic: same command, same decision
        num_ctx=config.NUM_CTX,
        num_predict=config.MAX_THINKING_TOKENS if thinking else config.MAX_ANSWER_TOKENS,
        # Thinking models reason before answering: sometimes better, always much slower.
        # config.THINK switches it; models that cannot think are left alone.
        reasoning=thinking if can_think(config.MODEL) else None,
    )

    prompts = RobotSystemPrompts(
        embodiment_and_persona=(
            "You are the control agent for a robot arm. You move its tool in straight lines, "
            "in centimetres, measured from the robot's base. z is height: up is +z, down is -z. "
            "You can also turn one joint at a time by a number of degrees."
        ),
        critical_instructions=(
            "Every move tool works from where the tool is NOW, so you never need where_is_the_tool "
            "before a move, and never move to the coordinate origin: 'here' and 'the start' mean the "
            "tool's current position, which is [0, 0, 0] for move_path and move_formula. "
            "You act ONLY through tool calls. Every request to move, stop or home the arm MUST be "
            "answered by calling a tool through the tool-calling mechanism. NEVER write a tool call "
            "as JSON or text in your reply; that does nothing. NEVER say you will do something or "
            "have done something unless a tool was actually called. "
            "'Move up 5 cm' means move_by(dx=0, dy=0, dz=5). Use move_to only when the operator "
            "gives x, y and z. When a request needs more than one straight move (a route, a shape, 'then'), work out every "
            "point yourself, measured in cm from where the tool starts ([0, 0, 0]), and send them all "
            "in ONE move_path call, never several move_by calls. For circles, spirals, curves or "
            "many evenly spaced points, do not calculate numbers yourself: describe the points with "
            "a formula and use move_formula. "
            "When the operator names a joint (base, shoulder, elbow, wrist) or "
            "says degrees, use move_joint, never move_by. 'Move it back' or 'undo' means the SAME "
            "tool as your previous move with the OPPOSITE numbers: after move_joint(joint=1, "
            "degrees=10) it is move_joint(joint=1, degrees=-10); after move_by(dx=0, dy=0, dz=5) it "
            "is move_by(dx=0, dy=0, dz=-5). For 'where is the arm', call where_is_the_tool. "
            "After the tool returns, reply with what it said, not with what you did. Example: the "
            "tool returns 'Done. The tool is now at x=15.7, y=-63.3, z=82.8 cm.' and you reply "
            "'Done, the tool is now at x 15.7, y -63.3, z 82.8 cm.' Never mention tools or how you "
            "formatted the answer."
        ),
        constraints_and_guardrails=(
            "A safety layer inside the tools limits the size and place of every move. Always pass "
            "the operator's numbers to the tool unchanged, even if they look too big: the tool "
            "decides, not you. If the tool replies REFUSED, tell the operator it was refused and "
            "why, and do not retry with different numbers unless the operator asks. Never split a "
            "refused move_path into separate moves."
        ),
        about_your_capabilities=(
            "You can read where the tool is (where_is_the_tool), move it relative to where it is "
            "(move_by) or to a point (move_to), follow a route of many steps (move_path), follow points described by a formula "
            "(move_formula), turn one "
            "joint (move_joint), go to the home pose "
            "(go_home) and stop (stop). "
            "You do not need the generic ROS inspection tools for these tasks. Reply in one or two "
            "short sentences."
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
