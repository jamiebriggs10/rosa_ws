"""What Rosa says, at each moment of the loop, in two personas.

The console composes every spoken line from facts it measured (the plan's distance and
duration, the tool's recorded outcome), never from the model's prose, so the words are
always true. Two personas say the same facts in different words: A, the technician, terse;
B, the colleague, a little warmer. Refusals have one wording in both: no personality
when something is wrong.

    line("propose", "A", label="Point 10", distance_cm=14, seconds=6)
    -> "Point 10. 14 centimetres, 6 seconds, path clear. Confirm?"
"""

PERSONAS = ("A", "B")

# One template per kind and persona. {facts} are filled with str.format.
_LINES = {
    "preface": {"A": "On it.", "B": "Let me see."},
    "propose": {"A": "{label}. {distance_cm} centimetres, {seconds} seconds, path clear. Confirm?",
                "B": "{label}'s {distance_cm} centimetres away, about {seconds} seconds, path's clear. Confirm?"},
    "propose_joint": {"A": "{label}. {largest_turn_deg} degrees, {seconds} seconds, clear. Confirm?",
                      "B": "{label}, {largest_turn_deg} degrees, about {seconds} seconds. Confirm?"},
    "propose_trigger": {"A": "{label}. This moves the robot. Confirm?",
                        "B": "{label}. That one moves the robot, so: confirm?"},
    "changed": {"A": "{label} instead. {distance_cm} centimetres, {seconds} seconds, path clear. Confirm?",
                "B": "Okay, {label} instead: {distance_cm} centimetres, about {seconds} seconds. Confirm?"},
    "done": {"A": "At {label}.", "B": "There, {label}."},
    "done_trigger": {"A": "Done: {label}.", "B": "Done, {label}."},
    "cancel": {"A": "Cancelled.", "B": "Okay, cancelled."},
    "timeout": {"A": "{label} cancelled.", "B": "No answer, so {label} is cancelled."},
    "stopped": {"A": "Stopped at x {x}, y {y}, z {z}.", "B": "Stopped. We're at x {x}, y {y}, z {z}."},
    "refused": {"A": "{reason}", "B": "{reason}"},
    "result": {"A": "{text}", "B": "{text}"},
    "tour_intro": {"A": "{group} offers {count}.", "B": "{group} can do {count}."},
    "tour_item": {"A": "{name}: say \"{say}\".", "B": "{name}: just say \"{say}\"."},
    "tour_all": {"A": "{count} nodes are connected: {groups}. Say \"what can I do with\" and a name.",
                 "B": "{count} nodes are connected: {groups}. Ask me \"what can I do with\" any of them."},
    "movers": {"A": "These move the robot and will ask you to confirm: {names}.",
               "B": "These ones move the robot, so they'll always ask first: {names}."},
    "trace": {"A": "The last command went to {group}, {ros}.", "B": "That one went to {group}, through {ros}."},
}
KINDS = tuple(_LINES)
_MAX_RESULT = 200


def line(kind, persona="A", **facts):
    """The spoken line for `kind` with these facts. KeyError for a kind that does not exist."""
    templates = _LINES[kind]
    persona = persona if persona in templates else "A"
    facts = {k: _tidy(v) for k, v in facts.items()}
    if kind == "result":
        facts["text"] = _trim(str(facts.get("text", "")))
    if kind == "tour_intro":
        n = facts.get("count", 0)
        facts["count"] = f"{n} thing" + ("" if n == 1 else "s")
    if kind in ("tour_all", "movers") and isinstance(facts.get("groups" if kind == "tour_all" else "names"), (list, tuple)):
        key = "groups" if kind == "tour_all" else "names"
        facts[key] = ", ".join(str(x) for x in facts[key])
    text = templates[persona].format(**facts)
    if kind == "tour_item" and facts.get("moves") in (True, "True"):
        text += " It will ask you to confirm."
    return text


def _tidy(value):
    """Numbers without trailing .0, so lines read "14 centimetres" not "14.0 centimetres"."""
    if isinstance(value, float) and value == int(value):
        return int(value)
    if isinstance(value, float):
        return round(value, 1)
    return value


def _trim(text):
    """At most _MAX_RESULT characters, cut at the end of a sentence."""
    text = " ".join(text.split())
    if len(text) <= _MAX_RESULT:
        return text
    cut = text[:_MAX_RESULT]
    for mark in (". ", "! ", "? "):
        i = cut.rfind(mark)
        if i > 40:
            return cut[:i + 1]
    return cut[:_MAX_RESULT - 1].rstrip() + "."
