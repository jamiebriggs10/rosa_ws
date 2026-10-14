#!/usr/bin/env python3
"""Test helper, not part of the product: send commands to a running console and print
what happened, the way an operator at the page would see it.

    python3 tests/sim/console_session.py http://192.168.1.145:8000 "move up 5 cm" "go home"
    python3 tests/sim/console_session.py --decline http://... "run the touchdown"

Each command is sent, the confirm card (if one appears) is answered yes, or no with
--decline, and the tool calls, outcomes, timings and the reply are printed as a
Markdown table row, ready for a log. Needs only Python; runs from any machine.
"""
import argparse
import json
import sys
import time
import urllib.request


def call(base, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def run(base, command, confirm, timeout=240):
    status, body = call(base, "/command", {"text": command})
    if status != 200:
        return {"command": command, "reply": f"not sent: {body.get('error')}", "events": [], "total_s": 0}
    seen = len(call(base, "/state")[1]["messages"]) - 1
    t0, answered = time.time(), set()
    while time.time() - t0 < timeout:
        s = call(base, "/state")[1]
        p = s["pending"]
        if p and p["id"] not in answered:
            answered.add(p["id"])
            time.sleep(0.5)                               # a human would look at the card first
            call(base, "/confirm", {"id": p["id"], "ok": confirm})
        if not s["busy"] and len(s["messages"]) > seen:
            m = s["messages"][-1]
            return {"command": command, "reply": m["text"], "events": m.get("events", []),
                    "total_s": m.get("total_s", 0), "confirmed": bool(answered) and confirm,
                    "declined": bool(answered) and not confirm}
        time.sleep(0.5)
    return {"command": command, "reply": "(timed out)", "events": [], "total_s": timeout}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("commands", nargs="+")
    ap.add_argument("--decline", action="store_true", help="answer no to any confirm card")
    a = ap.parse_args()
    print("| Command | Tool calls | Outcome | Reply | Time |")
    print("|---|---|---|---|---|")
    for c in a.commands:
        r = run(a.base.rstrip("/"), c, confirm=not a.decline)
        calls = "; ".join(f"{e['tool']}({', '.join(f'{k}={v}' for k, v in e['args'].items())})" for e in r["events"]) or "none"
        outcome = ", ".join(e["outcome"] for e in r["events"]) or "no tool"
        if r.get("declined"):
            outcome += " (card declined)"
        elif r.get("confirmed"):
            outcome += " (card confirmed)"
        reply = r["reply"].replace("|", "/").replace("\n", " ")[:160]
        print(f"| {c} | {calls} | {outcome} | {reply} | {r['total_s']} s |")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
