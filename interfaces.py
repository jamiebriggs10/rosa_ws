"""The description file: what the robot system's interfaces mean, and which move the robot.

Discovery (discovery.py) tells us a service called /touchdown_test/execute exists
and takes a std_srvs/Trigger. It cannot tell us that calling it drives a probe
onto a part. The system's owner writes that down once, in interfaces.yaml:

    system: UR_humble inspection cell
    interfaces:
      /multiview_scan/capture:
        means: Take one 3D view from where the camera is now and add it to the scan.
      /touchdown_test/execute:
        means: Run the previewed touchdown. The arm presses the probe on the surface.
        moves_robot: true

Rules the policy applies to every write (service call or publish):
- not in the file: refused, unless config.ALLOW_UNLISTED_WRITES is on, and then it
  still goes through the operator's confirm card;
- in the file without moves_robot: treated as moves_robot: true (confirm card);
- in the file with moves_robot: false: goes ahead at once.
Reads are always allowed.
"""
import pathlib

import yaml


class Description:
    def __init__(self, system="", entries=None):
        self.system = system
        self.entries = entries or {}      # name -> {"means": str, "moves_robot": bool}

    @classmethod
    def load(cls, path):
        """Read interfaces.yaml. A missing or empty file gives an empty description."""
        path = pathlib.Path(path).expanduser()
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except OSError:
            return cls()
        if not isinstance(data, dict):
            raise ValueError(f"{path} must be a YAML mapping with 'system' and 'interfaces'.")
        entries = {}
        for name, entry in (data.get("interfaces") or {}).items():
            entry = entry or {}
            if not isinstance(entry, dict):
                raise ValueError(f"{path}: entry {name} must be a mapping (means:, moves_robot:).")
            entries[str(name)] = {"means": str(entry.get("means", "")).strip(),
                                  "moves_robot": bool(entry.get("moves_robot", True))}
        return cls(str(data.get("system", "")).strip(), entries)

    def get(self, name):
        return self.entries.get(name)


def policy(description, name, allow_unlisted):
    """What a write to `name` may do: ("ok" | "confirm" | "refused", reason)."""
    entry = description.get(name)
    if entry is None:
        if allow_unlisted:
            return "confirm", f"{name} is not in the description file, so the operator must confirm it."
        return "refused", (f"{name} is not in the description file (interfaces.yaml), so writing to it is "
                           "not allowed. Ask the operator to describe it, or to allow unlisted writes.")
    if entry["moves_robot"]:
        return "confirm", f"{name} can move the robot, so the operator must confirm it."
    return "ok", ""


def report(description, interfaces):
    """One row per interface for the console: everything on the graph, plus anything
    the file describes that is missing from the graph."""
    rows, seen = [], set()
    for i in interfaces:
        entry = description.get(i.name)
        seen.add(i.name)
        rows.append({"name": i.name, "kind": i.kind, "type": i.type,
                     "means": entry["means"] if entry else "",
                     "moves_robot": entry["moves_robot"] if entry else None,
                     "status": "ready" if entry else "undescribed"})
    for name, entry in description.entries.items():
        if name not in seen:
            rows.append({"name": name, "kind": "", "type": "", "means": entry["means"],
                         "moves_robot": entry["moves_robot"], "status": "missing"})
    return rows
