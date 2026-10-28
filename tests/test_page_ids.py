"""The pages' scripts only ever ask for elements that exist, and every script parses.

Pure Python (plus `node --check` when node is installed), so it runs on the Mac:
    python3 -m pytest tests/test_page_ids.py
"""
import pathlib
import re
import shutil
import subprocess

import pytest

STATIC = pathlib.Path(__file__).resolve().parent.parent / "static"
PAGES = ["stage.html"]
# Ids a shared script may ask for that a page is allowed not to have (the scripts guard them).
OPTIONAL = {"settings"}
ASKED = re.compile(r'(?:\$|getElementById)\(\s*"([\w-]+)"\s*\)')
QUERY = re.compile(r'querySelector(?:All)?\(\s*["\']#([\w-]+)')
DEFINED = re.compile(r'id="([\w-]+)"')
SCRIPT = re.compile(r"<script(?P<attrs>[^>]*)>(?P<body>.*?)</script>", re.S)
SRC = re.compile(r'src="/static/([\w.-]+)"')
JS_IDS = re.compile(r'\bid:\s*"([\w-]+)"|\bid="([\w-]+)"')     # elements a script builds itself


def scripts_of(page_text):
    """The inline script bodies and the static script files a page uses."""
    inline, files = [], []
    for m in SCRIPT.finditer(page_text):
        src = SRC.search(m.group("attrs"))
        if src:
            files.append(STATIC / src.group(1))
        elif "importmap" not in m.group("attrs"):
            inline.append(m.group("body"))
    return inline, files


def existing_pages():
    return [p for p in PAGES if (STATIC / p).exists()]


@pytest.mark.parametrize("page", existing_pages())
def test_every_id_the_scripts_ask_for_exists(page):
    text = (STATIC / page).read_text()
    defined = set(DEFINED.findall(text))
    inline, files = scripts_of(text)
    code = "\n".join(inline + [f.read_text() for f in files])
    for a, b in JS_IDS.findall(code):
        defined.add(a or b)
    asked = set(ASKED.findall(code)) | set(QUERY.findall(code))
    missing = sorted(i for i in asked if i not in defined and i not in OPTIONAL and not i.startswith("voice-load-"))
    assert not missing, f"{page} asks for ids it does not have: {missing}"


def test_scripts_parse():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    checked = 0
    for page in existing_pages():
        inline, files = scripts_of((STATIC / page).read_text())
        for i, body in enumerate(inline):
            path = STATIC / f".check-{page}-{i}.js"
            path.write_text(body)
            try:
                subprocess.run([node, "--check", str(path)], check=True, capture_output=True)
            finally:
                path.unlink()
            checked += 1
        for f in files:
            subprocess.run([node, "--check", str(f)], check=True, capture_output=True)
            checked += 1
    assert checked > 0


MAP_PROBE = r"""
const fs = require("fs"); const vm = require("vm");
const ctx = {console, window: {}, d3: {}};
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.argv[2], "utf8") + "\nglobalThis.RosaMap = RosaMap;", ctx);
const rows = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const out = {};
for (const showRos of [false, true]) {
  const sel = ctx.RosaMap.select(rows, showRos);
  out[showRos ? "ros" : "plain"] = {groups: sel.groups.map(g => g.id), offers: sel.offers.length,
    collapsed: sel.groups.filter(g => g.collapsed).map(g => g.id)};
}
console.log(JSON.stringify(out));
"""


def test_map_select_hides_the_undescribed_by_default(tmp_path):
    node = shutil.which("node")
    if not node or not (STATIC / "map.js").exists():
        pytest.skip("node or map.js missing")
    import json
    rows = [{"name": "/multiview_scan/capture", "node": "multiview_scan", "status": "ready", "say": "take a scan view", "moves_robot": False, "means": "Take one view", "kind": "service", "type": "Trigger"},
            {"name": "/multiview_scan/fused", "node": "multiview_scan", "status": "undescribed", "say": "/multiview_scan/fused", "moves_robot": None, "means": "", "kind": "topic", "type": "PointCloud2"},
            {"name": "go_home", "node": "moveit", "status": "ready", "say": "go home", "moves_robot": True, "means": "Home", "kind": "tool", "type": "MoveGroup action"}]
    for i in range(160):
        rows.append({"name": f"/controller_{i % 8}/thing_{i}", "node": f"controller_{i % 8}", "status": "undescribed", "say": f"/controller_{i % 8}/thing_{i}",
                     "moves_robot": None, "means": "", "kind": "topic", "type": "std_msgs/msg/Float64"})
    probe, data = tmp_path / "probe.js", tmp_path / "rows.json"
    probe.write_text(MAP_PROBE)
    data.write_text(json.dumps(rows))
    out = json.loads(subprocess.run([node, str(probe), str(STATIC / "map.js"), str(data)], check=True, capture_output=True, text=True).stdout)
    assert out["plain"]["groups"] == ["multiview_scan", "moveit"] and out["plain"]["offers"] == 2
    assert out["ros"]["offers"] == 3                                   # the described group's undescribed row joins it
    assert len(out["ros"]["collapsed"]) == 8 and out["ros"]["collapsed"][0].startswith("controller_")


def test_the_voice_form_has_no_generated_parts():
    page = (STATIC / "stage.html").read_text()
    text = page + (STATIC / "panels.js").read_text()
    for gone in ("voice-stages", "voice-preset", "voice-url", "renderVoiceStages", "vapi(", "voice-settings-body", "workbench"):
        assert gone not in text, gone
    for there in ("s-input-device", "s-wake-threshold", "s-silence-ms", "s-output-device", "s-speech-enabled", "s-sounds-enabled"):
        assert f'id="{there}"' in page, there


def test_every_style_rule_has_something_to_style():
    """A class in panels.css must be used by the page or a script, or it is dead."""
    css = (STATIC / "panels.css").read_text()
    used = "\n".join((STATIC / f).read_text() for f in ("stage.html", "panels.js", "map.js", "view.js"))
    classes = set(re.findall(r"\.([a-zA-Z][\w-]*)", re.sub(r"/\*.*?\*/", "", css, flags=re.S)))
    dead = sorted(c for c in classes if not re.search(r"\b" + re.escape(c) + r"\b", used))
    assert not dead, f"panels.css styles nothing with: {dead}"


def test_the_code_no_longer_speaks_of_the_voice_app():
    root = STATIC.parent
    files = [root / n for n in ("web_console.py", "config.py", "lines.py")] + list((root / "voice").glob("*.py")) + [f for f in STATIC.iterdir() if f.suffix in (".html", ".js", ".css")]
    hits = [f"{f.name}:{i}" for f in files for i, line in enumerate(f.read_text().splitlines(), 1) if "voice app" in line.lower()]
    assert not hits, hits
