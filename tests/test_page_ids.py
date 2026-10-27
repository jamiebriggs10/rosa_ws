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
PAGES = ["index.html", "stage.html"]
# Ids a shared script may ask for that a page is allowed not to have (the scripts guard them).
OPTIONAL = {"view-sethome", "view-cell", "saved-count", "saved-list", "voice-lamp", "voice-text", "voice-toggle",
            "voice-settings", "robot-widget", "robot-lamp", "robot-text", "tools", "think-time", "cancel",
            "hint", "pendant", "help", "settings-panel", "summary", "limits-edit", "limits-form", "limits-chips",
            "s-ollama", "s-moveit", "s-joint_states", "jointlist", "scroll", "messages", "gate", "thinking",
            "robot-mode-sim", "robot-mode-real", "robot-recheck", "robot-dialog", "voice-dialog"}
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
