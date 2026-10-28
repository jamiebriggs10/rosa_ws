"""The web page: it is served, its ids exist, every URL its script calls exists in app.py,
and the script parses. No worker thread is started, so no models load."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

import app as app_mod

ROOT = Path(__file__).resolve().parent.parent
HTML = (ROOT / "static" / "index.html").read_text()


@pytest.fixture()
def client(tmp_path):
    settings = tmp_path / "settings.yaml"
    shutil.copy(ROOT / "settings.example.yaml", settings)
    app = app_mod.create_app(settings_path=settings, log_path=tmp_path / "live.jsonl")
    return app, app.test_client()


def test_index_served_with_ids(client):
    _, c = client
    r = c.get("/")
    assert r.status_code == 200
    page = r.get_data(as_text=True)
    ids = set(re.findall(r'\$\("([\w-]+)"\)', HTML)) | set(re.findall(r'getElementById\("([\w-]+)"\)', HTML))
    assert ids, "script should look up elements by id"
    for i in ids:
        assert f'id="{i}"' in page, f"missing element id {i}"


def test_script_urls_exist_in_app(client):
    app, _ = client
    urls = set(re.findall(r'(?:fetch|api|EventSource)\(\s*[`"\'](/[^`"\'?$]*)', HTML))
    urls.update(re.findall(r'api\(`(/[^`$?]*)', HTML))
    urls.update(re.findall(r'"(/ptt/\w+)"', HTML))      # chosen by a ternary, so not directly inside api(
    assert {"/events", "/settings", "/options", "/status", "/history", "/presets", "/wake", "/ptt/down", "/ptt/up"} <= urls
    patterns = [re.sub(r"<[^>]+>", "X", r.rule) for r in app.url_map.iter_rules()]
    for u in urls:
        if u.startswith("/static/"):                  # plain files Flask serves itself (e.g. help.json)
            assert (Path(app.static_folder) / u[len("/static/"):]).exists(), f"{u} missing"
            continue
        probe = re.sub(r"/[^/]*$", "/X", u) if u.endswith("/") else u
        assert u in patterns or probe in patterns, f"{u} not served by app.py"
    # the templated preset URLs
    assert "/presets/X" in patterns and "/presets/X/load" in patterns


def test_options_lists_four_stages(client):
    _, c = client
    opts = c.get("/options").get_json()["options"]
    assert set(opts) == {"trigger", "vad", "endpoint", "stt"}


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_script_parses(tmp_path):
    js = re.search(r"<script>(.*)</script>", HTML, re.S).group(1)
    f = tmp_path / "page.js"
    f.write_text(js)
    r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
