# 2026-10-28: the voice app moves into the console's repository

**What:** this folder, the Voice Control Prototype, is now `rosa_ws/voice_app/`
inside the console's git repository (`github.com/jamiebriggs10/rosa_ws`). On the Mac
the old path `Dissertation/Voice Control Prototype` is a symlink to it. It is still
its own process on port 8100 with its own venv; nothing in the code moved.

**Why:** it had no git history (the speech work of 2026-10-27, `speech.py` and the
`/say`, `/play`, `/followup` routes, was done in a folder nobody could diff), and the
lab PC needed a hand copy of the folder. Now one `git pull` updates the console and
the voice app together, and `start.sh` already starts both. Jamie's question was
whether to fold it into the console as one program; the answer was no: the speech
libraries are heavy and pinned differently from ROS's Python, an audio crash must
not stop the console, and the tests run on the Mac without ROS.

**Not in git:** `models/` (4 GB), `.venv/`, `runs/` (every utterance's JSONL),
`bench/testset/` (46 MB of recordings, including Jamie's voice), `cache/` (spoken
phrases), `tests/.clips/`. The lab PC's copies of models and settings are moved by
hand once (`docs/06-lab-pc-setup.md` in the project folder, section 5b).

**Checked:** `./.venv/bin/python -m pytest -m "not model"` from the new folder: 95
passed (the moved venv still works; its `python` links to Homebrew's 3.11).
