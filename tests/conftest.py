"""Keep the tests away from the real settings file: config reads ~/.config/rosa_ws/settings.json
at import, and an operator's saved limits or model must not change what the tests see."""
import os
import tempfile

os.environ["ROSA_SETTINGS"] = os.path.join(tempfile.mkdtemp(), "settings.json")   # does not exist: defaults apply
