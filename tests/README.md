# Running the tests

On the lab PC (or any Ubuntu with ROS 2 Humble): `./venv/bin/python -m pytest`.

On the Mac there is no rclpy, so the console's tests run in Docker:

```bash
docker build -t rosa-test -f tests/Dockerfile .          # once, and again when requirements.txt changes
tar c --exclude venv --exclude .git --exclude models . | docker run --rm -i rosa-test \
  bash -c 'mkdir -p /ws && tar x -C /ws && cd /ws && python3 -m pytest -q'
```

The pure-Python tests (words, audio, stages, pipeline, speech, the Voice object, the settings,
the lines, the page ids, the mock) also run natively in the Mac venv:
`venv/bin/python -m pytest tests/test_words.py tests/test_voice_*.py tests/test_speech.py tests/test_settings.py tests/test_lines.py tests/test_page_ids.py tests/test_mock_console.py`.

Tests marked `model` load the real speech models (downloaded into `models/` on first run) and
are skipped unless asked for: `venv/bin/python -m pytest -m model`.
