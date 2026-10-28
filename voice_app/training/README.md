# training/: making the "Truffles" wake word (phase 7, option A: Google Colab)

Training runs on Google Colab (a free GPU in the browser), **not on the Mac**. The only thing that comes back is a small model file, `truffles.onnx`, which the prototype then runs fully offline.

| File | What |
|---|---|
| `truffles_wake_word.ipynb` | The notebook to upload to Colab. Self-contained: the two scripts below are embedded in it. |
| `prepare_data.py` | Downloads the training data on Colab (room echoes, background audio, pre-computed negative features). |
| `check_model.py` | First check of the new model on held-out synthetic clips; writes `training_report.json`. |

Background: what training does and why, `briefs/2026-10-08-phase7-wake-word.md`.

## Step by step

1. Go to https://colab.research.google.com and sign in with your Google account.
2. **File → Upload notebook →** choose `training/truffles_wake_word.ipynb` from this folder.
3. **Runtime → Change runtime type →** Hardware accelerator **T4 GPU** → **Save**.
4. **Quick test first.** The first code cell says `QUICK_TEST = True`; leave it. **Runtime → Run all.** If Colab warns that the notebook wasn't written by Google, click **Run anyway**.
   - Takes about 15 minutes. Cells run top to bottom; a green tick appears beside each finished one.
   - It ends with **"Quick test passed"**. If instead a cell stops with red error text, copy that text and send it to Claude.
5. **The real run.** In the first cell change `QUICK_TEST = True` to `QUICK_TEST = False`, then **Runtime → Run all** again.
   - Takes about 1.5–2.5 hours. Keep the tab open (a closed or idle tab can be disconnected). If it disconnects, **Run all** again: it carries on from the clips it has already made.
6. At the end your browser downloads three files: `truffles.onnx`, `training_report.json`, `truffles_training_config.yaml`.
7. Put them in the prototype:
   - `truffles.onnx` → `models/truffles.onnx`
   - the other two → `training/results/` (kept for the dissertation: exact settings and first numbers)
8. Tell Claude. Next: restart the app, pick `models/truffles.onnx` in the wake-word dropdown, say "Truffles, go home", then the benchmark against `hey_jarvis`.

## What was changed from openWakeWord's official notebook (and why)

The official `automatic_model_training.ipynb` (2023) no longer runs on Colab as of October 2026. Checked by reading its code and package lists:

| Problem | Fix |
|---|---|
| `piper-phonemize` has builds only up to Python 3.11; Colab is 3.12+ | `piper-phonemize-fix` 1.2.2: same `piper_phonemize` module, built for 3.12/3.13 |
| piper-sample-generator was reorganised in March 2026; `generate_samples.py` moved, so `train.py` can't import it | pin to the v2.0.0 tag (matches `train.py` and the `.pt` voice model) |
| torchaudio 2.9+ removed `load()`/`info()`, which openWakeWord's training code uses | pin `torch==2.8.0`, `torchaudio==2.8.0` |
| Colab's pre-installed torchvision is built for its newer PyTorch; after pinning 2.8, `speechbrain` crashes importing it (`operator torchvision::nms does not exist`), found on the first Colab run | pin `torchvision==0.23.0` (the version that matches torch 2.8.0) |
| PyTorch 2.6+ won't load a whole pickled model with `torch.load` by default (the Piper voice file) | `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for that trusted file |
| AudioSet `bal_train09.tar` link is dead (404); now Parquet files | read `data/bal_train/00.parquet`, `01.parquet` directly |
| Music source `rudraml/fma` is a `datasets` loading script that new versions don't run | dropped; AudioSet's balanced set already includes music |
| `tensorflow-cpu==2.8.1` / `onnx_tf` don't install on new Python | not installed: only needed for the optional `.tflite` export; the prototype uses `.onnx` |
| openWakeWord bug: `train.py`'s flags default to the string `"False"` (truthy), so it always attempts the TFLite conversion after saving the ONNX model, and fails without TensorFlow; found on the first Colab run | `sed` the defaults to real `False` after checkout |
| `check_model.py` runs in `/content`, next to the repository folder also named `openwakeword`, which Python imported instead of the installed package (`no attribute get_pretrained_model_paths`); found on the first Colab run, reproduced locally | the script removes its own folder from Python's search path |
| quick test and real run would share a folder, and later steps reuse existing features | separate `quick_test_model/` and `my_custom_model/` |
| `deep-phonemizer` only used for words missing from the CMU dictionary | not needed: "truffles" is in it |

Pinned openWakeWord commit: `368c03716d1e92591906a84949bc477f3a834455`. Training settings: 6,000 positive + 6,000 near-miss clips, 1,500 of each held out, 30,000 steps, openWakeWord's defaults otherwise (`dnn`, layer size 32, target ≤ 0.2 false positives per hour).
