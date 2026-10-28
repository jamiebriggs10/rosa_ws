# Brief 2026-10-08: phase 7, training a custom wake word ("Truffles")

*Updated 8 Oct: the robot is now called Truffles, so the wake word is "Truffles" instead of "hey Rosa". Phrase decided: exactly "Truffles".*

**Status: option A chosen (Jamie, 8 Oct). Notebook ready: `training/truffles_wake_word.ipynb`, instructions in `training/README.md`.** Everything else (phases 0–6) is built; the plug-in already accepts a custom model (`trigger.openwakeword.model: models/truffles.onnx`, which appears in the page's dropdown once the file exists) and the benchmark can already score it.

## What training involves

openWakeWord's own training process (`openwakeword/train.py`, driven by `notebooks/automatic_model_training.ipynb` in its GitHub repo):

1. **Generate positive clips**: thousands of synthetic "Truffles" clips with Piper TTS (`piper-sample-generator`, many voices), plus "adversarial" near-misses ("truffle", "trouble", "shuffles", "ruffles").
2. **Augment**: mix with background noise/music and room reverb (impulse responses).
3. **Features**: run everything through openWakeWord's frozen mel + embedding models.
4. **Train** a small classifier (a few hundred kB) against ~2,000 h of pre-computed negative features; pick the model with the fewest false positives per hour on an 11 h validation set.
5. **Export** to ONNX (the only format we need) → `models/truffles.onnx`.

## What it needs (researched 8 Oct 2026; sizes approximate)

| Download | Size |
|---|---|
| ACAV100M negative features (`davidscripka/openwakeword_features`) | ~17.3 GB |
| Background audio (FMA small, converted to 16 kHz WAV) | ~8 GB |
| False-positive validation features | ~0.2 GB |
| MIT room impulse responses | ~0.05 GB |
| Piper TTS voice model (libritts_r medium) | ~0.06 GB |
| **Total data** | **~26 GB**, plus ~3–5 GB of Python packages and generated clips |

Packages: **PyTorch** + torchaudio, speechbrain, audiomentations, torch-audiomentations, datasets, piper-phonemize, and the notebook pins **tensorflow-cpu 2.8.1 + onnx-tf** (only for a TFLite export we don't need).

Time: ~75–90 min on a Colab T4 GPU; CPU-only is possible but many times slower (hours).

## The problem on the Mac

- `tensorflow-cpu==2.8.1` has no Apple-silicon / Python 3.11 build, and `piper-phonemize` has no arm64 wheel (needs building from source).
- The TFLite step can be skipped (we only need ONNX), but the script imports these pins, so it would need patching.
- So on the Mac it is: ~30 GB of disk, PyTorch (against the "ask first" rule), a source build, a patched training script, and several CPU hours. Possible, fiddly, and not reproducible on the lab PC without the same patching.

## Options for Jamie

| Option | Disk on the Mac | PyTorch on the Mac | Effort | Notes |
|---|---|---|---|---|
| **A. Official Colab notebook** (recommended) | none | no | lowest: run the notebook with `target_phrase: "truffles"`, download the `.onnx` | Training runs on Google's GPU. Nothing leaves except the phrase. The *runtime* stays fully local (the local-only rule is about speech processing at run time; check you're happy with this for the dissertation). |
| B. Lab PC (Linux + NVIDIA) when it arrives | none | no (PyTorch on the lab PC) | medium: all wheels exist on Linux x86 | Also local; delays phase 7 until the PC arrives. |
| C. On this Mac, in a separate venv (`.venv-train/`, never the app's venv) | ~30 GB | yes, in the separate venv | high: source-build piper-phonemize, patch out the TF/TFLite export, CPU hours | Keeps the app's venv PyTorch-free. Most likely to hit problems. |

## The phrase

**Decided (Jamie, 8 Oct): exactly "Truffles"**, one word. Known risk: openWakeWord recommends 3+ syllables; a two-syllable real word with close neighbours ("trouble", "truffle", "ruffles") may false-trigger more. We measure it (false wake-ups per hour) rather than guess, and tune the threshold.

## After the model exists (any option)

1. Copy it to `models/truffles.onnx`, then restart the app so the dropdown lists it.
2. `./.venv/bin/python bench/make_synthetic.py --wake "Truffles" --out bench/testset/synthetic-truffles` (synthetic wake clips; real ones from `bench/record_set.py`).
3. Run `presets/wake-truffles.yaml` and `presets/wake-jarvis.yaml` through `bench/run_bench.py --kinds wake,noise --dir bench/testset/synthetic-truffles` (and the Jarvis set for wake-jarvis), sweeping `threshold` 0.3–0.7: detection rate against false wake-ups per hour.
4. Live: pick the model in the page's dropdown, say "Truffles, go home".

**Done when** (from the build brief, with the new name): "Truffles" works live and its numbers are in the results.
