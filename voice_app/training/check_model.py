"""First sanity check of a freshly trained wake-word model (runs on Colab).

    python check_model.py <output_dir> <model_name>

Plays held-out SYNTHETIC test clips (made during training but never trained on)
through the model the same way the prototype does (80 ms chunks), and writes
training_report.json. These numbers are an optimistic first look: the real
test is Jamie's voice, in the prototype's benchmark.
"""

import glob
import json
import sys

# This script sits in /content next to the openWakeWord *repository* folder, which is
# also called "openwakeword". Python searches the script's own folder first, so it would
# import that folder instead of the installed package. Remove the script's folder from
# the search path (found on the first Colab run: "has no attribute get_pretrained_model_paths").
sys.path.pop(0)

import numpy as np
import soundfile as sf
from openwakeword.model import Model

out_dir, name = sys.argv[1], sys.argv[2]
model = Model(wakeword_models=[f"{out_dir}/{name}.onnx"], inference_framework="onnx")


def peak_score(path):
    """Highest score the model gives anywhere in the clip (with 1 s of silence either side)."""
    audio, _ = sf.read(path, dtype="int16")
    pad = np.zeros(16000, np.int16)
    audio = np.concatenate([pad, audio, pad])
    model.reset()
    return max(list(model.predict(audio[i:i + 1280]).values())[0]
               for i in range(0, len(audio) - 1279, 1280))


positives = [peak_score(p) for p in sorted(glob.glob(f"{out_dir}/{name}/positive_test/*.wav"))[:300]]
near_misses = [peak_score(p) for p in sorted(glob.glob(f"{out_dir}/{name}/negative_test/*.wav"))[:300]]

report = {"model": name, "synthetic_test_clips": True,
          "positive_clips": len(positives), "near_miss_clips": len(near_misses)}
for threshold in (0.3, 0.5, 0.7):
    report[f"detected_at_{threshold}"] = round(float(np.mean([s >= threshold for s in positives])), 3)
    report[f"near_miss_fired_at_{threshold}"] = round(float(np.mean([s >= threshold for s in near_misses])), 3)
json.dump(report, open(f"{out_dir}/training_report.json", "w"), indent=2)
print(json.dumps(report, indent=2))
