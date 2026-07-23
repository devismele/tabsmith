"""Guard: the foundation pipeline must import and run without PyTorch.

Runs a subprocess with a meta-path hook that blocks torch/onnx imports, then
exercises schema -> synth -> features -> baseline -> metrics -> importer. If any
foundation module imported torch at module load, the subprocess would fail.
"""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_SCRIPT = r"""
import sys

class _Blocker:
    _blocked = ("torch", "onnx", "onnxruntime")
    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in self._blocked:
            raise ImportError(f"blocked heavy dependency: {name}")
        return None

sys.meta_path.insert(0, _Blocker())

from ml.preprocessing.synth_generator import generate_track
from ml.preprocessing.features import extract_features
from ml.evaluation.baselines import chroma_template_predict
from ml.evaluation.metrics import evaluate_regions
from ml.preprocessing.import_tabsmith_ref import import_tabsmith_references

track, samples, sr = generate_track(0, 1, 6.0, (120, 120))
features = extract_features(samples, sr)
predicted = chroma_template_predict(features)
result = evaluate_regions(track.chords, predicted)
tracks, summary = import_tabsmith_references()

assert "torch" not in sys.modules, "torch was imported by a foundation module"
assert result["rootAccuracy"] >= 0
assert summary["totalSongs"] >= 1
print("OK")
"""


class NoTorchDependencyTests(unittest.TestCase):
    def test_foundation_runs_without_torch(self):
        result = subprocess.run([sys.executable, "-c", _SCRIPT], cwd=REPO_ROOT,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, msg=f"stderr:\n{result.stderr}")
        self.assertIn("OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
