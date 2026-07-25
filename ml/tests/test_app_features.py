r"""harmony-features-v1 (app spectralChroma port) tests — torch-free.

Numerical identity with the TS `chromaForChordFrame` was verified out-of-band on
real audio (max abs error ~1e-16). These lock the port's behaviour: pure tones map
to the right pitch class, chroma is L1-normalized, and the frame extractor emits
the right layout/version.

Run: .\.venv\Scripts\python.exe -m unittest discover -s ml/tests
"""
from __future__ import annotations

import unittest

import numpy as np

from ml.preprocessing.app_features import (
    FEATURE_PIPELINE_VERSION,
    FRAME_SIZE,
    app_spectral_chroma,
    extract_app_features,
)

SR = 44100


def _tone(freq: float, n: int = FRAME_SIZE) -> np.ndarray:
    return 0.5 * np.sin(2 * np.pi * freq * np.arange(n) / SR)


class AppFeatureTests(unittest.TestCase):
    def test_pure_tone_maps_to_pitch_class(self):
        # 440 Hz = A (pitch class 9); 261.63 Hz = C4 (pitch class 0).
        chroma_a, _ = app_spectral_chroma(_tone(440.0), SR)
        self.assertEqual(int(chroma_a.argmax()), 9)
        chroma_c, _ = app_spectral_chroma(_tone(261.63), SR)
        self.assertEqual(int(chroma_c.argmax()), 0)

    def test_chroma_is_l1_normalized(self):
        chroma, root = app_spectral_chroma(_tone(330.0), SR)
        self.assertAlmostEqual(float(chroma.sum()), 1.0, places=6)
        self.assertAlmostEqual(float(root.sum()), 1.0, places=6)

    def test_extract_layout_and_version(self):
        signal = _tone(440.0, SR * 2)  # 2 seconds
        frames = extract_app_features(signal, SR, hop_seconds=0.25)
        self.assertEqual(frames.pipeline_version, FEATURE_PIPELINE_VERSION)
        self.assertEqual(frames.chroma.shape[1], 12)
        self.assertEqual(frames.bass_chroma.shape[1], 12)
        self.assertEqual(frames.stacked().shape[1], 25)
        self.assertGreater(len(frames.times), 0)
        # ~2 s at 0.25 s hop -> ~8 frames.
        self.assertGreaterEqual(len(frames.times), 6)


if __name__ == "__main__":
    unittest.main()
