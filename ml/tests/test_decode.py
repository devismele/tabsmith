r"""Viterbi decode tests (stdlib unittest, torch-free — pure numpy decoder).

Run: .\.venv\Scripts\python.exe -m unittest discover -s ml/tests
"""
from __future__ import annotations

import unittest

import numpy as np

from ml.evaluation.decode import viterbi_decode


def _probs(root_seq, quality_idx=0):
    """One-hot-ish frame probabilities for a sequence of root pitch classes."""
    T = len(root_seq)
    root = np.full((T, 12), 0.01)
    quality = np.full((T, 3), 0.01)
    nochord = np.full(T, 0.01)
    for i, r in enumerate(root_seq):
        root[i] = 0.01
        root[i, r] = 0.9
        quality[i, quality_idx] = 0.9
    return root, quality, nochord


class ViterbiDecodeTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(viterbi_decode([], np.zeros((0, 12)), np.zeros((0, 3)), np.zeros(0), 0.1), [])

    def test_stable_sequence_one_region(self):
        root, quality, nochord = _probs([0] * 10)
        times = np.arange(10) * 0.1
        regions = viterbi_decode(times, root, quality, nochord, 0.1, transition_penalty=4.0)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].label, "C:maj")

    def test_penalty_absorbs_single_frame_flicker(self):
        # A lone G frame inside a run of C should be smoothed away at high penalty.
        root, quality, nochord = _probs([0, 0, 0, 7, 0, 0, 0])
        times = np.arange(7) * 0.1
        strong = viterbi_decode(times, root, quality, nochord, 0.1, transition_penalty=8.0)
        self.assertEqual([r.label for r in strong], ["C:maj"])
        # With no penalty the flicker survives as its own region.
        weak = viterbi_decode(times, root, quality, nochord, 0.1, transition_penalty=0.0)
        self.assertIn("G:maj", [r.label for r in weak])

    def test_genuine_change_is_kept(self):
        # A real, sustained change from C to G must survive even a stiff penalty.
        root, quality, nochord = _probs([0] * 6 + [7] * 6)
        times = np.arange(12) * 0.1
        regions = viterbi_decode(times, root, quality, nochord, 0.1, transition_penalty=6.0)
        self.assertEqual([r.label for r in regions], ["C:maj", "G:maj"])

    def test_nochord_state(self):
        root, quality, nochord = _probs([0] * 4)
        nochord[:] = 0.99
        times = np.arange(4) * 0.1
        regions = viterbi_decode(times, root, quality, nochord, 0.1, transition_penalty=4.0)
        self.assertEqual([r.label for r in regions], ["N"])


if __name__ == "__main__":
    unittest.main()
