r"""Billboard feature-path tests (stdlib unittest, torch-free).

A synthetic ``bothchroma.csv`` fixture encodes the documented layout (col 0 =
timestamp, cols 1..24 = bass 12 then treble 12, bin 0 == A). These lock the
parsing/rolling contract; the BASS_FIRST / origin constants still need a real-file
check via ``calibrate_against_annotations`` on acquisition.

Run: .\.venv\Scripts\python.exe -m unittest discover -s ml/tests
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ml.preprocessing.billboard_features import (
    FEATURE_PIPELINE_VERSION,
    calibrate_against_annotations,
    load_billboard_bothchroma,
)
from ml.preprocessing.feature_source import frames_for_track
from ml.schema import ChordRegion, Track


def _row(time: float, bass_pc_raw: int, treble_pc_raw: int) -> str:
    """One CSV row: timestamp + 12 bass + 12 treble, spike at the given raw bins."""
    bass = [0.01] * 12
    treble = [0.01] * 12
    bass[bass_pc_raw] = 1.0
    treble[treble_pc_raw] = 1.0
    return ",".join(str(x) for x in [time, *bass, *treble])


class BillboardFeatureTests(unittest.TestCase):
    def _write_csv(self, tmp: str) -> Path:
        # Raw bin 3 is A-based index 3 -> C-based pitch class (3+9)%12 == 0 (C).
        # Raw bin 10 -> pitch class (10+9)%12 == 7 (G).
        rows = [
            _row(0.0, 3, 3),     # C
            _row(0.1, 3, 3),     # C
            _row(0.2, 10, 10),   # G
        ]
        path = Path(tmp) / "bothchroma.csv"
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return path

    def test_shapes_versions_and_hop(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = load_billboard_bothchroma(self._write_csv(tmp))
        self.assertEqual(frames.pipeline_version, FEATURE_PIPELINE_VERSION)
        self.assertEqual(frames.chroma.shape, (3, 12))
        self.assertEqual(frames.bass_chroma.shape, (3, 12))
        self.assertEqual(frames.stacked().shape, (3, 25))
        self.assertAlmostEqual(frames.hop_seconds, 0.1, places=3)
        self.assertFalse(getattr(frames, "energy_available"))

    def test_pitch_class_rolled_to_c_based(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = load_billboard_bothchroma(self._write_csv(tmp))
        # Frame 0/1 should peak on C (0), frame 2 on G (7), in both bands.
        self.assertEqual(int(frames.bass_chroma[0].argmax()), 0)
        self.assertEqual(int(frames.chroma[0].argmax()), 0)
        self.assertEqual(int(frames.bass_chroma[2].argmax()), 7)
        # L1-normalized.
        self.assertAlmostEqual(float(frames.chroma[0].sum()), 1.0, places=5)

    def test_calibration_matches_aligned_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = load_billboard_bothchroma(self._write_csv(tmp))
        regions = [ChordRegion(0.0, 0.2, "C:maj"), ChordRegion(0.2, 0.3, "G:maj")]
        report = calibrate_against_annotations(frames, regions)
        self.assertEqual(report["frames_scored"], 3)
        self.assertEqual(report["root_match_fraction"], 1.0)

    def test_feature_source_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._write_csv(tmp)
            track = Track(
                track_id="billboard-0001", artist="Billboard", title="0001",
                duration=0.3, source="billboard", audio_availability="features",
                license="ddmal-cc0-features", feature_path=str(csv_path),
                chords=[ChordRegion(0.0, 0.3, "C:maj")],
            ).validate()
            self.assertTrue(track.is_feature_trainable())
            self.assertTrue(track.is_trainable())
            frames = frames_for_track(track)
        self.assertEqual(frames.chroma.shape, (3, 12))

    def test_features_availability_requires_feature_path(self):
        with self.assertRaises(ValueError):
            Track(
                track_id="x", artist="a", title="t", duration=1.0, source="billboard",
                audio_availability="features",  # no feature_path -> must fail
                chords=[ChordRegion(0.0, 1.0, "C:maj")],
            ).validate()


if __name__ == "__main__":
    unittest.main()
