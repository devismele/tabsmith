"""Foundation-pass tests (stdlib unittest, no pytest/torch needed).

Run: .\.venv\Scripts\python.exe -m unittest discover -s ml/tests
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml.evaluation.baselines import chroma_template_predict
from ml.evaluation.metrics import evaluate_regions
from ml.preprocessing.features import extract_features
from ml.preprocessing.import_tabsmith_ref import import_tabsmith_references
from ml.preprocessing.synth_generator import generate_track, write_wav
from ml.schema import ChordRegion, Track, parse_chord_label, transpose_label
from ml.splits.make_splits import build_split_assignment


class SchemaTests(unittest.TestCase):
    def test_chord_label_parsing(self):
        self.assertEqual(parse_chord_label("A:maj").majmin, "A:maj")
        self.assertEqual(parse_chord_label("F#:7").detailed, "F#:7")
        self.assertEqual(parse_chord_label("Am").family, "min")
        self.assertTrue(parse_chord_label("N").is_no_chord)
        self.assertEqual(parse_chord_label("C#:min/E").bass, parse_chord_label("E").root)

    def test_transpose_applies_capo(self):
        # Displayed Bm with capo 2 sounds as C#m.
        self.assertEqual(parse_chord_label(transpose_label("Bm", 2)).detailed, "C#:min")

    def test_track_roundtrip_and_validation(self):
        track = Track(track_id="t", artist="A", title="T", duration=4.0, source="synthetic",
                      audio_availability="annotations",
                      chords=[ChordRegion(0, 2, "A:maj"), ChordRegion(2, 4, "E:maj")]).validate()
        restored = Track.from_dict(track.to_dict())
        self.assertEqual(restored.chords[1].label, "E:maj")

    def test_audio_availability_requires_path(self):
        with self.assertRaises(ValueError):
            Track(track_id="t", artist="A", title="T", duration=1.0, source="synthetic",
                  audio_availability="audio").validate()


class SyntheticTests(unittest.TestCase):
    def test_generation_is_deterministic(self):
        a, samples_a, _ = generate_track(0, seed=1, duration=8.0, tempo_range=(100, 100))
        b, samples_b, _ = generate_track(0, seed=1, duration=8.0, tempo_range=(100, 100))
        self.assertEqual([c.label for c in a.chords], [c.label for c in b.chords])
        self.assertTrue(np.allclose(samples_a, samples_b))
        self.assertGreater(len(a.chords), 0)

    def test_features_have_expected_shape(self):
        _track, samples, sr = generate_track(0, seed=2, duration=6.0, tempo_range=(120, 120))
        features = extract_features(samples, sr)
        self.assertEqual(features.chroma.shape[1], 12)
        self.assertEqual(features.bass_chroma.shape[1], 12)
        self.assertEqual(len(features.times), features.chroma.shape[0])


class MetricsTests(unittest.TestCase):
    def test_perfect_prediction_scores_one(self):
        ref = [ChordRegion(0, 2, "A:maj"), ChordRegion(2, 4, "E:maj")]
        result = evaluate_regions(ref, ref, bpm=120)
        self.assertEqual(result["rootAccuracy"], 1.0)
        self.assertEqual(result["detailedAccuracy"], 1.0)
        self.assertEqual(result["fragmentationRate"], 0.0)

    def test_wrong_quality_keeps_root_but_loses_detail(self):
        ref = [ChordRegion(0, 4, "A:maj")]
        pred = [ChordRegion(0, 4, "A:min")]
        result = evaluate_regions(ref, pred)
        self.assertEqual(result["rootAccuracy"], 1.0)
        self.assertLess(result["detailedAccuracy"], 1.0)

    def test_template_baseline_runs_end_to_end(self):
        _track, samples, sr = generate_track(0, seed=3, duration=8.0, tempo_range=(110, 110))
        features = extract_features(samples, sr)
        predicted = chroma_template_predict(features)
        self.assertTrue(predicted)
        result = evaluate_regions([ChordRegion(0, 8, "A:maj")], predicted)
        self.assertIn("boundaryF1", result)


class SplitAndImportTests(unittest.TestCase):
    def test_artist_level_split_no_leakage_for_reserved_test(self):
        tracks = [
            Track(track_id="s1", artist="Synthetic", title="x", duration=1, source="synthetic",
                  audio_availability="annotations", chords=[ChordRegion(0, 1, "C:maj")]),
            Track(track_id="q1", artist="Queen", title="a", duration=1, source="tabsmith-reference",
                  split="test", audio_availability="annotations", chords=[ChordRegion(0, 1, "C:maj")]),
            Track(track_id="q2", artist="Queen", title="b", duration=1, source="tabsmith-reference",
                  split="test", audio_availability="annotations", chords=[ChordRegion(0, 1, "C:maj")]),
        ]
        assignment = build_split_assignment(tracks, seed=1)
        self.assertEqual(assignment["trackSplit"]["q1"], "test")
        self.assertEqual(assignment["trackSplit"]["q2"], "test")
        self.assertEqual(assignment["trackSplit"]["s1"], "training")

    def test_tabsmith_reference_import_runs_on_repo_file(self):
        tracks, summary = import_tabsmith_references()
        self.assertGreaterEqual(summary["totalSongs"], 1)
        # Hotel California's verified section should import with sounding chords.
        hotel = [t for t in tracks if "hotel" in t.track_id]
        self.assertTrue(hotel, "expected the Hotel California verified section to import")
        self.assertGreater(len(hotel[0].chords), 0)


class WavRoundTripTests(unittest.TestCase):
    def test_write_and_feature_extract_from_wav(self):
        from ml.preprocessing.features import extract_from_wav
        _track, samples, sr = generate_track(0, seed=4, duration=5.0, tempo_range=(120, 120))
        with tempfile.TemporaryDirectory() as tmp:
            path = write_wav(Path(tmp) / "a.wav", samples, sr)
            features = extract_from_wav(path)
            self.assertGreater(len(features.times), 0)


if __name__ == "__main__":
    unittest.main()
