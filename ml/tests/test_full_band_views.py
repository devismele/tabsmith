r"""Full-band source-view construction tests (offline, synthetic audio).

Check stem classification, per-view stem selection, mixing behaviour, and the
guitar-absent / percussion-only controls that make the Phase 12 comparison
meaningful.

Run: .\.venv\Scripts\python.exe -m unittest discover -s ml/tests -t .
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml.full_band.views import (
    VIEW_NAMES,
    Stem,
    ViewError,
    build_view,
    mix_stems,
    select_stems,
    stems_from_metadata,
    view_availability,
)


def _metadata(classes: dict[str, str], unrendered: set[str] = frozenset()) -> dict:
    return {
        "stems": {
            stem_id: {"inst_class": name,
                      "audio_rendered": stem_id not in unrendered}
            for stem_id, name in classes.items()
        }
    }


CLASSES = {"S00": "Guitar", "S01": "Bass", "S02": "Drums",
           "S03": "Piano", "S04": "Guitar"}


def _write_stems(root: Path, classes=CLASSES, seconds=1.0, rate=22050):
    stem_dir = root / "stems"
    stem_dir.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    for i, stem_id in enumerate(classes):
        tone = 0.1 * np.sin(
            2 * np.pi * (110 * (i + 1)) * np.arange(int(rate * seconds)) / rate)
        sf.write(str(stem_dir / f"{stem_id}.flac"), tone.astype(np.float32), rate)
    mix = 0.2 * np.sin(2 * np.pi * 220 * np.arange(int(rate * seconds)) / rate)
    sf.write(str(root / "mix.flac"), mix.astype(np.float32), rate)
    return rate


class StemClassificationTests(unittest.TestCase):
    def test_metadata_maps_to_existing_files_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_stems(root)
            stems = stems_from_metadata(_metadata(CLASSES), root / "stems")
            self.assertEqual([s.stem_id for s in stems],
                             ["S00", "S01", "S02", "S03", "S04"])

    def test_unrendered_stems_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_stems(root)
            stems = stems_from_metadata(
                _metadata(CLASSES, unrendered={"S02"}), root / "stems")
            self.assertNotIn("S02", [s.stem_id for s in stems])

    def test_missing_files_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "stems").mkdir()
            self.assertEqual(stems_from_metadata(_metadata(CLASSES), root / "stems"), [])

    def test_class_predicates(self):
        self.assertTrue(Stem("a", "Guitar", Path(".")).is_guitar)
        self.assertTrue(Stem("a", "Bass", Path(".")).is_bass)
        self.assertTrue(Stem("a", "Drums", Path(".")).is_percussive)
        self.assertTrue(Stem("a", "Piano", Path(".")).is_harmonic)
        self.assertFalse(Stem("a", "Drums", Path(".")).is_harmonic)


class ViewSelectionTests(unittest.TestCase):
    def setUp(self):
        self.stems = [Stem(k, v, Path(".")) for k, v in CLASSES.items()]

    def test_oracle_guitar_selects_only_guitars(self):
        picked = select_stems(self.stems, "oracle-guitar")
        self.assertEqual([s.stem_id for s in picked], ["S00", "S04"])

    def test_guitar_plus_bass(self):
        picked = select_stems(self.stems, "guitar-plus-bass")
        self.assertEqual([s.stem_id for s in picked], ["S00", "S01", "S04"])

    def test_oracle_harmonic_excludes_drums(self):
        picked = select_stems(self.stems, "oracle-harmonic")
        self.assertNotIn("S02", [s.stem_id for s in picked])
        self.assertEqual(len(picked), 4)

    def test_guitar_absent_harmonic_removes_every_guitar(self):
        picked = select_stems(self.stems, "guitar-absent-harmonic")
        ids = [s.stem_id for s in picked]
        self.assertNotIn("S00", ids)
        self.assertNotIn("S04", ids)
        self.assertEqual(ids, ["S01", "S03"])

    def test_percussion_only_selects_drums(self):
        picked = select_stems(self.stems, "percussion-only")
        self.assertEqual([s.stem_id for s in picked], ["S02"])

    def test_full_mix_is_not_summed_from_stems(self):
        with self.assertRaises(ViewError):
            select_stems(self.stems, "full-mix")

    def test_unknown_view_rejected(self):
        with self.assertRaises(ViewError):
            select_stems(self.stems, "vocals-only")


class MixingTests(unittest.TestCase):
    def test_mix_sums_and_matches_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rate = _write_stems(root)
            stems = stems_from_metadata(_metadata(CLASSES), root / "stems")
            mixed, out_rate = mix_stems(select_stems(stems, "oracle-guitar"))
            self.assertEqual(out_rate, rate)
            self.assertEqual(len(mixed), rate)

    def test_empty_selection_raises(self):
        with self.assertRaises(ViewError):
            mix_stems([])

    def test_mix_is_peak_limited(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_stems(root)
            stems = stems_from_metadata(_metadata(CLASSES), root / "stems")
            mixed, _ = mix_stems(stems)
            self.assertLessEqual(float(np.max(np.abs(mixed))), 1.0 + 1e-6)


class BuildViewTests(unittest.TestCase):
    def test_every_named_view_builds_or_reports_absence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rate = _write_stems(root)
            metadata = _metadata(CLASSES)
            for view in VIEW_NAMES:
                result = build_view(root, metadata, view)
                self.assertIsNotNone(result, view)
                audio, out_rate = result
                self.assertEqual(out_rate, rate, view)
                self.assertGreater(len(audio), 0, view)

    def test_guitar_view_is_none_on_a_guitar_absent_track(self):
        """An absent view is a legitimate outcome, not an error."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            classes = {"S00": "Piano", "S01": "Drums"}
            _write_stems(root, classes)
            self.assertIsNone(build_view(root, _metadata(classes), "oracle-guitar"))

    def test_missing_mix_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "stems").mkdir()
            with self.assertRaises(ViewError):
                build_view(root, _metadata({}), "full-mix")

    def test_availability_counts_without_reading_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_stems(root)
            counts = view_availability(_metadata(CLASSES), root)
            self.assertEqual(counts["full-mix"], 1)
            self.assertEqual(counts["oracle-guitar"], 2)
            self.assertEqual(counts["guitar-plus-bass"], 3)
            self.assertEqual(counts["guitar-absent-harmonic"], 2)
            self.assertEqual(counts["percussion-only"], 1)


if __name__ == "__main__":
    unittest.main()
