r"""GuitarSet importer tests (stdlib unittest, torch-free).

Uses a small synthetic JAMS fixture that encodes the format assumptions the
importer relies on (two ``chord`` annotations with the lead-sheet first, a
``beat_position`` annotation, GuitarSet track-id layout). Verify against a real
GuitarSet file once acquired — these lock the parsing contract, not the data.

Run: .\.venv\Scripts\python.exe -m unittest discover -s ml/tests
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ml.preprocessing.import_guitarset import (
    _parse_track_id,
    import_guitarset_track,
    scan_guitarset_dir,
)


def _fixture_jams() -> dict:
    return {
        "annotations": [
            {
                "namespace": "chord",  # lead-sheet (preferred) — comes first
                "data": [
                    {"time": 0.0, "duration": 1.0, "value": "C:maj", "confidence": 1.0},
                    {"time": 1.0, "duration": 1.0, "value": "A:min", "confidence": 1.0},
                    {"time": 2.0, "duration": 1.0, "value": "D:7", "confidence": 1.0},
                ],
            },
            {
                "namespace": "chord",  # inferred — should be ignored by default
                "data": [{"time": 0.0, "duration": 3.0, "value": "C:maj", "confidence": 0.5}],
            },
            {
                "namespace": "beat_position",
                "data": [
                    {"time": 0.0, "duration": 0.0, "value": {"position": 1}},
                    {"time": 0.5, "duration": 0.0, "value": {"position": 2}},
                    {"time": 1.0, "duration": 0.0, "value": {"position": 1}},
                ],
            },
        ]
    }


class GuitarSetImporterTests(unittest.TestCase):
    def test_track_id_parsing(self):
        player, genre, mode = _parse_track_id("05_Jazz3-150-C_solo")
        self.assertEqual(player, "05")
        self.assertEqual(genre, "Jazz")
        self.assertEqual(mode, "solo")
        # Bossa Nova style code + comping mode.
        self.assertEqual(_parse_track_id("00_BN1-129-Eb_comp")[1], "Bossa Nova")

    def test_import_prefers_leadsheet_chords(self):
        with tempfile.TemporaryDirectory() as tmp:
            jams_path = Path(tmp) / "03_Rock1-120-C_comp.jams"
            jams_path.write_text(json.dumps(_fixture_jams()), encoding="utf-8")
            track = import_guitarset_track(jams_path)

        self.assertEqual(track.source, "guitarset")
        self.assertEqual(track.artist, "guitarset-p03")
        self.assertEqual(track.license, "CC-BY-4.0")
        # Lead-sheet annotation has 3 chords; inferred (1 chord) must be ignored.
        self.assertEqual([c.label for c in track.chords], ["C:maj", "A:min", "D:7"])
        self.assertEqual(track.duration, 3.0)
        self.assertEqual(track.downbeats, [0.0, 1.0])
        # No audio dir supplied -> annotations-only, never falsely audio-trainable.
        self.assertEqual(track.audio_availability, "annotations")
        self.assertFalse(track.is_audio_trainable())

    def test_audio_pairing_marks_audio_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "annotation").mkdir()
            (root / "audio_mono-mic").mkdir()
            track_id = "01_Funk2-100-A_solo"
            (root / "annotation" / f"{track_id}.jams").write_text(
                json.dumps(_fixture_jams()), encoding="utf-8")
            (root / "audio_mono-mic" / f"{track_id}_mic.wav").write_bytes(b"RIFF....WAVE")

            tracks = scan_guitarset_dir(root)

        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].audio_availability, "audio")
        self.assertTrue(tracks[0].is_audio_trainable())
        self.assertIn("genre=Funk", tracks[0].notes)


if __name__ == "__main__":
    unittest.main()
