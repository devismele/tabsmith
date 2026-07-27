r"""Slakh preparation tests (offline, no archive, no network).

Build small synthetic tar.gz archives with the Slakh layout and check the
streaming symbolic pass, composition grouping by MIDI content, split-leakage
auditing, label/instrument statistics, deterministic pilot selection and the
traversal-guarded bounded audio extraction.

Run: .\.venv\Scripts\python.exe -m unittest discover -s ml/tests -t .
"""
from __future__ import annotations

import gzip
import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from ml.full_band.prepare import (
    TrackRecord,
    audit_split_leakage,
    build_report,
    deterministic_pilot_subset,
    extract_audio_for,
    group_by_composition,
    instrument_statistics,
    iter_symbolic_tracks,
    label_statistics,
)
from ml.full_band.slakh import SLAKH_ZENODO, SlakhIntegrityError

TOP = SLAKH_ZENODO["topLevelDir"]


def _varlen(n: int) -> bytes:
    out = bytearray([n & 0x7F])
    n >>= 7
    while n:
        out.insert(0, (n & 0x7F) | 0x80)
        n >>= 7
    return bytes(out)


def _midi(pitches=(60, 64, 67), quarter_ticks=480, quarters=4) -> bytes:
    header = (b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big")
              + (1).to_bytes(2, "big") + quarter_ticks.to_bytes(2, "big"))
    body = bytearray()
    for p in pitches:
        body += _varlen(0) + bytes([0x90, p, 64])
    body += _varlen(quarter_ticks * quarters) + bytes([0x80, pitches[0], 0])
    for p in pitches[1:]:
        body += _varlen(0) + bytes([0x80, p, 0])
    body += _varlen(0) + bytes([0xFF, 0x2F, 0x00])
    track = b"MTrk" + len(body).to_bytes(4, "big") + bytes(body)
    return header + track


def _metadata(classes=("Guitar", "Bass", "Drums")) -> bytes:
    lines = ["stems:"]
    for i, name in enumerate(classes):
        lines += [f"  S{i:02d}:", f"    inst_class: {name}", "    audio_rendered: true"]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _archive(members: dict[str, bytes]) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return gzip.compress(raw.getvalue())


def _standard_archive() -> bytes:
    guitar_midi = _midi((60, 64, 67))
    other_midi = _midi((62, 65, 69))
    return _archive({
        f"{TOP}/train/Track00001/all_src.mid": guitar_midi,
        f"{TOP}/train/Track00001/metadata.yaml": _metadata(("Guitar", "Bass", "Drums")),
        f"{TOP}/train/Track00001/mix.flac": b"A" * 512,
        f"{TOP}/train/Track00002/all_src.mid": other_midi,
        f"{TOP}/train/Track00002/metadata.yaml": _metadata(("Piano", "Drums")),
        f"{TOP}/train/Track00002/mix.flac": b"B" * 512,
        f"{TOP}/validation/Track01500/all_src.mid": _midi((57, 60, 64)),
        f"{TOP}/validation/Track01500/metadata.yaml": _metadata(("Guitar", "Strings")),
        f"{TOP}/test/Track01900/all_src.mid": _midi((55, 59, 62)),
        f"{TOP}/test/Track01900/metadata.yaml": _metadata(("Organ",)),
    })


class SymbolicPassTests(unittest.TestCase):
    def test_reads_all_splits_and_derives_chords(self):
        records = list(iter_symbolic_tracks(io.BytesIO(_standard_archive())))
        self.assertEqual([r.track_id for r in records],
                         ["Track00001", "Track00002", "Track01500", "Track01900"])
        self.assertEqual(records[0].official_split, "train")
        self.assertEqual(records[2].official_split, "validation")
        self.assertTrue(all(r.error is None for r in records))
        self.assertEqual(records[0].chord_regions[0].label, "C:maj")

    def test_split_filter_is_respected(self):
        records = list(iter_symbolic_tracks(io.BytesIO(_standard_archive()), splits=("test",)))
        self.assertEqual([r.track_id for r in records], ["Track01900"])

    def test_instrument_classes_and_guitar_flags(self):
        records = list(iter_symbolic_tracks(io.BytesIO(_standard_archive()), splits=("train",)))
        first, second = records
        self.assertEqual(first.instrument_classes, ["Guitar", "Bass", "Drums"])
        self.assertTrue(first.has_guitar)
        self.assertTrue(first.has_bass)
        self.assertFalse(second.has_guitar)
        self.assertTrue(second.has_harmonic)  # Piano is harmonic

    def test_unrendered_stems_are_ignored(self):
        metadata = b"stems:\n  S00:\n    inst_class: Guitar\n    audio_rendered: false\n"
        archive = _archive({
            f"{TOP}/train/Track00001/all_src.mid": _midi(),
            f"{TOP}/train/Track00001/metadata.yaml": metadata,
        })
        record = next(iter_symbolic_tracks(io.BytesIO(archive), splits=("train",)))
        self.assertFalse(record.has_guitar)
        self.assertEqual(record.stem_count, 0)

    def test_corrupt_midi_is_recorded_not_raised(self):
        """One bad track must not invalidate a multi-hour pass."""
        archive = _archive({
            f"{TOP}/train/Track00001/all_src.mid": b"NOT-A-MIDI-FILE",
            f"{TOP}/train/Track00001/metadata.yaml": _metadata(),
            f"{TOP}/train/Track00002/all_src.mid": _midi(),
            f"{TOP}/train/Track00002/metadata.yaml": _metadata(),
        })
        records = list(iter_symbolic_tracks(io.BytesIO(archive), splits=("train",)))
        self.assertEqual(len(records), 2)
        self.assertIsNotNone(records[0].error)
        self.assertIsNone(records[1].error)

    def test_missing_midi_is_recorded(self):
        archive = _archive({f"{TOP}/train/Track00001/metadata.yaml": _metadata()})
        records = list(iter_symbolic_tracks(io.BytesIO(archive), splits=("train",)))
        self.assertEqual(records[0].error, "no MIDI member")

    def test_bounded_scan_stops_early(self):
        archive = _archive({
            f"{TOP}/train/Track{i:05d}/all_src.mid": _midi((60 + i, 64 + i, 67 + i))
            for i in range(1, 6)
        })
        records = list(iter_symbolic_tracks(
            io.BytesIO(archive), splits=("train",), max_tracks_per_split=2))
        self.assertEqual(len(records), 2)

    def test_audio_members_are_never_read_in_the_symbolic_pass(self):
        """The 100 GB of FLAC must not be buffered while deriving labels."""
        huge = b"Z" * (4 << 20)
        archive = _archive({
            f"{TOP}/train/Track00001/all_src.mid": _midi(),
            f"{TOP}/train/Track00001/metadata.yaml": _metadata(),
            f"{TOP}/train/Track00001/mix.flac": huge,
            f"{TOP}/train/Track00001/stems/S00.flac": huge,
        })
        record = next(iter_symbolic_tracks(io.BytesIO(archive), splits=("train",)))
        self.assertIsNone(record.error)
        self.assertEqual(record.chord_regions[0].label, "C:maj")


class CompositionGroupingTests(unittest.TestCase):
    def _records(self):
        return [
            TrackRecord("Track00001", "train", "aaa", 10.0),
            TrackRecord("Track00002", "train", "aaa", 10.0),   # duplicate composition
            TrackRecord("Track00003", "train", "bbb", 10.0),
            TrackRecord("Track01500", "validation", "ccc", 10.0),
        ]

    def test_duplicates_group_by_midi_content(self):
        groups = group_by_composition(self._records())
        self.assertEqual(len(groups), 3)
        self.assertEqual(sorted(r.track_id for r in groups["aaa"]),
                         ["Track00001", "Track00002"])

    def test_failed_tracks_are_excluded_from_grouping(self):
        records = self._records() + [TrackRecord("Bad", "train", "", 0.0, error="boom")]
        self.assertEqual(len(group_by_composition(records)), 3)

    def test_clean_splits_report_no_leakage(self):
        audit = audit_split_leakage(group_by_composition(self._records()))
        self.assertTrue(audit["leakageFree"])
        self.assertEqual(audit["duplicateCompositions"], 1)
        self.assertEqual(audit["duplicateTrackCount"], 2)

    def test_composition_straddling_official_splits_is_flagged(self):
        records = self._records()
        records.append(TrackRecord("Track01900", "test", "aaa", 10.0))
        audit = audit_split_leakage(group_by_composition(records))
        self.assertFalse(audit["leakageFree"])
        self.assertEqual(audit["compositionsStraddlingOfficialSplits"], 1)
        self.assertEqual(audit["straddling"][0]["splits"], ["test", "train"])


class PilotSubsetTests(unittest.TestCase):
    def _groups(self):
        records = [TrackRecord(f"Track{i:05d}", "train", hashlib.sha256(
            str(i).encode()).hexdigest(), 60.0) for i in range(20)]
        return group_by_composition(records)

    def test_subset_is_bounded_and_deterministic(self):
        groups = self._groups()
        first = [r.track_id for r in deterministic_pilot_subset(groups, size=5)]
        second = [r.track_id for r in deterministic_pilot_subset(groups, size=5)]
        self.assertEqual(len(first), 5)
        self.assertEqual(first, second)

    def test_subset_takes_one_rendering_per_composition(self):
        records = [
            TrackRecord("Track00001", "train", "aaa", 60.0),
            TrackRecord("Track00002", "train", "aaa", 60.0),
            TrackRecord("Track00003", "train", "bbb", 60.0),
        ]
        chosen = deterministic_pilot_subset(group_by_composition(records), size=10)
        self.assertEqual(len(chosen), 2)
        self.assertEqual({r.midi_sha256 for r in chosen}, {"aaa", "bbb"})

    def test_subset_only_draws_from_the_requested_split(self):
        records = [
            TrackRecord("Track00001", "train", "aaa", 60.0),
            TrackRecord("Track01900", "test", "bbb", 60.0),
        ]
        chosen = deterministic_pilot_subset(group_by_composition(records), size=10)
        self.assertEqual([r.track_id for r in chosen], ["Track00001"])


class StatisticsTests(unittest.TestCase):
    def test_label_statistics_cover_no_chord_and_qualities(self):
        records = list(iter_symbolic_tracks(io.BytesIO(_standard_archive())))
        stats = label_statistics(records)
        self.assertEqual(stats["usableTracks"], 4)
        self.assertGreater(stats["totalDurationSeconds"], 0)
        self.assertIn("maj", stats["qualitySeconds"])
        self.assertGreaterEqual(stats["noChordFraction"], 0.0)
        self.assertLessEqual(stats["noChordFraction"], 1.0)
        self.assertAlmostEqual(sum(stats["qualityFraction"].values()), 1.0, places=4)

    def test_instrument_statistics_balance(self):
        records = list(iter_symbolic_tracks(io.BytesIO(_standard_archive())))
        stats = instrument_statistics(records)
        self.assertEqual(stats["trackCount"], 4)
        self.assertEqual(stats["guitarPresentTracks"] + stats["guitarAbsentTracks"], 4)
        self.assertIn("Guitar", stats["instrumentClassTrackCounts"])

    def test_report_is_json_serialisable_and_complete(self):
        import json

        records = list(iter_symbolic_tracks(io.BytesIO(_standard_archive())))
        report = build_report(records, pilot_size=2)
        json.dumps(report)  # must not raise
        self.assertEqual(report["usableTracks"], 4)
        self.assertEqual(report["license"], "cc-by-4.0")
        self.assertEqual(report["pilotSubset"]["selectedSize"], 2)
        self.assertEqual(len(report["pilotSubset"]["checksum"]), 64)
        self.assertIn("tracksByOfficialSplit", report)


class BoundedExtractionTests(unittest.TestCase):
    def test_only_requested_tracks_and_members_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "a.tar.gz"
            archive.write_bytes(_standard_archive())
            destination = Path(tmp) / "out"
            summary = extract_audio_for(archive, {"Track00001"}, destination)
            self.assertEqual(summary["writtenMembers"], 1)
            self.assertTrue((destination / "Track00001" / "mix.flac").exists())
            self.assertFalse((destination / "Track00002").exists())

    def test_traversal_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "evil.tar.gz"
            archive.write_bytes(_archive({f"{TOP}/../../escape/mix.flac": b"x"}))
            with self.assertRaises(SlakhIntegrityError):
                extract_audio_for(archive, {"escape"}, Path(tmp) / "out")

    def test_nothing_is_written_for_an_empty_track_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "a.tar.gz"
            archive.write_bytes(_standard_archive())
            destination = Path(tmp) / "out"
            summary = extract_audio_for(archive, set(), destination)
            self.assertEqual(summary["writtenMembers"], 0)
            self.assertEqual(list(destination.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
