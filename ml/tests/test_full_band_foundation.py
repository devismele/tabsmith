from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ml.full_band.loader import materialize_training_tracks
from ml.full_band.freeze_splits import freeze_splits
from ml.full_band.manifest import (
    REQUIRED_AUDIO_VIEWS,
    FullBandManifestError,
    portable_manifest_checksum,
    validate_manifest,
)
from ml.full_band.readiness import build_readiness_report
from ml.full_band.splits import build_grouped_splits, validate_grouped_splits
from ml.full_band.symbolic import derive_chord_regions, normalize_chord_regions


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def approved_payload(track_count: int = 8, *, audio_bytes: bytes = b"audio") -> dict:
    checksum = _sha256(audio_bytes)
    splits = ("training", "development", "validation", "test")
    tracks = []
    for index in range(track_count):
        track_id = f"licensed-track-{index:02d}"
        tracks.append({
            "trackId": track_id,
            "compositionId": f"composition-{index:02d}",
            "recordingId": f"recording-{index:02d}",
            "artistId": f"artist-{index:02d}",
            "splitGroupId": f"group-{index:02d}",
            "title": f"Track {index}",
            "genre": "test-genre",
            "durationSeconds": 4.0,
            "split": splits[index % len(splits)],
            "views": {
                view: {
                    "path": f"audio/{track_id}/{view}.wav",
                    "sha256": checksum,
                    "origin": "dataset-mix" if view == "full-mix" else "oracle-stem",
                }
                for view in REQUIRED_AUDIO_VIEWS
            },
            "chordRegions": [
                {"start": 0.0, "end": 2.0, "label": "C:maj"},
                {"start": 2.0, "end": 4.0, "label": "N"},
            ],
            "symbolicEvents": [],
        })
    return {
        "schemaVersion": 1,
        "datasetId": "licensed-full-band-fixture",
        "datasetVersion": "1.0",
        "license": {
            "status": "verified",
            "licenseId": "fixture-license",
            "licenseUrl": "https://example.invalid/license",
            "permittedUses": ["model-training", "model-evaluation"],
            "attribution": "Fixture attribution",
            "verifiedBy": "fixture-reviewer",
            "verifiedAt": "2026-07-27",
            "audioRedistributionAllowed": False,
            "modelArtifactDistributionAllowed": False,
        },
        "provenance": {
            "sourceUrl": "https://example.invalid/dataset",
            "distributionId": "fixture-distribution",
            "acquisitionMethod": "official-download",
            "archiveChecksums": {"fixture.zip": "a" * 64},
        },
        "tracks": tracks,
        "splitManifest": {
            "schemaVersion": 1,
            "strategy": "fixture-frozen-groups",
            "seed": 20260727,
            "assignments": {
                track["trackId"]: track["split"] for track in tracks
            },
        },
    }


def write_audio_views(root: Path, payload: dict, audio_bytes: bytes = b"audio") -> None:
    for track in payload["tracks"]:
        for view in track["views"].values():
            path = root / view["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(audio_bytes)


class FullBandManifestTests(unittest.TestCase):
    def test_manifest_is_portable_and_checksum_is_deterministic(self):
        manifest = validate_manifest(approved_payload())
        self.assertEqual(
            portable_manifest_checksum(manifest),
            portable_manifest_checksum(json.loads(json.dumps(manifest))),
        )
        self.assertFalse(any(Path(view["path"]).is_absolute()
                             for track in manifest["tracks"]
                             for view in track["views"].values()))

    def test_absolute_and_traversal_paths_are_rejected(self):
        for unsafe in ("C:/Users/person/song.wav", "../song.wav", "/tmp/song.wav"):
            with self.subTest(path=unsafe):
                payload = approved_payload()
                payload["tracks"][0]["views"]["full-mix"]["path"] = unsafe
                with self.assertRaises(FullBandManifestError):
                    validate_manifest(payload)

    def test_unverified_or_ambiguous_license_cannot_enter_loader(self):
        payload = approved_payload()
        payload["license"]["status"] = "unverified"
        with self.assertRaisesRegex(FullBandManifestError, "verified"):
            validate_manifest(payload)

    def test_local_provenance_path_cannot_enter_portable_manifest(self):
        payload = approved_payload()
        payload["provenance"]["sourceUrl"] = "C:/private/dataset"
        with self.assertRaisesRegex(FullBandManifestError, "public URL"):
            validate_manifest(payload)


class FullBandSplitTests(unittest.TestCase):
    def test_grouped_split_is_deterministic_and_keeps_compositions_together(self):
        manifest = validate_manifest(approved_payload(12))
        for track in manifest["tracks"]:
            track.pop("split", None)
        manifest["tracks"][1]["compositionId"] = manifest["tracks"][0]["compositionId"]
        first = build_grouped_splits(manifest["tracks"], seed=91)
        second = build_grouped_splits(manifest["tracks"], seed=91)
        self.assertEqual(first, second)
        self.assertEqual(
            first["assignments"][manifest["tracks"][0]["trackId"]],
            first["assignments"][manifest["tracks"][1]["trackId"]],
        )
        validate_grouped_splits(manifest["tracks"], first)

    def test_split_validator_rejects_artist_leakage(self):
        manifest = validate_manifest(approved_payload())
        for track in manifest["tracks"]:
            track.pop("split", None)
        manifest["tracks"][1]["artistId"] = manifest["tracks"][0]["artistId"]
        assignments = {
            track["trackId"]: ("training" if index != 1 else "test")
            for index, track in enumerate(manifest["tracks"])
        }
        with self.assertRaisesRegex(FullBandManifestError, "artistId"):
            validate_grouped_splits(
                manifest["tracks"],
                {"assignments": assignments},
            )

    def test_frozen_split_contains_only_portable_identities(self):
        payload = approved_payload(12)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            output_path = root / "split.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            split = freeze_splits(manifest_path, output_path, seed=44)
            serialized = output_path.read_text(encoding="utf-8")
        self.assertTrue(split["portableIdsOnly"])
        self.assertNotIn(str(root.resolve()), serialized)
        self.assertEqual(set(split["assignments"]), {
            track["trackId"] for track in payload["tracks"]
        })


class FullBandSymbolicTests(unittest.TestCase):
    def test_symbolic_derivation_is_contiguous_and_adds_no_chord(self):
        regions = derive_chord_regions([
            {"start": 0.0, "end": 1.0, "pitches": [60, 64, 67]},
            {"start": 2.0, "end": 3.0, "pitches": [62, 65, 69]},
        ], 4.0)
        self.assertEqual(
            [(region.start, region.end, region.label) for region in regions],
            [
                (0.0, 1.0, "C:maj"),
                (1.0, 2.0, "N"),
                (2.0, 3.0, "D:min"),
                (3.0, 4.0, "N"),
            ],
        )

    def test_empty_symbolic_example_is_full_duration_no_chord(self):
        regions = derive_chord_regions([], 5.0)
        self.assertEqual([(region.start, region.end, region.label) for region in regions],
                         [(0.0, 5.0, "N")])

    def test_timed_regions_fill_gaps_and_reject_overlap(self):
        regions = normalize_chord_regions(
            [{"start": 1.0, "end": 2.0, "label": "A:min"}], 3.0
        )
        self.assertEqual([region.label for region in regions], ["N", "A:min", "N"])
        with self.assertRaisesRegex(FullBandManifestError, "overlap"):
            normalize_chord_regions([
                {"start": 0.0, "end": 2.0, "label": "C:maj"},
                {"start": 1.0, "end": 3.0, "label": "G:maj"},
            ], 3.0)


class FullBandLoaderAndReadinessTests(unittest.TestCase):
    def test_loader_materializes_all_views_and_verifies_checksums(self):
        payload = approved_payload(1)
        manifest = validate_manifest(payload)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_audio_views(root, payload)
            tracks = materialize_training_tracks(
                manifest, data_root=root, views=REQUIRED_AUDIO_VIEWS
            )
            self.assertEqual(len(tracks), 4)
            self.assertEqual(
                {track.track_id.split("@", 1)[1] for track in tracks},
                set(REQUIRED_AUDIO_VIEWS),
            )
            self.assertTrue(all(track.source.startswith("full-band:") for track in tracks))

            corrupted = root / payload["tracks"][0]["views"]["full-mix"]["path"]
            corrupted.write_bytes(b"corrupt")
            with self.assertRaisesRegex(FullBandManifestError, "checksum mismatch"):
                materialize_training_tracks(
                    manifest, data_root=root, views=("full-mix",)
                )

    def test_loader_requires_frozen_grouped_split(self):
        payload = approved_payload(1)
        payload.pop("splitManifest")
        manifest = validate_manifest(payload)
        with self.assertRaisesRegex(FullBandManifestError, "frozen grouped split"):
            materialize_training_tracks(
                manifest,
                data_root="unused",
                views=("full-mix",),
                require_files=False,
            )

    def test_ready_report_has_paired_baselines_without_local_paths(self):
        payload = approved_payload()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_audio_views(root, payload)
            report = build_readiness_report(payload, data_root=root)
        self.assertEqual(report["status"], "ready")
        self.assertTrue(report["trainingAuthorized"])
        self.assertFalse(report["largeTrainingRunAuthorized"])
        self.assertEqual(len(report["baselineMatrix"]), 12)
        self.assertTrue(all(item["ready"] for item in report["baselineMatrix"]))
        self.assertEqual(len(report["sourceSeparationComparisons"]), 3)
        self.assertNotIn(str(root.resolve()), json.dumps(report))

    def test_empty_template_is_an_honest_blocker_not_a_training_authorization(self):
        template_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "full-band-harmony-v1-manifest.example.json"
        )
        payload = json.loads(template_path.read_text(encoding="utf-8"))
        report = build_readiness_report(payload)
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["trainingAuthorized"])
        self.assertFalse(report["largeTrainingRunAuthorized"])
        self.assertEqual(report["inventory"]["trackCount"], 0)
        self.assertTrue(report["blockers"])


if __name__ == "__main__":
    unittest.main()
