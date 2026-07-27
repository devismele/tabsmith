r"""Full-band Slakh integration tests (offline, dependency-free).

Validate the MIDI reader, deterministic chord derivation, streaming tar subset
extraction with a path-traversal guard, bounded/deterministic selection, archive
checksum verification, and the pinned CC-BY-4.0 licence record — all without any
network access or the 104 GB archive.

Run: .\.venv\Scripts\python.exe -m unittest discover -s ml/tests
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from ml.full_band import download_slakh as dl
from ml.full_band.download_slakh import _open_range
from ml.full_band.download_slakh import download as download_archive
from ml.full_band.midi import read_note_events
from ml.full_band.slakh import (
    SLAKH_ZENODO,
    SlakhIntegrityError,
    derive_track_chords,
    iter_tar_tracks,
    safe_member_name,
    verify_archive,
)


def _varlen(n: int) -> bytes:
    out = bytearray([n & 0x7F])
    n >>= 7
    while n:
        out.insert(0, (n & 0x7F) | 0x80)
        n >>= 7
    return bytes(out)


def _minimal_midi(pitches=(60, 64, 67), quarter_ticks=480) -> bytes:
    header = b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big") + (1).to_bytes(2, "big") + quarter_ticks.to_bytes(2, "big")
    body = bytearray()
    for p in pitches:                       # simultaneous note-ons at t=0
        body += _varlen(0) + bytes([0x90, p, 64])
    body += _varlen(quarter_ticks) + bytes([0x80, pitches[0], 0])  # note-offs after 1 quarter
    for p in pitches[1:]:
        body += _varlen(0) + bytes([0x80, p, 0])
    body += _varlen(0) + bytes([0xFF, 0x2F, 0x00])
    track = b"MTrk" + len(body).to_bytes(4, "big") + bytes(body)
    return header + track


def _make_tar_gz(members: dict[str, bytes]) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return gzip.compress(raw.getvalue())


class MidiDerivationTests(unittest.TestCase):
    def test_reads_simultaneous_notes(self):
        events, duration = read_note_events(_minimal_midi())
        self.assertEqual(len(events), 3)
        self.assertAlmostEqual(duration, 0.5, places=3)  # 480 ticks @ 500000us/qn

    def test_c_major_derivation(self):
        regions = derive_track_chords(_minimal_midi((60, 64, 67)))
        self.assertTrue(regions)
        self.assertEqual(regions[0].label, "C:maj")

    def test_a_minor_derivation(self):
        regions = derive_track_chords(_minimal_midi((69, 72, 76)))  # A C E
        self.assertEqual(regions[0].label, "A:min")

    def test_empty_midi_is_no_chord(self):
        header = b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big") + (1).to_bytes(2, "big") + (480).to_bytes(2, "big")
        body = _varlen(0) + bytes([0xFF, 0x2F, 0x00])
        empty = header + b"MTrk" + len(body).to_bytes(4, "big") + bytes(body)
        regions = derive_track_chords(empty, duration_override=2.0)
        self.assertEqual([r.label for r in regions], ["N"])


class TarSubsetTests(unittest.TestCase):
    def _archive(self):
        midi = _minimal_midi()
        top = SLAKH_ZENODO["topLevelDir"]
        return _make_tar_gz({
            f"{top}/train/Track00001/all_src.mid": midi,
            f"{top}/train/Track00001/metadata.yaml": b"instruments: {}\n",
            f"{top}/train/Track00001/mix.flac": b"X" * 4096,      # large -> skipped
            f"{top}/train/Track00002/all_src.mid": midi,
            f"{top}/validation/Track00100/all_src.mid": midi,      # other split
        })

    def test_streams_only_requested_split_and_reads_small_members(self):
        tracks = list(iter_tar_tracks(io.BytesIO(self._archive()), splits=("train",)))
        self.assertEqual([t.track_id for t in tracks], ["Track00001", "Track00002"])
        self.assertIn("all_src.mid", tracks[0].files)
        self.assertNotIn("mix.flac", tracks[0].files)   # large audio skipped
        self.assertIsNotNone(tracks[0].midi_bytes())

    def test_max_tracks_is_bounded_and_deterministic(self):
        one = list(iter_tar_tracks(io.BytesIO(self._archive()), splits=("train",), max_tracks=1))
        self.assertEqual([t.track_id for t in one], ["Track00001"])

    def test_path_traversal_rejected(self):
        top = SLAKH_ZENODO["topLevelDir"]
        for bad in (f"{top}/../evil.mid", "/etc/passwd", f"{top}/train/../../x"):
            with self.assertRaises(SlakhIntegrityError):
                safe_member_name(bad, top_level=top)
        self.assertEqual(safe_member_name(f"{top}/train/Track1/x.mid", top_level=top),
                         f"{top}/train/Track1/x.mid")


class ArchiveVerificationTests(unittest.TestCase):
    def test_checksum_verify_pass_and_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.bin"
            path.write_bytes(b"hello world")
            md5 = hashlib.md5(b"hello world").hexdigest()
            verify_archive(path, expected_size=11, expected_md5=md5)  # no raise
            with self.assertRaises(SlakhIntegrityError):
                verify_archive(path, expected_size=11, expected_md5="0" * 32)
            with self.assertRaises(SlakhIntegrityError):
                verify_archive(path, expected_size=999, expected_md5=md5)


class _FakeResponse:
    """Minimal stand-in for a streamed ``requests`` response."""

    def __init__(self, status_code, body=b"", headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        self.closed = False

    def iter_content(self, chunk_size):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class _FakeSession:
    """Replays a scripted list of responses and records the Range headers seen."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.headers = {}
        self.ranges = []

    def get(self, url, headers=None, stream=False, timeout=None):
        self.ranges.append((headers or {}).get("Range"))
        return self._responses.pop(0)


class ResumeSafetyTests(unittest.TestCase):
    """The resume path must never append to a partial without proof of offset."""

    def test_range_ignored_is_rejected(self):
        session = _FakeSession([_FakeResponse(200, b"whole file again")])
        with self.assertRaises(SlakhIntegrityError) as ctx:
            _open_range(session, "http://example/x", 4096, 30)
        self.assertIn("ignored Range", str(ctx.exception))

    def test_wrong_resume_offset_is_rejected(self):
        session = _FakeSession([
            _FakeResponse(206, b"", {"Content-Range": "bytes 0-99/100000"}),
        ])
        with self.assertRaises(SlakhIntegrityError) as ctx:
            _open_range(session, "http://example/x", 4096, 30)
        self.assertIn("resumed at 0", str(ctx.exception))

    def test_unparsable_content_range_is_rejected(self):
        session = _FakeSession([_FakeResponse(206, b"", {"Content-Range": "garbage"})])
        with self.assertRaises(SlakhIntegrityError):
            _open_range(session, "http://example/x", 4096, 30)

    def test_correct_offset_accepted(self):
        resp = _FakeResponse(206, b"tail", {"Content-Range": "bytes 4096-9999/10000"})
        session = _FakeSession([resp])
        self.assertIs(_open_range(session, "http://example/x", 4096, 30), resp)
        self.assertEqual(session.ranges, ["bytes=4096-"])

    def test_partial_larger_than_official_size_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            downloads = root / "downloads"
            downloads.mkdir()
            partial = downloads / (SLAKH_ZENODO["archiveName"] + ".download")
            partial.write_bytes(b"X" * 50)
            with self.assertRaises(SlakhIntegrityError):
                download_archive(downloads, expected_size=10, expected_md5="0" * 32)


class DownloadOutcomeTests(unittest.TestCase):
    def _run(self, body, expected_md5):
        tmp = tempfile.mkdtemp()
        root = Path(tmp)
        downloads = root / "downloads"
        session = _FakeSession([_FakeResponse(200, body)])
        return root, downloads, session, expected_md5

    def test_verified_download_renames_and_reports_progress(self):
        body = b"slakh-bytes" * 100
        md5 = hashlib.md5(body).hexdigest()
        root, downloads, session, md5 = self._run(body, md5)
        with unittest.mock.patch.object(dl, "requests", create=True):
            path = download_archive(
                downloads, expected_size=len(body), expected_md5=md5,
                quarantine_dir=root / "quarantine", session=session,
            )
        self.assertEqual(path.name, SLAKH_ZENODO["archiveName"])
        self.assertEqual(path.read_bytes(), body)
        self.assertFalse(path.with_suffix(path.suffix + ".download").exists())
        progress = json.loads(
            (downloads / (SLAKH_ZENODO["archiveName"] + ".progress.json")).read_text()
        )
        self.assertEqual(progress["state"], "verified")
        self.assertEqual(progress["bytes_done"], len(body))

    def test_md5_mismatch_quarantines_and_raises(self):
        body = b"corrupted-payload"
        root, downloads, session, _ = self._run(body, "0" * 32)
        with self.assertRaises(SlakhIntegrityError):
            download_archive(
                downloads, expected_size=len(body), expected_md5="0" * 32,
                quarantine_dir=root / "quarantine", session=session,
            )
        # The bad archive never takes the final name and is moved outside downloads.
        self.assertFalse((downloads / SLAKH_ZENODO["archiveName"]).exists())
        quarantined = list((root / "quarantine").glob("*.badmd5"))
        self.assertEqual(len(quarantined), 1)
        self.assertEqual(quarantined[0].read_bytes(), body)
        progress = json.loads(
            (downloads / (SLAKH_ZENODO["archiveName"] + ".progress.json")).read_text()
        )
        self.assertEqual(progress["state"], "failed")


class LicenceTests(unittest.TestCase):
    def test_pinned_licence_is_commercial_compatible(self):
        self.assertEqual(SLAKH_ZENODO["license"], "cc-by-4.0")
        self.assertTrue(SLAKH_ZENODO["commercialUseCompatible"])
        self.assertEqual(len(SLAKH_ZENODO["archiveMd5"]), 32)

    def test_download_url_targets_the_official_record(self):
        self.assertIn(SLAKH_ZENODO["recid"], dl.API_CONTENT_URL)
        self.assertIn(SLAKH_ZENODO["archiveName"], dl.API_CONTENT_URL)
        self.assertTrue(dl.API_CONTENT_URL.startswith("https://zenodo.org/"))


if __name__ == "__main__":
    unittest.main()
