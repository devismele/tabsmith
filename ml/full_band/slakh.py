"""Slakh2100 acquisition and chord-derivation integration.

Legal + safe by construction:

* Official Zenodo identity, size and MD5 are pinned constants (queried from the
  official record, never invented). Licence is CC-BY-4.0 (commercial-compatible).
* Downloads are resumable via HTTP Range to a ``.download`` temp and only renamed
  atomically after the full size + MD5 verify. A checksum mismatch aborts.
* The tarball is iterated as a stream (``r|gz``) with a path-traversal guard on
  every member, so a bounded, deterministic pilot subset can be prepared without
  materialising the whole 104 GB archive.
* Chord targets are derived deterministically from the aligned MIDI via the
  existing ``derive_chord_regions`` (maj/min/7/N vocabulary).

Nothing here is run against the full archive in this branch; that requires an
operator-provided data root and a completed download.
"""
from __future__ import annotations

import hashlib
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

from .midi import read_note_events
from .symbolic import derive_chord_regions

# Pinned from the official Zenodo record 4599665 -> 4599666 (queried, not invented).
SLAKH_ZENODO = {
    "conceptRecid": "4599665",
    "recid": "4599666",
    "title": "Slakh2100",
    "license": "cc-by-4.0",
    "commercialUseCompatible": True,
    "archiveName": "slakh2100_flac_redux.tar.gz",
    "archiveSizeBytes": 104322767708,
    "archiveMd5": "f4b71b6c45ac9b506f59788456b3f0c4",
    "downloadUrl": "https://zenodo.org/records/4599666/files/slakh2100_flac_redux.tar.gz",
    "topLevelDir": "slakh2100_flac_redux",
    "splits": ("train", "validation", "test"),
}


class SlakhIntegrityError(RuntimeError):
    pass


def _md5_of(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_archive(path: Path, *, expected_size: int, expected_md5: str) -> None:
    """Raise unless the file matches the official size AND MD5."""
    size = path.stat().st_size
    if size != expected_size:
        raise SlakhIntegrityError(f"size mismatch: {size} != {expected_size}")
    digest = _md5_of(path)
    if digest != expected_md5:
        raise SlakhIntegrityError(f"md5 mismatch: {digest} != {expected_md5}")


def resumable_download(dest_dir: Path, *, url: str | None = None,
                       expected_size: int | None = None, expected_md5: str | None = None,
                       session=None) -> Path:
    """Download the official archive with byte-range resume + checksum verify.

    Resumes an existing ``<name>.download`` from its current size (never appends
    blindly). Verifies size + MD5 before the atomic rename to the final name.
    Requires ``requests``; raises on any checksum mismatch (never continues).
    """
    url = url or SLAKH_ZENODO["downloadUrl"]
    expected_size = expected_size or SLAKH_ZENODO["archiveSizeBytes"]
    expected_md5 = expected_md5 or SLAKH_ZENODO["archiveMd5"]
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / SLAKH_ZENODO["archiveName"]
    partial = final.with_suffix(final.suffix + ".download")
    if final.exists():
        verify_archive(final, expected_size=expected_size, expected_md5=expected_md5)
        return final

    if session is None:
        import requests  # deferred; only needed for a real download
        session = requests.Session()
    have = partial.stat().st_size if partial.exists() else 0
    if have > expected_size:
        raise SlakhIntegrityError("partial larger than expected; refusing to guess")
    headers = {"Range": f"bytes={have}-"} if have else {}
    with session.get(url, headers=headers, stream=True, timeout=60) as resp:
        if have and resp.status_code != 206:
            raise SlakhIntegrityError("server ignored Range; refusing append-on-resume")
        resp.raise_for_status()
        mode = "ab" if have else "wb"
        with partial.open(mode) as fh:
            for chunk in resp.iter_content(1 << 20):
                fh.write(chunk)
    verify_archive(partial, expected_size=expected_size, expected_md5=expected_md5)
    partial.replace(final)
    return final


def safe_member_name(name: str, *, top_level: str) -> str:
    """Return a normalised relative path, or raise on traversal/absolute paths."""
    norm = name.replace("\\", "/")
    if norm.startswith("/") or ".." in norm.split("/") or ":" in norm.split("/")[0]:
        raise SlakhIntegrityError(f"unsafe archive path: {name}")
    if not norm.startswith(top_level + "/") and norm != top_level:
        raise SlakhIntegrityError(f"member outside expected top-level: {name}")
    return norm


@dataclass
class SlakhTrack:
    track_id: str
    split: str
    files: dict[str, bytes]     # relpath (below track dir) -> bytes (only read members)

    def midi_bytes(self) -> bytes | None:
        for key in ("all_src.mid", "MIDI/all_src.mid"):
            if key in self.files:
                return self.files[key]
        for k, v in self.files.items():
            if k.endswith(".mid"):
                return v
        return None


def iter_tar_tracks(fileobj: BinaryIO, *, max_tracks: int | None = None,
                    splits: tuple[str, ...] = ("train",),
                    read_suffixes: tuple[str, ...] = (".mid", ".yaml"),
                    top_level: str = SLAKH_ZENODO["topLevelDir"]) -> Iterator[SlakhTrack]:
    """Stream tracks from the tarball, reading only small members (MIDI/metadata)
    into memory and skipping large audio. Deterministic (archive order), bounded
    by ``max_tracks``. Every member path is traversal-checked."""
    tar = tarfile.open(fileobj=fileobj, mode="r|gz")
    current_id = current_split = None
    files: dict[str, bytes] = {}
    yielded = 0
    for member in tar:
        if not member.isfile():
            continue
        name = safe_member_name(member.name, top_level=top_level)
        parts = name.split("/")
        if len(parts) < 3:
            continue
        split, track_id = parts[1], parts[2]
        if split not in splits:
            continue
        rel = "/".join(parts[3:])
        if track_id != current_id:
            if current_id is not None and files:
                yield SlakhTrack(current_id, current_split, files)
                yielded += 1
                if max_tracks is not None and yielded >= max_tracks:
                    return
            current_id, current_split, files = track_id, split, {}
        if rel.endswith(read_suffixes):
            extracted = tar.extractfile(member)
            if extracted is not None:
                files[rel] = extracted.read()
    if current_id is not None and files and (max_tracks is None or yielded < max_tracks):
        yield SlakhTrack(current_id, current_split, files)


def derive_track_chords(midi_bytes: bytes, *, duration_override: float | None = None):
    """Aligned MIDI bytes -> deterministic chord regions (maj/min/7/N)."""
    events, duration = read_note_events(midi_bytes)
    duration = duration_override or duration
    if duration <= 0:
        return []
    return derive_chord_regions(events, duration)
