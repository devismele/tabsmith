"""Prepare a bounded, deterministic Slakh2100 pilot subset from the archive.

Runs only after the archive's exact size and full official MD5 have both
verified. Deliberately conservative about what it materialises:

* **Metadata and MIDI first.** One streaming pass reads only ``metadata.yaml``
  and the aligned MIDI (small members) for every track in the requested splits.
  The 100 GB of FLAC is never touched in this pass, so labels, statistics and
  split decisions are all settled before any audio is extracted.
* **Composition grouping by content.** Slakh2100 contains tracks rendered from
  the same underlying composition. Grouping on the SHA-256 of the aligned MIDI
  makes duplicates detectable from content rather than from a name convention,
  so re-rendered duplicates cannot straddle a split boundary.
* **Official splits respected, then re-checked.** Slakh's own train/validation/
  test identities are carried through, but a composition appearing in more than
  one official split is quarantined rather than silently trusted.
* **Bounded audio extraction.** Audio is only ever extracted for an explicitly
  chosen, deterministic subset, with a path-traversal guard on every member.

Nothing here writes inside the repository: the caller supplies a dataset root.
"""
from __future__ import annotations

import hashlib
import json
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Iterator

from .midi import MidiError, read_note_events
from .slakh import SLAKH_ZENODO, SlakhIntegrityError, safe_member_name
from .symbolic import derive_chord_regions

# Instrument classes Slakh uses; these three drive the guitar/harmony views.
# The exact ``inst_class`` vocabulary this dataset uses, confirmed by scanning
# the verified archive. Note "Strings (continued)" -- a distinct class name that
# covers 1389 of 1709 tracks and is easy to miss by guessing.
GUITAR_CLASSES = {"Guitar"}
BASS_CLASSES = {"Bass"}
HARMONIC_CLASSES = {"Guitar", "Bass", "Piano", "Organ", "Strings",
                    "Strings (continued)", "Brass", "Reed", "Pipe",
                    "Synth Lead", "Synth Pad", "Chromatic Percussion"}
PERCUSSIVE_CLASSES = {"Drums", "Percussive"}

METADATA_NAME = "metadata.yaml"
# The aligned, all-source MIDI at the track root. Per-stem MIDI lives under
# ``MIDI/SXX.mid`` and must NOT be used for chord derivation: a single stem is
# one instrument, not the harmony.
MIDI_NAME = "all_src.mid"
# Slakh2100-redux ships its own exclusions under this split. They are scanned so
# they can be counted and audited, but never offered for selection.
OMITTED_SPLIT = "omitted"
SELECTABLE_SPLITS = ("train", "validation", "test")


@dataclass
class TrackRecord:
    """One Slakh track's symbolic content and derived labels."""

    track_id: str
    official_split: str
    midi_sha256: str
    duration_seconds: float
    chord_regions: list[Any] = field(default_factory=list)
    instrument_classes: list[str] = field(default_factory=list)
    stem_count: int = 0
    has_guitar: bool = False
    has_bass: bool = False
    has_harmonic: bool = False
    error: str | None = None

    def label_durations(self) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        for region in self.chord_regions:
            out[region.label] += max(0.0, region.end - region.start)
        return dict(out)

    def as_dict(self) -> dict[str, Any]:
        return {
            "trackId": self.track_id,
            "officialSplit": self.official_split,
            "midiSha256": self.midi_sha256,
            "durationSeconds": round(self.duration_seconds, 3),
            "chordRegionCount": len(self.chord_regions),
            "instrumentClasses": sorted(set(self.instrument_classes)),
            "stemCount": self.stem_count,
            "hasGuitar": self.has_guitar,
            "hasBass": self.has_bass,
            "hasHarmonic": self.has_harmonic,
            "error": self.error,
        }


def _parse_metadata(raw: bytes) -> dict[str, Any]:
    import yaml

    try:
        parsed = yaml.safe_load(raw.decode("utf-8", errors="replace"))
    except Exception as exc:  # malformed metadata must not kill the pass
        return {"_error": f"unparsable metadata: {exc}"}
    return parsed if isinstance(parsed, dict) else {"_error": "metadata is not a mapping"}


def _instrument_classes(metadata: dict[str, Any]) -> tuple[list[str], int]:
    stems = metadata.get("stems")
    if not isinstance(stems, dict):
        return [], 0
    classes: list[str] = []
    for stem in stems.values():
        if not isinstance(stem, dict):
            continue
        # Skip stems the dataset marks as absent from the rendered mix.
        if stem.get("audio_rendered") is False:
            continue
        name = stem.get("inst_class")
        if isinstance(name, str) and name.strip():
            classes.append(name.strip())
    return classes, len(classes)


def collect_symbolic_members(
    fileobj: BinaryIO,
    *,
    splits: tuple[str, ...],
    top_level: str = SLAKH_ZENODO["topLevelDir"],
) -> dict[tuple[str, str], dict[str, bytes]]:
    """Buffer every track's ``metadata.yaml`` and ``all_src.mid`` in one pass.

    **Members in this archive are interleaved, not grouped by track.** Files
    belonging to one track are scattered throughout the tar, so a streaming
    reader must not assume it has seen a whole track when the track id changes:
    doing so pairs a track with whatever fragment happened to be adjacent and
    silently derives chords from a single instrument stem.

    Only the two small root members are read. Per-stem ``MIDI/SXX.mid`` and all
    audio are skipped, which keeps the buffer near 100 MB for the full 2100-track
    archive instead of holding the 100 GB of FLAC.
    """
    collected: dict[tuple[str, str], dict[str, bytes]] = {}
    tar = tarfile.open(fileobj=fileobj, mode="r|gz")
    for member in tar:
        if not member.isfile():
            continue
        name = safe_member_name(member.name, top_level=top_level)
        parts = name.split("/")
        if len(parts) < 4:
            continue
        split, track_id = parts[1], parts[2]
        if split not in splits:
            continue
        relative = "/".join(parts[3:])
        if relative not in (METADATA_NAME, MIDI_NAME):
            continue
        extracted = tar.extractfile(member)
        if extracted is not None:
            collected.setdefault((split, track_id), {})[relative] = extracted.read()
    return collected


def _record_from_members(split: str, track_id: str,
                         files: dict[str, bytes]) -> TrackRecord:
    """Turn one track's buffered members into a labelled record."""
    midi_bytes = files.get(MIDI_NAME)
    if midi_bytes is None:
        return TrackRecord(track_id, split, "", 0.0, error=f"no {MIDI_NAME}")

    digest = hashlib.sha256(midi_bytes).hexdigest()
    metadata = _parse_metadata(files[METADATA_NAME]) if METADATA_NAME in files else {}
    classes, stem_count = _instrument_classes(metadata)

    try:
        events, duration = read_note_events(midi_bytes)
    except Exception as exc:  # noqa: BLE001 - record and continue
        return TrackRecord(track_id, split, digest, 0.0,
                           instrument_classes=classes, stem_count=stem_count,
                           error=f"midi parse failed: {exc}")
    if duration <= 0:
        return TrackRecord(track_id, split, digest, 0.0,
                           instrument_classes=classes, stem_count=stem_count,
                           error="non-positive duration")
    return TrackRecord(
        track_id=track_id,
        official_split=split,
        midi_sha256=digest,
        duration_seconds=duration,
        chord_regions=derive_chord_regions(events, duration),
        instrument_classes=classes,
        stem_count=stem_count,
        has_guitar=any(c in GUITAR_CLASSES for c in classes),
        has_bass=any(c in BASS_CLASSES for c in classes),
        has_harmonic=any(c in HARMONIC_CLASSES for c in classes),
    )


def iter_symbolic_tracks(
    fileobj: BinaryIO,
    *,
    splits: tuple[str, ...] = SELECTABLE_SPLITS,
    max_tracks_per_split: int | None = None,
    top_level: str = SLAKH_ZENODO["topLevelDir"],
) -> Iterator[TrackRecord]:
    """Yield one labelled record per track, in deterministic (split, id) order.

    The whole archive is streamed before anything is yielded, because members
    are interleaved (see :func:`collect_symbolic_members`). A track whose MIDI
    is corrupt yields a record carrying ``error`` rather than aborting the pass,
    so one bad file cannot invalidate a multi-hour scan.
    """
    collected = collect_symbolic_members(fileobj, splits=splits, top_level=top_level)
    per_split: Counter[str] = Counter()
    for split, track_id in sorted(collected):
        if max_tracks_per_split is not None and per_split[split] >= max_tracks_per_split:
            continue
        per_split[split] += 1
        yield _record_from_members(split, track_id, collected[(split, track_id)])


def group_by_composition(records: list[TrackRecord]) -> dict[str, list[TrackRecord]]:
    """Group tracks by aligned-MIDI content hash (the composition identity)."""
    groups: dict[str, list[TrackRecord]] = defaultdict(list)
    for record in records:
        if record.error or not record.midi_sha256:
            continue
        groups[record.midi_sha256].append(record)
    return dict(groups)


def audit_split_leakage(groups: dict[str, list[TrackRecord]]) -> dict[str, Any]:
    """Find compositions whose renderings span more than one official split."""
    straddling = []
    for digest, members in sorted(groups.items()):
        splits = sorted({m.official_split for m in members})
        if len(splits) > 1:
            straddling.append({
                "midiSha256": digest,
                "splits": splits,
                "trackIds": sorted(m.track_id for m in members),
            })
    duplicates = [
        {"midiSha256": digest, "trackIds": sorted(m.track_id for m in members)}
        for digest, members in sorted(groups.items()) if len(members) > 1
    ]
    return {
        "compositions": len(groups),
        "duplicateCompositions": len(duplicates),
        "duplicateTrackCount": sum(len(d["trackIds"]) for d in duplicates),
        "compositionsStraddlingOfficialSplits": len(straddling),
        "straddling": straddling[:50],
        "duplicates": duplicates[:50],
        "leakageFree": not straddling,
    }


def label_statistics(records: list[TrackRecord]) -> dict[str, Any]:
    """Chord-quality distribution, no-chord coverage and duration totals."""
    quality_seconds: dict[str, float] = defaultdict(float)
    root_seconds: dict[str, float] = defaultdict(float)
    total_seconds = 0.0
    no_chord_seconds = 0.0
    region_count = 0
    for record in records:
        if record.error:
            continue
        total_seconds += record.duration_seconds
        for label, seconds in record.label_durations().items():
            if label == "N":
                no_chord_seconds += seconds
                quality_seconds["N"] += seconds
                continue
            root, _, quality = label.partition(":")
            quality_seconds[quality or "unknown"] += seconds
            root_seconds[root] += seconds
        region_count += len(record.chord_regions)
    return {
        "usableTracks": sum(1 for r in records if not r.error),
        "totalDurationSeconds": round(total_seconds, 2),
        "totalDurationHours": round(total_seconds / 3600.0, 3),
        "chordRegionCount": region_count,
        "noChordSeconds": round(no_chord_seconds, 2),
        "noChordFraction": round(no_chord_seconds / total_seconds, 6) if total_seconds else 0.0,
        "qualitySeconds": {k: round(v, 2) for k, v in sorted(quality_seconds.items())},
        "qualityFraction": {
            k: round(v / total_seconds, 6) for k, v in sorted(quality_seconds.items())
        } if total_seconds else {},
        "rootSeconds": {k: round(v, 2) for k, v in sorted(root_seconds.items())},
        "meanRegionsPerMinute": round(
            region_count * 60.0 / total_seconds, 3) if total_seconds else 0.0,
    }


def instrument_statistics(records: list[TrackRecord]) -> dict[str, Any]:
    """Instrument coverage and guitar-present/absent balance."""
    usable = [r for r in records if not r.error]
    class_counts: Counter[str] = Counter()
    for record in usable:
        class_counts.update(set(record.instrument_classes))
    guitar_present = [r for r in usable if r.has_guitar]
    guitar_absent = [r for r in usable if not r.has_guitar]
    return {
        "trackCount": len(usable),
        "instrumentClassTrackCounts": dict(sorted(class_counts.items())),
        "guitarPresentTracks": len(guitar_present),
        "guitarAbsentTracks": len(guitar_absent),
        "guitarPresentFraction": round(len(guitar_present) / len(usable), 6) if usable else 0.0,
        "guitarPresentDurationHours": round(
            sum(r.duration_seconds for r in guitar_present) / 3600.0, 3),
        "guitarAbsentDurationHours": round(
            sum(r.duration_seconds for r in guitar_absent) / 3600.0, 3),
        "bassPresentTracks": sum(1 for r in usable if r.has_bass),
        "harmonicPresentTracks": sum(1 for r in usable if r.has_harmonic),
        "meanStemsPerTrack": round(
            sum(r.stem_count for r in usable) / len(usable), 3) if usable else 0.0,
    }


def deterministic_pilot_subset(
    groups: dict[str, list[TrackRecord]],
    *,
    size: int,
    split: str = "train",
) -> list[TrackRecord]:
    """Pick a bounded, reproducible subset that keeps compositions intact.

    Ordering is by composition hash, which is content-derived and therefore
    stable across machines and archive orderings. One rendering per composition
    is taken so the subset cannot contain two views of the same music.
    """
    chosen: list[TrackRecord] = []
    for digest in sorted(groups):
        members = [m for m in groups[digest] if m.official_split == split]
        if not members:
            continue
        chosen.append(sorted(members, key=lambda m: m.track_id)[0])
        if len(chosen) >= size:
            break
    return chosen


def build_report(records: list[TrackRecord], *, pilot_size: int = 0) -> dict[str, Any]:
    """One portable summary of everything the symbolic pass established.

    Tracks under the dataset's own ``omitted`` split are counted and audited but
    excluded from every statistic and from selection, because Slakh2100-redux
    ships them as its own exclusions.
    """
    omitted = [r for r in records if r.official_split == OMITTED_SPLIT]
    selectable = [r for r in records if r.official_split != OMITTED_SPLIT]

    groups = group_by_composition(selectable)
    leakage = audit_split_leakage(groups)
    failed = [r for r in selectable if r.error]
    by_split: Counter[str] = Counter(r.official_split for r in selectable if not r.error)

    # Does anything the dataset omitted share a composition with a kept track?
    omitted_digests = {r.midi_sha256 for r in omitted if r.midi_sha256 and not r.error}
    collisions = sorted(d for d in omitted_digests if d in groups)

    report = {
        "schemaVersion": 1,
        "datasetId": "slakh2100-flac-redux",
        "zenodoRecord": SLAKH_ZENODO["recid"],
        "license": SLAKH_ZENODO["license"],
        "scannedTracks": len(records),
        "usableTracks": len(selectable) - len(failed),
        "failedTracks": len(failed),
        "failures": [{"trackId": r.track_id, "error": r.error} for r in failed[:50]],
        "tracksByOfficialSplit": dict(sorted(by_split.items())),
        "officiallyOmitted": {
            "trackCount": len(omitted),
            "distinctCompositions": len(omitted_digests),
            "compositionsSharedWithSelectableSplits": len(collisions),
            "note": ("Slakh2100-redux ships these exclusions itself. They are excluded "
                     "from all statistics and from selection; the shared-composition "
                     "count shows whether keeping them would have leaked."),
        },
        "compositionGrouping": leakage,
        "labelStatistics": label_statistics(selectable),
        "instrumentStatistics": instrument_statistics(selectable),
    }
    if pilot_size:
        pilot = deterministic_pilot_subset(groups, size=pilot_size)
        report["pilotSubset"] = {
            "requestedSize": pilot_size,
            "selectedSize": len(pilot),
            "trackIds": [r.track_id for r in pilot],
            "durationHours": round(sum(r.duration_seconds for r in pilot) / 3600.0, 3),
            "guitarPresentTracks": sum(1 for r in pilot if r.has_guitar),
            "selectionRule": "one rendering per composition, ordered by aligned-MIDI SHA-256",
            "checksum": hashlib.sha256(
                "\n".join(sorted(r.track_id for r in pilot)).encode("ascii")
            ).hexdigest(),
        }
    return report


def prepare_from_archive(
    archive_path: Path,
    *,
    splits: tuple[str, ...] = ("train", "validation", "test"),
    max_tracks_per_split: int | None = None,
    pilot_size: int = 0,
    manifest_dir: Path | None = None,
    verify: bool = True,
) -> dict[str, Any]:
    """Verify, stream, derive labels and write the symbolic manifests."""
    archive_path = Path(archive_path)
    if verify:
        from .slakh import verify_archive

        verify_archive(archive_path,
                       expected_size=SLAKH_ZENODO["archiveSizeBytes"],
                       expected_md5=SLAKH_ZENODO["archiveMd5"])

    records: list[TrackRecord] = []
    with archive_path.open("rb") as handle:
        for record in iter_symbolic_tracks(
            handle, splits=splits, max_tracks_per_split=max_tracks_per_split
        ):
            records.append(record)

    report = build_report(records, pilot_size=pilot_size)
    if manifest_dir is not None:
        manifest_dir = Path(manifest_dir)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (manifest_dir / "slakh-symbolic-report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        (manifest_dir / "slakh-tracks.json").write_text(
            json.dumps([r.as_dict() for r in records], indent=2), encoding="utf-8")
    return report


def extract_audio_for(
    archive_path: Path,
    track_ids: set[str],
    destination: Path,
    *,
    members: tuple[str, ...] = ("mix.flac",),
    top_level: str = SLAKH_ZENODO["topLevelDir"],
) -> dict[str, Any]:
    """Extract only the named members of an explicit track list.

    Every member path is traversal-checked before it is written, and only paths
    under ``destination`` are ever created.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    total_bytes = 0
    with Path(archive_path).open("rb") as handle:
        tar = tarfile.open(fileobj=handle, mode="r|gz")
        for member in tar:
            if not member.isfile():
                continue
            name = safe_member_name(member.name, top_level=top_level)
            parts = name.split("/")
            if len(parts) < 4:
                continue
            track_id = parts[2]
            relative = "/".join(parts[3:])
            if track_id not in track_ids or relative not in members:
                continue
            target = (destination / track_id / relative).resolve()
            if not str(target).startswith(str(destination.resolve())):
                raise SlakhIntegrityError(f"member escapes destination: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            payload = extracted.read()
            target.write_bytes(payload)
            written.append(f"{track_id}/{relative}")
            total_bytes += len(payload)
    return {
        "requestedTracks": len(track_ids),
        "writtenMembers": len(written),
        "totalBytes": total_bytes,
        "totalGigabytes": round(total_bytes / (1 << 30), 3),
    }
