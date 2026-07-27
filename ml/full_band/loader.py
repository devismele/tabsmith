"""Adapt approved full-band audio views to the existing Track training schema."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

from ..schema import Track
from .manifest import FullBandManifestError, load_manifest
from .splits import validate_grouped_splits
from .symbolic import derive_chord_regions, normalize_chord_regions


def _resolved_view_path(data_root: Path, relative_path: str) -> Path:
    root = data_root.resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise FullBandManifestError("audio view escapes the configured data root") from error
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_training_tracks(
    manifest: dict[str, Any],
    *,
    data_root: str | Path,
    views: Iterable[str],
    require_files: bool = True,
) -> list[Track]:
    """Create one existing-schema Track per requested alternate audio view."""
    root = Path(data_root)
    result: list[Track] = []
    requested_views = tuple(dict.fromkeys(views))
    if not requested_views:
        raise FullBandManifestError("at least one audio view is required")
    split_manifest = manifest.get("splitManifest")
    if not split_manifest:
        raise FullBandManifestError(
            "full-band training requires a frozen grouped split manifest"
        )
    validate_grouped_splits(manifest["tracks"], split_manifest)
    assignments = split_manifest["assignments"]
    for source_track in manifest["tracks"]:
        assigned_split = assignments[source_track["trackId"]]
        if source_track.get("split") not in (None, assigned_split):
            raise FullBandManifestError(
                f"{source_track['trackId']} split disagrees with frozen split manifest"
            )
        regions = (
            normalize_chord_regions(
                source_track["chordRegions"], source_track["durationSeconds"]
            )
            if source_track["chordRegions"]
            else derive_chord_regions(
                source_track["symbolicEvents"], source_track["durationSeconds"]
            )
        )
        for view_name in requested_views:
            if view_name not in source_track["views"]:
                continue
            audio_path = _resolved_view_path(root, source_track["views"][view_name]["path"])
            if require_files and not audio_path.is_file():
                raise FullBandManifestError(
                    f"{source_track['trackId']}:{view_name} audio file is missing"
                )
            if require_files and _sha256(audio_path) != source_track["views"][view_name]["sha256"]:
                raise FullBandManifestError(
                    f"{source_track['trackId']}:{view_name} audio checksum mismatch"
                )
            track = Track(
                track_id=f"{source_track['trackId']}@{view_name}",
                artist=source_track["artistId"],
                title=source_track["title"],
                duration=source_track["durationSeconds"],
                source=f"full-band:{manifest['datasetId']}",
                audio_availability="audio",
                split=assigned_split,
                license=manifest["license"]["licenseId"],
                source_url=manifest["provenance"]["sourceUrl"],
                audio_path=str(audio_path),
                chords=[
                    type(region)(region.start, region.end, region.label)
                    for region in regions
                ],
                notes=(
                    f"recording={source_track['recordingId']}; "
                    f"composition={source_track['compositionId']}; view={view_name}; "
                    "portable manifest paths resolved only at runtime"
                ),
            )
            track.validate()
            result.append(track)
    if not result:
        raise FullBandManifestError("no requested full-band audio views were materialized")
    return result


def load_full_band_training_tracks(
    manifest_path: str | Path,
    *,
    data_root: str | Path,
    views: Iterable[str],
    require_files: bool = True,
) -> list[Track]:
    manifest = load_manifest(manifest_path, require_approved=True)
    return materialize_training_tracks(
        manifest, data_root=data_root, views=views, require_files=require_files
    )
