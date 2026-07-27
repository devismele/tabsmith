"""Portable, license-gated manifest contract for full-band harmony data."""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

SCHEMA_VERSION = 1
REQUIRED_AUDIO_VIEWS = (
    "full-mix",
    "harmony-stem",
    "guitar-stem",
    "guitar-plus-bass",
)
ALLOWED_SPLITS = {"training", "development", "validation", "test"}
REQUIRED_PERMITTED_USES = {"model-training", "model-evaluation"}
ALLOWED_VIEW_ORIGINS = {
    "dataset-mix",
    "oracle-stem",
    "separator-output",
    "generated-mix",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class FullBandManifestError(ValueError):
    """Raised when a full-band manifest is unsafe or methodologically invalid."""


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FullBandManifestError(f"{field} must be a non-empty string")
    return value.strip()


def _portable_relative_path(value: Any, field: str) -> str:
    raw = _require_string(value, field).replace("\\", "/")
    posix = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise FullBandManifestError(f"{field} must be a portable relative path")
    return posix.as_posix()


def _public_reference(value: Any, field: str) -> str:
    reference = _require_string(value, field)
    if not reference.startswith(("https://", "http://", "doi:", "urn:")):
        raise FullBandManifestError(
            f"{field} must be a public URL, DOI, or portable URN"
        )
    return reference


def _validate_checksum(value: Any, field: str, *, required: bool) -> str | None:
    if value in (None, "") and not required:
        return None
    checksum = _require_string(value, field).lower()
    if not SHA256_PATTERN.fullmatch(checksum):
        raise FullBandManifestError(f"{field} must be a lowercase SHA-256 digest")
    return checksum


def _validate_license(license_record: Any, *, require_approved: bool) -> dict[str, Any]:
    if not isinstance(license_record, dict):
        raise FullBandManifestError("license must be an object")
    status = _require_string(license_record.get("status"), "license.status")
    permitted = license_record.get("permittedUses")
    if not isinstance(permitted, list) or not all(isinstance(item, str) for item in permitted):
        raise FullBandManifestError("license.permittedUses must be a string array")
    result = {
        **license_record,
        "status": status,
        "licenseId": _require_string(license_record.get("licenseId"), "license.licenseId"),
        "licenseUrl": _public_reference(
            license_record.get("licenseUrl"), "license.licenseUrl"
        ),
        "permittedUses": sorted(set(permitted)),
        "attribution": _require_string(
            license_record.get("attribution"), "license.attribution"
        ),
        "audioRedistributionAllowed": bool(
            license_record.get("audioRedistributionAllowed", False)
        ),
        "modelArtifactDistributionAllowed": bool(
            license_record.get("modelArtifactDistributionAllowed", False)
        ),
    }
    if require_approved:
        if status != "verified":
            raise FullBandManifestError("license.status must be verified before data use")
        missing = REQUIRED_PERMITTED_USES - set(result["permittedUses"])
        if missing:
            raise FullBandManifestError(
                f"license does not permit required uses: {sorted(missing)}"
            )
        _require_string(license_record.get("verifiedBy"), "license.verifiedBy")
        _require_string(license_record.get("verifiedAt"), "license.verifiedAt")
    return result


def _validate_provenance(provenance: Any, *, require_approved: bool) -> dict[str, Any]:
    if not isinstance(provenance, dict):
        raise FullBandManifestError("provenance must be an object")
    acquisition = _require_string(
        provenance.get("acquisitionMethod"), "provenance.acquisitionMethod"
    )
    if acquisition not in {"user-supplied", "official-download", "generated-symbolic"}:
        raise FullBandManifestError("unsupported provenance.acquisitionMethod")
    result = {
        **provenance,
        "sourceUrl": _public_reference(
            provenance.get("sourceUrl"), "provenance.sourceUrl"
        ),
        "distributionId": _require_string(
            provenance.get("distributionId"), "provenance.distributionId"
        ),
        "acquisitionMethod": acquisition,
        "archiveChecksums": {},
    }
    for name, checksum in dict(provenance.get("archiveChecksums", {})).items():
        portable_name = _portable_relative_path(
            name, "provenance.archiveChecksums key"
        )
        result["archiveChecksums"][portable_name] = _validate_checksum(
            checksum, f"provenance.archiveChecksums.{portable_name}", required=True
        )
    if require_approved and not result["archiveChecksums"]:
        raise FullBandManifestError(
            "approved datasets require at least one provenance archive checksum"
        )
    return result


def _validate_region(region: Any, field: str, duration: float) -> dict[str, Any]:
    if not isinstance(region, dict):
        raise FullBandManifestError(f"{field} must be an object")
    start = float(region.get("start", -1))
    end = float(region.get("end", -1))
    label = _require_string(region.get("label"), f"{field}.label")
    if start < 0 or end <= start or end > duration + 1e-6:
        raise FullBandManifestError(f"{field} has an invalid time range")
    return {"start": start, "end": min(end, duration), "label": label}


def _validate_symbolic_event(event: Any, field: str, duration: float) -> dict[str, Any]:
    if not isinstance(event, dict):
        raise FullBandManifestError(f"{field} must be an object")
    start = float(event.get("start", -1))
    end = float(event.get("end", -1))
    pitches = event.get("pitches")
    if start < 0 or end <= start or end > duration + 1e-6:
        raise FullBandManifestError(f"{field} has an invalid time range")
    if (
        not isinstance(pitches, list)
        or not pitches
        or any(not isinstance(pitch, int) or pitch < 0 or pitch > 127 for pitch in pitches)
    ):
        raise FullBandManifestError(f"{field}.pitches must contain MIDI pitches 0..127")
    velocity = float(event.get("velocity", 1.0))
    if velocity <= 0:
        raise FullBandManifestError(f"{field}.velocity must be positive")
    return {
        "start": start,
        "end": min(end, duration),
        "pitches": pitches,
        "velocity": velocity,
    }


def _validate_track(track: Any, index: int, *, require_approved: bool) -> dict[str, Any]:
    field = f"tracks[{index}]"
    if not isinstance(track, dict):
        raise FullBandManifestError(f"{field} must be an object")
    duration = float(track.get("durationSeconds", 0))
    if duration <= 0:
        raise FullBandManifestError(f"{field}.durationSeconds must be positive")
    result = {
        **track,
        "trackId": _require_string(track.get("trackId"), f"{field}.trackId"),
        "compositionId": _require_string(
            track.get("compositionId"), f"{field}.compositionId"
        ),
        "recordingId": _require_string(track.get("recordingId"), f"{field}.recordingId"),
        "artistId": _require_string(track.get("artistId"), f"{field}.artistId"),
        "splitGroupId": _require_string(
            track.get("splitGroupId"), f"{field}.splitGroupId"
        ),
        "title": _require_string(track.get("title"), f"{field}.title"),
        "genre": _require_string(track.get("genre"), f"{field}.genre"),
        "durationSeconds": duration,
    }
    split = track.get("split")
    if split is not None and split not in ALLOWED_SPLITS:
        raise FullBandManifestError(f"{field}.split is invalid")
    if split is not None:
        result["split"] = split

    views = track.get("views")
    if not isinstance(views, dict):
        raise FullBandManifestError(f"{field}.views must be an object")
    clean_views: dict[str, Any] = {}
    for view_name, view in sorted(views.items()):
        if not isinstance(view, dict):
            raise FullBandManifestError(f"{field}.views.{view_name} must be an object")
        origin = _require_string(
            view.get("origin"), f"{field}.views.{view_name}.origin"
        )
        if origin not in ALLOWED_VIEW_ORIGINS:
            raise FullBandManifestError(
                f"{field}.views.{view_name}.origin is unsupported"
            )
        clean_views[view_name] = {
            **view,
            "path": _portable_relative_path(
                view.get("path"), f"{field}.views.{view_name}.path"
            ),
            "sha256": _validate_checksum(
                view.get("sha256"),
                f"{field}.views.{view_name}.sha256",
                required=require_approved,
            ),
            "origin": origin,
        }
    result["views"] = clean_views

    chord_regions = [
        _validate_region(region, f"{field}.chordRegions[{region_index}]", duration)
        for region_index, region in enumerate(track.get("chordRegions", []))
    ]
    ordered_regions = sorted(chord_regions, key=lambda region: (region["start"], region["end"]))
    for previous, current in zip(ordered_regions, ordered_regions[1:]):
        if current["start"] < previous["end"] - 1e-6:
            raise FullBandManifestError(f"{field}.chordRegions overlap")
    symbolic_events = [
        _validate_symbolic_event(
            event, f"{field}.symbolicEvents[{event_index}]", duration
        )
        for event_index, event in enumerate(track.get("symbolicEvents", []))
    ]
    if require_approved and not chord_regions and not symbolic_events:
        raise FullBandManifestError(
            f"{field} requires timed chord regions or symbolic note events"
        )
    result["chordRegions"] = chord_regions
    result["symbolicEvents"] = symbolic_events
    return result


def validate_manifest(
    payload: Any,
    *,
    require_approved: bool = True,
    require_all_views: bool = False,
) -> dict[str, Any]:
    """Validate and return a normalized portable manifest.

    ``require_approved=False`` is only for readiness templates.  Training and
    evaluation loaders always use the strict default.
    """
    if not isinstance(payload, dict):
        raise FullBandManifestError("manifest must be an object")
    if payload.get("schemaVersion") != SCHEMA_VERSION:
        raise FullBandManifestError(f"schemaVersion must be {SCHEMA_VERSION}")
    tracks = [
        _validate_track(track, index, require_approved=require_approved)
        for index, track in enumerate(payload.get("tracks", []))
    ]
    track_ids = [track["trackId"] for track in tracks]
    if len(track_ids) != len(set(track_ids)):
        raise FullBandManifestError("trackId values must be unique")
    if require_all_views:
        for track in tracks:
            missing = set(REQUIRED_AUDIO_VIEWS) - set(track["views"])
            if missing:
                raise FullBandManifestError(
                    f"{track['trackId']} is missing required views: {sorted(missing)}"
                )
    result = {
        **deepcopy(payload),
        "schemaVersion": SCHEMA_VERSION,
        "datasetId": _require_string(payload.get("datasetId"), "datasetId"),
        "datasetVersion": _require_string(payload.get("datasetVersion"), "datasetVersion"),
        "license": _validate_license(
            payload.get("license"), require_approved=require_approved
        ),
        "provenance": _validate_provenance(
            payload.get("provenance"), require_approved=require_approved
        ),
        "tracks": tracks,
    }
    return result


def load_manifest(
    path: str | Path,
    *,
    require_approved: bool = True,
    require_all_views: bool = False,
) -> dict[str, Any]:
    return validate_manifest(
        json.loads(Path(path).read_text(encoding="utf-8")),
        require_approved=require_approved,
        require_all_views=require_all_views,
    )


def portable_manifest_checksum(payload: dict[str, Any]) -> str:
    """Hash only the portable manifest; local data roots never enter identity."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
