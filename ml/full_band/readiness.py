"""Generate a portable full-band data/evaluation readiness report."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ..schema import parse_chord_label
from .manifest import (
    REQUIRED_AUDIO_VIEWS,
    FullBandManifestError,
    portable_manifest_checksum,
    validate_manifest,
)
from .splits import build_grouped_splits, validate_grouped_splits
from .symbolic import derive_chord_regions, normalize_chord_regions

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "ml" / "configs" / "full-band-harmony-v1-manifest.example.json"
DEFAULT_PROTOCOL = REPO_ROOT / "ml" / "configs" / "full-band-harmony-v1-evaluation.json"
DEFAULT_JSON_REPORT = REPO_ROOT / "evaluation" / "reports" / "full-band-harmony-v1-readiness.json"
DEFAULT_MARKDOWN_REPORT = REPO_ROOT / "evaluation" / "reports" / "full-band-harmony-v1-readiness.md"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_data_path(root: Path, relative_path: str) -> Path:
    resolved_root = root.resolve()
    candidate = (resolved_root / relative_path).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise FullBandManifestError("manifest path escapes the local data root") from error
    return candidate


def _regions_for_track(track: dict[str, Any]):
    if track["chordRegions"]:
        return normalize_chord_regions(track["chordRegions"], track["durationSeconds"])
    return derive_chord_regions(track["symbolicEvents"], track["durationSeconds"])


def build_readiness_report(
    payload: dict[str, Any],
    *,
    data_root: str | Path | None = None,
    evaluation_protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit a manifest without leaking its optional local data root."""
    manifest = validate_manifest(payload, require_approved=False)
    protocol = evaluation_protocol or json.loads(
        DEFAULT_PROTOCOL.read_text(encoding="utf-8")
    )
    if protocol.get("schemaVersion") != 1:
        raise FullBandManifestError("evaluation protocol schemaVersion must be 1")
    if tuple(protocol.get("inputViews", ())) != REQUIRED_AUDIO_VIEWS:
        raise FullBandManifestError(
            "evaluation protocol views must match the frozen required view order"
        )
    engines = protocol.get("engines")
    if not isinstance(engines, list) or not engines:
        raise FullBandManifestError("evaluation protocol must define engines")
    blockers: list[str] = []
    warnings: list[str] = []

    try:
        validate_manifest(payload, require_approved=True)
        license_gate = "passed"
    except FullBandManifestError as error:
        license_gate = "blocked"
        blockers.append(f"License/provenance gate: {error}")

    tracks = manifest["tracks"]
    if not tracks:
        blockers.append("No legally approved full-band tracks are configured.")

    view_counts = {
        view: sum(view in track["views"] for track in tracks)
        for view in REQUIRED_AUDIO_VIEWS
    }
    missing_view_tracks = {
        view: [track["trackId"] for track in tracks if view not in track["views"]]
        for view in REQUIRED_AUDIO_VIEWS
    }
    for view, missing in missing_view_tracks.items():
        if missing:
            blockers.append(
                f"{view} is missing for {len(missing)} configured track(s)."
            )

    total_duration = sum(track["durationSeconds"] for track in tracks)
    no_chord_duration = 0.0
    labelled_duration = 0.0
    annotation_failures = []
    for track in tracks:
        try:
            regions = _regions_for_track(track)
            labelled_duration += sum(region.duration() for region in regions)
            no_chord_duration += sum(
                region.duration()
                for region in regions
                if parse_chord_label(region.label).is_no_chord
            )
        except (FullBandManifestError, KeyError, TypeError, ValueError) as error:
            annotation_failures.append({"trackId": track["trackId"], "reason": str(error)})
    if annotation_failures:
        blockers.append(
            f"Chord/no-chord derivation failed for {len(annotation_failures)} track(s)."
        )

    split_payload = manifest.get("splitManifest")
    split_status = "blocked"
    if tracks and split_payload:
        try:
            validate_grouped_splits(tracks, split_payload)
            split_counts = Counter(split_payload["assignments"].values())
            missing_splits = [
                split
                for split in ("training", "development", "validation", "test")
                if split_counts[split] == 0
            ]
            if missing_splits:
                blockers.append(
                    f"Grouped split has no tracks in: {', '.join(missing_splits)}."
                )
            else:
                split_status = "passed"
        except FullBandManifestError as error:
            blockers.append(f"Grouped split gate: {error}")
    elif tracks:
        blockers.append("No frozen composition/artist/group-aware split manifest exists.")

    local_files = {
        "checked": data_root is not None,
        "expected": sum(len(track["views"]) for track in tracks),
        "present": 0,
        "checksumMatched": 0,
        "missingPortableIds": [],
        "checksumMismatchPortableIds": [],
    }
    if data_root is not None:
        root = Path(data_root)
        for track in tracks:
            for view_name, view in track["views"].items():
                portable_id = f"{track['trackId']}:{view_name}"
                path = _portable_data_path(root, view["path"])
                if not path.is_file():
                    local_files["missingPortableIds"].append(portable_id)
                    continue
                local_files["present"] += 1
                if view["sha256"] and _file_sha256(path) == view["sha256"]:
                    local_files["checksumMatched"] += 1
                else:
                    local_files["checksumMismatchPortableIds"].append(portable_id)
        if local_files["missingPortableIds"]:
            blockers.append(
                f"{len(local_files['missingPortableIds'])} declared audio view file(s) are missing."
            )
        if local_files["checksumMismatchPortableIds"]:
            blockers.append(
                f"{len(local_files['checksumMismatchPortableIds'])} audio view checksum(s) do not match."
            )
    elif tracks:
        warnings.append(
            "Local audio files and per-view checksums were not inspected; pass --data-root locally."
        )

    baseline_matrix = [
        {
            "baselineId": f"{engine}:{view}",
            "engine": engine,
            "inputView": view,
            "trackCount": view_counts[view],
            "ready": bool(tracks) and view_counts[view] == len(tracks),
        }
        for view in REQUIRED_AUDIO_VIEWS
        for engine in engines
    ]
    separation_comparisons = [
        {
            **comparison,
            "pairedBy": protocol["pairingKeys"],
            "metrics": protocol["harmonyMetrics"] + protocol["downstreamMetrics"],
            "ready": (
                bool(tracks)
                and view_counts[comparison["referenceInputView"]] == len(tracks)
                and view_counts[comparison["candidateInputView"]] == len(tracks)
            ),
        }
        for comparison in protocol["sourceSeparationComparisons"]
    ]

    report = {
        "schemaVersion": 1,
        "status": "ready" if not blockers else "blocked",
        "trainingAuthorized": not blockers,
        "largeTrainingRunAuthorized": False,
        "dataset": {
            "datasetId": manifest["datasetId"],
            "datasetVersion": manifest["datasetVersion"],
            "portableManifestChecksum": portable_manifest_checksum(manifest),
            "licenseId": manifest["license"]["licenseId"],
            "licenseStatus": manifest["license"]["status"],
            "licenseGate": license_gate,
            "audioRedistributionAllowed": manifest["license"][
                "audioRedistributionAllowed"
            ],
            "modelArtifactDistributionAllowed": manifest["license"][
                "modelArtifactDistributionAllowed"
            ],
        },
        "evaluationProtocol": {
            "protocolId": protocol["protocolId"],
            "checksum": hashlib.sha256(
                json.dumps(
                    protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True
                ).encode("utf-8")
            ).hexdigest(),
            "statistics": protocol["statistics"],
            "gates": protocol["gates"],
        },
        "inventory": {
            "trackCount": len(tracks),
            "artistCount": len({track["artistId"] for track in tracks}),
            "compositionCount": len({track["compositionId"] for track in tracks}),
            "recordingCount": len({track["recordingId"] for track in tracks}),
            "genreCounts": dict(sorted(Counter(track["genre"] for track in tracks).items())),
            "durationSeconds": total_duration,
            "labelledDurationSeconds": labelled_duration,
            "noChordDurationSeconds": no_chord_duration,
            "viewTrackCounts": view_counts,
        },
        "splitAudit": {
            "status": split_status,
            "strategy": split_payload.get("strategy") if split_payload else None,
            "seed": split_payload.get("seed") if split_payload else None,
            "assignments": split_payload.get("assignments", {}) if split_payload else {},
        },
        "annotationFailures": annotation_failures,
        "localFileAudit": local_files,
        "baselineMatrix": baseline_matrix,
        "sourceSeparationComparisons": separation_comparisons,
        "methodology": {
            "alternateViewsArePaired": True,
            "alternateViewsCountAsIndependentSongs": False,
            "selectionUnit": "composition/artist/split-group connected component",
            "noChordPolicy": "explicit N fills every unlabelled interval",
            "labelDerivation": (
                "Timed chord regions when present; otherwise deterministic symbolic-pitch-sets-v1."
            ),
            "trainingLoader": "ml.full_band.loader materializes approved views into existing Track schema",
            "guitarSetSupportPreserved": True,
        },
        "blockers": blockers,
        "warnings": warnings,
    }
    serialized = json.dumps(report, sort_keys=True)
    if str(Path.home()).lower() in serialized.lower():
        raise FullBandManifestError("absolute local path leaked into readiness report")
    return report


def freeze_split_manifest(payload: dict[str, Any], *, seed: int) -> dict[str, Any]:
    manifest = validate_manifest(payload, require_approved=True)
    return build_grouped_splits(manifest["tracks"], seed=seed)


def _markdown(report: dict[str, Any]) -> str:
    inventory = report["inventory"]
    lines = [
        "# Full-band harmony v1 readiness",
        "",
        f"Status: **{report['status']}**. Large training is **not authorized**.",
        "",
        "This foundation contains no audio and performs no downloads. A local dataset "
        "may be used only after its license, provenance, checksums, grouped splits, "
        "timed labels, no-chord coverage, and alternate audio views pass the gates.",
        "",
        "## Inventory",
        "",
        "| Tracks | Artists | Compositions | Hours | License gate | Split gate |",
        "|---:|---:|---:|---:|---|---|",
        f"| {inventory['trackCount']} | {inventory['artistCount']} | "
        f"{inventory['compositionCount']} | {inventory['durationSeconds'] / 3600:.2f} | "
        f"{report['dataset']['licenseGate']} | {report['splitAudit']['status']} |",
        "",
        "## Planned paired baselines",
        "",
        "Every approved track must be evaluated through the same annotations and metric "
        "implementation using full mix, harmony stem, guitar stem, and guitar-plus-bass. "
        "Rule v3, v1 ML-only, and v1 hybrid are compared per view. Alternate views are "
        "paired observations, never independent songs.",
        "",
        "## Blockers",
        "",
    ]
    lines.extend(f"- {blocker}" for blocker in report["blockers"])
    if not report["blockers"]:
        lines.append("- None.")
    lines.extend([
        "",
        "## Promotion gate",
        "",
        "No model training, production-weight change, or promotion is authorized by this "
        "report. A frozen artist/composition-disjoint full-band test set and downstream "
        "source-separation comparisons remain mandatory.",
        "",
    ])
    return "\n".join(lines)


def write_reports(
    report: dict[str, Any],
    *,
    json_path: str | Path = DEFAULT_JSON_REPORT,
    markdown_path: str | Path = DEFAULT_MARKDOWN_REPORT,
) -> None:
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_output.write_text(_markdown(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--json-output", default=str(DEFAULT_JSON_REPORT))
    parser.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN_REPORT))
    args = parser.parse_args()
    payload = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    report = build_readiness_report(payload, data_root=args.data_root)
    write_reports(
        report, json_path=args.json_output, markdown_path=args.markdown_output
    )
    print(json.dumps({
        "status": report["status"],
        "trackCount": report["inventory"]["trackCount"],
        "blockerCount": len(report["blockers"]),
        "trainingAuthorized": report["trainingAuthorized"],
        "largeTrainingRunAuthorized": report["largeTrainingRunAuthorized"],
    }, indent=2))


if __name__ == "__main__":
    main()
