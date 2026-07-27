"""CLI for the Slakh symbolic preparation pass (Phase 11).

Verifies the archive, streams metadata and MIDI only, derives chord labels,
audits composition grouping and split leakage, and writes portable manifests
plus a human-readable report. No audio is extracted here.

    python -m ml.full_band.run_preparation --root "%USERPROFILE%\\Datasets\\Slakh2100"

The dataset root is always operator-supplied and outside the repository. The
markdown report is checked for absolute paths before it is written, so a report
can be committed without leaking a local layout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .prepare import prepare_from_archive
from .slakh import SLAKH_ZENODO

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "evaluation" / "reports"


def _assert_portable(text: str) -> str:
    lowered = text.replace("\\", "/").lower()
    for needle in ("c:/users", "/users/devis", str(REPO_ROOT).replace("\\", "/").lower()):
        if needle and needle in lowered:
            raise ValueError(f"absolute path leaked into report: {needle}")
    return text


def build_markdown(report: dict[str, Any]) -> str:
    labels = report["labelStatistics"]
    instruments = report["instrumentStatistics"]
    grouping = report["compositionGrouping"]
    lines = [
        "# Slakh2100 symbolic preparation",
        "",
        f"Official Zenodo record {report['zenodoRecord']} (`{SLAKH_ZENODO['archiveName']}`), "
        f"licence {report['license']}. Archive verified by exact byte size and full MD5",
        "before this pass ran. Labels are derived deterministically from the aligned MIDI;",
        "no audio was read.",
        "",
        "## Coverage",
        "",
        f"- scanned tracks: {report['scannedTracks']}",
        f"- usable tracks: {report['usableTracks']}",
        f"- failed tracks: {report['failedTracks']}",
        f"- tracks by official split: {json.dumps(report['tracksByOfficialSplit'], sort_keys=True)}",
        f"- total duration: {labels['totalDurationHours']} h",
        "",
        "## Composition grouping and split integrity",
        "",
        f"- distinct compositions (by aligned-MIDI SHA-256): {grouping['compositions']}",
        f"- compositions with more than one rendering: {grouping['duplicateCompositions']} "
        f"({grouping['duplicateTrackCount']} tracks)",
        f"- compositions straddling official splits: {grouping['compositionsStraddlingOfficialSplits']}",
        f"- **leakage free: {grouping['leakageFree']}**",
        "",
        "## Chord labels",
        "",
        f"- chord regions: {labels['chordRegionCount']}",
        f"- mean regions/minute: {labels['meanRegionsPerMinute']}",
        f"- no-chord duration: {labels['noChordSeconds']} s "
        f"({labels['noChordFraction']:.4f} of total)",
        "",
        "| quality | seconds | fraction |",
        "|---|---|---|",
    ]
    for quality, seconds in labels["qualitySeconds"].items():
        lines.append(f"| {quality} | {seconds} | {labels['qualityFraction'].get(quality, 0.0):.4f} |")
    lines += [
        "",
        "## Instrument coverage",
        "",
        f"- guitar present: {instruments['guitarPresentTracks']} tracks "
        f"({instruments['guitarPresentFraction']:.4f}), {instruments['guitarPresentDurationHours']} h",
        f"- guitar absent: {instruments['guitarAbsentTracks']} tracks, "
        f"{instruments['guitarAbsentDurationHours']} h",
        f"- bass present: {instruments['bassPresentTracks']} tracks",
        f"- harmonic instrument present: {instruments['harmonicPresentTracks']} tracks",
        f"- mean rendered stems per track: {instruments['meanStemsPerTrack']}",
        "",
        "| instrument class | tracks |",
        "|---|---|",
    ]
    for name, count in instruments["instrumentClassTrackCounts"].items():
        lines.append(f"| {name} | {count} |")

    pilot = report.get("pilotSubset")
    if pilot:
        lines += [
            "",
            "## Deterministic pilot subset",
            "",
            f"- requested: {pilot['requestedSize']}, selected: {pilot['selectedSize']}",
            f"- duration: {pilot['durationHours']} h",
            f"- guitar-present tracks: {pilot['guitarPresentTracks']}",
            f"- selection rule: {pilot['selectionRule']}",
            f"- subset checksum: `{pilot['checksum']}`",
            "",
        ]
    if report.get("failures"):
        lines += ["## Failures", "", "| track | error |", "|---|---|"]
        for failure in report["failures"]:
            lines.append(f"| {failure['trackId']} | {failure['error']} |")
        lines.append("")
    lines += [
        "## Interpretation limit",
        "",
        "Slakh2100 is rendered from MIDI. Everything derived here is synthetic-domain",
        "evidence. A future untouched, real-recorded, legally licensed full-band",
        "benchmark remains required before any production claim.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Slakh symbolic subset.")
    parser.add_argument("--root", required=True,
                        help="Dataset root outside the repository (contains downloads/).")
    parser.add_argument("--splits", nargs="*", default=["train", "validation", "test"])
    parser.add_argument("--max-tracks-per-split", type=int, default=None)
    parser.add_argument("--pilot-size", type=int, default=120)
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip the MD5 re-check (only when it has just been verified).")
    args = parser.parse_args()

    root = Path(args.root)
    archive = root / "downloads" / SLAKH_ZENODO["archiveName"]
    if not archive.exists():
        parser.error(f"verified archive not found: {archive.name}")

    report = prepare_from_archive(
        archive,
        splits=tuple(args.splits),
        max_tracks_per_split=args.max_tracks_per_split,
        pilot_size=args.pilot_size,
        manifest_dir=root / "manifests",
        verify=not args.skip_verify,
    )

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "full-band-slakh-preparation.md").write_text(
        _assert_portable(build_markdown(report)), encoding="utf-8")
    (report_dir / "full-band-slakh-preparation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    print(f"scanned {report['scannedTracks']} tracks, usable {report['usableTracks']}, "
          f"failed {report['failedTracks']}")
    print(f"compositions {report['compositionGrouping']['compositions']}, "
          f"leakage free: {report['compositionGrouping']['leakageFree']}")
    print(f"total duration {report['labelStatistics']['totalDurationHours']} h, "
          f"no-chord {report['labelStatistics']['noChordFraction']:.4f}")


if __name__ == "__main__":
    main()
