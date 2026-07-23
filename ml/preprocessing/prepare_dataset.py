"""Dataset assembly: python -m ml.preprocessing.prepare_dataset

Generates synthetic audio, imports every configured real annotation source
(Tabsmith references always; Isophonics/Billboard if a local directory is
supplied), builds immutable artist-level splits, and writes a reproducibility
manifest (seed, git commit, dataset versions, split assignment, per-file
checksums, class distribution).

No audio is downloaded. Real recordings are only referenced when the operator
points to a locally licensed directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .features import FEATURE_PIPELINE_VERSION
from .import_billboard import scan_billboard_dir
from .import_isophonics import scan_isophonics_dir
from .import_tabsmith_ref import import_tabsmith_references
from .synth_generator import render_dataset
from ..schema import SCHEMA_VERSION, Track, parse_chord_label
from ..splits.make_splits import build_split_assignment

ML_ROOT = Path(__file__).resolve().parents[1]


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ML_ROOT.parent,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _class_distribution(tracks: list[Track]) -> dict:
    duration: dict[str, float] = {}
    for track in tracks:
        for region in track.chords:
            family = parse_chord_label(region.label).family
            duration[family] = duration.get(family, 0.0) + region.duration()
    total = sum(duration.values()) or 1.0
    return {family: round(seconds / total, 4) for family, seconds in sorted(duration.items())}


def _availability_breakdown(tracks: list[Track]) -> dict:
    counts: dict[str, int] = {}
    for track in tracks:
        counts[track.audio_availability] = counts.get(track.audio_availability, 0) + 1
    return counts


def prepare(out_dir: Path, config: dict, isophonics_dir: str | None, billboard_dir: str | None) -> dict:
    seed = int(config.get("seed", 20260723))
    synth = config.get("synthetic", {})
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets: list[dict] = []
    all_tracks: list[Track] = []

    # 1. Synthetic (always available, runnable).
    synthetic_tracks = render_dataset(
        out_dir / "synthetic",
        count=int(synth.get("trackCount", 6)),
        seed=seed,
        duration=float(synth.get("durationSeconds", 24.0)),
        tempo_range=tuple(synth.get("tempoRange", [72, 132])),
    )
    all_tracks += synthetic_tracks
    datasets.append({"name": "synthetic", "version": "synth-v1", "license": "CC0-synthetic",
                     "trackCount": len(synthetic_tracks),
                     "audioAvailability": _availability_breakdown(synthetic_tracks)})

    # 2. Tabsmith manual references (real repo data, annotations-only unless local audio).
    tabsmith_tracks, tabsmith_summary = import_tabsmith_references()
    ann_out = out_dir / "tabsmith-reference" / "annotations"
    for track in tabsmith_tracks:
        track.save(ann_out / f"{track.track_id}.json")
    all_tracks += tabsmith_tracks
    datasets.append({"name": "tabsmith-reference", "version": "reference-v1",
                     "license": "user-supplied-recording", "trackCount": len(tabsmith_tracks),
                     "audioAvailability": _availability_breakdown(tabsmith_tracks),
                     "importSummary": tabsmith_summary})

    # 3. Optional licensed academic datasets (only if a local directory is supplied).
    for name, directory, scanner in (
        ("isophonics", isophonics_dir, scan_isophonics_dir),
        ("billboard", billboard_dir, scan_billboard_dir),
    ):
        if not directory:
            datasets.append({"name": name, "version": "not-supplied", "trackCount": 0,
                             "note": f"Pass --{name}-dir to import locally licensed {name} annotations."})
            continue
        scanned = scanner(directory)
        ann_out = out_dir / name / "annotations"
        for track in scanned:
            track.save(ann_out / f"{track.track_id}.json")
        all_tracks += scanned
        datasets.append({"name": name, "version": "local", "trackCount": len(scanned),
                         "audioAvailability": _availability_breakdown(scanned)})

    split_assignment = build_split_assignment(all_tracks, seed)

    # Checksums of every emitted annotation file.
    checksums = {p.name: _sha256(p) for p in sorted(out_dir.rglob("*.json"))}

    manifest = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "gitCommit": _git_commit(),
        "schemaVersion": SCHEMA_VERSION,
        "featurePipelineVersion": FEATURE_PIPELINE_VERSION,
        "config": config,
        "datasets": datasets,
        "trackCount": len(all_tracks),
        "audioTrainableTracks": sum(1 for t in all_tracks if t.is_audio_trainable()),
        "classDistribution": _class_distribution(all_tracks),
        "splitAssignment": split_assignment,
        "annotationChecksums": checksums,
        "trainingMetrics": None,
        "validationMetrics": None,
        "modelChecksum": None,
    }
    manifest_path = out_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble the learned-harmony-v1 dataset.")
    parser.add_argument("--out", default=str(ML_ROOT / "datasets"))
    parser.add_argument("--config", default=str(ML_ROOT / "configs" / "default.json"))
    parser.add_argument("--isophonics-dir", default=None)
    parser.add_argument("--billboard-dir", default=None)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    manifest = prepare(Path(args.out), config, args.isophonics_dir, args.billboard_dir)

    print(f"Tracks: {manifest['trackCount']} "
          f"({manifest['audioTrainableTracks']} audio-trainable)")
    print(f"Splits: {manifest['splitAssignment']['counts']}")
    print(f"Class distribution (by duration): {manifest['classDistribution']}")
    if manifest["splitAssignment"]["warnings"]:
        print("Leakage/review warnings:")
        for warning in manifest["splitAssignment"]["warnings"]:
            print(f"  - {warning}")
    print(f"Manifest: {Path(args.out) / 'dataset_manifest.json'}")


if __name__ == "__main__":
    main()
