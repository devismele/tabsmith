"""Evaluation entry point: python -m ml.evaluation.evaluate

Foundation-pass scope: score the ``chroma-template-v0`` reference predictor on
the synthetic dataset (perfect ground truth) using the shared metrics, and print
the recorded ``harmonic-context-v3`` baseline for context. Once the learned
model lands, its predictions slot into the same ``evaluate_regions`` call so the
three-way comparison (baseline / ML / hybrid) is apples-to-apples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

from .baselines import BASELINE_NAME, chroma_template_predict
from .metrics import evaluate_regions
from ..preprocessing.features import extract_from_wav
from ..preprocessing.import_tabsmith_ref import import_tabsmith_references
from ..schema import Track

ML_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ML_ROOT / "datasets" / "synthetic" / "annotations"
REPORT_DIR = ML_ROOT / "evaluation" / "reports"


def _bpm_from_beats(beats: list[float]) -> float | None:
    if len(beats) < 2:
        return None
    diffs = [b - a for a, b in zip(beats, beats[1:]) if b > a]
    return round(60.0 / median(diffs), 2) if diffs else None


def _mean(values: list[float]) -> float | None:
    vals = [v for v in values if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 4) if vals else None


def evaluate_synthetic(dataset_dir: Path) -> dict:
    annotation_files = sorted(dataset_dir.glob("*.json"))
    if not annotation_files:
        raise FileNotFoundError(
            f"No synthetic annotations in {dataset_dir}. Run "
            f"`python -m ml.preprocessing.prepare_dataset` first."
        )
    per_track = []
    for path in annotation_files:
        track = Track.load(path)
        if not track.is_audio_trainable():
            continue
        features = extract_from_wav(track.audio_path)
        predicted = chroma_template_predict(features)
        metrics = evaluate_regions(track.chords, predicted, bpm=_bpm_from_beats(track.beats))
        per_track.append({"trackId": track.track_id, "metrics": metrics})

    aggregate_keys = ["rootAccuracy", "majorMinorAccuracy", "detailedAccuracy",
                      "medianBoundaryErrorMs", "fragmentationRate"]
    aggregate = {key: _mean([t["metrics"][key] for t in per_track]) for key in aggregate_keys}
    aggregate["boundaryF1_250ms"] = _mean([t["metrics"]["boundaryF1"]["250ms"]["f1"] for t in per_track])
    return {"predictor": BASELINE_NAME, "trackCount": len(per_track),
            "aggregate": aggregate, "perTrack": per_track}


def recorded_baseline() -> dict:
    """The harmonic-context-v3 numbers already recorded in the reference file."""
    _tracks, _summary = import_tabsmith_references()
    reference_path = ML_ROOT.parent / "evaluation" / "chord-reference-songs.json"
    songs = json.loads(reference_path.read_text(encoding="utf-8"))
    recorded = []
    for song in songs:
        for section in song.get("sections", []) or []:
            latest = section.get("latestEvaluation")
            if latest:
                recorded.append({
                    "song": f"{song['artist']} — {song['title']}",
                    "pipeline": latest.get("pipeline"),
                    "rootAccuracy": latest.get("rootAccuracy"),
                    "majorMinorAccuracy": latest.get("majorMinorAccuracy"),
                    "medianBoundaryErrorMs": latest.get("medianBoundaryErrorMs"),
                    "fragmentationRate": latest.get("fragmentationRate"),
                })
    return {"pipeline": "2026-07-harmonic-context-v3-reduced-latency", "sections": recorded}


def _print_report(synthetic: dict, baseline: dict) -> None:
    print("=" * 68)
    print("learned-harmony-v1 - foundation evaluation")
    print("=" * 68)
    print(f"\nSynthetic dataset | predictor={synthetic['predictor']} | "
          f"{synthetic['trackCount']} tracks")
    for key, value in synthetic["aggregate"].items():
        print(f"  {key:24s}: {value}")
    print("\nRecorded production baseline (real audio, different songs - context only):")
    print(f"  pipeline: {baseline['pipeline']}")
    for section in baseline["sections"]:
        print(f"  {section['song'].replace(chr(8212), '-')}: root={section['rootAccuracy']} "
              f"majmin={section['majorMinorAccuracy']} "
              f"medianBoundary={section['medianBoundaryErrorMs']}ms "
              f"frag={section['fragmentationRate']}")
    print("\nNote: the synthetic predictor is a non-learned floor (chroma-template-v0).")
    print("It is NOT comparable head-to-head with the baseline yet — different data")
    print("and no learned model. The learned model plugs into the same metrics.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate chord predictions against ground truth.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--report", default=str(REPORT_DIR / "foundation-eval.json"))
    args = parser.parse_args()

    synthetic = evaluate_synthetic(Path(args.dataset))
    baseline = recorded_baseline()
    _print_report(synthetic, baseline)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"synthetic": synthetic, "recordedBaseline": baseline}, indent=2),
                           encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
