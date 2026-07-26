"""Offline experimental comparison: python -m ml.evaluation.compare_experimental

Scores rule-based / ml-only / hybrid predictors on a HELD-OUT synthetic test
split (its own seed namespace) and prints clearly-labelled synthetic-only
metrics. Requires the training extra + a trained checkpoint.

Explicitly does NOT evaluate on Hotel California — that section is development
material and far too small to call validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

from .adapters import predict_hybrid, predict_ml, predict_rule
from .metrics import evaluate_regions
from ..preprocessing.features import extract_features
from ..preprocessing.synth_generator import build_split_tracks
from ..training.checkpoint import load_checkpoint

ML_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ML_ROOT / "checkpoints" / "temporal-baseline-v0.pt"
REPORT = ML_ROOT / "evaluation" / "reports" / "experimental-comparison.json"

_KEYS = [
    "rootAccuracy", "majorMinorAccuracy", "detailedAccuracy",
    "fragmentationRate", "noChordRecall", "predictedRegions",
]


def _bpm(beats):
    diffs = [b - a for a, b in zip(beats, beats[1:]) if b > a]
    return round(60.0 / median(diffs), 2) if diffs else None


def _mean(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 4) if vals else None


def _label_at(regions, time):
    for region in regions:
        if region.start <= time < region.end:
            return region.label
    return "N"


def _override_count(rule_regions, hybrid_regions):
    """Count hybrid regions whose midpoint differs from the rule proxy."""
    return sum(
        1 for region in hybrid_regions
        if _label_at(rule_regions, (region.start + region.end) / 2) != region.label
    )


def compare(config: dict, checkpoint_path: Path) -> dict:
    model, metadata = load_checkpoint(checkpoint_path)
    data = config["data"]
    tracks = build_split_tracks("synthetic_test", int(data["syntheticTestTracks"]),
                                float(data["durationSeconds"]), tuple(data.get("tempoRange", [72, 140])))
    predictors = {
        "rule-based (chroma-template-v0)": lambda f: predict_rule(f),
        "ml-only (temporal-baseline-v0)": lambda f: predict_ml(model, f),
        "hybrid-experimental": lambda f: predict_hybrid(model, f),
    }
    results = {name: [] for name in predictors}
    for track, audio, sr in tracks:
        features = extract_features(audio, sr)
        predictions = {name: predict(features) for name, predict in predictors.items()}
        for name, predict in predictors.items():
            metrics = evaluate_regions(track.chords, predictions[name], bpm=_bpm(track.beats))
            if name == "hybrid-experimental":
                metrics["mlOverridesVsRule"] = _override_count(
                    predictions["rule-based (chroma-template-v0)"],
                    predictions[name],
                )
            results[name].append(metrics)

    aggregate = {}
    for name, metric_list in results.items():
        agg = {key: _mean([m[key] for m in metric_list]) for key in _KEYS}
        for tol in ("100ms", "250ms", "500ms"):
            agg[f"boundaryF1_{tol}"] = _mean([m["boundaryF1"][tol]["f1"] for m in metric_list])
        agg["meanBoundaryErrorMs"] = _mean([m["meanBoundaryErrorMs"] for m in metric_list])
        agg["medianBoundaryErrorMs"] = _mean([m["medianBoundaryErrorMs"] for m in metric_list])
        agg["meanSignedBoundaryErrorMs"] = _mean([m["meanSignedBoundaryErrorMs"] for m in metric_list])
        agg["extraChords"] = _mean([m["extraChords"] for m in metric_list])
        agg["missingChords"] = _mean([m["missingChords"] for m in metric_list])
        if name == "hybrid-experimental":
            agg["mlOverridesVsRule"] = _mean(
                [m["mlOverridesVsRule"] for m in metric_list]
            )
        aggregate[name] = agg
    return {"trackCount": len(tracks), "modelChecksum": metadata.get("checksum"), "aggregate": aggregate}


def _print(report: dict) -> None:
    print("#" * 72)
    print("# SYNTHETIC PIPELINE-VALIDATION METRICS ONLY")
    print("# Not representative of commercial music. No accuracy claim is made.")
    print("#" * 72)
    print(f"\nHeld-out synthetic test tracks: {report['trackCount']}\n")
    for name, agg in report["aggregate"].items():
        print(f"[{name}]")
        for key, value in agg.items():
            print(f"    {key:26s}: {value}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare rule / ML / hybrid on synthetic held-out data.")
    parser.add_argument("--config", default=str(ML_ROOT / "configs" / "temporal-baseline.json"))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--report", default=str(REPORT))
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    report = compare(config, Path(args.checkpoint))
    _print(report)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
