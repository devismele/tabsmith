"""Audit and report the completed temporal-harmony-v2 grouped ablation.

The module never loads p00.  Its optional checkpoint re-evaluation reproduces
the frozen p01-p05 metrics, measures inference runtime, and compares the
probability filters whose defaults existed before the ablation.  Smoothing
diagnostics are explicitly ineligible for model selection.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .temporal_v2_ablation import (
    CAPTURES,
    FORBIDDEN_SELECTION_PERFORMER,
    _aggregate_metrics,
    _capture,
    _evaluate_model,
    _extract_feature_map,
    _load_tracks,
    _portable_checksum,
    candidate_config,
    evaluate_selection_gates,
)
from .probability_smoothing import smooth_learned_probabilities

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "ml" / "configs"
DEFAULT_RUN_DIR = REPO_ROOT / "ml" / "runs" / "temporal-harmony-v2"
DEFAULT_REPORT_DIR = REPO_ROOT / "evaluation" / "reports"
CONFIG_PATHS = (
    CONFIG_ROOT / "temporal-harmony-v2.json",
    CONFIG_ROOT / "temporal-harmony-v2-splits.json",
    CONFIG_ROOT / "temporal-harmony-v2-ablations.json",
    CONFIG_ROOT / "temporal-harmony-v2-augmentations.json",
)
DEFAULT_SMOOTHING = CONFIG_ROOT / "temporal-harmony-v2-smoothing-diagnostics.json"
BOOTSTRAP_SEED = 20260726
BOOTSTRAP_ITERATIONS = 10_000


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def frozen_configuration_checksum(paths: tuple[Path, ...] = CONFIG_PATHS) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_completed_matrix(
    report: dict[str, Any],
    base: dict[str, Any],
    splits: dict[str, Any],
    ablations: dict[str, Any],
    augmentations: dict[str, Any],
) -> dict[str, Any]:
    if report.get("status") != "completed" or report.get("validForSelection") is not True:
        raise ValueError("ablation report is not a completed frozen selection matrix")
    if (
        report.get("expectedRunCount"),
        report.get("completedRunCount"),
        report.get("failedRunCount"),
    ) != (30, 30, 0):
        raise ValueError("ablation matrix must contain 30 completed and zero failed runs")
    if report.get("p00UsedForSelection") is not False:
        raise ValueError("p00 was used during model selection")

    folds = {fold["foldId"]: fold for fold in splits["folds"]}
    candidates = {candidate["id"]: candidate for candidate in ablations["candidates"]}
    expected = {(fold_id, candidate_id) for fold_id in folds for candidate_id in candidates}
    rows = report["foldResults"]
    keys = [(row["foldId"], row["candidateId"]) for row in rows]
    if len(rows) != len(expected) or len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("candidate/fold coverage is duplicated or incomplete")

    augmentation_checksum = _portable_checksum(augmentations)
    for row in rows:
        fold = folds[row["foldId"]]
        candidate = candidates[row["candidateId"]]
        config = candidate_config(base, candidate)
        config["splitIdentity"] = {
            "splitId": splits["splitId"],
            "foldId": fold["foldId"],
            "trainPerformers": fold["trainPerformers"],
            "validationPerformers": fold["validationPerformers"],
        }
        config["augmentationIdentity"] = (
            {
                "augmentationId": augmentations["augmentationId"],
                "manifestChecksum": augmentation_checksum,
            }
            if candidate.get("useAugmentation", False)
            else None
        )
        if row["trainingConfigChecksum"] != _portable_checksum(config):
            raise ValueError(f"{row['foldId']}/{row['candidateId']}: config checksum mismatch")
        if not row["parity"]["pythonOnnx"]["passed"]:
            raise ValueError(f"{row['foldId']}/{row['candidateId']}: ONNX parity failed")
        if not row["parity"]["typescript"]["passed"]:
            raise ValueError(f"{row['foldId']}/{row['candidateId']}: TypeScript parity failed")
        track_rows = row["metrics"]["perTrack"]
        identities = [(item["trackId"], item["capture"]) for item in track_rows]
        if len(track_rows) != 120 or len(identities) != len(set(identities)):
            raise ValueError(f"{row['foldId']}/{row['candidateId']}: invalid track coverage")
        if any(
            FORBIDDEN_SELECTION_PERFORMER in item["trackId"].lower()
            or item["trackId"].startswith("00_")
            for item in track_rows
        ):
            raise ValueError("p00 track entered the completed selection report")

    return {
        "status": "passed",
        "runCount": len(rows),
        "foldCounts": dict(sorted(Counter(fold for fold, _ in keys).items())),
        "candidateCounts": dict(sorted(Counter(candidate for _, candidate in keys).items())),
        "p00UsedForSelection": False,
        "allPythonOnnxParityPassed": True,
        "allTypeScriptParityPassed": True,
    }


def _candidate_rows(report: dict[str, Any], candidate_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold_result in report["foldResults"]:
        if fold_result["candidateId"] != candidate_id:
            continue
        for row in fold_result["metrics"]["perTrack"]:
            rows.append({**row, "foldId": fold_result["foldId"]})
    return rows


def _paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["foldId"], row["trackId"])].append(row)
    paired = []
    for (fold_id, track_id), captures in sorted(grouped.items()):
        if {row["capture"] for row in captures} != set(CAPTURES):
            raise ValueError(f"{fold_id}/{track_id}: missing alternate capture")

        def average(name: str) -> float:
            values = [float(row[name]) for row in captures]
            return sum(values) / len(values)

        def optional_average(name: str) -> float | None:
            values = [
                float(row[name])
                for row in captures
                if row[name] is not None
            ]
            return sum(values) / len(values) if values else None

        boundary_f1 = {
            tolerance: {
                key: sum(float(row["boundaryF1"][tolerance][key]) for row in captures) / 2
                for key in ("precision", "recall", "f1")
            }
            for tolerance in ("100ms", "250ms", "500ms", "1000ms")
        }
        paired.append({
            "foldId": fold_id,
            "trackId": track_id,
            "capture": "paired-capture-mean",
            "evaluatedDurationSeconds": average("evaluatedDurationSeconds"),
            "rootAccuracy": average("rootAccuracy"),
            "majorMinorAccuracy": average("majorMinorAccuracy"),
            "detailedAccuracy": average("detailedAccuracy"),
            "chordSymbolRecall": average("chordSymbolRecall"),
            "noChordPrecision": average("noChordPrecision"),
            "noChordRecall": average("noChordRecall"),
            "meanBoundaryErrorMs": optional_average("meanBoundaryErrorMs"),
            "medianBoundaryErrorMs": optional_average("medianBoundaryErrorMs"),
            "meanSignedBoundaryErrorMs": optional_average("meanSignedBoundaryErrorMs"),
            "fragmentationRate": average("fragmentationRate"),
            "referenceRegions": average("referenceRegions"),
            "predictedRegions": average("predictedRegions"),
            "missingChords": average("missingChords"),
            "extraChords": average("extraChords"),
            "oneWindowRegionCount": average("oneWindowRegionCount"),
            "shortRegionCount": average("shortRegionCount"),
            "flickerCount": average("flickerCount"),
            "meanTopChordConfidence": average("meanTopChordConfidence"),
            "meanLearnedEntropy": average("meanLearnedEntropy"),
            "meanFrameSwitchProbability": average("meanFrameSwitchProbability"),
            "meanBoundaryProbability": average("meanBoundaryProbability"),
            "boundaryF1": boundary_f1,
        })
    return paired


def _extended_aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    metrics = _aggregate_metrics(rows)
    reference_boundaries = sum(max(0, row["referenceRegions"] - 1) for row in rows)
    predicted_boundaries = sum(max(0, row["predictedRegions"] - 1) for row in rows)
    reference_regions = sum(row["referenceRegions"] for row in rows)
    predicted_regions = sum(row["predictedRegions"] for row in rows)

    def reference_weighted(name: str) -> float:
        return sum(
            float(row[name] or 0.0) * max(0, row["referenceRegions"] - 1)
            for row in rows
        ) / max(1, reference_boundaries)

    def boundary_stat(tolerance: str, name: str, *, predicted: bool = False) -> float:
        weight_name = "predictedRegions" if predicted else "referenceRegions"
        denominator = predicted_boundaries if predicted else reference_boundaries
        return sum(
            float(row["boundaryF1"][tolerance][name])
            * max(0, row[weight_name] - 1)
            for row in rows
        ) / max(1, denominator)

    no_chord_precision = metrics["noChordPrecision"]
    no_chord_recall = metrics["noChordRecall"]
    switch_precision = boundary_stat("250ms", "precision", predicted=True)
    switch_recall = boundary_stat("250ms", "recall")
    metrics.update({
        "weightedChordSymbolRecall": metrics["detailedAccuracy"],
        "noChordF1": (
            2 * no_chord_precision * no_chord_recall
            / (no_chord_precision + no_chord_recall)
            if no_chord_precision + no_chord_recall
            else 0.0
        ),
        "extraRegionCount": float(sum(row["extraChords"] for row in rows)),
        "missingRegionCount": float(sum(row["missingChords"] for row in rows)),
        "averageChordRegionDurationSeconds": (
            metrics["evaluatedDurationSeconds"] / max(1, predicted_regions)
        ),
        "meanSignedBoundaryErrorMs": reference_weighted("meanSignedBoundaryErrorMs"),
        **{
            f"boundariesWithin{tolerance}": boundary_stat(tolerance, "recall")
            for tolerance in ("100ms", "250ms", "500ms", "1000ms")
        },
        "chordSwitchPrecision": switch_precision,
        "chordSwitchRecall": switch_recall,
        "chordSwitchF1": (
            2 * switch_precision * switch_recall / (switch_precision + switch_recall)
            if switch_precision + switch_recall
            else 0.0
        ),
        "referenceRegionCount": float(reference_regions),
    })
    return metrics


def aggregate_candidates(report: dict[str, Any]) -> dict[str, Any]:
    aggregates = {}
    for candidate in report["candidates"]:
        candidate_id = candidate["id"]
        rows = _candidate_rows(report, candidate_id)
        by_capture = {
            capture: _extended_aggregate([row for row in rows if row["capture"] == capture])
            for capture in CAPTURES
        }
        aggregates[candidate_id] = {
            "byCapture": by_capture,
            "pairedPerformances": _extended_aggregate(_paired_rows(rows)),
        }
    return aggregates


def _track_metric(row: dict[str, Any], metric: str) -> float:
    if metric == "regionsPerMinute":
        return float(row["predictedRegions"]) * 60 / max(float(row["evaluatedDurationSeconds"]), 1e-9)
    if metric == "meanAbsoluteBoundaryErrorMs":
        return float(row["meanBoundaryErrorMs"] or 0.0)
    return float(row[metric])


def paired_bootstrap(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    metric: str,
    *,
    seed: int = BOOTSTRAP_SEED,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    baseline = {(row["foldId"], row["trackId"]): row for row in _paired_rows(baseline_rows)}
    candidate = {(row["foldId"], row["trackId"]): row for row in _paired_rows(candidate_rows)}
    if baseline.keys() != candidate.keys():
        raise ValueError("paired bootstrap received different performance sets")
    by_fold: dict[str, list[float]] = defaultdict(list)
    differences = []
    for key in sorted(baseline):
        difference = _track_metric(candidate[key], metric) - _track_metric(baseline[key], metric)
        by_fold[key[0]].append(difference)
        differences.append(difference)
    fold_ids = sorted(by_fold)
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        chosen = rng.choice(fold_ids, size=len(fold_ids), replace=True)
        values = [value for fold_id in chosen for value in by_fold[fold_id]]
        samples[index] = float(np.mean(values))
    epsilon = 1e-12
    lower_is_better = metric in {
        "fragmentationRate",
        "regionsPerMinute",
        "meanAbsoluteBoundaryErrorMs",
    }
    improved = (
        sum(value < -epsilon for value in differences)
        if lower_is_better
        else sum(value > epsilon for value in differences)
    )
    worsened = (
        sum(value > epsilon for value in differences)
        if lower_is_better
        else sum(value < -epsilon for value in differences)
    )
    return {
        "meanDifference": float(np.mean(differences)),
        "medianDifference": float(np.median(differences)),
        "confidenceInterval95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "improved": improved,
        "tied": sum(abs(value) <= epsilon for value in differences),
        "worsened": worsened,
        "performanceCount": len(differences),
        "clusterCount": len(fold_ids),
        "seed": seed,
        "iterations": iterations,
    }


def pareto_frontier(aggregates: dict[str, Any]) -> list[str]:
    objectives = (
        ("detailedAccuracy", True),
        ("rootAccuracy", True),
        ("fragmentationRate", False),
        ("regionsPerMinute", False),
        ("meanAbsoluteBoundaryErrorMs", False),
    )
    values = {candidate: data["pairedPerformances"] for candidate, data in aggregates.items()}
    frontier = []
    for candidate, metrics in values.items():
        dominated = False
        for other, other_metrics in values.items():
            if other == candidate:
                continue
            no_worse = all(
                other_metrics[name] >= metrics[name] if maximize
                else other_metrics[name] <= metrics[name]
                for name, maximize in objectives
            )
            strictly_better = any(
                other_metrics[name] > metrics[name] if maximize
                else other_metrics[name] < metrics[name]
                for name, maximize in objectives
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return frontier


def _boundary_state_switch_agreement(
    model: Any,
    tracks: list[Any],
    feature_map: dict[str, Any],
    smoothing: dict[str, Any],
) -> dict[str, dict[str, float]]:
    from .adapters import forward_probabilities

    totals = {
        capture: {"weightedAgreement": 0.0, "duration": 0.0}
        for capture in CAPTURES
    }
    for track in tracks:
        features = feature_map[track.track_id]
        learned = forward_probabilities(model, features)
        root, quality, no_chord, boundary = smooth_learned_probabilities(
            *learned, smoothing
        )
        chord_states = (
            root[:, :, None]
            * quality[:, None, :]
            * (1.0 - no_chord[:, None, None])
        ).reshape(len(root), -1)
        states = np.concatenate([chord_states, no_chord[:, None]], axis=1)
        frame_switch = (
            0.5 * np.abs(states[1:] - states[:-1]).sum(axis=1)
            if len(states) > 1
            else np.zeros(0, dtype=np.float32)
        )
        agreement = (
            float(1.0 - np.mean(np.abs(boundary[1:] - frame_switch)))
            if len(frame_switch)
            else 1.0
        )
        agreement = float(np.clip(agreement, 0.0, 1.0))
        capture = _capture(track)
        duration = float(track.duration)
        totals[capture]["weightedAgreement"] += agreement * duration
        totals[capture]["duration"] += duration
    return {
        capture: {
            "boundaryStateSwitchAgreement": (
                values["weightedAgreement"] / max(values["duration"], 1e-9)
            ),
            "evaluatedDurationSeconds": values["duration"],
        }
        for capture, values in totals.items()
    }


def run_checkpoint_diagnostics(
    *,
    report: dict[str, Any],
    base: dict[str, Any],
    splits: dict[str, Any],
    ablations: dict[str, Any],
    smoothing: dict[str, Any],
    annotation_dir: Path,
    audio_dirs: dict[str, Path],
    run_dir: Path,
) -> dict[str, Any]:
    from ..training.checkpoint import load_checkpoint

    tracks = _load_tracks(annotation_dir, audio_dirs)
    allowed = {item["performerId"] for item in splits["modelSelectionPerformers"]}
    tracks = [track for track in tracks if track.artist in allowed]
    if not tracks or any(track.artist == FORBIDDEN_SELECTION_PERFORMER for track in tracks):
        raise ValueError("diagnostic track loading violated the p01-p05-only protocol")
    feature_map = _extract_feature_map(tracks, base["features"]["pipelineVersion"])
    candidate_map = {candidate["id"]: candidate for candidate in ablations["candidates"]}
    stored = {(row["foldId"], row["candidateId"]): row for row in report["foldResults"]}
    runtime_rows = []
    frozen_evaluations: dict[tuple[str, str], dict[str, Any]] = {}

    for fold in splits["folds"]:
        validation = set(fold["validationPerformers"])
        validation_tracks = [track for track in tracks if track.artist in validation]
        for candidate in ablations["candidates"]:
            key = (fold["foldId"], candidate["id"])
            model, _ = load_checkpoint(run_dir / fold["foldId"] / candidate["id"] / "model.pt")
            config = candidate_config(base, candidate)
            started = time.perf_counter()
            evaluated = _evaluate_model(model, validation_tracks, feature_map, config)
            elapsed = time.perf_counter() - started
            agreement = _boundary_state_switch_agreement(
                model,
                validation_tracks,
                feature_map,
                config.get("decoding", {}).get("probabilitySmoothing", {"method": "none"}),
            )
            frozen_evaluations[key] = evaluated
            for capture in CAPTURES:
                expected = stored[key]["metrics"]["byCapture"][capture]
                actual = evaluated["byCapture"][capture]
                for metric in (
                    "rootAccuracy",
                    "majorMinorAccuracy",
                    "detailedAccuracy",
                    "fragmentationRate",
                    "regionsPerMinute",
                    "meanAbsoluteBoundaryErrorMs",
                ):
                    if not math.isclose(float(actual[metric]), float(expected[metric]), abs_tol=1e-9):
                        raise ValueError(f"{key}/{capture}: checkpoint reproduction mismatch for {metric}")
            minutes = sum(
                evaluated["byCapture"][capture]["evaluatedDurationSeconds"]
                for capture in CAPTURES
            ) / 60
            runtime_rows.append({
                "foldId": fold["foldId"],
                "candidateId": candidate["id"],
                "runtimeSeconds": elapsed,
                "evaluatedAudioMinutes": minutes,
                "runtimeSecondsPerAudioMinute": elapsed / max(minutes, 1e-9),
                "byCaptureSequenceDiagnostics": agreement,
            })

    smoothing_rows = []
    checkpoint_candidate = smoothing["checkpointCandidate"]
    if checkpoint_candidate != "full-objective-unsmoothed":
        raise ValueError("smoothing diagnostics must use the frozen unsmoothed checkpoint")
    for fold in splits["folds"]:
        validation = set(fold["validationPerformers"])
        validation_tracks = [track for track in tracks if track.artist in validation]
        model, _ = load_checkpoint(run_dir / fold["foldId"] / checkpoint_candidate / "model.pt")
        base_candidate = candidate_map[checkpoint_candidate]
        for method in smoothing["methods"]:
            if method["id"] == "none":
                evaluated = frozen_evaluations[(fold["foldId"], checkpoint_candidate)]
                elapsed = next(
                    row["runtimeSeconds"] for row in runtime_rows
                    if row["foldId"] == fold["foldId"]
                    and row["candidateId"] == checkpoint_candidate
                )
                settings = method["settings"]
            else:
                config = candidate_config(base, base_candidate)
                config["decoding"]["probabilitySmoothing"] = copy.deepcopy(method["settings"])
                started = time.perf_counter()
                evaluated = _evaluate_model(model, validation_tracks, feature_map, config)
                elapsed = time.perf_counter() - started
                settings = config["decoding"]["probabilitySmoothing"]
            agreement = _boundary_state_switch_agreement(
                model, validation_tracks, feature_map, settings
            )
            smoothing_rows.append({
                "foldId": fold["foldId"],
                "methodId": method["id"],
                "settings": method["settings"],
                "metrics": evaluated,
                "runtimeSeconds": elapsed,
                "byCaptureSequenceDiagnostics": agreement,
            })
    return {
        "schemaVersion": 1,
        "status": "completed",
        "p00Used": False,
        "selectionEligible": False,
        "diagnosticId": smoothing["diagnosticId"],
        "methodManifestChecksum": _portable_checksum(smoothing),
        "checkpointReproduction": {
            "status": "passed",
            "runCount": len(runtime_rows),
        },
        "runtimeRows": runtime_rows,
        "smoothingRows": smoothing_rows,
    }


def build_reports(
    report: dict[str, Any],
    base: dict[str, Any],
    splits: dict[str, Any],
    ablations: dict[str, Any],
    augmentations: dict[str, Any],
    diagnostics: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    audit = validate_completed_matrix(report, base, splits, ablations, augmentations)
    aggregates = aggregate_candidates(report)
    baseline_id = "v1-objective"
    gates = {}
    comparisons = {}
    baseline_rows = _candidate_rows(report, baseline_id)
    for candidate in ablations["candidates"][1:]:
        candidate_id = candidate["id"]
        gate = evaluate_selection_gates(
            aggregates[baseline_id]["byCapture"],
            aggregates[candidate_id]["byCapture"],
            base["selectionGates"],
            parity_passed=True,
            hybrid_improvement_validated=False,
        )
        metric_checks = gate["checks"]
        gates[candidate_id] = {
            **gate,
            "metricGatesPassed": all(check["passed"] for check in metric_checks),
            "hybridGateStatus": "not-evaluated-candidate-ineligible-on-metric-gates",
        }
        candidate_rows = _candidate_rows(report, candidate_id)
        comparisons[candidate_id] = {
            metric: paired_bootstrap(baseline_rows, candidate_rows, metric)
            for metric in (
                "rootAccuracy",
                "detailedAccuracy",
                "fragmentationRate",
                "regionsPerMinute",
                "meanAbsoluteBoundaryErrorMs",
            )
        }

    eligible = [
        candidate_id for candidate_id, result in gates.items()
        if result["metricGatesPassed"] and result["passed"]
    ]
    if eligible:
        raise ValueError("automatic winner selection is forbidden; freeze an explicit decision")

    parity = {
        "runCount": len(report["foldResults"]),
        "allPythonOnnxPassed": all(
            row["parity"]["pythonOnnx"]["passed"] for row in report["foldResults"]
        ),
        "allTypeScriptPassed": all(
            row["parity"]["typescript"]["passed"] for row in report["foldResults"]
        ),
        "maximumPythonOnnxError": max(
            row["parity"]["pythonOnnx"]["maximumAbsoluteError"]
            for row in report["foldResults"]
        ),
        "maximumTypeScriptError": max(
            row["parity"]["typescript"]["maximumAbsoluteError"]
            for row in report["foldResults"]
        ),
        "tolerance": 1e-4,
    }
    contrast_metrics = (
        "rootAccuracy",
        "detailedAccuracy",
        "fragmentationRate",
        "regionsPerMinute",
        "meanAbsoluteBoundaryErrorMs",
        "chordSwitchPrecision",
        "chordSwitchRecall",
        "chordSwitchF1",
    )
    contrast_specs = (
        (
            "longerReceptiveField",
            "v1-objective",
            "longer-context",
            "Longer receptive field; all sequence-loss weights remain zero.",
        ),
        (
            "durationConsistency",
            "longer-context",
            "duration-objective",
            "Duration-consistency loss added to the longer-context model.",
        ),
        (
            "boundaryAndSwitchSupervision",
            "longer-context",
            "boundary-switch-objective",
            "Switch, hard-boundary, and boundary/state agreement losses added together.",
        ),
        (
            "combinedSequenceObjectives",
            "longer-context",
            "full-objective-unsmoothed",
            "All sequence objectives added together without augmentation or smoothing.",
        ),
        (
            "augmentationAndEma",
            "full-objective-unsmoothed",
            "full-v2",
            "Deterministic feature augmentation and EMA are added together; this contrast cannot isolate either component.",
        ),
    )
    ablation_contrasts = {}
    for contrast_id, baseline_candidate, candidate, limitation in contrast_specs:
        ablation_contrasts[contrast_id] = {
            "baselineCandidate": baseline_candidate,
            "candidate": candidate,
            "interpretation": limitation,
            "byCaptureDelta": {
                capture: {
                    metric: (
                        aggregates[candidate]["byCapture"][capture][metric]
                        - aggregates[baseline_candidate]["byCapture"][capture][metric]
                    )
                    for metric in contrast_metrics
                }
                for capture in CAPTURES
            },
        }
    sequence_diagnostics: dict[str, Any] = {}
    if diagnostics:
        for candidate_id in aggregates:
            rows = [
                row for row in diagnostics["runtimeRows"]
                if row["candidateId"] == candidate_id
            ]
            sequence_diagnostics[candidate_id] = {}
            for capture in CAPTURES:
                duration = sum(
                    row["byCaptureSequenceDiagnostics"][capture]["evaluatedDurationSeconds"]
                    for row in rows
                )
                sequence_diagnostics[candidate_id][capture] = {
                    "boundaryStateSwitchAgreement": sum(
                        row["byCaptureSequenceDiagnostics"][capture][
                            "boundaryStateSwitchAgreement"
                        ]
                        * row["byCaptureSequenceDiagnostics"][capture][
                            "evaluatedDurationSeconds"
                        ]
                        for row in rows
                    ) / max(duration, 1e-9)
                }
    summary = {
        "schemaVersion": 1,
        "status": "completed",
        "study": report["ablationId"],
        "seed": report["seed"],
        "configurationChecksum": frozen_configuration_checksum(),
        "reportChecksum": hashlib.sha256(
            json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "splitId": splits["splitId"],
        "datasetIdentity": splits["datasetIdentity"],
        "folds": splits["folds"],
        "candidates": report["candidates"],
        "expectedRunCount": report["expectedRunCount"],
        "completedRunCount": report["completedRunCount"],
        "failedRunCount": report["failedRunCount"],
        "loadedPerformanceCount": report["loadedPerformanceCount"],
        "loadedTrackCaptureCount": report["loadedTrackCaptureCount"],
        "totalTrainingSeconds": sum(
            float(row["training"]["trainSeconds"]) for row in report["foldResults"]
        ),
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "iterations": BOOTSTRAP_ITERATIONS,
            "unit": "paired performance clustered by held-out performer fold",
        },
        "selectionPerformers": [
            item["performerId"] for item in splits["modelSelectionPerformers"]
        ],
        "p00UsedForSelection": False,
        "audit": audit,
        "aggregates": aggregates,
        "ablationContrasts": ablation_contrasts,
        "pairedDifferencesVersusV1Objective": comparisons,
        "selectionGates": gates,
        "eligibleCandidateCount": len(eligible),
        "paretoFrontier": pareto_frontier(aggregates),
        "parity": parity,
        "runtimeDiagnostics": diagnostics["runtimeRows"] if diagnostics else None,
        "sequenceDiagnostics": sequence_diagnostics if diagnostics else None,
        "methodologicalLimitations": [
            "All p01-p05 performers overlap learned-model development; this is integration-domain development evidence.",
            "GuitarSet contains solo guitar, not a legally licensed artist-disjoint full-band benchmark.",
            "The p00 performer was not accessed because no v2 candidate passed the frozen replacement gates.",
        ],
    }
    decision = {
        "schemaVersion": 1,
        "status": "frozen",
        "decision": "retain-v1",
        "selectedCandidate": None,
        "selectedSmoothing": None,
        "eligibleCandidateCount": 0,
        "p00AccessAuthorized": False,
        "p00Accessed": False,
        "reason": (
            "Every v2 candidate failed the frozen fragmentation and regions-per-minute "
            "gates on both microphone and pickup captures. Accuracy and boundary gains "
            "cannot override the predeclared stability requirements."
        ),
        "configurationChecksum": summary["configurationChecksum"],
        "ablationReportChecksum": summary["reportChecksum"],
        "gateEvidence": {
            candidate_id: [
                check for check in result["checks"] if not check["passed"]
            ]
            for candidate_id, result in gates.items()
        },
        "parity": parity,
        "productionMutation": False,
        "hybridDefaultMutation": False,
    }
    return summary, decision


def _markdown(summary: dict[str, Any], decision: dict[str, Any]) -> str:
    lines = [
        "# Temporal harmony v2 grouped ablation",
        "",
        "## Executive summary",
        "",
        "The frozen 6-candidate x 5-fold study completed all 30 runs with no failures. "
        "No v2 candidate is eligible to replace v1: every candidate violated the frozen "
        "fragmentation and regions-per-minute gates on both captures.",
        "",
        "Decision: **retain v1**. p00 remains sealed and was not evaluated.",
        "",
        "## Aggregate results",
        "",
        "| Candidate | Capture | Root | Detailed | Fragmentation | Regions/min | Boundary MAE (ms) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for candidate, data in summary["aggregates"].items():
        for capture, label in (
            ("audio_mono-mic", "Microphone"),
            ("audio_mono-pickup_mix", "Pickup mix"),
        ):
            metrics = data["byCapture"][capture]
            lines.append(
                f"| {candidate} | {label} | {metrics['rootAccuracy']:.4f} | "
                f"{metrics['detailedAccuracy']:.4f} | {metrics['fragmentationRate']:.4f} | "
                f"{metrics['regionsPerMinute']:.3f} | "
                f"{metrics['meanAbsoluteBoundaryErrorMs']:.1f} |"
            )
    lines.extend([
        "",
        "## Gate result",
        "",
        f"Eligible candidates: **{summary['eligibleCandidateCount']}**. "
        f"Pareto frontier: {', '.join(summary['paretoFrontier'])}.",
        "",
        "All 30 checkpoints passed PyTorch/ONNX and actual TypeScript parity at the "
        f"{summary['parity']['tolerance']:.0e} tolerance.",
        "",
        "## Methodological limits",
        "",
    ])
    lines.extend(f"- {item}" for item in summary["methodologicalLimitations"])
    lines.append("")
    return "\n".join(lines)


def write_report_files(
    output_dir: Path,
    summary: dict[str, Any],
    decision: dict[str, Any],
    report: dict[str, Any],
    diagnostics: dict[str, Any] | None,
) -> None:
    _write_json(output_dir / "temporal-v2-ablation-summary.json", summary)
    (output_dir / "temporal-v2-ablation-summary.md").write_text(
        _markdown(summary, decision), encoding="utf-8"
    )
    _write_json(output_dir / "temporal-v2-pareto-frontier.json", {
        "schemaVersion": 1,
        "frontier": summary["paretoFrontier"],
        "objectives": {
            "maximize": ["detailedAccuracy", "rootAccuracy"],
            "minimize": ["fragmentationRate", "regionsPerMinute", "meanAbsoluteBoundaryErrorMs"],
        },
        "aggregates": {
            candidate: data["pairedPerformances"]
            for candidate, data in summary["aggregates"].items()
        },
    })
    _write_json(output_dir / "temporal-v2-parity.json", summary["parity"])
    _write_json(output_dir / "temporal-v2-selection.json", decision)
    (output_dir / "temporal-v2-selection.md").write_text(
        "# Temporal harmony v2 frozen selection\n\n"
        "**Decision: retain v1.**\n\n"
        f"{decision['reason']}\n\n"
        "p00 was not accessed. Production weights and hybrid defaults remain unchanged.\n",
        encoding="utf-8",
    )
    if diagnostics:
        smoothing_summary: dict[str, Any] = {
            "schemaVersion": 1,
            "status": diagnostics["status"],
            "diagnosticId": diagnostics["diagnosticId"],
            "selectionEligible": False,
            "p00Used": False,
            "methods": {},
        }
        for method_id in sorted({row["methodId"] for row in diagnostics["smoothingRows"]}):
            method_rows = [
                row for row in diagnostics["smoothingRows"] if row["methodId"] == method_id
            ]
            per_track = [
                track
                for row in method_rows
                for track in row["metrics"]["perTrack"]
            ]
            smoothing_summary["methods"][method_id] = {
                "settings": method_rows[0]["settings"],
                "byCapture": {
                    capture: _extended_aggregate([
                        row for row in per_track if row["capture"] == capture
                    ])
                    for capture in CAPTURES
                },
                "byCaptureSequenceDiagnostics": {
                    capture: {
                        "boundaryStateSwitchAgreement": sum(
                            row["byCaptureSequenceDiagnostics"][capture][
                                "boundaryStateSwitchAgreement"
                            ]
                            * row["byCaptureSequenceDiagnostics"][capture][
                                "evaluatedDurationSeconds"
                            ]
                            for row in method_rows
                        ) / max(
                            sum(
                                row["byCaptureSequenceDiagnostics"][capture][
                                    "evaluatedDurationSeconds"
                                ]
                                for row in method_rows
                            ),
                            1e-9,
                        )
                    }
                    for capture in CAPTURES
                },
                "runtimeSeconds": sum(row["runtimeSeconds"] for row in method_rows),
            }
        _write_json(
            output_dir / "temporal-v2-smoothing-comparison.json",
            smoothing_summary,
        )

    with (output_dir / "temporal-v2-fold-results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = [
            "foldId", "candidateId", "capture", "rootAccuracy",
            "majorMinorAccuracy", "detailedAccuracy", "fragmentationRate",
            "regionsPerMinute", "predictedRegionCount", "shortRegionRate",
            "flickerCount", "meanAbsoluteBoundaryErrorMs",
            "medianAbsoluteBoundaryErrorMs", "boundaryF1At100ms",
            "boundaryF1At250ms", "boundaryF1At500ms", "boundaryF1At1000ms",
            "weightedChordSymbolRecall", "noChordF1",
            "meanSignedBoundaryErrorMs", "boundariesWithin100ms",
            "boundariesWithin250ms", "boundariesWithin500ms",
            "boundariesWithin1000ms", "chordSwitchPrecision",
            "chordSwitchRecall", "chordSwitchF1",
            "boundaryStateSwitchAgreement",
            "trainSeconds", "epochsRun", "checkpointChecksum",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["foldResults"]:
            for capture in CAPTURES:
                source_rows = [
                    item for item in row["metrics"]["perTrack"]
                    if item["capture"] == capture
                ]
                metrics = _extended_aggregate(source_rows)
                runtime_row = next(
                    (
                        item for item in (diagnostics or {}).get("runtimeRows", [])
                        if item["foldId"] == row["foldId"]
                        and item["candidateId"] == row["candidateId"]
                    ),
                    None,
                )
                sequence = (
                    runtime_row["byCaptureSequenceDiagnostics"][capture]
                    if runtime_row
                    else {"boundaryStateSwitchAgreement": ""}
                )
                writer.writerow({
                    "foldId": row["foldId"],
                    "candidateId": row["candidateId"],
                    "capture": capture,
                    **{key: metrics[key] for key in fieldnames if key in metrics},
                    "boundaryStateSwitchAgreement": sequence[
                        "boundaryStateSwitchAgreement"
                    ],
                    "trainSeconds": row["training"]["trainSeconds"],
                    "epochsRun": row["training"]["epochsRun"],
                    "checkpointChecksum": row["training"]["stateDictChecksum"],
                })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--diagnostics-output", default=None)
    parser.add_argument("--run-checkpoint-diagnostics", action="store_true")
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--mic-audio", default=None)
    parser.add_argument("--pickup-audio", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    base, splits, ablations, augmentations = (_read_json(path) for path in CONFIG_PATHS)
    smoothing = _read_json(DEFAULT_SMOOTHING)
    report = _read_json(run_dir / "ablation-report.json")
    validate_completed_matrix(report, base, splits, ablations, augmentations)

    diagnostics_path = (
        Path(args.diagnostics_output).resolve()
        if args.diagnostics_output
        else run_dir / "checkpoint-diagnostics.json"
    )
    if args.run_checkpoint_diagnostics:
        required = (args.annotations, args.mic_audio, args.pickup_audio)
        if not all(required):
            parser.error("checkpoint diagnostics require annotation, mic and pickup paths")
        diagnostics = run_checkpoint_diagnostics(
            report=report,
            base=base,
            splits=splits,
            ablations=ablations,
            smoothing=smoothing,
            annotation_dir=Path(args.annotations).resolve(),
            audio_dirs={
                "audio_mono-mic": Path(args.mic_audio).resolve(),
                "audio_mono-pickup_mix": Path(args.pickup_audio).resolve(),
            },
            run_dir=run_dir,
        )
        _write_json(diagnostics_path, diagnostics)
    else:
        diagnostics = _read_json(diagnostics_path) if diagnostics_path.exists() else None
    summary, decision = build_reports(
        report, base, splits, ablations, augmentations, diagnostics
    )
    output_dir = Path(args.output_dir).resolve()
    write_report_files(output_dir, summary, decision, report, diagnostics)
    serialized = "\n".join(
        path.read_text(encoding="utf-8")
        for path in output_dir.glob("temporal-v2-*")
        if path.suffix in {".json", ".md", ".csv"}
    )
    if str(Path.home()).lower() in serialized.lower():
        raise ValueError("absolute local path leaked into tracked temporal-v2 reports")
    print(json.dumps({
        "status": "completed",
        "decision": decision["decision"],
        "eligibleCandidateCount": decision["eligibleCandidateCount"],
        "p00Accessed": decision["p00Accessed"],
        "configurationChecksum": summary["configurationChecksum"],
    }, indent=2))


if __name__ == "__main__":
    main()
