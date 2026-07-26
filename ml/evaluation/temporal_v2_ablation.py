"""Frozen performer-grouped ablation harness for temporal-harmony-v2.

The default command prints a portable plan. ``--run`` performs the fixed six
candidate, five-fold experiment on local GuitarSet audio. It never loads p00,
never mutates application weights, and writes heavyweight artifacts only under
an ignored run directory.

Examples:
    python -m ml.evaluation.temporal_v2_ablation
    python -m ml.evaluation.temporal_v2_ablation --run \
        --annotations <annotation-dir> --mic-audio <mic-dir> \
        --pickup-audio <pickup-dir>
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import traceback
from collections import defaultdict
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = REPO_ROOT / "ml"
DEFAULT_CONFIG = ML_ROOT / "configs" / "temporal-harmony-v2.json"
DEFAULT_SPLITS = ML_ROOT / "configs" / "temporal-harmony-v2-splits.json"
DEFAULT_ABLATIONS = ML_ROOT / "configs" / "temporal-harmony-v2-ablations.json"
DEFAULT_AUGMENTATIONS = ML_ROOT / "configs" / "temporal-harmony-v2-augmentations.json"
DEFAULT_RUN_DIR = ML_ROOT / "runs" / "temporal-harmony-v2"
FORBIDDEN_SELECTION_PERFORMER = "guitarset-p00"
CAPTURES = ("audio_mono-mic", "audio_mono-pickup_mix")


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _portable_checksum(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _load_completed_run(path: Path, expected_identity: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = _read_json(path)
    if payload.get("runIdentity") != expected_identity:
        raise ValueError(
            f"{expected_identity['foldId']}/{expected_identity['candidateId']}: "
            "existing completed result has a different frozen identity"
        )
    if payload.get("status") != "completed" or not isinstance(payload.get("result"), dict):
        return None
    return payload["result"]


def validate_split_manifest(manifest: dict[str, Any]) -> None:
    performers = {item["performerId"] for item in manifest["modelSelectionPerformers"]}
    excluded = {item["performerId"] for item in manifest["excludedFromModelSelection"]}
    if FORBIDDEN_SELECTION_PERFORMER not in excluded:
        raise ValueError("p00 must be explicitly excluded from v2 model selection")
    if FORBIDDEN_SELECTION_PERFORMER in performers:
        raise ValueError("p00 cannot appear in modelSelectionPerformers")
    if manifest.get("capturePolicy", "").strip() == "":
        raise ValueError("split manifest must define alternate-capture grouping")
    validation_seen: set[str] = set()
    for fold in manifest["folds"]:
        train = set(fold["trainPerformers"])
        validation = set(fold["validationPerformers"])
        if train & validation:
            raise ValueError(f"{fold['foldId']}: train/validation performer overlap")
        if (train | validation) - performers:
            raise ValueError(f"{fold['foldId']}: unknown performer in fold")
        if excluded & (train | validation):
            raise ValueError(f"{fold['foldId']}: excluded performer entered selection")
        validation_seen |= validation
    if validation_seen != performers:
        raise ValueError("every model-selection performer must be held out by exactly one fold")


def validate_ablation_manifest(manifest: dict[str, Any]) -> None:
    candidates = manifest.get("candidates", [])
    maximum = int(manifest["constraints"]["candidateCountMaximum"])
    if not candidates or len(candidates) > maximum:
        raise ValueError(f"ablation candidate count must be between 1 and {maximum}")
    ids = [candidate["id"] for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("ablation candidate IDs must be unique")
    if ids[0] != "v1-objective":
        raise ValueError("the first ablation candidate must be the v1 objective reference")
    if not manifest["constraints"].get("p00Forbidden", False):
        raise ValueError("ablation manifest must forbid p00")


def candidate_config(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config["modelName"] = f"temporal-harmony-v2-{candidate['id']}"
    for section, override_key in (
        ("model", "modelOverrides"),
        ("training", "trainingOverrides"),
        ("labels", "labelOverrides"),
    ):
        config.setdefault(section, {}).update(candidate.get(override_key, {}))
    if "smoothing" in candidate:
        config.setdefault("decoding", {})["probabilitySmoothing"] = copy.deepcopy(candidate["smoothing"])
    return config


def build_ablation_plan(
    base_config: dict[str, Any],
    split_manifest: dict[str, Any],
    ablation_manifest: dict[str, Any],
    augmentation_manifest: dict[str, Any],
) -> dict[str, Any]:
    validate_split_manifest(split_manifest)
    validate_ablation_manifest(ablation_manifest)
    candidates = []
    for candidate in ablation_manifest["candidates"]:
        config = candidate_config(base_config, candidate)
        model = config["model"]
        receptive_field_frames = (
            1
            + 2
            * (int(model["kernelSize"]) - 1)
            * sum(int(dilation) for dilation in model["dilations"])
        )
        candidates.append({
            "id": candidate["id"],
            "description": candidate["description"],
            "configChecksum": _portable_checksum(config),
            "receptiveFieldFrames": receptive_field_frames,
            "receptiveFieldSecondsAt250msHop": receptive_field_frames * 0.25,
            "useAugmentation": bool(candidate.get("useAugmentation", False)),
            "smoothing": config.get("decoding", {}).get("probabilitySmoothing", {"method": "none"}),
            "temporalLossWeights": {
                key: config["training"].get(key, 0.0)
                for key in (
                    "durationConsistencyLossWeight",
                    "switchLossWeight",
                    "hardBoundaryLossWeight",
                    "boundarySwitchAgreementLossWeight",
                )
            },
        })
    return {
        "schemaVersion": 1,
        "ablationId": ablation_manifest["ablationId"],
        "status": "planned",
        "seed": ablation_manifest["seed"],
        "splitId": split_manifest["splitId"],
        "splitManifestChecksum": _portable_checksum(split_manifest),
        "augmentationId": augmentation_manifest["augmentationId"],
        "augmentationManifestChecksum": _portable_checksum(augmentation_manifest),
        "p00UsedForSelection": False,
        "captures": list(CAPTURES),
        "folds": split_manifest["folds"],
        "candidates": candidates,
        "candidateFoldRuns": len(candidates) * len(split_manifest["folds"]),
        "finalEvaluationGate": split_manifest["finalEvaluationGate"],
        "productionMutation": False,
        "hybridDefaultMutation": False,
    }


def evaluate_selection_gates(
    baseline_by_capture: dict[str, dict[str, float]],
    candidate_by_capture: dict[str, dict[str, float]],
    gates: dict[str, Any],
    *,
    parity_passed: bool = True,
    hybrid_improvement_validated: bool = False,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for capture in CAPTURES:
        baseline = baseline_by_capture[capture]
        candidate = candidate_by_capture[capture]
        values = {
            "rootAccuracy": candidate["rootAccuracy"] - baseline["rootAccuracy"],
            "detailedAccuracy": candidate["detailedAccuracy"] - baseline["detailedAccuracy"],
            "fragmentationRelativeReduction": (
                (baseline["fragmentationRate"] - candidate["fragmentationRate"])
                / max(baseline["fragmentationRate"], 1e-9)
            ),
            "regionsPerMinuteRelativeReduction": (
                (baseline["regionsPerMinute"] - candidate["regionsPerMinute"])
                / max(baseline["regionsPerMinute"], 1e-9)
            ),
            "boundaryMeanAbsoluteErrorRelativeIncrease": (
                (candidate["meanAbsoluteBoundaryErrorMs"] - baseline["meanAbsoluteBoundaryErrorMs"])
                / max(baseline["meanAbsoluteBoundaryErrorMs"], 1e-9)
            ),
        }
        thresholds = {
            "rootAccuracy": float(gates["rootAccuracyDeltaMinimum"]),
            "detailedAccuracy": float(gates["detailedAccuracyDeltaMinimum"]),
            "fragmentationRelativeReduction": float(gates["fragmentationRelativeReductionMinimum"]),
            "regionsPerMinuteRelativeReduction": float(gates["regionsPerMinuteRelativeReductionMinimum"]),
            "boundaryMeanAbsoluteErrorRelativeIncrease": float(
                gates["boundaryMeanAbsoluteErrorRelativeIncreaseMaximum"]
            ),
        }
        for name, value in values.items():
            if name == "boundaryMeanAbsoluteErrorRelativeIncrease":
                passed = value <= thresholds[name]
                comparison = "maximum"
            else:
                passed = value >= thresholds[name]
                comparison = "minimum"
            checks.append({
                "capture": capture,
                "gate": name,
                "value": value,
                "threshold": thresholds[name],
                "comparison": comparison,
                "passed": passed,
            })
    metric_gates_passed = all(check["passed"] for check in checks)
    replacement_checks = [
        {
            "gate": "pythonTypeScriptParity",
            "passed": parity_passed,
        },
        {
            "gate": "hybridImprovementWithFrozenSafeguards",
            "passed": hybrid_improvement_validated,
        },
    ]
    return {
        "developmentMetricGatesPassed": metric_gates_passed,
        "replacementGatePassed": (
            metric_gates_passed
            and all(check["passed"] for check in replacement_checks)
        ),
        "passed": (
            metric_gates_passed
            and all(check["passed"] for check in replacement_checks)
        ),
        "checks": checks,
        "replacementChecks": replacement_checks,
    }


def _resolve_data_path(argument: str | None, environment_name: str) -> Path | None:
    value = argument or os.environ.get(environment_name)
    return Path(value).expanduser().resolve() if value else None


def _load_tracks(annotation_dir: Path, audio_dirs: dict[str, Path]) -> list[Any]:
    from ..preprocessing.import_guitarset import import_guitarset_track

    tracks = []
    suffixes = {"audio_mono-mic": "_mic.wav", "audio_mono-pickup_mix": "_mix.wav"}
    for jams_path in sorted(annotation_dir.rglob("*.jams")):
        for capture in CAPTURES:
            audio_path = audio_dirs[capture] / f"{jams_path.stem}{suffixes[capture]}"
            if not audio_path.exists():
                raise FileNotFoundError(f"missing {capture} audio for {jams_path.stem}")
            track = import_guitarset_track(jams_path, audio_dir=None)
            track.audio_availability = "audio"
            track.audio_path = str(audio_path)
            track.track_id = f"{track.track_id}@{capture}"
            track.notes = f"{track.notes} capture={capture}"
            track.validate()
            tracks.append(track)
    if not tracks:
        raise ValueError("zero GuitarSet tracks loaded")
    if any(track.artist == FORBIDDEN_SELECTION_PERFORMER for track in tracks):
        # Loading p00 is allowed from disk, but it is removed before feature
        # extraction and never reaches any candidate below.
        tracks = [track for track in tracks if track.artist != FORBIDDEN_SELECTION_PERFORMER]
    if any(track.artist == FORBIDDEN_SELECTION_PERFORMER for track in tracks):
        raise AssertionError("p00 entered v2 model selection")
    return tracks


def _capture(track: Any) -> str:
    return track.track_id.rsplit("@", 1)[-1]


def _base_track_id(track: Any) -> str:
    return track.track_id.rsplit("@", 1)[0]


def validate_local_dataset(
    annotation_dir: Path,
    audio_dirs: dict[str, Path],
    split_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Validate portable IDs and capture pairing without extracting features."""
    validate_split_manifest(split_manifest)
    tracks = _load_tracks(annotation_dir, audio_dirs)
    allowed = {item["performerId"] for item in split_manifest["modelSelectionPerformers"]}
    tracks = [track for track in tracks if track.artist in allowed]
    pair_captures: dict[str, set[str]] = defaultdict(set)
    performer_counts: dict[str, set[str]] = defaultdict(set)
    for track in tracks:
        pair_captures[_base_track_id(track)].add(_capture(track))
        performer_counts[track.artist].add(_base_track_id(track))
    incomplete = sorted(
        track_id for track_id, captures in pair_captures.items()
        if captures != set(CAPTURES)
    )
    if incomplete:
        raise ValueError(f"{len(incomplete)} performances lack paired captures")
    if any(performer == FORBIDDEN_SELECTION_PERFORMER for performer in performer_counts):
        raise AssertionError("p00 entered local model-selection data")
    portable_tracks = {}
    for track in tracks:
        portable_tracks.setdefault(_base_track_id(track), track)
    portable_rows = [
        {
            "trackId": track_id,
            "performerId": track.artist,
            "duration": track.duration,
            "chords": [
                {"start": region.start, "end": region.end, "label": region.label}
                for region in track.chords
            ],
        }
        for track_id, track in sorted(portable_tracks.items())
    ]
    portable_manifest_checksum = _portable_checksum(portable_rows)
    expected_checksum = split_manifest["datasetIdentity"]["modelSelectionAnnotationManifestChecksum"]
    if portable_manifest_checksum != expected_checksum:
        raise ValueError("local model-selection annotation manifest checksum does not match frozen split")
    duration_per_capture = sum(track.duration for track in portable_tracks.values())
    expected_duration = float(split_manifest["modelSelectionDurationSecondsPerCapture"])
    if abs(duration_per_capture - expected_duration) > 1e-6:
        raise ValueError("local model-selection duration does not match frozen split")
    return {
        "schemaVersion": 1,
        "status": "passed",
        "splitId": split_manifest["splitId"],
        "modelSelectionAnnotationCount": len(pair_captures),
        "performanceCount": len(pair_captures),
        "trackCaptureCount": len(tracks),
        "durationSecondsPerCapture": duration_per_capture,
        "portableAnnotationManifestChecksum": portable_manifest_checksum,
        "captureCounts": {
            capture: sum(1 for track in tracks if _capture(track) == capture)
            for capture in CAPTURES
        },
        "performerPerformanceCounts": {
            performer: len(track_ids)
            for performer, track_ids in sorted(performer_counts.items())
        },
        "incompleteCapturePairs": incomplete,
        "p00UsedForSelection": False,
    }


def _extract_feature_map(tracks: list[Any], pipeline: str) -> dict[str, Any]:
    from ..preprocessing.feature_source import frames_for_track

    return {
        track.track_id: frames_for_track(track, audio_feature=pipeline)
        for track in tracks
    }


def _training_samples(
    tracks: list[Any],
    feature_map: dict[str, Any],
    config: dict[str, Any],
    augmentation_manifest: dict[str, Any] | None,
) -> list[Any]:
    from ..training.augmentation import build_augmented_samples
    from ..training.dataset import chunk_sample, pitch_shift_sample, sample_from_features

    tolerance = float(config["labels"]["boundaryToleranceSeconds"])
    assembled = []
    for track in tracks:
        sample = sample_from_features(track, feature_map[track.track_id], tolerance)
        if sample is None:
            continue
        sources = build_augmented_samples(sample, augmentation_manifest) if augmentation_manifest else [sample]
        for source in sources:
            for chunk in chunk_sample(source, 500):
                assembled.append(chunk)
                for shift in (-2, -1, 1, 2):
                    assembled.append(pitch_shift_sample(chunk, shift))
    return assembled


def _development_samples(
    tracks: list[Any], feature_map: dict[str, Any], config: dict[str, Any],
) -> list[Any]:
    from ..training.dataset import chunk_sample, sample_from_features

    tolerance = float(config["labels"]["boundaryToleranceSeconds"])
    assembled = []
    for track in tracks:
        sample = sample_from_features(track, feature_map[track.track_id], tolerance)
        if sample is not None:
            assembled.extend(chunk_sample(sample, 500))
    return assembled


def _aggregate_metrics(per_track: list[dict[str, Any]]) -> dict[str, float]:
    if not per_track:
        raise ValueError("cannot aggregate zero tracks")
    total_duration = sum(row["evaluatedDurationSeconds"] for row in per_track)
    total_reference = sum(row["referenceRegions"] for row in per_track)
    total_predicted = sum(row["predictedRegions"] for row in per_track)
    boundary_weight = sum(max(0, row["referenceRegions"] - 1) for row in per_track)

    def duration_weighted(name: str) -> float:
        return sum(row[name] * row["evaluatedDurationSeconds"] for row in per_track) / total_duration

    mean_boundary = (
        sum(
            (row["meanBoundaryErrorMs"] or 0.0) * max(0, row["referenceRegions"] - 1)
            for row in per_track
        ) / max(1, boundary_weight)
    )
    return {
        "trackCount": float(len(per_track)),
        "evaluatedDurationSeconds": total_duration,
        "rootAccuracy": duration_weighted("rootAccuracy"),
        "majorMinorAccuracy": duration_weighted("majorMinorAccuracy"),
        "detailedAccuracy": duration_weighted("detailedAccuracy"),
        "noChordPrecision": duration_weighted("noChordPrecision"),
        "noChordRecall": duration_weighted("noChordRecall"),
        "fragmentationRate": (
            sum(row["fragmentationRate"] * row["referenceRegions"] for row in per_track)
            / max(1, total_reference)
        ),
        "regionsPerMinute": total_predicted * 60.0 / total_duration,
        "predictedRegionCount": float(total_predicted),
        "shortRegionRate": (
            sum(row["shortRegionCount"] for row in per_track) / max(1, total_predicted)
        ),
        "oneWindowRegionRate": (
            sum(row["oneWindowRegionCount"] for row in per_track) / max(1, total_predicted)
        ),
        "flickerCount": float(sum(row["flickerCount"] for row in per_track)),
        "meanTopChordConfidence": duration_weighted("meanTopChordConfidence"),
        "meanLearnedEntropy": duration_weighted("meanLearnedEntropy"),
        "meanFrameSwitchProbability": duration_weighted("meanFrameSwitchProbability"),
        "meanBoundaryProbability": duration_weighted("meanBoundaryProbability"),
        "meanAbsoluteBoundaryErrorMs": mean_boundary,
        "medianAbsoluteBoundaryErrorMs": (
            sum(
                (row["medianBoundaryErrorMs"] or 0.0) * max(0, row["referenceRegions"] - 1)
                for row in per_track
            ) / max(1, boundary_weight)
        ),
        **{
            f"boundaryF1At{tolerance}": (
                sum(
                    row["boundaryF1"][tolerance]["f1"] * max(0, row["referenceRegions"] - 1)
                    for row in per_track
                ) / max(1, boundary_weight)
            )
            for tolerance in ("100ms", "250ms", "500ms", "1000ms")
        },
    }


def _evaluate_model(
    model: Any,
    tracks: list[Any],
    feature_map: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    from .adapters import forward_probabilities
    from .decode import viterbi_decode
    from .metrics import evaluate_regions
    from .probability_smoothing import smooth_learned_probabilities

    rows = []
    smoothing = config.get("decoding", {}).get("probabilitySmoothing", {"method": "none"})
    transition_penalty = float(config.get("decoding", {}).get("transitionPenalty", 4.0))
    for track in tracks:
        features = feature_map[track.track_id]
        learned = forward_probabilities(model, features)
        root, quality, no_chord, boundary = smooth_learned_probabilities(*learned, smoothing)
        chord_states = (
            root[:, :, None]
            * quality[:, None, :]
            * (1.0 - no_chord[:, None, None])
        ).reshape(len(root), -1)
        states = np.concatenate([chord_states, no_chord[:, None]], axis=1)
        state_entropy = -np.sum(states * np.log(np.maximum(states, 1e-9)), axis=1) / np.log(37.0)
        frame_switch = (
            0.5 * np.abs(states[1:] - states[:-1]).sum(axis=1)
            if len(states) > 1
            else np.zeros(0, dtype=np.float32)
        )
        predicted = viterbi_decode(
            features.times,
            root,
            quality,
            no_chord,
            features.hop_seconds,
            transition_penalty,
        )
        metrics = evaluate_regions(track.chords, predicted, tolerances=(0.1, 0.25, 0.5, 1.0))
        durations = [max(0.0, region.end - region.start) for region in predicted]
        one_window_count = sum(
            1 for duration in durations
            if duration <= features.hop_seconds * 1.5
        )
        short_count = sum(1 for duration in durations if duration < 0.5)
        flicker_count = sum(
            1
            for index in range(1, len(predicted) - 1)
            if predicted[index - 1].label == predicted[index + 1].label
            and predicted[index].label != predicted[index - 1].label
            and predicted[index].duration() < 1.0
        )
        rows.append({
            "trackId": _base_track_id(track),
            "capture": _capture(track),
            **metrics,
            "oneWindowRegionCount": one_window_count,
            "shortRegionCount": short_count,
            "flickerCount": flicker_count,
            "meanTopChordConfidence": float(np.mean(np.max(states, axis=1))) if len(states) else 0.0,
            "meanLearnedEntropy": float(np.mean(state_entropy)) if len(states) else 0.0,
            "meanFrameSwitchProbability": float(np.mean(frame_switch)) if len(frame_switch) else 0.0,
            "meanBoundaryProbability": float(np.mean(boundary)) if len(boundary) else 0.0,
        })
    by_capture = {
        capture: _aggregate_metrics([row for row in rows if row["capture"] == capture])
        for capture in CAPTURES
    }
    return {"byCapture": by_capture, "perTrack": rows}


def _python_onnx_parity(model: Any, onnx_path: Path, features: np.ndarray) -> dict[str, Any]:
    import onnxruntime as ort
    import torch

    sample = features[:min(64, len(features))].astype(np.float32)[None, :, :]
    with torch.no_grad():
        torch_output = model(torch.from_numpy(sample))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {"features": sample})
    names = ("root", "quality", "nochord", "boundary")
    errors = {
        name: float(np.max(np.abs(torch_output[name].numpy() - onnx_output[index])))
        for index, name in enumerate(names)
    }
    maximum = max(errors.values())
    return {"passed": maximum <= 1e-4, "maximumAbsoluteError": maximum, "headErrors": errors}


def _typescript_parity(checkpoint_path: Path, export_path: Path) -> dict[str, Any]:
    from ..exports.export_tcn_json import export

    export(checkpoint_path, export_path, feature_version="harmony-features-v1")
    npx = "npx.cmd" if os.name == "nt" else "npx"
    completed = subprocess.run(
        [
            npx,
            "vite-node",
            "evaluation/verify-temporal-v2-parity.ts",
            "--weights",
            str(export_path),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"TypeScript parity failed: {completed.stderr.strip()}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def run_ablation(
    *,
    annotation_dir: Path,
    audio_dirs: dict[str, Path],
    base_config: dict[str, Any],
    split_manifest: dict[str, Any],
    ablation_manifest: dict[str, Any],
    augmentation_manifest: dict[str, Any],
    run_dir: Path,
    selected_candidates: set[str] | None = None,
    selected_folds: set[str] | None = None,
    epochs_override: int | None = None,
) -> dict[str, Any]:
    from ..training.checkpoint import load_checkpoint
    from ..training.train import fit_samples

    plan = build_ablation_plan(base_config, split_manifest, ablation_manifest, augmentation_manifest)
    tracks = _load_tracks(annotation_dir, audio_dirs)
    allowed_performers = {item["performerId"] for item in split_manifest["modelSelectionPerformers"]}
    tracks = [track for track in tracks if track.artist in allowed_performers]
    expected_per_capture = len({_base_track_id(track) for track in tracks}) 
    if expected_per_capture == 0:
        raise ValueError("zero model-selection tracks loaded")
    feature_map = _extract_feature_map(tracks, base_config["features"]["pipelineVersion"])
    run_dir.mkdir(parents=True, exist_ok=True)
    fold_results = []
    failed_runs: list[dict[str, Any]] = []
    selected_fold_ids = {
        fold["foldId"]
        for fold in split_manifest["folds"]
        if not selected_folds or fold["foldId"] in selected_folds
    }
    selected_candidate_ids = {
        candidate["id"]
        for candidate in ablation_manifest["candidates"]
        if not selected_candidates or candidate["id"] in selected_candidates
    }
    expected_run_count = len(selected_fold_ids) * len(selected_candidate_ids)
    full_frozen_matrix = (
        selected_fold_ids == {fold["foldId"] for fold in split_manifest["folds"]}
        and selected_candidate_ids == {
            candidate["id"] for candidate in ablation_manifest["candidates"]
        }
    )

    for fold in split_manifest["folds"]:
        if selected_folds and fold["foldId"] not in selected_folds:
            continue
        train_performers = set(fold["trainPerformers"])
        validation_performers = set(fold["validationPerformers"])
        train_tracks = [track for track in tracks if track.artist in train_performers]
        validation_tracks = [track for track in tracks if track.artist in validation_performers]
        if not train_tracks or not validation_tracks:
            raise ValueError(f"{fold['foldId']}: zero train or validation tracks")
        for candidate in ablation_manifest["candidates"]:
            if selected_candidates and candidate["id"] not in selected_candidates:
                continue
            config = candidate_config(base_config, candidate)
            config["splitIdentity"] = {
                "splitId": split_manifest["splitId"],
                "foldId": fold["foldId"],
                "trainPerformers": fold["trainPerformers"],
                "validationPerformers": fold["validationPerformers"],
            }
            config["augmentationIdentity"] = (
                {
                    "augmentationId": augmentation_manifest["augmentationId"],
                    "manifestChecksum": _portable_checksum(augmentation_manifest),
                }
                if candidate.get("useAugmentation", False)
                else None
            )
            if epochs_override is not None:
                config["training"]["epochs"] = epochs_override
            checkpoint_dir = run_dir / fold["foldId"] / candidate["id"]
            checkpoint_path = checkpoint_dir / "model.pt"
            training_config_checksum = _portable_checksum(config)
            run_identity = {
                "foldId": fold["foldId"],
                "candidateId": candidate["id"],
                "trainingConfigChecksum": training_config_checksum,
                "seed": int(config["seed"]),
                "epochsOverride": epochs_override,
            }
            result_path = checkpoint_dir / "run-result.json"
            completed_result = _load_completed_run(result_path, run_identity)
            if completed_result is not None:
                print(f"resume: {fold['foldId']}/{candidate['id']} already completed")
                fold_results.append(completed_result)
                continue

            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            status_path = checkpoint_dir / "run-status.json"
            log_path = checkpoint_dir / "training.log"
            _write_json_atomic(status_path, {
                "schemaVersion": 1,
                "status": "running",
                "runIdentity": run_identity,
                "startedAt": datetime.now(timezone.utc).isoformat(),
            })
            print(f"run: {fold['foldId']}/{candidate['id']}")
            try:
                with log_path.open("a", encoding="utf-8") as log_handle, redirect_stdout(log_handle):
                    print(
                        f"starting {fold['foldId']}/{candidate['id']} "
                        f"at {datetime.now(timezone.utc).isoformat()}"
                    )
                    train_samples = _training_samples(
                        train_tracks,
                        feature_map,
                        config,
                        augmentation_manifest if candidate.get("useAugmentation", False) else None,
                    )
                    dev_samples = _development_samples(validation_tracks, feature_map, config)
                    summary = fit_samples(
                        config,
                        train_samples,
                        dev_samples,
                        checkpoint_path,
                        quiet=False,
                    )
                    model, metadata = load_checkpoint(checkpoint_path)
                    metrics = _evaluate_model(model, validation_tracks, feature_map, config)
                    first_features = feature_map[validation_tracks[0].track_id].stacked()
                    python_parity = _python_onnx_parity(
                        model,
                        checkpoint_path.with_suffix(".onnx"),
                        first_features,
                    )
                    typescript_parity = _typescript_parity(
                        checkpoint_path,
                        checkpoint_dir / "model.weights.json",
                    )
                    if not python_parity["passed"] or not typescript_parity["passed"]:
                        raise RuntimeError(
                            f"{fold['foldId']}/{candidate['id']}: export parity failed"
                        )
                    result = {
                        "foldId": fold["foldId"],
                        "candidateId": candidate["id"],
                        "trainingConfigChecksum": training_config_checksum,
                        "training": {
                            "trainSampleCount": len(train_samples),
                            "developmentSampleCount": len(dev_samples),
                            "bestEpoch": summary["bestEpoch"],
                            "bestDevLoss": summary["bestDevLoss"],
                            "epochsRun": summary["epochsRun"],
                            "trainSeconds": summary["trainSeconds"],
                            "stateDictChecksum": metadata["checksum"],
                            "datasetManifestHash": metadata["datasetManifestHash"],
                        },
                        "metrics": metrics,
                        "parity": {
                            "pythonOnnx": python_parity,
                            "typescript": typescript_parity,
                        },
                    }
                    print(
                        f"completed {fold['foldId']}/{candidate['id']} "
                        f"at {datetime.now(timezone.utc).isoformat()}"
                    )
                _write_json_atomic(result_path, {
                    "schemaVersion": 1,
                    "status": "completed",
                    "runIdentity": run_identity,
                    "result": result,
                })
                _write_json_atomic(status_path, {
                    "schemaVersion": 1,
                    "status": "completed",
                    "runIdentity": run_identity,
                    "completedAt": datetime.now(timezone.utc).isoformat(),
                    "stateDictChecksum": result["training"]["stateDictChecksum"],
                })
                fold_results.append(result)
            except Exception as error:
                with log_path.open("a", encoding="utf-8") as log_handle:
                    traceback.print_exc(file=log_handle)
                message = str(error)
                for local_path in (
                    str(Path.home()),
                    str(annotation_dir),
                    *(str(path) for path in audio_dirs.values()),
                ):
                    if local_path:
                        message = message.replace(local_path, "[local-path]")
                failure = {
                    "foldId": fold["foldId"],
                    "candidateId": candidate["id"],
                    "trainingConfigChecksum": training_config_checksum,
                    "errorType": type(error).__name__,
                    "message": message,
                }
                failed_runs.append(failure)
                _write_json_atomic(status_path, {
                    "schemaVersion": 1,
                    "status": "failed",
                    "runIdentity": run_identity,
                    "failedAt": datetime.now(timezone.utc).isoformat(),
                    "failure": failure,
                })
                print(
                    f"failed: {fold['foldId']}/{candidate['id']} "
                    f"({type(error).__name__})"
                )

    # Gate only complete candidate sets against the same-fold v1 reference.
    gate_results = []
    by_fold_candidate = {
        (row["foldId"], row["candidateId"]): row
        for row in fold_results
    }
    for fold in split_manifest["folds"]:
        baseline = by_fold_candidate.get((fold["foldId"], "v1-objective"))
        if baseline is None:
            continue
        for candidate in ablation_manifest["candidates"][1:]:
            row = by_fold_candidate.get((fold["foldId"], candidate["id"]))
            if row is None:
                continue
            gate_results.append({
                "foldId": fold["foldId"],
                "candidateId": candidate["id"],
                **evaluate_selection_gates(
                    baseline["metrics"]["byCapture"],
                    row["metrics"]["byCapture"],
                    base_config["selectionGates"],
                    parity_passed=(
                        row["parity"]["pythonOnnx"]["passed"]
                        and row["parity"]["typescript"]["passed"]
                    ),
                    hybrid_improvement_validated=False,
                ),
            })

    matrix_complete = len(fold_results) == expected_run_count and not failed_runs
    report = {
        **plan,
        "status": "completed" if matrix_complete else "incomplete",
        "validForSelection": (
            epochs_override is None
            and full_frozen_matrix
            and matrix_complete
        ),
        "expectedRunCount": expected_run_count,
        "completedRunCount": len(fold_results),
        "failedRunCount": len(failed_runs),
        "failedRuns": failed_runs,
        "resumePolicy": (
            "Completed candidate-fold results are reused only when their fold, "
            "candidate, seed, epoch mode, and training-config checksum match. "
            "Interrupted or failed individual runs restart from epoch zero."
        ),
        "loadedPerformanceCount": expected_per_capture,
        "loadedTrackCaptureCount": len(tracks),
        "p00UsedForSelection": False,
        "foldResults": fold_results,
        "selectionGateResults": gate_results,
        "selectedCandidate": None,
        "selectionNote": (
            "No candidate is selected automatically. Review grouped-fold gates, "
            "then freeze one candidate before any p00 comparison."
        ),
    }
    serialized = json.dumps(report, indent=2)
    for forbidden in (str(Path.home()), str(annotation_dir), *(str(path) for path in audio_dirs.values())):
        if forbidden and forbidden.lower() in serialized.lower():
            raise ValueError("absolute local path leaked into ablation report")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or run the fixed temporal-harmony-v2 ablation.")
    parser.add_argument("--run", action="store_true", help="Execute training; default prints the frozen plan.")
    parser.add_argument("--validate-data", action="store_true",
                        help="Validate local IDs/capture pairing without feature extraction or training.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS))
    parser.add_argument("--ablations", default=str(DEFAULT_ABLATIONS))
    parser.add_argument("--augmentations", default=str(DEFAULT_AUGMENTATIONS))
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--mic-audio", default=None)
    parser.add_argument("--pickup-audio", default=None)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--output", default=None)
    parser.add_argument("--candidate", action="append", default=None)
    parser.add_argument("--fold", action="append", default=None)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Smoke-only override; reports are marked invalid for selection.")
    args = parser.parse_args()

    base_config = _read_json(args.config)
    split_manifest = _read_json(args.splits)
    ablation_manifest = _read_json(args.ablations)
    augmentation_manifest = _read_json(args.augmentations)
    if not args.run and not args.validate_data:
        print(json.dumps(
            build_ablation_plan(base_config, split_manifest, ablation_manifest, augmentation_manifest),
            indent=2,
        ))
        return

    annotation_dir = _resolve_data_path(args.annotations, "TABSMITH_GUITARSET_ANNOTATIONS")
    mic_audio = _resolve_data_path(args.mic_audio, "TABSMITH_GUITARSET_MIC_AUDIO")
    pickup_audio = _resolve_data_path(args.pickup_audio, "TABSMITH_GUITARSET_PICKUP_AUDIO")
    if not annotation_dir or not mic_audio or not pickup_audio:
        parser.error(
            "--run/--validate-data requires annotations, mic audio and pickup audio paths via arguments or "
            "TABSMITH_GUITARSET_ANNOTATIONS/TABSMITH_GUITARSET_MIC_AUDIO/"
            "TABSMITH_GUITARSET_PICKUP_AUDIO"
        )
    audio_dirs = {"audio_mono-mic": mic_audio, "audio_mono-pickup_mix": pickup_audio}
    if args.validate_data and not args.run:
        print(json.dumps(
            validate_local_dataset(annotation_dir, audio_dirs, split_manifest),
            indent=2,
        ))
        return
    report = run_ablation(
        annotation_dir=annotation_dir,
        audio_dirs=audio_dirs,
        base_config=base_config,
        split_manifest=split_manifest,
        ablation_manifest=ablation_manifest,
        augmentation_manifest=augmentation_manifest,
        run_dir=Path(args.run_dir).resolve(),
        selected_candidates=set(args.candidate) if args.candidate else None,
        selected_folds=set(args.fold) if args.fold else None,
        epochs_override=args.epochs,
    )
    output = Path(args.output).resolve() if args.output else Path(args.run_dir).resolve() / "ablation-report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output, report)
    print(
        f"Temporal v2 ablation {report['status']}: "
        f"{report['completedRunCount']}/{report['expectedRunCount']} candidate-fold runs"
    )
    print(f"Report: {output}")
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
