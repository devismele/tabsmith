"""Resumable CPU training for the frozen boundary-calibration candidates.

Reuses the temporal-v2 harness internals that are already tested (track
loading, feature extraction, sample assembly, ``fit_samples``, checkpointing)
but keeps its own manifest schema and its own protocol guards, so the
temporal-v2 validator -- which mandates a ``v1-objective`` first candidate --
is left untouched rather than relaxed.

Protocol guards enforced here, not assumed:

* only p01-p05 are ever loaded; a p00 track anywhere aborts the run
* microphone and pickup captures of a performance stay in the same fold
* candidates declaring ``reusesCheckpoint`` are never trained
* a completed (fold, candidate) run is skipped only when its recorded identity
  -- config checksum, seed, epoch override -- matches exactly, so a changed
  candidate can never silently resume onto a stale checkpoint

One fold-candidate pair trains at a time; the loop is restart-safe at that
granularity and writes a status file before each run.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..temporal_v2_ablation import (
    DEFAULT_AUGMENTATIONS,
    DEFAULT_CONFIG,
    DEFAULT_SPLITS,
    FORBIDDEN_SELECTION_PERFORMER,
    _base_track_id,
    _capture,
    _development_samples,
    _extract_feature_map,
    _load_completed_run,
    _load_tracks,
    _portable_checksum,
    _python_onnx_parity,
    _read_json,
    _resolve_data_path,
    _training_samples,
    _typescript_parity,
    _write_json_atomic,
    candidate_config,
)

ML_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATES = ML_ROOT / "configs" / "boundary-calibration-v3-candidates.json"
DEFAULT_RUN_DIR = ML_ROOT / "runs" / "boundary-calibration-v3"


def validate_candidate_manifest(manifest: dict[str, Any]) -> None:
    candidates = manifest.get("candidates", [])
    if not candidates:
        raise ValueError("no candidates defined")
    if len(candidates) > 5:
        raise ValueError("bounded study: at most five candidates including the control")
    ids = [c["id"] for c in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate ids must be unique")
    constraints = manifest.get("constraints", {})
    if constraints.get("p00UsedForSelection", True):
        raise ValueError("manifest must forbid p00 for selection")
    if constraints.get("productionMutation", True):
        raise ValueError("manifest must forbid production mutation")
    if FORBIDDEN_SELECTION_PERFORMER in constraints.get("modelSelectionPerformers", []):
        raise ValueError("p00 must not appear in the model-selection performers")


def trainable_candidates(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Candidates that actually need a training run (not checkpoint reusers)."""
    return [c for c in manifest["candidates"] if not c.get("reusesCheckpoint")]


def train(
    *,
    annotation_dir: Path,
    audio_dirs: dict[str, Path],
    base_config: dict[str, Any],
    split_manifest: dict[str, Any],
    candidate_manifest: dict[str, Any],
    augmentation_manifest: dict[str, Any],
    run_dir: Path,
    selected_candidates: set[str] | None = None,
    selected_folds: set[str] | None = None,
    epochs_override: int | None = None,
) -> dict[str, Any]:
    from ...training.checkpoint import load_checkpoint
    from ...training.train import fit_samples

    validate_candidate_manifest(candidate_manifest)
    tracks = _load_tracks(annotation_dir, audio_dirs)
    allowed = {item["performerId"] for item in split_manifest["modelSelectionPerformers"]}
    if FORBIDDEN_SELECTION_PERFORMER in allowed:
        raise AssertionError("p00 present in model-selection performers")
    tracks = [t for t in tracks if t.artist in allowed]
    if any(t.artist == FORBIDDEN_SELECTION_PERFORMER for t in tracks):
        raise AssertionError("p00 entered boundary-calibration training")
    if not tracks:
        raise ValueError("zero model-selection tracks loaded")

    # Capture pairing: every performance must contribute both captures.
    by_performance: dict[str, set[str]] = {}
    for t in tracks:
        by_performance.setdefault(_base_track_id(t), set()).add(_capture(t))
    unpaired = sorted(p for p, caps in by_performance.items() if len(caps) != 2)
    if unpaired:
        raise ValueError(f"unpaired captures for {len(unpaired)} performances, e.g. {unpaired[:3]}")

    feature_map = _extract_feature_map(tracks, base_config["features"]["pipelineVersion"])
    run_dir.mkdir(parents=True, exist_ok=True)

    candidates = trainable_candidates(candidate_manifest)
    if selected_candidates:
        candidates = [c for c in candidates if c["id"] in selected_candidates]

    runs: list[dict[str, Any]] = []
    started = time.time()
    for fold in split_manifest["folds"]:
        if selected_folds and fold["foldId"] not in selected_folds:
            continue
        train_tracks = [t for t in tracks if t.artist in set(fold["trainPerformers"])]
        validation_tracks = [t for t in tracks if t.artist in set(fold["validationPerformers"])]
        if not train_tracks or not validation_tracks:
            raise ValueError(f"{fold['foldId']}: zero train or validation tracks")
        if set(fold["trainPerformers"]) & set(fold["validationPerformers"]):
            raise AssertionError(f"{fold['foldId']}: train/validation performer overlap")

        for candidate in candidates:
            config = candidate_config(base_config, candidate)
            config["modelName"] = f"boundary-calibration-v3-{candidate['id']}"
            config["splitIdentity"] = {
                "splitId": split_manifest["splitId"],
                "foldId": fold["foldId"],
                "trainPerformers": fold["trainPerformers"],
                "validationPerformers": fold["validationPerformers"],
            }
            use_augmentation = bool(candidate.get("useAugmentation", False))
            config["augmentationIdentity"] = (
                {"augmentationId": augmentation_manifest["augmentationId"],
                 "manifestChecksum": _portable_checksum(augmentation_manifest)}
                if use_augmentation else None
            )
            if epochs_override is not None:
                config["training"]["epochs"] = epochs_override

            checkpoint_dir = run_dir / fold["foldId"] / candidate["id"]
            result_path = checkpoint_dir / "run-result.json"
            run_identity = {
                "foldId": fold["foldId"],
                "candidateId": candidate["id"],
                "trainingConfigChecksum": _portable_checksum(config),
                "seed": int(config["seed"]),
                "epochsOverride": epochs_override,
            }
            completed = _load_completed_run(result_path, run_identity)
            if completed is not None:
                print(f"resume: {fold['foldId']}/{candidate['id']} already completed", flush=True)
                runs.append(completed)
                continue

            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            _write_json_atomic(checkpoint_dir / "run-status.json", {
                "schemaVersion": 1, "status": "running", "runIdentity": run_identity,
                "startedAt": datetime.now(timezone.utc).isoformat(),
            })
            print(f"run: {fold['foldId']}/{candidate['id']}", flush=True)
            run_started = time.time()
            train_samples = _training_samples(
                train_tracks, feature_map, config,
                augmentation_manifest if use_augmentation else None)
            dev_samples = _development_samples(validation_tracks, feature_map, config)
            checkpoint_path = checkpoint_dir / "model.pt"
            history = fit_samples(config, train_samples, dev_samples, checkpoint_path, quiet=False)
            elapsed = time.time() - run_started

            # Export parity is part of every run, not only the final study: a
            # candidate that cannot round-trip to ONNX/TypeScript is ineligible
            # regardless of its metrics, and finding that out now is cheap.
            model, metadata = load_checkpoint(checkpoint_path)
            first_features = feature_map[validation_tracks[0].track_id].stacked()
            python_parity = _python_onnx_parity(
                model, checkpoint_path.with_suffix(".onnx"), first_features)
            typescript_parity = _typescript_parity(
                checkpoint_path, checkpoint_dir / "model.weights.json")
            if not python_parity["passed"] or not typescript_parity["passed"]:
                raise RuntimeError(f"{fold['foldId']}/{candidate['id']}: export parity failed")

            result = {
                "schemaVersion": 1,
                "status": "completed",
                "runIdentity": run_identity,
                "result": {
                    "candidateId": candidate["id"],
                    "foldId": fold["foldId"],
                    "trainSamples": len(train_samples),
                    "developmentSamples": len(dev_samples),
                    "bestEpoch": history.get("bestEpoch"),
                    "bestDevLoss": history.get("bestDevLoss"),
                    "epochsRun": history.get("epochsRun"),
                    "trainSeconds": round(elapsed, 2),
                    "stateDictChecksum": metadata["checksum"],
                    "parameterCount": history.get("parameterCount"),
                    "epochsOverride": epochs_override,
                    "parity": {"pythonOnnx": python_parity, "typescript": typescript_parity},
                },
            }
            _write_json_atomic(result_path, result)
            _write_json_atomic(checkpoint_dir / "run-status.json", {
                "schemaVersion": 1, "status": "completed", "runIdentity": run_identity,
                "finishedAt": datetime.now(timezone.utc).isoformat(),
            })
            print(f"done: {fold['foldId']}/{candidate['id']} in {elapsed / 60:.1f} min "
                  f"(best epoch {history.get('bestEpoch')}, dev {history.get('bestDevLoss')})",
                  flush=True)
            runs.append(result)

    return {
        "schemaVersion": 1,
        "candidateSetId": candidate_manifest["ablationId"],
        "candidateManifestChecksum": _portable_checksum(candidate_manifest),
        "splitId": split_manifest["splitId"],
        "epochsOverride": epochs_override,
        "validForSelection": epochs_override is None,
        "p00UsedForSelection": False,
        "runs": runs,
        "elapsedSeconds": round(time.time() - started, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the frozen boundary-calibration candidates.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS))
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--augmentations", default=str(DEFAULT_AUGMENTATIONS))
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--mic-audio", default=None)
    parser.add_argument("--pickup-audio", default=None)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--candidate", action="append", default=None)
    parser.add_argument("--fold", action="append", default=None)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Smoke-only override; the report is marked invalid for selection.")
    args = parser.parse_args()

    annotation_dir = _resolve_data_path(args.annotations, "TABSMITH_GUITARSET_ANNOTATIONS")
    mic = _resolve_data_path(args.mic_audio, "TABSMITH_GUITARSET_MIC_AUDIO")
    pickup = _resolve_data_path(args.pickup_audio, "TABSMITH_GUITARSET_PICKUP_AUDIO")
    if not annotation_dir or not mic or not pickup:
        parser.error("requires annotation/mic/pickup paths via args or TABSMITH_GUITARSET_* env vars")

    report = train(
        annotation_dir=annotation_dir,
        audio_dirs={"audio_mono-mic": mic, "audio_mono-pickup_mix": pickup},
        base_config=_read_json(args.config),
        split_manifest=_read_json(args.splits),
        candidate_manifest=_read_json(args.candidates),
        augmentation_manifest=_read_json(args.augmentations),
        run_dir=Path(args.run_dir).resolve(),
        selected_candidates=set(args.candidate) if args.candidate else None,
        selected_folds=set(args.fold) if args.fold else None,
        epochs_override=args.epochs,
    )
    output = Path(args.run_dir).resolve() / "training-report.json"
    _write_json_atomic(output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
