"""Experimental training smoke pass: python -m ml.training.train

Trains the compact TCN on *synthetic* data to validate the full lifecycle
(train -> save -> reload -> predict -> export -> evaluate). This is NOT a
production chord model and makes no accuracy claim about real music.

Requires the training extra (ml/requirements-training.txt).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .checkpoint import build_metadata, save_checkpoint
from .dataset import collate, compute_class_weights, make_samples, FrameSample
from .losses import combined_loss
from ..models.temporal_baseline import ModelConfig, TemporalBaseline, export_onnx
from ..preprocessing.synth_generator import build_split_tracks

ML_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ML_ROOT / "configs" / "temporal-baseline.json"
DEFAULT_CHECKPOINT = ML_ROOT / "checkpoints" / "temporal-baseline-v0.pt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _iterate_batches(samples: list[FrameSample], batch_size: int, rng: np.random.Generator):
    order = rng.permutation(len(samples))
    for start in range(0, len(samples), batch_size):
        yield collate([samples[i] for i in order[start:start + batch_size]])


def _epoch_loss(model, samples, weights, config, optimizer, rng, train: bool):
    model.train(train)
    batch_size = int(config["training"]["batchSize"])
    totals: dict[str, float] = {}
    count = 0
    for batch in _iterate_batches(samples, batch_size, rng):
        with torch.set_grad_enabled(train):
            outputs = model(batch["features"])
            loss, parts = combined_loss(outputs, batch, weights, config)
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"].get("gradClip", 5.0)))
            optimizer.step()
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    return {key: value / max(1, count) for key, value in totals.items()}


def run_training(config: dict, checkpoint_path: Path, quiet: bool = False) -> dict:
    seed = int(config.get("seed", 20260723))
    set_seed(seed)
    data = config["data"]
    tempo = tuple(data.get("tempoRange", [72, 140]))
    tolerance = float(config["labels"]["boundaryToleranceSeconds"])

    train_tracks = build_split_tracks("training", int(data["trainingTracks"]), float(data["durationSeconds"]), tempo)
    dev_tracks = build_split_tracks("development", int(data["developmentTracks"]), float(data["durationSeconds"]), tempo)
    train_samples = make_samples(train_tracks, tolerance)
    dev_samples = make_samples(dev_tracks, tolerance)
    weights = compute_class_weights(train_samples)

    model = TemporalBaseline(ModelConfig.from_dict(config))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learningRate"]),
                                 weight_decay=float(config["training"].get("weightDecay", 0.0)))

    epochs = int(config["training"]["epochs"])
    patience = int(config["training"].get("earlyStoppingPatience", 6))
    rng = np.random.default_rng(seed)
    history: list[dict] = []
    best_dev = float("inf")
    best_state = None
    best_epoch = -1
    stale = 0
    start_time = time.time()

    for epoch in range(epochs):
        train_metrics = _epoch_loss(model, train_samples, weights, config, optimizer, rng, train=True)
        with torch.no_grad():
            dev_metrics = _epoch_loss(model, dev_samples, weights, config, optimizer, rng, train=False)
        history.append({"epoch": epoch, "train": train_metrics, "dev": dev_metrics})
        if not quiet:
            print(f"epoch {epoch:02d} | train {train_metrics['total']:.4f} | dev {dev_metrics['total']:.4f} "
                  f"(root {dev_metrics['root']:.3f} qual {dev_metrics['quality']:.3f} "
                  f"nc {dev_metrics['nochord']:.3f} bnd {dev_metrics['boundary']:.3f})")
        if dev_metrics["total"] < best_dev - 1e-4:
            best_dev, best_state, best_epoch, stale = dev_metrics["total"], {k: v.clone() for k, v in model.state_dict().items()}, epoch, 0
        else:
            stale += 1
            if stale >= patience:
                if not quiet:
                    print(f"early stopping at epoch {epoch} (best dev {best_dev:.4f} @ {best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    train_seconds = time.time() - start_time

    manifest_hash = hashlib.sha256(json.dumps({
        "config": config, "trainTracks": len(train_samples), "devTracks": len(dev_samples),
    }, sort_keys=True).encode()).hexdigest()
    metadata = build_metadata(config, manifest_hash, seed, model.parameter_count(), model)
    save_checkpoint(checkpoint_path, model, config, metadata)

    history_path = checkpoint_path.with_suffix(".history.json")
    history_path.write_text(json.dumps({
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "seed": seed, "bestEpoch": best_epoch, "bestDevLoss": round(best_dev, 5),
        "epochsRun": len(history), "trainSeconds": round(train_seconds, 2),
        "parameterCount": model.parameter_count(), "history": history,
        "disclaimer": "Synthetic pipeline-validation only. Not representative of commercial music.",
    }, indent=2), encoding="utf-8")

    onnx_path = checkpoint_path.with_suffix(".onnx")
    export_onnx(model, str(onnx_path), input_dim=int(config["features"]["inputDim"]))

    return {
        "checkpoint": str(checkpoint_path),
        "onnx": str(onnx_path),
        "parameterCount": model.parameter_count(),
        "checkpointBytes": checkpoint_path.stat().st_size,
        "onnxBytes": onnx_path.stat().st_size,
        "trainSeconds": round(train_seconds, 2),
        "bestDevLoss": round(best_dev, 5),
        "bestEpoch": best_epoch,
        "epochsRun": len(history),
        "trainTracks": len(train_samples),
        "devTracks": len(dev_samples),
        "checksum": metadata["checksum"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental temporal-baseline training smoke pass.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    summary = run_training(config, Path(args.checkpoint))
    print("\n" + "=" * 60)
    print("EXPERIMENTAL smoke training complete (synthetic data only).")
    print("No production-quality learned chord model has been created.")
    print("=" * 60)
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
