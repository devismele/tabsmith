"""Checkpoint save/reload with embedded, checksummed metadata (torch)."""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from ..models.temporal_baseline import QUALITIES, ROOTS, ModelConfig, TemporalBaseline

ML_ROOT = Path(__file__).resolve().parents[1]


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ML_ROOT.parent,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def state_dict_checksum(model: torch.nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def build_metadata(config: dict, dataset_manifest_hash: str, seed: int,
                   param_count: int, model: TemporalBaseline) -> dict:
    return {
        "modelVersion": config.get("modelName", "temporal-baseline-v0"),
        "featureVersion": config.get("features", {}).get("pipelineVersion", "numpy-chroma-v1"),
        "trainingConfig": config,
        "datasetManifestHash": dataset_manifest_hash,
        "trainingSeed": seed,
        "chordVocabulary": {"roots": ROOTS, "qualities": list(QUALITIES), "noChord": True},
        "inputDim": config.get("features", {}).get("inputDim", 25),
        "gitCommit": git_commit(),
        "parameterCount": param_count,
        "checksum": state_dict_checksum(model),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "disclaimer": config.get(
            "artifactDisclaimer",
            "Experimental research model. Not approved for production or representative full-band use.",
        ),
    }


def save_checkpoint(path: str | Path, model: TemporalBaseline, config: dict, metadata: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_config": config.get("model", {}),
                "features": config.get("features", {}), "metadata": metadata}, path)
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def load_checkpoint(path: str | Path) -> tuple[TemporalBaseline, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = {"model": checkpoint.get("model_config", {}), "features": checkpoint.get("features", {})}
    model = TemporalBaseline(ModelConfig.from_dict(config))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint.get("metadata", {})
