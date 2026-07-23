"""Export a trained checkpoint to ONNX and verify agreement.

    python -m ml.exports.export_model --checkpoint ml/checkpoints/temporal-baseline-v0.pt

Verifies PyTorch vs ONNX Runtime outputs agree within tolerance and prints the
embedded, checksummed metadata. Requires the training extra. The exported model
is NOT bundled into the Electron installer in this pass.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..models.temporal_baseline import export_onnx
from ..training.checkpoint import load_checkpoint

ML_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ML_ROOT / "checkpoints" / "temporal-baseline-v0.pt"


def verify_onnx_agreement(model, onnx_path: str, input_dim: int, tolerance: float = 1e-4) -> dict:
    import onnxruntime as ort

    x = np.random.default_rng(0).standard_normal((1, 24, input_dim)).astype(np.float32)
    with torch.no_grad():
        torch_out = model(torch.from_numpy(x))
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"features": x})
    names = ["root", "quality", "nochord", "boundary"]
    diffs = {name: float(np.max(np.abs(torch_out[name].numpy() - onnx_out[i])))
             for i, name in enumerate(names)}
    return {"maxAbsDiff": diffs, "withinTolerance": all(v <= tolerance for v in diffs.values()),
            "tolerance": tolerance}


def main() -> int:
    parser = argparse.ArgumentParser(description="Export + verify the temporal-baseline model.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--onnx", default=None)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"No checkpoint at {checkpoint_path}. Run `python -m ml.training.train` first.")
        return 1

    model, metadata = load_checkpoint(checkpoint_path)
    input_dim = int(metadata.get("inputDim", 25))
    onnx_path = args.onnx or str(checkpoint_path.with_suffix(".onnx"))
    export_onnx(model, onnx_path, input_dim=input_dim)
    agreement = verify_onnx_agreement(model, onnx_path, input_dim)

    print("Model metadata:")
    print(json.dumps(metadata, indent=2))
    print(f"\nONNX export: {onnx_path}")
    print(f"PyTorch/ONNX agreement: {agreement}")
    return 0 if agreement["withinTolerance"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
