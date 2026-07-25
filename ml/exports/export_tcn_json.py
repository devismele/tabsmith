"""Export a trained temporal-baseline checkpoint to JSON weights for the pure-TS
provider: python -m ml.exports.export_tcn_json --checkpoint <pt> --out <json>

The app runs the learned model without an ONNX runtime by reimplementing this
compact TCN's forward pass in TypeScript (src/learnedHarmony/onnxProvider.ts).
This dumps every weight plus a deterministic (input, logits) sample so the TS
port can be verified numerically identical to PyTorch. Requires the training
extra (torch).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..models.temporal_baseline import QUALITIES
from ..training.checkpoint import load_checkpoint

ML_ROOT = Path(__file__).resolve().parents[1]


def _bn(sd: dict, prefix: str) -> dict:
    return {
        "gamma": sd[f"{prefix}.weight"].tolist(),
        "beta": sd[f"{prefix}.bias"].tolist(),
        "mean": sd[f"{prefix}.running_mean"].tolist(),
        "var": sd[f"{prefix}.running_var"].tolist(),
        "eps": 1e-5,
    }


def _conv(sd: dict, prefix: str, squeeze_kernel: bool) -> dict:
    weight = sd[f"{prefix}.weight"]           # (out, in, k)
    if squeeze_kernel:
        weight = weight[:, :, 0]              # 1x1 conv -> (out, in)
    return {"weight": weight.tolist(), "bias": sd[f"{prefix}.bias"].tolist()}


def serialize_model(model, metadata: dict, sample_frames: int = 20) -> dict:
    """TCN weights + a deterministic (input, logits) parity sample as a JSON dict."""
    model.eval()
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    config = model.config

    blocks = []
    for i in range(len(config.dilations)):
        blocks.append({
            "dilation": int(config.dilations[i]),
            "conv1": _conv(sd, f"blocks.{i}.conv1", squeeze_kernel=False),
            "conv2": _conv(sd, f"blocks.{i}.conv2", squeeze_kernel=False),
            "norm1": _bn(sd, f"blocks.{i}.norm1"),
            "norm2": _bn(sd, f"blocks.{i}.norm2"),
        })

    # Deterministic parity sample: fixed random (T, inputDim) -> raw head logits.
    rng = np.random.default_rng(7)
    sample_in = rng.standard_normal((sample_frames, config.input_dim)).astype(np.float32)
    with torch.no_grad():
        out = model(torch.from_numpy(sample_in).unsqueeze(0))
    sample = {
        "input": sample_in.tolist(),
        "root": out["root"][0].numpy().tolist(),
        "quality": out["quality"][0].numpy().tolist(),
        "nochord": out["nochord"][0].numpy().tolist(),
        "boundary": out["boundary"][0].numpy().tolist(),
    }

    payload = {
        "modelVersion": metadata.get("modelVersion", "temporal-baseline"),
        "modelChecksum": metadata.get("checksum", ""),
        "featureVersion": metadata.get("featureVersion", "harmony-features-v1"),
        "config": {
            "inputDim": config.input_dim,
            "channels": config.channels,
            "kernelSize": config.kernel_size,
            "dilations": list(config.dilations),
            "qualities": list(QUALITIES),
        },
        "inputProj": _conv(sd, "input_proj", squeeze_kernel=True),
        "blocks": blocks,
        "heads": {
            "root": _conv(sd, "root_head", squeeze_kernel=True),
            "quality": _conv(sd, "quality_head", squeeze_kernel=True),
            "nochord": _conv(sd, "nochord_head", squeeze_kernel=True),
            "boundary": _conv(sd, "boundary_head", squeeze_kernel=True),
        },
        "paritySample": sample,
    }
    return payload


def export(checkpoint_path: Path, out_path: Path) -> dict:
    model, metadata = load_checkpoint(checkpoint_path)
    payload = serialize_model(model, metadata)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a TCN checkpoint to JSON weights for the TS provider.")
    parser.add_argument("--checkpoint", default=str(ML_ROOT / "checkpoints" / "temporal-baseline-app-v0.pt"))
    parser.add_argument("--out", default=str(ML_ROOT / "exports" / "temporal-baseline-app-v0.weights.json"))
    args = parser.parse_args()
    payload = export(Path(args.checkpoint), Path(args.out))
    size = Path(args.out).stat().st_size
    print(f"Exported {payload['modelVersion']} ({payload['config']['channels']}ch, "
          f"{len(payload['blocks'])} blocks) -> {args.out} ({size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
