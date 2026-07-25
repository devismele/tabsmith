"""Generate a small TCN parity fixture for the TypeScript provider test:
python -m ml.exports.make_parity_fixture

Builds a tiny (deterministic) temporal-baseline, serializes its weights and a
sample (input -> PyTorch head logits), and writes a small JSON the vitest test
(tests/learnedHarmony.tcn.test.ts) uses to prove the pure-TS forward pass
(src/learnedHarmony/onnxProvider.ts) is numerically identical to PyTorch —
without committing a multi-MB trained model. Requires torch.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .export_tcn_json import serialize_model
from ..models.temporal_baseline import ModelConfig, TemporalBaseline

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "src" / "learnedHarmony" / "__fixtures__" / "tcn-parity.json"


def main() -> None:
    torch.manual_seed(20260725)
    # Tiny but exercises every layer type: input_dim 25 (real), 2 dilated blocks,
    # and non-default BatchNorm stats so the eval-mode normalization is non-trivial.
    config = ModelConfig(input_dim=25, channels=4, kernel_size=3, dilations=(1, 2), dropout=0.0)
    model = TemporalBaseline(config)
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            module.running_mean = torch.randn_like(module.running_mean)
            module.running_var = torch.rand_like(module.running_var) + 0.5  # positive
    metadata = {"modelVersion": "tcn-parity-fixture", "checksum": "fixture", "featureVersion": "harmony-features-v1"}
    payload = serialize_model(model, metadata, sample_frames=6)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Wrote parity fixture -> {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
