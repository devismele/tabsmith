"""Compact dilated TCN for learned-harmony-v1 (experimental smoke model).

Deliberately small: a few dilated Conv1d blocks with residual connections and
four output heads. Chosen over a Transformer for fast CPU training, easy
debugging, straightforward ONNX export, and low memory — suitable for later
Windows ARM64 / x64 inference. This is a *pipeline-validation* model, not a
production chord recogniser.

Requires torch (ml/requirements-training.txt). Nothing in the foundation pass
imports this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..schema import pitch_class_name

# Output vocabulary v0.
ROOTS = 12
QUALITIES = ("maj", "min", "7")


@dataclass
class ModelConfig:
    input_dim: int = 25
    channels: int = 48
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.1

    @staticmethod
    def from_dict(data: dict) -> "ModelConfig":
        model = data.get("model", {})
        return ModelConfig(
            input_dim=int(data.get("features", {}).get("inputDim", 25)),
            channels=int(model.get("channels", 48)),
            kernel_size=int(model.get("kernelSize", 3)),
            dilations=tuple(model.get("dilations", (1, 2, 4, 8))),
            dropout=float(model.get("dropout", 0.1)),
        )

    def receptive_field_frames(self) -> int:
        """Symmetric temporal context covered by the two convolutions per block."""
        return 1 + 2 * (self.kernel_size - 1) * sum(self.dilations)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2  # symmetric -> length preserved
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dropout(F.relu(self.norm1(self.conv1(x))))
        x = self.dropout(F.relu(self.norm2(self.conv2(x))))
        return x + residual


class TemporalBaseline(nn.Module):
    """Input (B, T, F) -> per-frame root/quality/no-chord/boundary logits."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Conv1d(config.input_dim, config.channels, 1)
        self.blocks = nn.ModuleList(
            _ResidualBlock(config.channels, config.kernel_size, d, config.dropout)
            for d in config.dilations
        )
        self.root_head = nn.Conv1d(config.channels, ROOTS, 1)
        self.quality_head = nn.Conv1d(config.channels, len(QUALITIES), 1)
        self.nochord_head = nn.Conv1d(config.channels, 1, 1)
        self.boundary_head = nn.Conv1d(config.channels, 1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.input_proj(x.transpose(1, 2))  # (B, C, T)
        for block in self.blocks:
            h = block(h)
        return {
            "root": self.root_head(h).transpose(1, 2),        # (B, T, 12)
            "quality": self.quality_head(h).transpose(1, 2),  # (B, T, 3)
            "nochord": self.nochord_head(h).transpose(1, 2).squeeze(-1),   # (B, T)
            "boundary": self.boundary_head(h).transpose(1, 2).squeeze(-1),  # (B, T)
        }

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class _ExportWrapper(nn.Module):
    """Flattens the dict output to a stable tuple for ONNX export."""

    def __init__(self, model: TemporalBaseline):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        return out["root"], out["quality"], out["nochord"], out["boundary"]


def export_onnx(model: TemporalBaseline, path: str, input_dim: int, opset: int = 17) -> str:
    was_training = model.training
    model.eval()
    dummy = torch.zeros(1, 8, input_dim, dtype=torch.float32)
    # Use the legacy TorchScript exporter (dynamo=False) so export does not pull
    # in onnxscript; keeps the training extra minimal (torch/onnx/onnxruntime).
    torch.onnx.export(
        _ExportWrapper(model),
        dummy,
        path,
        input_names=["features"],
        output_names=["root", "quality", "nochord", "boundary"],
        dynamic_axes={"features": {0: "batch", 1: "time"},
                      "root": {0: "batch", 1: "time"},
                      "quality": {0: "batch", 1: "time"},
                      "nochord": {0: "batch", 1: "time"},
                      "boundary": {0: "batch", 1: "time"}},
        opset_version=opset,
        dynamo=False,
    )
    # The legacy exporter leaves modules in training mode; restore prior state so
    # later inference forwards are not silently using BatchNorm batch statistics.
    model.train(was_training)
    return path


def quality_token(index: int) -> str:
    return QUALITIES[index]


def frame_labels_to_regions(
    times,
    root_idx,
    quality_idx,
    nochord_prob,
    hop_seconds: float,
    min_region_seconds: float = 0.4,
):
    """Greedy frame -> region decoding (argmax + merge)."""
    from ..schema import ChordRegion
    labels = []
    for i in range(len(times)):
        if nochord_prob[i] > 0.5:
            labels.append("N")
        else:
            labels.append(f"{pitch_class_name(int(root_idx[i]))}:{quality_token(int(quality_idx[i]))}")
    regions: list = []
    if not labels:
        return regions
    start_i = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start_i]:
            start = float(times[start_i]) - hop_seconds / 2
            end = float(times[i - 1]) + hop_seconds / 2
            regions.append(ChordRegion(round(max(0.0, start), 4), round(end, 4), labels[start_i]))
            start_i = i
    # Absorb sub-minimum flicker into the previous region.
    merged = [regions[0]]
    for region in regions[1:]:
        prev = merged[-1]
        if region.label == prev.label or region.duration() < min_region_seconds:
            prev.end = region.end
        else:
            merged.append(region)
    return merged
