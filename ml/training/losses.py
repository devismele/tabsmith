"""Combined multi-task loss for the temporal baseline.

    total = root CE + quality CE + w_nc · no-chord BCE + w_b · boundary BCE

Root/quality are scored only on frames that carry a chord. Boundary frames are
rare, so a positive class weight (``boundaryPosWeight``) makes "never predict a
change" a losing strategy.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def combined_loss(outputs: dict, targets: dict, weights: dict, config: dict) -> tuple[torch.Tensor, dict]:
    training = config.get("training", {})
    pad = targets["pad_mask"]
    chord_mask = pad * (1.0 - targets["nochord"])
    pad_sum = pad.sum().clamp(min=1.0)
    chord_sum = chord_mask.sum().clamp(min=1.0)

    root_ce = F.cross_entropy(
        outputs["root"].reshape(-1, outputs["root"].shape[-1]),
        targets["root"].reshape(-1).clamp(min=0),
        weight=weights["root"].to(outputs["root"].device),
        reduction="none",
    )
    root_ce = (root_ce * chord_mask.reshape(-1)).sum() / chord_sum

    quality_ce = F.cross_entropy(
        outputs["quality"].reshape(-1, outputs["quality"].shape[-1]),
        targets["quality"].reshape(-1).clamp(min=0),
        weight=weights["quality"].to(outputs["quality"].device),
        reduction="none",
    )
    quality_ce = (quality_ce * chord_mask.reshape(-1)).sum() / chord_sum

    nochord_bce = F.binary_cross_entropy_with_logits(outputs["nochord"], targets["nochord"], reduction="none")
    nochord_bce = (nochord_bce * pad).sum() / pad_sum

    pos_weight = torch.tensor(float(training.get("boundaryPosWeight", 8.0)), device=outputs["boundary"].device)
    boundary_bce = F.binary_cross_entropy_with_logits(
        outputs["boundary"], targets["boundary"], pos_weight=pos_weight, reduction="none")
    boundary_bce = (boundary_bce * pad).sum() / pad_sum

    w_nc = float(training.get("noChordLossWeight", 1.0))
    w_b = float(training.get("boundaryLossWeight", 2.0))
    total = root_ce + quality_ce + w_nc * nochord_bce + w_b * boundary_bce
    return total, {
        "root": float(root_ce.detach()),
        "quality": float(quality_ce.detach()),
        "nochord": float(nochord_bce.detach()),
        "boundary": float(boundary_bce.detach()),
        "total": float(total.detach()),
    }
