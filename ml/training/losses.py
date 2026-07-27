"""Multi-task and sequence-aware losses for temporal chord models.

The v1 objective is preserved when all new temporal weights are omitted:

    root CE + quality CE + w_nc * no-chord BCE + w_b * soft-boundary BCE

Temporal-harmony-v2 adds opt-in terms that train distributions, not argmax
labels: within-chord consistency, supervised chord-state switching, an exact
change target for the boundary head, and agreement between identity switches
and the boundary head.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def chord_state_probabilities(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return a normalized 37-state distribution (12 roots x 3 qualities + N)."""
    root = torch.softmax(outputs["root"], dim=-1)
    quality = torch.softmax(outputs["quality"], dim=-1)
    no_chord = torch.sigmoid(outputs["nochord"]).unsqueeze(-1)
    chord = (
        root.unsqueeze(-1)
        * quality.unsqueeze(-2)
        * (1.0 - no_chord).unsqueeze(-1)
    ).flatten(start_dim=-2)
    return torch.cat([chord, no_chord], dim=-1)


def predicted_switch_probability(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Total-variation change between adjacent chord-state distributions.

    Identical uncertain distributions yield zero rather than being mistaken for
    a switch; a complete move between disjoint chord states yields one.
    """
    states = chord_state_probabilities(outputs)
    if states.shape[1] < 2:
        return states.new_zeros((states.shape[0], 0))
    change = 0.5 * (states[:, 1:] - states[:, :-1]).abs().sum(dim=-1)
    return change.clamp(min=1e-6, max=1.0 - 1e-6)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def focal_modulation(logits: torch.Tensor, targets: torch.Tensor, gamma: float) -> torch.Tensor:
    """``(1 - p_t) ** gamma`` for a soft-target binary problem.

    ``p_t`` is the probability assigned to the *observed* target mass, so a
    frame the head already predicts well is down-weighted and the ambiguous
    mid-range keeps its gradient. Soft boundary targets interpolate between the
    positive and negative branch, which keeps the triangular tolerance window
    meaningful instead of collapsing it to a hard label.

    ``gamma <= 0`` returns ones, so omitting it leaves the objective unchanged.
    """
    if gamma <= 0.0:
        return torch.ones_like(logits)
    probability = torch.sigmoid(logits)
    p_t = targets * probability + (1.0 - targets) * (1.0 - probability)
    return (1.0 - p_t).clamp(min=0.0, max=1.0).pow(gamma)


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

    nochord_bce = F.binary_cross_entropy_with_logits(
        outputs["nochord"], targets["nochord"], reduction="none")
    nochord_bce = (nochord_bce * pad).sum() / pad_sum

    pos_weight = torch.tensor(
        float(training.get("boundaryPosWeight", 8.0)),
        device=outputs["boundary"].device,
    )
    boundary_gamma = float(training.get("boundaryFocalGamma", 0.0))
    boundary_bce = F.binary_cross_entropy_with_logits(
        outputs["boundary"], targets["boundary"], pos_weight=pos_weight, reduction="none")
    boundary_bce = boundary_bce * focal_modulation(
        outputs["boundary"], targets["boundary"], boundary_gamma)
    boundary_bce = (boundary_bce * pad).sum() / pad_sum

    zero = outputs["root"].new_zeros(())
    duration_consistency = zero
    switch_loss = zero
    hard_boundary_bce = zero
    boundary_switch_agreement = zero
    if outputs["root"].shape[1] > 1:
        transition_mask = pad[:, 1:] * pad[:, :-1]
        states = chord_state_probabilities(outputs)
        switch_probability = predicted_switch_probability(outputs)

        # Do not suppress a real change: consistency applies only where the
        # reference chord persists across adjacent frames.
        stable_mask = targets.get("stable", 1.0 - targets["change"])[:, 1:] * transition_mask
        state_delta = (states[:, 1:] - states[:, :-1]).square().sum(dim=-1)
        duration_consistency = _masked_mean(state_delta, stable_mask)

        switch_loss = _masked_mean(
            F.binary_cross_entropy(
                switch_probability,
                targets["change"][:, 1:],
                reduction="none",
            ),
            transition_mask,
        )

        hard_pos_weight = torch.tensor(
            float(training.get("hardBoundaryPosWeight", training.get("boundaryPosWeight", 8.0))),
            device=outputs["boundary"].device,
        )
        hard_boundary = F.binary_cross_entropy_with_logits(
            outputs["boundary"][:, 1:],
            targets["change"][:, 1:],
            pos_weight=hard_pos_weight,
            reduction="none",
        )
        hard_boundary = hard_boundary * focal_modulation(
            outputs["boundary"][:, 1:], targets["change"][:, 1:],
            float(training.get("hardBoundaryFocalGamma", boundary_gamma)),
        )
        hard_boundary_bce = _masked_mean(hard_boundary, transition_mask)

        boundary_probability = torch.sigmoid(outputs["boundary"][:, 1:])
        boundary_switch_agreement = _masked_mean(
            (boundary_probability - switch_probability).square(),
            transition_mask,
        )

    w_nc = float(training.get("noChordLossWeight", 1.0))
    w_b = float(training.get("boundaryLossWeight", 2.0))
    w_duration = float(training.get("durationConsistencyLossWeight", 0.0))
    w_switch = float(training.get("switchLossWeight", 0.0))
    w_hard_boundary = float(training.get("hardBoundaryLossWeight", 0.0))
    w_agreement = float(training.get("boundarySwitchAgreementLossWeight", 0.0))
    total = (
        root_ce
        + quality_ce
        + w_nc * nochord_bce
        + w_b * boundary_bce
        + w_duration * duration_consistency
        + w_switch * switch_loss
        + w_hard_boundary * hard_boundary_bce
        + w_agreement * boundary_switch_agreement
    )
    return total, {
        "root": float(root_ce.detach()),
        "quality": float(quality_ce.detach()),
        "nochord": float(nochord_bce.detach()),
        "boundary": float(boundary_bce.detach()),
        "durationConsistency": float(duration_consistency.detach()),
        "switch": float(switch_loss.detach()),
        "hardBoundary": float(hard_boundary_bce.detach()),
        "boundarySwitchAgreement": float(boundary_switch_agreement.detach()),
        "total": float(total.detach()),
    }
