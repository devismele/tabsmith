"""Separate boundary-retraining effects from decoder and chord-posterior effects.

A candidate in this study changes the *model*, and every model is then read by
two decoders. A raw "candidate beat the control" number therefore conflates
three different things, and this branch's whole question is which one moved.

The 2x2 (control vs candidate) x (full-v2 decode vs segmental-full) design
supports a clean decomposition:

* **decoder effect**   M(control, segmental) - M(control, existing)
      what the decoder buys on an unchanged model
* **model effect**     M(candidate, D) - M(control, D)   for each decoder D
      what retraining buys under a fixed decoder
* **interaction**      model effect under segmental - model effect under existing
      whether retraining and the segmental decoder help each other, which is
      the only term that can be credited to boundary/decoder *cooperation*

Because the full-v2 decode ignores the boundary head entirely, the model effect
measured under it cannot come from the boundary head at all -- it is pure
chord-posterior movement. That is what makes the attribution possible:

    model effect under full-v2 decode        -> chord-posterior channel
    interaction (segmental minus full-v2)    -> boundary channel

So an intervention that was *aimed* at the boundary head but shows its gain in
the full-v2 column has in fact moved the chord posterior, and should not be
described as a boundary result. The chord-channel diagnostics below (root and
detailed accuracy, posterior margin, top-1 agreement with the control) make
that visible directly rather than by inference.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..segmental.seg_metrics import CAPTURES

EXISTING = "full-v2-existing"
SEGMENTAL = "segmental-full"

# Lower-is-better metrics, so a negative delta is an improvement.
LOWER_IS_BETTER = {
    "fragmentationRate", "regionsPerMinute", "meanAbsoluteBoundaryErrorMs",
    "predictedRegionCount", "abaCount", "flickerCount",
}
ATTRIBUTED_METRICS = (
    "rootAccuracy", "detailedAccuracy", "fragmentationRate",
    "regionsPerMinute", "meanAbsoluteBoundaryErrorMs", "flickerCount",
)


def _delta(candidate: float, reference: float, metric: str) -> float:
    """Signed improvement: positive always means better."""
    raw = candidate - reference
    return -raw if metric in LOWER_IS_BETTER else raw


def decompose(control: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Decompose one candidate's effect against the control, per capture.

    ``control`` and ``candidate`` are the ``decoders`` maps produced by
    :func:`ml.evaluation.boundary.evaluate.evaluate_candidate`.
    """
    out: dict[str, Any] = {}
    for capture in CAPTURES:
        control_existing = control[EXISTING]["byCapture"][capture]
        control_segmental = control[SEGMENTAL]["byCapture"][capture]
        candidate_existing = candidate[EXISTING]["byCapture"][capture]
        candidate_segmental = candidate[SEGMENTAL]["byCapture"][capture]

        rows: dict[str, Any] = {}
        for metric in ATTRIBUTED_METRICS:
            decoder_effect = _delta(control_segmental[metric], control_existing[metric], metric)
            model_existing = _delta(candidate_existing[metric], control_existing[metric], metric)
            model_segmental = _delta(candidate_segmental[metric], control_segmental[metric], metric)
            rows[metric] = {
                "controlExisting": round(float(control_existing[metric]), 4),
                "controlSegmental": round(float(control_segmental[metric]), 4),
                "candidateExisting": round(float(candidate_existing[metric]), 4),
                "candidateSegmental": round(float(candidate_segmental[metric]), 4),
                "decoderEffect": round(decoder_effect, 4),
                "modelEffectUnderExisting": round(model_existing, 4),
                "modelEffectUnderSegmental": round(model_segmental, 4),
                "interaction": round(model_segmental - model_existing, 4),
                # The full-v2 decode never reads the boundary head, so any model
                # effect visible there is chord-posterior movement by definition.
                "chordPosteriorChannel": round(model_existing, 4),
                "boundaryChannel": round(model_segmental - model_existing, 4),
            }
        out[capture] = rows
    return out


def boundary_channel_summary(control: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Did the boundary head itself actually change, and by how much?"""
    out: dict[str, Any] = {}
    for capture in CAPTURES:
        control_bq = control[SEGMENTAL]["boundaryQuality"][capture]
        candidate_bq = candidate[SEGMENTAL]["boundaryQuality"][capture]
        out[capture] = {
            "precision": {"control": control_bq["precision"], "candidate": candidate_bq["precision"],
                          "delta": round(candidate_bq["precision"] - control_bq["precision"], 4)},
            "recall": {"control": control_bq["recall"], "candidate": candidate_bq["recall"],
                       "delta": round(candidate_bq["recall"] - control_bq["recall"], 4)},
            "f1": {"control": control_bq["f1"], "candidate": candidate_bq["f1"],
                   "delta": round(candidate_bq["f1"] - control_bq["f1"], 4)},
            "expectedCalibrationError": {
                "control": control_bq["expectedCalibrationError"],
                "candidate": candidate_bq["expectedCalibrationError"],
                "delta": round(candidate_bq["expectedCalibrationError"]
                               - control_bq["expectedCalibrationError"], 6)},
            "stateChangeBoundaryAgreement": {
                "control": control_bq["stateChangeBoundaryAgreement"],
                "candidate": candidate_bq["stateChangeBoundaryAgreement"],
                "delta": round(candidate_bq["stateChangeBoundaryAgreement"]
                               - control_bq["stateChangeBoundaryAgreement"], 4)},
            "operatingThreshold": {
                "control": control_bq["meanOperatingThreshold"],
                "candidate": candidate_bq["meanOperatingThreshold"]},
        }
    return out


def chord_channel_summary(control_entries: list, candidate_entries: list) -> dict[str, Any]:
    """Compare the chord posterior itself between two models, frame by frame.

    Measures how far the chord heads moved independently of any decoding:
    top-1 argmax agreement, mean posterior margin, and mean no-chord
    probability. A "boundary" intervention that shifts these has changed the
    chord posterior, whatever it was aimed at.
    """
    from .diagnose import chord_posteriors

    by_key = {(e.performance_id, e.capture): e for e in control_entries}
    agreements: list[float] = []
    control_margins: list[float] = []
    candidate_margins: list[float] = []
    control_nochord: list[float] = []
    candidate_nochord: list[float] = []

    for candidate_entry in candidate_entries:
        control_entry = by_key.get((candidate_entry.performance_id, candidate_entry.capture))
        if control_entry is None:
            continue
        length = min(len(control_entry.times), len(candidate_entry.times))
        if length == 0:
            continue
        control_states = chord_posteriors(
            control_entry.root[:length], control_entry.quality[:length],
            control_entry.nochord[:length])
        candidate_states = chord_posteriors(
            candidate_entry.root[:length], candidate_entry.quality[:length],
            candidate_entry.nochord[:length])
        agreements.append(float(np.mean(
            np.argmax(control_states, axis=1) == np.argmax(candidate_states, axis=1))))
        for states, margins in ((control_states, control_margins),
                                (candidate_states, candidate_margins)):
            ordered = np.sort(states, axis=1)
            margins.append(float(np.mean(ordered[:, -1] - ordered[:, -2])))
        control_nochord.append(float(np.mean(control_entry.nochord[:length])))
        candidate_nochord.append(float(np.mean(candidate_entry.nochord[:length])))

    def mean(values: list[float]) -> float:
        return round(float(np.mean(values)), 4) if values else 0.0

    return {
        "comparedCaptures": len(agreements),
        "top1AgreementWithControl": mean(agreements),
        "controlMeanMargin": mean(control_margins),
        "candidateMeanMargin": mean(candidate_margins),
        "marginDelta": round(mean(candidate_margins) - mean(control_margins), 4),
        "controlMeanNoChord": mean(control_nochord),
        "candidateMeanNoChord": mean(candidate_nochord),
    }


def interpret(decomposition: dict[str, Any], boundary: dict[str, Any],
              chord: dict[str, Any]) -> dict[str, Any]:
    """State, per capture, which channel actually moved fragmentation."""
    findings: dict[str, Any] = {}
    for capture in CAPTURES:
        frag = decomposition[capture]["fragmentationRate"]
        chord_part = frag["chordPosteriorChannel"]
        boundary_part = frag["boundaryChannel"]
        total = chord_part + boundary_part
        dominant = "neither (no material movement)"
        if abs(total) >= 0.002:
            dominant = ("chord posterior" if abs(chord_part) > abs(boundary_part)
                        else "boundary channel")
        findings[capture] = {
            "fragmentationImprovementTotal": round(total, 4),
            "viaChordPosterior": round(chord_part, 4),
            "viaBoundaryChannel": round(boundary_part, 4),
            "dominantChannel": dominant,
            "boundaryF1Delta": boundary[capture]["f1"]["delta"],
            "chordTop1AgreementWithControl": chord.get("top1AgreementWithControl", 0.0),
            "note": (
                "The full-v2 decode does not read the boundary head, so a model "
                "effect visible there is chord-posterior movement by definition."
            ),
        }
    return findings
