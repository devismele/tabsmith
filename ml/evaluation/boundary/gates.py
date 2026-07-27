"""Apply the frozen boundary-calibration gates.

The gate values live in ``ml/configs/boundary-calibration-v3-gates.json`` and
were committed before any candidate was trained. Nothing here may relax them:
this module only reads them and reports pass/fail with the measured value that
decided each one, so a failure is always traceable to a number.

A candidate is eligible only if every gate passes on *both* captures.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ML_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATES = ML_ROOT / "configs" / "boundary-calibration-v3-gates.json"
CAPTURES = ("audio_mono-mic", "audio_mono-pickup_mix")


def load_gates(path: Path = DEFAULT_GATES) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _check(name: str, passed: bool, measured, required, capture: str) -> dict[str, Any]:
    return {"gate": name, "capture": capture, "passed": bool(passed),
            "measured": measured, "required": required}


def evaluate_candidate_gates(
    candidate_id: str,
    decoder_results: dict[str, Any],
    *,
    gates: dict[str, Any],
    baseline_full_v2: dict[str, Any],
    baseline_boundary: dict[str, Any],
) -> dict[str, Any]:
    """Apply every frozen gate to one candidate under one decoder."""
    g = gates["gates"]
    accuracy = g["accuracy"]
    stability = g["stabilityThroughSegmentalDecoding"]
    boundary = g["boundary"]
    checks: list[dict[str, Any]] = []

    for cap in CAPTURES:
        metrics = decoder_results["byCapture"][cap]
        bq = decoder_results["boundaryQuality"][cap]
        taxonomy = decoder_results["regionTaxonomy"][cap]
        base = baseline_full_v2[cap]
        base_bq = baseline_boundary

        checks.append(_check("rootAccuracyMinimum",
                             metrics["rootAccuracy"] >= accuracy["rootAccuracyMinimum"],
                             round(metrics["rootAccuracy"], 4),
                             accuracy["rootAccuracyMinimum"], cap))
        checks.append(_check("detailedAccuracyMinimum",
                             metrics["detailedAccuracy"] >= accuracy["detailedAccuracyMinimum"],
                             round(metrics["detailedAccuracy"], 4),
                             accuracy["detailedAccuracyMinimum"], cap))

        per_performer = decoder_results.get("perPerformer", {}).get(cap, {})
        worst = min((v["rootAccuracy"] for v in per_performer.values()), default=1.0)
        checks.append(_check("perPerformerRootAccuracyMinimum",
                             worst >= accuracy["perPerformerRootAccuracyMinimum"],
                             round(worst, 4),
                             accuracy["perPerformerRootAccuracyMinimum"], cap))

        checks.append(_check("fragmentationRateMaximum",
                             metrics["fragmentationRate"] <= stability["fragmentationRateMaximum"],
                             round(metrics["fragmentationRate"], 4),
                             stability["fragmentationRateMaximum"], cap))
        checks.append(_check("regionsPerMinuteMaximum",
                             metrics["regionsPerMinute"] <= stability["regionsPerMinuteMaximum"],
                             round(metrics["regionsPerMinute"], 3),
                             stability["regionsPerMinuteMaximum"], cap))
        checks.append(_check("flickerBelowFullV2",
                             metrics["flickerCount"] < base["flickerCount"],
                             metrics["flickerCount"], f"< {base['flickerCount']}", cap))

        base_false_long = base_bq["falseLong"][cap]
        false_long = taxonomy.get("false_long", 0)
        required_false_long = base_false_long * (1.0 - stability["falseLongRelativeReductionMinimum"])
        checks.append(_check("falseLongMateriallyLower",
                             false_long <= required_false_long,
                             false_long, f"<= {required_false_long:.1f}", cap))

        checks.append(_check("boundaryMaeMaximum",
                             metrics["meanAbsoluteBoundaryErrorMs"] <= boundary["meanAbsoluteBoundaryErrorMsMaximum"],
                             round(metrics["meanAbsoluteBoundaryErrorMs"], 1),
                             boundary["meanAbsoluteBoundaryErrorMsMaximum"], cap))

        base_precision = base_bq["precision"][cap]
        required_precision = base_precision + boundary["boundaryPrecisionAbsoluteImprovementMinimum"]
        checks.append(_check("boundaryPrecisionMateriallyImproves",
                             bq["precision"] >= required_precision,
                             bq["precision"], round(required_precision, 4), cap))

        base_recall = base_bq["recall"][cap]
        recall_floor = base_recall * boundary["boundaryRecallRelativeFloorVsFullV2"]
        checks.append(_check("boundaryRecallDoesNotCollapse",
                             bq["recall"] >= recall_floor,
                             bq["recall"], round(recall_floor, 4), cap))

        base_agreement = base_bq["agreement"][cap]
        required_agreement = base_agreement + boundary["stateChangeBoundaryAgreementAbsoluteImprovementMinimum"]
        checks.append(_check("stateChangeBoundaryAgreementImproves",
                             bq["stateChangeBoundaryAgreement"] >= required_agreement,
                             bq["stateChangeBoundaryAgreement"],
                             round(required_agreement, 4), cap))

    failed = [c for c in checks if not c["passed"]]
    return {
        "candidateId": candidate_id,
        "eligible": not failed,
        "checks": checks,
        "failedGates": [f"{c['gate']}@{c['capture']}" for c in failed],
        "firstFailure": (
            f"{failed[0]['gate']} on {failed[0]['capture']}: "
            f"measured {failed[0]['measured']} vs required {failed[0]['required']}"
            if failed else None
        ),
    }
