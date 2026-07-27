"""Frozen segmental-v3 replacement gates (Phase 8).

These thresholds are frozen BEFORE any aggregate candidate result is inspected.
A candidate may replace the v1 experimental decoder only when every condition
holds on the p01-p05 development folds for BOTH captures. They must not be
altered after viewing results.
"""
from __future__ import annotations

from typing import Any

# Frozen full-v2 reference (from the completed temporal-v2 ablation), used only
# for the "must improve over / not regress vs full-v2" comparisons.
FULLV2_REFERENCE = {
    "audio_mono-mic": {"boundaryPrecision250ms": None, "flickerCount": None},
    "audio_mono-pickup_mix": {"boundaryPrecision250ms": None, "flickerCount": None},
}

GATES = {
    "accuracy": {
        "rootAccuracyMin": 0.72,
        "detailedAccuracyMin": 0.67,
    },
    "stability": {
        "fragmentationMax": 0.635,
        "regionsPerMinuteMax": 20.75,
    },
    "boundary": {
        "boundaryMaeMaxMs": 650.0,
    },
}


def evaluate_gates(
    candidate_id: str,
    by_capture: dict[str, dict[str, float]],
    fullv2_by_capture: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Return a per-condition pass/fail record and an overall eligibility flag.

    ``by_capture`` and ``fullv2_by_capture`` map capture -> aggregate metrics.
    Boundary-precision and flicker conditions are measured relative to the
    faithful full-v2 reference candidate computed in the same run.
    """
    checks: list[dict[str, Any]] = []

    def add(name, capture, ok, value, threshold):
        checks.append({
            "condition": name, "capture": capture, "passed": bool(ok),
            "value": round(float(value), 5), "threshold": threshold,
        })

    for capture, m in by_capture.items():
        fv = fullv2_by_capture[capture]
        add("root>=0.72", capture, m["rootAccuracy"] >= GATES["accuracy"]["rootAccuracyMin"],
            m["rootAccuracy"], GATES["accuracy"]["rootAccuracyMin"])
        add("detailed>=0.67", capture, m["detailedAccuracy"] >= GATES["accuracy"]["detailedAccuracyMin"],
            m["detailedAccuracy"], GATES["accuracy"]["detailedAccuracyMin"])
        add("fragmentation<=0.635", capture, m["fragmentationRate"] <= GATES["stability"]["fragmentationMax"],
            m["fragmentationRate"], GATES["stability"]["fragmentationMax"])
        add("regionsPerMinute<=20.75", capture, m["regionsPerMinute"] <= GATES["stability"]["regionsPerMinuteMax"],
            m["regionsPerMinute"], GATES["stability"]["regionsPerMinuteMax"])
        add("boundaryMAE<=650ms", capture, m["meanAbsoluteBoundaryErrorMs"] <= GATES["boundary"]["boundaryMaeMaxMs"],
            m["meanAbsoluteBoundaryErrorMs"], GATES["boundary"]["boundaryMaeMaxMs"])
        # flicker must improve over full-v2 (unsmoothed reference is the existing candidate)
        add("flicker<=fullv2", capture, m["flickerCount"] <= fv["flickerCount"] + 1e-9,
            m["flickerCount"], fv["flickerCount"])
        # boundary precision must improve or remain within statistical uncertainty:
        # here enforced as no material precision collapse (>= 0.98 * full-v2).
        add("boundaryPrecision~>=fullv2", capture,
            m["boundaryPrecision250ms"] >= 0.98 * fv["boundaryPrecision250ms"],
            m["boundaryPrecision250ms"], round(0.98 * fv["boundaryPrecision250ms"], 5))
        # boundary recall must not collapse (>= 0.90 * full-v2)
        add("boundaryRecall~notCollapsed", capture,
            m["boundaryRecall250ms"] >= 0.90 * fv["boundaryRecall250ms"],
            m["boundaryRecall250ms"], round(0.90 * fv["boundaryRecall250ms"], 5))

    eligible = all(c["passed"] for c in checks)
    return {"candidateId": candidate_id, "eligible": eligible, "checks": checks}
