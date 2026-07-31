"""Per-track metrics, aggregation, and performer-paired bootstrap for the
segmental-v3 study. Shared metrics reuse the exact weighting of the frozen
temporal-v2 report so a faithful ``existing`` candidate reproduces full-v2."""
from __future__ import annotations

import random
from collections import defaultdict
from statistics import median
from typing import Any, Callable

import numpy as np

from ..metrics import _boundary_f1, evaluate_regions

CAPTURES = ("audio_mono-mic", "audio_mono-pickup_mix")
_SWITCH_TOL = 0.25


def _region_tuples(regions):
    out = []
    for r in regions:
        if hasattr(r, "start"):
            out.append((float(r.start), float(r.end), r.label))
        else:
            out.append((float(r["start"]), float(r["end"]), r.get("label") or r.get("name") or "N"))
    return out


def _frame_diagnostics(root, quality, nochord, boundary) -> dict[str, float]:
    chord = (root[:, :, None] * quality[:, None, :] * (1.0 - nochord[:, None, None])).reshape(len(root), -1)
    states = np.concatenate([chord, nochord[:, None]], axis=1)
    entropy = -np.sum(states * np.log(np.maximum(states, 1e-9)), axis=1) / np.log(37.0)
    switch = 0.5 * np.abs(states[1:] - states[:-1]).sum(axis=1) if len(states) > 1 else np.zeros(0)
    return {
        "meanTopChordConfidence": float(np.mean(np.max(states, axis=1))) if len(states) else 0.0,
        "meanLearnedEntropy": float(np.mean(entropy)) if len(states) else 0.0,
        "meanFrameSwitchProbability": float(np.mean(switch)) if len(switch) else 0.0,
        "meanBoundaryProbability": float(np.mean(boundary)) if len(boundary) else 0.0,
    }


def track_metrics(
    *, reference, predicted, hop_seconds: float,
    root=None, quality=None, nochord=None, boundary=None,
) -> dict[str, Any]:
    """One track's metrics: core accuracy/boundary + region-shape diagnostics."""
    core = evaluate_regions(reference, predicted, tolerances=(0.1, 0.25, 0.5, 1.0))
    pred = _region_tuples(predicted)
    durations = [max(0.0, e - s) for s, e, _ in pred]
    one_window = sum(1 for d in durations if d <= hop_seconds * 1.5)
    sub250 = sum(1 for d in durations if d < 0.25)
    sub500 = sum(1 for d in durations if d < 0.5)
    sub1000 = sum(1 for d in durations if d < 1.0)
    aba = sum(
        1 for i in range(1, len(pred) - 1)
        if pred[i - 1][2] == pred[i + 1][2] and pred[i][2] != pred[i - 1][2]
    )
    flicker = sum(
        1 for i in range(1, len(pred) - 1)
        if pred[i - 1][2] == pred[i + 1][2] and pred[i][2] != pred[i - 1][2]
        and (pred[i][1] - pred[i][0]) < 1.0
    )
    # state-switch: decoded region-change times vs reference region-change times
    ref = _region_tuples(reference)
    ref_changes = [r[0] for r in ref[1:]]
    pred_changes = [p[0] for p in pred[1:]]
    switch_f1 = _boundary_f1(ref_changes, pred_changes, _SWITCH_TOL)
    switch_hits = round(switch_f1["recall"] * len(ref_changes)) if ref_changes else 0
    row: dict[str, Any] = dict(core)
    row.update({
        "oneWindowRegionCount": one_window,
        "subQuarterSecondCount": sub250,
        "shortRegionCount": sub500,
        "subOneSecondCount": sub1000,
        "abaCount": aba,
        "flickerCount": flicker,
        "predictedChangeCount": len(pred_changes),
        "referenceChangeCount": len(ref_changes),
        "stateSwitchHits": switch_hits,
        "stateSwitchPrecision": switch_f1["precision"],
        "stateSwitchRecall": switch_f1["recall"],
        "stateSwitchF1": switch_f1["f1"],
    })
    if root is not None:
        row.update(_frame_diagnostics(root, quality, nochord, boundary))
    else:
        row.update({k: 0.0 for k in (
            "meanTopChordConfidence", "meanLearnedEntropy",
            "meanFrameSwitchProbability", "meanBoundaryProbability")})
    return row


def _duration_weighted(rows, name):
    total = sum(r["evaluatedDurationSeconds"] for r in rows)
    return sum(r[name] * r["evaluatedDurationSeconds"] for r in rows) / total if total else 0.0


def aggregate_capture(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate one capture's per-track rows using the frozen report weighting."""
    if not rows:
        raise ValueError("cannot aggregate zero tracks")
    total_dur = sum(r["evaluatedDurationSeconds"] for r in rows)
    total_pred = sum(r["predictedRegions"] for r in rows)
    total_ref = sum(r["referenceRegions"] for r in rows)
    bw = sum(max(0, r["referenceRegions"] - 1) for r in rows)

    def ref_weighted(name):
        return sum((r[name] or 0.0) * max(0, r["referenceRegions"] - 1) for r in rows) / max(1, bw)

    def f1_weighted(tol):
        return sum(r["boundaryF1"][tol]["f1"] * max(0, r["referenceRegions"] - 1) for r in rows) / max(1, bw)

    def recall_weighted(tol):
        return sum(r["boundaryF1"][tol]["recall"] * max(0, r["referenceRegions"] - 1) for r in rows) / max(1, bw)

    pred_changes = sum(r["predictedChangeCount"] for r in rows)
    ref_changes = sum(r["referenceChangeCount"] for r in rows)
    switch_hits = sum(r["stateSwitchHits"] for r in rows)
    sw_prec = switch_hits / pred_changes if pred_changes else 0.0
    sw_rec = switch_hits / ref_changes if ref_changes else 0.0
    return {
        "trackCount": float(len(rows)),
        "evaluatedDurationSeconds": total_dur,
        "rootAccuracy": _duration_weighted(rows, "rootAccuracy"),
        "majorMinorAccuracy": _duration_weighted(rows, "majorMinorAccuracy"),
        "detailedAccuracy": _duration_weighted(rows, "detailedAccuracy"),
        "weightedChordSymbolRecall": _duration_weighted(rows, "detailedAccuracy"),
        "noChordPrecision": _duration_weighted(rows, "noChordPrecision"),
        "noChordRecall": _duration_weighted(rows, "noChordRecall"),
        "fragmentationRate": sum(r["fragmentationRate"] * r["referenceRegions"] for r in rows) / max(1, total_ref),
        "regionsPerMinute": total_pred * 60.0 / total_dur if total_dur else 0.0,
        "predictedRegionCount": float(total_pred),
        "referenceRegionCount": float(total_ref),
        "averageChordRegionDurationSeconds": total_dur / max(1, total_pred),
        "extraRegionCount": float(sum(r["extraChords"] for r in rows)),
        "missingRegionCount": float(sum(r["missingChords"] for r in rows)),
        "oneWindowRegionRate": sum(r["oneWindowRegionCount"] for r in rows) / max(1, total_pred),
        "twoWindowRegionRate": sum(r["subQuarterSecondCount"] for r in rows) / max(1, total_pred),
        "subQuarterSecondRate": sum(r["subQuarterSecondCount"] for r in rows) / max(1, total_pred),
        "subHalfSecondRate": sum(r["shortRegionCount"] for r in rows) / max(1, total_pred),
        "subOneSecondRate": sum(r["subOneSecondCount"] for r in rows) / max(1, total_pred),
        "shortRegionRate": sum(r["shortRegionCount"] for r in rows) / max(1, total_pred),
        "abaCount": float(sum(r["abaCount"] for r in rows)),
        "flickerCount": float(sum(r["flickerCount"] for r in rows)),
        "meanTopChordConfidence": _duration_weighted(rows, "meanTopChordConfidence"),
        "meanLearnedEntropy": _duration_weighted(rows, "meanLearnedEntropy"),
        "meanFrameSwitchProbability": _duration_weighted(rows, "meanFrameSwitchProbability"),
        "meanBoundaryProbability": _duration_weighted(rows, "meanBoundaryProbability"),
        "meanAbsoluteBoundaryErrorMs": ref_weighted("meanBoundaryErrorMs"),
        "medianAbsoluteBoundaryErrorMs": ref_weighted("medianBoundaryErrorMs"),
        "meanSignedBoundaryErrorMs": ref_weighted("meanSignedBoundaryErrorMs"),
        "boundariesWithin100ms": recall_weighted("100ms"),
        "boundariesWithin250ms": recall_weighted("250ms"),
        "boundariesWithin500ms": recall_weighted("500ms"),
        "boundariesWithin1000ms": recall_weighted("1000ms"),
        "boundaryPrecision250ms": sum(
            r["boundaryF1"]["250ms"]["precision"] * max(0, r["predictedRegions"] - 1) for r in rows
        ) / max(1, sum(max(0, r["predictedRegions"] - 1) for r in rows)),
        "boundaryRecall250ms": recall_weighted("250ms"),
        "boundaryF1At100ms": f1_weighted("100ms"),
        "boundaryF1At250ms": f1_weighted("250ms"),
        "boundaryF1At500ms": f1_weighted("500ms"),
        "boundaryF1At1000ms": f1_weighted("1000ms"),
        "stateSwitchPrecision": sw_prec,
        "stateSwitchRecall": sw_rec,
        "stateSwitchF1": 2 * sw_prec * sw_rec / (sw_prec + sw_rec) if (sw_prec + sw_rec) else 0.0,
    }


def performer_paired_bootstrap(
    rows: list[dict[str, Any]], capture: str, metrics: list[str],
    *, iterations: int = 1000, seed: int = 20260727,
) -> dict[str, dict[str, float]]:
    """95% CIs by resampling performances (both captures kept together) with
    replacement, grouped by performer to respect leave-one-performer-out."""
    by_perf: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["capture"] == capture:
            by_perf[(r["performerId"], r["performanceId"])].append(r)
    groups = list(by_perf.values())
    if not groups:
        raise ValueError("no rows for capture")
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {m: [] for m in metrics}
    n = len(groups)
    for _ in range(iterations):
        drawn = [row for _ in range(n) for row in groups[rng.randrange(n)]]
        agg = aggregate_capture(drawn)
        for m in metrics:
            samples[m].append(agg[m])
    out = {}
    for m in metrics:
        vals = sorted(samples[m])
        lo = vals[int(0.025 * (len(vals) - 1))]
        hi = vals[int(0.975 * (len(vals) - 1))]
        out[m] = {"lo": lo, "hi": hi}
    return out
