"""Phase 5 boundary-calibration diagnosis for full-v2 (p01-p05 only).

Reuses the frozen segmental-v3 inference cache, so nothing is re-inferred and
p00 cannot enter (it was never cached). For each capture this measures, in one
pass:

* frame-level boundary calibration -- reliability diagram, ECE/MCE, base rate
* boundary precision/recall/F1 across a threshold sweep (peak-picked)
* decoded chord-state-switch precision/recall/F1 and boundary agreement
* boundary probability and chord-posterior margin on four disjoint categories:
  true transitions, false transitions, missed transitions, sustained frames
* region shape around each proposed switch (previous/proposed duration,
  A->B->A interruptions, false-long vs false-short)
* boundary timing bias, early vs late, and mic/pickup disagreement
* bass-root agreement, per-performer breakdowns

The decoder is held fixed at the faithful full-v2 path (EMA smoothing +
penalty-4 Viterbi) so every number here describes the *model*, not a decoder
variant.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from ..decode import viterbi_decode
from ..probability_smoothing import smooth_learned_probabilities
from ..segmental.diagnostics import _nearest, _peak_boundaries
from ..segmental.seg_metrics import _region_tuples
from .calibration import (
    expected_calibration_error,
    maximum_calibration_error,
    reliability_bins,
)

TOLERANCE = 0.25
THRESHOLD_SWEEP = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                   0.45, 0.50, 0.55, 0.60, 0.70, 0.80)
PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def chord_posteriors(root, quality, nochord):
    """(T, 37) chord-state posterior: 12 roots x 3 qualities, plus no-chord."""
    chord = (root[:, :, None] * quality[:, None, :] * (1.0 - nochord[:, None, None]))
    chord = chord.reshape(len(root), -1)
    return np.concatenate([chord, nochord[:, None]], axis=1)


def _margin_and_top(states: np.ndarray):
    srt = np.sort(states, axis=1)
    return srt[:, -1] - srt[:, -2], srt[:, -1]


def _frame_boundary_labels(times: np.ndarray, ref_changes: list[float],
                           tolerance: float = TOLERANCE) -> np.ndarray:
    """1.0 where a reference chord change falls within ``tolerance`` of the frame."""
    labels = np.zeros(len(times), dtype=np.float64)
    if not ref_changes:
        return labels
    changes = np.asarray(sorted(ref_changes), dtype=np.float64)
    idx = np.searchsorted(changes, times)
    left = np.clip(idx - 1, 0, len(changes) - 1)
    right = np.clip(idx, 0, len(changes) - 1)
    nearest = np.minimum(np.abs(times - changes[left]), np.abs(times - changes[right]))
    labels[nearest <= tolerance] = 1.0
    return labels


def _threshold_sweep(times, boundary, ref_changes, hop, thresholds=THRESHOLD_SWEEP):
    """Peak-picked boundary P/R/F1 at each threshold (accumulated as counts)."""
    out = {}
    for th in thresholds:
        peaks = _peak_boundaries(times, boundary, hop, threshold=th)
        hits = sum(1 for c in ref_changes if _nearest(c, peaks, TOLERANCE) is not None)
        out[th] = {"pred": len(peaks), "ref": len(ref_changes), "hits": hits}
    return out


def _prf(hits: int, pred: int, ref: int) -> dict[str, float]:
    precision = hits / pred if pred else 0.0
    recall = hits / ref if ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "p10": round(float(np.percentile(arr, 10)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
    }


def _bass_root_agreement(bass_chroma: np.ndarray, predicted, times, hop) -> tuple[int, int]:
    """How often the decoded root equals the argmax bass pitch class."""
    if bass_chroma is None or bass_chroma.size == 0:
        return 0, 0
    bass_root = np.argmax(bass_chroma, axis=1)
    agree = total = 0
    for start, end, label in _region_tuples(predicted):
        if label in ("N", "X") or ":" not in label:
            continue
        name = label.split(":")[0]
        if name not in PITCH_CLASSES:
            continue
        target = PITCH_CLASSES.index(name)
        lo = int(np.searchsorted(times, start))
        hi = int(np.searchsorted(times, end))
        if hi <= lo:
            continue
        agree += int(np.sum(bass_root[lo:hi] == target))
        total += hi - lo
    return agree, total


def diagnose_entries(entries: list[Any], smoothing: dict | None) -> dict[str, Any]:
    """Full boundary diagnosis over one capture's held-out entries."""
    cal_probs: list[np.ndarray] = []
    cal_labels: list[np.ndarray] = []
    sweep_totals = {th: {"pred": 0, "ref": 0, "hits": 0} for th in THRESHOLD_SWEEP}

    categories = {
        "trueTransition": {"boundaryProb": [], "margin": [], "topConfidence": []},
        "falseTransition": {"boundaryProb": [], "margin": [], "topConfidence": []},
        "missedTransition": {"boundaryProb": [], "margin": [], "topConfidence": []},
        "sustainedNonBoundary": {"boundaryProb": [], "margin": [], "topConfidence": []},
    }
    shape = {
        "falsePreviousDurationS": [], "falseProposedDurationS": [],
        "truePreviousDurationS": [], "trueProposedDurationS": [],
    }
    taxonomy: dict[str, int] = defaultdict(int)
    timing_signed_ms: list[float] = []
    sc_pred = sc_ref = sc_hits = 0
    agree_changes = agree_boundary = 0
    agreement_sweep = {th: {"changes": 0, "supported": 0} for th in THRESHOLD_SWEEP}
    bass_agree = bass_total = 0
    per_performer: dict[str, dict[str, float]] = defaultdict(
        lambda: {"trueTransitions": 0, "falseTransitions": 0, "missedTransitions": 0,
                 "predictedRegions": 0, "referenceRegions": 0,
                 "trueBoundaryProbSum": 0.0, "falseBoundaryProbSum": 0.0})
    switch_by_performance: dict[str, dict[str, int]] = defaultdict(dict)

    for e in entries:
        root, quality, nochord, boundary = smooth_learned_probabilities(
            e.root, e.quality, e.nochord, e.boundary, smoothing)
        predicted = viterbi_decode(e.times, root, quality, nochord, e.hop_seconds, 4.0)
        pred = _region_tuples(predicted)
        ref = _region_tuples(e.reference)
        ref_changes = [r[0] for r in ref[1:]]
        pred_changes = [p[0] for p in pred[1:]]

        states = chord_posteriors(root, quality, nochord)
        margin, topconf = _margin_and_top(states)

        # --- frame-level calibration -------------------------------------
        labels = _frame_boundary_labels(e.times, ref_changes)
        cal_probs.append(np.asarray(boundary, dtype=np.float64))
        cal_labels.append(labels)

        # --- boundary threshold sweep ------------------------------------
        for th, counts in _threshold_sweep(e.times, boundary, ref_changes, e.hop_seconds).items():
            for key, value in counts.items():
                sweep_totals[th][key] += value

        # --- decoded state-switch quality --------------------------------
        hits = sum(1 for c in ref_changes if _nearest(c, pred_changes, TOLERANCE) is not None)
        sc_pred += len(pred_changes)
        sc_ref += len(ref_changes)
        sc_hits += hits
        peaks = _peak_boundaries(e.times, boundary, e.hop_seconds)
        for c in pred_changes:
            agree_changes += 1
            if _nearest(c, peaks, TOLERANCE) is not None:
                agree_boundary += 1

        # Agreement is a function of the peak-picking threshold, not only of the
        # heads: at 0.5 the head emits far fewer peaks than the decoder emits
        # changes, so a low value there conflates disagreement with starvation.
        for th in THRESHOLD_SWEEP:
            th_peaks = _peak_boundaries(e.times, boundary, e.hop_seconds, threshold=th)
            agreement_sweep[th]["changes"] += len(pred_changes)
            agreement_sweep[th]["supported"] += sum(
                1 for c in pred_changes if _nearest(c, th_peaks, TOLERANCE) is not None)

        perf = per_performer[e.performer_id]
        perf["predictedRegions"] += len(pred)
        perf["referenceRegions"] += len(ref)

        def frame_of(t: float) -> int:
            i = int(np.searchsorted(e.times, t))
            return min(max(i, 0), len(e.times) - 1)

        # --- categories at proposed switches -----------------------------
        switch_flags: dict[str, int] = {}
        for i in range(1, len(pred)):
            t = pred[i][0]
            fi = frame_of(t)
            matched = _nearest(t, ref_changes, TOLERANCE)
            bucket = "trueTransition" if matched is not None else "falseTransition"
            categories[bucket]["boundaryProb"].append(float(boundary[fi]))
            categories[bucket]["margin"].append(float(margin[fi]))
            categories[bucket]["topConfidence"].append(float(topconf[fi]))
            previous_duration = pred[i - 1][1] - pred[i - 1][0]
            proposed_duration = pred[i][1] - pred[i][0]
            is_aba = (i < len(pred) - 1 and pred[i - 1][2] == pred[i + 1][2]
                      and pred[i][2] != pred[i - 1][2])
            if matched is not None:
                taxonomy["true_boundary"] += 1
                shape["truePreviousDurationS"].append(previous_duration)
                shape["trueProposedDurationS"].append(proposed_duration)
                timing_signed_ms.append((t - matched) * 1000.0)
                perf["trueTransitions"] += 1
                perf["trueBoundaryProbSum"] += float(boundary[fi])
                switch_flags[f"{round(t, 2)}"] = 1
            else:
                shape["falsePreviousDurationS"].append(previous_duration)
                shape["falseProposedDurationS"].append(proposed_duration)
                perf["falseTransitions"] += 1
                perf["falseBoundaryProbSum"] += float(boundary[fi])
                if is_aba:
                    taxonomy["false_aba_flicker"] += 1
                elif proposed_duration <= e.hop_seconds * 1.5:
                    taxonomy["false_one_window"] += 1
                elif proposed_duration < 1.0:
                    taxonomy["false_short_multi_window"] += 1
                else:
                    taxonomy["false_long"] += 1
                switch_flags[f"{round(t, 2)}"] = 0

        # --- missed reference transitions --------------------------------
        for c in ref_changes:
            if _nearest(c, pred_changes, TOLERANCE) is None:
                fi = frame_of(c)
                categories["missedTransition"]["boundaryProb"].append(float(boundary[fi]))
                categories["missedTransition"]["margin"].append(float(margin[fi]))
                categories["missedTransition"]["topConfidence"].append(float(topconf[fi]))
                taxonomy["missed_boundary"] += 1
                perf["missedTransitions"] += 1

        # --- sustained non-boundary frames -------------------------------
        sustained = labels == 0.0
        if sustained.any():
            categories["sustainedNonBoundary"]["boundaryProb"].extend(
                boundary[sustained].tolist())
            categories["sustainedNonBoundary"]["margin"].extend(margin[sustained].tolist())
            categories["sustainedNonBoundary"]["topConfidence"].extend(
                topconf[sustained].tolist())

        agree, total = _bass_root_agreement(e.bass_chroma, predicted, e.times, e.hop_seconds)
        bass_agree += agree
        bass_total += total
        switch_by_performance[e.performance_id] = switch_flags

    probs = np.concatenate(cal_probs) if cal_probs else np.zeros(0)
    labels = np.concatenate(cal_labels) if cal_labels else np.zeros(0)

    sweep = {}
    for th, counts in sweep_totals.items():
        sweep[f"{th:.2f}"] = dict(_prf(counts["hits"], counts["pred"], counts["ref"]),
                                  predicted=counts["pred"], reference=counts["ref"])
    best_threshold = max(sweep, key=lambda k: sweep[k]["f1"]) if sweep else "0.50"

    signed = np.asarray(timing_signed_ms, dtype=np.float64)
    return {
        "captureEntries": len(entries),
        "calibration": {
            "frames": int(probs.size),
            "positiveRate": round(float(labels.mean()), 6) if labels.size else 0.0,
            "meanPredicted": round(float(probs.mean()), 6) if probs.size else 0.0,
            "expectedCalibrationError": round(expected_calibration_error(probs, labels), 6),
            "maximumCalibrationError": round(maximum_calibration_error(probs, labels), 6),
            "reliability": reliability_bins(probs, labels),
        },
        "boundaryThresholdSweep": sweep,
        "bestF1Threshold": best_threshold,
        "stateSwitch": _prf(sc_hits, sc_pred, sc_ref),
        "stateChangeBoundaryAgreement": round(agree_boundary / agree_changes, 4) if agree_changes else 0.0,
        "stateChangeBoundaryAgreementByThreshold": {
            f"{th:.2f}": round(v["supported"] / v["changes"], 4) if v["changes"] else 0.0
            for th, v in agreement_sweep.items()
        },
        "taxonomy": dict(taxonomy),
        "categoryEvidence": {
            name: {metric: _stats(values) for metric, values in metrics.items()}
            for name, metrics in categories.items()
        },
        "regionShape": {name: _stats(values) for name, values in shape.items()},
        "timing": {
            "meanSignedErrorMs": round(float(signed.mean()), 2) if signed.size else 0.0,
            "medianSignedErrorMs": round(float(np.median(signed)), 2) if signed.size else 0.0,
            "meanAbsoluteErrorMs": round(float(np.abs(signed).mean()), 2) if signed.size else 0.0,
            "earlyCount": int((signed < 0).sum()),
            "lateCount": int((signed > 0).sum()),
        },
        "bassRootAgreement": round(bass_agree / bass_total, 4) if bass_total else 0.0,
        "perPerformer": {
            pid: {
                "trueTransitions": int(v["trueTransitions"]),
                "falseTransitions": int(v["falseTransitions"]),
                "missedTransitions": int(v["missedTransitions"]),
                "predictedRegions": int(v["predictedRegions"]),
                "referenceRegions": int(v["referenceRegions"]),
                "meanTrueBoundaryProb": round(
                    v["trueBoundaryProbSum"] / v["trueTransitions"], 4) if v["trueTransitions"] else 0.0,
                "meanFalseBoundaryProb": round(
                    v["falseBoundaryProbSum"] / v["falseTransitions"], 4) if v["falseTransitions"] else 0.0,
            }
            for pid, v in sorted(per_performer.items())
        },
        "_switchFlags": switch_by_performance,
    }


def capture_disagreement(mic: dict[str, Any], pickup: dict[str, Any]) -> dict[str, float]:
    """How often the two captures of the same performance agree on a switch."""
    mic_flags = mic.get("_switchFlags", {})
    pickup_flags = pickup.get("_switchFlags", {})
    shared = set(mic_flags) & set(pickup_flags)
    agree = total = 0
    for performance in shared:
        keys = set(mic_flags[performance]) | set(pickup_flags[performance])
        for key in keys:
            total += 1
            if key in mic_flags[performance] and key in pickup_flags[performance]:
                agree += 1
    return {
        "sharedPerformances": len(shared),
        "switchSites": total,
        "captureSwitchAgreement": round(agree / total, 4) if total else 0.0,
    }


def classify_dominant_issue(diagnosis: dict[str, Any]) -> dict[str, Any]:
    """Name the dominant model-side failure mode from the measured numbers.

    Deliberately rule-based and reported alongside the evidence, so the chosen
    intervention is traceable to measurements rather than intuition.
    """
    cal = diagnosis["calibration"]
    evidence = diagnosis["categoryEvidence"]
    sweep = diagnosis["boundaryThresholdSweep"]
    true_prob = evidence["trueTransition"]["boundaryProb"]["mean"]
    false_prob = evidence["falseTransition"]["boundaryProb"]["mean"]
    at_half = sweep.get("0.50", {})
    best = sweep.get(diagnosis["bestF1Threshold"], {})

    findings = []
    # Under/over-confidence: sign of (mean predicted - observed base rate).
    skew = cal["meanPredicted"] - cal["positiveRate"]
    if skew < -0.02:
        findings.append("boundary underconfidence")
    elif skew > 0.02:
        findings.append("boundary overconfidence")

    # Thresholding: does moving off 0.5 buy a materially better F1?
    if at_half and best and best["f1"] - at_half["f1"] >= 0.02:
        findings.append("poor thresholding")

    # Discrimination: separation between true and false transition probability.
    separation = true_prob - false_prob
    if separation < 0.10:
        findings.append("weak discrimination")

    # Head agreement.
    if diagnosis["stateChangeBoundaryAgreement"] < 0.5:
        findings.append("poor boundary/chord-head agreement")

    # Recall/precision balance at the operating point.
    if at_half and at_half["recall"] < 0.5 <= at_half["precision"]:
        findings.append("recall starvation at the operating threshold")
    if at_half and at_half["precision"] < 0.5 <= at_half["recall"]:
        findings.append("excessive recall at low precision")

    return {
        "findings": findings,
        "confidenceSkew": round(skew, 6),
        "trueFalseSeparation": round(separation, 4),
        "f1AtHalf": at_half.get("f1", 0.0),
        "bestF1": best.get("f1", 0.0),
        "bestThreshold": diagnosis["bestF1Threshold"],
    }
