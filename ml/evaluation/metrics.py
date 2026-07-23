"""Chord + boundary metrics, ported from server/evaluation.mjs.

Kept numerically consistent with the JavaScript harness so learned-harmony-v1
numbers are directly comparable to the recorded harmonic-context-v3 baseline.
Regions may be ChordRegion objects or dicts with start/end/label(or name).
"""
from __future__ import annotations

from statistics import median

from ..schema import parse_chord_label


def _region(obj) -> tuple[float, float, str]:
    if hasattr(obj, "start"):
        return float(obj.start), float(obj.end), getattr(obj, "label", None) or getattr(obj, "name", "N")
    return float(obj["start"]), float(obj["end"]), obj.get("label") or obj.get("name") or "N"


def _chord_at(regions: list[tuple[float, float, str]], time: float) -> tuple[float, float, str] | None:
    for region in regions:
        if region[0] <= time < region[1]:
            return region
    return None


def _nearest_error(boundary: float, predicted: list[float]) -> float | None:
    if not predicted:
        return None
    best = min(predicted, key=lambda p: abs(p - boundary))
    return best - boundary


def _boundary_f1(ref_boundaries: list[float], pred_boundaries: list[float], tolerance: float) -> dict:
    used = set()
    hits = 0
    for boundary in ref_boundaries:
        best_j, best_d = -1, tolerance + 1e-9
        for j, pred in enumerate(pred_boundaries):
            if j in used:
                continue
            d = abs(pred - boundary)
            if d <= tolerance and d < best_d:
                best_j, best_d = j, d
        if best_j >= 0:
            used.add(best_j)
            hits += 1
    precision = hits / len(pred_boundaries) if pred_boundaries else 0.0
    recall = hits / len(ref_boundaries) if ref_boundaries else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def evaluate_regions(reference, predicted, bpm: float | None = None,
                     tolerances=(0.1, 0.25, 0.5)) -> dict:
    ref = [_region(r) for r in reference]
    pred = [_region(r) for r in predicted]
    if not ref:
        raise ValueError("reference has no chord regions")

    eval_start = min(r[0] for r in ref)
    eval_end = max(r[1] for r in ref)
    pred = [p for p in pred if p[1] > eval_start and p[0] < eval_end]

    points = sorted({p for region in ref + pred for p in (region[0], region[1])
                     if eval_start - 1e-9 <= p <= eval_end + 1e-9})

    total = root_ok = family_ok = detail_ok = 0.0
    ref_n_dur = pred_n_dur = n_true = 0.0
    for i in range(len(points) - 1):
        start, end = points[i], points[i + 1]
        if end <= start:
            continue
        mid = (start + end) / 2
        expected_region = _chord_at(ref, mid)
        if not expected_region:
            continue
        detected_region = _chord_at(pred, mid)
        expected = parse_chord_label(expected_region[2])
        detected = parse_chord_label(detected_region[2] if detected_region else "N")
        dur = end - start
        total += dur
        if expected.root == detected.root:
            root_ok += dur
        if expected.majmin == detected.majmin:
            family_ok += dur
        if expected.detailed == detected.detailed:
            detail_ok += dur
        if expected.is_no_chord:
            ref_n_dur += dur
        if detected.is_no_chord:
            pred_n_dur += dur
        if expected.is_no_chord and detected.is_no_chord:
            n_true += dur

    ref_boundaries = [r[0] for r in ref[1:]]
    pred_boundaries = [p[0] for p in pred[1:]]
    errors = [e for e in (_nearest_error(b, pred_boundaries) for b in ref_boundaries) if e is not None]
    abs_errors = [abs(e) for e in errors]

    fragmented = sum(1 for r in ref if sum(1 for p in pred if p[0] < r[1] and p[1] > r[0]) > 1)
    ref_symbols = {parse_chord_label(r[2]).detailed for r in ref}
    pred_symbols = {parse_chord_label(p[2]).detailed for p in pred}

    def pct(numer, denom):
        return round(numer / denom, 4) if denom else 0.0

    beat = (60.0 / bpm) if (bpm and bpm > 0) else None
    return {
        "evaluatedDurationSeconds": round(total, 3),
        "rootAccuracy": pct(root_ok, total),
        "majorMinorAccuracy": pct(family_ok, total),
        "detailedAccuracy": pct(detail_ok, total),
        "chordSymbolRecall": pct(len(ref_symbols & pred_symbols), len(ref_symbols)),
        "noChordPrecision": pct(n_true, pred_n_dur),
        "noChordRecall": pct(n_true, ref_n_dur),
        "meanBoundaryErrorMs": round(sum(abs_errors) / len(abs_errors) * 1000, 1) if abs_errors else None,
        "medianBoundaryErrorMs": round(median(abs_errors) * 1000, 1) if abs_errors else None,
        "meanSignedBoundaryErrorMs": round(sum(errors) / len(errors) * 1000, 1) if errors else None,
        "percentBoundariesOverOneBeatLate": (
            pct(sum(1 for e in errors if beat and e > beat), len(errors)) if (beat and errors) else None
        ),
        "boundaryF1": {f"{int(t * 1000)}ms": _boundary_f1(ref_boundaries, pred_boundaries, t) for t in tolerances},
        "fragmentationRate": pct(fragmented, len(ref)),
        "referenceRegions": len(ref),
        "predictedRegions": len(pred),
        "missingChords": max(0, len(ref) - len(pred)),
        "extraChords": max(0, len(pred) - len(ref)),
    }
