"""Phase 9 evaluation for the frozen boundary-calibration candidates.

For every (candidate, decoder, fold) this:

1. loads that candidate's cached four-head responses for the held-out performer
   (the frozen full-v2 cache for checkpoint reusers, a candidate-specific cache
   for retrained candidates);
2. fits the candidate's boundary calibrator on the *training* performers of the
   fold only, and picks the boundary operating threshold on those same
   performers -- never the held-out fold, never p00;
3. applies the calibrator to the held-out entries' boundary channel;
4. decodes with both the existing full-v2 path and the frozen segmental-full
   decoder, so model effects stay separable from decoder effects;
5. scores with the same metrics and the same performer-paired bootstrap the
   segmental-v3 study used, then applies the frozen gates.

The existing full-v2 decode does not consume the boundary head, so calibration
is inert on that path by construction. That is reported rather than hidden: it
is what makes the segmental-full column interpretable as a boundary effect.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..probability_smoothing import smooth_learned_probabilities
from ..segmental.evaluate import DEFAULT_CACHE_DIR as FULLV2_CACHE
from ..segmental.evaluate import load_cache_entries
from ..segmental.inference_cache import CacheEntry
from ..segmental.pipeline import decode_regions
from ..segmental.priors import derive_priors, params_for_candidate
from ..segmental.seg_metrics import (
    CAPTURES,
    aggregate_capture,
    performer_paired_bootstrap,
    track_metrics,
)
from .calibration import expected_calibration_error, fit_calibrator
from .diagnose import THRESHOLD_SWEEP, _frame_boundary_labels, _nearest, _peak_boundaries, _prf

ML_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATES = ML_ROOT / "configs" / "boundary-calibration-v3-candidates.json"
DEFAULT_GATES = ML_ROOT / "configs" / "boundary-calibration-v3-gates.json"
DEFAULT_RUN_DIR = ML_ROOT / "runs" / "boundary-calibration-v3"

BOOTSTRAP_METRICS = [
    "rootAccuracy", "detailedAccuracy", "fragmentationRate",
    "regionsPerMinute", "meanAbsoluteBoundaryErrorMs",
]
TOLERANCE = 0.25

SEGMENTAL_FULL_KNOBS = {
    "segmentSwitchPenalty": 8.0, "maxSegmentFrames": 40, "useSegmentDuration": True,
    "segmentDurationPenalty": 4.0, "maxSegmentMinSeconds": 0.75, "useBoundary": True,
    "boundaryGain": 3.0, "switchCostFloor": 2.0, "useConfirm": True, "confirmMass": 2.0,
    "useFlicker": True,
}
EXISTING_KNOBS = {"transitionPenalty": 4.0}

# Smoothing is a property of the frozen DECODER, not of this study's candidates.
# The frozen full-v2 path applies EMA smoothing; the frozen segmental-v3
# `segmental-full` candidate declares none and decodes raw responses. Applying
# the candidate's smoothing to both would silently redefine segmental-full and
# stop the control from reproducing the published segmental-full baseline.
DECODER_SMOOTHING = {
    "full-v2-existing": {"method": "ema", "alpha": 0.65, "boundaryAlpha": 0.35},
    "segmental-full": None,
}


def candidate_cache_dir(candidate: dict, run_dir: Path) -> Path:
    """Frozen full-v2 cache for reusers; a per-candidate cache otherwise."""
    if candidate.get("reusesCheckpoint") == "full-v2":
        return FULLV2_CACHE
    return Path(run_dir) / "cache" / candidate["id"]


def _calibrated(entry: CacheEntry, calibrator, smoothing: dict | None) -> CacheEntry:
    """Smooth, then recalibrate the boundary channel only."""
    root, quality, nochord, boundary = smooth_learned_probabilities(
        entry.root, entry.quality, entry.nochord, entry.boundary, smoothing)
    boundary = np.asarray(calibrator(boundary), dtype=np.float64)
    return CacheEntry(
        identity=entry.identity, dataset_id=entry.dataset_id, split_id=entry.split_id,
        fold_id=entry.fold_id, capture=entry.capture, performance_id=entry.performance_id,
        performer_id=entry.performer_id, model_checksum=entry.model_checksum,
        feature_version=entry.feature_version, pipeline_version=entry.pipeline_version,
        hop_seconds=entry.hop_seconds, times=entry.times, root=root, quality=quality,
        nochord=nochord, boundary=boundary, bass_chroma=entry.bass_chroma,
        reference=entry.reference)


def _fit_fold_calibration(training_entries: list[CacheEntry], family: str,
                          smoothing: dict | None) -> tuple[Any, float, dict]:
    """Fit calibrator + operating threshold on training performers only."""
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for e in training_entries:
        _, _, _, boundary = smooth_learned_probabilities(
            e.root, e.quality, e.nochord, e.boundary, smoothing)
        ref_changes = [float(r["start"]) for r in e.reference[1:]]
        probs.append(np.asarray(boundary, dtype=np.float64))
        labels.append(_frame_boundary_labels(e.times, ref_changes, TOLERANCE))
    flat_probs = np.concatenate(probs) if probs else np.zeros(0)
    flat_labels = np.concatenate(labels) if labels else np.zeros(0)
    calibrator = fit_calibrator(family, flat_probs, flat_labels)

    # Operating threshold: maximise peak-picked boundary F1 on the SAME
    # training performers, after calibration.
    counts = {th: {"pred": 0, "ref": 0, "hits": 0} for th in THRESHOLD_SWEEP}
    for e in training_entries:
        _, _, _, boundary = smooth_learned_probabilities(
            e.root, e.quality, e.nochord, e.boundary, smoothing)
        boundary = np.asarray(calibrator(boundary), dtype=np.float64)
        ref_changes = [float(r["start"]) for r in e.reference[1:]]
        for th in THRESHOLD_SWEEP:
            peaks = _peak_boundaries(e.times, boundary, e.hop_seconds, threshold=th)
            counts[th]["pred"] += len(peaks)
            counts[th]["ref"] += len(ref_changes)
            counts[th]["hits"] += sum(
                1 for c in ref_changes if _nearest(c, peaks, TOLERANCE) is not None)
    scored = {th: _prf(c["hits"], c["pred"], c["ref"])["f1"] for th, c in counts.items()}
    best = max(scored, key=lambda th: scored[th])
    return calibrator, float(best), {
        "calibrator": calibrator.as_dict(),
        "operatingThreshold": float(best),
        "trainingF1AtThreshold": round(scored[best], 4),
        "trainingEceBefore": round(expected_calibration_error(flat_probs, flat_labels), 6),
        "trainingEceAfter": round(
            expected_calibration_error(calibrator(flat_probs), flat_labels), 6),
        "trainingFrames": int(flat_probs.size),
        "trainingPerformers": sorted({e.performer_id for e in training_entries}),
    }


def _boundary_quality(entries: list[CacheEntry], calibrators: dict[str, Any],
                      smoothing, thresholds: dict[str, float],
                      predicted_by_id: dict[str, list]) -> dict[str, Any]:
    """Boundary P/R/F1, agreement and calibration on the held-out entries.

    Each entry is scored with *its own fold's* calibrator and operating
    threshold -- the same ones the decoder saw -- so these numbers describe the
    configuration actually evaluated rather than an averaged approximation.
    """
    counts = {"pred": 0, "ref": 0, "hits": 0}
    agree_changes = agree_supported = 0
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for e in entries:
        _, _, _, boundary = smooth_learned_probabilities(
            e.root, e.quality, e.nochord, e.boundary, smoothing)
        boundary = np.asarray(calibrators[e.fold_id](boundary), dtype=np.float64)
        ref_changes = [float(r["start"]) for r in e.reference[1:]]
        peaks = _peak_boundaries(e.times, boundary, e.hop_seconds,
                                 threshold=thresholds[e.fold_id])
        counts["pred"] += len(peaks)
        counts["ref"] += len(ref_changes)
        counts["hits"] += sum(1 for c in ref_changes if _nearest(c, peaks, TOLERANCE) is not None)
        probs.append(boundary)
        labels.append(_frame_boundary_labels(e.times, ref_changes, TOLERANCE))
        predicted = predicted_by_id.get(f"{e.performance_id}|{e.capture}", [])
        changes = [float(r.start) for r in predicted[1:]] if predicted else []
        for c in changes:
            agree_changes += 1
            if _nearest(c, peaks, TOLERANCE) is not None:
                agree_supported += 1
    flat_probs = np.concatenate(probs) if probs else np.zeros(0)
    flat_labels = np.concatenate(labels) if labels else np.zeros(0)
    out = _prf(counts["hits"], counts["pred"], counts["ref"])
    out.update({
        "predictedBoundaries": counts["pred"],
        "referenceBoundaries": counts["ref"],
        "stateChangeBoundaryAgreement": round(agree_supported / agree_changes, 4) if agree_changes else 0.0,
        "expectedCalibrationError": round(expected_calibration_error(flat_probs, flat_labels), 6),
    })
    return out


def _region_taxonomy(entries: list[CacheEntry], predicted_by_id: dict[str, list]) -> dict[str, int]:
    """False-long / false-short / A->B->A accounting over decoded regions."""
    taxonomy: dict[str, int] = defaultdict(int)
    for e in entries:
        predicted = predicted_by_id.get(f"{e.performance_id}|{e.capture}", [])
        pred = [(float(r.start), float(r.end), r.label) for r in predicted]
        ref_changes = [float(r["start"]) for r in e.reference[1:]]
        for i in range(1, len(pred)):
            t = pred[i][0]
            duration = pred[i][1] - pred[i][0]
            is_aba = (i < len(pred) - 1 and pred[i - 1][2] == pred[i + 1][2]
                      and pred[i][2] != pred[i - 1][2])
            if _nearest(t, ref_changes, TOLERANCE) is not None:
                taxonomy["true"] += 1
            elif is_aba:
                taxonomy["false_aba"] += 1
            elif duration < 1.0:
                taxonomy["false_short"] += 1
            else:
                taxonomy["false_long"] += 1
        taxonomy["missed"] += sum(
            1 for c in ref_changes
            if _nearest(c, [p[0] for p in pred[1:]], TOLERANCE) is None)
    return dict(taxonomy)


def evaluate_candidate(candidate: dict, *, run_dir: Path) -> dict[str, Any]:
    """Evaluate one candidate under both decoders across all folds."""
    cache_dir = candidate_cache_dir(candidate, run_dir)
    entries = load_cache_entries(cache_dir)
    family = candidate.get("boundaryCalibration", "identity")
    folds = sorted({e.fold_id for e in entries})

    by_fold: dict[str, list[CacheEntry]] = defaultdict(list)
    for e in entries:
        by_fold[e.fold_id].append(e)

    priors_by_fold = {}
    for fold in folds:
        priors_by_fold[fold] = derive_priors([e for e in entries if e.fold_id != fold])

    decoders = {"full-v2-existing": ("existing", EXISTING_KNOBS),
                "segmental-full": ("segmental-full", SEGMENTAL_FULL_KNOBS)}
    results: dict[str, Any] = {}
    fold_calibration: dict[str, dict] = {}
    for decoder_id, (kind, knobs) in decoders.items():
        # Calibration is fitted against the probabilities this decoder actually
        # consumes, so the smoothing policy has to be settled before fitting.
        smoothing = DECODER_SMOOTHING[decoder_id]
        calibrators: dict[str, Any] = {}
        thresholds: dict[str, float] = {}
        for fold in folds:
            training = [e for e in entries if e.fold_id != fold]
            calibrator, threshold, record = _fit_fold_calibration(training, family, smoothing)
            calibrators[fold] = calibrator
            thresholds[fold] = threshold
            fold_calibration[f"{decoder_id}|{fold}"] = record

        rows: list[dict] = []
        predicted_by_id: dict[str, list] = {}
        runtime = 0.0
        for fold in folds:
            params = params_for_candidate(kind, knobs, priors_by_fold[fold])
            for entry in by_fold[fold]:
                decode_entry = _calibrated(entry, calibrators[fold], smoothing)
                started = time.perf_counter()
                predicted = decode_regions(kind, decode_entry, params)
                runtime += time.perf_counter() - started
                predicted_by_id[f"{entry.performance_id}|{entry.capture}"] = predicted
                row = track_metrics(
                    reference=entry.reference, predicted=predicted,
                    hop_seconds=entry.hop_seconds, root=decode_entry.root,
                    quality=decode_entry.quality, nochord=decode_entry.nochord,
                    boundary=decode_entry.boundary)
                row.update({"foldId": fold, "capture": entry.capture,
                            "performerId": entry.performer_id,
                            "performanceId": entry.performance_id})
                rows.append(row)

        by_capture = {}
        confidence = {}
        per_fold = {}
        boundary_quality = {}
        taxonomy = {}
        per_performer = {}
        for cap in CAPTURES:
            cap_rows = [r for r in rows if r["capture"] == cap]
            by_capture[cap] = aggregate_capture(cap_rows)
            confidence[cap] = performer_paired_bootstrap(rows, cap, BOOTSTRAP_METRICS)
            cap_entries = [e for e in entries if e.capture == cap]
            boundary_quality[cap] = _boundary_quality(
                cap_entries, calibrators, smoothing, thresholds, predicted_by_id)
            boundary_quality[cap]["meanOperatingThreshold"] = round(
                float(np.mean([thresholds[e.fold_id] for e in cap_entries])), 4)
            boundary_quality[cap]["thresholdByFold"] = {
                fold: round(thresholds[fold], 4) for fold in folds}
            taxonomy[cap] = _region_taxonomy(cap_entries, predicted_by_id)
            per_performer[cap] = {
                pid: aggregate_capture([r for r in cap_rows if r["performerId"] == pid])
                for pid in sorted({r["performerId"] for r in cap_rows})
            }
        for fold in folds:
            per_fold[fold] = {
                cap: aggregate_capture([r for r in rows if r["capture"] == cap and r["foldId"] == fold])
                for cap in CAPTURES}

        results[decoder_id] = {
            "byCapture": by_capture,
            "confidenceIntervals": confidence,
            "perFold": per_fold,
            "perPerformer": per_performer,
            "boundaryQuality": boundary_quality,
            "regionTaxonomy": taxonomy,
            "runtimeSeconds": round(runtime, 3),
            "consumesBoundaryHead": kind != "existing",
            "smoothing": smoothing,
        }

    return {
        "candidateId": candidate["id"],
        "cacheDir": cache_dir.name,
        "calibrationFamily": family,
        "foldCalibration": fold_calibration,
        "decoders": results,
    }
