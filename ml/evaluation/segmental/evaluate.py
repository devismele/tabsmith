"""Segmental-v3 frozen development evaluation orchestrator (Phases 4-8).

Reuses the single full-v2 inference cache. For each leave-one-performer-out fold
it derives decoder priors from the OTHER model-selection performers, decodes every
frozen candidate on the held-out performer, scores against reference, aggregates
per capture with performer-paired bootstrap CIs, and applies the frozen gates.
p00 is never loaded here (it is not in the cache).
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..probability_smoothing import smooth_learned_probabilities
from .gates import GATES, evaluate_gates
from .inference_cache import CacheEntry, InferenceCache
from .pipeline import decode_regions
from .priors import derive_priors, params_for_candidate
from .seg_metrics import CAPTURES, aggregate_capture, performer_paired_bootstrap, track_metrics

BOOTSTRAP_METRICS = [
    "rootAccuracy", "detailedAccuracy", "fragmentationRate",
    "regionsPerMinute", "meanAbsoluteBoundaryErrorMs",
]
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = REPO_ROOT / "ml" / "runs" / "segmental-harmony-v3" / "cache"
DEFAULT_CANDIDATES = REPO_ROOT / "ml" / "configs" / "segmental-harmony-v3-candidates.json"


def _checksum(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_candidates(path: Path) -> tuple[dict, str]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    return config, _checksum(config)


def load_cache_entries(cache_dir: Path) -> list[CacheEntry]:
    cache = InferenceCache(cache_dir)
    entries = []
    for npz in sorted(Path(cache_dir).glob("*.npz")):
        entries.append(cache.load(npz.stem))
    if not entries:
        raise ValueError(f"zero cache entries under {cache_dir}")
    return entries


def _apply_smoothing(entry: CacheEntry, smoothing: dict | None):
    root, quality, nochord, boundary = smooth_learned_probabilities(
        entry.root, entry.quality, entry.nochord, entry.boundary, smoothing)
    clone = CacheEntry(
        identity=entry.identity, dataset_id=entry.dataset_id, split_id=entry.split_id,
        fold_id=entry.fold_id, capture=entry.capture, performance_id=entry.performance_id,
        performer_id=entry.performer_id, model_checksum=entry.model_checksum,
        feature_version=entry.feature_version, pipeline_version=entry.pipeline_version,
        hop_seconds=entry.hop_seconds, times=entry.times, root=root, quality=quality,
        nochord=nochord, boundary=boundary, bass_chroma=entry.bass_chroma, reference=entry.reference)
    return clone


def run_evaluation(*, cache_dir: Path = DEFAULT_CACHE_DIR,
                   candidates_path: Path = DEFAULT_CANDIDATES) -> dict[str, Any]:
    config, candidate_checksum = load_candidates(candidates_path)
    entries = load_cache_entries(cache_dir)
    folds = sorted({e.fold_id for e in entries})
    by_fold: dict[str, list[CacheEntry]] = defaultdict(list)
    for e in entries:
        by_fold[e.fold_id].append(e)

    # per-fold priors from OTHER performers (never the held-out fold, never p00)
    fold_priors = {}
    for fold in folds:
        others = [e for e in entries if e.fold_id != fold]
        fold_priors[fold] = derive_priors(others)

    candidate_rows: dict[str, list[dict]] = defaultdict(list)
    runtime_seconds: dict[str, float] = defaultdict(float)
    audio_seconds = 0.0
    counted_audio = False

    for candidate in config["candidates"]:
        cid = candidate["id"]
        kind = candidate["kind"]
        knobs = candidate.get("knobs", {})
        smoothing = candidate.get("smoothing")
        for fold in folds:
            priors = fold_priors[fold]
            params = params_for_candidate(kind, knobs, priors)
            for entry in by_fold[fold]:
                decode_entry = _apply_smoothing(entry, smoothing) if smoothing else entry
                t0 = time.perf_counter()
                predicted = decode_regions(kind, decode_entry, params)
                runtime_seconds[cid] += time.perf_counter() - t0
                if not counted_audio:
                    audio_seconds += float(entry.times[-1] - entry.times[0]) if len(entry.times) else 0.0
                row = track_metrics(
                    reference=entry.reference, predicted=predicted, hop_seconds=entry.hop_seconds,
                    root=decode_entry.root, quality=decode_entry.quality,
                    nochord=decode_entry.nochord, boundary=decode_entry.boundary)
                row.update({
                    "candidateId": cid, "foldId": fold, "capture": entry.capture,
                    "performerId": entry.performer_id, "performanceId": entry.performance_id})
                candidate_rows[cid].append(row)
        counted_audio = True

    # aggregate + bootstrap + fold-level
    results: dict[str, Any] = {}
    for candidate in config["candidates"]:
        cid = candidate["id"]
        rows = candidate_rows[cid]
        by_capture = {cap: aggregate_capture([r for r in rows if r["capture"] == cap]) for cap in CAPTURES}
        ci = {cap: performer_paired_bootstrap(rows, cap, BOOTSTRAP_METRICS) for cap in CAPTURES}
        per_fold = {}
        for fold in folds:
            per_fold[fold] = {
                cap: aggregate_capture([r for r in rows if r["capture"] == cap and r["foldId"] == fold])
                for cap in CAPTURES}
        results[cid] = {
            "kind": candidate["kind"], "byCapture": by_capture, "confidenceIntervals": ci,
            "perFold": per_fold, "runtimeSeconds": round(runtime_seconds[cid], 3),
            "runtimePerAudioMinute": round(runtime_seconds[cid] / (audio_seconds / 60.0), 5) if audio_seconds else 0.0}

    fullv2 = results["full-v2-existing"]["byCapture"]
    gate_results = {}
    for candidate in config["candidates"]:
        cid = candidate["id"]
        if cid == "full-v2-existing":
            continue
        gate_results[cid] = evaluate_gates(cid, results[cid]["byCapture"], fullv2)

    eligible = [cid for cid, g in gate_results.items() if g["eligible"]]
    pareto = _pareto(results)
    return {
        "candidateSetId": config["candidateSetId"],
        "candidateChecksum": candidate_checksum,
        "gates": GATES,
        "folds": folds,
        "priors": {fold: fold_priors[fold].as_dict() for fold in folds},
        "audioSeconds": round(audio_seconds, 1),
        "results": results,
        "gateResults": gate_results,
        "eligibleCandidates": eligible,
        "pareto": pareto,
        "p00Accessed": False,
    }


def _pareto(results: dict[str, Any]) -> list[str]:
    """Pareto frontier on paired-capture (root up, regions/min down, frag down)."""
    def point(cid):
        mic = results[cid]["byCapture"]["audio_mono-mic"]
        pk = results[cid]["byCapture"]["audio_mono-pickup_mix"]
        return (
            (mic["rootAccuracy"] + pk["rootAccuracy"]) / 2,
            (mic["regionsPerMinute"] + pk["regionsPerMinute"]) / 2,
            (mic["fragmentationRate"] + pk["fragmentationRate"]) / 2,
        )
    cids = list(results)
    pts = {c: point(c) for c in cids}
    frontier = []
    for c in cids:
        r, rpm, frag = pts[c]
        dominated = any(
            o != c and pts[o][0] >= r and pts[o][1] <= rpm and pts[o][2] <= frag
            and (pts[o][0] > r or pts[o][1] < rpm or pts[o][2] < frag)
            for o in cids)
        if not dominated:
            frontier.append(c)
    return sorted(frontier)
