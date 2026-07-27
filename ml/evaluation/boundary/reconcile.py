"""Phase 4 baseline reconciliation for the boundary-calibration study.

Before any training code is touched, prove that the inputs this study inherits
are exactly the ones the segmental-v3 study used:

* only p01-p05 appear, and p00 appears nowhere in the cache (by performer id
  *and* by the ``00_`` performance-id prefix, so a mislabelled entry is caught)
* microphone and pickup captures stay paired for every performance
* every cached entry's model checksum matches the fold checkpoint on disk
* the fold checkpoints are the ones the previous study recorded
* cached inference equals freshly recomputed inference on a sampled subset

The last check is the expensive one and is sampled by default; it is the only
part that needs audio, so it degrades to ``skipped`` when GuitarSet is not
locally available rather than failing the whole reconciliation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..segmental.build_cache import FULLV2_CHECKPOINTS
from ..segmental.evaluate import DEFAULT_CACHE_DIR, load_cache_entries

EXPECTED_PERFORMERS = ("guitarset-p01", "guitarset-p02", "guitarset-p03",
                       "guitarset-p04", "guitarset-p05")
FORBIDDEN_PERFORMER = "guitarset-p00"
CAPTURES = ("audio_mono-mic", "audio_mono-pickup_mix")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_p00_sealed(entries: list[Any]) -> dict[str, Any]:
    by_performer = sorted({e.performer_id for e in entries})
    forbidden_by_id = [e.performance_id for e in entries if e.performer_id == FORBIDDEN_PERFORMER]
    # GuitarSet performance ids are "<performer index>_<piece>"; p00's start "00_".
    forbidden_by_prefix = [e.performance_id for e in entries
                           if str(e.performance_id).startswith("00_")]
    return {
        "performers": by_performer,
        "expectedPerformers": list(EXPECTED_PERFORMERS),
        "performersMatch": by_performer == sorted(EXPECTED_PERFORMERS),
        "p00EntriesByPerformerId": len(forbidden_by_id),
        "p00EntriesByPerformanceIdPrefix": len(forbidden_by_prefix),
        "p00Sealed": not forbidden_by_id and not forbidden_by_prefix,
    }


def check_capture_pairing(entries: list[Any]) -> dict[str, Any]:
    seen: dict[str, set[str]] = {}
    for e in entries:
        seen.setdefault(e.performance_id, set()).add(e.capture)
    unpaired = sorted(pid for pid, caps in seen.items() if set(caps) != set(CAPTURES))
    fold_of: dict[str, set[str]] = {}
    for e in entries:
        fold_of.setdefault(e.performance_id, set()).add(e.fold_id)
    split_across_folds = sorted(pid for pid, folds in fold_of.items() if len(folds) > 1)
    return {
        "performances": len(seen),
        "unpairedPerformances": unpaired,
        "allPaired": not unpaired,
        "performancesSplitAcrossFolds": split_across_folds,
        "pairingHeldWithinFold": not split_across_folds,
    }


def check_checkpoints(entries: list[Any], checkpoints_root: Path = FULLV2_CHECKPOINTS) -> dict[str, Any]:
    """Cached model checksums must equal each fold checkpoint's recorded checksum."""
    cached: dict[str, set[str]] = {}
    for e in entries:
        cached.setdefault(e.fold_id, set()).add(e.model_checksum)

    folds = {}
    consistent = True
    for fold_id, checksums in sorted(cached.items()):
        metadata_path = Path(checkpoints_root) / fold_id / "full-v2" / "model.metadata.json"
        record: dict[str, Any] = {
            "cachedChecksums": sorted(checksums),
            "singleCheckpointPerFold": len(checksums) == 1,
        }
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            record["checkpointChecksum"] = metadata.get("checksum")
            record["modelVersion"] = metadata.get("modelVersion")
            record["featureVersion"] = metadata.get("featureVersion")
            record["parameterCount"] = metadata.get("parameterCount")
            record["weightFileSha256"] = _sha256_file(metadata_path.with_name("model.pt"))
            record["matches"] = metadata.get("checksum") in checksums
        else:
            record["checkpointChecksum"] = None
            record["matches"] = False
        consistent = consistent and record["matches"] and record["singleCheckpointPerFold"]
        folds[fold_id] = record
    return {"folds": folds, "allFoldsMatch": consistent}


def check_cache_equality(entries: list[Any], *, sample: int = 6,
                         checkpoints_root: Path = FULLV2_CHECKPOINTS) -> dict[str, Any]:
    """Recompute inference for a deterministic sample and compare to the cache."""
    from ..segmental.build_cache import _base_track_id, _capture, _load_tracks
    from ..temporal_v2_ablation import _resolve_data_path

    annotations = _resolve_data_path(None, "TABSMITH_GUITARSET_ANNOTATIONS")
    mic = _resolve_data_path(None, "TABSMITH_GUITARSET_MIC_AUDIO")
    pickup = _resolve_data_path(None, "TABSMITH_GUITARSET_PICKUP_AUDIO")
    if not annotations or not mic or not pickup:
        return {"status": "skipped", "reason": "GuitarSet paths unavailable (TABSMITH_GUITARSET_*)"}

    import numpy as np

    from ...preprocessing.feature_source import frames_for_track
    from ...training.checkpoint import load_checkpoint
    from ..adapters import forward_probabilities

    # Deterministic sample: first N entries in identity order, spread over folds.
    ordered = sorted(entries, key=lambda e: (e.fold_id, e.capture, e.performance_id))
    by_fold: dict[str, list[Any]] = {}
    for e in ordered:
        by_fold.setdefault(e.fold_id, []).append(e)
    chosen: list[Any] = []
    index = 0
    while len(chosen) < sample and any(len(v) > index for v in by_fold.values()):
        for fold in sorted(by_fold):
            if len(by_fold[fold]) > index and len(chosen) < sample:
                chosen.append(by_fold[fold][index])
        index += 1

    wanted = {(e.performance_id, e.capture) for e in chosen}
    tracks = {}
    for track in _load_tracks(annotations, {"audio_mono-mic": mic, "audio_mono-pickup_mix": pickup}):
        key = (_base_track_id(track).replace("guitarset-", ""), _capture(track))
        if key in wanted:
            tracks[key] = track

    compared = []
    models: dict[str, Any] = {}
    for entry in chosen:
        key = (entry.performance_id, entry.capture)
        track = tracks.get(key)
        if track is None:
            compared.append({"performanceId": entry.performance_id, "capture": entry.capture,
                             "status": "track-not-found"})
            continue
        if entry.fold_id not in models:
            models[entry.fold_id] = load_checkpoint(
                Path(checkpoints_root) / entry.fold_id / "full-v2" / "model.pt")
        model, metadata = models[entry.fold_id]
        features = frames_for_track(track, audio_feature=entry.pipeline_version)
        root, quality, nochord, boundary = forward_probabilities(model, features)
        deltas = {
            "root": float(np.max(np.abs(root - entry.root))),
            "quality": float(np.max(np.abs(quality - entry.quality))),
            "nochord": float(np.max(np.abs(nochord - entry.nochord))),
            "boundary": float(np.max(np.abs(boundary - entry.boundary))),
        }
        compared.append({
            "performanceId": entry.performance_id,
            "capture": entry.capture,
            "foldId": entry.fold_id,
            "checksumMatches": metadata.get("checksum") == entry.model_checksum,
            "maxAbsoluteDelta": round(max(deltas.values()), 12),
            "perHead": {k: round(v, 12) for k, v in deltas.items()},
            "status": "compared",
        })

    successes = [c for c in compared if c["status"] == "compared"]
    return {
        "status": "compared" if successes else "unavailable",
        "sampled": len(compared),
        "maxAbsoluteDelta": round(max((c["maxAbsoluteDelta"] for c in successes), default=0.0), 12),
        "allWithinTolerance": all(c["maxAbsoluteDelta"] <= 1e-6 for c in successes) if successes else False,
        "allChecksumsMatch": all(c["checksumMatches"] for c in successes) if successes else False,
        "entries": compared,
    }


def reconcile(*, cache_dir: Path = DEFAULT_CACHE_DIR, sample: int = 6) -> dict[str, Any]:
    entries = load_cache_entries(Path(cache_dir))
    sealing = check_p00_sealed(entries)
    pairing = check_capture_pairing(entries)
    checkpoints = check_checkpoints(entries)
    equality = check_cache_equality(entries, sample=sample)
    blocking = [
        sealing["p00Sealed"],
        sealing["performersMatch"],
        pairing["allPaired"],
        pairing["pairingHeldWithinFold"],
        checkpoints["allFoldsMatch"],
    ]
    # Cache equality only blocks when it actually ran.
    if equality["status"] == "compared":
        blocking.append(equality["allWithinTolerance"] and equality["allChecksumsMatch"])
    return {
        "cacheEntries": len(entries),
        "folds": sorted({e.fold_id for e in entries}),
        "sealing": sealing,
        "pairing": pairing,
        "checkpoints": checkpoints,
        "cacheEquality": equality,
        "reconciled": all(blocking),
    }
