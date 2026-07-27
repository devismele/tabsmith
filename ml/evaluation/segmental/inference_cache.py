"""Reusable, decoder-agnostic full-v2 inference cache (torch-free).

Every segmental decoder candidate must receive *identical* full-v2 model output,
so we run the four-head network exactly once per (performance, capture) and reuse
the cached frame-level distributions for all candidates.

Cache identity deliberately includes every input that could change the numbers:
dataset + split identity, the held-out fold, the capture, the performance id, the
full-v2 checkpoint checksum, the feature version, and the feature pipeline
version. If any of those change, the identity changes and the stale entry is
never silently reused.

No audio is stored. Caches live under ``ml/runs`` which is git-ignored.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def cache_identity(
    *,
    dataset_id: str,
    split_id: str,
    fold_id: str,
    capture: str,
    performance_id: str,
    model_checksum: str,
    feature_version: str,
    pipeline_version: str,
) -> str:
    """Deterministic content hash for one cached inference response."""
    payload = {
        "schemaVersion": SCHEMA_VERSION,
        "datasetId": dataset_id,
        "splitId": split_id,
        "foldId": fold_id,
        "capture": capture,
        "performanceId": performance_id,
        "modelChecksum": model_checksum,
        "featureVersion": feature_version,
        "pipelineVersion": pipeline_version,
    }
    return hashlib.sha256(_canonical(payload).encode("ascii")).hexdigest()


@dataclass
class CacheEntry:
    """One immutable full-v2 response plus the reference needed to score it."""

    identity: str
    dataset_id: str
    split_id: str
    fold_id: str
    capture: str
    performance_id: str
    performer_id: str
    model_checksum: str
    feature_version: str
    pipeline_version: str
    hop_seconds: float
    times: np.ndarray            # (T,)
    root: np.ndarray             # (T, 12)
    quality: np.ndarray          # (T, 3)
    nochord: np.ndarray          # (T,)
    boundary: np.ndarray         # (T,)
    bass_chroma: np.ndarray      # (T, 12) optional bass/context evidence
    reference: list[dict[str, Any]] = field(default_factory=list)  # start/end/label

    def meta(self) -> dict[str, Any]:
        return {
            "schemaVersion": SCHEMA_VERSION,
            "identity": self.identity,
            "datasetId": self.dataset_id,
            "splitId": self.split_id,
            "foldId": self.fold_id,
            "capture": self.capture,
            "performanceId": self.performance_id,
            "performerId": self.performer_id,
            "modelChecksum": self.model_checksum,
            "featureVersion": self.feature_version,
            "pipelineVersion": self.pipeline_version,
            "hopSeconds": self.hop_seconds,
            "frameCount": int(len(self.times)),
            "reference": self.reference,
        }


class InferenceCache:
    """Content-addressed store of full-v2 responses under a git-ignored root."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, identity: str) -> Path:
        return self.root / f"{identity}.npz"

    def has(self, identity: str) -> bool:
        return self._path(identity).exists()

    def store(self, entry: CacheEntry) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(entry.identity)
        tmp = path.with_suffix(".npz.tmp")
        # savez_compressed auto-appends ".npz" to a path arg, so write via a handle.
        with tmp.open("wb") as handle:
            np.savez_compressed(
                handle,
                meta=np.frombuffer(_canonical(entry.meta()).encode("utf-8"), dtype=np.uint8),
                times=entry.times.astype(np.float64),
                root=entry.root.astype(np.float32),
                quality=entry.quality.astype(np.float32),
                nochord=entry.nochord.astype(np.float32),
                boundary=entry.boundary.astype(np.float32),
                bass_chroma=entry.bass_chroma.astype(np.float32),
            )
        tmp.replace(path)
        return path

    def load(self, identity: str) -> CacheEntry:
        with np.load(self._path(identity)) as data:
            meta = json.loads(bytes(data["meta"].tobytes()).decode("utf-8"))
            if meta["identity"] != identity:
                raise ValueError(f"cache identity mismatch: {meta['identity']} != {identity}")
            return CacheEntry(
                identity=identity,
                dataset_id=meta["datasetId"],
                split_id=meta["splitId"],
                fold_id=meta["foldId"],
                capture=meta["capture"],
                performance_id=meta["performanceId"],
                performer_id=meta["performerId"],
                model_checksum=meta["modelChecksum"],
                feature_version=meta["featureVersion"],
                pipeline_version=meta["pipelineVersion"],
                hop_seconds=float(meta["hopSeconds"]),
                times=np.asarray(data["times"], dtype=np.float64),
                root=np.asarray(data["root"], dtype=np.float64),
                quality=np.asarray(data["quality"], dtype=np.float64),
                nochord=np.asarray(data["nochord"], dtype=np.float64),
                boundary=np.asarray(data["boundary"], dtype=np.float64),
                bass_chroma=np.asarray(data["bass_chroma"], dtype=np.float64),
                reference=meta["reference"],
            )


def entries_match(a: CacheEntry, b: CacheEntry, *, atol: float = 1e-6) -> bool:
    """Structural + numerical equality used to prove cached == uncached output."""
    if a.identity != b.identity or a.performance_id != b.performance_id:
        return False
    if abs(a.hop_seconds - b.hop_seconds) > 1e-12:
        return False
    for name in ("times", "root", "quality", "nochord", "boundary", "bass_chroma"):
        left = np.asarray(getattr(a, name), dtype=np.float64)
        right = np.asarray(getattr(b, name), dtype=np.float64)
        if left.shape != right.shape or not np.allclose(left, right, atol=atol, rtol=0.0):
            return False
    return a.reference == b.reference
