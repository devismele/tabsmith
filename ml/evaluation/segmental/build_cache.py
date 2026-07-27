"""Build the reusable full-v2 inference cache for the segmental-v3 study.

Runs the frozen full-v2 network exactly once per held-out (performance, capture)
using that fold's own checkpoint, and stores the four-head frame distributions so
every decoder candidate later reuses identical model output. Feature extraction is
the only heavy cost; inference is milliseconds. Resumable: existing cache entries
are skipped.

p00 never enters: only the model-selection performers (p01-p05) are cached, and
each performance is cached from the fold in which it is held out.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ..temporal_v2_ablation import (
    DEFAULT_CONFIG,
    DEFAULT_SPLITS,
    _base_track_id,
    _capture,
    _load_tracks,
    _read_json,
    _resolve_data_path,
)
from .inference_cache import CacheEntry, InferenceCache, cache_identity

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "ml" / "runs" / "segmental-harmony-v3" / "cache"
FULLV2_CHECKPOINTS = Path(__file__).resolve().parents[3] / "ml" / "runs" / "temporal-harmony-v2"


def build_cache(
    *,
    annotation_dir: Path,
    audio_dirs: dict[str, Path],
    config: dict,
    split_manifest: dict,
    cache_dir: Path,
    checkpoints_root: Path = FULLV2_CHECKPOINTS,
) -> dict:
    from ...preprocessing.feature_source import frames_for_track
    from ...training.checkpoint import load_checkpoint
    from ..adapters import forward_probabilities

    pipeline = config["features"]["pipelineVersion"]
    dataset_id = split_manifest["datasetIdentity"]["datasetId"]
    split_id = split_manifest["splitId"]
    allowed = {item["performerId"] for item in split_manifest["modelSelectionPerformers"]}
    tracks = [t for t in _load_tracks(annotation_dir, audio_dirs) if t.artist in allowed]
    cache = InferenceCache(cache_dir)

    by_fold: dict[str, list] = {}
    for fold in split_manifest["folds"]:
        validation = set(fold["validationPerformers"])
        by_fold[fold["foldId"]] = [t for t in tracks if t.artist in validation]

    stored = skipped = 0
    identities: list[dict] = []
    start = time.time()
    for fold in split_manifest["folds"]:
        fold_id = fold["foldId"]
        ckpt_path = checkpoints_root / fold_id / "full-v2" / "model.pt"
        model = meta = None
        for track in by_fold[fold_id]:
            capture = _capture(track)
            performance_id = _base_track_id(track).replace("guitarset-", "")
            # identity needs the model checksum -> load checkpoint lazily (once/fold)
            if model is None:
                model, meta = load_checkpoint(ckpt_path)
            identity = cache_identity(
                dataset_id=dataset_id, split_id=split_id, fold_id=fold_id, capture=capture,
                performance_id=performance_id, model_checksum=meta["checksum"],
                feature_version=meta["featureVersion"], pipeline_version=pipeline,
            )
            identities.append({
                "foldId": fold_id, "capture": capture, "performanceId": performance_id,
                "performerId": track.artist, "identity": identity,
            })
            if cache.has(identity):
                skipped += 1
                continue
            feats = frames_for_track(track, audio_feature=pipeline)
            root, quality, nochord, boundary = forward_probabilities(model, feats)
            entry = CacheEntry(
                identity=identity, dataset_id=dataset_id, split_id=split_id, fold_id=fold_id,
                capture=capture, performance_id=performance_id, performer_id=track.artist,
                model_checksum=meta["checksum"], feature_version=meta["featureVersion"],
                pipeline_version=pipeline, hop_seconds=feats.hop_seconds,
                times=np.asarray(feats.times), root=root, quality=quality, nochord=nochord,
                boundary=boundary, bass_chroma=feats.bass_chroma,
                reference=[{"start": r.start, "end": r.end, "label": r.label} for r in track.chords],
            )
            cache.store(entry)
            stored += 1
            if stored % 50 == 0:
                print(f"  cached {stored} (skipped {skipped}) at {time.time() - start:.0f}s", flush=True)
    return {
        "datasetId": dataset_id, "splitId": split_id, "pipelineVersion": pipeline,
        "captureCount": len(identities), "stored": stored, "skipped": skipped,
        "elapsedSeconds": round(time.time() - start, 1), "identities": identities,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the segmental-v3 full-v2 inference cache.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS))
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--mic-audio", default=None)
    parser.add_argument("--pickup-audio", default=None)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    args = parser.parse_args()

    config = _read_json(args.config)
    split_manifest = _read_json(args.splits)
    annotation_dir = _resolve_data_path(args.annotations, "TABSMITH_GUITARSET_ANNOTATIONS")
    mic = _resolve_data_path(args.mic_audio, "TABSMITH_GUITARSET_MIC_AUDIO")
    pickup = _resolve_data_path(args.pickup_audio, "TABSMITH_GUITARSET_PICKUP_AUDIO")
    if not annotation_dir or not mic or not pickup:
        parser.error("requires annotation/mic/pickup paths via args or TABSMITH_GUITARSET_* env vars")
    summary = build_cache(
        annotation_dir=annotation_dir,
        audio_dirs={"audio_mono-mic": mic, "audio_mono-pickup_mix": pickup},
        config=config, split_manifest=split_manifest, cache_dir=Path(args.cache_dir),
    )
    print(
        f"cache build complete: stored {summary['stored']}, skipped {summary['skipped']}, "
        f"{summary['captureCount']} captures in {summary['elapsedSeconds']}s"
    )


if __name__ == "__main__":
    main()
