"""Build per-candidate inference caches for the retrained boundary candidates.

Each retrained candidate needs its own four-head responses on the held-out
performers, produced by that candidate's own fold checkpoint. Checkpoint
reusers (the control and the post-hoc calibration candidate) share the frozen
full-v2 cache and are skipped here.

Resumable: existing cache entries are detected by content identity and skipped,
so an interrupted build continues rather than recomputing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..segmental.build_cache import build_cache
from ..temporal_v2_ablation import (
    DEFAULT_CONFIG,
    DEFAULT_SPLITS,
    _read_json,
    _resolve_data_path,
)
from .evaluate import DEFAULT_CANDIDATES, DEFAULT_RUN_DIR, candidate_cache_dir
from .train_candidates import trainable_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Build boundary-candidate inference caches.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--splits", default=str(DEFAULT_SPLITS))
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--mic-audio", default=None)
    parser.add_argument("--pickup-audio", default=None)
    parser.add_argument("--candidate", action="append", default=None)
    args = parser.parse_args()

    annotations = _resolve_data_path(args.annotations, "TABSMITH_GUITARSET_ANNOTATIONS")
    mic = _resolve_data_path(args.mic_audio, "TABSMITH_GUITARSET_MIC_AUDIO")
    pickup = _resolve_data_path(args.pickup_audio, "TABSMITH_GUITARSET_PICKUP_AUDIO")
    if not annotations or not mic or not pickup:
        parser.error("requires annotation/mic/pickup paths via args or TABSMITH_GUITARSET_* env vars")

    manifest = _read_json(args.candidates)
    config = _read_json(args.config)
    splits = _read_json(args.splits)
    run_dir = Path(args.run_dir).resolve()

    candidates = trainable_candidates(manifest)
    if args.candidate:
        candidates = [c for c in candidates if c["id"] in set(args.candidate)]

    summaries = {}
    for candidate in candidates:
        cache_dir = candidate_cache_dir(candidate, run_dir)
        print(f"building cache for {candidate['id']} -> {cache_dir.name}", flush=True)
        summary = build_cache(
            annotation_dir=annotations,
            audio_dirs={"audio_mono-mic": mic, "audio_mono-pickup_mix": pickup},
            config=config,
            split_manifest=splits,
            cache_dir=cache_dir,
            checkpoints_root=run_dir,
            candidate_dir=candidate["id"],
        )
        summaries[candidate["id"]] = {
            k: v for k, v in summary.items() if k != "identities"}
        print(f"  stored {summary['stored']}, skipped {summary['skipped']}, "
              f"{summary['captureCount']} captures in {summary['elapsedSeconds']}s", flush=True)

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
