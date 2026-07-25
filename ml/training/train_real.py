"""Real-data training: python -m ml.training.train_real --guitarset-dir <path>

Trains the temporal baseline on **real** licensed data (GuitarSet audio and/or
McGill Billboard features) instead of synthetic audio. Uses artist/player-level
splits so no performer crosses train/dev, and holds out validation/test.

This is still an EXPERIMENTAL, non-shipping model: it must beat the production
`harmonic-context-v3` baseline on held-out real data (via the offline three-way
comparison) before it could ever become a selectable engine. Nothing here touches
the production app. Requires the training extra (ml/requirements-training.txt).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import make_samples_from_tracks
from .train import fit_samples
from ..preprocessing.import_billboard import scan_billboard_dir
from ..preprocessing.import_guitarset import scan_guitarset_dir
from ..schema import Track
from ..splits.make_splits import _artist_key, _hash_split, build_split_assignment

ML_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ML_ROOT / "configs" / "temporal-baseline.json"
DEFAULT_CHECKPOINT = ML_ROOT / "checkpoints" / "temporal-baseline-real-v0.pt"


def gather_tracks(guitarset_dir: str | None, billboard_dir: str | None,
                  billboard_limit: int | None = None) -> list[Track]:
    tracks: list[Track] = []
    if guitarset_dir:
        tracks += [t for t in scan_guitarset_dir(guitarset_dir) if t.is_trainable()]
    if billboard_dir:
        bb = [t for t in scan_billboard_dir(billboard_dir) if t.is_trainable()]
        if billboard_limit is not None and len(bb) > billboard_limit:
            # Deterministic stride subsample keeps artist spread (they are sorted
            # by id); Billboard songs are long so a subset stays tractable.
            stride = len(bb) / billboard_limit
            bb = [bb[int(i * stride)] for i in range(billboard_limit)]
        tracks += bb
    return tracks


def partition(tracks: list[Track], seed: int, cap: int | None) -> tuple[list[Track], list[Track], dict]:
    # GuitarSet/Billboard carry no canonical split, and the importer default
    # ("development") would be treated as authoritative and starve training.
    # Pre-assign each artist/player a deterministic hash split so it is honored
    # consistently (build_split_assignment then still runs leakage detection).
    for t in tracks:
        t.split = _hash_split(_artist_key(t.artist), seed)
    assignment = build_split_assignment(tracks, seed)
    track_split = assignment["trackSplit"]
    train = [t for t in tracks if track_split.get(t.track_id) == "training"]
    dev = [t for t in tracks if track_split.get(t.track_id) == "development"]
    if cap is not None:
        train, dev = train[:cap], dev[:max(1, cap // 4)]
    return train, dev, assignment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the temporal baseline on real licensed data.")
    parser.add_argument("--guitarset-dir", default=None)
    parser.add_argument("--billboard-dir", default=None)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-tracks-per-split", type=int, default=None,
                        help="Cap tracks per split for a fast first run (default: use all).")
    parser.add_argument("--billboard-limit", type=int, default=None,
                        help="Subsample Billboard to N tracks (they are long; keeps a first combined run tractable).")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if not args.guitarset_dir and not args.billboard_dir:
        parser.error("provide at least one of --guitarset-dir / --billboard-dir")

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.seed is not None:
        config["seed"] = args.seed
    seed = int(config.get("seed", 20260723))
    tolerance = float(config["labels"]["boundaryToleranceSeconds"])

    tracks = gather_tracks(args.guitarset_dir, args.billboard_dir, args.billboard_limit)
    if not tracks:
        parser.error("no trainable tracks found — check the acquired data paths")
    train_tracks, dev_tracks, assignment = partition(tracks, seed, args.max_tracks_per_split)

    print(f"Real tracks: {len(tracks)} trainable "
          f"(train {len(train_tracks)} / dev {len(dev_tracks)}); split counts {assignment['counts']}")
    if assignment["warnings"]:
        print("Split warnings:")
        for w in assignment["warnings"]:
            print(f"  - {w}")

    print("Extracting features (audio -> numpy-chroma-v1, features -> billboard-bothchroma-v1)...")
    train_samples = make_samples_from_tracks(train_tracks, tolerance)
    dev_samples = make_samples_from_tracks(dev_tracks, tolerance)
    print(f"Assembled samples: train {len(train_samples)} / dev {len(dev_samples)}")

    summary = fit_samples(config, train_samples, dev_samples, Path(args.checkpoint))
    print("\n" + "=" * 60)
    print("EXPERIMENTAL real-data training complete.")
    print("NOT a shipping model — must beat harmonic-context-v3 on held-out")
    print("real data (offline three-way comparison) before any promotion.")
    print("=" * 60)
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
