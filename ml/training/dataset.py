"""Frame-aligned targets and batching for the temporal baseline (torch).

Turns (Track, FeatureFrames) into per-frame supervision:
  root (0..11 or -1), quality (0..2 or -1), no-chord (0/1), boundary (soft 0..1).

Root/quality are masked on no-chord frames. Boundary targets are soft — a
triangular window around each annotated region start — so a prediction a few
frames away is not penalised like one far away.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..preprocessing.features import FeatureFrames, extract_features
from ..schema import Track, parse_chord_label

QUALITY_TO_INDEX = {
    "maj": 0, "maj7": 0, "aug": 0, "sus2": 0, "sus4": 0, "5": 0, "add9": 0,
    "min": 1, "min7": 1, "dim": 1,
    "7": 2,
}


def _region_at(track: Track, time: float):
    for region in track.chords:
        if region.start <= time < region.end:
            return region
    return None


def build_frame_targets(track: Track, features: FeatureFrames, tolerance_seconds: float) -> dict:
    T = len(features.times)
    root = np.full(T, -1, dtype=np.int64)
    quality = np.full(T, -1, dtype=np.int64)
    nochord = np.zeros(T, dtype=np.float32)
    boundary = np.zeros(T, dtype=np.float32)

    for i, t in enumerate(features.times):
        region = _region_at(track, float(t))
        symbol = parse_chord_label(region.label) if region else None
        if symbol is None or symbol.is_no_chord or symbol.root is None:
            nochord[i] = 1.0
            continue
        root[i] = symbol.root
        quality[i] = QUALITY_TO_INDEX.get(symbol.quality, 0)

    boundaries = [region.start for region in track.chords[1:]]
    if tolerance_seconds > 0:
        for b in boundaries:
            for i, t in enumerate(features.times):
                dist = abs(float(t) - b)
                if dist <= tolerance_seconds:
                    boundary[i] = max(boundary[i], 1.0 - dist / tolerance_seconds)

    return {"root": root, "quality": quality, "nochord": nochord, "boundary": boundary}


@dataclass
class FrameSample:
    features: np.ndarray   # (T, F)
    root: np.ndarray
    quality: np.ndarray
    nochord: np.ndarray
    boundary: np.ndarray


def make_samples(tracks_with_audio: list[tuple[Track, np.ndarray, int]], tolerance_seconds: float) -> list[FrameSample]:
    samples: list[FrameSample] = []
    for track, audio, sr in tracks_with_audio:
        features = extract_features(audio, sr)
        if len(features.times) == 0:
            continue
        targets = build_frame_targets(track, features, tolerance_seconds)
        samples.append(FrameSample(
            features=features.stacked().astype(np.float32),
            root=targets["root"], quality=targets["quality"],
            nochord=targets["nochord"], boundary=targets["boundary"],
        ))
    return samples


def collate(batch: list[FrameSample]) -> dict[str, torch.Tensor]:
    max_t = max(sample.features.shape[0] for sample in batch)
    feat_dim = batch[0].features.shape[1]
    b = len(batch)
    feats = torch.zeros(b, max_t, feat_dim, dtype=torch.float32)
    root = torch.full((b, max_t), -1, dtype=torch.long)
    quality = torch.full((b, max_t), -1, dtype=torch.long)
    nochord = torch.zeros(b, max_t, dtype=torch.float32)
    boundary = torch.zeros(b, max_t, dtype=torch.float32)
    pad_mask = torch.zeros(b, max_t, dtype=torch.float32)
    for i, sample in enumerate(batch):
        t = sample.features.shape[0]
        feats[i, :t] = torch.from_numpy(sample.features)
        root[i, :t] = torch.from_numpy(sample.root)
        quality[i, :t] = torch.from_numpy(sample.quality)
        nochord[i, :t] = torch.from_numpy(sample.nochord)
        boundary[i, :t] = torch.from_numpy(sample.boundary)
        pad_mask[i, :t] = 1.0
    return {"features": feats, "root": root, "quality": quality,
            "nochord": nochord, "boundary": boundary, "pad_mask": pad_mask}


def compute_class_weights(samples: list[FrameSample]) -> dict[str, torch.Tensor]:
    root_counts = np.ones(12, dtype=np.float64)
    quality_counts = np.ones(3, dtype=np.float64)
    for sample in samples:
        for r in sample.root[sample.root >= 0]:
            root_counts[r] += 1
        for q in sample.quality[sample.quality >= 0]:
            quality_counts[q] += 1
    root_w = root_counts.sum() / (len(root_counts) * root_counts)
    quality_w = quality_counts.sum() / (len(quality_counts) * quality_counts)
    return {"root": torch.tensor(root_w, dtype=torch.float32),
            "quality": torch.tensor(quality_w, dtype=torch.float32)}
