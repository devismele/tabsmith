"""Non-learned reference predictors.

``chroma-template-v0`` is a deliberately simple template-matching chord detector.
It exists to (a) exercise the feature -> prediction -> metric path end-to-end in
the foundation pass and (b) give learned-harmony-v1 a concrete floor to beat. It
is NOT the learned model and NOT a candidate for production.
"""
from __future__ import annotations

import numpy as np

from ..preprocessing.features import FeatureFrames
from ..schema import ChordRegion, pitch_class_name

BASELINE_NAME = "chroma-template-v0"

_MAJ = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=float)
_MIN = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float)


def _templates() -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    for root in range(12):
        maj = np.roll(_MAJ, root)
        minor = np.roll(_MIN, root)
        out.append((f"{pitch_class_name(root)}:maj", maj / maj.sum()))
        out.append((f"{pitch_class_name(root)}:min", minor / minor.sum()))
    return out


def chroma_template_predict(features: FeatureFrames, min_region_seconds: float = 0.5,
                            silence_energy: float = 0.01) -> list[ChordRegion]:
    templates = _templates()
    hop = features.hop_seconds
    labels: list[str] = []
    for i in range(len(features.times)):
        if features.energy[i] < silence_energy:
            labels.append("N")
            continue
        chroma = features.chroma[i]
        bass = features.bass_chroma[i]
        best_label, best_score = "N", -1e9
        for label, template in templates:
            root = int(label.split(":")[0] and _root_index(label))
            score = float(chroma @ template) + 0.25 * float(bass[root])
            if score > best_score:
                best_label, best_score = label, score
        labels.append(best_label)

    # Merge consecutive identical labels into regions.
    regions: list[ChordRegion] = []
    if not labels:
        return regions
    start_i = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start_i]:
            start = features.times[start_i] - hop / 2
            end = features.times[i - 1] + hop / 2
            regions.append(ChordRegion(round(max(0.0, start), 4), round(end, 4), labels[start_i]))
            start_i = i

    return _merge_short(regions, min_region_seconds)


def _root_index(label: str) -> int:
    from ..schema import pitch_class_index
    return pitch_class_index(label.split(":")[0]) or 0


def _merge_short(regions: list[ChordRegion], min_seconds: float) -> list[ChordRegion]:
    if not regions:
        return regions
    merged = [regions[0]]
    for region in regions[1:]:
        prev = merged[-1]
        if region.duration() < min_seconds and region.label != prev.label:
            prev.end = region.end  # absorb the flicker into the previous region
        elif region.label == prev.label:
            prev.end = region.end
        else:
            merged.append(region)
    return merged
