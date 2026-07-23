"""Model-independent prediction adapters for the offline comparison (torch).

Three engines share one interface — ``predict(features) -> [ChordRegion]`` — so
the existing metrics score them identically:

* rule-based  : chroma-template-v0 (stands in offline for harmonic-context-v3,
                which lives in JS; the real hybrid wires ML probabilities into
                that decoder in-app later)
* ml-only     : temporal-baseline-v0 argmax decode
* hybrid       : ML chord/boundary probabilities blended with the bass-chroma
                root evidence already in the feature pipeline

The hybrid adapter is exercised offline on synthetic data only; it is NOT a
user-facing engine yet.
"""
from __future__ import annotations

import numpy as np
import torch

from .baselines import chroma_template_predict
from ..models.temporal_baseline import frame_labels_to_regions
from ..preprocessing.features import FeatureFrames


def predict_rule(features: FeatureFrames):
    return chroma_template_predict(features)


def _forward(model, features: FeatureFrames):
    x = torch.from_numpy(features.stacked().astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
    root = torch.softmax(out["root"][0], dim=-1).numpy()
    quality = torch.softmax(out["quality"][0], dim=-1).numpy()
    nochord = torch.sigmoid(out["nochord"][0]).numpy()
    boundary = torch.sigmoid(out["boundary"][0]).numpy()
    return root, quality, nochord, boundary


def predict_ml(model, features: FeatureFrames):
    root, quality, nochord, _boundary = _forward(model, features)
    return frame_labels_to_regions(
        features.times, root.argmax(axis=-1), quality.argmax(axis=-1), nochord, features.hop_seconds)


def predict_hybrid(model, features: FeatureFrames, bass_weight: float = 0.35):
    """ML probabilities + existing bass-root evidence -> decode."""
    root, quality, nochord, _boundary = _forward(model, features)
    # Bass chroma (12) is already aligned to root pitch classes 0..11.
    blended_root = root + bass_weight * features.bass_chroma
    return frame_labels_to_regions(
        features.times, blended_root.argmax(axis=-1), quality.argmax(axis=-1), nochord, features.hop_seconds)


ADAPTERS = {
    "rule-based (chroma-template-v0)": predict_rule,
    "ml-only (temporal-baseline-v0)": None,     # bound to a model at runtime
    "hybrid-experimental": None,
}
