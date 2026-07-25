"""Model-independent prediction adapters for the offline comparison (torch).

Three engines share one interface — ``predict(features) -> [ChordRegion]`` — so
the existing metrics score them identically:

* rule-based  : chroma-template-v0 (stands in offline for harmonic-context-v3,
                which lives in JS; the real hybrid wires ML probabilities into
                that decoder in-app later)
* ml-only     : temporal-baseline-v0 argmax decode
* hybrid proxy : ML root probabilities blended with existing bass-chroma evidence

This Python hybrid is a lightweight evaluation proxy, not an implementation-
parity copy of the TypeScript observation fusion and Tabsmith temporal decoder.
ML-only remains internal evaluation support and is not a product engine.
"""
from __future__ import annotations

import numpy as np
import torch

from .baselines import chroma_template_predict
from .decode import viterbi_decode
from ..preprocessing.features import FeatureFrames

# Self-transition penalty for Viterbi decode. Tuned on the held-out GuitarSet
# player: ~4 improves root/quality accuracy over argmax while cutting
# fragmentation; higher values reduce fragmentation further but lag boundaries.
DEFAULT_TRANSITION_PENALTY = 4.0


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


def predict_ml(model, features: FeatureFrames, transition_penalty: float = DEFAULT_TRANSITION_PENALTY):
    root, quality, nochord, _boundary = _forward(model, features)
    return viterbi_decode(features.times, root, quality, nochord, features.hop_seconds, transition_penalty)


def predict_hybrid(model, features: FeatureFrames, bass_weight: float = 0.35,
                   transition_penalty: float = DEFAULT_TRANSITION_PENALTY):
    """ML probabilities + existing bass-root evidence -> Viterbi decode."""
    root, quality, nochord, _boundary = _forward(model, features)
    # Bass chroma (12) is already aligned to root pitch classes 0..11; renormalize
    # the blend back to a distribution so the decoder's log-emissions stay valid.
    blended_root = root + bass_weight * features.bass_chroma
    blended_root = blended_root / blended_root.sum(axis=-1, keepdims=True)
    return viterbi_decode(features.times, blended_root, quality, nochord, features.hop_seconds, transition_penalty)


ADAPTERS = {
    "rule-based (chroma-template-v0)": predict_rule,
    "ml-only (temporal-baseline-v0)": None,     # bound to a model at runtime
    "hybrid-experimental": None,
}
