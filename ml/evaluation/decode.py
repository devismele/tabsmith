"""Viterbi chord decoding with a self-transition penalty (torch-free, numpy only).

Frame-argmax decoding fragments badly: the model changes its mind mid-chord, so
one long reference chord is split across many predicted regions. This decodes the
per-frame probabilities globally with a switching cost — a chord only changes when
the frame evidence outweighs ``transition_penalty`` — which cuts fragmentation
*and* (empirically, on held-out GuitarSet) improves root/quality accuracy versus
argmax. Pure numpy so it runs in the foundation environment and is unit-testable
without torch.

Transition model: staying in a state costs 0, switching to any other state costs
``transition_penalty`` from the best previous state (uniform switch cost). Higher
penalty = stickier chords = less fragmentation but laggier boundaries.
"""
from __future__ import annotations

import numpy as np

from ..schema import ChordRegion, pitch_class_name

# Must match models.temporal_baseline.QUALITIES (kept torch-free by not importing it).
QUALITY_TOKENS = ("maj", "min", "7")
_N_STATE = 36              # 12 roots * 3 qualities = 36 chord states; index 36 = no-chord
_NUM_STATES = 37
_EPS = 1e-8


def _emissions(root_prob: np.ndarray, quality_prob: np.ndarray, nochord_prob: np.ndarray) -> np.ndarray:
    """(T, 37) log-emission per frame: chord states + a no-chord state."""
    T = len(nochord_prob)
    E = np.full((T, _NUM_STATES), -1e9)
    log_root = np.log(root_prob + _EPS)
    log_quality = np.log(quality_prob + _EPS)
    log_nc = np.log(nochord_prob + _EPS)
    log_chord = np.log(1.0 - nochord_prob + _EPS)
    for r in range(12):
        for q in range(3):
            E[:, r * 3 + q] = log_root[:, r] + log_quality[:, q] + log_chord
    E[:, _N_STATE] = log_nc
    return E


def _viterbi_path(E: np.ndarray, penalty: float) -> np.ndarray:
    T, S = E.shape
    dp = E[0].copy()
    back = np.zeros((T, S), dtype=np.int32)
    states = np.arange(S)
    for t in range(1, T):
        best_idx = int(dp.argmax())
        switch_score = dp[best_idx] - penalty        # cheapest way to arrive by switching
        take_stay = dp >= switch_score               # else the state kept itself
        dp = np.where(take_stay, dp, switch_score) + E[t]
        back[t] = np.where(take_stay, states, best_idx)
    path = np.zeros(T, dtype=np.int32)
    path[-1] = int(dp.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _state_label(state: int) -> str:
    if state == _N_STATE:
        return "N"
    return f"{pitch_class_name(state // 3)}:{QUALITY_TOKENS[state % 3]}"


def viterbi_decode(times, root_prob, quality_prob, nochord_prob, hop_seconds: float,
                   transition_penalty: float = 4.0) -> list[ChordRegion]:
    """Decode per-frame probabilities into merged chord regions."""
    times = np.asarray(times, dtype=np.float64)
    if len(times) == 0:
        return []
    E = _emissions(np.asarray(root_prob, dtype=np.float64),
                   np.asarray(quality_prob, dtype=np.float64),
                   np.asarray(nochord_prob, dtype=np.float64))
    path = _viterbi_path(E, float(transition_penalty))
    labels = [_state_label(int(s)) for s in path]

    regions: list[ChordRegion] = []
    start_i = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start_i]:
            start = float(times[start_i]) - hop_seconds / 2
            end = float(times[i - 1]) + hop_seconds / 2
            regions.append(ChordRegion(round(max(0.0, start), 4), round(end, 4), labels[start_i]))
            start_i = i
    return regions
