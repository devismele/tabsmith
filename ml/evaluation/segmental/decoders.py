"""Reusable segmental decoding components (numpy-only, torch-free).

All decoders consume the same full-v2 response — root/quality/no-chord/boundary
frame distributions plus observation timestamps — and return valid, non-
overlapping ``ChordRegion`` timelines using the exact region-emission convention
of the existing ``viterbi_decode`` (so metrics stay comparable).

Components, each independently reusable:

* ``duration_viterbi``    - transition cost grows when a chord is left before a
                            reference-derived minimum dwell (soft, never a hard
                            minimum that would delete genuine short chords).
* boundary gating         - the boundary head can *reduce* a switch cost but a
                            floor keeps it from forcing a switch on its own.
* ``confirm_chord_evidence`` - a proposed new chord must hold enough posterior
                            evidence for a number of frames / accumulated mass.
* ``semi_markov_decode``  - segment-level DP where one chord explains a whole
                            bounded-length segment rather than independent frames.
* ``reconcile_flicker``   - merges short A->B->A interruptions when B lacks both
                            boundary and chord support; keeps genuinely supported
                            short B chords.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..decode import _emissions, _state_label, viterbi_decode  # noqa: F401  (reused convention)
from ...schema import ChordRegion

_N_STATE = 36
_NUM_STATES = 37


@dataclass
class DecoderParams:
    """All decoder tunables. Fold-derived values are injected from priors; the
    remaining scalars are frozen in the candidate configuration."""

    transition_penalty: float = 4.0
    # Duration constraint (soft): extra switch cost ramps linearly from
    # ``duration_penalty`` at dwell 0 down to 0 once dwell >= min_dwell_frames.
    min_dwell_frames: int = 0
    duration_penalty: float = 0.0
    # Boundary gating: switch cost is reduced by ``boundary_gain * boundary`` but
    # never below ``switch_cost_floor`` (boundary cannot force a switch alone).
    boundary_gain: float = 0.0
    switch_cost_floor: float = 0.5
    # Chord-evidence confirmation.
    confirm_frames: int = 0
    confirm_mass: float = 0.0
    confirm_margin: float = 0.0
    # Semi-Markov segment length bound (frames) and short-segment duration prior.
    max_segment_frames: int = 40
    segment_switch_penalty: float = 4.0
    segment_min_frames: int = 0
    segment_duration_penalty: float = 0.0
    # Flicker reconciliation.
    flicker_max_seconds: float = 0.0
    flicker_boundary_threshold: float = 1.0
    flicker_margin_threshold: float = 0.0


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _regions_from_path(times: np.ndarray, path: np.ndarray, hop_seconds: float) -> list[ChordRegion]:
    times = np.asarray(times, dtype=np.float64)
    if len(times) == 0:
        return []
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


def _dur_extra(dwell: np.ndarray, min_frames: int, penalty: float) -> np.ndarray:
    """Soft short-dwell penalty: full ``penalty`` at dwell 0, zero at >=min."""
    if min_frames <= 0 or penalty <= 0.0:
        return np.zeros_like(dwell, dtype=np.float64)
    ramp = np.clip((min_frames - dwell) / float(min_frames), 0.0, 1.0)
    return penalty * ramp


# --------------------------------------------------------------------------- #
# A+B+D building blocks
# --------------------------------------------------------------------------- #
def duration_viterbi(
    times, root_prob, quality_prob, nochord_prob, boundary_prob, hop_seconds: float,
    params: DecoderParams,
) -> list[ChordRegion]:
    """Frame Viterbi with dwell-dependent (A) and optionally boundary-gated (B)
    switch costs. Reduces to the plain uniform-penalty decoder when the duration
    and boundary terms are zero."""
    times = np.asarray(times, dtype=np.float64)
    if len(times) == 0:
        return []
    E = _emissions(
        np.asarray(root_prob, dtype=np.float64),
        np.asarray(quality_prob, dtype=np.float64),
        np.asarray(nochord_prob, dtype=np.float64),
    )
    boundary = np.asarray(boundary_prob, dtype=np.float64)
    T, S = E.shape
    base = float(params.transition_penalty)
    gain = float(params.boundary_gain)
    floor = float(params.switch_cost_floor)

    dp = E[0].copy()
    dwell = np.ones(S, dtype=np.int64)
    back = np.zeros((T, S), dtype=np.int32)
    for t in range(1, T):
        # departure cost per source state i (charged on leaving i at frame t)
        cost = base + _dur_extra(dwell, params.min_dwell_frames, params.duration_penalty)
        if gain > 0.0:
            cost = cost - gain * boundary[t]
        cost = np.maximum(cost, floor)
        departure = dp - cost
        i1 = int(departure.argmax())
        v1 = departure[i1]
        masked = departure.copy()
        masked[i1] = -np.inf
        i2 = int(masked.argmax())
        v2 = masked[i2]
        best_dep = np.full(S, v1)
        best_dep[i1] = v2
        src = np.full(S, i1, dtype=np.int32)
        src[i1] = i2
        stay = dp  # staying costs 0
        take_stay = stay >= best_dep
        dp = np.where(take_stay, stay, best_dep) + E[t]
        dwell = np.where(take_stay, dwell + 1, 1)
        back[t] = np.where(take_stay, np.arange(S, dtype=np.int32), src)
    path = np.zeros(T, dtype=np.int32)
    path[-1] = int(dp.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return _regions_from_path(times, path, hop_seconds)


# --------------------------------------------------------------------------- #
# D. semi-Markov (segment-level) decoding
# --------------------------------------------------------------------------- #
def semi_markov_decode(
    times, root_prob, quality_prob, nochord_prob, boundary_prob, hop_seconds: float,
    params: DecoderParams, *, use_boundary: bool = False,
) -> list[ChordRegion]:
    """Segment-level DP: each candidate chord explains a whole bounded segment.

    ``best[t]`` = best score for covering frames ``[0, t)`` ending a segment at
    ``t``. Segment emission sums come from a per-state prefix sum in O(1). A
    short-segment duration prior discourages tiny segments; a per-segment switch
    penalty discourages many segments; when ``use_boundary`` the switch penalty
    at a segment start is reduced by boundary evidence (never below a floor)."""
    times = np.asarray(times, dtype=np.float64)
    if len(times) == 0:
        return []
    E = _emissions(
        np.asarray(root_prob, dtype=np.float64),
        np.asarray(quality_prob, dtype=np.float64),
        np.asarray(nochord_prob, dtype=np.float64),
    )
    boundary = np.asarray(boundary_prob, dtype=np.float64)
    T, S = E.shape
    L = max(1, int(params.max_segment_frames))
    prefix = np.zeros((T + 1, S), dtype=np.float64)
    np.cumsum(E, axis=0, out=prefix[1:])

    switch_pen = float(params.segment_switch_penalty)
    gain = float(params.boundary_gain) if use_boundary else 0.0
    floor = float(params.switch_cost_floor)

    NEG = -1e18
    best = np.full(T + 1, NEG)
    best[0] = 0.0
    back_start = np.full(T + 1, -1, dtype=np.int64)
    back_state = np.full(T + 1, -1, dtype=np.int64)
    for t in range(1, T + 1):
        s_lo = max(0, t - L)
        for s in range(s_lo, t):
            if best[s] <= NEG / 2:
                continue
            seg_len = t - s
            seg_scores = prefix[t] - prefix[s]           # (S,) emission over [s,t)
            dur_extra = 0.0
            if params.segment_min_frames > 0 and params.segment_duration_penalty > 0.0:
                ramp = max(0.0, (params.segment_min_frames - seg_len) / float(params.segment_min_frames))
                dur_extra = params.segment_duration_penalty * ramp
            # switch cost paid for starting a new segment at s (except the first)
            if s == 0:
                switch_cost = 0.0
            else:
                switch_cost = switch_pen
                if gain > 0.0:
                    switch_cost = max(floor, switch_cost - gain * boundary[s])
            cand = best[s] + seg_scores - dur_extra - switch_cost
            j = int(cand.argmax())
            score = cand[j]
            if score > best[t]:
                best[t] = score
                back_start[t] = s
                back_state[t] = j
    # backtrace segments -> frame path
    path = np.zeros(T, dtype=np.int32)
    t = T
    while t > 0:
        s = int(back_start[t])
        j = int(back_state[t])
        path[s:t] = j
        t = s
    return _regions_from_path(times, path, hop_seconds)


# --------------------------------------------------------------------------- #
# C. chord-evidence confirmation (post-process on regions)
# --------------------------------------------------------------------------- #
def _label_state_index(label: str) -> int:
    for s in range(_NUM_STATES):
        if _state_label(s) == label:
            return s
    return _N_STATE


def _region_frame_span(times: np.ndarray, hop: float, start: float, end: float) -> tuple[int, int]:
    lo = int(np.searchsorted(times, start - 1e-9, side="left"))
    hi = int(np.searchsorted(times, end - 1e-9, side="right"))
    return max(0, lo), min(len(times), max(lo + 1, hi))


def confirm_chord_evidence(
    regions: list[ChordRegion], times, root_prob, quality_prob, nochord_prob,
    hop_seconds: float, params: DecoderParams,
) -> list[ChordRegion]:
    """Absorb a region into its predecessor when the proposed chord never gains
    enough posterior support (too few confirming frames AND too little mass AND
    too small a margin over the predecessor). Genuinely supported regions stay."""
    if params.confirm_frames <= 0 and params.confirm_mass <= 0.0:
        return regions
    times = np.asarray(times, dtype=np.float64)
    E = _emissions(
        np.asarray(root_prob, dtype=np.float64),
        np.asarray(quality_prob, dtype=np.float64),
        np.asarray(nochord_prob, dtype=np.float64),
    )
    posterior = np.exp(E - E.max(axis=1, keepdims=True))
    posterior /= posterior.sum(axis=1, keepdims=True)
    out: list[ChordRegion] = []
    for region in regions:
        if not out:
            out.append(region)
            continue
        s = _label_state_index(region.label)
        lo, hi = _region_frame_span(times, hop_seconds, region.start, region.end)
        span = posterior[lo:hi, s] if hi > lo else np.zeros(0)
        prev_s = _label_state_index(out[-1].label)
        prev_mass = posterior[lo:hi, prev_s] if hi > lo else np.zeros(0)
        confirming = int(np.sum(span >= 0.5))
        mass = float(np.sum(span)) if len(span) else 0.0
        margin = float(np.mean(span - prev_mass)) if len(span) else 0.0
        supported = (
            (params.confirm_frames <= 0 or confirming >= params.confirm_frames)
            or (params.confirm_mass > 0.0 and mass >= params.confirm_mass)
            or (params.confirm_margin > 0.0 and margin >= params.confirm_margin)
        )
        if supported:
            out.append(region)
        else:
            # absorb into predecessor (extend its end)
            out[-1] = ChordRegion(out[-1].start, region.end, out[-1].label)
    return _merge_adjacent(out)


# --------------------------------------------------------------------------- #
# E. flicker reconciliation
# --------------------------------------------------------------------------- #
def reconcile_flicker(
    regions: list[ChordRegion], times, root_prob, quality_prob, nochord_prob, boundary_prob,
    hop_seconds: float, params: DecoderParams,
) -> list[ChordRegion]:
    """Merge short A->B->A flicker where B lacks *both* boundary and chord
    support. A short B backed by a real boundary and a clear chord margin is
    retained."""
    if params.flicker_max_seconds <= 0.0 or len(regions) < 3:
        return regions
    times = np.asarray(times, dtype=np.float64)
    E = _emissions(
        np.asarray(root_prob, dtype=np.float64),
        np.asarray(quality_prob, dtype=np.float64),
        np.asarray(nochord_prob, dtype=np.float64),
    )
    posterior = np.exp(E - E.max(axis=1, keepdims=True))
    posterior /= posterior.sum(axis=1, keepdims=True)
    boundary = np.asarray(boundary_prob, dtype=np.float64)

    out = list(regions)
    i = 1
    while i < len(out) - 1:
        a, b, c = out[i - 1], out[i], out[i + 1]
        if a.label == c.label and a.label != b.label and b.duration() <= params.flicker_max_seconds:
            lo, hi = _region_frame_span(times, hop_seconds, b.start, b.end)
            b_state = _label_state_index(b.label)
            a_state = _label_state_index(a.label)
            mean_boundary = float(np.mean(boundary[lo:hi])) if hi > lo else 0.0
            margin = float(np.mean(posterior[lo:hi, b_state] - posterior[lo:hi, a_state])) if hi > lo else 0.0
            boundary_supported = mean_boundary >= params.flicker_boundary_threshold
            chord_supported = margin >= params.flicker_margin_threshold
            if not boundary_supported and not chord_supported:
                # B lacks BOTH boundary and chord support -> unsupported flicker;
                # merge B and C back into A's label. A short B backed by either a
                # real boundary or a clear chord margin is retained.
                out[i - 1] = ChordRegion(a.start, c.end, a.label)
                del out[i:i + 2]
                i = max(1, i - 1)
                continue
        i += 1
    return _merge_adjacent(out)


def _merge_adjacent(regions: list[ChordRegion]) -> list[ChordRegion]:
    if not regions:
        return regions
    merged = [regions[0]]
    for region in regions[1:]:
        if region.label == merged[-1].label:
            merged[-1] = ChordRegion(merged[-1].start, region.end, merged[-1].label)
        else:
            merged.append(region)
    return merged
