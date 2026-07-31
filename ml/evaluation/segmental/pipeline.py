"""Candidate -> region dispatch. Each candidate composes reusable decoder
components; every candidate receives the identical cached full-v2 response."""
from __future__ import annotations

from ..decode import viterbi_decode
from .decoders import (
    DecoderParams,
    confirm_chord_evidence,
    duration_viterbi,
    reconcile_flicker,
    semi_markov_decode,
)
from .inference_cache import CacheEntry

CANDIDATE_KINDS = (
    "existing",
    "duration-viterbi",
    "boundary-gated-viterbi",
    "duration-boundary-viterbi",
    "semi-markov-chord",
    "semi-markov-chord-boundary",
    "segmental-full",
)


def decode_regions(kind: str, entry: CacheEntry, params: DecoderParams):
    t, r, q, n, b, hop = (
        entry.times, entry.root, entry.quality, entry.nochord, entry.boundary, entry.hop_seconds,
    )
    if kind == "existing":
        return viterbi_decode(t, r, q, n, hop, params.transition_penalty)
    if kind in ("duration-viterbi", "boundary-gated-viterbi", "duration-boundary-viterbi"):
        return duration_viterbi(t, r, q, n, b, hop, params)
    if kind == "semi-markov-chord":
        return semi_markov_decode(t, r, q, n, b, hop, params, use_boundary=False)
    if kind == "semi-markov-chord-boundary":
        return semi_markov_decode(t, r, q, n, b, hop, params, use_boundary=True)
    if kind == "segmental-full":
        regions = semi_markov_decode(t, r, q, n, b, hop, params, use_boundary=True)
        regions = confirm_chord_evidence(regions, t, r, q, n, hop, params)
        regions = reconcile_flicker(regions, t, r, q, n, b, hop, params)
        return regions
    raise ValueError(f"unknown candidate kind: {kind}")
