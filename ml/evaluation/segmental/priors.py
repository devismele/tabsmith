"""Fold-local decoder priors, derived from development-fold training partitions.

For each leave-one-performer-out fold the decoder parameters must be derived
without the held-out performer and without p00. Duration priors come purely from
the *other* model-selection performers' reference annotations (model-independent).
Boundary-reliability and posterior-margin priors come from those same other
performers' cached full-v2 responses (each cached once from its own held-out
fold), so a single inference cache is reused and the held-out performer's data
never fits its own decoder.

All derived values are deterministic statistics (duration/boundary/margin
quantiles + a short-chord rate) so a candidate configuration is reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np

from .decoders import DecoderParams
from .inference_cache import CacheEntry


@dataclass
class Priors:
    performerCount: int
    referenceRegionCount: int
    durationP10Seconds: float
    durationP25Seconds: float
    durationMedianSeconds: float
    shortRegionRateUnder500ms: float
    shortRegionRateUnder1s: float
    boundaryP50: float
    boundaryP75: float
    boundaryP90: float
    marginP10: float
    marginP25: float
    marginMedian: float
    medianHopSeconds: float

    def as_dict(self) -> dict:
        return asdict(self)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def derive_priors(entries: Iterable[CacheEntry]) -> Priors:
    entries = list(entries)
    if not entries:
        raise ValueError("cannot derive priors from zero entries")
    performers = {e.performer_id for e in entries}
    durations: list[float] = []
    for e in entries:
        for region in e.reference:
            durations.append(float(region["end"]) - float(region["start"]))
    boundary_all: list[float] = []
    margins: list[float] = []
    hops: list[float] = []
    for e in entries:
        boundary_all.extend(e.boundary.tolist())
        hops.append(e.hop_seconds)
        chord = (
            e.root[:, :, None] * e.quality[:, None, :] * (1.0 - e.nochord[:, None, None])
        ).reshape(len(e.root), -1)
        states = np.concatenate([chord, e.nochord[:, None]], axis=1)
        srt = np.sort(states, axis=1)
        margins.extend((srt[:, -1] - srt[:, -2]).tolist())
    total = len(durations)
    return Priors(
        performerCount=len(performers),
        referenceRegionCount=total,
        durationP10Seconds=_quantile(durations, 0.10),
        durationP25Seconds=_quantile(durations, 0.25),
        durationMedianSeconds=_quantile(durations, 0.50),
        shortRegionRateUnder500ms=(sum(1 for d in durations if d < 0.5) / total) if total else 0.0,
        shortRegionRateUnder1s=(sum(1 for d in durations if d < 1.0) / total) if total else 0.0,
        boundaryP50=_quantile(boundary_all, 0.50),
        boundaryP75=_quantile(boundary_all, 0.75),
        boundaryP90=_quantile(boundary_all, 0.90),
        marginP10=_quantile(margins, 0.10),
        marginP25=_quantile(margins, 0.25),
        marginMedian=_quantile(margins, 0.50),
        medianHopSeconds=float(np.median(hops)) if hops else 0.25,
    )


def params_for_candidate(kind: str, knobs: dict, priors: Priors) -> DecoderParams:
    """Translate frozen candidate knobs + fold priors into concrete parameters.

    ``knobs`` are the frozen scalar dials from the candidate configuration;
    ``priors`` supply the fold-derived duration/boundary/margin statistics.
    """
    hop = max(1e-6, priors.medianHopSeconds)
    # A soft minimum dwell at the 25th-percentile reference duration: short enough
    # to keep genuine short chords reachable, long enough to suppress sub-chord
    # flicker. Capped so a slow song cannot erase fast progressions.
    min_dwell = int(round(min(priors.durationP25Seconds, knobs.get("maxMinDwellSeconds", 1.0)) / hop))
    seg_min = int(round(min(priors.durationP10Seconds, knobs.get("maxSegmentMinSeconds", 0.75)) / hop))
    p = DecoderParams(
        transition_penalty=float(knobs.get("transitionPenalty", 4.0)),
        min_dwell_frames=max(0, min_dwell) if knobs.get("useDuration") else 0,
        duration_penalty=float(knobs.get("durationPenalty", 0.0)) if knobs.get("useDuration") else 0.0,
        boundary_gain=float(knobs.get("boundaryGain", 0.0)) if knobs.get("useBoundary") else 0.0,
        switch_cost_floor=float(knobs.get("switchCostFloor", 0.5)),
        confirm_frames=int(round(priors.durationP10Seconds / hop)) if knobs.get("useConfirm") else 0,
        confirm_mass=float(knobs.get("confirmMass", 0.0)) if knobs.get("useConfirm") else 0.0,
        confirm_margin=float(priors.marginP25) if knobs.get("useConfirm") else 0.0,
        max_segment_frames=int(knobs.get("maxSegmentFrames", 40)),
        segment_switch_penalty=float(knobs.get("segmentSwitchPenalty", 4.0)),
        segment_min_frames=max(0, seg_min) if knobs.get("useSegmentDuration") else 0,
        segment_duration_penalty=float(knobs.get("segmentDurationPenalty", 0.0)) if knobs.get("useSegmentDuration") else 0.0,
        flicker_max_seconds=float(priors.durationP10Seconds) if knobs.get("useFlicker") else 0.0,
        flicker_boundary_threshold=float(priors.boundaryP75) if knobs.get("useFlicker") else 1.0,
        flicker_margin_threshold=float(priors.marginP25) if knobs.get("useFlicker") else 0.0,
    )
    return p
