"""Deterministic symbolic-note to chord-region derivation."""
from __future__ import annotations

from typing import Any

from ..schema import ChordRegion, parse_chord_label
from .manifest import FullBandManifestError

PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
QUALITY_INTERVALS = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "7": (0, 4, 7, 10),
}


def normalize_chord_regions(
    regions: list[dict[str, Any]] | list[ChordRegion],
    duration: float,
) -> list[ChordRegion]:
    """Return contiguous, monotonic regions with explicit ``N`` gaps."""
    if duration <= 0:
        raise FullBandManifestError("track duration must be positive")
    clean = []
    for region in regions:
        start = float(region.start if isinstance(region, ChordRegion) else region["start"])
        end = float(region.end if isinstance(region, ChordRegion) else region["end"])
        label = region.label if isinstance(region, ChordRegion) else str(region["label"])
        if start < 0 or end <= start or end > duration + 1e-6:
            raise FullBandManifestError("chord region has an invalid range")
        if parse_chord_label(label).is_unknown:
            label = "N"
        clean.append(ChordRegion(start, min(end, duration), label))
    clean.sort(key=lambda region: (region.start, region.end, region.label))

    normalized: list[ChordRegion] = []
    cursor = 0.0
    for region in clean:
        if region.start < cursor - 1e-6:
            raise FullBandManifestError("chord regions overlap")
        if region.start > cursor + 1e-6:
            normalized.append(ChordRegion(cursor, region.start, "N"))
        start = max(cursor, region.start)
        normalized.append(ChordRegion(start, region.end, region.label))
        cursor = region.end
    if cursor < duration - 1e-6:
        normalized.append(ChordRegion(cursor, duration, "N"))
    if not normalized:
        normalized = [ChordRegion(0.0, duration, "N")]

    merged: list[ChordRegion] = []
    for region in normalized:
        canonical = parse_chord_label(region.label).detailed
        if merged and merged[-1].label == canonical and abs(merged[-1].end - region.start) <= 1e-6:
            merged[-1].end = region.end
        else:
            merged.append(ChordRegion(region.start, region.end, canonical))
    merged[0].start = 0.0
    merged[-1].end = duration
    return merged


def _segment_label(
    active_events: list[dict[str, Any]],
    *,
    minimum_distinct_pitch_classes: int,
    minimum_chord_tone_share: float,
) -> str:
    weights = [0.0] * 12
    for event in active_events:
        velocity = float(event.get("velocity", 1.0))
        for pitch in event["pitches"]:
            weights[int(pitch) % 12] += velocity
    active_classes = sum(weight > 0 for weight in weights)
    total = sum(weights)
    if active_classes < minimum_distinct_pitch_classes or total <= 0:
        return "N"

    candidates = []
    for root in range(12):
        for quality_index, (quality, intervals) in enumerate(QUALITY_INTERVALS.items()):
            tones = {(root + interval) % 12 for interval in intervals}
            chord_weight = sum(weights[pitch] for pitch in tones)
            outside_weight = total - chord_weight
            coverage = chord_weight / total
            root_support = weights[root] / total
            score = coverage + 0.12 * root_support - 0.18 * outside_weight / total
            candidates.append((score, coverage, root_support, -quality_index, -root, root, quality))
    _, coverage, _, _, _, root, quality = max(candidates)
    if coverage < minimum_chord_tone_share:
        return "N"
    return f"{PITCH_NAMES[root]}:{quality}"


def derive_chord_regions(
    symbolic_events: list[dict[str, Any]],
    duration: float,
    *,
    minimum_distinct_pitch_classes: int = 2,
    minimum_chord_tone_share: float = 0.66,
    minimum_region_seconds: float = 0.10,
) -> list[ChordRegion]:
    """Derive deterministic root/quality regions from normalized MIDI-like events.

    An upstream MIDI parser only needs to provide ``start``, ``end``, ``pitches``
    and optional ``velocity``. Empty intervals and no-note examples become ``N``.
    """
    if duration <= 0:
        raise FullBandManifestError("track duration must be positive")
    events = []
    for event in symbolic_events:
        start = float(event["start"])
        end = float(event["end"])
        pitches = [int(pitch) for pitch in event["pitches"]]
        if start < 0 or end <= start or end > duration + 1e-6:
            raise FullBandManifestError("symbolic event has an invalid time range")
        if not pitches or any(pitch < 0 or pitch > 127 for pitch in pitches):
            raise FullBandManifestError("symbolic event pitches must be MIDI values 0..127")
        events.append({
            "start": start,
            "end": min(end, duration),
            "pitches": pitches,
            "velocity": float(event.get("velocity", 1.0)),
        })
    boundaries = sorted({0.0, duration, *(
        time for event in events for time in (event["start"], event["end"])
    )})
    raw: list[ChordRegion] = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end - start <= 1e-9:
            continue
        active = [
            event for event in events
            if event["start"] < end - 1e-9 and event["end"] > start + 1e-9
        ]
        label = _segment_label(
            active,
            minimum_distinct_pitch_classes=minimum_distinct_pitch_classes,
            minimum_chord_tone_share=minimum_chord_tone_share,
        )
        if raw and raw[-1].label == label:
            raw[-1].end = end
        else:
            raw.append(ChordRegion(start, end, label))

    # Absorb tiny symbolic slivers deterministically into the longer neighbour.
    for index in range(len(raw)):
        region = raw[index]
        if region.duration() >= minimum_region_seconds or len(raw) == 1:
            continue
        previous = raw[index - 1] if index > 0 else None
        following = raw[index + 1] if index + 1 < len(raw) else None
        replacement = max(
            (candidate for candidate in (previous, following) if candidate is not None),
            key=lambda candidate: (candidate.duration(), candidate.label),
        )
        region.label = replacement.label
    return normalize_chord_regions(raw, duration)
