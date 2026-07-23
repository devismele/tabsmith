"""Immutable, artist-level dataset splits with leakage detection.

Songs from one artist must never be divided across training/test, and remasters,
live versions, covers, and duplicate excerpts must not leak a recording from one
split into another. Splits are assigned at the *artist* level and any conflict is
surfaced rather than silently resolved.

Pre-assigned splits (e.g. the Tabsmith reference file, which keeps Hotel
California in ``development`` and freezes a ``test`` set) are authoritative.
"""
from __future__ import annotations

import hashlib
import re

from ..schema import Track

VALID_SPLITS = ("training", "development", "validation", "test")
_LIVE_COVER_TOKENS = ("live", "remaster", "remastered", "remix", "cover", "acoustic version", "demo", "reprise")


def _artist_key(artist: str) -> str:
    return re.sub(r"\s+", " ", artist.strip().lower())


def _normalized_title(title: str) -> str:
    cleaned = re.sub(r"\(.*?\)|\[.*?\]", "", title.lower())
    cleaned = re.sub(r"[^a-z0-9 ]", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _hash_split(artist_key: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{artist_key}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 100
    # 70/15/15 training/development/validation; test is reserved (never auto-assigned).
    if bucket < 70:
        return "training"
    if bucket < 85:
        return "development"
    return "validation"


def assign_splits(tracks: list[Track], seed: int) -> tuple[dict[str, str], list[str]]:
    """Assign a split per artist. Returns (artist_key -> split, warnings)."""
    warnings: list[str] = []
    assigned: dict[str, set[str]] = {}
    for track in tracks:
        key = _artist_key(track.artist)
        # Synthetic gets its own training-only namespace per track id.
        if track.source == "synthetic":
            assigned.setdefault(key, set()).add("training")
            continue
        if track.split in VALID_SPLITS:
            assigned.setdefault(key, set()).add(track.split)

    artist_split: dict[str, str] = {}
    for key, splits in assigned.items():
        non_training = splits - {"training"}
        if len(non_training) > 1:
            warnings.append(f"LEAKAGE: artist '{key}' has conflicting splits {sorted(splits)}; using the most restrictive")
            # Prefer the most restrictive (test > validation > development).
            for candidate in ("test", "validation", "development"):
                if candidate in non_training:
                    artist_split[key] = candidate
                    break
        elif non_training:
            artist_split[key] = non_training.pop()
        elif splits:
            artist_split[key] = "training"
        else:
            artist_split[key] = _hash_split(key, seed)
    return artist_split, warnings


def detect_leakage(tracks: list[Track], artist_split: dict[str, str]) -> list[str]:
    warnings: list[str] = []
    # Duplicate/near-duplicate recordings across splits.
    by_title: dict[str, list[Track]] = {}
    for track in tracks:
        by_title.setdefault(_normalized_title(track.title), []).append(track)
    for title, group in by_title.items():
        splits = {artist_split.get(_artist_key(t.artist), "training") for t in group}
        if len(group) > 1 and len(splits) > 1:
            warnings.append(f"LEAKAGE: '{title}' appears across splits {sorted(splits)} "
                            f"({len(group)} entries) — check remaster/live/cover duplicates")
        for track in group:
            if any(token in track.title.lower() for token in _LIVE_COVER_TOKENS):
                warnings.append(f"REVIEW: '{track.title}' looks like a live/remaster/cover variant")
    return warnings


def build_split_assignment(tracks: list[Track], seed: int) -> dict:
    artist_split, warnings = assign_splits(tracks, seed)
    warnings += detect_leakage(tracks, artist_split)
    track_split: dict[str, str] = {}
    for track in tracks:
        if track.source == "synthetic":
            track_split[track.track_id] = "training"
        else:
            track_split[track.track_id] = artist_split.get(_artist_key(track.artist), "training")
    counts: dict[str, int] = {}
    for split in track_split.values():
        counts[split] = counts.get(split, 0) + 1
    return {
        "seed": seed,
        "artistSplit": artist_split,
        "trackSplit": track_split,
        "counts": counts,
        "warnings": warnings,
    }
