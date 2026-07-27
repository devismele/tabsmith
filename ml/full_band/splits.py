"""Deterministic composition/artist/group-aware full-band splits."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from .manifest import ALLOWED_SPLITS, FullBandManifestError

DEFAULT_RATIOS = {
    "training": 0.70,
    "development": 0.15,
    "validation": 0.05,
    "test": 0.10,
}


class _DisjointSet:
    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _connected_groups(tracks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    ids = [track["trackId"] for track in tracks]
    groups = _DisjointSet(ids)
    indexes: dict[tuple[str, str], list[str]] = defaultdict(list)
    for track in tracks:
        for key in ("compositionId", "artistId", "splitGroupId"):
            indexes[(key, track[key])].append(track["trackId"])
    by_id = {track["trackId"]: track for track in tracks}
    for members in indexes.values():
        for member in members[1:]:
            groups.union(members[0], member)
    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for track_id in ids:
        components[groups.find(track_id)].append(by_id[track_id])
    return [
        sorted(component, key=lambda track: track["trackId"])
        for _, component in sorted(components.items())
    ]


def _validate_ratios(ratios: dict[str, float]) -> dict[str, float]:
    if set(ratios) != ALLOWED_SPLITS:
        raise FullBandManifestError(
            f"split ratios must define exactly {sorted(ALLOWED_SPLITS)}"
        )
    clean = {name: float(value) for name, value in ratios.items()}
    if any(value < 0 for value in clean.values()) or abs(sum(clean.values()) - 1.0) > 1e-9:
        raise FullBandManifestError("split ratios must be non-negative and sum to one")
    return clean


def build_grouped_splits(
    tracks: list[dict[str, Any]],
    *,
    seed: int,
    ratios: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Assign connected composition/artist/split groups without leakage."""
    if not tracks:
        raise FullBandManifestError("cannot split an empty full-band manifest")
    clean_ratios = _validate_ratios(ratios or DEFAULT_RATIOS)
    components = _connected_groups(tracks)
    total_duration = sum(track["durationSeconds"] for track in tracks)
    targets = {name: total_duration * clean_ratios[name] for name in sorted(clean_ratios)}
    durations = {name: 0.0 for name in clean_ratios}
    assignments: dict[str, str] = {}

    def component_key(component: list[dict[str, Any]]) -> tuple[str, str]:
        identity = "\0".join(track["trackId"] for track in component)
        digest = hashlib.sha256(f"{seed}\0{identity}".encode("utf-8")).hexdigest()
        return digest, identity

    ordered = sorted(
        components,
        key=lambda component: (
            -sum(track["durationSeconds"] for track in component),
            component_key(component),
        ),
    )
    for component in ordered:
        duration = sum(track["durationSeconds"] for track in component)
        declared = {track["split"] for track in component if track.get("split")}
        if len(declared) > 1:
            raise FullBandManifestError(
                "connected composition/artist/split group has conflicting preassigned splits"
            )
        split = (
            next(iter(declared))
            if declared
            else min(
                sorted(clean_ratios),
                key=lambda name: (
                    (durations[name] + duration - targets[name])
                    / max(targets[name], 1e-9),
                    durations[name] / max(targets[name], 1e-9),
                    hashlib.sha256(
                        f"{seed}\0{name}".encode("utf-8")
                    ).hexdigest(),
                ),
            )
        )
        for track in component:
            assignments[track["trackId"]] = split
        durations[split] += duration

    split_manifest = {
        "schemaVersion": 1,
        "strategy": "connected-composition-artist-group-duration-balance",
        "seed": seed,
        "ratios": clean_ratios,
        "assignments": dict(sorted(assignments.items())),
        "durationSeconds": durations,
        "componentCount": len(components),
    }
    validate_grouped_splits(tracks, split_manifest)
    return split_manifest


def validate_grouped_splits(
    tracks: list[dict[str, Any]], split_manifest: dict[str, Any]
) -> None:
    assignments = split_manifest.get("assignments", {})
    expected = {track["trackId"] for track in tracks}
    if set(assignments) != expected:
        raise FullBandManifestError("split assignments do not match manifest tracks")
    if any(split not in ALLOWED_SPLITS for split in assignments.values()):
        raise FullBandManifestError("split assignment contains an invalid split")
    for track in tracks:
        declared = track.get("split")
        assigned = assignments[track["trackId"]]
        if declared is not None and declared != assigned:
            raise FullBandManifestError(
                f"{track['trackId']} declares {declared} but split manifest assigns {assigned}"
            )

    for group_key in ("compositionId", "artistId", "splitGroupId"):
        seen: dict[str, str] = {}
        for track in tracks:
            value = track[group_key]
            split = assignments[track["trackId"]]
            if value in seen and seen[value] != split:
                raise FullBandManifestError(
                    f"{group_key} {value} leaks across {seen[value]} and {split}"
                )
            seen[value] = split
