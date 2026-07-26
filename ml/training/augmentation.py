"""Deterministic, manifest-driven temporal-harmony-v2 augmentation.

The pipeline operates on probability-preserving harmonic feature frames and
never writes generated audio. It provides reproducible proxies for timbre,
leakage, and imperfect separation while keeping real microphone/pickup captures
as distinct input examples. Optional future stem-backed transforms can reuse
the same recipe and seed identities.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import FrameSample


def load_augmentation_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(manifest.get("schemaVersion", 0)) != 1:
        raise ValueError("unsupported augmentation manifest schema")
    if int(manifest.get("variantsPerTrack", 0)) < 0:
        raise ValueError("variantsPerTrack must be non-negative")
    recipes = manifest.get("recipes", [])
    ids = [str(recipe.get("id", "")) for recipe in recipes]
    if not recipes or any(not recipe_id for recipe_id in ids):
        raise ValueError("augmentation manifest requires named recipes")
    if len(ids) != len(set(ids)):
        raise ValueError("augmentation recipe IDs must be unique")
    if any(float(recipe.get("weight", 0.0)) <= 0 for recipe in recipes):
        raise ValueError("augmentation recipe weights must be positive")
    return manifest


def augmentation_manifest_checksum(manifest: dict[str, Any]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def deterministic_augmentation_seed(base_seed: int, track_id: str, recipe_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{track_id}:{recipe_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _normalize_chroma(values: np.ndarray) -> np.ndarray:
    values = np.maximum(values, 0.0)
    sums = values.sum(axis=1, keepdims=True)
    return np.divide(values, sums, out=np.zeros_like(values), where=sums > 0)


def _slow_harmonic_contaminant(length: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    chroma = np.zeros((length, 12), dtype=np.float32)
    bass = np.zeros((length, 12), dtype=np.float32)
    position = 0
    while position < length:
        segment = int(rng.integers(6, 25))
        end = min(length, position + segment)
        root = int(rng.integers(0, 12))
        quality = (0, 4, 7) if rng.random() < 0.65 else (0, 3, 7)
        for interval, amplitude in zip(quality, (1.0, 0.7, 0.75)):
            chroma[position:end, (root + interval) % 12] = amplitude
        bass[position:end, root] = 1.0
        position = end
    return _normalize_chroma(chroma), _normalize_chroma(bass)


def _vocal_contaminant(length: int, rng: np.random.Generator) -> np.ndarray:
    chroma = np.zeros((length, 12), dtype=np.float32)
    pitch = int(rng.integers(0, 12))
    for index in range(length):
        if index % int(rng.integers(4, 13)) == 0:
            pitch = (pitch + int(rng.choice((-2, -1, 1, 2, 5)))) % 12
        chroma[index, pitch] = 1.0
        chroma[index, (pitch + 7) % 12] = 0.2
    return _normalize_chroma(chroma)


def _blend(original: np.ndarray, contaminant: np.ndarray, strength: float) -> np.ndarray:
    return _normalize_chroma((1.0 - strength) * original + strength * contaminant)


def _copy_sample(sample: FrameSample, *, features: np.ndarray, augmentation_id: str,
                 force_no_chord: bool = False) -> FrameSample:
    length = features.shape[0]
    if force_no_chord:
        root = np.full(length, -1, dtype=np.int64)
        quality = np.full(length, -1, dtype=np.int64)
        no_chord = np.ones(length, dtype=np.float32)
        boundary = np.zeros(length, dtype=np.float32)
        change = np.zeros(length, dtype=np.float32)
        stable = np.ones(length, dtype=np.float32)
        if length:
            stable[0] = 0.0
    else:
        root = sample.root.copy()
        quality = sample.quality.copy()
        no_chord = sample.nochord.copy()
        boundary = sample.boundary.copy()
        change = None if sample.change is None else sample.change.copy()
        stable = None if sample.stable is None else sample.stable.copy()
    return FrameSample(
        features=features.astype(np.float32),
        root=root,
        quality=quality,
        nochord=no_chord,
        boundary=boundary,
        change=change,
        stable=stable,
        track_id=sample.track_id,
        capture_type=sample.capture_type,
        augmentation_id=augmentation_id,
    )


def augment_sample(sample: FrameSample, recipe: dict[str, Any], base_seed: int) -> FrameSample:
    """Apply one recipe without mutating the clean sample."""
    recipe_id = str(recipe["id"])
    rng = np.random.default_rng(
        deterministic_augmentation_seed(base_seed, sample.track_id, recipe_id)
    )
    low, high = (float(value) for value in recipe.get("strengthRange", (0.1, 0.3)))
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError(f"{recipe_id}: strengthRange must satisfy 0 <= low <= high <= 1")
    strength = float(rng.uniform(low, high))
    kind = str(recipe["kind"])
    features = sample.features.copy().astype(np.float32)
    length = features.shape[0]
    chroma = features[:, :12]
    bass = features[:, 12:24]
    energy = features[:, 24]
    harmonic, harmonic_bass = _slow_harmonic_contaminant(length, rng)

    if kind == "compression":
        peak = float(np.max(energy)) or 1.0
        features[:, 24] = peak * np.power(np.clip(energy / peak, 0.0, 1.0), 1.0 - strength)
    elif kind == "reverb":
        for index in range(1, length):
            chroma[index] = (1.0 - strength) * chroma[index] + strength * chroma[index - 1]
            bass[index] = (1.0 - strength) * bass[index] + strength * bass[index - 1]
        features[:, :12] = _normalize_chroma(chroma)
        features[:, 12:24] = _normalize_chroma(bass)
    elif kind == "distortion":
        exponent = max(0.45, 1.0 - strength)
        features[:, :12] = _normalize_chroma(np.power(np.maximum(chroma, 0.0), exponent))
        features[:, 12:24] = _normalize_chroma(np.power(np.maximum(bass, 0.0), exponent))
        features[:, 24] = np.tanh(energy * (1.0 + 3.0 * strength))
    elif kind == "bass-leakage":
        features[:, 12:24] = _blend(bass, harmonic_bass, strength)
        features[:, :12] = _blend(chroma, harmonic, strength * 0.25)
    elif kind == "harmonic-contamination":
        features[:, :12] = _blend(chroma, harmonic, strength)
        features[:, 12:24] = _blend(bass, harmonic_bass, strength * 0.35)
    elif kind == "percussive-leakage":
        uniform = np.full_like(chroma, 1.0 / 12.0)
        features[:, :12] = _blend(chroma, uniform, strength * 0.35)
        spikes = (rng.random(length) < 0.12).astype(np.float32)
        features[:, 24] = np.maximum(energy, spikes * strength)
    elif kind == "vocal-leakage":
        vocal = _vocal_contaminant(length, rng)
        features[:, :12] = _blend(chroma, vocal, strength)
    elif kind == "separation-artifacts":
        smeared = chroma.copy()
        if length > 1:
            smeared[1:] = 0.7 * chroma[1:] + 0.3 * chroma[:-1]
        drop_mask = rng.random(chroma.shape) < (0.08 + 0.25 * strength)
        smeared[drop_mask] *= rng.uniform(0.0, 0.35)
        features[:, :12] = _normalize_chroma(smeared)
        features[:, 12:24] = _normalize_chroma(
            bass * rng.uniform(0.65, 1.0, size=bass.shape).astype(np.float32)
        )
    elif kind == "full-mixture":
        vocal = _vocal_contaminant(length, rng)
        mixed = 0.55 * harmonic + 0.25 * vocal + 0.20 / 12.0
        features[:, :12] = _blend(chroma, mixed, strength)
        features[:, 12:24] = _blend(bass, harmonic_bass, strength * 0.7)
        features[:, 24] = np.maximum(energy, rng.random(length).astype(np.float32) * strength)
    elif kind == "no-harmonic-negative":
        # Drums, unpitched vocals, and ambience only. Harmonic-instrument-only
        # material must never be converted to N by this recipe.
        noise = rng.random((length, 12)).astype(np.float32)
        features[:, :12] = _normalize_chroma(noise)
        features[:, 12:24] = _normalize_chroma(rng.random((length, 12)).astype(np.float32))
        features[:, 24] = rng.uniform(0.0, strength, size=length).astype(np.float32)
        return _copy_sample(
            sample,
            features=features,
            augmentation_id=recipe_id,
            force_no_chord=True,
        )
    else:
        raise ValueError(f"{recipe_id}: unsupported augmentation kind {kind!r}")

    return _copy_sample(sample, features=features, augmentation_id=recipe_id)


def build_augmented_samples(sample: FrameSample, manifest: dict[str, Any]) -> list[FrameSample]:
    """Return the clean sample plus a deterministic weighted recipe subset."""
    recipes = list(manifest["recipes"])
    count = min(int(manifest.get("variantsPerTrack", 0)), len(recipes))
    base_seed = int(manifest["seed"])
    rng = np.random.default_rng(
        deterministic_augmentation_seed(base_seed, sample.track_id, "recipe-selection")
    )
    weights = np.asarray([float(recipe["weight"]) for recipe in recipes], dtype=np.float64)
    weights /= weights.sum()
    selected_indices = rng.choice(len(recipes), size=count, replace=False, p=weights)
    output = [sample] if bool(manifest.get("alwaysIncludeClean", True)) else []
    output.extend(augment_sample(sample, recipes[int(index)], base_seed) for index in selected_indices)
    return output
