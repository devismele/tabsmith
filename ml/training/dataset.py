"""Frame-aligned targets and batching for the temporal baseline (torch).

Turns (Track, FeatureFrames) into per-frame supervision:
  root (0..11 or -1), quality (0..2 or -1), no-chord (0/1),
  boundary (soft 0..1), change (hard 0/1), and stable (hard 0/1).

Root/quality are masked on no-chord frames. Boundary targets are soft — a
triangular window around each annotated region start — so a prediction a few
frames away is not penalised like one far away.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..preprocessing.features import FeatureFrames, extract_features
from ..schema import Track, parse_chord_label

QUALITY_TO_INDEX = {
    "maj": 0, "maj7": 0, "aug": 0, "sus2": 0, "sus4": 0, "5": 0, "add9": 0,
    "min": 1, "min7": 1, "dim": 1,
    "7": 2,
}


def _region_at(track: Track, time: float):
    for region in track.chords:
        if region.start <= time < region.end:
            return region
    return None


def build_frame_targets(track: Track, features: FeatureFrames, tolerance_seconds: float) -> dict:
    T = len(features.times)
    root = np.full(T, -1, dtype=np.int64)
    quality = np.full(T, -1, dtype=np.int64)
    nochord = np.zeros(T, dtype=np.float32)
    boundary = np.zeros(T, dtype=np.float32)
    change = np.zeros(T, dtype=np.float32)
    stable = np.zeros(T, dtype=np.float32)

    for i, t in enumerate(features.times):
        region = _region_at(track, float(t))
        symbol = parse_chord_label(region.label) if region else None
        if symbol is None or symbol.is_no_chord or symbol.root is None:
            nochord[i] = 1.0
            continue
        root[i] = symbol.root
        quality[i] = QUALITY_TO_INDEX.get(symbol.quality, 0)

    boundaries = [region.start for region in track.chords[1:]]
    if tolerance_seconds > 0:
        for b in boundaries:
            for i, t in enumerate(features.times):
                dist = abs(float(t) - b)
                if dist <= tolerance_seconds:
                    boundary[i] = max(boundary[i], 1.0 - dist / tolerance_seconds)

    # A hard transition target complements the tolerant boundary target. It is
    # derived from the chord identity actually supervised at adjacent frames, so
    # it stays aligned even when an annotated boundary falls between frame times.
    for i in range(1, T):
        previous = (int(root[i - 1]), int(quality[i - 1]), bool(nochord[i - 1]))
        current = (int(root[i]), int(quality[i]), bool(nochord[i]))
        if current == previous:
            stable[i] = 1.0
        else:
            change[i] = 1.0

    return {
        "root": root,
        "quality": quality,
        "nochord": nochord,
        "boundary": boundary,
        "change": change,
        "stable": stable,
    }


@dataclass
class FrameSample:
    features: np.ndarray   # (T, F)
    root: np.ndarray
    quality: np.ndarray
    nochord: np.ndarray
    boundary: np.ndarray
    change: np.ndarray | None = None
    stable: np.ndarray | None = None
    track_id: str = ""
    capture_type: str = ""
    augmentation_id: str = "clean"


def sample_from_features(track: Track, features, tolerance_seconds: float) -> FrameSample | None:
    if len(features.times) == 0:
        return None
    targets = build_frame_targets(track, features, tolerance_seconds)
    audio_name = str(getattr(track, "audio_path", "")).replace("\\", "/").lower()
    capture_type = (
        "audio_mono-pickup_mix" if audio_name.endswith("_mix.wav")
        else "audio_mono-mic" if audio_name.endswith("_mic.wav")
        else features.pipeline_version
    )
    return FrameSample(
        features=features.stacked().astype(np.float32),
        root=targets["root"], quality=targets["quality"],
        nochord=targets["nochord"], boundary=targets["boundary"],
        change=targets["change"], stable=targets["stable"],
        track_id=track.track_id,
        capture_type=capture_type,
    )


def make_samples(tracks_with_audio: list[tuple[Track, np.ndarray, int]], tolerance_seconds: float) -> list[FrameSample]:
    """Assemble samples from in-memory (Track, audio, sr) tuples (synthetic path)."""
    samples: list[FrameSample] = []
    for track, audio, sr in tracks_with_audio:
        sample = sample_from_features(track, extract_features(audio, sr), tolerance_seconds)
        if sample is not None:
            samples.append(sample)
    return samples


def chunk_sample(sample: FrameSample, chunk_frames: int) -> list[FrameSample]:
    """Split a long per-track sample into fixed-length windows.

    Uniform-length chunks keep batches from being dominated by padding (Billboard
    songs are ~10x longer than GuitarSet clips) and add samples. A short tail
    (< a quarter chunk) is dropped rather than padded.
    """
    T = sample.features.shape[0]
    if chunk_frames <= 0 or T <= chunk_frames:
        return [sample]
    chunks: list[FrameSample] = []
    for start in range(0, T, chunk_frames):
        end = min(start + chunk_frames, T)
        if end - start < max(1, chunk_frames // 4):
            continue
        chunks.append(FrameSample(
            features=sample.features[start:end], root=sample.root[start:end],
            quality=sample.quality[start:end], nochord=sample.nochord[start:end],
            boundary=sample.boundary[start:end],
            change=None if sample.change is None else sample.change[start:end].copy(),
            stable=None if sample.stable is None else sample.stable[start:end].copy(),
            track_id=sample.track_id,
            capture_type=sample.capture_type,
            augmentation_id=f"{sample.augmentation_id}:chunk-{start}-{end}",
        ))
    return chunks


def pitch_shift_sample(sample: FrameSample, semitones: int) -> FrameSample:
    """Transpose a sample by rolling chroma+bass-chroma and the root targets.

    Cheap, musically valid augmentation: pitch class c -> c+semitones. Quality,
    no-chord, boundary, and the energy channel are unchanged.
    """
    if semitones % 12 == 0:
        return sample
    features = sample.features.copy()
    features[:, 0:12] = np.roll(features[:, 0:12], semitones, axis=1)    # harmonic chroma
    features[:, 12:24] = np.roll(features[:, 12:24], semitones, axis=1)  # bass chroma
    root = sample.root.copy()
    voiced = root >= 0
    root[voiced] = (root[voiced] + semitones) % 12
    return FrameSample(features=features, root=root, quality=sample.quality.copy(),
                       nochord=sample.nochord.copy(), boundary=sample.boundary.copy(),
                       change=None if sample.change is None else sample.change.copy(),
                       stable=None if sample.stable is None else sample.stable.copy(),
                       track_id=sample.track_id,
                       capture_type=sample.capture_type,
                       augmentation_id=f"{sample.augmentation_id}:pitch-{semitones:+d}")


def make_samples_from_tracks(tracks: list[Track], tolerance_seconds: float,
                             chunk_frames: int = 0, pitch_shifts: tuple[int, ...] = (),
                             audio_feature: str = "numpy-chroma-v1",
                             augmentation_manifest: dict | None = None) -> list[FrameSample]:
    """Assemble samples from real Track objects — audio *or* precomputed features.

    Optionally chunk long tracks into ``chunk_frames`` windows and add pitch-shifted
    copies for each ``pitch_shifts`` semitone offset. ``audio_feature`` selects the
    audio pipeline (``numpy-chroma-v1`` or app-identical ``harmony-features-v1``).
    Skips tracks that are neither audio- nor feature-trainable, and any whose source
    can't be read, so a partial local acquisition still trains.
    """
    from ..preprocessing.feature_source import frames_for_track

    samples: list[FrameSample] = []
    for track in tracks:
        if not track.is_trainable():
            continue
        try:
            features = frames_for_track(track, audio_feature=audio_feature)
        except Exception:
            continue
        sample = sample_from_features(track, features, tolerance_seconds)
        if sample is None:
            continue
        source_samples = [sample]
        if augmentation_manifest is not None:
            from .augmentation import build_augmented_samples
            source_samples = build_augmented_samples(sample, augmentation_manifest)
        for source_sample in source_samples:
            for chunk in chunk_sample(source_sample, chunk_frames):
                samples.append(chunk)
                for semitones in pitch_shifts:
                    samples.append(pitch_shift_sample(chunk, semitones))
    return samples


def collate(batch: list[FrameSample]) -> dict[str, torch.Tensor]:
    max_t = max(sample.features.shape[0] for sample in batch)
    feat_dim = batch[0].features.shape[1]
    b = len(batch)
    feats = torch.zeros(b, max_t, feat_dim, dtype=torch.float32)
    root = torch.full((b, max_t), -1, dtype=torch.long)
    quality = torch.full((b, max_t), -1, dtype=torch.long)
    nochord = torch.zeros(b, max_t, dtype=torch.float32)
    boundary = torch.zeros(b, max_t, dtype=torch.float32)
    change = torch.zeros(b, max_t, dtype=torch.float32)
    stable = torch.zeros(b, max_t, dtype=torch.float32)
    pad_mask = torch.zeros(b, max_t, dtype=torch.float32)
    for i, sample in enumerate(batch):
        t = sample.features.shape[0]
        feats[i, :t] = torch.from_numpy(sample.features)
        root[i, :t] = torch.from_numpy(sample.root)
        quality[i, :t] = torch.from_numpy(sample.quality)
        nochord[i, :t] = torch.from_numpy(sample.nochord)
        boundary[i, :t] = torch.from_numpy(sample.boundary)
        if sample.change is not None:
            change[i, :t] = torch.from_numpy(sample.change)
        else:
            identity_change = (
                (sample.root[1:] != sample.root[:-1])
                | (sample.quality[1:] != sample.quality[:-1])
                | (sample.nochord[1:] != sample.nochord[:-1])
            )
            change[i, 1:t] = torch.from_numpy(identity_change.astype(np.float32))
        if sample.stable is not None:
            stable[i, :t] = torch.from_numpy(sample.stable)
        elif t > 1:
            stable[i, 1:t] = 1.0 - change[i, 1:t]
        pad_mask[i, :t] = 1.0
    return {"features": feats, "root": root, "quality": quality,
            "nochord": nochord, "boundary": boundary, "change": change,
            "stable": stable, "pad_mask": pad_mask}


def compute_class_weights(samples: list[FrameSample]) -> dict[str, torch.Tensor]:
    root_counts = np.ones(12, dtype=np.float64)
    quality_counts = np.ones(3, dtype=np.float64)
    for sample in samples:
        for r in sample.root[sample.root >= 0]:
            root_counts[r] += 1
        for q in sample.quality[sample.quality >= 0]:
            quality_counts[q] += 1
    root_w = root_counts.sum() / (len(root_counts) * root_counts)
    quality_w = quality_counts.sum() / (len(quality_counts) * quality_counts)
    return {"root": torch.tensor(root_w, dtype=torch.float32),
            "quality": torch.tensor(quality_w, dtype=torch.float32)}
