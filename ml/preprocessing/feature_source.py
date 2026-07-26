"""Resolve a Track to FeatureFrames, whichever way its data is available.

One entry point (`frames_for_track`) so the training assembler and evaluation
don't care whether a track is backed by raw audio (GuitarSet, synthetic) or by
precomputed features (McGill Billboard). Audio tracks go through the canonical
`numpy-chroma-v1` pipeline; feature tracks through `billboard-bothchroma-v1`.

Real GuitarSet audio is 44.1 kHz and may not be 16-bit PCM, which the stdlib
foundation reader rejects. So audio loading prefers librosa (already in the
training extra) and falls back to the stdlib 16-bit reader for synthetic WAVs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .app_features import extract_app_features
from .billboard_features import load_billboard_bothchroma
from .features import FeatureFrames, extract_features, read_wav_mono
from ..schema import Track

# Audio feature pipelines selectable for audio-backed tracks. "numpy-chroma-v1" is
# the dependency-light experimental pipeline; "harmony-features-v1" matches the
# app's spectralChroma exactly (verified identical) so a model trained on it can be
# wired into the app.
AUDIO_FEATURE_PIPELINES = ("numpy-chroma-v1", "harmony-features-v1")

# The app decodes/analyzes audio at 22.05 kHz; harmony-features-v1 is computed at
# this rate for train/serve consistency (see analysis pipeline resampleToMono).
APP_SAMPLE_RATE = 22050


def load_audio_mono(path: str | Path, target_sr: int = 22050) -> tuple[np.ndarray, int]:
    """Mono float32 audio. librosa if available (any bit depth / sample rate),
    else the stdlib 16-bit reader used by the synthetic pipeline."""
    try:
        import librosa  # type: ignore
        samples, sr = librosa.load(str(path), sr=target_sr, mono=True)
        return samples.astype(np.float32), int(sr)
    except ImportError:
        return read_wav_mono(path)


def frames_for_track(track: Track, audio_feature: str = "numpy-chroma-v1", **audio_kwargs) -> FeatureFrames:
    """FeatureFrames for a trainable track, dispatched on availability.

    ``audio_feature`` selects the pipeline for audio-backed tracks:
    ``numpy-chroma-v1`` (default) or ``harmony-features-v1`` (app-identical).
    Precomputed-feature tracks (Billboard) always use their own pipeline.
    """
    if track.is_audio_trainable():
        if audio_feature == "harmony-features-v1":
            # Compute at the app's analysis rate (22.05 kHz) so training features
            # match what the app feeds the model at inference — the chroma bins
            # depend on sampleRate/FRAME_SIZE, so the rate must be the app's.
            samples, sr = load_audio_mono(track.audio_path, target_sr=APP_SAMPLE_RATE)
            return extract_app_features(samples, sr, **audio_kwargs)
        samples, sr = load_audio_mono(track.audio_path)
        return extract_features(samples, sr, **audio_kwargs)
    if track.is_feature_trainable():
        return load_billboard_bothchroma(track.feature_path)
    raise ValueError(
        f"{track.track_id}: not trainable (availability={track.audio_availability!r}, "
        f"audio_path={bool(track.audio_path)}, feature_path={bool(track.feature_path)})"
    )
