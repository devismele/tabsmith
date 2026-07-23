"""Framewise harmonic features for learned-harmony-v1.

This is the canonical feature contract every model consumes. The foundation
pass implements a dependency-light numpy version (STFT chroma, bass chroma,
spectral energy) so the pipeline runs today. A later pass may swap in a
librosa/CQT implementation behind the same ``FeatureFrames`` interface and the
same feature-pipeline version — the cache key includes that version so stale
features are never reused.

Timing, bass, melody, and musical-context features described in the task spec
(beat phase, downbeat probability, cyclic bar position, transition history) are
layered on top of these audio features by the training data assembler, not here;
this module owns only the audio-derived part.
"""
from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FEATURE_PIPELINE_VERSION = "numpy-chroma-v1"


@dataclass
class FeatureFrames:
    times: np.ndarray          # (T,) frame center times in seconds
    chroma: np.ndarray         # (T, 12) L1-normalized harmonic chroma
    bass_chroma: np.ndarray    # (T, 12) low-band chroma for root evidence
    energy: np.ndarray         # (T,) per-frame RMS energy
    sample_rate: int
    hop_seconds: float
    pipeline_version: str = FEATURE_PIPELINE_VERSION

    def stacked(self) -> np.ndarray:
        """(T, 25) feature matrix: chroma | bass_chroma | energy."""
        return np.concatenate([self.chroma, self.bass_chroma, self.energy[:, None]], axis=1)


def read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV as mono float32 in [-1, 1] (stdlib only)."""
    with wave.open(str(path), "rb") as handle:
        sr = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError(f"{path}: only 16-bit PCM WAV is supported in the foundation pass")
    data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, sr


def _pitch_class_map(freqs: np.ndarray, tuning_hz: float) -> np.ndarray:
    pc = np.full(freqs.shape, -1, dtype=np.int64)
    positive = freqs > 0
    midi = 69.0 + 12.0 * np.log2(np.where(positive, freqs, 1.0) / tuning_hz)
    pc_float = np.round(midi).astype(np.int64) % 12
    pc[positive] = pc_float[positive]
    return pc


def extract_features(
    samples: np.ndarray,
    sample_rate: int,
    hop_seconds: float = 0.09288,
    frame_seconds: float = 0.371,
    bass_max_hz: float = 320.0,
    tuning_hz: float = 440.0,
) -> FeatureFrames:
    n_fft = 1 << int(np.ceil(np.log2(max(256, frame_seconds * sample_rate))))
    hop = max(1, int(hop_seconds * sample_rate))
    window = np.hanning(n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    pc = _pitch_class_map(freqs, tuning_hz)
    audible = freqs >= 55.0  # ignore DC / sub-audible rumble
    bass_band = audible & (freqs <= bass_max_hz)

    if len(samples) < n_fft:
        samples = np.pad(samples, (0, n_fft - len(samples)))
    starts = range(0, len(samples) - n_fft + 1, hop)

    times, chroma_rows, bass_rows, energy_rows = [], [], [], []
    for start in starts:
        frame = samples[start:start + n_fft] * window
        spectrum = np.abs(np.fft.rfft(frame))
        chroma = np.zeros(12)
        bass = np.zeros(12)
        for cls in range(12):
            chroma[cls] = spectrum[audible & (pc == cls)].sum()
            bass[cls] = spectrum[bass_band & (pc == cls)].sum()
        chroma_rows.append(chroma / (chroma.sum() or 1.0))
        bass_rows.append(bass / (bass.sum() or 1.0))
        energy_rows.append(float(np.sqrt(np.mean(frame ** 2))))
        times.append((start + n_fft / 2) / sample_rate)

    return FeatureFrames(
        times=np.asarray(times),
        chroma=np.asarray(chroma_rows) if chroma_rows else np.zeros((0, 12)),
        bass_chroma=np.asarray(bass_rows) if bass_rows else np.zeros((0, 12)),
        energy=np.asarray(energy_rows) if energy_rows else np.zeros((0,)),
        sample_rate=sample_rate,
        hop_seconds=hop / sample_rate,
    )


def extract_from_wav(path: str | Path, **kwargs) -> FeatureFrames:
    samples, sr = read_wav_mono(path)
    return extract_features(samples, sr, **kwargs)
