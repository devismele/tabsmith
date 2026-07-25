"""Python port of the app's spectralChroma (src/chordAnalysis.ts) — harmony-features-v1.

This produces features numerically identical to what the Electron app computes at
inference (`chromaForChordFrame` / `spectralChroma` + RMS), so a model trained on
these features consumes exactly the representation the app will feed it. That
train/serve match is the prerequisite for wiring the learned engine into the app
(the FREEZE doc's two feature layers finally reconciled onto one).

Faithful to the TS:
- Hann window over FRAME_SIZE = 8192 (np.hanning matches 0.5 - 0.5*cos(2πi/(n-1))),
- magnitude spectrum |rfft| over bins 0..n/2-1,
- chroma[pc] += sqrt(magnitude) * weight, binned by round(69 + 12*log2(f/440)) % 12,
- treble chroma over 70-1400 Hz; root/bass chroma over 42-320 Hz with a
  low-frequency bias sqrt(lowHz / max(lowHz, f)); each L1-normalized,
- energy = RMS of the zero-padded frame (divided by FRAME_SIZE).

The 25-dim model layout maps as: chroma | rootChroma (as bass_chroma) | rms.
"""
from __future__ import annotations

import numpy as np

from .features import FeatureFrames

FEATURE_PIPELINE_VERSION = "harmony-features-v1"
FRAME_SIZE = 8192
DEFAULT_HOP_SECONDS = 0.25  # app default: settings.rawFrameHopSeconds

_WINDOW = np.hanning(FRAME_SIZE)


def _chroma_from_magnitudes(mag: np.ndarray, frame_length: int, sample_rate: float,
                            low_hz: float, high_hz: float, low_bias: bool = False) -> np.ndarray:
    chroma = np.zeros(12)
    bin_hz = sample_rate / frame_length
    for b in range(1, len(mag)):
        freq = b * bin_hz
        if freq < low_hz or freq > high_hz:
            continue
        midi = 69.0 + 12.0 * np.log2(freq / 440.0)
        pc = int(round(midi)) % 12
        weight = np.sqrt(low_hz / max(low_hz, freq)) if low_bias else 1.0
        chroma[pc] += np.sqrt(mag[b]) * weight
    total = chroma.sum()
    return chroma / total if total > 0 else chroma


def app_spectral_chroma(frame: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """(treble chroma, root chroma) for one frame, matching the app's spectralChroma."""
    padded = np.zeros(FRAME_SIZE)
    n = min(len(frame), FRAME_SIZE)
    padded[:n] = frame[:n]
    spectrum = np.fft.rfft(padded * _WINDOW)
    mag = np.abs(spectrum)[:FRAME_SIZE // 2]  # bins 0..n/2-1 to match the TS magnitude length
    chroma = _chroma_from_magnitudes(mag, FRAME_SIZE, sample_rate, 70.0, 1400.0)
    root_chroma = _chroma_from_magnitudes(mag, FRAME_SIZE, sample_rate, 42.0, 320.0, low_bias=True)
    return chroma, root_chroma


def extract_app_features(samples: np.ndarray, sample_rate: int,
                         hop_seconds: float = DEFAULT_HOP_SECONDS) -> FeatureFrames:
    """Frame-wise harmony-features-v1 over an audio signal, matching createRawFrames."""
    hop = max(1, round(hop_seconds * sample_rate))
    times, chroma_rows, bass_rows, energy = [], [], [], []
    for start in range(0, max(1, len(samples)), hop):
        frame = samples[start:start + FRAME_SIZE]
        chroma, root_chroma = app_spectral_chroma(frame, sample_rate)
        padded = np.zeros(FRAME_SIZE)
        padded[:min(len(frame), FRAME_SIZE)] = frame[:FRAME_SIZE]
        rms = float(np.sqrt(np.mean(padded ** 2)))
        times.append(start / sample_rate)          # app frame.start time
        chroma_rows.append(chroma)
        bass_rows.append(root_chroma)
        energy.append(rms)

    return FeatureFrames(
        times=np.asarray(times),
        chroma=np.asarray(chroma_rows) if chroma_rows else np.zeros((0, 12)),
        bass_chroma=np.asarray(bass_rows) if bass_rows else np.zeros((0, 12)),
        energy=np.asarray(energy) if energy else np.zeros((0,)),
        sample_rate=sample_rate,
        hop_seconds=hop / sample_rate,
        pipeline_version=FEATURE_PIPELINE_VERSION,
    )
