"""Load McGill Billboard precomputed NNLS-chroma into the harmony feature layout.

McGill Billboard ships **no audio** — only Chordino (NNLS Chroma Vamp plugin)
features under CC0. This adapter maps a song's ``bothchroma.csv`` into the same
25-dim ``FeatureFrames`` layout the audio pipeline (``numpy-chroma-v1``) emits, so
a single model can train across GuitarSet audio + Billboard features. It carries
its own pipeline version ``billboard-bothchroma-v1`` (distinct from
``numpy-chroma-v1``) so the two sources are always distinguishable in caches,
manifests, and per-source normalization.

CSV format (as distributed / parsed by mirdata's billboard loader):
  * column 0 = frame timestamp in seconds (mirdata drops it as "metadata"),
  * columns 1..24 = the 24 NNLS chroma bins.

Two facts about the 24 bins are **assumptions that must be verified against a real
file on acquisition** — they are isolated here as single constants so a mismatch
is a one-line fix, not a rewrite. Both are checkable with
``calibrate_against_annotations`` (correlate chroma energy with annotated roots):

  * ``BASS_FIRST`` — whether bins 0..11 are bass and 12..23 treble, or vice versa.
  * ``NNLS_BIN_ORIGIN_PC`` — the pitch class of bin 0 (NNLS chroma is conventionally
    A-based, i.e. 9); we roll so output index 0 == C to match ``numpy-chroma-v1``.

Energy is not recoverable from chroma, so a per-frame magnitude *proxy* (total
pre-normalization chroma mass, scaled to unit max) fills the energy channel and is
flagged unavailable via ``energy_available=False``. No audio is downloaded here.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .features import FeatureFrames

FEATURE_PIPELINE_VERSION = "billboard-bothchroma-v1"

# --- Verify-on-real-file assumptions (see module docstring) ---------------------
BASS_FIRST = True            # bins 0..11 = bass, 12..23 = treble
NNLS_BIN_ORIGIN_PC = 9       # bin 0 == A (pitch class 9); roll to C-based order
# --------------------------------------------------------------------------------


def _to_c_based(vec12: np.ndarray, origin_pc: int) -> np.ndarray:
    """Reorder a 12-vector whose index 0 == ``origin_pc`` so index 0 == C.

    out[j] = vec[(j - origin_pc) mod 12] == np.roll(vec, origin_pc).
    """
    return np.roll(vec12, origin_pc)


def _l1(vec: np.ndarray) -> np.ndarray:
    total = float(vec.sum())
    return vec / total if total > 0 else vec


def load_billboard_bothchroma(
    csv_path: str | Path,
    *,
    bass_first: bool = BASS_FIRST,
    origin_pc: int = NNLS_BIN_ORIGIN_PC,
) -> FeatureFrames:
    """Parse a Billboard ``bothchroma.csv`` into 25-dim FeatureFrames."""
    csv_path = Path(csv_path)
    times: list[float] = []
    raw_rows: list[np.ndarray] = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as handle:
        for row in csv.reader(handle):
            # The DDMAL release prefixes each row with the source filename
            # ("/tmp/audio.wav"): col0=name, col1=time, col2..25 = 24 chroma. A
            # bare release omits it: col0=time, col1..24 = chroma. Detect which by
            # whether col0 parses as a float, so both line up.
            try:
                float(row[0])
                base = 0
            except (ValueError, IndexError):
                base = 1
            if len(row) < base + 25:
                continue
            try:
                time = float(row[base])
                chroma = [float(x) for x in row[base + 1:base + 25]]
            except ValueError:
                continue  # header or malformed line
            times.append(time)
            raw_rows.append(np.asarray(chroma, dtype=np.float64))

    if not raw_rows:
        return FeatureFrames(
            times=np.zeros((0,)), chroma=np.zeros((0, 12)), bass_chroma=np.zeros((0, 12)),
            energy=np.zeros((0,)), sample_rate=0, hop_seconds=0.0,
            pipeline_version=FEATURE_PIPELINE_VERSION,
        )

    raw = np.vstack(raw_rows)                       # (T, 24)
    first, second = raw[:, :12], raw[:, 12:]
    bass_raw, treble_raw = (first, second) if bass_first else (second, first)

    # Energy proxy from total pre-normalization mass, scaled to unit max.
    mass = raw.sum(axis=1)
    energy = mass / (mass.max() or 1.0)

    chroma = np.vstack([_l1(_to_c_based(treble_raw[i], origin_pc)) for i in range(len(raw))])
    bass_chroma = np.vstack([_l1(_to_c_based(bass_raw[i], origin_pc)) for i in range(len(raw))])

    times_arr = np.asarray(times, dtype=np.float64)
    diffs = np.diff(times_arr)
    hop = float(np.median(diffs)) if len(diffs) else 0.0

    frames = FeatureFrames(
        times=times_arr,
        chroma=chroma,
        bass_chroma=bass_chroma,
        energy=energy.astype(np.float64),
        sample_rate=0,               # unknown / not applicable for feature-only sources
        hop_seconds=hop,
        pipeline_version=FEATURE_PIPELINE_VERSION,
    )
    # Mark the synthetic-energy channel so downstream code can treat it specially.
    frames.energy_available = False  # type: ignore[attr-defined]
    return frames


def calibrate_against_annotations(frames: FeatureFrames, chord_regions) -> dict:
    """Sanity-check bin order/origin: does bass-chroma energy peak on annotated roots?

    Returns the fraction of chord frames whose argmax bass pitch class equals the
    annotated root. A well-aligned ``origin_pc``/``bass_first`` gives a high value;
    a low value means the constants above need adjusting for the acquired release.
    Runs on real data at acquisition time — no assertion here.
    """
    from ..schema import parse_chord_label

    hits = 0
    total = 0
    for i, t in enumerate(frames.times):
        region = next((r for r in chord_regions if r.start <= t < r.end), None)
        if region is None:
            continue
        symbol = parse_chord_label(region.label)
        if symbol.root is None:
            continue
        total += 1
        if int(np.argmax(frames.bass_chroma[i])) == symbol.root:
            hits += 1
    return {"frames_scored": total, "root_match_fraction": (hits / total) if total else 0.0}
