"""Construct the full-band source views the Phase 12 baselines compare.

Slakh ships per-instrument stems plus the rendered mix, which makes it possible
to separate *what the model can do* from *what a separator can deliver*. Each
view below is a deterministic mix of a chosen stem subset:

===========================  ==================================================
view                         contents
===========================  ==================================================
``full-mix``                 the dataset's own rendered mix
``oracle-harmonic``          every pitched/harmonic stem, no drums or percussion
``oracle-guitar``            guitar-family stems only
``guitar-plus-bass``         guitar-family plus bass stems
``guitar-absent-harmonic``   harmonic stems with all guitar removed
``percussion-only``          drums and percussion only (a no-harmony control)
===========================  ==================================================

The oracle views are upper bounds: they use ground-truth stems, so a real
separator can only approach them. Reporting oracle and full-mix side by side is
what makes a separation gap measurable rather than assumed.

``percussion-only`` exists to catch a model that hallucinates harmony from
rhythm alone; the correct answer there is almost entirely no-chord.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

GUITAR_CLASSES = {"Guitar"}
BASS_CLASSES = {"Bass"}
PERCUSSIVE_CLASSES = {"Drums", "Percussive"}
# Everything pitched that can carry harmony. Deliberately excludes percussion.
# Vocabulary confirmed against the verified archive; "Strings (continued)" is a
# real class name covering 1389 of 1709 tracks.
HARMONIC_CLASSES = {
    "Guitar", "Bass", "Piano", "Organ", "Strings", "Strings (continued)",
    "Brass", "Reed", "Pipe", "Synth Lead", "Synth Pad", "Chromatic Percussion",
}

VIEW_NAMES = (
    "full-mix",
    "oracle-harmonic",
    "oracle-guitar",
    "guitar-plus-bass",
    "guitar-absent-harmonic",
    "percussion-only",
)


class ViewError(RuntimeError):
    pass


@dataclass
class Stem:
    stem_id: str
    inst_class: str
    path: Path

    @property
    def is_guitar(self) -> bool:
        return self.inst_class in GUITAR_CLASSES

    @property
    def is_bass(self) -> bool:
        return self.inst_class in BASS_CLASSES

    @property
    def is_percussive(self) -> bool:
        return self.inst_class in PERCUSSIVE_CLASSES

    @property
    def is_harmonic(self) -> bool:
        return self.inst_class in HARMONIC_CLASSES


def stems_from_metadata(metadata: dict[str, Any], stem_dir: Path,
                        *, suffix: str = ".flac") -> list[Stem]:
    """Map a track's metadata to the stem files that actually exist on disk."""
    entries = metadata.get("stems")
    if not isinstance(entries, dict):
        return []
    stems: list[Stem] = []
    for stem_id, info in sorted(entries.items()):
        if not isinstance(info, dict) or info.get("audio_rendered") is False:
            continue
        inst_class = info.get("inst_class")
        if not isinstance(inst_class, str):
            continue
        path = Path(stem_dir) / f"{stem_id}{suffix}"
        if path.exists():
            stems.append(Stem(stem_id, inst_class.strip(), path))
    return stems


def select_stems(stems: Iterable[Stem], view: str) -> list[Stem]:
    """Which stems belong to ``view``. Raises on an unknown view name."""
    stems = list(stems)
    if view == "oracle-harmonic":
        return [s for s in stems if s.is_harmonic]
    if view == "oracle-guitar":
        return [s for s in stems if s.is_guitar]
    if view == "guitar-plus-bass":
        return [s for s in stems if s.is_guitar or s.is_bass]
    if view == "guitar-absent-harmonic":
        return [s for s in stems if s.is_harmonic and not s.is_guitar]
    if view == "percussion-only":
        return [s for s in stems if s.is_percussive]
    if view == "full-mix":
        raise ViewError("full-mix is read from the dataset mix, not summed from stems")
    raise ViewError(f"unknown view: {view}")


def _read_audio(path: Path, target_sample_rate: int | None = None):
    import soundfile as sf

    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if target_sample_rate and sample_rate != target_sample_rate:
        import librosa

        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
        sample_rate = target_sample_rate
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def mix_stems(stems: Iterable[Stem], *, sample_rate: int | None = None,
              peak_normalize: bool = True) -> tuple[np.ndarray, int]:
    """Sum stems to one mono signal.

    Summing is what the dataset itself does, so an oracle view stays comparable
    to the rendered mix. Peak normalisation only rescales when the sum would
    clip, so quiet views are not artificially boosted relative to the full mix.
    """
    stems = list(stems)
    if not stems:
        raise ViewError("no stems selected for this view")
    mixed: np.ndarray | None = None
    rate = sample_rate
    for stem in stems:
        audio, stem_rate = _read_audio(stem.path, rate)
        rate = rate or stem_rate
        if stem_rate != rate:
            raise ViewError(f"sample-rate mismatch in {stem.stem_id}: {stem_rate} != {rate}")
        if mixed is None:
            mixed = audio.copy()
        else:
            length = max(len(mixed), len(audio))
            if len(mixed) < length:
                mixed = np.pad(mixed, (0, length - len(mixed)))
            if len(audio) < length:
                audio = np.pad(audio, (0, length - len(audio)))
            mixed += audio
    assert mixed is not None
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak_normalize and peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32), int(rate or 0)


def build_view(track_dir: Path, metadata: dict[str, Any], view: str,
               *, sample_rate: int | None = None,
               mix_name: str = "mix.flac", stem_subdir: str = "stems"):
    """Return ``(audio, sample_rate)`` for one view of one track.

    Returns ``None`` when the view has no stems for this track (for example
    ``oracle-guitar`` on a guitar-absent piece), which is a legitimate outcome
    the caller must account for rather than an error.
    """
    track_dir = Path(track_dir)
    if view == "full-mix":
        path = track_dir / mix_name
        if not path.exists():
            raise ViewError(f"missing rendered mix: {mix_name}")
        return _read_audio(path, sample_rate)
    selected = select_stems(stems_from_metadata(metadata, track_dir / stem_subdir), view)
    if not selected:
        return None
    return mix_stems(selected, sample_rate=sample_rate)


def view_availability(metadata: dict[str, Any], track_dir: Path,
                      *, stem_subdir: str = "stems") -> dict[str, int]:
    """Stem count per view, so coverage can be reported before any audio is read."""
    stems = stems_from_metadata(metadata, Path(track_dir) / stem_subdir)
    out: dict[str, int] = {}
    for view in VIEW_NAMES:
        if view == "full-mix":
            out[view] = 1 if (Path(track_dir) / "mix.flac").exists() else 0
            continue
        out[view] = len(select_stems(stems, view))
    return out
