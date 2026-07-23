"""Importer for Isophonics reference annotations (Beatles, Queen, Carole King…).

Isophonics distributes timed chord/beat/key/segment annotations as ``.lab``
files (``start end label`` per line) but NOT the copyrighted recordings. So an
imported track is ``annotations``-only unless the operator supplies a locally
licensed audio file. Nothing here downloads audio.

License: Isophonics annotations are released for research use; record the exact
collection + version in the dataset manifest. See ml/README.md.
"""
from __future__ import annotations

from pathlib import Path

from ..schema import ChordRegion, Section, Track


def _read_lab(path: Path) -> list[tuple[float, float, str]]:
    rows: list[tuple[float, float, str]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            start, end = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        rows.append((start, end, " ".join(parts[2:])))
    return rows


def import_isophonics_track(
    chord_lab: str | Path,
    artist: str,
    title: str,
    *,
    split: str = "development",
    audio_path: str = "",
    beat_lab: str | Path | None = None,
    key_lab: str | Path | None = None,
    segment_lab: str | Path | None = None,
    collection_version: str = "isophonics-unversioned",
) -> Track:
    chord_lab = Path(chord_lab)
    rows = _read_lab(chord_lab)
    if not rows:
        raise ValueError(f"No chord rows parsed from {chord_lab}")
    chords = [ChordRegion(round(s, 4), round(e, 4), label) for s, e, label in rows]
    duration = chords[-1].end

    beats: list[float] = []
    downbeats: list[float] = []
    if beat_lab and Path(beat_lab).exists():
        for start, _end, label in _read_lab(Path(beat_lab)):
            beats.append(round(start, 4))
            if label.strip() in ("1", "1.0"):
                downbeats.append(round(start, 4))

    key = None
    if key_lab and Path(key_lab).exists():
        key_rows = _read_lab(Path(key_lab))
        if key_rows:
            key = key_rows[0][2]

    sections: list[Section] = []
    if segment_lab and Path(segment_lab).exists():
        sections = [Section(round(s, 4), round(e, 4), label) for s, e, label in _read_lab(Path(segment_lab))]

    return Track(
        track_id=f"isophonics-{artist}-{title}".lower().replace(" ", "-"),
        artist=artist,
        title=title,
        duration=round(duration, 4),
        source="isophonics",
        audio_availability="audio" if audio_path else "annotations",
        split=split,
        key=key,
        license="isophonics-research",
        source_url="https://isophonics.net/datasets",
        audio_path=audio_path,
        beats=beats,
        downbeats=downbeats,
        chords=chords,
        sections=sections,
        notes=f"Imported from Isophonics ({collection_version}).",
    ).validate()


def scan_isophonics_dir(root: str | Path, split: str = "development") -> list[Track]:
    """Best-effort scan: expects ``<root>/<Artist>/<...>/<Title>.lab`` chord files."""
    root = Path(root)
    if not root.exists():
        return []
    tracks: list[Track] = []
    for chord_lab in sorted(root.rglob("*.lab")):
        # Heuristic artist/title from the path; operators can override later.
        artist = chord_lab.parts[len(root.parts)] if len(chord_lab.parts) > len(root.parts) else "Unknown"
        title = chord_lab.stem
        try:
            tracks.append(import_isophonics_track(chord_lab, artist, title, split=split))
        except ValueError:
            continue
    return tracks
