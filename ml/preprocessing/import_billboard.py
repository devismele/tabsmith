"""Importer for the McGill Billboard chord dataset.

McGill Billboard supplies chord/structure annotations and *precomputed* chroma
features, but not the copyrighted recordings. Accordingly:

* a MIREX-style ``majmin.lab`` / ``full.lab`` import is ``annotations``-only;
* if the matching ``bothchroma.csv`` feature file is present, the track is
  marked ``features`` (usable for feature-based training, not raw audio).

Nothing here downloads audio. License: research use per the DDMAL terms; record
the dataset release in the manifest. See ml/README.md.
"""
from __future__ import annotations

from pathlib import Path

from ..schema import ChordRegion, Track


def _read_lab(path: Path) -> list[tuple[float, float, str]]:
    rows: list[tuple[float, float, str]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.replace("\t", " ").split()
        if len(parts) < 3:
            continue
        try:
            start, end = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        rows.append((start, end, parts[2]))
    return rows


def import_billboard_entry(
    entry_dir: str | Path,
    *,
    split: str = "development",
    lab_name: str = "majmin.lab",
    dataset_version: str = "mcgill-billboard-2012",
) -> Track:
    entry_dir = Path(entry_dir)
    lab_path = entry_dir / lab_name
    if not lab_path.exists():
        raise ValueError(f"No {lab_name} in {entry_dir}")
    rows = _read_lab(lab_path)
    if not rows:
        raise ValueError(f"No chord rows parsed from {lab_path}")
    chords = [ChordRegion(round(s, 4), round(e, 4), label) for s, e, label in rows]
    has_features = (entry_dir / "bothchroma.csv").exists() or (entry_dir / "echonest.json").exists()

    return Track(
        track_id=f"billboard-{entry_dir.name}",
        artist="Billboard",
        title=entry_dir.name,
        duration=round(chords[-1].end, 4),
        source="billboard",
        audio_availability="features" if has_features else "annotations",
        split=split,
        license="ddmal-research",
        source_url="https://ddmal.music.mcgill.ca/research/The_McGill_Billboard_Project",
        chords=chords,
        notes=f"Imported from McGill Billboard ({dataset_version}); "
              f"{'chroma features present' if has_features else 'annotations only'}.",
    ).validate()


def scan_billboard_dir(root: str | Path, split: str = "development") -> list[Track]:
    root = Path(root)
    if not root.exists():
        return []
    tracks: list[Track] = []
    for entry in sorted(p for p in root.iterdir() if p.is_dir()):
        for lab_name in ("majmin.lab", "full.lab", "majmininv.lab"):
            if (entry / lab_name).exists():
                try:
                    tracks.append(import_billboard_entry(entry, split=split, lab_name=lab_name))
                except ValueError:
                    pass
                break
    return tracks
