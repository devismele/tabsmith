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

import csv
from pathlib import Path

from ..schema import ChordRegion, Track


def load_billboard_artist_index(index_path: str | Path) -> dict[str, str]:
    """Map zero-padded song id -> artist from billboard-2.0-index.csv.

    Without this every Billboard track shares one artist and cannot be split by
    artist (all would land in one split). Entries with no artist are skipped.
    """
    index_path = Path(index_path)
    if not index_path.exists():
        return {}
    mapping: dict[str, str] = {}
    with open(index_path, "r", encoding="utf-8", errors="ignore") as handle:
        for row in csv.DictReader(handle):
            artist = (row.get("artist") or "").strip()
            raw_id = (row.get("id") or "").strip()
            if artist and raw_id.isdigit():
                mapping[f"{int(raw_id):04d}"] = artist
    return mapping


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
    artist: str = "Billboard",
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
    bothchroma = entry_dir / "bothchroma.csv"
    feature_path = str(bothchroma) if bothchroma.exists() else ""
    has_features = bool(feature_path)

    return Track(
        track_id=f"billboard-{entry_dir.name}",
        artist=artist,
        title=entry_dir.name,
        duration=round(chords[-1].end, 4),
        source="billboard",
        audio_availability="features" if has_features else "annotations",
        split=split,
        license="ddmal-cc0-features",
        source_url="https://ddmal.music.mcgill.ca/research/The_McGill_Billboard_Project",
        feature_path=feature_path,
        chords=chords,
        notes=f"Imported from McGill Billboard ({dataset_version}); "
              f"{'NNLS-chroma features (billboard-bothchroma-v1)' if has_features else 'annotations only'}.",
    ).validate()


def scan_billboard_dir(root: str | Path, split: str = "development") -> list[Track]:
    root = Path(root)
    if not root.exists():
        return []
    # Auto-detect the artist index next to, above, or inside the data dir.
    index = {}
    for candidate in (root / "index.csv", root.parent / "index.csv",
                      root / "billboard-2.0-index.csv", root.parent / "billboard-2.0-index.csv"):
        if candidate.exists():
            index = load_billboard_artist_index(candidate)
            break
    tracks: list[Track] = []
    for entry in sorted(p for p in root.iterdir() if p.is_dir()):
        for lab_name in ("majmin.lab", "full.lab", "majmininv.lab"):
            if (entry / lab_name).exists():
                try:
                    artist = index.get(entry.name, "Billboard")
                    tracks.append(import_billboard_entry(entry, split=split, lab_name=lab_name, artist=artist))
                except ValueError:
                    pass
                break
    return tracks
