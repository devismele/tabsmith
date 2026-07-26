"""Importer for GuitarSet (NYU MARL) — real guitar audio with chord annotations.

Unlike Isophonics/Billboard, GuitarSet distributes the *audio* itself under
**CC-BY 4.0** (commercial use permitted with attribution), so an imported track
is genuinely ``audio``-trainable through the ``numpy-chroma-v1`` pipeline — no
feature reconciliation needed. It spans five musical styles (Rock, Singer-
Songwriter, Bossa Nova, Jazz, Funk) played by six guitarists, which is why it is
the first real multi-genre source. Production style is uniform (solo acoustic
guitar), so it broadens *musical* genre but not full-band production diversity;
McGill Billboard features cover that later.

Layout expected (as distributed in GuitarSet 1.1.0 on Zenodo record 3371780)::

    <root>/annotation/<track_id>.jams          # standard JAMS JSON
    <root>/audio_mono-mic/<track_id>_mic.wav   # mono reference mic recording

A ``track_id`` looks like ``05_Jazz3-150-C_solo`` = ``<player>_<style><n>-<tempo>-<key>_<mode>``.
The player id becomes the ``artist`` so splits stay performer-separated. Nothing
here downloads audio; point ``--guitarset-dir`` at a locally acquired copy.

Attribution obligation if used: credit GuitarSet (Xi et al., ISMIR 2018), CC-BY 4.0.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..schema import ChordRegion, Track

# Style codes embedded in the track id -> human-readable genre for notes.
_STYLE_GENRES = {
    "BN": "Bossa Nova",
    "Funk": "Funk",
    "Jazz": "Jazz",
    "Rock": "Rock",
    "SS": "Singer-Songwriter",
}


def _parse_track_id(track_id: str) -> tuple[str, str, str]:
    """Return (player_id, genre, mode) parsed from a GuitarSet track id.

    ``05_Jazz3-150-C_solo`` -> ("05", "Jazz", "solo"). Falls back gracefully on
    ids that don't match, so a slightly different distribution never crashes import.
    """
    player = track_id.split("_", 1)[0] if "_" in track_id else "unknown"
    mode = track_id.rsplit("_", 1)[-1] if "_" in track_id else "unknown"
    genre = "unknown"
    parts = track_id.split("_")
    if len(parts) >= 2:
        style_token = parts[1].split("-", 1)[0]  # "Jazz3" -> "Jazz3"
        style_key = style_token.rstrip("0123456789")  # -> "Jazz"
        genre = _STYLE_GENRES.get(style_key, style_key or "unknown")
    return player, genre, mode


def _chord_regions_from_jams(jams: dict, prefer: str = "leadsheet") -> list[ChordRegion]:
    """Extract chord regions from a JAMS dict.

    GuitarSet stores two ``chord`` annotations: the lead-sheet (intended) chords
    first and inferred chords second. The lead-sheet version is the reliable
    ground truth, so it is preferred; ``prefer="inferred"`` selects the second.
    """
    chord_annotations = [a for a in jams.get("annotations", []) if a.get("namespace") == "chord"]
    if not chord_annotations:
        return []
    index = 1 if prefer == "inferred" and len(chord_annotations) > 1 else 0
    data = chord_annotations[index].get("data", [])
    regions: list[ChordRegion] = []
    for obs in data:
        start = float(obs["time"])
        end = start + float(obs["duration"])
        if end <= start:
            continue
        regions.append(ChordRegion(round(start, 4), round(end, 4), str(obs["value"])))
    regions.sort(key=lambda r: r.start)
    return regions


def _beats_from_jams(jams: dict) -> tuple[list[float], list[float]]:
    beats: list[float] = []
    downbeats: list[float] = []
    for annotation in jams.get("annotations", []):
        if annotation.get("namespace") != "beat_position":
            continue
        for obs in annotation.get("data", []):
            t = round(float(obs["time"]), 4)
            beats.append(t)
            value = obs.get("value")
            position = value.get("position") if isinstance(value, dict) else value
            if position in (1, 1.0, "1"):
                downbeats.append(t)
        break
    return beats, downbeats


def import_guitarset_track(
    jams_path: str | Path,
    *,
    split: str = "development",
    audio_dir: str | Path | None = None,
    prefer: str = "leadsheet",
    dataset_version: str = "guitarset-zenodo-3371780-v1.1.0",
) -> Track:
    jams_path = Path(jams_path)
    jams = json.loads(jams_path.read_text(encoding="utf-8", errors="ignore"))
    track_id = jams_path.stem
    chords = _chord_regions_from_jams(jams, prefer=prefer)
    if not chords:
        raise ValueError(f"No chord annotation parsed from {jams_path}")

    beats, downbeats = _beats_from_jams(jams)
    player, genre, mode = _parse_track_id(track_id)

    # Prefer the mono reference mic recording next to the annotations.
    audio_path = ""
    if audio_dir is not None:
        candidate = Path(audio_dir) / f"{track_id}_mic.wav"
        if candidate.exists():
            audio_path = str(candidate)

    return Track(
        track_id=f"guitarset-{track_id}",
        artist=f"guitarset-p{player}",
        title=track_id,
        duration=round(chords[-1].end, 4),
        source="guitarset",
        audio_availability="audio" if audio_path else "annotations",
        split=split,
        license="CC-BY-4.0",
        source_url="https://zenodo.org/records/3371780",
        audio_path=audio_path,
        beats=beats,
        downbeats=downbeats,
        chords=chords,
        notes=f"GuitarSet ({dataset_version}); genre={genre}; mode={mode}; player={player}. "
              f"CC-BY 4.0 — attribution required.",
    ).validate()


def scan_guitarset_dir(root: str | Path, split: str = "development") -> list[Track]:
    """Scan ``<root>/annotation/*.jams`` and pair with ``<root>/audio_mono-mic``."""
    root = Path(root)
    if not root.exists():
        return []
    annotation_dir = root / "annotation" if (root / "annotation").exists() else root
    audio_dir = root / "audio_mono-mic"
    audio_dir = audio_dir if audio_dir.exists() else None
    tracks: list[Track] = []
    for jams_path in sorted(annotation_dir.rglob("*.jams")):
        try:
            tracks.append(import_guitarset_track(jams_path, split=split, audio_dir=audio_dir))
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
    return tracks
