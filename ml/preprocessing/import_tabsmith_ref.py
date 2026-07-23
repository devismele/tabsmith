"""Importer for Tabsmith's own manually aligned chord references.

Reads ``evaluation/chord-reference-songs.json`` — the artist-level split file the
existing evaluation harness already uses — and converts every *manually
verified* section into an internal-schema Track with sounding-pitch chord
regions (capo applied). Songs still marked ``pending`` carry no chord data and
are reported but skipped.

This importer runs against real repository data today; it does not need audio.
The referenced recordings are user-supplied, so tracks are ``annotations``-only
unless a local ``audioPath`` is present.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..schema import ChordRegion, Track, transpose_label

DEFAULT_REFERENCE = Path(__file__).resolve().parents[2] / "evaluation" / "chord-reference-songs.json"


def _section_track(song: dict, section: dict) -> Track | None:
    sequence = section.get("normalizedChordSequence")
    starts = section.get("manualStartSeconds")
    if not sequence or not starts or len(sequence) != len(starts):
        return None
    capo = int(section.get("normalization", {}).get("capo", 0) or 0)
    section_end = float(section.get("endSeconds", starts[-1] + 4))
    chords: list[ChordRegion] = []
    for i, displayed in enumerate(sequence):
        start = float(starts[i])
        end = float(starts[i + 1]) if i + 1 < len(starts) else section_end
        if end <= start:
            end = start + 0.1
        # Convert displayed (capo-relative) shape to sounding pitch.
        chords.append(ChordRegion(round(start, 4), round(end, 4), transpose_label(displayed, capo)))

    audio_path = song.get("audioPath", "") or ""
    return Track(
        track_id=section.get("id", f"{song['artist']}-{song['title']}").lower().replace(" ", "-"),
        artist=song.get("artist", ""),
        title=song.get("title", ""),
        duration=round(chords[-1].end - chords[0].start, 4),
        source="tabsmith-reference",
        audio_availability="audio" if audio_path else "annotations",
        split=song.get("split", "development"),
        key=section.get("normalization", {}).get("soundingKey"),
        license="user-supplied-recording",
        source_url=song.get("referenceUrl", ""),
        audio_path=audio_path,
        chords=chords,
        notes=f"Manually verified section ({section.get('status', 'unknown')}); "
              f"capo {capo} applied to reach sounding pitch. No lyrics/page body retained.",
    ).validate()


def import_tabsmith_references(reference_path: str | Path = DEFAULT_REFERENCE) -> tuple[list[Track], dict]:
    """Returns (verified tracks, summary of what was imported vs skipped)."""
    reference_path = Path(reference_path)
    songs = json.loads(reference_path.read_text(encoding="utf-8"))
    tracks: list[Track] = []
    pending: list[str] = []
    for song in songs:
        sections = song.get("sections") or []
        verified = [s for s in sections if s.get("status") == "manually-verified"]
        if not verified:
            pending.append(f"{song.get('artist')} — {song.get('title')} [{song.get('status', 'pending')}]")
            continue
        for section in verified:
            track = _section_track(song, section)
            if track:
                tracks.append(track)
    summary = {
        "referencePath": str(reference_path),
        "totalSongs": len(songs),
        "verifiedTracks": len(tracks),
        "pendingSongs": pending,
        "splitCounts": _split_counts(songs),
    }
    return tracks, summary


def _split_counts(songs: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for song in songs:
        split = song.get("split", "development")
        counts[split] = counts.get(split, 0) + 1
    return counts


def _main() -> None:
    tracks, summary = import_tabsmith_references()
    print(json.dumps(summary, indent=2))
    for track in tracks:
        print(f"  {track.track_id}: {len(track.chords)} chords, split={track.split}, "
              f"audio={track.audio_availability}")


if __name__ == "__main__":
    _main()
