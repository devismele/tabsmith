"""Internal JAMS-compatible annotation schema for learned-harmony-v1.

Every dataset importer (Isophonics, McGill Billboard, Tabsmith reference,
synthetic) converts to this one format so downstream feature extraction,
splitting, and evaluation never depend on a dataset's original layout.

The chord label vocabulary mirrors ``src/chordAnalysis.ts`` /
``server/evaluation.mjs`` (root + major/minor family + detailed quality) so the
learned engine can be scored by the *existing* Tabsmith evaluation harness.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

SCHEMA_VERSION = 1

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_PITCH_INDEX = {
    "C": 0, "B#": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "FB": 4,
    "E#": 5, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9,
    "A#": 10, "BB": 10, "B": 11, "CB": 11,
}

# Canonical quality tokens (Harte-style, after ``:``).
_QUALITY_ALIASES = {
    "": "maj", "maj": "maj", "major": "maj",
    "min": "min", "m": "min", "minor": "min",
    "7": "7", "dom7": "7",
    "maj7": "maj7", "min7": "min7", "m7": "min7",
    "dim": "dim", "hdim7": "dim", "dim7": "dim",
    "aug": "aug",
    "sus2": "sus2", "sus4": "sus4",
    "add9": "add9", "5": "5",
}

# Audio availability, mirroring the three cases the task calls out. Only "audio"
# entries may be used for raw-audio training; "features" for precomputed feature
# training; "annotations" entries are label-only and must never be treated as
# audio-trainable.
AUDIO_AVAILABILITY = ("audio", "features", "annotations")


def _clean(token: str) -> str:
    return token.strip().replace("♯", "#").replace("♭", "b")


def pitch_class_index(name: str) -> Optional[int]:
    if not name:
        return None
    idx = _PITCH_INDEX.get(_clean(name).upper())
    return idx


def pitch_class_name(index: int) -> str:
    return PITCH_CLASSES[index % 12]


@dataclass(frozen=True)
class ChordSymbol:
    """Parsed chord label with the fields the evaluation harness compares on."""
    label: str
    root: Optional[int]          # pitch-class index, or None for N/X
    quality: str                 # canonical quality token or "N"/"X"
    bass: Optional[int]          # pitch-class index for inversions, else None
    is_no_chord: bool
    is_unknown: bool

    @property
    def family(self) -> str:
        """maj/min/dim family used by the major-minor accuracy metric."""
        if self.is_no_chord:
            return "N"
        if self.quality in ("min", "min7"):
            return "min"
        if self.quality == "dim":
            return "dim"
        return "maj"

    @property
    def root_name(self) -> Optional[str]:
        return None if self.root is None else pitch_class_name(self.root)

    @property
    def majmin(self) -> str:
        return "N" if self.is_no_chord else f"{self.root_name}:{self.family}"

    @property
    def detailed(self) -> str:
        return "N" if self.is_no_chord else f"{self.root_name}:{self.quality}"


def parse_chord_label(label: str) -> ChordSymbol:
    """Parse Harte-ish (``A:maj``, ``F#:7``, ``C#:min/E``) or plain (``Am``) labels."""
    raw = _clean(str(label))
    if raw in ("", "N", "N.C.", "n"):
        return ChordSymbol(label or "N", None, "N", None, True, False)
    if raw.upper() in ("X", "?"):
        return ChordSymbol(label, None, "X", None, False, True)

    bass_pc = None
    body = raw
    if "/" in body:
        body, bass_part = body.split("/", 1)
        bass_pc = pitch_class_index(bass_part)

    if ":" in body:
        root_part, quality_part = body.split(":", 1)
    else:
        # Plain form like "Am", "F#maj7": root is 1-2 chars, rest is quality.
        root_len = 2 if len(body) > 1 and body[1] in "#b♯♭" else 1
        root_part, quality_part = body[:root_len], body[root_len:]

    root_pc = pitch_class_index(root_part)
    if root_pc is None:
        return ChordSymbol(label, None, "X", None, False, True)
    quality = _QUALITY_ALIASES.get(quality_part.lower())
    if quality is None:
        # Unknown extension: keep the root, degrade quality to nearest family.
        quality = "min" if quality_part.lower().startswith("m") and not quality_part.lower().startswith("maj") else "maj"
    return ChordSymbol(label, root_pc, quality, bass_pc, False, False)


def transpose_label(label: str, semitones: int) -> str:
    """Transpose a chord label; used by pitch-shift augmentation later."""
    sym = parse_chord_label(label)
    if sym.is_no_chord or sym.is_unknown or sym.root is None:
        return sym.label
    root = pitch_class_name(sym.root + semitones)
    quality = "" if sym.quality == "maj" else sym.quality
    suffix = f":{quality}" if quality else ":maj"
    bass = f"/{pitch_class_name(sym.bass + semitones)}" if sym.bass is not None else ""
    return f"{root}{suffix}{bass}"


@dataclass
class ChordRegion:
    start: float
    end: float
    label: str

    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Section:
    start: float
    end: float
    label: str


@dataclass
class Track:
    """One annotated (and optionally audio-backed) example."""
    track_id: str
    artist: str
    title: str
    duration: float
    source: str                          # synthetic|isophonics|billboard|tabsmith-reference
    audio_availability: str              # one of AUDIO_AVAILABILITY
    split: str = "development"           # training|development|validation|test
    key: Optional[str] = None
    license: str = "unknown"
    source_url: str = ""
    audio_path: str = ""
    beats: list[float] = field(default_factory=list)
    downbeats: list[float] = field(default_factory=list)
    chords: list[ChordRegion] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    notes: str = ""

    def validate(self) -> "Track":
        if self.audio_availability not in AUDIO_AVAILABILITY:
            raise ValueError(f"{self.track_id}: audio_availability must be one of {AUDIO_AVAILABILITY}")
        if self.audio_availability == "audio" and not self.audio_path:
            raise ValueError(f"{self.track_id}: audio availability is 'audio' but no audio_path was given")
        last = -1e-9
        for region in self.chords:
            if region.end <= region.start:
                raise ValueError(f"{self.track_id}: chord region {region} is non-positive length")
            if region.start < last - 1e-6:
                raise ValueError(f"{self.track_id}: chord regions overlap or are unsorted near {region.start}")
            last = region.start
        return self

    def is_audio_trainable(self) -> bool:
        return self.audio_availability == "audio" and bool(self.audio_path)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["schemaVersion"] = SCHEMA_VERSION
        return data

    @staticmethod
    def from_dict(data: dict) -> "Track":
        return Track(
            track_id=data["track_id"],
            artist=data.get("artist", ""),
            title=data.get("title", ""),
            duration=float(data["duration"]),
            source=data["source"],
            audio_availability=data.get("audio_availability", "annotations"),
            split=data.get("split", "development"),
            key=data.get("key"),
            license=data.get("license", "unknown"),
            source_url=data.get("source_url", ""),
            audio_path=data.get("audio_path", ""),
            beats=list(data.get("beats", [])),
            downbeats=list(data.get("downbeats", [])),
            chords=[ChordRegion(**c) for c in data.get("chords", [])],
            sections=[Section(**s) for s in data.get("sections", [])],
            notes=data.get("notes", ""),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @staticmethod
    def load(path: str | Path) -> "Track":
        return Track.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
