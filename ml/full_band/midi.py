"""Minimal dependency-free Standard MIDI File reader.

Slakh2100 ships aligned MIDI (``all_src.mid`` per track and per-stem MIDI). We
only need timed note events to feed the existing deterministic chord derivation
(`ml.full_band.symbolic.derive_chord_regions`), so this parses format-0/1 SMF into
note events without pulling in ``mido``/``pretty_midi``. It handles delta-time
varints, running status, tempo maps (multiple set-tempo), note-on/off (including
velocity-0 note-off), meta and sysex skipping.
"""
from __future__ import annotations

from dataclasses import dataclass


# General MIDI reserves channel 10 (0-indexed 9) for percussion. Note numbers on
# that channel select drum sounds and carry no pitch, so they must be excluded
# from any harmonic analysis.
DRUM_CHANNEL = 9


class MidiError(ValueError):
    pass


def _read_varlen(data: bytes, i: int) -> tuple[int, int]:
    value = 0
    while True:
        if i >= len(data):
            raise MidiError("truncated variable-length quantity")
        byte = data[i]
        i += 1
        value = (value << 7) | (byte & 0x7F)
        if not (byte & 0x80):
            return value, i


@dataclass
class NoteEvent:
    start: float
    end: float
    pitch: int
    velocity: float


def _parse_track(data: bytes, division: int, default_us_per_qn: int,
                 include_drum_channel: bool = False):
    """Yield (tick, kind, pitch, velocity) note edges plus tempo changes."""
    i = 0
    tick = 0
    status = 0
    edges = []          # (tick, pitch, velocity_or_0)
    tempos = []         # (tick, us_per_qn)
    while i < len(data):
        delta, i = _read_varlen(data, i)
        tick += delta
        if i >= len(data):
            break
        byte = data[i]
        if byte & 0x80:
            status = byte
            i += 1
        # else running status: reuse previous status, do not advance
        if status == 0xFF:  # meta
            meta_type = data[i]; i += 1
            length, i = _read_varlen(data, i)
            payload = data[i:i + length]; i += length
            if meta_type == 0x51 and length == 3:
                tempos.append((tick, (payload[0] << 16) | (payload[1] << 8) | payload[2]))
            # 0x2F end-of-track and others: ignored
        elif status in (0xF0, 0xF7):  # sysex
            length, i = _read_varlen(data, i)
            i += length
        else:
            event = status & 0xF0
            if event in (0x80, 0x90):
                pitch = data[i]; vel = data[i + 1]; i += 2
                on = event == 0x90 and vel > 0
                # GM channel 10 (0-indexed 9) is percussion: its "pitches" are
                # drum sounds, not harmony. Slakh's all_src.mid merges every
                # stem including Drums, so keeping these would feed kick/snare
                # note numbers straight into chord derivation.
                if (status & 0x0F) != DRUM_CHANNEL or include_drum_channel:
                    edges.append((tick, pitch, vel if on else 0))
            elif event in (0xA0, 0xB0, 0xE0):  # 2 data bytes
                i += 2
            elif event in (0xC0, 0xD0):        # 1 data byte
                i += 1
            else:
                raise MidiError(f"unsupported status byte {status:#x}")
    return edges, tempos


def read_note_events(data: bytes, *,
                     include_drum_channel: bool = False) -> tuple[list[dict], float]:
    """Parse SMF bytes into note events and total duration in seconds.

    Percussion (GM channel 10) is excluded by default: those note numbers are
    drum sounds, not pitches, and Slakh's ``all_src.mid`` merges every stem
    including Drums. Including them feeds kick/snare note numbers into chord
    derivation as if they were harmony.
    """
    if data[:4] != b"MThd":
        raise MidiError("not a Standard MIDI File (missing MThd)")
    header_len = int.from_bytes(data[4:8], "big")
    fmt = int.from_bytes(data[8:10], "big")
    ntrks = int.from_bytes(data[10:12], "big")
    division = int.from_bytes(data[12:14], "big")
    if division & 0x8000:
        raise MidiError("SMPTE time division is not supported")
    if division <= 0:
        raise MidiError("invalid tick division")
    pos = 8 + header_len

    # collect all tracks' edges/tempos on a shared tick timeline
    all_edges: list[tuple[int, int, int]] = []
    all_tempos: list[tuple[int, int]] = []
    for _ in range(ntrks):
        if data[pos:pos + 4] != b"MTrk":
            break
        length = int.from_bytes(data[pos + 4:pos + 8], "big")
        chunk = data[pos + 8:pos + 8 + length]
        pos += 8 + length
        edges, tempos = _parse_track(chunk, division, 500000,
                                     include_drum_channel=include_drum_channel)
        all_edges.extend(edges)
        all_tempos.extend(tempos)

    # tick -> seconds via a piecewise-constant tempo map
    tempo_map = sorted(all_tempos) or [(0, 500000)]
    if tempo_map[0][0] != 0:
        tempo_map = [(0, 500000)] + tempo_map

    def tick_to_seconds(target: int) -> float:
        seconds = 0.0
        for idx, (t_tick, us) in enumerate(tempo_map):
            nxt = tempo_map[idx + 1][0] if idx + 1 < len(tempo_map) else None
            if nxt is not None and nxt <= target:
                seconds += (nxt - t_tick) * (us / 1e6) / division
            else:
                seconds += (target - t_tick) * (us / 1e6) / division
                break
        return seconds

    # pair note-on with the next note-off of the same pitch
    active: dict[int, list[int]] = {}
    events: list[dict] = []
    for tick, pitch, vel in sorted(all_edges, key=lambda e: e[0]):
        if vel > 0:
            active.setdefault(pitch, []).append(tick)
        elif active.get(pitch):
            start_tick = active[pitch].pop(0)
            s = tick_to_seconds(start_tick)
            e = tick_to_seconds(tick)
            if e > s:
                events.append({"start": s, "end": e, "pitches": [pitch], "velocity": max(1.0, float(vel or 64))})
    duration = max((ev["end"] for ev in events), default=0.0)
    return events, duration
