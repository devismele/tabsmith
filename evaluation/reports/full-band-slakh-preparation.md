# Slakh2100 symbolic preparation

Official Zenodo record 4599666 (`slakh2100_flac_redux.tar.gz`), licence cc-by-4.0. Archive verified by exact byte size and full MD5
before this pass ran. Labels are derived deterministically from the aligned MIDI;
no audio was read.

## Coverage

- scanned tracks: 2100
- usable tracks: 1709
- failed tracks: 1
- tracks by official split: {"test": 151, "train": 1289, "validation": 269}
- total duration: 115.584 h

## Composition grouping and split integrity

- distinct compositions (by aligned-MIDI SHA-256): 1709
- compositions with more than one rendering: 0 (0 tracks)
- compositions straddling official splits: 0
- **leakage free: True**

## Chord labels

- chord regions: 697129
- mean regions/minute: 100.522
- no-chord duration: 46330.19 s (0.1113 of total)

| quality | seconds | fraction |
|---|---|---|
| 7 | 47364.13 | 0.1138 |
| N | 46330.19 | 0.1113 |
| maj | 235908.21 | 0.5669 |
| min | 86501.15 | 0.2079 |

## Instrument coverage

- guitar present: 1707 tracks (0.9988), 115.465 h
- guitar absent: 2 tracks, 0.12 h
- bass present: 1595 tracks
- harmonic instrument present: 1709 tracks
- mean rendered stems per track: 10.48

| instrument class | tracks |
|---|---|
| Bass | 1595 |
| Brass | 432 |
| Chromatic Percussion | 308 |
| Drums | 1709 |
| Guitar | 1707 |
| Organ | 473 |
| Piano | 1709 |
| Pipe | 427 |
| Reed | 522 |
| Strings | 250 |
| Strings (continued) | 1388 |
| Synth Lead | 376 |
| Synth Pad | 572 |

## Deterministic pilot subset

- requested: 120, selected: 120
- duration: 7.967 h
- guitar-present tracks: 120
- selection rule: one rendering per composition, ordered by aligned-MIDI SHA-256
- subset checksum: `f600afba8c9bd1e683e364125cb2cf5d4a56ddd9bc55182f50ae3e9b44f6ee51`

## Failures

| track | error |
|---|---|
| Track01657 | midi parse failed: truncated variable-length quantity |

## Label granularity (read before comparing fragmentation)

- mean derived region duration: **0.597 s**
- derived regions/minute: **100.5** vs GuitarSet reference ~20 (5.0x denser)

**These labels are note-level, not chord-level.** The symbolic derivation
segments on note content, and dense full-band arrangements change notes far
more often than they change chords. Consequences:

- Root, quality, detailed accuracy and no-chord metrics remain meaningful.
- Fragmentation rate and regions/minute are **not** comparable to GuitarSet,
  and a fragmentation gate calibrated on GuitarSet cannot be applied to these
  labels as-is.
- A chord-level derivation (harmonic-rhythm aware segmentation, or a minimum
  region on the order of a beat rather than 0.10 s) is prerequisite work
  before full-band stability gates mean anything.

This is reported rather than silently corrected: retuning the derivation after
seeing the numbers would be fitting labels to a desired result.

## Interpretation limit

Slakh2100 is rendered from MIDI. Everything derived here is synthetic-domain
evidence. A future untouched, real-recorded, legally licensed full-band
benchmark remains required before any production claim.
