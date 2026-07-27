# Full-band Slakh2100 dataset audit

## Official source and licence (verified, not invented)

Queried the official Zenodo record API (concept `4599665` → version `4599666`):

| field | value |
|---|---|
| title | Slakh2100 |
| licence | **CC-BY-4.0** (permissive; commercial/product use compatible with attribution) |
| access | open |
| archive | `slakh2100_flac_redux.tar.gz` (single file) |
| size | 104,322,767,708 bytes (104.32 GB) |
| md5 | `f4b71b6c45ac9b506f59788456b3f0c4` |

CC-BY-4.0 permits training weights intended for a possible commercial Tabsmith release
(with attribution), unlike MedleyDB (non-commercial) which is therefore excluded from
weight training. Slakh is synthetic (Lakh MIDI rendered with virtual instruments),
primarily instrumental — recorded-audio realism is a documented limitation.

## Hardware and chosen scale

| resource | value |
|---|---|
| CPU logical cores | 12 |
| CUDA | not available (torch 2.13.0+cpu) |
| free disk | ~411 GB |
| Zenodo throughput (measured, 100 MB ranged fetch) | ~4.25 MB/s |
| HTTP range / resume | supported (HTTP 206, `accept-ranges: bytes`) |

Decision: **CPU-only ⇒ Plan B (workday pilot)** rather than a GPU complete experiment.
The single monolithic 104.32 GB archive at ~4.25 MB/s implies **~6.8 hours of download
alone** before any CPU feature extraction or training, and the official *test* split sits
at the end of the tarball (reachable only via a full download). Whole-archive MD5 can be
verified only on a completed download.

## Integration built (scalable to the full dataset)

- `ml/full_band/slakh.py` — pinned official identity/size/MD5; resumable byte-range
  downloader that verifies size + MD5 before an atomic rename and aborts on mismatch
  (never appends blindly — avoids the earlier GuitarSet append-on-resume failure mode);
  streaming `r|gz` tar iterator that reads only small members (MIDI/metadata), skips
  large audio, guards every member path against traversal/absolute paths, and selects a
  bounded, deterministic subset by split.
- `ml/full_band/midi.py` — dependency-free Standard-MIDI reader (varint delta-times,
  running status, tempo maps, note-on/off) feeding the existing deterministic
  `derive_chord_regions` (maj/min/7/N). No `mido`/`pretty_midi` dependency.

## Offline validation (no network, no 104 GB)

`ml/tests/test_full_band_slakh.py` (9 tests, passing): C-major and A-minor derivation
from synthesized MIDI; empty MIDI → single `N`; streaming subset reads MIDI/metadata and
skips audio; bounded/deterministic selection; path-traversal rejection; archive
size+MD5 verify pass/fail; pinned CC-BY-4.0 licence record.

## Status

Licence verified and commercial-compatible; provenance and integrity gates implemented
and tested; the pipeline scales to the full archive. The heavy download + CPU pilot
training + untouched-test-split evaluation were **not** completed in this session — see
`full-band-training-protocol.md` for the blocker and recommendation. No audio, MIDI, or
features are committed; the data root stays outside Git.
