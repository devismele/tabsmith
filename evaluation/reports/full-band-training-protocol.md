# Full-band training protocol and blocker

## Intended protocol (unchanged, ready to run once data is local)

Plan B CPU pilot on Slakh2100 (CC-BY-4.0):

1. Download + verify the official archive (resumable, MD5-checked).
2. Duplicate/grouping audit (Slakh documents replicated MIDI); composition-grouped
   train/dev/**untouched-test** splits with no MIDI/view/stem leakage.
3. Deterministic MIDI→chord derivation (`ml.full_band.slakh.derive_track_chords`) with
   label statistics (quality distribution, no-chord duration, guitar-present vs absent).
4. Full-band baselines: rule v3, v1 ML-only, v1 hybrid on full-mix + oracle stems vs
   separated stems (Phase 8) to localise the bottleneck before training.
5. Mixed-domain fine-tune from v1/full-v2 preserving the four-head contract and GuitarSet
   performance (domain-balanced sampling, GuitarSet rehearsal), smoke test first.
6. Frozen gates (full-band improvement + GuitarSet non-regression + parity), then evaluate
   the untouched Slakh test split once. p00 stays sealed.

## Blocker (honest stop this session)

A genuine trained-and-test-evaluated full-band candidate was **not** produced, and per the
task's own rule ("do not claim success without completed real evaluation") none is claimed.

Root causes, measured:

- **Download budget:** the dataset is a single 104.32 GB tarball; measured Zenodo
  throughput is ~4.25 MB/s ⇒ ~6.8 h to download the whole archive before any CPU work.
- **Monolithic archive:** the official *untouched test* split is at the end of the
  tarball, so valid final-test evaluation (Phase 15) requires the full download; a
  streamed front-of-archive prefix would be train-only and non-representative, which would
  violate the "do not lower evaluation standards to get a positive result" rule.
- **Compute:** CPU-only (no CUDA). Even after download, feature extraction over ~2100
  synthetic tracks plus a mixed-domain fine-tune is a multi-hour-to-multi-day job; a
  meaningful, resumable pilot is feasible but cannot complete alongside a 6.8 h download in
  one workday.
- **Whole-archive checksum** can only be verified on a completed download; the pipeline
  records the official MD5 and refuses to proceed on mismatch.

What IS complete and committed: verified licence/provenance, a tested resumable+checksummed
downloader, a tested streaming subset extractor with traversal guards, a dependency-free
MIDI reader + deterministic chord derivation validated offline, and this audit. The
pipeline scales to the full dataset unchanged.

## Recommendation

- **Retain v1.** No full-band candidate exists yet; nothing replaces production weights,
  application defaults, hybrid settings, or release gating (all untouched).
- **Continue researching** the full-band candidate: pre-fetch and verify the 104 GB archive
  out-of-band (overnight at ~4.25 MB/s, or from a faster mirror), then run the committed
  pipeline for the grouped split, label statistics, v1 baselines, and a resumable CPU
  pilot fine-tune. Only then apply the frozen gates and the untouched-test evaluation.
- Slakh is synthetic; a future untouched real-recorded full-band benchmark remains required
  even if the pilot succeeds. GuitarSet p00 remains sealed (not needed for this task).
