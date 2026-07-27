# Full-band harmony v1 foundation

This branch prepares the next data and evaluation gate. It does not contain,
download, train on, or authorize any full-band audio.

## Safety contract

A dataset is usable only when its portable manifest records:

- a verified license that explicitly permits model training and evaluation;
- source/distribution identity, attribution, and archive checksums;
- relative audio paths plus per-view SHA-256 checksums;
- artist, composition, recording, and split-group IDs;
- timed chord regions or normalized symbolic note events;
- explicit no-chord coverage;
- a frozen group-aware split with no artist/composition leakage.

The loader rejects absolute paths and path traversal. A local data root is
provided at execution time and is never written to reports.

The repository does not scrape or download commercial music. Dataset licensing
must be reviewed by the operator; these checks are engineering gates, not legal
advice.

## Audio views and baselines

Each recording should provide these paired views:

1. `full-mix`
2. `harmony-stem`
3. `guitar-stem`
4. `guitar-plus-bass`

The readiness plan evaluates rule v3, v1 ML-only, and v1 hybrid on every view
using identical tracks, chord regions, durations, vocabulary, and metrics.
Alternate views are paired captures of one performance, not independent songs.
The frozen planned matrix and paired-statistics contract live in
`ml/configs/full-band-harmony-v1-evaluation.json`.

The downstream source-separation comparison includes chord, note, tab, tuning,
and playable-string assignment metrics. It answers whether a separated source
improves Tabsmith, not merely whether it sounds clean.

## Symbolic labels

`ml.full_band.symbolic` accepts normalized MIDI-like note events (`start`,
`end`, `pitches`, optional `velocity`). It deterministically scores major,
minor, and dominant-seventh pitch sets, merges identical neighbours, and fills
all gaps with `N`. Empty/no-guitar examples become one full-duration `N`
region. This is deliberately reproducible and does not require generated audio.

## Local use

Copy `ml/configs/full-band-harmony-v1-manifest.example.json` outside the
repository and fill it only after license review. Then freeze a grouped split
and run readiness locally. Never commit the local manifest when it contains
private provenance or paths.

```powershell
npm run split:full-band -- `
  --manifest <portable-local-manifest.json> `
  --output <portable-frozen-split.json> `
  --seed 20260727
```

The tracked blocker report is generated with:

```powershell
npm run readiness:full-band
```

A real local manifest can be checked with:

```powershell
python -m ml.full_band.readiness `
  --manifest <portable-local-manifest.json> `
  --data-root <licensed-dataset-root>
```

After the readiness gate is cleared, the existing real-data trainer can consume
one or more approved views through `--full-band-manifest`,
`--full-band-root`, and `--full-band-views`. It preserves the frozen full-band
split while leaving GuitarSet/Billboard support unchanged. This branch does not
run that command.

The current tracked report must remain blocked until a suitable licensed
full-band dataset exists. No large training run should begin from this branch.
