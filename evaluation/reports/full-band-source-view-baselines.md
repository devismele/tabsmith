# Full-band source-view baselines (Slakh2100 pilot)

Deterministic pilot subset of 120 compositions (subset checksum `f600afba8c9bd1e6...`), one rendering per
composition. Inference only; no production weight, default or release gate is
touched. 2880 scored rows, 0 fallbacks, 0 skipped.

## Two limitations to read first

**1. Reference labels are note-level, not chord-level.** Mean derived region is
~0.6 s at ~100 regions/minute, against ~20 for GuitarSet human annotations. Root,
quality, detailed and no-chord accuracy remain meaningful. Fragmentation and
regions/minute are **not** comparable to GuitarSet, so the frozen full-band
fragmentation gate cannot honestly be applied to these labels.

**2. The percussion-only control does not measure what it was meant to.** It is
scored against each track's harmonic labels rather than an all-no-chord reference,
so it does not directly measure harmony hallucinated from rhythm. The frozen
`noChordRecallMinimum: 0.8` gate therefore cannot be evaluated as written. What the
column does show is how many chord regions each engine emits on drums-only audio,
which is indicative but not the gate.

Both are reported rather than reinterpreted to become passable.

## Accuracy by engine and source view

| engine | view | root | maj/min | detailed | no-chord P | R | F1 | regions/min |
|---|---|---|---|---|---|---|---|---|
| rule-v3 | full-mix | 0.590 | 0.562 | 0.519 | 0.86 | 0.24 | 0.38 | 28.1 |
| rule-v3 | oracle-harmonic | 0.603 | 0.577 | 0.532 | 0.86 | 0.26 | 0.40 | 28.8 |
| rule-v3 | oracle-guitar | 0.419 | 0.381 | 0.352 | 0.34 | 0.50 | 0.41 | 24.7 |
| rule-v3 | guitar-plus-bass | 0.549 | 0.490 | 0.446 | 0.53 | 0.39 | 0.45 | 28.4 |
| rule-v3 | guitar-absent-harmonic | 0.594 | 0.565 | 0.522 | 0.78 | 0.30 | 0.43 | 29.3 |
| rule-v3 | percussion-only | 0.132 | 0.077 | 0.077 | 0.30 | 0.50 | 0.38 | 12.7 |
| v1-ml | full-mix | 0.419 | 0.281 | 0.255 | 0.69 | 0.23 | 0.34 | 5.5 |
| v1-ml | oracle-harmonic | 0.451 | 0.327 | 0.297 | 0.74 | 0.25 | 0.38 | 6.3 |
| v1-ml | oracle-guitar | 0.370 | 0.248 | 0.223 | 0.43 | 0.41 | 0.42 | 5.8 |
| v1-ml | guitar-plus-bass | 0.430 | 0.303 | 0.272 | 0.57 | 0.31 | 0.40 | 6.8 |
| v1-ml | guitar-absent-harmonic | 0.457 | 0.343 | 0.310 | 0.73 | 0.27 | 0.40 | 6.8 |
| v1-ml | percussion-only | 0.130 | 0.079 | 0.069 | 0.40 | 0.37 | 0.39 | 1.5 |
| v1-hybrid | full-mix | 0.421 | 0.282 | 0.256 | 0.67 | 0.23 | 0.34 | 5.4 |
| v1-hybrid | oracle-harmonic | 0.452 | 0.329 | 0.299 | 0.72 | 0.25 | 0.38 | 6.3 |
| v1-hybrid | oracle-guitar | 0.370 | 0.248 | 0.223 | 0.42 | 0.42 | 0.42 | 5.7 |
| v1-hybrid | guitar-plus-bass | 0.438 | 0.309 | 0.277 | 0.56 | 0.31 | 0.40 | 7.0 |
| v1-hybrid | guitar-absent-harmonic | 0.460 | 0.347 | 0.313 | 0.72 | 0.27 | 0.40 | 6.7 |
| v1-hybrid | percussion-only | 0.140 | 0.080 | 0.070 | 0.39 | 0.37 | 0.38 | 1.4 |
| full-v2-ml | full-mix | 0.434 | 0.316 | 0.278 | 0.01 | 0.01 | 0.01 | 17.5 |
| full-v2-ml | oracle-harmonic | 0.453 | 0.326 | 0.285 | 0.02 | 0.01 | 0.01 | 18.4 |
| full-v2-ml | oracle-guitar | 0.353 | 0.258 | 0.226 | 0.02 | 0.00 | 0.00 | 16.8 |
| full-v2-ml | guitar-plus-bass | 0.430 | 0.303 | 0.263 | 0.02 | 0.00 | 0.00 | 18.9 |
| full-v2-ml | guitar-absent-harmonic | 0.447 | 0.319 | 0.278 | 0.02 | 0.01 | 0.01 | 18.7 |
| full-v2-ml | percussion-only | 0.086 | 0.061 | 0.048 | 0.05 | 0.01 | 0.01 | 4.1 |

## Separation versus oracle (detailed accuracy)

| engine | full mix | oracle harmonic | gain | oracle guitar | gain |
|---|---|---|---|---|---|
| rule-v3 | 0.519 | 0.532 | +0.013 | 0.352 | -0.167 |
| v1-ml | 0.255 | 0.297 | +0.042 | 0.223 | -0.032 |
| v1-hybrid | 0.256 | 0.299 | +0.043 | 0.223 | -0.033 |
| full-v2-ml | 0.278 | 0.285 | +0.007 | 0.226 | -0.052 |

Perfect harmonic separation buys between +0.007 and +0.043 detailed accuracy, and
isolating the guitar stem *costs* every engine (-0.03 to -0.17): a lone guitar has
less harmonic context than the full mix, which still carries bass and other
harmony. **Separation is not the bottleneck on this data; the chord model is.**

## Guitar presence

Slakh renders a guitar in 1707 of 1709 usable tracks (99.9%), and all 120 pilot
compositions contain one. A natural guitar-absent comparison is therefore not
available, and the guitar-absent test has to be read from the
`guitar-absent-harmonic` view, which removes guitar stems from every track.
On that view no engine collapses - detailed accuracy is within 0.003 of full mix
for the rule engine and *higher* for the learned engines - so there is no evidence
of guitar dependence.

## Interpretation

- The rule engine is roughly twice as accurate as every learned engine on
  full-band audio (0.519 detailed vs 0.255-0.278).
- v1 and full-v2 were trained exclusively on GuitarSet solo guitar and do not
  transfer to dense mixtures. That is the gap a full-band pilot would target.
- full-v2's no-chord head is effectively inert in this domain (P/R/F1 all around
  0.01 on every view) while v1 retains 0.34-0.42 F1. Whatever full-v2 gained on
  GuitarSet, its no-chord behaviour does not survive the domain shift.

Slakh2100 is rendered from MIDI. All of the above is synthetic-domain evidence.
