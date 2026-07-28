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
| v1-ml | full-mix | 0.638 | 0.542 | 0.488 | 0.91 | 0.22 | 0.35 | 20.1 |
| v1-ml | oracle-harmonic | 0.649 | 0.567 | 0.516 | 0.89 | 0.25 | 0.39 | 20.6 |
| v1-ml | oracle-guitar | 0.499 | 0.403 | 0.358 | 0.49 | 0.41 | 0.45 | 17.9 |
| v1-ml | guitar-plus-bass | 0.608 | 0.493 | 0.439 | 0.66 | 0.31 | 0.42 | 20.4 |
| v1-ml | guitar-absent-harmonic | 0.640 | 0.560 | 0.508 | 0.83 | 0.27 | 0.41 | 20.8 |
| v1-ml | percussion-only | 0.131 | 0.103 | 0.080 | 0.52 | 0.35 | 0.42 | 3.8 |
| v1-hybrid | full-mix | 0.641 | 0.544 | 0.490 | 0.91 | 0.22 | 0.35 | 20.2 |
| v1-hybrid | oracle-harmonic | 0.653 | 0.571 | 0.519 | 0.88 | 0.25 | 0.39 | 20.7 |
| v1-hybrid | oracle-guitar | 0.499 | 0.403 | 0.357 | 0.49 | 0.41 | 0.44 | 18.0 |
| v1-hybrid | guitar-plus-bass | 0.615 | 0.498 | 0.443 | 0.66 | 0.31 | 0.42 | 20.9 |
| v1-hybrid | guitar-absent-harmonic | 0.645 | 0.566 | 0.512 | 0.83 | 0.27 | 0.41 | 20.9 |
| v1-hybrid | percussion-only | 0.130 | 0.102 | 0.079 | 0.52 | 0.36 | 0.42 | 3.7 |
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
| v1-ml | 0.488 | 0.516 | +0.028 | 0.358 | -0.130 |
| v1-hybrid | 0.490 | 0.519 | +0.029 | 0.357 | -0.133 |
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

- **v1 transfers to full-band audio better than expected.** On the full mix it
  reaches 0.488 detailed accuracy against the rule engine's 0.519, and it is
  *ahead* on root accuracy (0.638 vs 0.590). The learned model is competitive,
  not outclassed.
- **full-v2 does not transfer.** At 0.278 detailed it is far behind both v1 and
  the rule engine, on its own correct feature pipeline. Whatever it gained on
  GuitarSet does not survive the domain shift.
- **full-v2's no-chord head is effectively inert here** (P/R/F1 all around 0.01
  on every view) while v1 retains ~0.35 F1.
- **Separation is not the bottleneck.** Perfect harmonic separation is worth only
  +0.007 to +0.029 detailed accuracy, and isolating the guitar stem costs every
  engine (-0.05 to -0.17).

### Correction notice

An earlier version of this report scored every engine through one global feature
pipeline. v1 was trained on `numpy-chroma-v1` but was being fed
`harmony-features-v1`, which understated it badly: full-mix detailed accuracy was
reported as 0.255 when it is actually 0.488, and root as 0.419 when it is 0.638.
Each engine now uses the pipeline recorded in its own checkpoint.

The earlier conclusion that 'the rule engine is roughly twice as accurate as every
learned engine' was an artifact of that bug and is withdrawn. The corrected gap
between the rule engine and v1 on the full mix is 0.031 detailed accuracy, not
0.264 - which materially weakens the motivation for a full-band fine-tune, since
there is far less headroom than the original numbers implied.

Slakh2100 is rendered from MIDI. All of the above is synthetic-domain evidence.
