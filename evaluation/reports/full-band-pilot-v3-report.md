# Full-band scale study result

**Decision: 2 candidates passed every evaluable gate (`rehearsal-heavy-120`, `scale-600`). The gates admit rather than rank, and no ranking rule was frozen in advance, so this study does not select between them.**

Pilot `full-band-pilot-v3-20260731` under gate set `full-band-pilot-gates-20260727`.
Strategy `scale-the-full-band-training-pool`.
The candidates were frozen before any v3 training run started. Only the size of the full-band training pool and the epoch budget vary; the domain ratio, objective, seed and initialisation are inherited unchanged from the rehearsal-heavy candidate that pilot v2 measured. The eligibility gates are the pilot-v1 set, reused unchanged for the fourth consecutive study. The Slakh test split was never used for selection.

## Full-band development (Slakh validation subset, 40 compositions)

| model | view | root | detailed | no-chord F1 | regions/min |
|---|---|---|---|---|---|
| v1 | full-mix | 0.6649 | 0.4961 | 0.378 | 20.85 |
| v1 | oracle-harmonic | 0.6861 | 0.5314 | 0.439 | 21.46 |
| v1 | guitar-absent-harmonic | 0.6725 | 0.5223 | 0.452 | 21.68 |
| rehearsal-heavy-120 | full-mix | 0.7197 | 0.6174 | 0.500 | 22.45 |
| rehearsal-heavy-120 | oracle-harmonic | 0.7232 | 0.6108 | 0.470 | 23.65 |
| rehearsal-heavy-120 | guitar-absent-harmonic | 0.6953 | 0.5851 | 0.440 | 24.27 |
| scale-600 | full-mix | 0.7157 | 0.6255 | 0.503 | 21.94 |
| scale-600 | oracle-harmonic | 0.7208 | 0.6307 | 0.475 | 23.33 |
| scale-600 | guitar-absent-harmonic | 0.6911 | 0.6033 | 0.434 | 23.83 |
| scale-600-longer | full-mix | 0.7353 | 0.6399 | 0.509 | 23.32 |
| scale-600-longer | oracle-harmonic | 0.7357 | 0.6430 | 0.486 | 24.11 |
| scale-600-longer | guitar-absent-harmonic | 0.7010 | 0.6084 | 0.448 | 24.31 |

## GuitarSet preservation (p01-p05, both captures)

| model | capture | root | detailed | fragmentation | regions/min |
|---|---|---|---|---|---|
| v1 | audio_mono-mic | 0.5198 | 0.3258 | 0.6871 | 22.72 |
| v1 | audio_mono-pickup_mix | 0.5334 | 0.3363 | 0.7021 | 23.18 |
| rehearsal-heavy-120 | audio_mono-mic | 0.5784 | 0.4437 | 0.6873 | 22.17 |
| rehearsal-heavy-120 | audio_mono-pickup_mix | 0.5638 | 0.4196 | 0.6653 | 21.07 |
| scale-600 | audio_mono-mic | 0.5581 | 0.3891 | 0.6723 | 21.25 |
| scale-600 | audio_mono-pickup_mix | 0.5393 | 0.3798 | 0.6341 | 20.05 |
| scale-600-longer | audio_mono-mic | 0.6120 | 0.4567 | 0.7026 | 22.65 |
| scale-600-longer | audio_mono-pickup_mix | 0.5904 | 0.4222 | 0.6731 | 21.19 |

## Gate outcomes

### rehearsal-heavy-120 — PASSED

| result | gate | scope | measured | required |
|---|---|---|---|---|
| PASS | guitarSetRootRegression | audio_mono-mic | -5.86 | <= 2.0 pp |
| PASS | guitarSetDetailedRegression | audio_mono-mic | -11.79 | <= 2.0 pp |
| PASS | guitarSetFragmentationIncrease | audio_mono-mic | 0.0002 | <= 0.015 |
| PASS | guitarSetRegionsPerMinuteIncrease | audio_mono-mic | -0.551 | <= 0.75 |
| PASS | guitarSetRootRegression | audio_mono-pickup_mix | -3.04 | <= 2.0 pp |
| PASS | guitarSetDetailedRegression | audio_mono-pickup_mix | -8.33 | <= 2.0 pp |
| PASS | guitarSetFragmentationIncrease | audio_mono-pickup_mix | -0.0368 | <= 0.015 |
| PASS | guitarSetRegionsPerMinuteIncrease | audio_mono-pickup_mix | -2.12 | <= 0.75 |
| PASS | fullBandDetailedImprovement | full-mix | 12.13 | >= 2.0 pp |
| PASS | fullBandNoChordF1Improvement | full-mix | 0.1225 | >= 0.02 |
| PASS | mustImproveOnFullMix | full-mix | 12.13 | > 0 pp |
| PASS | guitarAbsentNoCollapse | guitar-absent-harmonic | 0.9477 | >= 0.75 of full-mix |
| PASS | mustImproveOnImperfectSeparatedViews | guitar-absent-harmonic | 6.28 | > 0 pp |

### scale-600 — PASSED

| result | gate | scope | measured | required |
|---|---|---|---|---|
| PASS | guitarSetRootRegression | audio_mono-mic | -3.83 | <= 2.0 pp |
| PASS | guitarSetDetailedRegression | audio_mono-mic | -6.33 | <= 2.0 pp |
| PASS | guitarSetFragmentationIncrease | audio_mono-mic | -0.0148 | <= 0.015 |
| PASS | guitarSetRegionsPerMinuteIncrease | audio_mono-mic | -1.47 | <= 0.75 |
| PASS | guitarSetRootRegression | audio_mono-pickup_mix | -0.59 | <= 2.0 pp |
| PASS | guitarSetDetailedRegression | audio_mono-pickup_mix | -4.35 | <= 2.0 pp |
| PASS | guitarSetFragmentationIncrease | audio_mono-pickup_mix | -0.068 | <= 0.015 |
| PASS | guitarSetRegionsPerMinuteIncrease | audio_mono-pickup_mix | -3.131 | <= 0.75 |
| PASS | fullBandDetailedImprovement | full-mix | 12.94 | >= 2.0 pp |
| PASS | fullBandNoChordF1Improvement | full-mix | 0.1254 | >= 0.02 |
| PASS | mustImproveOnFullMix | full-mix | 12.94 | > 0 pp |
| PASS | guitarAbsentNoCollapse | guitar-absent-harmonic | 0.9645 | >= 0.75 of full-mix |
| PASS | mustImproveOnImperfectSeparatedViews | guitar-absent-harmonic | 8.1 | > 0 pp |

### scale-600-longer — FAILED

| result | gate | scope | measured | required |
|---|---|---|---|---|
| PASS | guitarSetRootRegression | audio_mono-mic | -9.22 | <= 2.0 pp |
| PASS | guitarSetDetailedRegression | audio_mono-mic | -13.09 | <= 2.0 pp |
| **FAIL** | guitarSetFragmentationIncrease | audio_mono-mic | 0.0155 | <= 0.015 |
| PASS | guitarSetRegionsPerMinuteIncrease | audio_mono-mic | -0.072 | <= 0.75 |
| PASS | guitarSetRootRegression | audio_mono-pickup_mix | -5.7 | <= 2.0 pp |
| PASS | guitarSetDetailedRegression | audio_mono-pickup_mix | -8.59 | <= 2.0 pp |
| PASS | guitarSetFragmentationIncrease | audio_mono-pickup_mix | -0.029 | <= 0.015 |
| PASS | guitarSetRegionsPerMinuteIncrease | audio_mono-pickup_mix | -1.995 | <= 0.75 |
| PASS | fullBandDetailedImprovement | full-mix | 14.38 | >= 2.0 pp |
| PASS | fullBandNoChordF1Improvement | full-mix | 0.1318 | >= 0.02 |
| PASS | mustImproveOnFullMix | full-mix | 14.38 | > 0 pp |
| PASS | guitarAbsentNoCollapse | guitar-absent-harmonic | 0.9508 | >= 0.75 of full-mix |
| PASS | mustImproveOnImperfectSeparatedViews | guitar-absent-harmonic | 8.61 | > 0 pp |

## Gates that could not be evaluated

Reported as not-evaluable rather than passed:

- **fullBandFragmentation** — Reference labels are note-level (~0.6 s mean region, ~100 regions/min) versus ~20 for GuitarSet human annotations, so a fragmentation bound calibrated on GuitarSet does not transfer to them.
- **percussionOnlyNoChordRecall** — The percussion-only view is scored against each track's harmonic labels rather than an all-no-chord reference, so this gate does not measure what it was written to measure.

## Interpretation

Every row above is the same frozen `duration-viterbi` decoder; only the weights
differ. Two configurations are eligible, and the one with the best accuracy of
any model measured in this workstream is not among them.

**More full-band data helps full-band accuracy.** `scale-600` trains on five
times as many compositions and improves full-mix detailed accuracy 0.6174 →
0.6255 while *reducing* fragmentation on both captures (0.6873 → 0.6723 mic,
0.6653 → 0.6341 pickup). It passes every gate.

**It also costs solo-guitar accuracy.** `scale-600` gives up GuitarSet root
0.5784 → 0.5581 and detailed 0.4437 → 0.3891 against `rehearsal-heavy-120`. The
domain ratio is identical, so this is what a fixed rehearsal share buys when the
other domain gets five times richer: per-epoch rehearsal is unchanged while the
full-band signal it competes with is far more varied. Both remain far above v1
(root 0.5198, detailed 0.3258), which is why both still clear the preservation
gates.

**The longer budget produces the best model on every accuracy axis, and it is
ineligible.** `scale-600-longer` is the strongest result this workstream has
measured — GuitarSet root 0.6120, GuitarSet detailed 0.4567, full-mix detailed
0.6399, all bests — and it fails the microphone fragmentation gate at 0.0155
against 0.015. **It misses by 0.0005.**

That gate is not being moved. It was frozen before any full-band result existed,
it has now bound four consecutive studies, and a 0.0005 miss is exactly the
situation where relaxing a bound would convert a real constraint into a
formality. The same gate was held against `precision-boundary-loss` in
boundary-v3 when it missed by 0.0008.

**The mechanism is worth naming.** Doubling the budget improved development loss
from 3.9879 to 3.8706 and improved every accuracy metric, while making the
decoded output less temporally stable. Better frame-level fit and worse
segmentation are not in tension here — a sharper posterior switches state more
readily, and the duration constraint that was sufficient at 12 epochs is not
sufficient at 24. This is the same axis segmental-v3 and boundary-v3 mapped from
other directions, reached this time by training longer rather than by decoding
differently.

**A noise floor, measured rather than assumed.** `scale-600` and
`scale-600-longer` share a seed, data and schedule for their first twelve
epochs, so those epochs should be identical. Their development loss at epoch 12
differs by 0.004 (3.9879 against 3.9918), which is CPU reduction-order
nondeterminism. Differences smaller than roughly 0.004 in development loss are
therefore not meaningful, and the 0.0005 fragmentation miss is well inside the
region where a rerun could plausibly land on either side of the line. That is an
argument for treating the candidate as unresolved, not for admitting it.

## The selection question this study deliberately does not answer

Two candidates are eligible and they are not ordered by the gates, which admit
rather than rank:

| | GuitarSet root (mic) | GuitarSet detailed (mic) | full-mix detailed | fragmentation (mic) |
|---|---|---|---|---|
| `rehearsal-heavy-120` | **0.5784** | **0.4437** | 0.6174 | 0.6873 |
| `scale-600` | 0.5581 | 0.3891 | **0.6255** | **0.6723** |

`rehearsal-heavy-120` is better on solo guitar; `scale-600` is better on
full-band audio and on segmentation. Choosing between them requires a criterion,
and no criterion for ranking eligible candidates was frozen in advance. Inventing
one now, with both results already visible, would be choosing the winner and then
writing the rule that selects it.

So this study reports two eligible configurations and stops there. A ranking rule
should be frozen before the comparison is made, and the obvious candidate for it
— relative weight on the guitar domain versus full-band audio — is a product
question about what Tabsmith is for, not a question the metrics can settle.

## Limits

- **p00 is spent**, so nothing here is a performer-level generalisation claim.
  The p00 confirmation showed GuitarSet gains on p01–p05 are substantially a
  rehearsal effect, and that applies to every GuitarSet number in this table.
- The full-band gain is the part that survived p00 scrutiny, and it is the axis
  where scaling helped most — but it is measured on MIDI-rendered audio and
  remains synthetic-domain evidence.
- The enlarged pool is a strict superset of the pilot's, so the comparison is
  nested rather than independent.
- `scale-600-longer` was still improving when its 24-epoch budget ended, so its
  accuracy figures are a lower bound and its fragmentation is likely still
  drifting in the wrong direction.
- Nothing is promoted. No production weights, application defaults or release
  gating changed.
