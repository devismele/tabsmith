# Full-band pilot v2 result

**Decision: retain v1. No pilot-v2 candidate is eligible.**

Pilot `full-band-pilot-v2-20260731` under gate set `full-band-pilot-gates-20260727`.
Strategy `preserve-root-identification-under-mixed-domain-finetune` and its candidates were frozen before any
v2 training run started. The eligibility gates are the pilot-v1 set reused
unchanged - the gate v1 failed on was not allowed to move for v2. p00 was never
accessed and the Slakh test split was never used for selection.

## Full-band development (Slakh validation subset, 40 compositions)

| model | view | root | detailed | no-chord F1 | regions/min |
|---|---|---|---|---|---|
| v1 | full-mix | 0.6649 | 0.4961 | 0.378 | 20.85 |
| v1 | oracle-harmonic | 0.6861 | 0.5314 | 0.439 | 21.46 |
| v1 | guitar-absent-harmonic | 0.6725 | 0.5223 | 0.452 | 21.68 |
| low-learning-rate | full-mix | 0.7127 | 0.6062 | 0.500 | 22.23 |
| low-learning-rate | oracle-harmonic | 0.7182 | 0.6116 | 0.491 | 23.66 |
| low-learning-rate | guitar-absent-harmonic | 0.6889 | 0.5902 | 0.456 | 24.75 |
| rehearsal-heavy | full-mix | 0.7227 | 0.6178 | 0.492 | 24.44 |
| rehearsal-heavy | oracle-harmonic | 0.7273 | 0.6137 | 0.472 | 25.67 |
| rehearsal-heavy | guitar-absent-harmonic | 0.6997 | 0.5887 | 0.449 | 26.68 |
| root-anchored-distillation | full-mix | 0.7207 | 0.6367 | 0.514 | 23.45 |
| root-anchored-distillation | oracle-harmonic | 0.7242 | 0.6332 | 0.490 | 24.75 |
| root-anchored-distillation | guitar-absent-harmonic | 0.7013 | 0.6125 | 0.473 | 25.38 |

## GuitarSet preservation (p01-p05, both captures)

| model | capture | root | detailed | fragmentation | regions/min |
|---|---|---|---|---|---|
| v1 | audio_mono-mic | 0.5198 | 0.3258 | 0.6871 | 22.72 |
| v1 | audio_mono-pickup_mix | 0.5334 | 0.3363 | 0.7021 | 23.18 |
| low-learning-rate | audio_mono-mic | 0.4388 | 0.2914 | 0.6174 | 20.89 |
| low-learning-rate | audio_mono-pickup_mix | 0.4259 | 0.2996 | 0.5672 | 19.67 |
| rehearsal-heavy | audio_mono-mic | 0.5802 | 0.4417 | 0.7439 | 25.64 |
| rehearsal-heavy | audio_mono-pickup_mix | 0.5653 | 0.4182 | 0.7230 | 24.64 |
| root-anchored-distillation | audio_mono-mic | 0.4522 | 0.3293 | 0.6277 | 20.92 |
| root-anchored-distillation | audio_mono-pickup_mix | 0.4304 | 0.3217 | 0.5774 | 19.28 |

## Gate outcomes

### low-learning-rate — FAILED

| result | gate | scope | measured | required |
|---|---|---|---|---|
| **FAIL** | guitarSetRootRegression | audio_mono-mic | 8.1 | <= 2.0 pp |
| **FAIL** | guitarSetDetailedRegression | audio_mono-mic | 3.44 | <= 2.0 pp |
| PASS | guitarSetFragmentationIncrease | audio_mono-mic | -0.0697 | <= 0.015 |
| PASS | guitarSetRegionsPerMinuteIncrease | audio_mono-mic | -1.825 | <= 0.75 |
| **FAIL** | guitarSetRootRegression | audio_mono-pickup_mix | 10.75 | <= 2.0 pp |
| **FAIL** | guitarSetDetailedRegression | audio_mono-pickup_mix | 3.67 | <= 2.0 pp |
| PASS | guitarSetFragmentationIncrease | audio_mono-pickup_mix | -0.1349 | <= 0.015 |
| PASS | guitarSetRegionsPerMinuteIncrease | audio_mono-pickup_mix | -3.518 | <= 0.75 |
| PASS | fullBandDetailedImprovement | full-mix | 11.01 | >= 2.0 pp |
| PASS | fullBandNoChordF1Improvement | full-mix | 0.1229 | >= 0.02 |
| PASS | mustImproveOnFullMix | full-mix | 11.01 | > 0 pp |
| PASS | guitarAbsentNoCollapse | guitar-absent-harmonic | 0.9736 | >= 0.75 of full-mix |
| PASS | mustImproveOnImperfectSeparatedViews | guitar-absent-harmonic | 6.79 | > 0 pp |

### rehearsal-heavy — FAILED

| result | gate | scope | measured | required |
|---|---|---|---|---|
| PASS | guitarSetRootRegression | audio_mono-mic | -6.04 | <= 2.0 pp |
| PASS | guitarSetDetailedRegression | audio_mono-mic | -11.59 | <= 2.0 pp |
| **FAIL** | guitarSetFragmentationIncrease | audio_mono-mic | 0.0568 | <= 0.015 |
| **FAIL** | guitarSetRegionsPerMinuteIncrease | audio_mono-mic | 2.921 | <= 0.75 |
| PASS | guitarSetRootRegression | audio_mono-pickup_mix | -3.19 | <= 2.0 pp |
| PASS | guitarSetDetailedRegression | audio_mono-pickup_mix | -8.19 | <= 2.0 pp |
| **FAIL** | guitarSetFragmentationIncrease | audio_mono-pickup_mix | 0.0209 | <= 0.015 |
| **FAIL** | guitarSetRegionsPerMinuteIncrease | audio_mono-pickup_mix | 1.457 | <= 0.75 |
| PASS | fullBandDetailedImprovement | full-mix | 12.17 | >= 2.0 pp |
| PASS | fullBandNoChordF1Improvement | full-mix | 0.1148 | >= 0.02 |
| PASS | mustImproveOnFullMix | full-mix | 12.17 | > 0 pp |
| PASS | guitarAbsentNoCollapse | guitar-absent-harmonic | 0.9529 | >= 0.75 of full-mix |
| PASS | mustImproveOnImperfectSeparatedViews | guitar-absent-harmonic | 6.64 | > 0 pp |

### root-anchored-distillation — FAILED

| result | gate | scope | measured | required |
|---|---|---|---|---|
| **FAIL** | guitarSetRootRegression | audio_mono-mic | 6.76 | <= 2.0 pp |
| PASS | guitarSetDetailedRegression | audio_mono-mic | -0.35 | <= 2.0 pp |
| PASS | guitarSetFragmentationIncrease | audio_mono-mic | -0.0594 | <= 0.015 |
| PASS | guitarSetRegionsPerMinuteIncrease | audio_mono-mic | -1.799 | <= 0.75 |
| **FAIL** | guitarSetRootRegression | audio_mono-pickup_mix | 10.3 | <= 2.0 pp |
| PASS | guitarSetDetailedRegression | audio_mono-pickup_mix | 1.46 | <= 2.0 pp |
| PASS | guitarSetFragmentationIncrease | audio_mono-pickup_mix | -0.1247 | <= 0.015 |
| PASS | guitarSetRegionsPerMinuteIncrease | audio_mono-pickup_mix | -3.906 | <= 0.75 |
| PASS | fullBandDetailedImprovement | full-mix | 14.06 | >= 2.0 pp |
| PASS | fullBandNoChordF1Improvement | full-mix | 0.1368 | >= 0.02 |
| PASS | mustImproveOnFullMix | full-mix | 14.06 | > 0 pp |
| PASS | guitarAbsentNoCollapse | guitar-absent-harmonic | 0.962 | >= 0.75 of full-mix |
| PASS | mustImproveOnImperfectSeparatedViews | guitar-absent-harmonic | 9.02 | > 0 pp |

## Gates that could not be evaluated

Reported as not-evaluable rather than passed:

- **fullBandFragmentation** — Reference labels are note-level (~0.6 s mean region, ~100 regions/min) versus ~20 for GuitarSet human annotations, so a fragmentation bound calibrated on GuitarSet does not transfer to them.
- **percussionOnlyNoChordRecall** — The percussion-only view is scored against each track's harmonic labels rather than an all-no-chord reference, so this gate does not measure what it was written to measure.

## Interpretation

**The preservation failure is solved, and rehearsal composition is what solved
it.** `rehearsal-heavy` does not merely stay inside the 2.0 pp root tolerance —
it *improves* GuitarSet root accuracy by **+6.04 pp** on microphone and
**+3.19 pp** on pickup, and detailed accuracy by **+11.59** and **+8.19 pp**,
while gaining **+12.17 pp** full-mix detailed accuracy and +0.114 no-chord F1 on
full-band audio. Against the pilot-v1 primary it converts a 4.8 pp root loss
into a 6.0 pp root gain for 2.66 pp of full-band detailed accuracy.

**The ablation rules out the cheap explanation.** `low-learning-rate` ran the
same 50/50 stream at 0.4x the step size and preserved *nothing*: root regressed
8.1 pp (microphone) and 10.75 pp (pickup), worse than the pilot-v1 primary, and
it also broke the detailed-accuracy gate. Drifting less is not what preserves
solo guitar; training on more of it is. Without this candidate,
`rehearsal-heavy`'s result could have been read as "any gentler fine-tune would
do", and that reading would have been wrong.

**The anchor did not bind.** `root-anchored-distillation` ran the identical
50/50 stream, schedule and seed as the pilot-v1 primary with a KL anchor to the
frozen v1 root head, and finished *worse* on the gate it targeted: 6.76 pp and
10.3 pp root regression against v1's 4.8 and 8.18. Its full-band detailed
accuracy also came in slightly below that primary (0.6367 against 0.6444). The
anchor term fell from 0.27 to 0.14 during training while the total loss sat
between 4 and 6, so at the frozen weight of 1.0 it was roughly two percent of
the objective — too weak to constrain anything. This bounds root distillation at
weight 1.0, exactly as the frozen configuration said it would; it does not
refute the mechanism, and the weight was deliberately not tuned because there
was no honest budget to tune it against.

**The binding constraint moved.** `rehearsal-heavy` fails on over-segmentation:
fragmentation rises +0.0568 (microphone) and +0.0209 (pickup) against a 0.015
allowance, and regions per minute rise +2.92 and +1.46 against 0.75. Every gate
it failed is a segmentation gate, and every accuracy gate it faced — on both
domains — it passed. That is the same wall the temporal-v2, segmental-v3 and
boundary-v3 studies each hit from a different direction: this model's residual
weakness is where it places boundaries, not what chords it names.

## What this changes

The full-band question and the fragmentation question are now separable in a way
they were not before pilot v2. A model exists that is better than v1 on
full-band audio *and* better than v1 on solo guitar in both accuracy metrics,
whose only remaining defect is that it changes chord too often. Prior studies
attacked fragmentation with decoders and boundary supervision and moved it by
±0.014; this candidate moved accuracy by 6 to 12 points and moved fragmentation
the wrong way by 0.057.

The next study is therefore over-segmentation control applied to a
rehearsal-heavy mixed-domain model — not another preservation mechanism, and not
another boundary-head intervention. `segmental-full` decoding was measured in
segmental-v3 to cut fragmentation 0.672 -> 0.649 on the full-v2 model without
costing accuracy; it has never been applied to a model in this family that
already passes the accuracy gates.

## Limits

- **`rehearsal-heavy`'s GuitarSet gain is measured on performers it rehearsed
  on.** p01-p05 are the performers v1 trained on and the ones this candidate
  trains on further, so +6.04 pp is a forgetting check that came out positive,
  **not** evidence of generalisation to unseen players. p00 remains sealed and is
  the only honest test of that claim, and it stays sealed because no candidate is
  eligible. This limit matters more now than in pilot v1, where the number moved
  the other way.
- Every candidate was still improving when the frozen 12-epoch budget ended, so
  all full-band figures are lower bounds rather than converged results.
- Reference labels are note-level (~0.6 s mean region), so full-band
  fragmentation and regions/minute are not comparable to GuitarSet and the
  full-band fragmentation gate is reported as not-evaluable.
- Slakh2100 is rendered from MIDI. Even a fully passing pilot would be
  synthetic-domain evidence and would not establish production readiness.
- The distillation result bounds one weight of one anchor, not the idea of
  anchoring.
