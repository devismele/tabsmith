# Full-band pilot result

**Decision: retain v1. No pilot candidate is eligible.**

Pilot `full-band-pilot-v1-20260728` under gate set `full-band-pilot-gates-20260727`. Strategy and
gates were frozen from the Phase 12 baselines before any pilot run started and were
not altered afterwards. p00 was never accessed and the Slakh test split was never
used for selection.

## Full-band development (Slakh validation subset, 40 compositions)

| model | view | root | detailed | no-chord F1 | regions/min |
|---|---|---|---|---|---|
| v1 | full-mix | 0.6649 | 0.4961 | 0.378 | 20.85 |
| v1 | oracle-harmonic | 0.6861 | 0.5314 | 0.439 | 21.46 |
| v1 | guitar-absent-harmonic | 0.6725 | 0.5223 | 0.452 | 21.68 |
| mixed-domain-finetune | full-mix | 0.7296 | 0.6444 | 0.512 | 24.08 |
| mixed-domain-finetune | oracle-harmonic | 0.7314 | 0.6424 | 0.497 | 25.35 |
| mixed-domain-finetune | guitar-absent-harmonic | 0.7016 | 0.6154 | 0.466 | 26.19 |
| guitarset-only-control | full-mix | 0.6523 | 0.5130 | 0.427 | 19.98 |
| guitarset-only-control | oracle-harmonic | 0.6639 | 0.5194 | 0.376 | 21.24 |
| guitarset-only-control | guitar-absent-harmonic | 0.6570 | 0.5091 | 0.404 | 21.37 |

## GuitarSet preservation (p01-p05, both captures)

| model | capture | root | detailed | fragmentation | regions/min |
|---|---|---|---|---|---|
| v1 | audio_mono-mic | 0.5198 | 0.3258 | 0.6871 | 22.72 |
| v1 | audio_mono-pickup_mix | 0.5334 | 0.3363 | 0.7021 | 23.18 |
| mixed-domain-finetune | audio_mono-mic | 0.4718 | 0.3412 | 0.6634 | 22.89 |
| mixed-domain-finetune | audio_mono-pickup_mix | 0.4516 | 0.3306 | 0.6004 | 20.95 |
| guitarset-only-control | audio_mono-mic | 0.5591 | 0.4198 | 0.6873 | 22.00 |
| guitarset-only-control | audio_mono-pickup_mix | 0.5494 | 0.4125 | 0.6763 | 21.98 |

## Gate outcomes

### mixed-domain-finetune — FAILED

| result | gate | scope | measured | required |
|---|---|---|---|---|
| **FAIL** | guitarSetRootRegression | audio_mono-mic | 4.8 | <= 2.0 pp |
| PASS | guitarSetDetailedRegression | audio_mono-mic | -1.54 | <= 2.0 pp |
| PASS | guitarSetFragmentationIncrease | audio_mono-mic | -0.0237 | <= 0.015 |
| PASS | guitarSetRegionsPerMinuteIncrease | audio_mono-mic | 0.171 | <= 0.75 |
| **FAIL** | guitarSetRootRegression | audio_mono-pickup_mix | 8.18 | <= 2.0 pp |
| PASS | guitarSetDetailedRegression | audio_mono-pickup_mix | 0.57 | <= 2.0 pp |
| PASS | guitarSetFragmentationIncrease | audio_mono-pickup_mix | -0.1017 | <= 0.015 |
| PASS | guitarSetRegionsPerMinuteIncrease | audio_mono-pickup_mix | -2.232 | <= 0.75 |
| PASS | fullBandDetailedImprovement | full-mix | 14.83 | >= 2.0 pp |
| PASS | fullBandNoChordF1Improvement | full-mix | 0.1346 | >= 0.02 |
| PASS | mustImproveOnFullMix | full-mix | 14.83 | > 0 pp |
| PASS | guitarAbsentNoCollapse | guitar-absent-harmonic | 0.955 | >= 0.75 of full-mix |
| PASS | mustImproveOnImperfectSeparatedViews | guitar-absent-harmonic | 9.31 | > 0 pp |

### guitarset-only-control — FAILED

| result | gate | scope | measured | required |
|---|---|---|---|---|
| PASS | guitarSetRootRegression | audio_mono-mic | -3.93 | <= 2.0 pp |
| PASS | guitarSetDetailedRegression | audio_mono-mic | -9.4 | <= 2.0 pp |
| PASS | guitarSetFragmentationIncrease | audio_mono-mic | 0.0002 | <= 0.015 |
| PASS | guitarSetRegionsPerMinuteIncrease | audio_mono-mic | -0.722 | <= 0.75 |
| PASS | guitarSetRootRegression | audio_mono-pickup_mix | -1.6 | <= 2.0 pp |
| PASS | guitarSetDetailedRegression | audio_mono-pickup_mix | -7.62 | <= 2.0 pp |
| PASS | guitarSetFragmentationIncrease | audio_mono-pickup_mix | -0.0258 | <= 0.015 |
| PASS | guitarSetRegionsPerMinuteIncrease | audio_mono-pickup_mix | -1.201 | <= 0.75 |
| **FAIL** | fullBandDetailedImprovement | full-mix | 1.69 | >= 2.0 pp |
| PASS | fullBandNoChordF1Improvement | full-mix | 0.0496 | >= 0.02 |
| PASS | mustImproveOnFullMix | full-mix | 1.69 | > 0 pp |
| PASS | guitarAbsentNoCollapse | guitar-absent-harmonic | 0.9924 | >= 0.75 of full-mix |
| **FAIL** | mustImproveOnImperfectSeparatedViews | guitar-absent-harmonic | -1.32 | > 0 pp |

## Gates that could not be evaluated

Reported as not-evaluable rather than passed:

- **fullBandFragmentation** — Reference labels are note-level (~0.6 s mean region, ~100 regions/min) versus ~20 for GuitarSet human annotations, so a fragmentation bound calibrated on GuitarSet does not transfer to them.
- **percussionOnlyNoChordRecall** — The percussion-only view is scored against each track's harmonic labels rather than an all-no-chord reference, so this gate does not measure what it was written to measure.

## Interpretation

- **Mixed-domain fine-tuning works on the target domain.** Full-mix detailed
  accuracy improves +14.83 pp over v1 and no-chord F1 rises
  +0.135. Every full-band gate passes,
  most by a wide margin, and there is no guitar-absent collapse.
- **The control isolates the cause.** GuitarSet-only training on the same
  schedule gains only +1.69 pp on the full mix, against the primary's
  +14.83 pp. The improvement comes from full-band exposure, not from
  continued optimisation.
- **It is rejected on preservation, not on capability.** The primary loses 4.8 pp
  (microphone) and 8.2 pp (pickup) of GuitarSet root accuracy against a frozen
  2.0 pp tolerance. A model that forgets solo guitar is not an improvement to
  Tabsmith's actual use case, which is why that gate is hard.
- Notably GuitarSet *detailed* accuracy did **not** regress (-1.54 pp on
  microphone is an improvement, +0.57 pp on pickup is inside tolerance), and
  fragmentation improved on both captures. The loss is specific to root
  identification.

### A correction

During training I described the control as degrading on full-band audio, based
on its Slakh development *loss* rising monotonically (7.66 to 8.55). The gate
metrics contradict that: its full-mix detailed accuracy actually improved
+1.69 pp. Loss and accuracy diverged, and the loss-based reading was
wrong. The comparison that matters is unchanged and in fact stronger on
accuracy: the primary gains roughly nine times as much as the control.

## Limits

- The primary was still improving when the frozen 12-epoch budget ended, so its
  full-band figures are a lower bound rather than a converged result.
- GuitarSet preservation is measured on p01-p05, which v1 was already trained on
  and which the frozen config rehearses. It is a catastrophic-forgetting check on
  data v1 already knew, not a held-out generalisation claim.
- Reference labels are note-level, so full-band fragmentation and regions/minute
  are not comparable to GuitarSet.
- Slakh2100 is rendered from MIDI. This is synthetic-domain evidence and does not
  establish production readiness.
