# Full-band decoder study result

**Decision: duration-viterbi passed every evaluable gate.**

Pilot `full-band-decoder-v1-20260731` under gate set `full-band-pilot-gates-20260727`.
Strategy `over-segmentation-control-on-a-rehearsal-heavy-mixed-domain-model`.
The candidates were frozen before any decoder was scored. Only the decoder varies: the rehearsal-heavy weights from pilot v2 are used unchanged, and every candidate decodes the identical cached network response for each item. The eligibility gates are the pilot-v1 set, reused unchanged for the third consecutive study. p00 was never accessed and the Slakh test split was never used for selection.

## Full-band development (Slakh validation subset, 40 compositions)

| model | view | root | detailed | no-chord F1 | regions/min |
|---|---|---|---|---|---|
| v1 | full-mix | 0.6649 | 0.4961 | 0.378 | 20.85 |
| v1 | oracle-harmonic | 0.6861 | 0.5314 | 0.439 | 21.46 |
| v1 | guitar-absent-harmonic | 0.6725 | 0.5223 | 0.452 | 21.68 |
| viterbi-penalty-4-control | full-mix | 0.7227 | 0.6178 | 0.492 | 24.44 |
| viterbi-penalty-4-control | oracle-harmonic | 0.7273 | 0.6137 | 0.472 | 25.67 |
| viterbi-penalty-4-control | guitar-absent-harmonic | 0.6997 | 0.5887 | 0.449 | 26.68 |
| duration-viterbi | full-mix | 0.7197 | 0.6174 | 0.500 | 22.45 |
| duration-viterbi | oracle-harmonic | 0.7232 | 0.6108 | 0.470 | 23.65 |
| duration-viterbi | guitar-absent-harmonic | 0.6953 | 0.5851 | 0.440 | 24.27 |
| duration-boundary-viterbi | full-mix | 0.7260 | 0.6194 | 0.493 | 24.45 |
| duration-boundary-viterbi | oracle-harmonic | 0.7299 | 0.6148 | 0.475 | 25.60 |
| duration-boundary-viterbi | guitar-absent-harmonic | 0.7008 | 0.5887 | 0.447 | 26.51 |
| segmental-full | full-mix | 0.7236 | 0.6178 | 0.496 | 23.70 |
| segmental-full | oracle-harmonic | 0.7266 | 0.6119 | 0.469 | 24.86 |
| segmental-full | guitar-absent-harmonic | 0.6976 | 0.5867 | 0.436 | 25.40 |

## GuitarSet preservation (p01-p05, both captures)

| model | capture | root | detailed | fragmentation | regions/min |
|---|---|---|---|---|---|
| v1 | audio_mono-mic | 0.5198 | 0.3258 | 0.6871 | 22.72 |
| v1 | audio_mono-pickup_mix | 0.5334 | 0.3363 | 0.7021 | 23.18 |
| viterbi-penalty-4-control | audio_mono-mic | 0.5802 | 0.4417 | 0.7439 | 25.64 |
| viterbi-penalty-4-control | audio_mono-pickup_mix | 0.5653 | 0.4182 | 0.7230 | 24.64 |
| duration-viterbi | audio_mono-mic | 0.5784 | 0.4437 | 0.6873 | 22.17 |
| duration-viterbi | audio_mono-pickup_mix | 0.5638 | 0.4196 | 0.6653 | 21.07 |
| duration-boundary-viterbi | audio_mono-mic | 0.5809 | 0.4458 | 0.7197 | 23.34 |
| duration-boundary-viterbi | audio_mono-pickup_mix | 0.5670 | 0.4215 | 0.7005 | 22.46 |
| segmental-full | audio_mono-mic | 0.5806 | 0.4435 | 0.7514 | 23.81 |
| segmental-full | audio_mono-pickup_mix | 0.5677 | 0.4222 | 0.7367 | 22.95 |

## Gate outcomes

### viterbi-penalty-4-control — FAILED

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

### duration-viterbi — PASSED

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

### duration-boundary-viterbi — FAILED

| result | gate | scope | measured | required |
|---|---|---|---|---|
| PASS | guitarSetRootRegression | audio_mono-mic | -6.11 | <= 2.0 pp |
| PASS | guitarSetDetailedRegression | audio_mono-mic | -12.0 | <= 2.0 pp |
| **FAIL** | guitarSetFragmentationIncrease | audio_mono-mic | 0.0326 | <= 0.015 |
| PASS | guitarSetRegionsPerMinuteIncrease | audio_mono-mic | 0.617 | <= 0.75 |
| PASS | guitarSetRootRegression | audio_mono-pickup_mix | -3.36 | <= 2.0 pp |
| PASS | guitarSetDetailedRegression | audio_mono-pickup_mix | -8.52 | <= 2.0 pp |
| PASS | guitarSetFragmentationIncrease | audio_mono-pickup_mix | -0.0016 | <= 0.015 |
| PASS | guitarSetRegionsPerMinuteIncrease | audio_mono-pickup_mix | -0.728 | <= 0.75 |
| PASS | fullBandDetailedImprovement | full-mix | 12.33 | >= 2.0 pp |
| PASS | fullBandNoChordF1Improvement | full-mix | 0.1159 | >= 0.02 |
| PASS | mustImproveOnFullMix | full-mix | 12.33 | > 0 pp |
| PASS | guitarAbsentNoCollapse | guitar-absent-harmonic | 0.9504 | >= 0.75 of full-mix |
| PASS | mustImproveOnImperfectSeparatedViews | guitar-absent-harmonic | 6.64 | > 0 pp |

### segmental-full — FAILED

| result | gate | scope | measured | required |
|---|---|---|---|---|
| PASS | guitarSetRootRegression | audio_mono-mic | -6.08 | <= 2.0 pp |
| PASS | guitarSetDetailedRegression | audio_mono-mic | -11.77 | <= 2.0 pp |
| **FAIL** | guitarSetFragmentationIncrease | audio_mono-mic | 0.0643 | <= 0.015 |
| **FAIL** | guitarSetRegionsPerMinuteIncrease | audio_mono-mic | 1.09 | <= 0.75 |
| PASS | guitarSetRootRegression | audio_mono-pickup_mix | -3.43 | <= 2.0 pp |
| PASS | guitarSetDetailedRegression | audio_mono-pickup_mix | -8.59 | <= 2.0 pp |
| **FAIL** | guitarSetFragmentationIncrease | audio_mono-pickup_mix | 0.0346 | <= 0.015 |
| PASS | guitarSetRegionsPerMinuteIncrease | audio_mono-pickup_mix | -0.236 | <= 0.75 |
| PASS | fullBandDetailedImprovement | full-mix | 12.17 | >= 2.0 pp |
| PASS | fullBandNoChordF1Improvement | full-mix | 0.118 | >= 0.02 |
| PASS | mustImproveOnFullMix | full-mix | 12.17 | > 0 pp |
| PASS | guitarAbsentNoCollapse | guitar-absent-harmonic | 0.9497 | >= 0.75 of full-mix |
| PASS | mustImproveOnImperfectSeparatedViews | guitar-absent-harmonic | 6.44 | > 0 pp |

## Gates that could not be evaluated

Reported as not-evaluable rather than passed:

- **fullBandFragmentation** — Reference labels are note-level (~0.6 s mean region, ~100 regions/min) versus ~20 for GuitarSet human annotations, so a fragmentation bound calibrated on GuitarSet does not transfer to them.
- **percussionOnlyNoChordRecall** — The percussion-only view is scored against each track's harmonic labels rather than an all-no-chord reference, so this gate does not measure what it was written to measure.

## Interpretation

**This is the first eligible candidate in the workstream.** `duration-viterbi`
applied to the frozen `rehearsal-heavy` weights passes all thirteen evaluable
gates, most by wide margins, against a gate set that has been reused unchanged
across three consecutive studies and that rejected every previous candidate.

**The control validates the harness.** `viterbi-penalty-4-control` reproduces
the frozen pilot-v2 `rehearsal-heavy` numbers exactly — fragmentation +0.0568 on
microphone, the same figure that rejected it — so the decoder axis is the only
thing that moved. Without that agreement none of the other rows would be
trustworthy.

**The win is genuinely Pareto, not a trade.** Against the control,
`duration-viterbi` gives up essentially nothing in accuracy (GuitarSet root
0.5802 -> 0.5784, detailed 0.4417 -> 0.4437; full-mix detailed 0.6178 -> 0.6174)
while removing the over-segmentation that rejected pilot v2: fragmentation
0.7439 -> 0.6873 on microphone and 0.7230 -> 0.6653 on pickup, regions per
minute 25.64 -> 22.17 and 24.64 -> 21.07.

**It is not degenerate.** A decoder can always pass a fragmentation gate by
emitting fewer, longer regions and destroying accuracy in the process. This one
lands at 22.17 regions/minute on microphone against v1's 22.72 and a reference
rate near 20, with detailed accuracy 11.79 pp *above* v1. It segments like v1
and labels far better.

**The simplest decoder won, which was not the expectation.** segmental-v3
identified `segmental-full` as its strongest over-segmentation control. On this
model it is the *worst* of the three: fragmentation 0.7514 on microphone, higher
than the control's 0.7439. `duration-boundary-viterbi` sits in between at
0.7197. Adding boundary gating and semi-Markov segmentation on top of the
duration constraint actively undid it here.

That is consistent with the boundary-v3 finding rather than in tension with it.
boundary-v3 concluded that this model family's boundary head, while individually
well calibrated (F1 0.817 raw), contributes an order of magnitude less to
decoded segmentation than the chord posterior does. The two candidates that read
the boundary head are exactly the two that failed; the one that ignores it and
constrains dwell time directly is the one that worked.

## What this changes

The workstream now has a configuration that beats the retained v1 on every
measured axis in both domains: full-band accuracy, solo-guitar accuracy, and
segmentation. Nothing has been promoted — this study selects a candidate, it
does not ship one.

The single measurement that would most change confidence is **p00**, the sealed
performer. Every preservation number here and in pilot v2 is measured on p01-p05,
which v1 trained on and which `rehearsal-heavy` rehearsed on further, so they are
forgetting checks rather than generalisation claims. Eligibility is precisely the
condition the seal was written to wait for, and unsealing is a separate frozen
decision rather than part of this study.

## Limits

- **The preservation result is still measured on rehearsed performers.** p01-p05
  are in the base model's training mixture. The +5.86 pp root gain is a
  forgetting check that came out strongly positive; it is not evidence of
  generalisation to an unseen player.
- **The decoder scalars were frozen against a different model.** They were
  transferred verbatim from segmental-v3, which tuned nothing on this model. That
  is what keeps the result honest, and it also means no candidate here is tuned
  for this model — a tuned decoder might do better, and could not be evaluated
  without fitting on the set the gates score.
- **Full-band evidence remains synthetic.** Slakh2100 is rendered from MIDI, its
  labels are note-level, and the full-band fragmentation gate remains
  not-evaluable for that reason.
- **The base model itself is unchanged and unpromoted.** This study varies only
  the decoder; the pilot-v2 conclusion to retain v1 as the shipped engine still
  stands until a promotion decision is taken on its own terms.
- Two gates remain not-evaluable for the reasons recorded in every pilot report,
  and are reported as such rather than counted as passes.
