# Segmental-v3 selection decision

**Decision: retain v1. No segmental candidate is eligible. p00 remains sealed.**

Candidate set `segmental-harmony-v3-frozen-20260727`. Evaluated decoder-only on the
frozen full-v2 inference cache across p01–p05 leave-one-performer-out (600 paired
captures). Full-v2 was NOT retrained. Gates were frozen before results were inspected
and were not altered afterwards.

## Pipeline validation

The faithful `full-v2-existing` candidate (EMA smoothing + penalty-4 Viterbi on the
cached responses) reproduces the frozen temporal-v2 full-v2 aggregate exactly:

| capture | root | detailed | frag | rpm | MAE ms |
|---|---|---|---|---|---|
| mic | 0.7565 | 0.7081 | 0.6719 | 22.207 | 429.6 |
| pickup | 0.7543 | 0.7051 | 0.6789 | 22.417 | 410.7 |

This confirms the cache, smoothing, decode and metric path are correct end-to-end.

## Frozen gates (both captures)

root ≥ 0.72; detailed ≥ 0.67; fragmentation ≤ 0.635; regions/min ≤ 20.75;
boundary MAE ≤ 650 ms; A→B→A flicker ≤ full-v2; boundary precision not materially
below full-v2; boundary recall not collapsed.

## Result: 0 eligible candidates

The structural decoders reduce over-segmentation and slightly improve recognition and
boundary timing, but the best candidates miss the fragmentation gate (≤ 0.635) and the
pickup regions/min gate (≤ 20.75) by a small margin.

| candidate | cap | root | detailed | frag | rpm | MAE | flicker | first failed gate |
|---|---|---|---|---|---|---|---|---|
| segmental-full | mic | 0.7583 | 0.7133 | 0.6486 | 20.783 | 479.0 | 4 | frag 0.6486 > 0.635; rpm 20.783 > 20.75 |
| segmental-full | pickup | 0.7571 | 0.7089 | 0.6481 | 20.927 | 475.7 | 5 | frag 0.6481 > 0.635; rpm 20.927 > 20.75 |
| semi-markov-chord | mic | 0.7491 | 0.7026 | 0.6417 | 20.638 | 519.3 | 5 | frag 0.6417 > 0.635 |
| semi-markov-chord | pickup | 0.7455 | 0.6967 | 0.6417 | 20.756 | 524.9 | 7 | frag 0.6417 > 0.635; rpm 20.756 > 20.75 |

`segmental-full` is the strongest on the over-segmentation objective: relative to full-v2
it cuts fragmentation 0.6719→0.6486 (mic) / 0.6789→0.6481 (pickup), regions/min
22.207→20.783 / 22.417→20.927, and A→B→A flicker 5→4 / 6→5, while *improving* detailed
accuracy (0.7081→0.7133 / 0.7051→0.7089) and keeping boundary MAE far under the 650 ms
gate. It nonetheless does not clear the frozen fragmentation gate on either capture, and
also misses the regions/min gate on both captures. `semi-markov-chord` reaches the lowest
fragmentation (0.6417) but still exceeds the 0.635 gate and trades away boundary timing.

## Consequence

- Retain v1 as the experimental decoder. Do not replace it.
- p00 was NOT accessed (no eligible candidate; the protocol forbids opening p00).
- No TypeScript decoder was promoted (nothing selected for the application path).
- Pareto frontier (paired root ↑, regions/min ↓, fragmentation ↓): semi-markov-chord,
  segmental-full, semi-markov-chord-boundary, duration-boundary-viterbi,
  boundary-gated-viterbi.

## Interpretation

Decoder-only segmental decoding is a genuine, cheap improvement on the over-segmentation
axis (and does not hurt accuracy), but it is not sufficient to satisfy the frozen v1
replacement gates. The remaining fragmentation is dominated by `false_long` spurious
regions (≈77% of excess boundaries) that occur where the boundary head is weak (mean
boundary prob 0.23 before false transitions vs 0.42 before true ones) and the chord
margin is small — evidence that closing the gap likely requires model-side boundary
calibration, not decoding alone. A future frozen protocol may revisit this with a
retrained boundary head.

This is grouped integration-domain development evidence on GuitarSet solo guitar, not
production readiness. Full-band validation remains separately blocked pending licensed
data.
