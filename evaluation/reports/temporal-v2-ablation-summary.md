# Temporal harmony v2 grouped ablation

## Executive summary

The frozen 6-candidate x 5-fold study completed all 30 runs with no failures. No v2 candidate is eligible to replace v1: every candidate violated the frozen fragmentation and regions-per-minute gates on both captures.

Decision: **retain v1**. p00 remains sealed and was not evaluated.

## Aggregate results

| Candidate | Capture | Root | Detailed | Fragmentation | Regions/min | Boundary MAE (ms) |
|---|---|---:|---:|---:|---:|---:|
| v1-objective | Microphone | 0.6026 | 0.5284 | 0.6167 | 19.857 | 1027.4 |
| v1-objective | Pickup mix | 0.6082 | 0.5311 | 0.6169 | 19.910 | 1143.6 |
| longer-context | Microphone | 0.6721 | 0.6002 | 0.6944 | 23.691 | 539.4 |
| longer-context | Pickup mix | 0.6737 | 0.6065 | 0.6958 | 23.815 | 520.8 |
| duration-objective | Microphone | 0.6662 | 0.6025 | 0.6900 | 23.599 | 525.8 |
| duration-objective | Pickup mix | 0.6722 | 0.6110 | 0.6917 | 23.638 | 502.5 |
| boundary-switch-objective | Microphone | 0.6622 | 0.5954 | 0.6883 | 23.448 | 537.2 |
| boundary-switch-objective | Pickup mix | 0.6716 | 0.6080 | 0.6939 | 23.566 | 540.8 |
| full-objective-unsmoothed | Microphone | 0.6673 | 0.6025 | 0.6808 | 23.100 | 566.5 |
| full-objective-unsmoothed | Pickup mix | 0.6798 | 0.6160 | 0.6917 | 23.474 | 529.6 |
| full-v2 | Microphone | 0.7565 | 0.7081 | 0.6719 | 22.207 | 429.6 |
| full-v2 | Pickup mix | 0.7543 | 0.7051 | 0.6789 | 22.417 | 410.7 |

## Gate result

Eligible candidates: **0**. Pareto frontier: v1-objective, full-v2.

All 30 checkpoints passed PyTorch/ONNX and actual TypeScript parity at the 1e-04 tolerance.

## Methodological limits

- All p01-p05 performers overlap learned-model development; this is integration-domain development evidence.
- GuitarSet contains solo guitar, not a legally licensed artist-disjoint full-band benchmark.
- The p00 performer was not accessed because no v2 candidate passed the frozen replacement gates.
