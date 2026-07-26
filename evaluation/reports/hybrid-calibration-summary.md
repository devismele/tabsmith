# Hybrid calibration summary

Status: **completed-non-p00-calibration**

## Protocol

- Calibration performers: guitarset-p01, guitarset-p02, guitarset-p03, guitarset-p04, guitarset-p05
- Sealed validation performer: guitarset-p00
- Strategy: leave-one-performer-out
- Track-captures: 600
- Candidates: 38
- Learned-model overlap: All non-p00 performers overlap learned-model fitting: p01/p02/p03/p05 training and p04 development. This is integration calibration, not independent model evaluation.

## Baseline gates

| Gate | Windows | % | Before | After | Blocked correct ML | Protected correct rule |
|---|---:|---:|---:|---:|---:|---:|
| availability | 8 | 0.0 | 0.3500 | 0.0000 | 0 | 4 |
| minimum-learned-confidence | 22951 | 67.2 | 0.3500 | 0.0000 | 4501 | 2218 |
| maximum-learned-entropy | 0 | 0.0 | 0.0000 | 0.0000 | 0 | 0 |
| learned-confidence-scaling | 11189 | 32.8 | 0.3500 | 0.2703 | 1881 | 270 |
| learned-entropy-scaling | 11189 | 32.8 | 0.2703 | 0.2388 | 1881 | 270 |
| rule-score-margin | 11189 | 32.8 | 0.2388 | 0.2085 | 1881 | 270 |
| rule-confidence-room | 11189 | 32.8 | 0.2085 | 0.1825 | 1881 | 270 |
| high-rule-confidence-protection | 8 | 0.0 | 0.0742 | 0.0582 | 2 | 0 |
| agreement-bonus | 0 | 0.0 | 0.0000 | 0.0000 | 0 | 0 |
| learned-flicker-suppression | 3550 | 10.4 | 0.1639 | 0.1050 | 816 | 166 |
| source-specific-maximum | 0 | 0.0 | 0.0000 | 0.0000 | 0 | 0 |

## Selection

Selected **current-adaptive** using only non-p00 calibration folds.

| Candidate | Detailed mean | Root mean | Worst fold detailed | Eligible |
|---|---:|---:|---:|---:|
| fixed-current-weight | 46.29% | 55.21% | 40.90% | no |
| random-22 | 42.74% | 51.90% | 35.17% | no |
| random-01 | 42.26% | 51.34% | 34.68% | no |
| random-09 | 41.91% | 51.04% | 33.99% | no |
| random-11 | 41.83% | 50.95% | 34.50% | no |
| adaptive-temperature-calibrated | 38.81% | 48.14% | 32.05% | no |
| adaptive-without-entropy-suppression | 38.53% | 47.98% | 31.80% | no |
| adaptive-without-rule-protection | 38.49% | 47.91% | 31.81% | no |
| chord-evidence-only | 38.48% | 47.87% | 31.80% | no |
| current-adaptive | 38.47% | 47.85% | 31.80% | no |

## Probability calibration

Global fitted temperature: 0.9
Fold-calibrated ECE: 0.0215

## Decision

No candidate satisfied all frozen stability constraints; the current experimental defaults are retained and the Pareto frontier is reported.

p00 was not loaded or scored by this command.

