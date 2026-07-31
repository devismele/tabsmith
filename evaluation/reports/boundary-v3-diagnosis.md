# Boundary-calibration-v3 diagnosis (full-v2, p01-p05)

Model-side boundary behaviour measured on the frozen segmental-v3 inference cache.
Decoder held fixed at the faithful full-v2 path (EMA smoothing + penalty-4 Viterbi),
so every number below describes the model, not a decoder variant. p00 never loaded.

## Baseline reconciliation

- cache entries: 600 across folds lopo-p01, lopo-p02, lopo-p03, lopo-p04, lopo-p05
- performers: guitarset-p01, guitarset-p02, guitarset-p03, guitarset-p04, guitarset-p05 (expected match: True)
- p00 sealed: True (by performer id: 0, by performance prefix: 0)
- mic/pickup paired for all 300 performances: True
- pairing held within fold: True
- fold checkpoints match cached checksums: True
- cached vs recomputed inference: 6 sampled, max |delta| 0.00e+00, within tolerance: True
- **reconciled: True**

## audio_mono-mic

### Calibration

- frames: 36740; boundary base rate (±0.25 s): 0.1796
- mean predicted boundary probability: 0.1654
- expected calibration error: 0.0520
- maximum calibration error: 0.4044

| bin | frames | mean predicted | observed | gap |
|---|---|---|---|---|
| 0.0-0.1 | 18015 | 0.0325 | 0.0485 | -0.0160 |
| 0.1-0.2 | 6058 | 0.1452 | 0.0852 | +0.0600 |
| 0.2-0.3 | 4501 | 0.2450 | 0.1880 | +0.0570 |
| 0.3-0.4 | 4028 | 0.3448 | 0.4615 | -0.1168 |
| 0.4-0.5 | 2052 | 0.4498 | 0.5673 | -0.1175 |
| 0.5-0.6 | 1519 | 0.5407 | 0.6847 | -0.1440 |
| 0.6-0.7 | 472 | 0.6411 | 0.5572 | +0.0839 |
| 0.7-0.8 | 81 | 0.7365 | 0.3951 | +0.3415 |
| 0.8-0.9 | 14 | 0.8330 | 0.4286 | +0.4044 |

### Boundary threshold sweep (peak-picked, ±0.25 s)

| threshold | predicted | precision | recall | F1 |
|---|---|---|---|---|
| 0.05 | 4079 | 0.6830 | 0.8442 | 0.7551 |
| 0.10 | 3915 | 0.7027 | 0.8336 | 0.7626 |
| 0.15 | 3780 | 0.7177 | 0.8221 | 0.7664 |
| 0.20 | 3625 | 0.7346 | 0.8070 | 0.7691 |
| 0.25 | 3454 | 0.7525 | 0.7876 | 0.7696 |
| 0.30 | 3249 | 0.7673 | 0.7555 | 0.7613 |
| 0.35 | 2759 | 0.7677 | 0.6418 | 0.6991 |
| 0.40 | 2279 | 0.7652 | 0.5285 | 0.6252 |
| 0.45 | 1937 | 0.7682 | 0.4509 | 0.5683 |
| 0.50 | 1369 | 0.7436 | 0.3085 | 0.4361 |
| 0.55 | 732 | 0.6366 | 0.1412 | 0.2312 |
| 0.60 | 398 | 0.5653 | 0.0682 | 0.1217 |
| 0.70 | 60 | 0.4333 | 0.0079 | 0.0155 |
| 0.80 | 5 | 0.6000 | 0.0009 | 0.0018 |

- best F1 threshold: 0.25

### Decoded state switches

- state-switch P/R/F1: 0.6880 / 0.6427 / 0.6646
- state-change/boundary agreement at peak threshold 0.50: 0.2692
- agreement by peak threshold: 0.05=0.713, 0.10=0.704, 0.15=0.695, 0.20=0.684, 0.25=0.670, 0.30=0.649, 0.35=0.550, 0.40=0.455, 0.45=0.393, 0.50=0.269, 0.55=0.121, 0.60=0.057, 0.70=0.005, 0.80=0.000
- taxonomy: {"false_aba_flicker": 116, "false_long": 742, "false_short_multi_window": 104, "missed_boundary": 1179, "true_boundary": 2121}
- bass-root agreement: 0.2580

### Evidence by category (boundary probability / chord margin)

| category | n | mean boundary prob | median | mean margin |
|---|---|---|---|---|
| trueTransition | 2121 | 0.4236 | 0.4463 | 0.2184 |
| falseTransition | 962 | 0.2342 | 0.2086 | 0.1804 |
| missedTransition | 1179 | 0.3970 | 0.4230 | 0.3321 |
| sustainedNonBoundary | 30140 | 0.1266 | 0.0774 | 0.6178 |

### Region shape and timing

- duration before FALSE switch: mean 2.688 s, median 2.250 s
- duration of FALSE proposed region: mean 2.521 s, median 2.250 s
- duration before TRUE switch: mean 2.563 s
- duration of TRUE proposed region: mean 2.603 s
- boundary timing: mean signed +7.3 ms, median signed +6.1 ms, MAE 82.7 ms
- early: 1045, late: 1076

### Per performer

| performer | true | false | missed | predicted regions | reference regions | mean prob true | mean prob false |
|---|---|---|---|---|---|---|---|
| guitarset-p01 | 344 | 236 | 316 | 640 | 720 | 0.4122 | 0.2587 |
| guitarset-p02 | 468 | 167 | 192 | 695 | 720 | 0.4246 | 0.2029 |
| guitarset-p03 | 480 | 149 | 180 | 689 | 720 | 0.4203 | 0.2122 |
| guitarset-p04 | 473 | 172 | 187 | 705 | 720 | 0.4271 | 0.2146 |
| guitarset-p05 | 356 | 238 | 304 | 654 | 720 | 0.4333 | 0.2596 |

### Dominant issue

- findings: poor thresholding, poor boundary/chord-head agreement, recall starvation at the operating threshold
- confidence skew (mean predicted − base rate): -0.0143
- true/false separation: 0.1894
- F1 at 0.50: 0.4361; best F1 0.7696 at 0.25

## audio_mono-pickup_mix

### Calibration

- frames: 36740; boundary base rate (±0.25 s): 0.1796
- mean predicted boundary probability: 0.1667
- expected calibration error: 0.0503
- maximum calibration error: 0.3587

| bin | frames | mean predicted | observed | gap |
|---|---|---|---|---|
| 0.0-0.1 | 17878 | 0.0326 | 0.0479 | -0.0153 |
| 0.1-0.2 | 6133 | 0.1455 | 0.0841 | +0.0613 |
| 0.2-0.3 | 4477 | 0.2454 | 0.1932 | +0.0522 |
| 0.3-0.4 | 4017 | 0.3448 | 0.4528 | -0.1080 |
| 0.4-0.5 | 2073 | 0.4483 | 0.5610 | -0.1128 |
| 0.5-0.6 | 1577 | 0.5398 | 0.6798 | -0.1400 |
| 0.6-0.7 | 481 | 0.6400 | 0.5405 | +0.0995 |
| 0.7-0.8 | 89 | 0.7342 | 0.4607 | +0.2735 |
| 0.8-0.9 | 15 | 0.8253 | 0.4667 | +0.3587 |

### Boundary threshold sweep (peak-picked, ±0.25 s)

| threshold | predicted | precision | recall | F1 |
|---|---|---|---|---|
| 0.05 | 4069 | 0.6815 | 0.8403 | 0.7526 |
| 0.10 | 3923 | 0.6972 | 0.8288 | 0.7573 |
| 0.15 | 3789 | 0.7134 | 0.8191 | 0.7626 |
| 0.20 | 3630 | 0.7331 | 0.8064 | 0.7680 |
| 0.25 | 3468 | 0.7503 | 0.7885 | 0.7689 |
| 0.30 | 3227 | 0.7688 | 0.7518 | 0.7602 |
| 0.35 | 2791 | 0.7668 | 0.6485 | 0.7027 |
| 0.40 | 2310 | 0.7623 | 0.5336 | 0.6278 |
| 0.45 | 1937 | 0.7713 | 0.4527 | 0.5706 |
| 0.50 | 1407 | 0.7413 | 0.3161 | 0.4432 |
| 0.55 | 738 | 0.6369 | 0.1424 | 0.2328 |
| 0.60 | 401 | 0.5686 | 0.0691 | 0.1232 |
| 0.70 | 65 | 0.5077 | 0.0100 | 0.0196 |
| 0.80 | 8 | 0.6250 | 0.0015 | 0.0030 |

- best F1 threshold: 0.25

### Decoded state switches

- state-switch P/R/F1: 0.6835 / 0.6452 / 0.6638
- state-change/boundary agreement at peak threshold 0.50: 0.2607
- agreement by peak threshold: 0.05=0.696, 0.10=0.690, 0.15=0.680, 0.20=0.671, 0.25=0.658, 0.30=0.629, 0.35=0.540, 0.40=0.443, 0.45=0.378, 0.50=0.261, 0.55=0.114, 0.60=0.054, 0.70=0.006, 0.80=0.000
- taxonomy: {"false_aba_flicker": 129, "false_long": 760, "false_short_multi_window": 97, "missed_boundary": 1171, "true_boundary": 2129}
- bass-root agreement: 0.2994

### Evidence by category (boundary probability / chord margin)

| category | n | mean boundary prob | median | mean margin |
|---|---|---|---|---|
| trueTransition | 2129 | 0.4220 | 0.4378 | 0.2128 |
| falseTransition | 986 | 0.2298 | 0.2061 | 0.1932 |
| missedTransition | 1171 | 0.4035 | 0.4226 | 0.3318 |
| sustainedNonBoundary | 30140 | 0.1279 | 0.0797 | 0.6235 |

### Region shape and timing

- duration before FALSE switch: mean 2.686 s, median 2.250 s
- duration of FALSE proposed region: mean 2.503 s, median 2.250 s
- duration before TRUE switch: mean 2.537 s
- duration of TRUE proposed region: mean 2.564 s
- boundary timing: mean signed +4.7 ms, median signed -5.6 ms, MAE 84.9 ms
- early: 1070, late: 1059

### Per performer

| performer | true | false | missed | predicted regions | reference regions | mean prob true | mean prob false |
|---|---|---|---|---|---|---|---|
| guitarset-p01 | 378 | 221 | 282 | 659 | 720 | 0.4133 | 0.2581 |
| guitarset-p02 | 446 | 183 | 214 | 689 | 720 | 0.4254 | 0.2053 |
| guitarset-p03 | 483 | 165 | 177 | 708 | 720 | 0.4156 | 0.1968 |
| guitarset-p04 | 471 | 183 | 189 | 714 | 720 | 0.4272 | 0.2136 |
| guitarset-p05 | 351 | 234 | 309 | 645 | 720 | 0.4285 | 0.2580 |

### Dominant issue

- findings: poor thresholding, poor boundary/chord-head agreement, recall starvation at the operating threshold
- confidence skew (mean predicted − base rate): -0.0129
- true/false separation: 0.1922
- F1 at 0.50: 0.4432; best F1 0.7689 at 0.25

## Smoothed vs raw boundary channel

The full-v2 decode applies EMA smoothing (boundaryAlpha 0.35) to the boundary
channel. Everything above is measured through that smoothing, which is the
configuration full-v2 actually runs. Measuring the raw head output separates a
weak head from a head degraded downstream:

| capture | channel | best F1 | at threshold | F1 at 0.50 | ECE | agreement at best |
|---|---|---|---|---|---|---|
| audio_mono-mic | smoothed | 0.7696 | 0.25 | 0.4361 | 0.0520 | 0.6705 |
| audio_mono-mic | raw | 0.8170 | 0.50 | 0.8170 | 0.0370 | 0.6508 |
| audio_mono-pickup_mix | smoothed | 0.7689 | 0.25 | 0.4432 | 0.0503 | 0.6581 |
| audio_mono-pickup_mix | raw | 0.8194 | 0.50 | 0.8194 | 0.0363 | 0.6456 |

Findings on the raw channel: audio_mono-mic: none triggered; audio_mono-pickup_mix: none triggered

## Microphone/pickup consistency

- shared performances: 300
- switch sites: 4007
- capture switch agreement: 0.5468
