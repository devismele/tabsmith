# Boundary-calibration-v3 candidate comparison (p01-p05 development)

Candidate set `boundary-calibration-v3-gates-20260727`. Gates were frozen before any candidate was
trained and were not altered afterwards. p00 was not accessed.

Each candidate is reported under both decoders. The existing full-v2 decode does
not consume the boundary head, so boundary calibration is inert on that path by
construction; it is shown to keep model effects separable from decoder effects.

## Decoder: full-v2-existing

### audio_mono-mic

| candidate | root | maj/min | detailed | frag | rpm | regions | MAE ms | A→B→A | flicker |
|---|---|---|---|---|---|---|---|---|---|
| full-v2-control | 0.7565 | 0.7142 | 0.7081 | 0.6719 | 22.207 | 3383 | 429.6 | 375 | 5 |
| calibrated-boundary | 0.7565 | 0.7142 | 0.7081 | 0.6719 | 22.207 | 3383 | 429.6 | 375 | 5 |
| precision-boundary-loss | 0.7554 | 0.7101 | 0.7010 | 0.6644 | 21.839 | 3327 | 553.9 | 350 | 8 |
| agreement-coupled | 0.8054 | 0.7642 | 0.7587 | 0.6853 | 22.594 | 3442 | 349.5 | 375 | 6 |

Boundary quality (at the fold-selected operating threshold):

| candidate | precision | recall | F1 | agreement | ECE | threshold | false-long | missed |
|---|---|---|---|---|---|---|---|---|
| full-v2-control | 0.7447 | 0.7927 | 0.7679 | 0.6750 | 0.0520 | 0.23 | 742 | 1179 |
| calibrated-boundary | 0.7177 | 0.7564 | 0.7365 | 0.6542 | 0.0041 | 0.15 | 742 | 1179 |
| precision-boundary-loss | 0.4105 | 0.4942 | 0.4485 | 0.4767 | 0.0048 | 0.15 | 837 | 1352 |
| agreement-coupled | 0.7315 | 0.7785 | 0.7543 | 0.6954 | 0.0018 | 0.10 | 580 | 933 |

95% performer-paired bootstrap CIs (root / rpm / frag):

- full-v2-control: root [0.7285, 0.7844] rpm [21.444, 23.009] frag [0.6561, 0.6873]
- calibrated-boundary: root [0.7285, 0.7844] rpm [21.444, 23.009] frag [0.6561, 0.6873]
- precision-boundary-loss: root [0.7256, 0.7846] rpm [21.035, 22.612] frag [0.6467, 0.6813]
- agreement-coupled: root [0.7796, 0.8289] rpm [21.696, 23.435] frag [0.6694, 0.7007]

### audio_mono-pickup_mix

| candidate | root | maj/min | detailed | frag | rpm | regions | MAE ms | A→B→A | flicker |
|---|---|---|---|---|---|---|---|---|---|
| full-v2-control | 0.7543 | 0.7126 | 0.7051 | 0.6789 | 22.417 | 3415 | 410.7 | 375 | 6 |
| calibrated-boundary | 0.7543 | 0.7126 | 0.7051 | 0.6789 | 22.417 | 3415 | 410.7 | 375 | 6 |
| precision-boundary-loss | 0.7497 | 0.7033 | 0.6942 | 0.6667 | 21.912 | 3338 | 559.7 | 364 | 7 |
| agreement-coupled | 0.8017 | 0.7598 | 0.7545 | 0.6808 | 22.529 | 3432 | 373.3 | 389 | 5 |

Boundary quality (at the fold-selected operating threshold):

| candidate | precision | recall | F1 | agreement | ECE | threshold | false-long | missed |
|---|---|---|---|---|---|---|---|---|
| full-v2-control | 0.7420 | 0.7906 | 0.7656 | 0.6620 | 0.0503 | 0.23 | 760 | 1171 |
| calibrated-boundary | 0.7140 | 0.7518 | 0.7324 | 0.6372 | 0.0037 | 0.15 | 760 | 1171 |
| precision-boundary-loss | 0.4010 | 0.4815 | 0.4376 | 0.4839 | 0.0063 | 0.15 | 841 | 1338 |
| agreement-coupled | 0.7278 | 0.7785 | 0.7523 | 0.6932 | 0.0023 | 0.10 | 570 | 937 |

95% performer-paired bootstrap CIs (root / rpm / frag):

- full-v2-control: root [0.7270, 0.7801] rpm [21.572, 23.221] frag [0.6644, 0.6941]
- calibrated-boundary: root [0.7270, 0.7801] rpm [21.572, 23.221] frag [0.6644, 0.6941]
- precision-boundary-loss: root [0.7188, 0.7775] rpm [21.059, 22.704] frag [0.6464, 0.6862]
- agreement-coupled: root [0.7759, 0.8244] rpm [21.693, 23.347] frag [0.6645, 0.6961]

## Decoder: segmental-full

### audio_mono-mic

| candidate | root | maj/min | detailed | frag | rpm | regions | MAE ms | A→B→A | flicker |
|---|---|---|---|---|---|---|---|---|---|
| full-v2-control | 0.7583 | 0.7181 | 0.7133 | 0.6486 | 20.783 | 3166 | 479.0 | 386 | 4 |
| calibrated-boundary | 0.7581 | 0.7179 | 0.7129 | 0.6461 | 20.743 | 3160 | 480.8 | 387 | 4 |
| precision-boundary-loss | 0.7533 | 0.7092 | 0.7006 | 0.6358 | 20.468 | 3118 | 592.7 | 329 | 7 |
| agreement-coupled | 0.8090 | 0.7677 | 0.7636 | 0.6628 | 21.551 | 3283 | 381.3 | 386 | 3 |

Boundary quality (at the fold-selected operating threshold):

| candidate | precision | recall | F1 | agreement | ECE | threshold | false-long | missed |
|---|---|---|---|---|---|---|---|---|
| full-v2-control | 0.7950 | 0.8370 | 0.8155 | 0.7854 | 0.0370 | 0.52 | 444 | 999 |
| calibrated-boundary | 0.7982 | 0.8400 | 0.8185 | 0.7790 | 0.0035 | 0.43 | 437 | 1000 |
| precision-boundary-loss | 0.6471 | 0.6833 | 0.6647 | 0.6725 | 0.0038 | 0.31 | 631 | 1265 |
| agreement-coupled | 0.8337 | 0.8370 | 0.8353 | 0.7841 | 0.0044 | 0.45 | 329 | 774 |

95% performer-paired bootstrap CIs (root / rpm / frag):

- full-v2-control: root [0.7300, 0.7863] rpm [20.044, 21.528] frag [0.6306, 0.6667]
- calibrated-boundary: root [0.7300, 0.7863] rpm [20.014, 21.499] frag [0.6277, 0.6641]
- precision-boundary-loss: root [0.7231, 0.7830] rpm [19.707, 21.235] frag [0.6155, 0.6567]
- agreement-coupled: root [0.7816, 0.8324] rpm [20.714, 22.338] frag [0.6461, 0.6803]

### audio_mono-pickup_mix

| candidate | root | maj/min | detailed | frag | rpm | regions | MAE ms | A→B→A | flicker |
|---|---|---|---|---|---|---|---|---|---|
| full-v2-control | 0.7571 | 0.7155 | 0.7089 | 0.6481 | 20.927 | 3188 | 475.7 | 366 | 5 |
| calibrated-boundary | 0.7560 | 0.7146 | 0.7080 | 0.6481 | 20.920 | 3187 | 474.9 | 363 | 5 |
| precision-boundary-loss | 0.7528 | 0.7069 | 0.6983 | 0.6306 | 20.402 | 3108 | 631.2 | 331 | 3 |
| agreement-coupled | 0.8054 | 0.7657 | 0.7609 | 0.6639 | 21.459 | 3269 | 396.1 | 387 | 6 |

Boundary quality (at the fold-selected operating threshold):

| candidate | precision | recall | F1 | agreement | ECE | threshold | false-long | missed |
|---|---|---|---|---|---|---|---|---|
| full-v2-control | 0.7977 | 0.8400 | 0.8183 | 0.7711 | 0.0363 | 0.52 | 472 | 1003 |
| calibrated-boundary | 0.7995 | 0.8400 | 0.8193 | 0.7645 | 0.0025 | 0.43 | 479 | 1009 |
| precision-boundary-loss | 0.6442 | 0.6945 | 0.6684 | 0.6777 | 0.0033 | 0.31 | 670 | 1303 |
| agreement-coupled | 0.8367 | 0.8339 | 0.8353 | 0.7804 | 0.0037 | 0.45 | 350 | 795 |

95% performer-paired bootstrap CIs (root / rpm / frag):

- full-v2-control: root [0.7282, 0.7829] rpm [20.088, 21.689] frag [0.6303, 0.6657]
- calibrated-boundary: root [0.7272, 0.7822] rpm [20.098, 21.676] frag [0.6304, 0.6662]
- precision-boundary-loss: root [0.7219, 0.7814] rpm [19.579, 21.153] frag [0.6081, 0.6518]
- agreement-coupled: root [0.7780, 0.8288] rpm [20.654, 22.238] frag [0.6464, 0.6809]

## Gate outcomes

Applied to each candidate under the segmental-full decoder (the path that
consumes the boundary head).

| candidate | eligible | first failure |
|---|---|---|
| calibrated-boundary | False | fragmentationRateMaximum on audio_mono-mic: measured 0.6461 vs required 0.635 |
| precision-boundary-loss | False | fragmentationRateMaximum on audio_mono-mic: measured 0.6358 vs required 0.635 |
| agreement-coupled | False | fragmentationRateMaximum on audio_mono-mic: measured 0.6628 vs required 0.635 |

**Eligible candidates: 0** (none)

## Effect attribution: boundary vs decoder vs chord posterior

The full-v2 decode does not read the boundary head. A model effect visible
under that decoder is therefore chord-posterior movement by definition, and the
extra effect that only appears under segmental-full is the boundary channel:

```
decoder effect          = control(segmental)   - control(existing)
chord-posterior channel = candidate(existing)  - control(existing)
boundary channel        = [candidate(segmental) - control(segmental)]
                          - chord-posterior channel
```

Signs are normalised so positive always means better.

### calibrated-boundary

**audio_mono-mic**

| effect | fragmentation | root accuracy |
|---|---|---|
| decoder (control only) | +0.0233 | +0.0018 |
| model under full-v2 decode (chord posterior) | -0.0000 | +0.0000 |
| model under segmental-full | +0.0025 | -0.0001 |
| boundary channel (interaction) | +0.0025 | -0.0001 |

- total fragmentation improvement: +0.0025 (**dominant channel: boundary channel**)
- boundary head F1 delta: +0.0030

**audio_mono-pickup_mix**

| effect | fragmentation | root accuracy |
|---|---|---|
| decoder (control only) | +0.0308 | +0.0028 |
| model under full-v2 decode (chord posterior) | -0.0000 | +0.0000 |
| model under segmental-full | +0.0000 | -0.0011 |
| boundary channel (interaction) | +0.0000 | -0.0011 |

- total fragmentation improvement: +0.0000 (**dominant channel: neither (no material movement)**)
- boundary head F1 delta: +0.0010

Chord posterior compared frame-by-frame against the control (600 captures):

- top-1 argmax agreement with control: 1.0000
- mean posterior margin: control 0.6194 -> candidate 0.6194 (+0.0000)
- mean no-chord probability: control 0.0009 -> candidate 0.0009

### precision-boundary-loss

**audio_mono-mic**

| effect | fragmentation | root accuracy |
|---|---|---|
| decoder (control only) | +0.0233 | +0.0018 |
| model under full-v2 decode (chord posterior) | +0.0075 | -0.0011 |
| model under segmental-full | +0.0128 | -0.0050 |
| boundary channel (interaction) | +0.0053 | -0.0039 |

- total fragmentation improvement: +0.0128 (**dominant channel: chord posterior**)
- boundary head F1 delta: -0.1508

**audio_mono-pickup_mix**

| effect | fragmentation | root accuracy |
|---|---|---|
| decoder (control only) | +0.0308 | +0.0028 |
| model under full-v2 decode (chord posterior) | +0.0122 | -0.0046 |
| model under segmental-full | +0.0175 | -0.0043 |
| boundary channel (interaction) | +0.0053 | +0.0003 |

- total fragmentation improvement: +0.0175 (**dominant channel: chord posterior**)
- boundary head F1 delta: -0.1499

Chord posterior compared frame-by-frame against the control (600 captures):

- top-1 argmax agreement with control: 0.6428
- mean posterior margin: control 0.6194 -> candidate 0.6381 (+0.0187)
- mean no-chord probability: control 0.0009 -> candidate 0.0007

### agreement-coupled

**audio_mono-mic**

| effect | fragmentation | root accuracy |
|---|---|---|
| decoder (control only) | +0.0233 | +0.0018 |
| model under full-v2 decode (chord posterior) | -0.0133 | +0.0489 |
| model under segmental-full | -0.0142 | +0.0507 |
| boundary channel (interaction) | -0.0008 | +0.0018 |

- total fragmentation improvement: -0.0141 (**dominant channel: chord posterior**)
- boundary head F1 delta: +0.0198

**audio_mono-pickup_mix**

| effect | fragmentation | root accuracy |
|---|---|---|
| decoder (control only) | +0.0308 | +0.0028 |
| model under full-v2 decode (chord posterior) | -0.0019 | +0.0474 |
| model under segmental-full | -0.0158 | +0.0483 |
| boundary channel (interaction) | -0.0139 | +0.0010 |

- total fragmentation improvement: -0.0158 (**dominant channel: boundary channel**)
- boundary head F1 delta: +0.0170

Chord posterior compared frame-by-frame against the control (600 captures):

- top-1 argmax agreement with control: 0.6812
- mean posterior margin: control 0.6194 -> candidate 0.6975 (+0.0781)
- mean no-chord probability: control 0.0009 -> candidate 0.0007

## Per-fold calibration fitted on training performers

| candidate | decoder / fold | calibrator | threshold | train ECE before | after |
|---|---|---|---|---|---|
| full-v2-control | full-v2-existing|lopo-p01 | identity | 0.25 | 0.0538 | 0.0538 |
| full-v2-control | full-v2-existing|lopo-p02 | identity | 0.20 | 0.0477 | 0.0477 |
| full-v2-control | full-v2-existing|lopo-p03 | identity | 0.25 | 0.0491 | 0.0491 |
| full-v2-control | full-v2-existing|lopo-p04 | identity | 0.20 | 0.0510 | 0.0510 |
| full-v2-control | full-v2-existing|lopo-p05 | identity | 0.25 | 0.0542 | 0.0542 |
| full-v2-control | segmental-full|lopo-p01 | identity | 0.55 | 0.0349 | 0.0349 |
| full-v2-control | segmental-full|lopo-p02 | identity | 0.50 | 0.0384 | 0.0384 |
| full-v2-control | segmental-full|lopo-p03 | identity | 0.55 | 0.0402 | 0.0402 |
| full-v2-control | segmental-full|lopo-p04 | identity | 0.50 | 0.0361 | 0.0361 |
| full-v2-control | segmental-full|lopo-p05 | identity | 0.50 | 0.0341 | 0.0341 |
| calibrated-boundary | full-v2-existing|lopo-p01 | isotonic | 0.15 | 0.0538 | 0.0000 |
| calibrated-boundary | full-v2-existing|lopo-p02 | isotonic | 0.15 | 0.0477 | 0.0000 |
| calibrated-boundary | full-v2-existing|lopo-p03 | isotonic | 0.15 | 0.0491 | 0.0000 |
| calibrated-boundary | full-v2-existing|lopo-p04 | isotonic | 0.15 | 0.0510 | 0.0000 |
| calibrated-boundary | full-v2-existing|lopo-p05 | isotonic | 0.15 | 0.0542 | 0.0000 |
| calibrated-boundary | segmental-full|lopo-p01 | isotonic | 0.45 | 0.0349 | 0.0000 |
| calibrated-boundary | segmental-full|lopo-p02 | isotonic | 0.45 | 0.0384 | 0.0000 |
| calibrated-boundary | segmental-full|lopo-p03 | isotonic | 0.40 | 0.0402 | 0.0000 |
| calibrated-boundary | segmental-full|lopo-p04 | isotonic | 0.40 | 0.0361 | 0.0000 |
| calibrated-boundary | segmental-full|lopo-p05 | isotonic | 0.45 | 0.0341 | 0.0000 |
| precision-boundary-loss | full-v2-existing|lopo-p01 | isotonic | 0.15 | 0.0366 | 0.0000 |
| precision-boundary-loss | full-v2-existing|lopo-p02 | isotonic | 0.15 | 0.0421 | 0.0000 |
| precision-boundary-loss | full-v2-existing|lopo-p03 | isotonic | 0.15 | 0.0488 | 0.0000 |
| precision-boundary-loss | full-v2-existing|lopo-p04 | isotonic | 0.15 | 0.0536 | 0.0000 |
| precision-boundary-loss | full-v2-existing|lopo-p05 | isotonic | 0.15 | 0.0527 | 0.0000 |
| precision-boundary-loss | segmental-full|lopo-p01 | isotonic | 0.30 | 0.0641 | 0.0000 |
| precision-boundary-loss | segmental-full|lopo-p02 | isotonic | 0.35 | 0.0701 | 0.0000 |
| precision-boundary-loss | segmental-full|lopo-p03 | isotonic | 0.30 | 0.0705 | 0.0000 |
| precision-boundary-loss | segmental-full|lopo-p04 | isotonic | 0.30 | 0.0714 | 0.0000 |
| precision-boundary-loss | segmental-full|lopo-p05 | isotonic | 0.30 | 0.0729 | 0.0000 |
| agreement-coupled | full-v2-existing|lopo-p01 | isotonic | 0.10 | 0.0692 | 0.0000 |
| agreement-coupled | full-v2-existing|lopo-p02 | isotonic | 0.10 | 0.0668 | 0.0000 |
| agreement-coupled | full-v2-existing|lopo-p03 | isotonic | 0.10 | 0.0646 | 0.0000 |
| agreement-coupled | full-v2-existing|lopo-p04 | isotonic | 0.10 | 0.0665 | 0.0000 |
| agreement-coupled | full-v2-existing|lopo-p05 | isotonic | 0.10 | 0.0644 | 0.0000 |
| agreement-coupled | segmental-full|lopo-p01 | isotonic | 0.45 | 0.0559 | 0.0000 |
| agreement-coupled | segmental-full|lopo-p02 | isotonic | 0.45 | 0.0534 | 0.0000 |
| agreement-coupled | segmental-full|lopo-p03 | isotonic | 0.45 | 0.0516 | 0.0000 |
| agreement-coupled | segmental-full|lopo-p04 | isotonic | 0.45 | 0.0513 | 0.0000 |
| agreement-coupled | segmental-full|lopo-p05 | isotonic | 0.45 | 0.0502 | 0.0000 |
