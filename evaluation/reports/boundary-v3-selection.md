# Boundary-calibration-v3 selection decision

**Decision: retain v1. No boundary candidate is eligible. p00 remains sealed.**

Candidate set `boundary-calibration-v3-frozen-20260727` under gate set `boundary-calibration-v3-gates-20260727`.
Gates were frozen before any candidate was trained and were not altered afterwards.

## Gate outcomes

| candidate | eligible | first failure |
|---|---|---|
| calibrated-boundary | False | fragmentationRateMaximum on audio_mono-mic: measured 0.6461 vs required 0.635 |
| precision-boundary-loss | False | fragmentationRateMaximum on audio_mono-mic: measured 0.6358 vs required 0.635 |
| agreement-coupled | False | fragmentationRateMaximum on audio_mono-mic: measured 0.6628 vs required 0.635 |

Eligible candidates: **0**

## Reference points (segmental-full decoder)

| model | capture | root | detailed | frag | rpm | MAE ms |
|---|---|---|---|---|---|---|
| full-v2-control | audio_mono-mic | 0.7583 | 0.7133 | 0.6486 | 20.783 | 479.0 |
| full-v2-control | audio_mono-pickup_mix | 0.7571 | 0.7089 | 0.6481 | 20.927 | 475.7 |
| calibrated-boundary | audio_mono-mic | 0.7581 | 0.7129 | 0.6461 | 20.743 | 480.8 |
| calibrated-boundary | audio_mono-pickup_mix | 0.7560 | 0.7080 | 0.6481 | 20.920 | 474.9 |
| precision-boundary-loss | audio_mono-mic | 0.7533 | 0.7006 | 0.6358 | 20.468 | 592.7 |
| precision-boundary-loss | audio_mono-pickup_mix | 0.7528 | 0.6983 | 0.6306 | 20.402 | 631.2 |
| agreement-coupled | audio_mono-mic | 0.8090 | 0.7636 | 0.6628 | 21.551 | 381.3 |
| agreement-coupled | audio_mono-pickup_mix | 0.8054 | 0.7609 | 0.6639 | 21.459 | 396.1 |

## Consequence

- Retain v1. Do not replace it.
- p00 was NOT accessed; the protocol forbids opening it without an eligible candidate.
- No replacement model exported; no production weight, application default or
  release gate changed.

## Checksums

- candidate manifest: `ea196afd25fa9875b310cc6c2f8b1c7582335023c2ad2474ae198b579185901c`
- gate set: `b91b2d88dbd0a07c26f51d26cabb16e025dd2b0538a9f23825d651b2dfc6212e`
- evaluation results: `44dbf696457736adee5e5995943279062571f377eaff641b5d9fb9f61c0c7017`

This is grouped integration-domain development evidence on GuitarSet solo
guitar, not production readiness.
