# Hybrid harmony cross-capture comparison

## Executive summary

The paired validation evidence does not establish a replicated accuracy/stability win across both captures. Keep the engine experimental.

This is held-out **validation** evidence on 60 paired
GuitarSet performances. Microphone and pickup recordings are alternate captures
and are never combined as independent tracks.

## Microphone capture

| Engine | Root | Maj/min | Detailed | N F1 | Fragmentation | Regions/min | Boundary error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rule v3 | 36.8% | 33.2% | 31.7% | 0.0% | 54.6% | 18.15 | 1319.1 ms |
| ML-only | 55.3% | 48.0% | 44.9% | 0.0% | 63.2% | 19.92 | 1054.7 ms |
| New hybrid | 44.1% | 40.0% | 38.8% | 0.0% | 57.6% | 18.87 | 1193.6 ms |
| Legacy region hybrid | 40.8% | 36.8% | 35.6% | 0.0% | 54.6% | 18.15 | 1319.1 ms |

Fallbacks: 0.

## Pickup-mix capture

| Engine | Root | Maj/min | Detailed | N F1 | Fragmentation | Regions/min | Boundary error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rule v3 | 39.0% | 34.9% | 33.6% | 0.0% | 54.3% | 18.25 | 1645.2 ms |
| ML-only | 54.6% | 47.1% | 44.0% | 0.0% | 63.7% | 19.40 | 977.7 ms |
| New hybrid | 43.5% | 39.3% | 37.7% | 0.0% | 56.1% | 18.61 | 1503.7 ms |
| Legacy region hybrid | 42.4% | 38.4% | 37.3% | 0.0% | 54.3% | 18.25 | 1645.2 ms |

Fallbacks: 0.

## Decision questions

- Hybrid beats rule on microphone with a root-accuracy CI above zero: **yes**
- Hybrid beats ML-only on microphone with a root-accuracy CI above zero: **no**
- Hybrid beats legacy region fusion on microphone: **yes**
- Rule-relative accuracy improvement reproduces on pickup mix: **yes**
- Fragmentation improves significantly on both captures: **no**
- Boundary timing improves significantly on both captures: **no**
- Hybrid fallbacks across both runs: **0**

## Recommendation

**keep-experimental**

No production-promotion conclusion is possible from validation-only,
solo-acoustic-guitar evidence without representative full-band raw audio.
