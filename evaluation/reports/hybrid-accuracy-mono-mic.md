# Hybrid harmony accuracy comparison

## Executive summary

The guitar-domain result does not establish a clear accuracy/stability win, and no representative full-band raw-audio result exists.

The actual TypeScript hybrid decoder was executed: **yes**.
Learned inference completed through the production temporal decoder on 60 of 60 requested tracks. Fallbacks: 0.

## Methodology

All primary engines used identical raw-audio intervals, annotations, sample rates,
feature frames, and chord normalization. ML-only and hybrid used the same captured
learned response object per successful track. Hybrid called the production
`runHybridHarmony()` path, which aligns the full probability distribution,
performs adaptive fusion, and returns the fused observations to
`decodeHarmonyObservations()`.

ML-only used a 37-state log-emission Viterbi decoder with a uniform switch penalty
of 4. It received no rule,
bass, beat, key, boundary-head, hysteresis, or cleanup evidence and remains
evaluation-only.

Paired bootstrap confidence intervals resample tracks with seed
20260725 for 2000
iterations.

## Dataset and split

- guitarset-zenodo-1492449, split `validation`: 60 tracks, 30.5 minutes, source type `solo-guitar`, capture `audio_mono-mic`, official distribution `zenodo-3371780` version `1.1.0`, annotation MD5 `b39b78e63d3446f2e54ddb7a54df9b10`, audio MD5 `275966d6610ac34999b58426beb119c3`, split definition `artist-hash seed 20260723`, manifest SHA-256 `a4a9c79549e362cd8e75adf91310d8237d22e1bfb098803938d68dcbd567e952`. Leakage audit: All configured held-out performers are excluded from training by the recorded artist-level split. The deterministic artist-hash split has training, development, and validation partitions; it does not create an untouched test partition. Capture audio_mono-mic is one alternate recording of each performance and is not treated as an independent track.

## Engine identities

- Rule-only: `2026-07-harmonic-context-v3-reduced-latency`.
- ML-only and hybrid model: `temporal-baseline-real-v1`,
  checksum `a8719a40942eb8afe700fd7ff003353abbc1444344a0eb4c74e2a35a8b51cb34`,
  feature version `harmony-features-v1`.
- Observation hybrid decoder version:
  `2`.
- Legacy region hybrid: evaluation-only reconstruction from `51392820c7c9c1a144069a3ed1371ef2cfe9f5b9`.

## Aggregate results

| Engine | Root | Maj/min | Detailed | N F1 | Fragmentation | Boundary error |
|---|---:|---:|---:|---:|---:|---:|
| Rule v3 | 36.8% | 33.2% | 31.7% | 0.0% | 54.6% | 1319.1 ms |
| ML-only | 55.3% | 48.0% | 44.9% | 0.0% | 63.2% | 1054.7 ms |
| New hybrid | 44.1% | 40.0% | 38.8% | 0.0% | 57.6% | 1193.6 ms |
| Legacy region hybrid | 40.8% | 36.8% | 35.6% | 0.0% | 54.6% | 1319.1 ms |

### solo-guitar

| Engine | Root | Maj/min | Detailed | N F1 | Fragmentation | Boundary error |
|---|---:|---:|---:|---:|---:|---:|
| Rule v3 | 36.8% | 33.2% | 31.7% | 0.0% | 54.6% | 1319.1 ms |
| ML-only | 55.3% | 48.0% | 44.9% | 0.0% | 63.2% | 1054.7 ms |
| New hybrid | 44.1% | 40.0% | 38.8% | 0.0% | 57.6% | 1193.6 ms |
| Legacy region hybrid | 40.8% | 36.8% | 35.6% | 0.0% | 54.6% | 1319.1 ms |

## Paired differences and confidence intervals

Positive accuracy/N-F1 differences favor hybrid. Negative fragmentation and
boundary-error differences favor hybrid.

| Comparison | Metric | Mean difference | Bootstrap 95% CI | Improved / tied / worsened |
|---|---|---:|---:|---:|
| hybrid-minus-rule | rootAccuracy | 0.0715 | [0.0443, 0.0999] | 29 / 30 / 1 |
| hybrid-minus-rule | majorMinorAccuracy | 0.0686 | [0.0428, 0.0989] | 28 / 29 / 3 |
| hybrid-minus-rule | detailedAccuracy | 0.0693 | [0.0436, 0.0995] | 30 / 29 / 1 |
| hybrid-minus-rule | noChordF1 | 0.0000 | [0.0000, 0.0000] | 0 / 60 / 0 |
| hybrid-minus-rule | fragmentationRate | 0.0231 | [0.0030, 0.0440] | 7 / 36 / 17 |
| hybrid-minus-rule | meanAbsoluteBoundaryErrorMs | -73.3156 | [-163.8823, 24.1799] | 24 / 28 / 8 |
| hybrid-minus-ml-only | rootAccuracy | -0.1201 | [-0.1536, -0.0900] | 9 / 0 / 51 |
| hybrid-minus-ml-only | majorMinorAccuracy | -0.0847 | [-0.1110, -0.0576] | 12 / 0 / 48 |
| hybrid-minus-ml-only | detailedAccuracy | -0.0605 | [-0.0881, -0.0333] | 15 / 2 / 43 |
| hybrid-minus-ml-only | noChordF1 | 0.0000 | [0.0000, 0.0000] | 0 / 60 / 0 |
| hybrid-minus-ml-only | fragmentationRate | -0.0610 | [-0.1113, -0.0118] | 35 / 8 / 17 |
| hybrid-minus-ml-only | meanAbsoluteBoundaryErrorMs | 260.0202 | [-273.4102, 625.0513] | 18 / 0 / 42 |
| hybrid-minus-legacy-region-hybrid | rootAccuracy | 0.0300 | [0.0134, 0.0483] | 22 / 29 / 9 |
| hybrid-minus-legacy-region-hybrid | majorMinorAccuracy | 0.0299 | [0.0158, 0.0462] | 22 / 28 / 10 |
| hybrid-minus-legacy-region-hybrid | detailedAccuracy | 0.0303 | [0.0129, 0.0471] | 21 / 28 / 11 |
| hybrid-minus-legacy-region-hybrid | noChordF1 | 0.0000 | [0.0000, 0.0000] | 0 / 60 / 0 |
| hybrid-minus-legacy-region-hybrid | fragmentationRate | 0.0231 | [0.0020, 0.0430] | 7 / 36 / 17 |
| hybrid-minus-legacy-region-hybrid | meanAbsoluteBoundaryErrorMs | -73.3156 | [-161.3824, 24.8365] | 24 / 28 / 8 |

## Fragmentation and runtime

- Rule v3: 18.15 regions/min, 0.0% one-window regions, 0.0% very-short regions, 0 flicker events, 0.924 runtime seconds/audio minute.
- ML-only: 19.92 regions/min, 0.5% one-window regions, 0.2% very-short regions, 0 flicker events, 0.491 runtime seconds/audio minute.
- New hybrid: 18.87 regions/min, 0.0% one-window regions, 0.0% very-short regions, 0 flicker events, 1.579 runtime seconds/audio minute.
- Legacy region hybrid: 18.15 regions/min, 0.0% one-window regions, 0.0% very-short regions, 0 flicker events, 1.036 runtime seconds/audio minute.

## Ablation results

Frozen settings ablation on guitarset-zenodo-1492449/validation.

| Variant | Root | Maj/min | Fragmentation | Boundary error |
|---|---:|---:|---:|---:|
| rule-only | 36.8% | 33.2% | 54.6% | 1319.1 ms |
| hybrid-chord-probabilities-only | 44.1% | 40.1% | 57.8% | 1190.2 ms |
| hybrid-without-boundary | 44.1% | 40.0% | 57.6% | 1193.6 ms |
| hybrid-without-adaptive-weighting | 50.2% | 45.2% | 59.6% | 1019.1 ms |
| full-hybrid | 44.1% | 40.0% | 57.6% | 1193.6 ms |

## Representative successes and regressions

- **adaptive-weighting-blocks-correct-ml** — `guitarset-00_BN1-129-Eb_comp` 0.000–0.511 s: The learned top candidate matched the annotation, but conservative fusion and temporal decoding retained a different label.
- **hybrid-worsens-rule** — `guitarset-00_BN1-129-Eb_comp` 0.511–1.185 s: The hybrid changed a rule candidate that matched the annotation.
- **quality-confusion** — `guitarset-00_BN1-129-Eb_comp` 5.230–5.904 s: The learned candidate had the annotated root but a different chord quality.
- **hybrid-corrects-uncertain-rule** — `guitarset-00_BN1-129-Eb_comp` 12.646–13.320 s: Confident learned evidence corrected a low-confidence rule candidate before temporal decoding.
- **rule-protection** — `guitarset-00_BN1-129-Eb_comp` 14.668–15.342 s: Rule/ML disagreement was resolved in favor of the protected rule result.
- **incorrect-no-chord** — `guitarset-00_BN3-154-E_comp` 24.752–24.932 s: The learned head favored no-chord, but the hybrid decoder retained a chord.

Missing evidence categories: `hybrid-fixes-ml-flicker`, `learned-boundary-timing`.

## Limitations

- Peak memory is a coarse Node heap high-water sample, not process-wide RSS profiling.
- ML-only uses a uniform transition penalty and intentionally omits the learned boundary head so it remains evidence-independent from Tabsmith's rule decoder.
- GuitarSet performer p00 was previously used to choose the ML-only Viterbi transition penalty in repository experiments; this is held-out validation evidence, not a pristine final test.
- No legally available representative full-band raw-audio dataset was configured, so the production promotion gate remains open.
- Billboard precomputed-feature results are intentionally excluded from this raw-audio app comparison.
- The selected lead-sheet references contain no annotated no-chord intervals; no-chord precision, recall, and F1 are therefore not informative for this run.

## Recommendation

- Keep experimental: **yes**
- Supports broader beta testing: **no**
- Supports production promotion: **no**

The guitar-domain result does not establish a clear accuracy/stability win, and no representative full-band raw-audio result exists.
