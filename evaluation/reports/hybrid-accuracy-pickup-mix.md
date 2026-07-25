# Hybrid harmony accuracy comparison

## Executive summary

Positive guitar-domain evidence, but no representative full-band raw-audio result exists; the production gate remains open.

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

- guitarset-zenodo-1492449, split `validation`: 60 tracks, 30.5 minutes, source type `solo-guitar`, capture `audio_mono-pickup_mix`, official distribution `zenodo-3371780` version `1.1.0`, annotation MD5 `b39b78e63d3446f2e54ddb7a54df9b10`, audio MD5 `aecce79f425a44e2055e46f680e10f6a`, split definition `artist-hash seed 20260723`, manifest SHA-256 `a4a9c79549e362cd8e75adf91310d8237d22e1bfb098803938d68dcbd567e952`. Leakage audit: All configured held-out performers are excluded from training by the recorded artist-level split. The deterministic artist-hash split has training, development, and validation partitions; it does not create an untouched test partition. Capture audio_mono-pickup_mix is one alternate recording of each performance and is not treated as an independent track.

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
| Rule v3 | 39.0% | 34.9% | 33.6% | 0.0% | 54.3% | 1645.2 ms |
| ML-only | 54.6% | 47.1% | 44.0% | 0.0% | 63.7% | 977.7 ms |
| New hybrid | 43.5% | 39.3% | 37.7% | 0.0% | 56.1% | 1503.7 ms |
| Legacy region hybrid | 42.4% | 38.4% | 37.3% | 0.0% | 54.3% | 1645.2 ms |

### solo-guitar

| Engine | Root | Maj/min | Detailed | N F1 | Fragmentation | Boundary error |
|---|---:|---:|---:|---:|---:|---:|
| Rule v3 | 39.0% | 34.9% | 33.6% | 0.0% | 54.3% | 1645.2 ms |
| ML-only | 54.6% | 47.1% | 44.0% | 0.0% | 63.7% | 977.7 ms |
| New hybrid | 43.5% | 39.3% | 37.7% | 0.0% | 56.1% | 1503.7 ms |
| Legacy region hybrid | 42.4% | 38.4% | 37.3% | 0.0% | 54.3% | 1645.2 ms |

## Paired differences and confidence intervals

Positive accuracy/N-F1 differences favor hybrid. Negative fragmentation and
boundary-error differences favor hybrid.

| Comparison | Metric | Mean difference | Bootstrap 95% CI | Improved / tied / worsened |
|---|---|---:|---:|---:|
| hybrid-minus-rule | rootAccuracy | 0.0454 | [0.0270, 0.0656] | 27 / 32 / 1 |
| hybrid-minus-rule | majorMinorAccuracy | 0.0434 | [0.0264, 0.0633] | 27 / 32 / 1 |
| hybrid-minus-rule | detailedAccuracy | 0.0397 | [0.0234, 0.0597] | 26 / 31 / 3 |
| hybrid-minus-rule | noChordF1 | 0.0000 | [0.0000, 0.0000] | 0 / 60 / 0 |
| hybrid-minus-rule | fragmentationRate | 0.0195 | [-0.0001, 0.0420] | 5 / 46 / 9 |
| hybrid-minus-rule | meanAbsoluteBoundaryErrorMs | -144.0967 | [-319.0960, -8.9060] | 15 / 30 / 13 |
| hybrid-minus-ml-only | rootAccuracy | -0.1044 | [-0.1423, -0.0639] | 11 / 0 / 49 |
| hybrid-minus-ml-only | majorMinorAccuracy | -0.0751 | [-0.1113, -0.0359] | 14 / 0 / 46 |
| hybrid-minus-ml-only | detailedAccuracy | -0.0580 | [-0.1006, -0.0122] | 16 / 3 / 41 |
| hybrid-minus-ml-only | noChordF1 | 0.0000 | [0.0000, 0.0000] | 0 / 60 / 0 |
| hybrid-minus-ml-only | fragmentationRate | -0.0742 | [-0.1342, -0.0189] | 30 / 12 / 18 |
| hybrid-minus-ml-only | meanAbsoluteBoundaryErrorMs | 435.6407 | [66.0937, 882.1403] | 17 / 0 / 41 |
| hybrid-minus-legacy-region-hybrid | rootAccuracy | 0.0105 | [-0.0022, 0.0257] | 14 / 33 / 13 |
| hybrid-minus-legacy-region-hybrid | majorMinorAccuracy | 0.0066 | [-0.0063, 0.0205] | 15 / 32 / 13 |
| hybrid-minus-legacy-region-hybrid | detailedAccuracy | 0.0032 | [-0.0095, 0.0173] | 12 / 32 / 16 |
| hybrid-minus-legacy-region-hybrid | noChordF1 | 0.0000 | [0.0000, 0.0000] | 0 / 60 / 0 |
| hybrid-minus-legacy-region-hybrid | fragmentationRate | 0.0195 | [-0.0002, 0.0417] | 5 / 46 / 9 |
| hybrid-minus-legacy-region-hybrid | meanAbsoluteBoundaryErrorMs | -144.0967 | [-327.4312, -12.4298] | 15 / 30 / 13 |

## Fragmentation and runtime

- Rule v3: 18.25 regions/min, 0.0% one-window regions, 0.0% very-short regions, 0 flicker events, 0.809 runtime seconds/audio minute.
- ML-only: 19.40 regions/min, 0.5% one-window regions, 0.0% very-short regions, 0 flicker events, 0.486 runtime seconds/audio minute.
- New hybrid: 18.61 regions/min, 0.0% one-window regions, 0.0% very-short regions, 0 flicker events, 1.353 runtime seconds/audio minute.
- Legacy region hybrid: 18.25 regions/min, 0.0% one-window regions, 0.0% very-short regions, 0 flicker events, 0.918 runtime seconds/audio minute.

## Ablation results

Frozen settings ablation on guitarset-zenodo-1492449/validation.

| Variant | Root | Maj/min | Fragmentation | Boundary error |
|---|---:|---:|---:|---:|
| rule-only | 39.0% | 34.9% | 54.3% | 1645.2 ms |
| hybrid-chord-probabilities-only | 43.6% | 39.5% | 56.1% | 1503.0 ms |
| hybrid-without-boundary | 43.5% | 39.5% | 56.1% | 1503.7 ms |
| hybrid-without-adaptive-weighting | 50.5% | 46.0% | 60.6% | 1195.8 ms |
| full-hybrid | 43.5% | 39.3% | 56.1% | 1503.7 ms |

## Representative successes and regressions

- **adaptive-weighting-blocks-correct-ml** — `guitarset-00_BN1-129-Eb_comp` 0.000–0.232 s: The learned top candidate matched the annotation, but conservative fusion and temporal decoding retained a different label.
- **hybrid-worsens-rule** — `guitarset-00_BN1-129-Eb_comp` 2.522–2.980 s: The hybrid changed a rule candidate that matched the annotation.
- **rule-protection** — `guitarset-00_BN1-129-Eb_comp` 5.270–5.728 s: Rule/ML disagreement was resolved in favor of the protected rule result.
- **quality-confusion** — `guitarset-00_BN1-129-Eb_comp` 5.270–5.728 s: The learned candidate had the annotated root but a different chord quality.
- **hybrid-corrects-uncertain-rule** — `guitarset-00_BN1-147-Gb_comp` 10.031–10.472 s: Confident learned evidence corrected a low-confidence rule candidate before temporal decoding.
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
- Supports broader beta testing: **yes**
- Supports production promotion: **no**

Positive guitar-domain evidence, but no representative full-band raw-audio result exists; the production gate remains open.
