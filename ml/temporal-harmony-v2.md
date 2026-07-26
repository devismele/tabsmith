# Temporal Harmony v2 development protocol

## Scope

Temporal-harmony-v2 targets one observed failure mode: the learned model has
useful chord identity accuracy but changes chord state too often. This milestone
does not change application code, bundled weights, hybrid settings, release
gating, or the production rule engine.

The model retains the four-head export contract:

- root logits
- quality logits
- no-chord logits
- boundary/change logits

Chord confidence is derived from the complete root, quality, and no-chord
distribution. The boundary head answers the separate question “should the chord
change now?” No hard label is smoothed before decoding.

## Frozen v1 starting point

Run:

```powershell
npm run baseline:temporal-v2
.\.venv\Scripts\python.exe -m ml.evaluation.temporal_v2_baseline --require-local-artifacts
npx vite-node evaluation/verify-temporal-v2-parity.ts --weights src/learnedHarmony/model/app-chord-model.weights.json
```

The machine-readable source is
`ml/baselines/temporal-harmony-v2-v1-baseline.json`. It reconciles the training
history, checkpoint/ONNX hashes, application export identity, TypeScript parity
sample, and the completed p00 reports.

Historical p00 ML-only validation:

| Capture | Root | Major/minor | Detailed | Fragmentation | Regions/min | Boundary MAE |
|---|---:|---:|---:|---:|---:|---:|
| Microphone | 55.29% | 48.04% | 44.87% | 0.6319 | 19.92 | 1055 ms |
| Pickup mix | 54.61% | 47.11% | 43.96% | 0.6375 | 19.40 | 978 ms |

p00 was already used for v1 validation and ML-only transition-penalty selection.
It is frozen historical evidence, not a v2 model-selection set.

## Temporal objectives

The v2 config extends the receptive field from 125 to 509 frames at the
application feature hop. It adds four opt-in sequence terms:

1. Within-chord consistency penalizes changes in the complete 37-state
   distribution only where the reference chord persists.
2. Switch supervision compares differentiable adjacent-state change probability
   with an exact frame-aligned change target.
3. Hard boundary supervision complements the existing tolerant boundary window.
4. Boundary/switch agreement encourages the identity and change heads to tell a
   coherent story.

All new loss weights default to zero when absent, so old v1 configs retain the
old objective.

Probability filters in `ml/evaluation/probability_smoothing.py` operate on full
distributions before Viterbi. The bounded study includes an unsmoothed full
objective, so smoothing cannot conceal whether the training objective itself
helped.

## Reproducible augmentation

`ml/configs/temporal-harmony-v2-augmentations.json` defines a fixed seed and
weighted recipes for compression, reverb, distortion, bass leakage,
piano/synth-like contamination, drum/vocal leakage, separation artifacts, and
full mixtures. Four deterministic variants plus the clean input are selected
per track.

These are feature-domain proxies and write no generated audio. Real microphone
and pickup-mix recordings are separate paired captures. The no-harmonic negative
contains unpitched interference only; valid piano- or synth-led harmony is never
relabeled as no-chord. A later legally sourced stem manifest may replace proxy
contaminants while keeping recipe and seed identities stable.

## Frozen grouped development evaluation

`ml/configs/temporal-harmony-v2-splits.json` freezes leave-one-performer-out
folds across p01–p05. Both captures of every performance remain in the same
fold. p00 is rejected if it appears in model selection.

All p01–p05 performers overlap v1 model development:

- p01, p02, p03, p05: v1 training
- p04: v1 development

The grouped study is therefore integration-domain development, not independent
generalization evidence. The final gate requires a still-unavailable legally
usable, artist-disjoint, raw-audio full-band set.

Validate local pairing without extracting features:

```powershell
python -m ml.evaluation.temporal_v2_ablation --validate-data `
  --annotations $env:TABSMITH_GUITARSET_ANNOTATIONS `
  --mic-audio $env:TABSMITH_GUITARSET_MIC_AUDIO `
  --pickup-audio $env:TABSMITH_GUITARSET_PICKUP_AUDIO
```

Print the frozen study plan:

```powershell
npm run ablate:temporal-v2
```

Execute the fixed study:

```powershell
.\.venv\Scripts\python.exe -m ml.evaluation.temporal_v2_ablation --run `
  --annotations $env:TABSMITH_GUITARSET_ANNOTATIONS `
  --mic-audio $env:TABSMITH_GUITARSET_MIC_AUDIO `
  --pickup-audio $env:TABSMITH_GUITARSET_PICKUP_AUDIO
```

The full plan is six causal candidates across five folds (30 training runs):

1. v1 objective
2. longer context
3. duration objective
4. boundary/switch objective
5. full objective without smoothing
6. full v2 with augmentation and probability EMA

This is deliberately not a parameter search. `--epochs` is available for smoke
testing, but any such report is marked invalid for model selection.

Each trained candidate must pass PyTorch/ONNX parity and the actual TypeScript
TCN forward path. Reports contain portable IDs and checksums, never local paths.
No candidate is selected automatically.

## Replacement gates

A v2 candidate may replace v1 only when both captures show:

- root and detailed accuracy preserved or improved
- fragmentation reduced by at least 8% relative
- regions per minute reduced by at least 5% relative
- boundary MAE worsened by no more than 2%
- Python/ONNX and Python/TypeScript parity
- improvement in the real hybrid with the existing safeguards unchanged

Passing grouped development metrics is not production promotion. The external
full-band raw-audio gate remains open, and the feature remains experimental.
