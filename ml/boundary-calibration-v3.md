# Boundary-calibration-v3

Experimental research branch. Changes no production weights, no application
defaults, no release gating, and exposes nothing new in release builds.

## Branch relationship

| | |
|---|---|
| branch | `ml/boundary-calibration-v3` |
| base commit | `54f514f` — "Freeze segmental-v3 selection: retain v1, no eligible decoder" |
| base branch | `ml/segmental-harmony-v3` |
| inherited from | `9d22a66` — "Freeze temporal v2 decision to retain v1" (via segmental-v3) |
| does **not** include | `ml/full-band-harmony-v1` (Slakh full-band work is a separate, unmerged track) |

The segmental-v3 study concluded *retain v1*: its best decoder (`segmental-full`)
cut over-segmentation materially but missed the frozen fragmentation gate
(0.6486 / 0.6481 vs ≤ 0.635) and the pickup regions/minute gate (20.927 vs
≤ 20.75). It attributed the residual to model-side boundary behaviour. This
branch tests that attribution and evaluates bounded interventions against it.

## Protocol invariants

- GuitarSet **p01–p05 only**, leave-one-performer-out; microphone and pickup
  captures of a performance always stay in the same fold.
- **p00 stays sealed** until a frozen, eligible selection decision exists. It is
  not in the inference cache and is never loaded by anything in this branch.
- Calibrators, thresholds and duration priors are fitted on *training
  performers only* — never on the held-out fold.
- The exported four-head contract (root / quality / no-chord / boundary) is
  unchanged; boundary calibration is a scalar transform applied downstream of
  the network, not a new head.

## Reproducing

The inference cache lives under the git-ignored `ml/runs/segmental-harmony-v3/cache`
and is reused unchanged from the segmental-v3 study.

```powershell
# Phase 4 reconciliation + Phase 5 diagnosis (cache-only, no training)
.\.venv\Scripts\python.exe -m ml.evaluation.boundary.run_diagnostics
```

The cached-vs-recomputed inference check additionally needs the GuitarSet paths
in `TABSMITH_GUITARSET_ANNOTATIONS`, `TABSMITH_GUITARSET_MIC_AUDIO` and
`TABSMITH_GUITARSET_PICKUP_AUDIO`; without them that one check reports
`skipped` rather than failing the reconciliation.

## Status

Phase 4 (reconciliation) and Phase 5 (diagnosis) complete — see
`evaluation/reports/boundary-v3-diagnosis.md`.
