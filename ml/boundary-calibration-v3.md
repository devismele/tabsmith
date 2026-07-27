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

## Headline finding so far

**The boundary head is not the problem.** The Phase 5 diagnosis initially
measured the boundary channel only *after* full-v2's EMA smoothing
(`boundaryAlpha` 0.35). Measured on the raw head output:

| capture | channel | best F1 | at threshold | F1 @ 0.50 | ECE |
|---|---|---|---|---|---|
| mic | smoothed | 0.7696 | 0.25 | 0.4361 | 0.0520 |
| mic | **raw** | **0.8170** | **0.50** | **0.8170** | 0.0370 |
| pickup | smoothed | 0.7689 | 0.25 | 0.4432 | 0.0503 |
| pickup | **raw** | **0.8194** | **0.50** | **0.8194** | 0.0363 |

On the raw channel the natural 0.50 threshold is already optimal and no
diagnostic finding triggers at all. The apparent "recall starvation", "poor
thresholding" and low head agreement (0.27) are artifacts of the smoothing
stage flattening boundary peaks while peak-picking stayed at 0.50 — not of a
weak or miscalibrated head. `segmental-full` already consumes the raw channel.

Accordingly, post-hoc calibration is nearly inert on decoded output: isotonic
cuts ECE tenfold (0.037 → 0.0035) but moves `segmental-full` fragmentation only
0.6486 → 0.6461 (mic) and 0.6481 → 0.6481 (pickup). The binding constraint in
every configuration measured so far is fragmentation against the 0.635 gate,
which is a chord-posterior/decoder property rather than a boundary one.

## Status

- Phase 4 reconciliation: **complete and passing** (`boundary-v3-diagnosis.md`).
- Phase 5 diagnosis: **complete**, including the raw-vs-smoothed correction.
- Phase 6/7 freeze: **committed before any candidate was trained** (`f9f6197`).
- Phase 8 smoke: **passed** (finite losses, all four heads training, checkpoint
  save/resume, ONNX export, TypeScript parity, resume-identity guard verified
  in both directions).
- Phase 9 evaluation: **implemented and validated** on the two candidates that
  need no training. `full-v2-control` reproduces both frozen baselines exactly;
  `calibrated-boundary` is ineligible (fragmentation 0.6461 > 0.635 on mic,
  regions/minute 20.920 > 20.75 on pickup).
- Retrained candidates (`precision-boundary-loss`, `agreement-coupled`):
  **running**, 10 fold-runs at roughly 80 minutes each on 12 CPU cores.
- Phase 10 freeze decision: **blocked** on those runs. p00 remains sealed.

## Resuming an interrupted run

Training is resumable at (fold, candidate) granularity. Re-running the same
command skips completed runs; a changed candidate config changes the frozen
run identity and the harness refuses to reuse the stale checkpoint rather than
silently continuing onto it.

```powershell
# 1. finish/resume training (safe to re-run at any time)
.\.venv\Scripts\python.exe -m ml.evaluation.boundary.train_candidates `
  --run-dir ml\runs\boundary-calibration-v3

# 2. build each retrained candidate's inference cache (resumable)
.\.venv\Scripts\python.exe -m ml.evaluation.boundary.build_caches `
  --run-dir ml\runs\boundary-calibration-v3

# 3. evaluate all candidates and apply the frozen gates
.\.venv\Scripts\python.exe -m ml.evaluation.boundary.run_study `
  --run-dir ml\runs\boundary-calibration-v3
```

All three need `TABSMITH_GUITARSET_ANNOTATIONS`, `TABSMITH_GUITARSET_MIC_AUDIO`
and `TABSMITH_GUITARSET_PICKUP_AUDIO` set. Step 3 reports a candidate whose
cache is missing rather than silently omitting it.
