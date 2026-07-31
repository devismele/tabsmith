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

## Outcome: retain v1 (0 of 3 candidates eligible)

All 10 fold-runs completed (29.4 CPU-hours), 10 distinct checkpoints, every run
valid for selection and clearing PyTorch/ONNX and TypeScript parity. The frozen
gates were applied unchanged; **no candidate passed and p00 stays sealed**.

| candidate | capture | root | detailed | frag | rpm | MAE ms |
|---|---|---|---|---|---|---|
| full-v2-control | mic | 0.7583 | 0.7133 | 0.6486 | 20.783 | 479.0 |
| calibrated-boundary | mic | 0.7581 | 0.7129 | 0.6461 | 20.743 | 480.8 |
| precision-boundary-loss | mic | 0.7533 | 0.7006 | **0.6358** | 20.468 | 592.7 |
| agreement-coupled | mic | **0.8090** | **0.7636** | 0.6628 | 21.551 | **381.3** |

Every candidate failed on fragmentation first. `precision-boundary-loss` missed
by **0.0008** (0.6358 vs 0.635) on microphone while passing on pickup — the gate
was not moved to accommodate it.

## The finding that matters more than the decision

**Both retrained candidates delivered their effect through the chord posterior,
not the boundary head — despite both being designed as boundary interventions.**

The attribution is identifiable because the full-v2 decode never reads the
boundary head, so any model effect visible there is chord-posterior movement:

| candidate | metric | chord-posterior channel | boundary channel |
|---|---|---|---|
| agreement-coupled | root accuracy (mic) | **+0.0489** | +0.0018 |
| agreement-coupled | detailed (mic) | **+0.0506** | −0.0003 |
| agreement-coupled | fragmentation (mic) | **−0.0133** | −0.0008 |
| precision-boundary-loss | fragmentation (mic) | **+0.0075** | +0.0053 |
| calibrated-boundary | fragmentation (mic) | −0.0000 | +0.0025 |

`agreement-coupled` gained **+5 accuracy points** — the largest movement in the
study — and 96% of it came from the chord posterior. Its top-1 chord agreement
with the control fell to 0.681 and its mean posterior margin rose +0.078: the
chord heads changed substantially. The same change cost fragmentation, which is
the trade-off that made it ineligible.

`precision-boundary-loss` bought its stability the same way (chord posterior
+0.0075/+0.0122 vs boundary +0.0053) while **destroying** the boundary head:
F1 0.8155 → 0.6647 and MAE 479 → 593 ms.

`calibrated-boundary` is the clean control: identical checkpoint, top-1 chord
agreement exactly 1.0000, entire effect in the boundary channel — and that
effect is +0.0025 fragmentation. Post-hoc calibration cut ECE tenfold (0.037 →
0.0035) and moved decoded output essentially not at all.

**Conclusion: boundary-head work on this model is exhausted.** The raw boundary
channel already reaches F1 0.817/0.819 at its natural 0.50 threshold with no
diagnostic finding triggered; boundary-channel effects across the whole study
span ±0.014, while chord-posterior effects reach ±0.05. The residual
over-segmentation is a chord-posterior property.

## Status

- Phase 4 reconciliation: **complete and passing** (`boundary-v3-diagnosis.md`).
- Phase 5 diagnosis: **complete**, including the raw-vs-smoothed correction.
- Phase 6/7 freeze: **committed before any candidate was trained** (`f9f6197`).
- Phase 8 smoke: **passed** (finite losses, all four heads training, checkpoint
  save/resume, ONNX export, TypeScript parity, resume-identity guard verified
  in both directions).
- Phase 8 training: **complete**, 10/10 runs, 29.4 CPU-hours.
- Phase 9 evaluation: **complete**. `full-v2-control` reproduces both frozen
  baselines exactly (full-v2 mic 0.7565/0.7081/0.6719/22.207/429.6 and
  segmental-full mic 0.7583/0.7133/0.6486/20.783/479.0).
- Phase 10 freeze decision: **complete — retain v1, 0 eligible, p00 sealed.**

## Recommended next branch

Chord-posterior persistence and segment-level state replacement, **not** further
boundary-head calibration. The evidence is direct: the one intervention that
moved accuracy materially did so entirely through the chord posterior, and every
boundary-channel effect measured in this study is an order of magnitude smaller.
`agreement-coupled`'s +5 accuracy points at the cost of fragmentation also shows
the axis is a genuine trade-off worth studying rather than a free win.

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
