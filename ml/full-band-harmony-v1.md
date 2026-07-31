# Full-band harmony v1 foundation

This branch prepares the next data and evaluation gate. It does not contain,
download, train on, or authorize any full-band audio.

## Safety contract

A dataset is usable only when its portable manifest records:

- a verified license that explicitly permits model training and evaluation;
- source/distribution identity, attribution, and archive checksums;
- relative audio paths plus per-view SHA-256 checksums;
- artist, composition, recording, and split-group IDs;
- timed chord regions or normalized symbolic note events;
- explicit no-chord coverage;
- a frozen group-aware split with no artist/composition leakage.

The loader rejects absolute paths and path traversal. A local data root is
provided at execution time and is never written to reports.

The repository does not scrape or download commercial music. Dataset licensing
must be reviewed by the operator; these checks are engineering gates, not legal
advice.

## Audio views and baselines

Each recording should provide these paired views:

1. `full-mix`
2. `harmony-stem`
3. `guitar-stem`
4. `guitar-plus-bass`

The readiness plan evaluates rule v3, v1 ML-only, and v1 hybrid on every view
using identical tracks, chord regions, durations, vocabulary, and metrics.
Alternate views are paired captures of one performance, not independent songs.
The frozen planned matrix and paired-statistics contract live in
`ml/configs/full-band-harmony-v1-evaluation.json`.

The downstream source-separation comparison includes chord, note, tab, tuning,
and playable-string assignment metrics. It answers whether a separated source
improves Tabsmith, not merely whether it sounds clean.

## Symbolic labels

`ml.full_band.symbolic` accepts normalized MIDI-like note events (`start`,
`end`, `pitches`, optional `velocity`). It deterministically scores major,
minor, and dominant-seventh pitch sets, merges identical neighbours, and fills
all gaps with `N`. Empty/no-guitar examples become one full-duration `N`
region. This is deliberately reproducible and does not require generated audio.

## Local use

Copy `ml/configs/full-band-harmony-v1-manifest.example.json` outside the
repository and fill it only after license review. Then freeze a grouped split
and run readiness locally. Never commit the local manifest when it contains
private provenance or paths.

```powershell
npm run split:full-band -- `
  --manifest <portable-local-manifest.json> `
  --output <portable-frozen-split.json> `
  --seed 20260727
```

The tracked blocker report is generated with:

```powershell
npm run readiness:full-band
```

A real local manifest can be checked with:

```powershell
python -m ml.full_band.readiness `
  --manifest <portable-local-manifest.json> `
  --data-root <licensed-dataset-root>
```

After the readiness gate is cleared, the existing real-data trainer can consume
one or more approved views through `--full-band-manifest`,
`--full-band-root`, and `--full-band-views`. It preserves the frozen full-band
split while leaving GuitarSet/Billboard support unchanged. This branch does not
run that command.

The current tracked report must remain blocked until a suitable licensed
full-band dataset exists. No large training run should begin from this branch.

## Pilot v2: preserving root identification

Pilot v1 concluded *retain v1*. Its primary passed every full-band gate
(+14.83 pp full-mix detailed accuracy, +0.135 no-chord F1) and failed only
GuitarSet root preservation, by 4.8 pp on microphone and 8.18 pp on pickup
against a 2.0 pp tolerance. The regression was specific to root identification:
detailed accuracy did not regress and fragmentation improved on both captures.

Pilot v2 attacks that failure and **reuses the eligibility gates unchanged**
(`full-band-pilot-gates-20260727`). The gate v1 failed on is not allowed to move
for v2. Three candidates, each moving exactly one lever, all initialised from v1
on the shared 12-epoch schedule and the same seed v1 used:

| candidate | lever | role |
|---|---|---|
| `rehearsal-heavy` | 0.7/0.3 domain sampling | primary |
| `root-anchored-distillation` | KL to the frozen v1 root head, rehearsal frames only | primary |
| `low-learning-rate` | 5e-4 → 2e-4 | ablation |

`rehearsal-heavy` is an isolated change rather than a dilution. The epoch mixer
sizes an epoch by the domain that can supply its share without repetition, so
with 600 GuitarSet and 240 Slakh samples a 0.7/0.3 split still draws all 240
Slakh samples — exactly as many as the 50/50 v1 primary saw — while GuitarSet
rises from 240 to 560 per epoch. Full-band exposure per epoch is identical and
only rehearsal moves.

`low-learning-rate` exists so a win cannot be misread: without it, preserved
root accuracy under either primary is indistinguishable from simply drifting
less.

The distillation weight is frozen at 1.0 and **not tuned** — there is no
held-out budget for it, and tuning against the Slakh development set would leak
selection into the dev metric. A null result bounds that mechanism at weight
1.0 rather than refuting it.

### Running and resuming

Training is resumable per candidate at epoch granularity, and both the extracted
features and the per-model evaluation metrics are cached under the run
directory, so an interrupted run resumes in seconds rather than re-paying ~13
minutes of feature extraction. Caches record the identity of what they hold
(feature pipeline, boundary tolerance, track set and order; for evaluation, the
checkpoint digest) and recompute rather than serve a stale entry.

```powershell
# 1. train (safe to re-run; add --candidate to run one at a time)
.\.venv\Scripts\python.exe -m ml.full_band.run_pilot `
  --root <slakh-root> `
  --pilot-config ml\configs\full-band-pilot-v2.json `
  --run-dir ml\runs\full-band-pilot-v2 `
  --v1-checkpoint <path-to-temporal-baseline-real-v1.pt> `
  --annotations <guitarset-annotation> `
  --mic-audio <guitarset-mic> --pickup-audio <guitarset-pickup>

# 2. apply the frozen gates (--output keeps v1's frozen results intact)
.\.venv\Scripts\python.exe -m ml.full_band.evaluate_pilot `
  --root <slakh-root> --run-dir ml\runs\full-band-pilot-v2 `
  --v1-checkpoint <path-to-temporal-baseline-real-v1.pt> `
  --output evaluation\reports\full-band-pilot-v2-results.json

# 3. render the report and freeze the decision
.\.venv\Scripts\python.exe -m ml.full_band.pilot_v2_report `
  --input evaluation\reports\full-band-pilot-v2-results.json
```

`v1` throughout this workstream is `temporal-baseline-real-v1`
(`numpy-chroma-v1`, 283,505 parameters). Its metadata does not distinguish it
from `temporal-baseline-app-v0`, which records the same feature version and
parameter count; the pilot-v1 checkpoints sit roughly 15x closer to it in mean
absolute weight distance, which is what identifies it as the parent.
