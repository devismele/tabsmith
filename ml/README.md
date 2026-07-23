# learned-harmony-v1 — experimental chord + boundary ML workspace

> **Status (2026-07-23): PAUSED at `learned-harmony-integration-v1`.** The
> foundation, smoke model, and app integration boundary are complete and frozen
> (see [FREEZE-learned-harmony-integration-v1.md](./FREEZE-learned-harmony-integration-v1.md)).
> The limiting factor is now trustworthy licensed real data, not code — see
> [DATASET_READINESS.md](./DATASET_READINESS.md). The production app is unchanged
> and stays entirely rule-based. The smoke model is labeled
> `temporal-baseline-v0-synthetic-only` and must never be released.

Dev-only research workspace for a supervised temporal chord-recognition model
that learns **chord identity** and **chord-change boundaries** jointly from
licensed timed annotations, bass, melody, beats, bar position, key, and chord
history. It is **separate from the Electron app**: the packaged Tabsmith will
eventually ship only an exported inference model, never this corpus or pipeline.

The current production detector stays the baseline and default:

```
2026-07-harmonic-context-v3-reduced-latency   (src/chordAnalysis.ts, chordAnalysisVersion 4)
```

`learned-harmony-v1` is promoted to an engine option only after it beats this
baseline on held-out songs (see "Promotion gates" below). The prioritized target
is the **hybrid** engine: ML chord/boundary probabilities feeding the existing
rule-based decoder (bass, beats, transitions, locked user corrections).

## Hard restrictions (enforced by design)

- **No scraping / no auto-download.** No Ultimate Guitar scraping, no automatic
  download of copyrighted recordings, no bypassing auth/CAPTCHA/paywalls.
  Ultimate Guitar stays a **manual** evaluation reference only.
- **No lyrics or full tablature** are stored. The Tabsmith importer keeps chord
  symbols and timings only.
- **Licensing is explicit.** A dataset is used only when its license/permitted
  use is documented. Every importer records source + license in the manifest.
- **Audio availability is honest.** Tracks are `audio` / `features` /
  `annotations`; only `audio` tracks are raw-audio-trainable. Annotation-only
  entries can never be silently treated as audio.
- **No training on user audio without opt-in**; corrections are local by default
  and retraining is an explicit offline step (next pass).
- **No split mixing.** Splits are immutable and artist-level (below).

## What the foundation pass includes (runnable today, numpy-only)

```
ml/
  schema/jams_like.py          internal JAMS-compatible annotation format
  preprocessing/
    synth_generator.py         deterministic synthetic audio + perfect ground truth
    features.py                framewise chroma / bass-chroma / energy (feature contract)
    import_isophonics.py       .lab chord/beat/key/segment importer (annotations-only)
    import_billboard.py        McGill Billboard importer (annotations / features)
    import_tabsmith_ref.py     imports evaluation/chord-reference-songs.json (real data)
    prepare_dataset.py         assembles corpus + reproducibility manifest
  splits/make_splits.py        immutable artist-level splits + leakage detection
  evaluation/
    metrics.py                 chord + boundary metrics (ported from evaluation.mjs)
    baselines.py               chroma-template-v0 non-learned floor
    evaluate.py                scores predictions vs ground truth
  models/temporal_baseline.py  compact dilated TCN (root/quality/no-chord/boundary heads)
  training/{dataset,losses,checkpoint,train}.py   experimental smoke training (torch)
  exports/export_model.py      ONNX export + PyTorch/ONNX agreement check
  evaluation/{adapters,compare_experimental}.py   rule / ML-only / hybrid comparison
```

## Experimental training smoke pass (needs the `ml[train]` extra)

`ml/requirements-training.txt` (torch, onnx, onnxruntime) is an **optional dev
extra**, never an app dependency. It validates the full lifecycle — train → save
→ reload → predict → ONNX export → evaluate — on **synthetic data only**. It is
NOT a production chord model and makes no accuracy claim about real music. The
`test` split stays frozen; Hotel California is never used as validation.

```powershell
pip install -r ml/requirements-training.txt
.\.venv\Scripts\python.exe -m ml.training.train                       # smoke train the TCN
.\.venv\Scripts\python.exe -m ml.evaluation.compare_experimental      # rule vs ML vs hybrid (synthetic held-out)
.\.venv\Scripts\python.exe -m ml.exports.export_model                 # ONNX + agreement check
.\.venv\Scripts\python.exe -m unittest discover -s ml/tests_training -t .   # heavy ML tests
```

## Reproducible commands

```powershell
.\.venv\Scripts\python.exe -m ml.preprocessing.prepare_dataset      # build corpus + manifest
.\.venv\Scripts\python.exe -m ml.evaluation.evaluate                # score baseline vs ground truth
.\.venv\Scripts\python.exe -m ml.preprocessing.import_tabsmith_ref  # inspect real reference import
.\.venv\Scripts\python.exe -m unittest discover -s ml/tests         # foundation tests
.\.venv\Scripts\python.exe -m ml.training.train                     # (stub) explains next pass
```

Bring your own licensed academic data:

```powershell
.\.venv\Scripts\python.exe -m ml.preprocessing.prepare_dataset `
    --isophonics-dir <local Isophonics annotations> `
    --billboard-dir  <local McGill Billboard release>
```

## Dataset splits

Splits are assigned at the **artist** level and pre-assigned splits in
`evaluation/chord-reference-songs.json` are authoritative (Hotel California stays
`development`; the `test` set stays frozen and is never auto-assigned).
`splits/make_splits.py` flags artists that appear in multiple splits and
title-level near-duplicates (remaster / live / cover) as leakage warnings.

## Reproducibility manifest

`prepare_dataset` writes `datasets/dataset_manifest.json` capturing: seed, git
commit, schema + feature-pipeline versions, config, per-dataset versions and
licenses, audio-availability breakdown, class distribution (by duration), the
split assignment, and per-file annotation checksums. Model/training/validation
metrics and the model checksum are filled in by later passes.

## Promotion gates (before it can become the default)

Beats the baseline on held-out validation; improves root and major/minor
accuracy; preserves or reduces boundary error; does not increase fragmentation;
holds across genres and with/without isolated stems; passes existing Tabsmith
tests; runs fast enough on supported Windows hardware; and has a documented model
+ dataset license inventory. The `test` split stays untouched until the model and
thresholds are frozen.
