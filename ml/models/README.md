# Models

`temporal_baseline.py` implements the compact model as an **experimental smoke
pass** (torch, behind `ml/requirements-training.txt`). It is a
pipeline-validation model trained on synthetic data only — not production, not
promoted. The foundation pass (schema/importers/features/eval) still runs with
numpy alone.

Architecture (implemented):

```
Feature encoder (chroma | bass chroma | energy; timing/bar-position/context next)
        ↓
Temporal model  (dilated TCN now → Transformer/Conformer later)
        ↓
Shared representation
  ├── chord-root head        (12 + N)
  ├── chord-quality head     (maj / min / dom7 / …)
  ├── no-chord head
  ├── boundary head          P(change at t), soft targets around annotated + beat/downbeat
  └── (aux) bass-root, key heads
```

Everything consumes `ml/preprocessing/features.py::FeatureFrames` and is scored
by `ml/evaluation/metrics.py`, so the model is a drop-in for the existing
comparison harness. Output vocabulary v1: 12 major, 12 minor, 12 dominant-7, N,
X (root and quality predicted separately).

The boundary head is explicit — boundaries are **not** inferred only from
frame-label changes. Decoding then combines ML chord/boundary probabilities with
the existing rule-based decoder (bass, beats, transitions) as the **hybrid**
engine, which is the prioritized deliverable.
