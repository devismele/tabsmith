# Full-band harmony v1 readiness

Status: **blocked**. Large training is **not authorized**.

This foundation contains no audio and performs no downloads. A local dataset may be used only after its license, provenance, checksums, grouped splits, timed labels, no-chord coverage, and alternate audio views pass the gates.

## Inventory

| Tracks | Artists | Compositions | Hours | License gate | Split gate |
|---:|---:|---:|---:|---|---|
| 0 | 0 | 0 | 0.00 | blocked | blocked |

## Planned paired baselines

Every approved track must be evaluated through the same annotations and metric implementation using full mix, harmony stem, guitar stem, and guitar-plus-bass. Rule v3, v1 ML-only, and v1 hybrid are compared per view. Alternate views are paired observations, never independent songs.

## Blockers

- License/provenance gate: license.status must be verified before data use
- No legally approved full-band tracks are configured.

## Promotion gate

No model training, production-weight change, or promotion is authorized by this report. A frozen artist/composition-disjoint full-band test set and downstream source-separation comparisons remain mandatory.
