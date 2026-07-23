# FREEZE — learned-harmony-integration-v1

Checkpoint of the learned-harmony **integration boundary**. This stands in for a
Git tag (`learned-harmony-integration-v1`) because the repository is not yet a
Git repo. If/when `git init` is run, tag this state with that name.

Frozen: 2026-07-23.

## Frozen facts

| Item | Value |
|---|---|
| Contract | **v1** (`shared/learned-harmony-contract.json`) |
| App feature format | `harmony-features-v1` (request/response contract) |
| Model feature pipeline | `numpy-chroma-v1` (`ml/preprocessing/features.py`) |
| Production chord engine | `2026-07-harmonic-context-v3-reduced-latency` (unchanged, only user-facing engine) |
| Experimental model | **not bundled**; disabled by default; no UI |
| Hybrid decoder version | 1 |
| Default provider | `DisabledLearnedHarmonyProvider` |

> Note the two feature-version layers are intentionally distinct: the app-side
> `harmony-features-v1` (what the contract carries) vs the model-side
> `numpy-chroma-v1` (what the current numpy pipeline emits). A future
> `OnnxLearnedHarmonyProvider` must reconcile them — the model's
> `expectedFeatureVersion` must match the feature package the app actually builds.

## Change policy

The contract must not change casually. Prefer:
- adding **optional** fields (unavailable ones stay explicitly unavailable), or
- a **new contract version** (bump `contractVersion`, keep v1 validation intact).

Both the TypeScript (`src/learnedHarmony/contract.ts`) and Python
(`ml/exports/inference_contract.py`) validators must be updated together and stay
in lock-step with the shared JSON.

## Synthetic smoke model (kept for integration tests only)

Labeled **`temporal-baseline-v0-synthetic-only`** — see
`ml/checkpoints/temporal-baseline-v0-synthetic-only.provenance.json`
(regenerate with `python -m ml.exports.label_synthetic_model`).

| Field | Value |
|---|---|
| state_dict checksum | `bdf1aabb507aa3ff4a0ab372c5dabfbf8eac35cb1d74588d53cbf9254bac8f25` |
| ONNX sha256 | `c8c651d71cebad67a106dcb46a73e28e009911d9f4d5bfee1b09e34afed8b233` |
| training seed | 20260723 |
| model feature version | `numpy-chroma-v1` |
| parameters | 58,529 |

**This model is synthetic-only and must never be released, bundled, or promoted.**

## File inventory

Integration boundary (TypeScript, shipped with the app but not imported by the
entry point, so tree-shaken out — zero bundle impact):
`shared/learned-harmony-contract.json`, `src/learnedHarmony/{types,contract,provider,flag,featurePackage,hybridDecoder,cacheKey,diagnostics,index}.ts`.

Dev-only ML workspace (never packaged): `ml/` (schema, preprocessing, splits,
evaluation, models, training, exports). Torch is only in `ml/requirements-training.txt`.

## Verification (this frozen state)

```powershell
npx tsc --noEmit
npx vitest run                                                   # 105 tests
.\.venv\Scripts\python.exe -m unittest discover -s ml/tests -t . # 13 (torch-free)
.\.venv\Scripts\python.exe -m unittest discover -s ml/tests_training -t .   # 13 (needs ml[train])
npm run build                                                    # bundle unaffected by learned-harmony
```

## Next legitimate step (blocked on data)

Replace `MockLearnedHarmonyProvider` with a dev-only `OnnxLearnedHarmonyProvider`
behind the existing `TABSMITH_EXPERIMENTAL_LEARNED_HARMONY` flag (excluded from
packaging), then compare `harmonic-context-v3` / ML-only / hybrid offline. See
`ml/DATASET_READINESS.md` for the data gate that must be cleared first.
