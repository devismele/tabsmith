"""Verify the frozen temporal-harmony-v2 starting baseline.

This command does not train, tune, or evaluate p00 again. It reconciles the
tracked v1 history/metadata, the local heavyweight training artifacts when
present, the bundled application export, and the already completed TypeScript
parity evaluation reports.

Run:
    python -m ml.evaluation.temporal_v2_baseline --require-local-artifacts
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = REPO_ROOT / "ml"
BASELINE_PATH = ML_ROOT / "baselines" / "temporal-harmony-v2-v1-baseline.json"
METADATA_PATH = ML_ROOT / "checkpoints" / "temporal-baseline-app-v0.metadata.json"
HISTORY_PATH = ML_ROOT / "checkpoints" / "temporal-baseline-app-v0.history.json"
CHECKPOINT_PATH = ML_ROOT / "checkpoints" / "temporal-baseline-app-v0.pt"
ONNX_PATH = ML_ROOT / "checkpoints" / "temporal-baseline-app-v0.onnx"
APP_WEIGHTS_PATH = REPO_ROOT / "src" / "learnedHarmony" / "model" / "app-chord-model.weights.json"
REPORT_PATHS = {
    "audio_mono-mic": REPO_ROOT / "evaluation" / "reports" / "hybrid-accuracy-mono-mic.json",
    "audio_mono-pickup_mix": REPO_ROOT / "evaluation" / "reports" / "hybrid-accuracy-pickup-mix.json",
}

METRIC_KEYS = (
    "rootAccuracy",
    "majorMinorAccuracy",
    "detailedAccuracy",
    "fragmentationRate",
    "regionsPerMinute",
    "meanAbsoluteBoundaryErrorMs",
    "medianAbsoluteBoundaryErrorMs",
    "boundariesWithin100ms",
    "boundariesWithin250ms",
    "boundariesWithin500ms",
    "boundariesWithin1000ms",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(checks: list[dict[str, Any]], name: str, actual: Any, expected: Any) -> None:
    checks.append({
        "name": name,
        "passed": actual == expected,
        "actual": actual,
        "expected": expected,
    })


def verify_baseline(*, require_local_artifacts: bool = False) -> dict[str, Any]:
    baseline = _read_json(BASELINE_PATH)
    training = baseline["training"]
    app_export = baseline["applicationExport"]
    historical = baseline["historicalValidation"]
    metadata = _read_json(METADATA_PATH)
    history = _read_json(HISTORY_PATH)
    weights = _read_json(APP_WEIGHTS_PATH)
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []

    _check(checks, "metadata.modelVersion", metadata["modelVersion"], training["modelVersion"])
    _check(checks, "metadata.declaredFeatureVersion",
           metadata["featureVersion"], training["declaredFeatureVersion"])
    _check(checks, "metadata.datasetManifestHash",
           metadata["datasetManifestHash"], training["datasetManifestHash"])
    _check(checks, "metadata.stateDictChecksum", metadata["checksum"], training["stateDictChecksum"])
    _check(checks, "metadata.parameterCount", metadata["parameterCount"], training["parameterCount"])
    _check(checks, "history.seed", history["seed"], training["trainingSeed"])
    _check(checks, "history.bestEpoch", history["bestEpoch"], training["bestEpoch"])
    _check(checks, "history.epochsRun", history["epochsRun"], training["epochsRun"])
    _check(checks, "history.bestDevLoss", history["bestDevLoss"], training["bestDevLoss"])
    _check(checks, "history.trainSeconds", history["trainSeconds"], training["trainSeconds"])
    _check(checks, "history.bestEpoch.train",
           history["history"][history["bestEpoch"]]["train"], training["bestEpochLosses"]["train"])
    _check(checks, "history.bestEpoch.development",
           history["history"][history["bestEpoch"]]["dev"], training["bestEpochLosses"]["development"])

    for path, expected_key, label in (
        (CHECKPOINT_PATH, "checkpointFileSha256", "checkpoint"),
        (ONNX_PATH, "onnxFileSha256", "onnx"),
    ):
        if path.exists():
            _check(checks, f"{label}.sha256", _sha256(path), training[expected_key])
        elif require_local_artifacts:
            _check(checks, f"{label}.present", False, True)
        else:
            warnings.append(
                f"{label} artifact is not present locally; its frozen checksum was not re-verified."
            )

    _check(checks, "applicationExport.modelVersion",
           weights["modelVersion"], app_export["modelVersion"])
    _check(checks, "applicationExport.modelChecksum",
           weights["modelChecksum"], app_export["modelChecksum"])
    _check(checks, "applicationExport.featureVersion",
           weights["featureVersion"], training["actualFeatureVersion"])
    _check(checks, "applicationExport.snapshotFeatureVersion",
           weights["featureVersion"], app_export["featureVersion"])
    _check(checks, "applicationExport.weightsFileSha256",
           _sha256(APP_WEIGHTS_PATH), app_export["weightsFileSha256"])
    _check(checks, "applicationExport.paritySample", bool(weights.get("paritySample")), True)

    for capture, report_path in REPORT_PATHS.items():
        report = _read_json(report_path)
        dataset = report["datasets"][0]
        ml_only = report["aggregate"]["engines"]["ml-only"]["durationWeighted"]
        expected = historical["captures"][capture]
        _check(checks, f"{capture}.performer",
               dataset["leakageAudit"]["heldOutPerformers"], [historical["performer"]])
        _check(checks, f"{capture}.trackCount", dataset["trackCount"], historical["trackCount"])
        _check(checks, f"{capture}.duration",
               dataset["totalEvaluatedDurationSeconds"], historical["evaluatedDurationSeconds"])
        for metric in METRIC_KEYS:
            _check(checks, f"{capture}.{metric}", ml_only[metric], expected[metric])

    failed = [check for check in checks if not check["passed"]]
    return {
        "schemaVersion": 1,
        "baselineId": baseline["baselineId"],
        "status": "passed" if not failed else "failed",
        "selectionEligibility": {
            "p00Eligible": False,
            "reason": historical["purpose"] if "purpose" in historical else baseline["purpose"],
        },
        "checks": checks,
        "warnings": warnings,
        "failedCheckNames": [check["name"] for check in failed],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the frozen v1 baseline for temporal-harmony-v2.")
    parser.add_argument("--require-local-artifacts", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print the complete machine-readable result.")
    args = parser.parse_args()
    result = verify_baseline(require_local_artifacts=args.require_local_artifacts)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        passed = sum(1 for check in result["checks"] if check["passed"])
        total = len(result["checks"])
        print(f"{result['baselineId']}: {result['status']} ({passed}/{total} checks)")
        for warning in result["warnings"]:
            print(f"warning: {warning}")
        for name in result["failedCheckNames"]:
            print(f"failed: {name}")
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
