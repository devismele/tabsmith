"""Python mirror of the learned-harmony integration contract.

Kept in lock-step with ``shared/learned-harmony-contract.json`` and
``src/learnedHarmony/contract.ts`` so a future ONNX/Python inference worker
validates exactly what the TypeScript app expects. Pure stdlib + math — no torch,
so it stays importable in the foundation pass.

Nothing here loads or downloads a model. ``build_model_manifest`` only describes
a checkpoint's metadata; installing a real model requires an explicit, verified
release artifact.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

CONTRACT_PATH = Path(__file__).resolve().parents[2] / "shared" / "learned-harmony-contract.json"
CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
CONTRACT_VERSION = CONTRACT["contractVersion"]
FEATURE_VERSION = CONTRACT["featureVersion"]
CHROMA_BINS = CONTRACT["limits"]["chromaBins"]
QUALITY_COUNT = len(CONTRACT["vocabulary"]["qualities"])


def _finite_vector(values, length=None) -> bool:
    if not isinstance(values, list):
        return False
    if length is not None and len(values) != length:
        return False
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def _finite_matrix(values, rows, cols) -> bool:
    return isinstance(values, list) and len(values) == rows and all(_finite_vector(r, cols) for r in values)


def _strictly_increasing(values) -> bool:
    return all(values[i] > values[i - 1] for i in range(1, len(values)))


def validate_request(request: dict) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if request.get("contractVersion") != CONTRACT_VERSION:
        errors.append("contractVersion mismatch")
    if not request.get("requestId"):
        errors.append("requestId is required")
    if request.get("featureVersion") != FEATURE_VERSION:
        errors.append("featureVersion mismatch")
    meta = request.get("modelMetadata", {})
    if meta.get("expectedFeatureVersion") != request.get("featureVersion"):
        errors.append("expectedFeatureVersion mismatch")
    if not (request.get("sectionEndSeconds", 0) > request.get("sectionStartSeconds", 0)):
        errors.append("section end must exceed start")

    frames = request.get("frameTimes")
    if not _finite_vector(frames) or not frames:
        errors.append("frameTimes must be a non-empty finite array")
        return False, errors
    if len(frames) > CONTRACT["limits"]["maxFrames"]:
        errors.append("frameTimes exceeds maxFrames")
    if not _strictly_increasing(frames):
        errors.append("frameTimes must be strictly increasing")

    t = len(frames)
    if not _finite_matrix(request.get("harmonicChroma"), t, CHROMA_BINS):
        errors.append("harmonicChroma shape/finite invalid")
    if not _finite_matrix(request.get("bassChroma"), t, CHROMA_BINS):
        errors.append("bassChroma shape/finite invalid")
    if not _finite_vector(request.get("onsetStrength"), t):
        errors.append("onsetStrength invalid")
    return len(errors) == 0, errors


def _probability_matrix(values, rows, cols) -> bool:
    if not _finite_matrix(values, rows, cols):
        return False
    tol = CONTRACT["limits"]["maxProbabilitySumError"]
    for row in values:
        if any(v < 0 or v > 1 for v in row):
            return False
        if abs(sum(row) - 1.0) > tol:
            return False
    return True


def validate_response(response: dict, request: dict) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if response.get("contractVersion") != CONTRACT_VERSION:
        errors.append("response contractVersion mismatch")
    if response.get("requestId") != request.get("requestId"):
        errors.append("requestId mismatch")
    if response.get("featureVersion") != request.get("featureVersion"):
        errors.append("featureVersion mismatch")
    frames = response.get("frameTimes")
    if not _finite_vector(frames) or len(frames) != len(request.get("frameTimes", [])):
        errors.append("response frameTimes length mismatch")
        return False, errors
    t = len(frames)
    if not _probability_matrix(response.get("rootProbabilities"), t, CHROMA_BINS):
        errors.append("rootProbabilities invalid")
    if not _probability_matrix(response.get("qualityProbabilities"), t, QUALITY_COUNT):
        errors.append("qualityProbabilities invalid")
    for key in ("noChordProbabilities", "boundaryProbabilities"):
        arr = response.get(key)
        if not _finite_vector(arr, t) or any(v < 0 or v > 1 for v in arr):
            errors.append(f"{key} invalid")
    return len(errors) == 0, errors


def build_model_manifest(checkpoint_metadata: dict) -> dict:
    """Describe an exported checkpoint as a model manifest (no model is bundled)."""
    vocab = checkpoint_metadata.get("chordVocabulary", {})
    return {
        "modelVersion": checkpoint_metadata.get("modelVersion", "temporal-baseline-v0"),
        "contractVersion": CONTRACT_VERSION,
        "featureVersion": FEATURE_VERSION,
        "checksum": checkpoint_metadata.get("checksum", ""),
        "format": "onnx",
        "vocabulary": {
            "roots": vocab.get("roots", CONTRACT["vocabulary"]["roots"]),
            "qualities": vocab.get("qualities", CONTRACT["vocabulary"]["qualities"]),
            "hasNoChordHead": bool(vocab.get("noChord", True)),
            "hasBoundaryHead": True,
        },
        "inputShapes": {"features": ["batch", "frames", checkpoint_metadata.get("inputDim", 25)]},
    }


def validate_model_manifest(manifest: dict) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if manifest.get("contractVersion") != CONTRACT_VERSION:
        errors.append("manifest contractVersion mismatch")
    if manifest.get("featureVersion") != FEATURE_VERSION:
        errors.append("manifest featureVersion mismatch")
    if not manifest.get("checksum"):
        errors.append("manifest checksum required")
    if manifest.get("format") not in ("onnx", "python"):
        errors.append("manifest format must be onnx or python")
    return len(errors) == 0, errors


if __name__ == "__main__":
    print(f"Contract v{CONTRACT_VERSION} feature={FEATURE_VERSION} "
          f"roots={CONTRACT['vocabulary']['roots']} qualities={CONTRACT['vocabulary']['qualities']}")
