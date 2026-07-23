"""Stamp the smoke-test model unmistakably as synthetic-only.

    python -m ml.exports.label_synthetic_model

Writes a provenance sidecar next to the checkpoint recording everything needed
to keep the model for integration tests while making it impossible to mistake
for a release model. Pure stdlib (no torch), so it stays in the foundation pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
LABEL = "temporal-baseline-v0-synthetic-only"
WARNING = ("SYNTHETIC-ONLY pipeline-validation model. NEVER a release model. Trained on "
           "generated audio; not representative of commercial music. Must not be bundled "
           "into Tabsmith or promoted over harmonic-context-v3-reduced-latency.")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_provenance(checkpoint: Path) -> dict:
    metadata = json.loads(checkpoint.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    history_path = checkpoint.with_suffix(".history.json")
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {}
    onnx_path = checkpoint.with_suffix(".onnx")
    return {
        "label": LABEL,
        "syntheticOnly": True,
        "warning": WARNING,
        "productionEngine": "2026-07-harmonic-context-v3-reduced-latency",
        "contractVersion": 1,
        "featureVersion": metadata.get("featureVersion"),
        "trainingSeed": metadata.get("trainingSeed"),
        "datasetManifestHash": metadata.get("datasetManifestHash"),
        "trainingConfig": metadata.get("trainingConfig"),
        "stateDictChecksum": metadata.get("checksum"),
        "checkpointFileSha256": _sha256(checkpoint),
        "onnxSha256": _sha256(onnx_path) if onnx_path.exists() else None,
        "parameterCount": metadata.get("parameterCount"),
        "bestEpoch": history.get("bestEpoch"),
        "bestDevLoss": history.get("bestDevLoss"),
        "labeledAt": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Label the smoke model as synthetic-only.")
    parser.add_argument("--checkpoint", default=str(ML_ROOT / "checkpoints" / "temporal-baseline-v0.pt"))
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"No checkpoint at {checkpoint}. Run `python -m ml.training.train` first.")
        return 1
    provenance = build_provenance(checkpoint)
    out = checkpoint.with_name(f"{LABEL}.provenance.json")
    out.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    print(json.dumps({k: provenance[k] for k in
                      ("label", "featureVersion", "trainingSeed", "stateDictChecksum", "onnxSha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
