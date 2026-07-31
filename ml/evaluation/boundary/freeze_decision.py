"""Phase 10: freeze the boundary-calibration selection decision.

Records an immutable, checksummed record of what was decided and why, before
p00 may be considered. The decision itself is mechanical -- it reads the frozen
gate outcomes and does not re-derive them -- so this module cannot quietly
rescue a candidate that failed.

Retention is the default. A candidate is selected only if it passed *every*
frozen gate on *both* captures; otherwise v1 is retained, no replacement model
is exported, and p00 stays sealed.

What gets frozen for a selected candidate:

* the per-fold checkpoint paths and their state-dict checksums
* the model configuration and training overrides
* the boundary calibration family and per-fold fitted parameters
* the decoder configuration and the per-fold operating thresholds
* the candidate-manifest and gate-set checksums that governed the decision
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ..segmental.seg_metrics import CAPTURES
from ..temporal_v2_ablation import _read_json
from .evaluate import DEFAULT_CANDIDATES, DEFAULT_RUN_DIR
from .gates import DEFAULT_GATES, load_gates

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = REPO_ROOT / "evaluation" / "reports"
RESULTS_NAME = "boundary-v3-results.json"


def _sha256_file(path: Path) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _checksum(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def collect_checkpoints(candidate_id: str, run_dir: Path) -> dict[str, Any]:
    """Per-fold checkpoint identity for a retrained candidate."""
    run_dir = Path(run_dir)
    folds: dict[str, Any] = {}
    for fold_dir in sorted(run_dir.glob("lopo-*")):
        checkpoint = fold_dir / candidate_id / "model.pt"
        metadata_path = fold_dir / candidate_id / "model.metadata.json"
        if not checkpoint.exists():
            continue
        record: dict[str, Any] = {"weightFileSha256": _sha256_file(checkpoint)}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            record["stateDictChecksum"] = metadata.get("checksum")
            record["modelVersion"] = metadata.get("modelVersion")
            record["featureVersion"] = metadata.get("featureVersion")
            record["parameterCount"] = metadata.get("parameterCount")
            record["trainingSeed"] = metadata.get("trainingSeed")
        folds[fold_dir.name] = record
    return folds


def decide(results: dict[str, Any], manifest: dict[str, Any], gates: dict[str, Any],
           *, run_dir: Path) -> dict[str, Any]:
    """Apply the frozen outcome and produce the freeze record."""
    gate_results = results.get("gateResults", {})
    eligible = list(results.get("eligibleCandidates", []))

    # Defensive: never trust a precomputed eligibility list over the checks.
    recomputed = [cid for cid, r in gate_results.items() if r.get("eligible")]
    if sorted(recomputed) != sorted(eligible):
        raise ValueError(
            f"eligibility disagrees with the recorded checks: {eligible} vs {recomputed}")

    selected = eligible[0] if len(eligible) == 1 else None
    if len(eligible) > 1:
        # Tie-break only among candidates that already passed every gate.
        def score(cid: str) -> tuple[float, float]:
            caps = results["evaluations"][cid]["decoders"]["segmental-full"]["byCapture"]
            frag = sum(caps[c]["fragmentationRate"] for c in CAPTURES) / len(CAPTURES)
            detailed = sum(caps[c]["detailedAccuracy"] for c in CAPTURES) / len(CAPTURES)
            return (-frag, detailed)
        selected = max(eligible, key=score)

    decision = "select-candidate" if selected else "retain-v1"
    record: dict[str, Any] = {
        "schemaVersion": 1,
        "decision": decision,
        "selectedCandidate": selected,
        "eligibleCandidates": eligible,
        "eligibleCandidateCount": len(eligible),
        "candidateSetId": manifest["ablationId"],
        "candidateManifestChecksum": _checksum(manifest),
        "gateSetId": gates["gateSetId"],
        "gateSetChecksum": _checksum(gates),
        "resultsChecksum": _checksum(results.get("evaluations", {})),
        "p00Accessed": False,
        "p00Sealed": selected is None,
        "productionWeightsReplaced": False,
        "applicationDefaultsChanged": False,
        "releaseGatingChanged": False,
        "gateOutcomes": {
            cid: {"eligible": r["eligible"], "failedGates": r["failedGates"],
                  "firstFailure": r["firstFailure"]}
            for cid, r in gate_results.items()
        },
    }

    if selected:
        candidate_def = next(c for c in manifest["candidates"] if c["id"] == selected)
        evaluation = results["evaluations"][selected]
        record["frozenConfiguration"] = {
            "candidate": candidate_def,
            "checkpoints": collect_checkpoints(selected, run_dir),
            "boundaryCalibration": {
                key: value["calibrator"]
                for key, value in evaluation["foldCalibration"].items()
            },
            "operatingThresholds": {
                key: value["operatingThreshold"]
                for key, value in evaluation["foldCalibration"].items()
            },
            "decoder": "segmental-full",
            "decoderKnobs": next(
                d for d in manifest["decoders"] if d["id"] == "segmental-full"),
            "metrics": {
                cap: evaluation["decoders"]["segmental-full"]["byCapture"][cap]
                for cap in CAPTURES
            },
        }
        record["nextStep"] = (
            "A single sealed p00 run is now permitted, after export parity is verified. "
            "Do not retune after p00 results."
        )
    else:
        record["retained"] = "v1"
        record["nextStep"] = (
            "p00 remains sealed. No replacement model is exported and no production "
            "weight, application default or release gate changes."
        )
    return record


def build_markdown(record: dict[str, Any], results: dict[str, Any]) -> str:
    selected = record["selectedCandidate"]
    lines = [
        "# Boundary-calibration-v3 selection decision",
        "",
        (f"**Decision: select `{selected}`.**" if selected
         else "**Decision: retain v1. No boundary candidate is eligible. p00 remains sealed.**"),
        "",
        f"Candidate set `{record['candidateSetId']}` under gate set `{record['gateSetId']}`.",
        "Gates were frozen before any candidate was trained and were not altered afterwards.",
        "",
        "## Gate outcomes",
        "",
        "| candidate | eligible | first failure |",
        "|---|---|---|",
    ]
    for cid, outcome in record["gateOutcomes"].items():
        lines.append(f"| {cid} | {outcome['eligible']} | {outcome['firstFailure'] or '-'} |")
    lines += ["", f"Eligible candidates: **{record['eligibleCandidateCount']}**", ""]

    control = results.get("evaluations", {}).get("full-v2-control")
    if control:
        lines += [
            "## Reference points (segmental-full decoder)",
            "",
            "| model | capture | root | detailed | frag | rpm | MAE ms |",
            "|---|---|---|---|---|---|---|",
        ]
        for cid, evaluation in results["evaluations"].items():
            if "error" in evaluation:
                continue
            for cap in CAPTURES:
                m = evaluation["decoders"]["segmental-full"]["byCapture"][cap]
                lines.append(
                    f"| {cid} | {cap} | {m['rootAccuracy']:.4f} | {m['detailedAccuracy']:.4f} | "
                    f"{m['fragmentationRate']:.4f} | {m['regionsPerMinute']:.3f} | "
                    f"{m['meanAbsoluteBoundaryErrorMs']:.1f} |")
        lines.append("")

    if selected:
        frozen = record["frozenConfiguration"]
        lines += [
            "## Frozen configuration",
            "",
            f"- decoder: `{frozen['decoder']}`",
            f"- boundary calibration family: "
            f"`{frozen['candidate'].get('boundaryCalibration', 'identity')}`",
            "",
            "| fold | checkpoint SHA-256 | state-dict checksum |",
            "|---|---|---|",
        ]
        for fold, info in frozen["checkpoints"].items():
            lines.append(f"| {fold} | `{info.get('weightFileSha256')}` | "
                         f"`{info.get('stateDictChecksum')}` |")
        lines += ["", f"- next step: {record['nextStep']}", ""]
    else:
        lines += [
            "## Consequence",
            "",
            "- Retain v1. Do not replace it.",
            "- p00 was NOT accessed; the protocol forbids opening it without an eligible candidate.",
            "- No replacement model exported; no production weight, application default or",
            "  release gate changed.",
            "",
        ]

    lines += [
        "## Checksums",
        "",
        f"- candidate manifest: `{record['candidateManifestChecksum']}`",
        f"- gate set: `{record['gateSetChecksum']}`",
        f"- evaluation results: `{record['resultsChecksum']}`",
        "",
        "This is grouped integration-domain development evidence on GuitarSet solo",
        "guitar, not production readiness.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the boundary-v3 selection decision.")
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--gates", default=str(DEFAULT_GATES))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    results_path = report_dir / RESULTS_NAME
    if not results_path.exists():
        parser.error(f"{RESULTS_NAME} not found; run ml.evaluation.boundary.run_study first")

    results = json.loads(results_path.read_text(encoding="utf-8"))
    manifest = _read_json(args.candidates)
    gates = load_gates(Path(args.gates))
    record = decide(results, manifest, gates, run_dir=Path(args.run_dir))

    (report_dir / "boundary-v3-selection.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    (report_dir / "boundary-v3-selection.md").write_text(
        build_markdown(record, results), encoding="utf-8")

    print(f"decision: {record['decision']}")
    print(f"eligible: {record['eligibleCandidates']}")
    print(f"p00 sealed: {record['p00Sealed']}")


if __name__ == "__main__":
    main()
