"""Run Phase 4 reconciliation + Phase 5 boundary diagnosis and write reports.

Reports are portable (no absolute paths, no frame dumps) and land under
``evaluation/reports``; the inference cache stays under the git-ignored
``ml/runs``. Nothing here trains, and p00 is never loaded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..segmental.evaluate import DEFAULT_CACHE_DIR, DEFAULT_CANDIDATES, load_candidates, load_cache_entries
from ..segmental.seg_metrics import CAPTURES
from .diagnose import capture_disagreement, classify_dominant_issue, diagnose_entries
from .reconcile import reconcile

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = REPO_ROOT / "evaluation" / "reports"


def _assert_portable(text: str) -> str:
    lowered = text.replace("\\", "/").lower()
    for needle in ("c:/users", "/users/devis", str(REPO_ROOT).replace("\\", "/").lower()):
        if needle and needle in lowered:
            raise ValueError(f"absolute path leaked into report: {needle}")
    return text


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_assert_portable(text), encoding="utf-8")


def _strip_private(payload: Any) -> Any:
    """Drop internal ``_``-prefixed working state before serialising a report."""
    if isinstance(payload, dict):
        return {k: _strip_private(v) for k, v in payload.items() if not k.startswith("_")}
    if isinstance(payload, list):
        return [_strip_private(v) for v in payload]
    return payload


def _reliability_table(rows: list[dict]) -> list[str]:
    out = ["| bin | frames | mean predicted | observed | gap |", "|---|---|---|---|---|"]
    for r in rows:
        if not r["count"]:
            continue
        gap = r["meanPredicted"] - r["observedFrequency"]
        out.append(f"| {r['binLow']:.1f}-{r['binHigh']:.1f} | {r['count']} | "
                   f"{r['meanPredicted']:.4f} | {r['observedFrequency']:.4f} | {gap:+.4f} |")
    return out


def _sweep_table(sweep: dict[str, dict]) -> list[str]:
    out = ["| threshold | predicted | precision | recall | F1 |", "|---|---|---|---|---|"]
    for th in sorted(sweep, key=float):
        row = sweep[th]
        out.append(f"| {th} | {row['predicted']} | {row['precision']:.4f} | "
                   f"{row['recall']:.4f} | {row['f1']:.4f} |")
    return out


def build_smoothing_section(diagnoses: dict, raw: dict, raw_issues: dict) -> list[str]:
    """Smoothed vs raw boundary channel, so downstream damage is attributable."""
    lines = [
        "## Smoothed vs raw boundary channel",
        "",
        "The full-v2 decode applies EMA smoothing (boundaryAlpha 0.35) to the boundary",
        "channel. Everything above is measured through that smoothing, which is the",
        "configuration full-v2 actually runs. Measuring the raw head output separates a",
        "weak head from a head degraded downstream:",
        "",
        "| capture | channel | best F1 | at threshold | F1 at 0.50 | ECE | agreement at best |",
        "|---|---|---|---|---|---|---|",
    ]
    for cap in CAPTURES:
        for label, d in (("smoothed", diagnoses[cap]), ("raw", raw[cap])):
            sweep = d["boundaryThresholdSweep"]
            best = d["bestF1Threshold"]
            agreement = d["stateChangeBoundaryAgreementByThreshold"].get(best, 0.0)
            lines.append(
                f"| {cap} | {label} | {sweep[best]['f1']:.4f} | {best} | "
                f"{sweep.get('0.50', {}).get('f1', 0.0):.4f} | "
                f"{d['calibration']['expectedCalibrationError']:.4f} | {agreement:.4f} |")
    lines += [
        "",
        "Findings on the raw channel: "
        + "; ".join(f"{cap}: {', '.join(raw_issues[cap]['findings']) or 'none triggered'}"
                    for cap in CAPTURES),
        "",
    ]
    return lines


def build_markdown(reconciliation: dict, diagnoses: dict, issues: dict,
                   disagreement: dict, raw: dict | None = None,
                   raw_issues: dict | None = None) -> str:
    lines = [
        "# Boundary-calibration-v3 diagnosis (full-v2, p01-p05)",
        "",
        "Model-side boundary behaviour measured on the frozen segmental-v3 inference cache.",
        "Decoder held fixed at the faithful full-v2 path (EMA smoothing + penalty-4 Viterbi),",
        "so every number below describes the model, not a decoder variant. p00 never loaded.",
        "",
        "## Baseline reconciliation",
        "",
        f"- cache entries: {reconciliation['cacheEntries']} across folds {', '.join(reconciliation['folds'])}",
        f"- performers: {', '.join(reconciliation['sealing']['performers'])} "
        f"(expected match: {reconciliation['sealing']['performersMatch']})",
        f"- p00 sealed: {reconciliation['sealing']['p00Sealed']} "
        f"(by performer id: {reconciliation['sealing']['p00EntriesByPerformerId']}, "
        f"by performance prefix: {reconciliation['sealing']['p00EntriesByPerformanceIdPrefix']})",
        f"- mic/pickup paired for all {reconciliation['pairing']['performances']} performances: "
        f"{reconciliation['pairing']['allPaired']}",
        f"- pairing held within fold: {reconciliation['pairing']['pairingHeldWithinFold']}",
        f"- fold checkpoints match cached checksums: {reconciliation['checkpoints']['allFoldsMatch']}",
    ]
    equality = reconciliation["cacheEquality"]
    if equality["status"] == "compared":
        lines.append(
            f"- cached vs recomputed inference: {equality['sampled']} sampled, "
            f"max |delta| {equality['maxAbsoluteDelta']:.2e}, "
            f"within tolerance: {equality['allWithinTolerance']}")
    else:
        lines.append(f"- cached vs recomputed inference: {equality['status']} "
                     f"({equality.get('reason', 'n/a')})")
    lines += [f"- **reconciled: {reconciliation['reconciled']}**", ""]

    for cap in CAPTURES:
        d = diagnoses[cap]
        cal = d["calibration"]
        ev = d["categoryEvidence"]
        lines += [
            f"## {cap}",
            "",
            "### Calibration",
            "",
            f"- frames: {cal['frames']}; boundary base rate (±0.25 s): {cal['positiveRate']:.4f}",
            f"- mean predicted boundary probability: {cal['meanPredicted']:.4f}",
            f"- expected calibration error: {cal['expectedCalibrationError']:.4f}",
            f"- maximum calibration error: {cal['maximumCalibrationError']:.4f}",
            "",
        ]
        lines += _reliability_table(cal["reliability"])
        lines += [
            "",
            "### Boundary threshold sweep (peak-picked, ±0.25 s)",
            "",
        ]
        lines += _sweep_table(d["boundaryThresholdSweep"])
        lines += [
            "",
            f"- best F1 threshold: {d['bestF1Threshold']}",
            "",
            "### Decoded state switches",
            "",
            f"- state-switch P/R/F1: {d['stateSwitch']['precision']:.4f} / "
            f"{d['stateSwitch']['recall']:.4f} / {d['stateSwitch']['f1']:.4f}",
            f"- state-change/boundary agreement at peak threshold 0.50: {d['stateChangeBoundaryAgreement']:.4f}",
            "- agreement by peak threshold: "
            + ", ".join(f"{th}={v:.3f}" for th, v in
                        sorted(d["stateChangeBoundaryAgreementByThreshold"].items(), key=lambda kv: float(kv[0]))),
            f"- taxonomy: {json.dumps(d['taxonomy'], sort_keys=True)}",
            f"- bass-root agreement: {d['bassRootAgreement']:.4f}",
            "",
            "### Evidence by category (boundary probability / chord margin)",
            "",
            "| category | n | mean boundary prob | median | mean margin |",
            "|---|---|---|---|---|",
        ]
        for name in ("trueTransition", "falseTransition", "missedTransition", "sustainedNonBoundary"):
            bp = ev[name]["boundaryProb"]
            mg = ev[name]["margin"]
            lines.append(f"| {name} | {bp['count']} | {bp['mean']:.4f} | {bp['median']:.4f} | {mg['mean']:.4f} |")
        shape = d["regionShape"]
        timing = d["timing"]
        lines += [
            "",
            "### Region shape and timing",
            "",
            f"- duration before FALSE switch: mean {shape['falsePreviousDurationS']['mean']:.3f} s, "
            f"median {shape['falsePreviousDurationS']['median']:.3f} s",
            f"- duration of FALSE proposed region: mean {shape['falseProposedDurationS']['mean']:.3f} s, "
            f"median {shape['falseProposedDurationS']['median']:.3f} s",
            f"- duration before TRUE switch: mean {shape['truePreviousDurationS']['mean']:.3f} s",
            f"- duration of TRUE proposed region: mean {shape['trueProposedDurationS']['mean']:.3f} s",
            f"- boundary timing: mean signed {timing['meanSignedErrorMs']:+.1f} ms, "
            f"median signed {timing['medianSignedErrorMs']:+.1f} ms, "
            f"MAE {timing['meanAbsoluteErrorMs']:.1f} ms",
            f"- early: {timing['earlyCount']}, late: {timing['lateCount']}",
            "",
            "### Per performer",
            "",
            "| performer | true | false | missed | predicted regions | reference regions | mean prob true | mean prob false |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for pid, v in d["perPerformer"].items():
            lines.append(
                f"| {pid} | {v['trueTransitions']} | {v['falseTransitions']} | {v['missedTransitions']} | "
                f"{v['predictedRegions']} | {v['referenceRegions']} | "
                f"{v['meanTrueBoundaryProb']:.4f} | {v['meanFalseBoundaryProb']:.4f} |")
        issue = issues[cap]
        lines += [
            "",
            "### Dominant issue",
            "",
            f"- findings: {', '.join(issue['findings']) or 'none triggered'}",
            f"- confidence skew (mean predicted − base rate): {issue['confidenceSkew']:+.4f}",
            f"- true/false separation: {issue['trueFalseSeparation']:.4f}",
            f"- F1 at 0.50: {issue['f1AtHalf']:.4f}; best F1 {issue['bestF1']:.4f} at {issue['bestThreshold']}",
            "",
        ]

    if raw is not None and raw_issues is not None:
        lines += build_smoothing_section(diagnoses, raw, raw_issues)

    lines += [
        "## Microphone/pickup consistency",
        "",
        f"- shared performances: {disagreement['sharedPerformances']}",
        f"- switch sites: {disagreement['switchSites']}",
        f"- capture switch agreement: {disagreement['captureSwitchAgreement']:.4f}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Boundary-calibration Phase 4-5 diagnostics.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    parser.add_argument("--equality-sample", type=int, default=6)
    args = parser.parse_args()

    reconciliation = reconcile(cache_dir=Path(args.cache_dir), sample=args.equality_sample)
    config, _ = load_candidates(Path(args.candidates))
    smoothing = next(c for c in config["candidates"] if c["id"] == "full-v2-existing").get("smoothing")

    entries = load_cache_entries(Path(args.cache_dir))
    diagnoses = {cap: diagnose_entries([e for e in entries if e.capture == cap], smoothing)
                 for cap in CAPTURES}
    issues = {cap: classify_dominant_issue(diagnoses[cap]) for cap in CAPTURES}
    disagreement = capture_disagreement(diagnoses[CAPTURES[0]], diagnoses[CAPTURES[1]])

    # The full-v2 path smooths the boundary channel (EMA, boundaryAlpha 0.35)
    # before anything sees it. Diagnosing only the smoothed channel cannot tell
    # a weak head apart from a head degraded downstream, so the raw responses
    # are diagnosed alongside it and the two are reported together.
    raw = {cap: diagnose_entries([e for e in entries if e.capture == cap], None)
           for cap in CAPTURES}
    raw_issues = {cap: classify_dominant_issue(raw[cap]) for cap in CAPTURES}

    report_dir = Path(args.report_dir)
    _write(report_dir / "boundary-v3-diagnosis.md",
           build_markdown(reconciliation, diagnoses, issues, disagreement, raw, raw_issues))
    _write(report_dir / "boundary-v3-diagnosis.json", json.dumps(_strip_private({
        "reconciliation": reconciliation,
        "smoothing": smoothing,
        "tolerance": 0.25,
        "diagnoses": diagnoses,
        "rawChannelDiagnoses": raw,
        "rawChannelDominantIssues": raw_issues,
        "dominantIssues": issues,
        "captureConsistency": disagreement,
        "p00Accessed": False,
    }), indent=2))

    print(f"reconciled: {reconciliation['reconciled']}")
    for cap in CAPTURES:
        print(f"{cap}: findings={issues[cap]['findings']} "
              f"ECE={diagnoses[cap]['calibration']['expectedCalibrationError']:.4f} "
              f"bestF1={issues[cap]['bestF1']:.4f}@{issues[cap]['bestThreshold']}")


if __name__ == "__main__":
    main()
