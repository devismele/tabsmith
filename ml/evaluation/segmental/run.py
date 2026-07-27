"""Run the segmental-v3 development evaluation and write Phase 12 reports.

Reports are portable: no absolute paths, no frame-level dumps. Reports land under
``evaluation/reports`` (tracked); the cache stays under ``ml/runs`` (ignored).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .diagnostics import diagnose_capture
from .evaluate import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CANDIDATES,
    load_cache_entries,
    load_candidates,
    run_evaluation,
)
from .seg_metrics import CAPTURES

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


def _fmt(m: dict[str, Any]) -> str:
    return (f"{m['rootAccuracy']:.4f} | {m['detailedAccuracy']:.4f} | {m['fragmentationRate']:.4f} | "
            f"{m['regionsPerMinute']:.3f} | {m['meanAbsoluteBoundaryErrorMs']:.1f} | "
            f"{m['flickerCount']:.0f} | {m['boundaryF1At250ms']:.4f}")


def write_reports(evaluation: dict[str, Any], diagnostics: dict[str, Any],
                  candidates_config: dict, candidate_checksum: str,
                  report_dir: Path = REPORT_DIR) -> list[Path]:
    written: list[Path] = []
    results = evaluation["results"]

    # 1. diagnostics.md
    lines = ["# Segmental-v3 over-segmentation diagnosis (full-v2)", "",
             "Faithful full-v2 decoder (EMA smoothing + penalty-4 Viterbi) on p01-p05 held-out captures.",
             "p00 sealed. Frozen full-v2 reference: mic rpm 22.21 / frag 0.6719; pickup rpm 22.42 / frag 0.6789;",
             "v1-objective reference: mic rpm 19.857 / frag 0.6167; pickup rpm 19.910 / frag 0.6169.", ""]
    for cap in CAPTURES:
        d = diagnostics[cap]
        lines += [f"## {cap}", "",
                  f"- extra regions vs reference: {d['extraRegions']}",
                  f"- boundary-head P/R/F1: {d['boundaryHead']['precision']} / {d['boundaryHead']['recall']} / {d['boundaryHead']['f1']}",
                  f"- state-change P/R/F1: {d['stateChange']['precision']} / {d['stateChange']['recall']} / {d['stateChange']['f1']}",
                  f"- state-change/boundary agreement: {d['stateChangeBoundaryAgreement']}",
                  f"- taxonomy: {json.dumps(d['taxonomy'])}",
                  f"- evidence before FALSE transition: {json.dumps(d['evidenceBeforeFalseTransition'])}",
                  f"- evidence before TRUE transition: {json.dumps(d['evidenceBeforeTrueTransition'])}", ""]
    _write(report_dir / "segmental-v3-diagnostics.md", "\n".join(lines))
    written.append(report_dir / "segmental-v3-diagnostics.md")

    # 2. candidates.json (frozen defs + checksum + derived priors)
    _write(report_dir / "segmental-v3-candidates.json", json.dumps({
        "candidateSetId": candidates_config["candidateSetId"],
        "candidateChecksum": candidate_checksum,
        "parameterPolicy": candidates_config["parameterPolicy"],
        "candidates": candidates_config["candidates"],
        "foldPriors": evaluation["priors"],
        "gates": evaluation["gates"],
    }, indent=2))
    written.append(report_dir / "segmental-v3-candidates.json")

    # 3. fold-results.csv
    csv_path = report_dir / "segmental-v3-fold-results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["candidateId", "foldId", "capture", "rootAccuracy", "detailedAccuracy",
                    "fragmentationRate", "regionsPerMinute", "predictedRegionCount",
                    "meanAbsoluteBoundaryErrorMs", "flickerCount", "boundaryF1At250ms",
                    "stateSwitchPrecision", "stateSwitchRecall"])
        for cid, res in results.items():
            for fold, caps in res["perFold"].items():
                for cap, m in caps.items():
                    w.writerow([cid, fold, cap, f"{m['rootAccuracy']:.4f}", f"{m['detailedAccuracy']:.4f}",
                                f"{m['fragmentationRate']:.4f}", f"{m['regionsPerMinute']:.3f}",
                                int(m['predictedRegionCount']), f"{m['meanAbsoluteBoundaryErrorMs']:.1f}",
                                int(m['flickerCount']), f"{m['boundaryF1At250ms']:.4f}",
                                f"{m['stateSwitchPrecision']:.4f}", f"{m['stateSwitchRecall']:.4f}"])
    written.append(csv_path)

    # 4. pareto.md (aggregate table + frontier + CIs)
    lines = ["# Segmental-v3 candidate comparison (p01-p05 development)", "",
             "root | detailed | frag | rpm | MAE ms | flicker | bF1@250 per capture.", ""]
    for cap in CAPTURES:
        lines += [f"## {cap}", "",
                  "| candidate | root | detailed | frag | rpm | MAE ms | flicker | bF1@250 |",
                  "|---|---|---|---|---|---|---|---|"]
        for cid, res in results.items():
            lines.append(f"| {cid} | {_fmt(res['byCapture'][cap])} |")
        lines.append("")
        lines.append("95% performer-paired bootstrap CIs (root / rpm / frag):")
        lines.append("")
        for cid, res in results.items():
            ci = res["confidenceIntervals"][cap]
            lines.append(f"- {cid}: root [{ci['rootAccuracy']['lo']:.4f}, {ci['rootAccuracy']['hi']:.4f}] "
                         f"rpm [{ci['regionsPerMinute']['lo']:.3f}, {ci['regionsPerMinute']['hi']:.3f}] "
                         f"frag [{ci['fragmentationRate']['lo']:.4f}, {ci['fragmentationRate']['hi']:.4f}]")
        lines.append("")
    lines += ["## Pareto frontier", "", ", ".join(evaluation["pareto"]), "",
              "## Runtime", ""]
    for cid, res in results.items():
        lines.append(f"- {cid}: {res['runtimePerAudioMinute']:.4f} s/audio-min "
                     f"({res['runtimeSeconds']:.2f}s total)")
    _write(report_dir / "segmental-v3-pareto.md", "\n".join(lines))
    written.append(report_dir / "segmental-v3-pareto.md")

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Run segmental-v3 development evaluation and reports.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    args = parser.parse_args()

    config, checksum = load_candidates(Path(args.candidates))
    evaluation = run_evaluation(cache_dir=Path(args.cache_dir), candidates_path=Path(args.candidates))
    entries = load_cache_entries(Path(args.cache_dir))
    smoothing = next(c for c in config["candidates"] if c["id"] == "full-v2-existing").get("smoothing")
    diagnostics = {cap: diagnose_capture([e for e in entries if e.capture == cap], smoothing) for cap in CAPTURES}
    written = write_reports(evaluation, diagnostics, config, checksum, Path(args.report_dir))
    print(f"eligible candidates: {evaluation['eligibleCandidates']}")
    print(f"pareto frontier: {evaluation['pareto']}")
    for p in written:
        print(f"wrote {p.name}")


if __name__ == "__main__":
    main()
