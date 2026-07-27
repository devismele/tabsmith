"""Run the frozen boundary-calibration study and write Phase 9 reports.

Evaluates every frozen candidate under both decoders, applies the frozen gates,
and writes portable reports. Requires each candidate's inference cache to exist
(``ml.evaluation.boundary.build_caches`` for the retrained ones); a candidate
whose cache is missing is reported as such rather than silently skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ..segmental.evaluate import load_cache_entries
from ..segmental.seg_metrics import CAPTURES
from ..temporal_v2_ablation import _read_json
from .attribution import (
    boundary_channel_summary,
    chord_channel_summary,
    decompose,
    interpret,
)
from .evaluate import DEFAULT_CANDIDATES, DEFAULT_RUN_DIR, candidate_cache_dir, evaluate_candidate
from .gates import DEFAULT_GATES, evaluate_candidate_gates, load_gates

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = REPO_ROOT / "evaluation" / "reports"
CONTROL_ID = "full-v2-control"
# Gates are applied on the decoder that actually consumes the boundary head.
GATING_DECODER = "segmental-full"


def _assert_portable(text: str) -> str:
    lowered = text.replace("\\", "/").lower()
    for needle in ("c:/users", "/users/devis", str(REPO_ROOT).replace("\\", "/").lower()):
        if needle and needle in lowered:
            raise ValueError(f"absolute path leaked into report: {needle}")
    return text


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_assert_portable(text), encoding="utf-8")


def _row(m: dict[str, Any]) -> str:
    return (f"{m['rootAccuracy']:.4f} | {m['majorMinorAccuracy']:.4f} | {m['detailedAccuracy']:.4f} | "
            f"{m['fragmentationRate']:.4f} | {m['regionsPerMinute']:.3f} | "
            f"{int(m['predictedRegionCount'])} | {m['meanAbsoluteBoundaryErrorMs']:.1f} | "
            f"{int(m['abaCount'])} | {int(m['flickerCount'])}")


def build_attribution_section(attribution: dict[str, Any]) -> list[str]:
    """Separate boundary retraining from decoder and chord-posterior effects."""
    if not attribution:
        return []
    lines = [
        "## Effect attribution: boundary vs decoder vs chord posterior",
        "",
        "The full-v2 decode does not read the boundary head. A model effect visible",
        "under that decoder is therefore chord-posterior movement by definition, and the",
        "extra effect that only appears under segmental-full is the boundary channel:",
        "",
        "```",
        "decoder effect          = control(segmental)   - control(existing)",
        "chord-posterior channel = candidate(existing)  - control(existing)",
        "boundary channel        = [candidate(segmental) - control(segmental)]",
        "                          - chord-posterior channel",
        "```",
        "",
        "Signs are normalised so positive always means better.",
        "",
    ]
    for cid, payload in attribution.items():
        lines += [f"### {cid}", ""]
        for capture in CAPTURES:
            frag = payload["decomposition"][capture]["fragmentationRate"]
            root = payload["decomposition"][capture]["rootAccuracy"]
            finding = payload["interpretation"][capture]
            lines += [
                f"**{capture}**",
                "",
                "| effect | fragmentation | root accuracy |",
                "|---|---|---|",
                f"| decoder (control only) | {frag['decoderEffect']:+.4f} | {root['decoderEffect']:+.4f} |",
                f"| model under full-v2 decode (chord posterior) | {frag['chordPosteriorChannel']:+.4f} | {root['chordPosteriorChannel']:+.4f} |",
                f"| model under segmental-full | {frag['modelEffectUnderSegmental']:+.4f} | {root['modelEffectUnderSegmental']:+.4f} |",
                f"| boundary channel (interaction) | {frag['boundaryChannel']:+.4f} | {root['boundaryChannel']:+.4f} |",
                "",
                f"- total fragmentation improvement: {finding['fragmentationImprovementTotal']:+.4f} "
                f"(**dominant channel: {finding['dominantChannel']}**)",
                f"- boundary head F1 delta: {finding['boundaryF1Delta']:+.4f}",
                "",
            ]
        chord = payload["chordChannel"]
        lines += [
            "Chord posterior compared frame-by-frame against the control "
            f"({chord['comparedCaptures']} captures):",
            "",
            f"- top-1 argmax agreement with control: {chord['top1AgreementWithControl']:.4f}",
            f"- mean posterior margin: control {chord['controlMeanMargin']:.4f} -> "
            f"candidate {chord['candidateMeanMargin']:.4f} ({chord['marginDelta']:+.4f})",
            f"- mean no-chord probability: control {chord['controlMeanNoChord']:.4f} -> "
            f"candidate {chord['candidateMeanNoChord']:.4f}",
            "",
        ]
    return lines


def build_markdown(evaluations: dict[str, Any], gate_results: dict[str, Any],
                   eligible: list[str], gates: dict[str, Any],
                   attribution: dict[str, Any] | None = None) -> str:
    lines = [
        "# Boundary-calibration-v3 candidate comparison (p01-p05 development)",
        "",
        f"Candidate set `{gates['gateSetId']}`. Gates were frozen before any candidate was",
        "trained and were not altered afterwards. p00 was not accessed.",
        "",
        "Each candidate is reported under both decoders. The existing full-v2 decode does",
        "not consume the boundary head, so boundary calibration is inert on that path by",
        "construction; it is shown to keep model effects separable from decoder effects.",
        "",
    ]
    for decoder in ("full-v2-existing", "segmental-full"):
        lines += [f"## Decoder: {decoder}", ""]
        for cap in CAPTURES:
            lines += [
                f"### {cap}", "",
                "| candidate | root | maj/min | detailed | frag | rpm | regions | MAE ms | A→B→A | flicker |",
                "|---|---|---|---|---|---|---|---|---|---|",
            ]
            for cid, ev in evaluations.items():
                if "error" in ev:
                    continue
                lines.append(f"| {cid} | {_row(ev['decoders'][decoder]['byCapture'][cap])} |")
            lines += ["", "Boundary quality (at the fold-selected operating threshold):", "",
                      "| candidate | precision | recall | F1 | agreement | ECE | threshold | false-long | missed |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for cid, ev in evaluations.items():
                if "error" in ev:
                    continue
                bq = ev["decoders"][decoder]["boundaryQuality"][cap]
                tax = ev["decoders"][decoder]["regionTaxonomy"][cap]
                lines.append(
                    f"| {cid} | {bq['precision']:.4f} | {bq['recall']:.4f} | {bq['f1']:.4f} | "
                    f"{bq['stateChangeBoundaryAgreement']:.4f} | {bq['expectedCalibrationError']:.4f} | "
                    f"{bq['meanOperatingThreshold']:.2f} | {tax.get('false_long', 0)} | {tax.get('missed', 0)} |")
            lines += ["", "95% performer-paired bootstrap CIs (root / rpm / frag):", ""]
            for cid, ev in evaluations.items():
                if "error" in ev:
                    continue
                ci = ev["decoders"][decoder]["confidenceIntervals"][cap]
                lines.append(
                    f"- {cid}: root [{ci['rootAccuracy']['lo']:.4f}, {ci['rootAccuracy']['hi']:.4f}] "
                    f"rpm [{ci['regionsPerMinute']['lo']:.3f}, {ci['regionsPerMinute']['hi']:.3f}] "
                    f"frag [{ci['fragmentationRate']['lo']:.4f}, {ci['fragmentationRate']['hi']:.4f}]")
            lines.append("")

    lines += ["## Gate outcomes", "",
              "Applied to each candidate under the segmental-full decoder (the path that",
              "consumes the boundary head).", "",
              "| candidate | eligible | first failure |", "|---|---|---|"]
    for cid, result in gate_results.items():
        lines.append(f"| {cid} | {result['eligible']} | {result['firstFailure'] or '-'} |")
    lines += ["", f"**Eligible candidates: {len(eligible)}** "
                  f"({', '.join(eligible) if eligible else 'none'})", ""]
    lines += build_attribution_section(attribution or {})

    lines += ["## Per-fold calibration fitted on training performers", "",
              "| candidate | decoder / fold | calibrator | threshold | train ECE before | after |",
              "|---|---|---|---|---|---|"]
    for cid, ev in evaluations.items():
        if "error" in ev:
            continue
        for key, record in ev["foldCalibration"].items():
            lines.append(
                f"| {cid} | {key} | {record['calibrator']['name']} | "
                f"{record['operatingThreshold']:.2f} | {record['trainingEceBefore']:.4f} | "
                f"{record['trainingEceAfter']:.4f} |")
    lines.append("")
    return "\n".join(lines)


def write_fold_csv(evaluations: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["candidateId", "decoderId", "foldId", "capture", "rootAccuracy",
                         "majorMinorAccuracy", "detailedAccuracy", "fragmentationRate",
                         "regionsPerMinute", "predictedRegionCount",
                         "meanAbsoluteBoundaryErrorMs", "medianAbsoluteBoundaryErrorMs",
                         "meanSignedBoundaryErrorMs", "boundariesWithin100ms",
                         "boundariesWithin250ms", "boundariesWithin500ms",
                         "boundariesWithin1000ms", "abaCount", "flickerCount",
                         "stateSwitchPrecision", "stateSwitchRecall", "stateSwitchF1"])
        for cid, ev in evaluations.items():
            if "error" in ev:
                continue
            for decoder, res in ev["decoders"].items():
                for fold, caps in res["perFold"].items():
                    for cap, m in caps.items():
                        writer.writerow([
                            cid, decoder, fold, cap,
                            f"{m['rootAccuracy']:.4f}", f"{m['majorMinorAccuracy']:.4f}",
                            f"{m['detailedAccuracy']:.4f}", f"{m['fragmentationRate']:.4f}",
                            f"{m['regionsPerMinute']:.3f}", int(m["predictedRegionCount"]),
                            f"{m['meanAbsoluteBoundaryErrorMs']:.1f}",
                            f"{m['medianAbsoluteBoundaryErrorMs']:.1f}",
                            f"{m['meanSignedBoundaryErrorMs']:.1f}",
                            f"{m['boundariesWithin100ms']:.4f}", f"{m['boundariesWithin250ms']:.4f}",
                            f"{m['boundariesWithin500ms']:.4f}", f"{m['boundariesWithin1000ms']:.4f}",
                            int(m["abaCount"]), int(m["flickerCount"]),
                            f"{m['stateSwitchPrecision']:.4f}", f"{m['stateSwitchRecall']:.4f}",
                            f"{m['stateSwitchF1']:.4f}"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen boundary-calibration study.")
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--gates", default=str(DEFAULT_GATES))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    args = parser.parse_args()

    manifest = _read_json(args.candidates)
    gates = load_gates(Path(args.gates))
    run_dir = Path(args.run_dir).resolve()

    evaluations: dict[str, Any] = {}
    for candidate in manifest["candidates"]:
        cache_dir = candidate_cache_dir(candidate, run_dir)
        if not cache_dir.exists() or not any(cache_dir.glob("*.npz")):
            print(f"skip {candidate['id']}: no inference cache at {cache_dir.name}", flush=True)
            evaluations[candidate["id"]] = {
                "candidateId": candidate["id"],
                "error": f"missing inference cache ({cache_dir.name})",
            }
            continue
        print(f"evaluating {candidate['id']}", flush=True)
        evaluations[candidate["id"]] = evaluate_candidate(candidate, run_dir=run_dir)

    control = evaluations.get(CONTROL_ID)
    if not control or "error" in control:
        raise SystemExit(f"control {CONTROL_ID} is required as the gate baseline")

    control_existing = control["decoders"]["full-v2-existing"]
    # The flicker gate keeps the frozen segmental-v3 convention: the reference is
    # full-v2 under its own decode, because that is the baseline being replaced.
    baseline_full_v2 = control_existing["byCapture"]

    # Boundary quality and false-long counts must come from the control under the
    # SAME decoder the candidate is gated on. The two decoders read different
    # channels -- full-v2 smooths the boundary head, segmental-full does not --
    # so a cross-decoder baseline would compare a smoothed reference against a
    # raw candidate and make the precision gate trivially passable.
    control_gating = control["decoders"][GATING_DECODER]
    baseline_boundary = {
        "precision": {c: control_gating["boundaryQuality"][c]["precision"] for c in CAPTURES},
        "recall": {c: control_gating["boundaryQuality"][c]["recall"] for c in CAPTURES},
        "agreement": {c: control_gating["boundaryQuality"][c]["stateChangeBoundaryAgreement"]
                      for c in CAPTURES},
        "falseLong": {c: control_gating["regionTaxonomy"][c].get("false_long", 0) for c in CAPTURES},
    }

    gate_results: dict[str, Any] = {}
    for cid, ev in evaluations.items():
        if "error" in ev or cid == CONTROL_ID:
            continue
        gate_results[cid] = evaluate_candidate_gates(
            cid, ev["decoders"][GATING_DECODER], gates=gates,
            baseline_full_v2=baseline_full_v2, baseline_boundary=baseline_boundary)
    eligible = [cid for cid, r in gate_results.items() if r["eligible"]]

    # Attribution: which channel actually moved, boundary or chord posterior?
    control_entries = load_cache_entries(candidate_cache_dir(
        next(c for c in manifest["candidates"] if c["id"] == CONTROL_ID), run_dir))
    attribution: dict[str, Any] = {}
    for cid, ev in evaluations.items():
        if "error" in ev or cid == CONTROL_ID:
            continue
        candidate_def = next(c for c in manifest["candidates"] if c["id"] == cid)
        decomposition = decompose(control["decoders"], ev["decoders"])
        boundary = boundary_channel_summary(control["decoders"], ev["decoders"])
        chord = chord_channel_summary(
            control_entries, load_cache_entries(candidate_cache_dir(candidate_def, run_dir)))
        attribution[cid] = {
            "decomposition": decomposition,
            "boundaryChannel": boundary,
            "chordChannel": chord,
            "interpretation": interpret(decomposition, boundary, chord),
        }

    report_dir = Path(args.report_dir)
    _write(report_dir / "boundary-v3-comparison.md",
           build_markdown(evaluations, gate_results, eligible, gates, attribution))
    write_fold_csv(evaluations, report_dir / "boundary-v3-fold-results.csv")
    _write(report_dir / "boundary-v3-results.json", json.dumps({
        "candidateSetId": manifest["ablationId"],
        "gateSetId": gates["gateSetId"],
        "baselineFullV2": baseline_full_v2,
        "baselineBoundary": baseline_boundary,
        "evaluations": evaluations,
        "gateResults": gate_results,
        "attribution": attribution,
        "eligibleCandidates": eligible,
        "p00Accessed": False,
    }, indent=2, default=float))

    print(f"eligible candidates: {eligible}")
    for cid, r in gate_results.items():
        print(f"  {cid}: eligible={r['eligible']} first_failure={r['firstFailure']}")


if __name__ == "__main__":
    main()
