"""Render the pilot-v2 result and freeze the decision.

The v1 writer hard-codes v1's two candidate names and its narrative, which was
correct for a frozen artifact and is why it is left untouched. This one renders
whatever candidates the evaluation actually produced and takes the interpretive
prose from a file, so the authored part of the report stays authored rather than
being assembled from templates that quietly imply conclusions.

Retention is the default: a candidate is adopted only if it passed every
evaluable frozen gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .pilot_report import REPO_ROOT, REPORT_DIR, _assert_portable, _checksum

CAPTURES = ("audio_mono-mic", "audio_mono-pickup_mix")
VIEWS = ("full-mix", "oracle-harmonic", "guitar-absent-harmonic")


def headline(eligible: list[str]) -> str:
    """The decision line.

    With more than one eligible candidate the gates have admitted several and
    ranked none - they are a filter, not an ordering. Naming the first as if it
    were the selection would be choosing a winner the study never chose, so the
    ambiguity is stated instead.
    """
    if not eligible:
        return "**Decision: retain v1. No candidate is eligible.**"
    if len(eligible) == 1:
        return f"**Decision: {eligible[0]} passed every evaluable gate.**"
    listed = ", ".join(f"`{name}`" for name in eligible)
    return (f"**Decision: {len(eligible)} candidates passed every evaluable gate "
            f"({listed}). The gates admit rather than rank, and no ranking rule was "
            "frozen in advance, so this study does not select between them.**")


def _models(payload: dict[str, Any]) -> list[str]:
    """Baseline first, then candidates in the order they were evaluated."""
    return ["v1"] + [cid for cid in payload["gateOutcomes"] if cid != "v1"]


def build_markdown(payload: dict[str, Any], pilot: dict[str, Any],
                   narrative: str | None = None) -> str:
    results = payload["results"]
    outcomes = payload["gateOutcomes"]
    eligible = payload["eligibleCandidates"]
    models = _models(payload)

    lines = [
        f"# {pilot.get('reportTitle', 'Full-band pilot v2 result')}",
        "",
        headline(eligible),
        "",
        f"Pilot `{payload['pilotId']}` under gate set `{payload['gateSetId']}`.",
        f"Strategy `{pilot['strategy']['id']}`.",
        pilot.get(
            "reportIntro",
            "The candidates were frozen before any run started. The eligibility gates are "
            "the pilot-v1 set reused unchanged - the gate v1 failed on was not allowed to "
            "move. p00 was never accessed and the Slakh test split was never used for "
            "selection."),
        "",
        "## Full-band development (Slakh validation subset, "
        f"{payload['developmentTracks']} compositions)",
        "",
        "| model | view | root | detailed | no-chord F1 | regions/min |",
        "|---|---|---|---|---|---|",
    ]
    for model in models:
        for view in VIEWS:
            m = results.get(model, {}).get("fullBand", {}).get(view)
            if not m:
                continue
            lines.append(f"| {model} | {view} | {m['rootAccuracy']:.4f} | "
                         f"{m['detailedAccuracy']:.4f} | {m['noChordF1']:.3f} | "
                         f"{m['regionsPerMinute']:.2f} |")

    lines += [
        "",
        "## GuitarSet preservation (p01-p05, both captures)",
        "",
        "| model | capture | root | detailed | fragmentation | regions/min |",
        "|---|---|---|---|---|---|",
    ]
    for model in models:
        for capture in CAPTURES:
            m = results.get(model, {}).get("guitarset", {}).get(capture)
            if not m:
                continue
            lines.append(f"| {model} | {capture} | {m['rootAccuracy']:.4f} | "
                         f"{m['detailedAccuracy']:.4f} | {m['fragmentationRate']:.4f} | "
                         f"{m['regionsPerMinute']:.2f} |")

    lines += ["", "## Gate outcomes", ""]
    for cid, outcome in outcomes.items():
        lines += [f"### {cid} — {'PASSED' if outcome['passed'] else 'FAILED'}", "",
                  "| result | gate | scope | measured | required |", "|---|---|---|---|---|"]
        for check in outcome["checks"]:
            mark = "PASS" if check["passed"] else "**FAIL**"
            lines.append(f"| {mark} | {check['gate']} | {check['scope']} | "
                         f"{check['measured']} | {check['required']} |")
        lines.append("")

    lines += ["## Gates that could not be evaluated", "",
              "Reported as not-evaluable rather than passed:", ""]
    for name, reason in outcomes[next(iter(outcomes))]["notEvaluable"].items():
        lines.append(f"- **{name}** — {reason}")

    if narrative:
        lines += ["", narrative.strip(), ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the full-band pilot-v2 report.")
    parser.add_argument("--input", default=str(REPORT_DIR / "full-band-pilot-v2-results.json"))
    parser.add_argument("--pilot-config", default=str(
        REPO_ROOT / "ml" / "configs" / "full-band-pilot-v2.json"))
    parser.add_argument("--narrative", default=None,
                        help="Markdown fragment appended after the gate tables.")
    parser.add_argument("--output", default=str(REPORT_DIR / "full-band-pilot-v2-report.md"))
    parser.add_argument("--decision", default=str(
        REPORT_DIR / "full-band-pilot-v2-decision.json"))
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    pilot = json.loads(Path(args.pilot_config).read_text(encoding="utf-8"))
    narrative = Path(args.narrative).read_text(encoding="utf-8") if args.narrative else None
    eligible = payload["eligibleCandidates"]

    if payload["gateSetId"] != pilot["gateSet"]:
        raise SystemExit(
            f"results were scored under {payload['gateSetId']} but the pilot froze "
            f"{pilot['gateSet']}; refusing to report a mismatched gate set")

    decision = {
        "schemaVersion": 1,
        "decision": ("retain-v1" if not eligible
                     else "select-candidate" if len(eligible) == 1
                     else "multiple-eligible-no-ranking-rule"),
        # Only a single eligible candidate is a selection. Several eligible
        # candidates with no pre-frozen ranking rule is an unresolved choice, and
        # recording one of them here would make it look decided.
        "selectedCandidate": eligible[0] if len(eligible) == 1 else None,
        "eligibleCandidates": eligible,
        "pilotId": payload["pilotId"],
        "gateSetId": payload["gateSetId"],
        "gateSetReusedFromPilotV1": True,
        "resultsChecksum": _checksum(payload["results"]),
        "pilotConfigChecksum": _checksum(pilot),
        "p00Accessed": False,
        "slakhTestSplitUsed": False,
        "productionWeightsReplaced": False,
        "applicationDefaultsChanged": False,
        "releaseGatingChanged": False,
        "gateOutcomes": {cid: {"passed": o["passed"], "failedGates": o["failedGates"],
                               "firstFailure": o["firstFailure"]}
                         for cid, o in payload["gateOutcomes"].items()},
        "notEvaluableGates": list(
            payload["gateOutcomes"][next(iter(payload["gateOutcomes"]))]["notEvaluable"]),
    }
    Path(args.output).write_text(
        _assert_portable(build_markdown(payload, pilot, narrative)), encoding="utf-8")
    Path(args.decision).write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(f"decision: {decision['decision']}")
    print(f"eligible: {eligible}")


if __name__ == "__main__":
    main()
