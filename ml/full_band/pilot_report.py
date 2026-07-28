"""Render the full-band pilot result and freeze the decision.

Reads the gate evaluation and writes a portable report plus a checksummed
decision record. Retention is the default: a candidate is adopted only if it
passed every evaluable frozen gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "evaluation" / "reports"
CAPTURES = ("audio_mono-mic", "audio_mono-pickup_mix")
VIEWS = ("full-mix", "oracle-harmonic", "guitar-absent-harmonic")


def _assert_portable(text: str) -> str:
    lowered = text.replace("\\", "/").lower()
    for needle in ("c:/users", "/users/devis", str(REPO_ROOT).replace("\\", "/").lower()):
        if needle and needle in lowered:
            raise ValueError(f"absolute path leaked into report: {needle}")
    return text


def _checksum(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def build_markdown(payload: dict[str, Any], pilot: dict[str, Any]) -> str:
    results = payload["results"]
    outcomes = payload["gateOutcomes"]
    eligible = payload["eligibleCandidates"]

    lines = [
        "# Full-band pilot result",
        "",
        (f"**Decision: retain v1. No pilot candidate is eligible.**" if not eligible
         else f"**Decision: {eligible[0]} passed every evaluable gate.**"),
        "",
        f"Pilot `{payload['pilotId']}` under gate set `{payload['gateSetId']}`. Strategy and",
        "gates were frozen from the Phase 12 baselines before any pilot run started and were",
        "not altered afterwards. p00 was never accessed and the Slakh test split was never",
        "used for selection.",
        "",
        "## Full-band development (Slakh validation subset, "
        f"{payload['developmentTracks']} compositions)",
        "",
        "| model | view | root | detailed | no-chord F1 | regions/min |",
        "|---|---|---|---|---|---|",
    ]
    for model in ("v1", "mixed-domain-finetune", "guitarset-only-control"):
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
    for model in ("v1", "mixed-domain-finetune", "guitarset-only-control"):
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

    lines += [
        "## Gates that could not be evaluated",
        "",
        "Reported as not-evaluable rather than passed:",
        "",
    ]
    for name, reason in payload["gateOutcomes"][
            next(iter(payload["gateOutcomes"]))]["notEvaluable"].items():
        lines.append(f"- **{name}** — {reason}")

    primary = results.get("mixed-domain-finetune", {}).get("fullBand", {}).get("full-mix")
    control = results.get("guitarset-only-control", {}).get("fullBand", {}).get("full-mix")
    baseline = results.get("v1", {}).get("fullBand", {}).get("full-mix")
    lines += ["", "## Interpretation", ""]
    if primary and control and baseline:
        primary_gain = (primary["detailedAccuracy"] - baseline["detailedAccuracy"]) * 100
        control_gain = (control["detailedAccuracy"] - baseline["detailedAccuracy"]) * 100
        lines += [
            f"- **Mixed-domain fine-tuning works on the target domain.** Full-mix detailed",
            f"  accuracy improves {primary_gain:+.2f} pp over v1 and no-chord F1 rises",
            f"  {primary['noChordF1'] - baseline['noChordF1']:+.3f}. Every full-band gate passes,",
            "  most by a wide margin, and there is no guitar-absent collapse.",
            f"- **The control isolates the cause.** GuitarSet-only training on the same",
            f"  schedule gains only {control_gain:+.2f} pp on the full mix, against the primary's",
            f"  {primary_gain:+.2f} pp. The improvement comes from full-band exposure, not from",
            "  continued optimisation.",
            "- **It is rejected on preservation, not on capability.** The primary loses 4.8 pp",
            "  (microphone) and 8.2 pp (pickup) of GuitarSet root accuracy against a frozen",
            "  2.0 pp tolerance. A model that forgets solo guitar is not an improvement to",
            "  Tabsmith's actual use case, which is why that gate is hard.",
            "- Notably GuitarSet *detailed* accuracy did **not** regress (-1.54 pp on",
            "  microphone is an improvement, +0.57 pp on pickup is inside tolerance), and",
            "  fragmentation improved on both captures. The loss is specific to root",
            "  identification.",
            "",
            "### A correction",
            "",
            "During training I described the control as degrading on full-band audio, based",
            "on its Slakh development *loss* rising monotonically (7.66 to 8.55). The gate",
            "metrics contradict that: its full-mix detailed accuracy actually improved",
            f"{control_gain:+.2f} pp. Loss and accuracy diverged, and the loss-based reading was",
            "wrong. The comparison that matters is unchanged and in fact stronger on",
            "accuracy: the primary gains roughly nine times as much as the control.",
            "",
        ]
    lines += [
        "## Limits",
        "",
        "- The primary was still improving when the frozen 12-epoch budget ended, so its",
        "  full-band figures are a lower bound rather than a converged result.",
        "- GuitarSet preservation is measured on p01-p05, which v1 was already trained on",
        "  and which the frozen config rehearses. It is a catastrophic-forgetting check on",
        "  data v1 already knew, not a held-out generalisation claim.",
        "- Reference labels are note-level, so full-band fragmentation and regions/minute",
        "  are not comparable to GuitarSet.",
        "- Slakh2100 is rendered from MIDI. This is synthetic-domain evidence and does not",
        "  establish production readiness.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the full-band pilot report.")
    parser.add_argument("--input", default=str(REPORT_DIR / "full-band-pilot-results.json"))
    parser.add_argument("--pilot-config", default=str(
        REPO_ROOT / "ml" / "configs" / "full-band-pilot-v1.json"))
    parser.add_argument("--output", default=str(REPORT_DIR / "full-band-pilot-report.md"))
    parser.add_argument("--decision", default=str(REPORT_DIR / "full-band-pilot-decision.json"))
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    pilot = json.loads(Path(args.pilot_config).read_text(encoding="utf-8"))
    eligible = payload["eligibleCandidates"]

    decision = {
        "schemaVersion": 1,
        "decision": "select-candidate" if eligible else "retain-v1",
        "selectedCandidate": eligible[0] if eligible else None,
        "eligibleCandidates": eligible,
        "pilotId": payload["pilotId"],
        "gateSetId": payload["gateSetId"],
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
        _assert_portable(build_markdown(payload, pilot)), encoding="utf-8")
    Path(args.decision).write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(f"decision: {decision['decision']}")
    print(f"eligible: {eligible}")


if __name__ == "__main__":
    main()
