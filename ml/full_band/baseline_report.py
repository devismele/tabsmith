"""Render the full-band source-view baseline report as portable markdown.

Reads the persisted baseline JSON so the report can be regenerated without
re-running the sweep, and states the two measurement limitations found on real
data up front rather than burying them under the tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "evaluation" / "reports"
VIEWS = ("full-mix", "oracle-harmonic", "oracle-guitar", "guitar-plus-bass",
         "guitar-absent-harmonic", "percussion-only")
ENGINE_ORDER = ("rule-v3", "v1-ml", "v1-hybrid", "full-v2-ml")


def _assert_portable(text: str) -> str:
    lowered = text.replace("\\", "/").lower()
    for needle in ("c:/users", "/users/devis", str(REPO_ROOT).replace("\\", "/").lower()):
        if needle and needle in lowered:
            raise ValueError(f"absolute path leaked into report: {needle}")
    return text


def build_markdown(payload: dict[str, Any]) -> str:
    aggregate = payload["aggregate"]
    by_view = aggregate["byEngineView"]
    lines = [
        "# Full-band source-view baselines (Slakh2100 pilot)",
        "",
        f"Deterministic pilot subset of {payload['pilotTracks']} compositions "
        f"(subset checksum `{payload['pilotChecksum'][:16]}...`), one rendering per",
        "composition. Inference only; no production weight, default or release gate is",
        f"touched. {aggregate['scoredRows']} scored rows, "
        f"{aggregate['totalFallbacks']} fallbacks, {len(payload['skippedTracks'])} skipped.",
        "",
        "## Two limitations to read first",
        "",
        "**1. Reference labels are note-level, not chord-level.** Mean derived region is",
        "~0.6 s at ~100 regions/minute, against ~20 for GuitarSet human annotations. Root,",
        "quality, detailed and no-chord accuracy remain meaningful. Fragmentation and",
        "regions/minute are **not** comparable to GuitarSet, so the frozen full-band",
        "fragmentation gate cannot honestly be applied to these labels.",
        "",
        "**2. The percussion-only control does not measure what it was meant to.** It is",
        "scored against each track's harmonic labels rather than an all-no-chord reference,",
        "so it does not directly measure harmony hallucinated from rhythm. The frozen",
        "`noChordRecallMinimum: 0.8` gate therefore cannot be evaluated as written. What the",
        "column does show is how many chord regions each engine emits on drums-only audio,",
        "which is indicative but not the gate.",
        "",
        "Both are reported rather than reinterpreted to become passable.",
        "",
        "## Accuracy by engine and source view",
        "",
        "| engine | view | root | maj/min | detailed | no-chord P | R | F1 | regions/min |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for engine in ENGINE_ORDER:
        for view in VIEWS:
            m = by_view.get(engine, {}).get(view)
            if not m:
                continue
            lines.append(
                f"| {engine} | {view} | {m['rootAccuracy']:.3f} | "
                f"{m['majorMinorAccuracy']:.3f} | {m['detailedAccuracy']:.3f} | "
                f"{m['noChordPrecision']:.2f} | {m['noChordRecall']:.2f} | "
                f"{m['noChordF1']:.2f} | {m['regionsPerMinute']:.1f} |")
    lines.append("")

    lines += [
        "## Separation versus oracle (detailed accuracy)",
        "",
        "| engine | full mix | oracle harmonic | gain | oracle guitar | gain |",
        "|---|---|---|---|---|---|",
    ]
    for engine in ENGINE_ORDER:
        g = payload["separationGap"].get(engine)
        if not g:
            continue
        guitar = g["oracleGuitarDetailed"]
        guitar_gain = g["guitarOracleGain"]
        lines.append(
            f"| {engine} | {g['fullMixDetailed']:.3f} | {g['oracleHarmonicDetailed']:.3f} | "
            f"{g['harmonicOracleGain']:+.3f} | "
            f"{guitar:.3f} | {guitar_gain:+.3f} |")
    lines += [
        "",
        "Perfect harmonic separation buys between +0.007 and +0.043 detailed accuracy, and",
        "isolating the guitar stem *costs* every engine (-0.03 to -0.17): a lone guitar has",
        "less harmonic context than the full mix, which still carries bass and other",
        "harmony. **Separation is not the bottleneck on this data; the chord model is.**",
        "",
        "## Guitar presence",
        "",
        "Slakh renders a guitar in 1707 of 1709 usable tracks (99.9%), and all 120 pilot",
        "compositions contain one. A natural guitar-absent comparison is therefore not",
        "available, and the guitar-absent test has to be read from the",
        "`guitar-absent-harmonic` view, which removes guitar stems from every track.",
        "On that view no engine collapses - detailed accuracy is within 0.003 of full mix",
        "for the rule engine and *higher* for the learned engines - so there is no evidence",
        "of guitar dependence.",
        "",
        "## Interpretation",
        "",
        "- The rule engine is roughly twice as accurate as every learned engine on",
        "  full-band audio (0.519 detailed vs 0.255-0.278).",
        "- v1 and full-v2 were trained exclusively on GuitarSet solo guitar and do not",
        "  transfer to dense mixtures. That is the gap a full-band pilot would target.",
        "- full-v2's no-chord head is effectively inert in this domain (P/R/F1 all around",
        "  0.01 on every view) while v1 retains 0.34-0.42 F1. Whatever full-v2 gained on",
        "  GuitarSet, its no-chord behaviour does not survive the domain shift.",
        "",
        "Slakh2100 is rendered from MIDI. All of the above is synthetic-domain evidence.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the full-band baseline report.")
    parser.add_argument("--input", default=str(REPORT_DIR / "full-band-source-view-baselines.json"))
    parser.add_argument("--output", default=str(REPORT_DIR / "full-band-source-view-baselines.md"))
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    Path(args.output).write_text(_assert_portable(build_markdown(payload)), encoding="utf-8")
    print(f"wrote {Path(args.output).name}")


if __name__ == "__main__":
    main()
