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
        "- **v1 transfers to full-band audio better than expected.** On the full mix it",
        "  reaches 0.488 detailed accuracy against the rule engine's 0.519, and it is",
        "  *ahead* on root accuracy (0.638 vs 0.590). The learned model is competitive,",
        "  not outclassed.",
        "- **full-v2 does not transfer.** At 0.278 detailed it is far behind both v1 and",
        "  the rule engine, on its own correct feature pipeline. Whatever it gained on",
        "  GuitarSet does not survive the domain shift.",
        "- **full-v2's no-chord head is effectively inert here** (P/R/F1 all around 0.01",
        "  on every view) while v1 retains ~0.35 F1.",
        "- **Separation is not the bottleneck.** Perfect harmonic separation is worth only",
        "  +0.007 to +0.029 detailed accuracy, and isolating the guitar stem costs every",
        "  engine (-0.05 to -0.17).",
        "",
        "### Correction notice",
        "",
        "An earlier version of this report scored every engine through one global feature",
        "pipeline. v1 was trained on `numpy-chroma-v1` but was being fed",
        "`harmony-features-v1`, which understated it badly: full-mix detailed accuracy was",
        "reported as 0.255 when it is actually 0.488, and root as 0.419 when it is 0.638.",
        "Each engine now uses the pipeline recorded in its own checkpoint.",
        "",
        "The earlier conclusion that 'the rule engine is roughly twice as accurate as every",
        "learned engine' was an artifact of that bug and is withdrawn. The corrected gap",
        "between the rule engine and v1 on the full mix is 0.031 detailed accuracy, not",
        "0.264 - which materially weakens the motivation for a full-band fine-tune, since",
        "there is far less headroom than the original numbers implied.",
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
