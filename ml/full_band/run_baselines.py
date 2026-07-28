"""Phase 12: run engine x source-view baselines on the extracted pilot subset.

Reference labels are re-derived from each track's extracted ``all_src.mid``
using the same deterministic derivation the preparation pass used, so the
labels scored against here are exactly the ones audited there.

Engines are whatever checkpoints the operator points at; a missing checkpoint
drops that engine rather than failing the run, so a partial local checkout
still produces a rule-vs-learned comparison.

Inference only -- nothing here trains, and no production weight or default is
touched.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from .baselines import DEFAULT_VIEWS, aggregate, build_engines, evaluate_track_views, write_report
from .midi import read_note_events
from .symbolic import derive_chord_regions

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "evaluation" / "reports"
DEFAULT_REPORT = REPORT_DIR / "full-band-slakh-preparation.json"


def _reference_regions(track_dir: Path):
    """Re-derive the audited chord labels from the track's aligned MIDI."""
    midi_path = track_dir / "all_src.mid"
    if not midi_path.exists():
        return None, 0.0
    events, duration = read_note_events(midi_path.read_bytes())
    if duration <= 0:
        return None, 0.0
    return derive_chord_regions(events, duration), duration


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full-band source-view baselines.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--views", nargs="*", default=list(DEFAULT_VIEWS))
    parser.add_argument("--v1-checkpoint", default=None)
    parser.add_argument("--full-v2-checkpoint", default=None)
    parser.add_argument("--pipeline", default="harmony-features-v1")
    args = parser.parse_args()

    root = Path(args.root)
    extracted = root / "extracted"
    prep = json.loads(Path(args.report).read_text(encoding="utf-8"))
    pilot = prep["pilotSubset"]
    track_ids = list(pilot["trackIds"])[: args.limit] if args.limit else list(pilot["trackIds"])

    checkpoints = {}
    if args.v1_checkpoint:
        checkpoints["v1-ml"] = Path(args.v1_checkpoint)
        checkpoints["v1-hybrid"] = Path(args.v1_checkpoint)
    if args.full_v2_checkpoint:
        checkpoints["full-v2-ml"] = Path(args.full_v2_checkpoint)
    engines = build_engines(checkpoints)
    print(f"engines: {sorted(engines)}", flush=True)
    print(f"pilot tracks: {len(track_ids)} (subset {pilot['checksum'][:16]}...)", flush=True)

    results = []
    skipped = []
    started = time.time()
    for index, track_id in enumerate(track_ids, 1):
        track_dir = extracted / track_id
        metadata_path = track_dir / "metadata.yaml"
        if not metadata_path.exists():
            skipped.append({"trackId": track_id, "reason": "no metadata"})
            continue
        reference, duration = _reference_regions(track_dir)
        if not reference:
            skipped.append({"trackId": track_id, "reason": "no derivable reference"})
            continue
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8", errors="replace"))
        classes = {s.get("inst_class") for s in (metadata.get("stems") or {}).values()
                   if isinstance(s, dict)}
        results.extend(evaluate_track_views(
            track_dir, metadata, reference, engines,
            track_id=track_id, has_guitar="Guitar" in classes,
            views=tuple(args.views), pipeline=args.pipeline))
        if index % 10 == 0:
            print(f"  {index}/{len(track_ids)} tracks at {time.time() - started:.0f}s", flush=True)

    # Persist the raw rows BEFORE aggregating. The sweep costs ~45 minutes and a
    # failure in summarisation must not throw that away: rows can be re-aggregated
    # offline without touching audio again.
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    rows_path = Path(args.root) / "manifests" / "baseline-rows.json"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.write_text(json.dumps([
        {"engineId": r.engine_id, "view": r.view, "trackId": r.track_id,
         "hasGuitar": r.has_guitar, "runtimeSeconds": r.runtime_seconds,
         "fellBack": r.fell_back, "metrics": r.metrics}
        for r in results
    ], indent=2, default=float), encoding="utf-8")
    print(f"persisted {len(results)} rows to {rows_path.name}", flush=True)

    aggregated = aggregate(results)
    write_report(aggregated, report_dir / "full-band-source-view-baselines.json", extra={
        "pilotChecksum": pilot["checksum"],
        "pilotTracks": len(track_ids),
        "skippedTracks": skipped,
        "featurePipeline": args.pipeline,
        "elapsedSeconds": round(time.time() - started, 1),
        "labelGranularityCaveat": (
            "Reference labels are note-level (mean region ~0.6 s, ~100 regions/min) "
            "versus ~20 regions/min for GuitarSet human annotations. Accuracy and "
            "no-chord metrics are meaningful; fragmentation and regions/minute are "
            "NOT comparable to GuitarSet."
        ),
    })
    print(json.dumps({"scoredRows": aggregated["scoredRows"],
                      "totalFallbacks": aggregated["totalFallbacks"],
                      "skipped": len(skipped),
                      "elapsedSeconds": round(time.time() - started, 1)}, indent=2))


if __name__ == "__main__":
    main()
