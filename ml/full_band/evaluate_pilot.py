"""Apply the frozen full-band pilot gates to the trained candidates.

Two halves, both hard:

1. **Full-band development** on the Slakh validation subset (never the test
   split) -- detailed accuracy and no-chord F1 against the v1 baseline.
2. **GuitarSet preservation** on p01-p05, both captures -- regression against v1
   must stay inside the frozen tolerances.

Gates that cannot be evaluated honestly are reported as ``not-evaluable`` with
the reason, never as passed. Two are known in advance: the full-band
fragmentation gate (reference labels are note-level, so fragmentation does not
transfer from GuitarSet) and the percussion-only no-chord-recall gate (scored
against harmonic labels rather than an all-no-chord reference).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from ..evaluation.metrics import evaluate_regions
from ..evaluation.temporal_v2_ablation import (
    _extract_feature_map,
    _load_tracks,
    _resolve_data_path,
)
from .baselines import _features
from .midi import read_note_events
from .symbolic import derive_chord_regions
from .views import build_view

ML_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_ROOT.parent
REPORT_DIR = REPO_ROOT / "evaluation" / "reports"
GATES = ML_ROOT / "configs" / "full-band-pilot-gates.json"
CAPTURES = ("audio_mono-mic", "audio_mono-pickup_mix")

NOT_EVALUABLE = {
    "fullBandFragmentation": (
        "Reference labels are note-level (~0.6 s mean region, ~100 regions/min) versus "
        "~20 for GuitarSet human annotations, so a fragmentation bound calibrated on "
        "GuitarSet does not transfer to them."),
    "percussionOnlyNoChordRecall": (
        "The percussion-only view is scored against each track's harmonic labels rather "
        "than an all-no-chord reference, so this gate does not measure what it was "
        "written to measure."),
}


def _predict(model, features, penalty: float = 4.0):
    from ..evaluation.adapters import predict_ml

    return predict_ml(model, features, transition_penalty=penalty)


def _weighted(rows: list[dict], name: str) -> float:
    usable = [(r.get(name), r.get("evaluatedDurationSeconds", 0.0)) for r in rows]
    usable = [(v, w) for v, w in usable if v is not None and w]
    total = sum(w for _v, w in usable)
    return sum(v * w for v, w in usable) / total if total else 0.0


def _summarise(rows: list[dict]) -> dict[str, float]:
    duration = sum(r.get("evaluatedDurationSeconds", 0.0) for r in rows)
    predicted = sum(r.get("predictedRegions", 0) for r in rows)
    precision = _weighted(rows, "noChordPrecision")
    recall = _weighted(rows, "noChordRecall")
    return {
        "tracks": len(rows),
        "rootAccuracy": round(_weighted(rows, "rootAccuracy"), 4),
        "majorMinorAccuracy": round(_weighted(rows, "majorMinorAccuracy"), 4),
        "detailedAccuracy": round(_weighted(rows, "detailedAccuracy"), 4),
        "noChordPrecision": round(precision, 4),
        "noChordRecall": round(recall, 4),
        "noChordF1": round(2 * precision * recall / (precision + recall), 4)
        if (precision + recall) else 0.0,
        "fragmentationRate": round(_weighted(rows, "fragmentationRate"), 4),
        "regionsPerMinute": round(predicted * 60.0 / duration, 3) if duration else 0.0,
    }


def evaluate_full_band(model, extracted: Path, track_ids: list[str],
                       views: tuple[str, ...], pipeline: str) -> dict[str, Any]:
    """Score one model on the Slakh development subset, per view."""
    by_view: dict[str, list[dict]] = {v: [] for v in views}
    for track_id in track_ids:
        track_dir = extracted / track_id
        midi_path = track_dir / "all_src.mid"
        metadata_path = track_dir / "metadata.yaml"
        if not midi_path.exists() or not metadata_path.exists():
            continue
        events, duration = read_note_events(midi_path.read_bytes())
        if duration <= 0:
            continue
        reference = derive_chord_regions(events, duration)
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8", errors="replace"))
        for view in views:
            built = build_view(track_dir, metadata, view, sample_rate=22050)
            if built is None:
                continue
            audio, rate = built
            predicted = _predict(model, _features(audio, rate, pipeline))
            if predicted:
                by_view[view].append(
                    evaluate_regions(reference, predicted, tolerances=(0.1, 0.25, 0.5, 1.0)))
    return {view: _summarise(rows) for view, rows in by_view.items() if rows}


def evaluate_guitarset(model, tracks, feature_map, capture_of) -> dict[str, Any]:
    """Score one model on GuitarSet, split by capture."""
    by_capture: dict[str, list[dict]] = {c: [] for c in CAPTURES}
    for track in tracks:
        frames = feature_map.get(track.track_id)
        if frames is None:
            continue
        predicted = _predict(model, frames)
        if not predicted:
            continue
        by_capture[capture_of(track)].append(
            evaluate_regions(track.chords, predicted, tolerances=(0.1, 0.25, 0.5, 1.0)))
    return {c: _summarise(rows) for c, rows in by_capture.items() if rows}


def file_digest(path: Path) -> str:
    """SHA-256 of a checkpoint, so cached metrics belong to specific weights."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cached_evaluation(cache_dir: Path | None, name: str, key: str, build):
    """Evaluate one model once, and keep it across an interrupted evaluation.

    Scoring every model takes far longer than one uninterrupted window, and the
    results were previously written only at the very end, so a stop discarded
    all of it. Metrics are cached per model under a key that includes the
    checkpoint digest: different weights are re-evaluated rather than read from
    a stale entry.
    """
    if cache_dir is None:
        return build()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}.json"
    if path.exists():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        if recorded.get("key") == key:
            print(f"  cache hit: {name}", flush=True)
            return recorded["value"]
        print(f"  cache stale: {name} (weights or inputs changed); re-evaluating", flush=True)
    value = build()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"key": key, "value": value}, indent=2), encoding="utf-8")
    tmp.replace(path)
    return value


def resolve_results_path(report_dir: Path, output: str | None, pilot_id: str) -> Path:
    """Where this pilot's results go, without clobbering another pilot's.

    The default filename predates pilot v2. A later pilot writing to it would
    overwrite the frozen results of an earlier one - the record that pilot's
    decision was read from - so a results file belonging to a different pilot is
    refused rather than replaced.
    """
    path = Path(output) if output else Path(report_dir) / "full-band-pilot-results.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8")).get("pilotId")
        except json.JSONDecodeError:
            existing = None
        if existing and existing != pilot_id:
            raise SystemExit(
                f"{path.name} holds results for {existing}, not {pilot_id}; "
                "pass --output to keep each pilot's frozen results intact")
    return path


def apply_gates(gates: dict[str, Any], baseline: dict[str, Any],
                candidate: dict[str, Any]) -> dict[str, Any]:
    """Apply both halves of the frozen gate set."""
    checks: list[dict[str, Any]] = []
    preservation = gates["preservationGatesVsV1GuitarSetDevelopment"]
    development = gates["fullBandDevelopmentGates"]

    for capture in CAPTURES:
        base = baseline["guitarset"].get(capture)
        cand = candidate["guitarset"].get(capture)
        if not base or not cand:
            continue
        root_drop = (base["rootAccuracy"] - cand["rootAccuracy"]) * 100.0
        detailed_drop = (base["detailedAccuracy"] - cand["detailedAccuracy"]) * 100.0
        frag_rise = cand["fragmentationRate"] - base["fragmentationRate"]
        rpm_rise = cand["regionsPerMinute"] - base["regionsPerMinute"]
        checks += [
            {"gate": "guitarSetRootRegression", "scope": capture,
             "measured": round(root_drop, 3),
             "required": f"<= {preservation['rootAccuracyRegressionMaximumPercentagePoints']} pp",
             "passed": root_drop <= preservation["rootAccuracyRegressionMaximumPercentagePoints"]},
            {"gate": "guitarSetDetailedRegression", "scope": capture,
             "measured": round(detailed_drop, 3),
             "required": f"<= {preservation['detailedAccuracyRegressionMaximumPercentagePoints']} pp",
             "passed": detailed_drop <= preservation["detailedAccuracyRegressionMaximumPercentagePoints"]},
            {"gate": "guitarSetFragmentationIncrease", "scope": capture,
             "measured": round(frag_rise, 4),
             "required": f"<= {preservation['fragmentationRateIncreaseMaximum']}",
             "passed": frag_rise <= preservation["fragmentationRateIncreaseMaximum"]},
            {"gate": "guitarSetRegionsPerMinuteIncrease", "scope": capture,
             "measured": round(rpm_rise, 3),
             "required": f"<= {preservation['regionsPerMinuteIncreaseMaximum']}",
             "passed": rpm_rise <= preservation["regionsPerMinuteIncreaseMaximum"]},
        ]

    base_mix = baseline["fullBand"].get("full-mix")
    cand_mix = candidate["fullBand"].get("full-mix")
    if base_mix and cand_mix:
        gain = (cand_mix["detailedAccuracy"] - base_mix["detailedAccuracy"]) * 100.0
        nc_gain = cand_mix["noChordF1"] - base_mix["noChordF1"]
        checks += [
            {"gate": "fullBandDetailedImprovement", "scope": "full-mix",
             "measured": round(gain, 3),
             "required": f">= {development['detailedAccuracyImprovementMinimumPercentagePoints']} pp",
             "passed": gain >= development["detailedAccuracyImprovementMinimumPercentagePoints"]},
            {"gate": "fullBandNoChordF1Improvement", "scope": "full-mix",
             "measured": round(nc_gain, 4),
             "required": f">= {development['noChordF1ImprovementMinimum']}",
             "passed": nc_gain >= development["noChordF1ImprovementMinimum"]},
            {"gate": "mustImproveOnFullMix", "scope": "full-mix",
             "measured": round(gain, 3), "required": "> 0 pp", "passed": gain > 0},
        ]

    base_absent = baseline["fullBand"].get("guitar-absent-harmonic")
    cand_absent = candidate["fullBand"].get("guitar-absent-harmonic")
    if cand_absent and cand_mix:
        floor = development["guitarAbsentDetailedAccuracyFloorRelativeToGuitarPresent"]
        ratio = (cand_absent["detailedAccuracy"] / cand_mix["detailedAccuracy"]
                 if cand_mix["detailedAccuracy"] else 0.0)
        checks.append({
            "gate": "guitarAbsentNoCollapse", "scope": "guitar-absent-harmonic",
            "measured": round(ratio, 4), "required": f">= {floor} of full-mix",
            "passed": ratio >= floor})
    if base_absent and cand_absent:
        gain = (cand_absent["detailedAccuracy"] - base_absent["detailedAccuracy"]) * 100.0
        checks.append({
            "gate": "mustImproveOnImperfectSeparatedViews",
            "scope": "guitar-absent-harmonic", "measured": round(gain, 3),
            "required": "> 0 pp", "passed": gain > 0})

    failed = [c for c in checks if not c["passed"]]
    return {
        "checks": checks,
        "notEvaluable": NOT_EVALUABLE,
        "passed": not failed,
        "failedGates": [f"{c['gate']}@{c['scope']}" for c in failed],
        "firstFailure": (
            f"{failed[0]['gate']} on {failed[0]['scope']}: measured "
            f"{failed[0]['measured']} vs required {failed[0]['required']}" if failed else None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the frozen full-band pilot gates.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--run-dir", default=str(ML_ROOT / "runs" / "full-band-pilot-v1"))
    parser.add_argument("--v1-checkpoint", required=True)
    parser.add_argument("--gates", default=str(GATES))
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    parser.add_argument("--no-eval-cache", action="store_true",
                        help="Re-evaluate every model instead of reusing cached metrics.")
    parser.add_argument("--output", default=None,
                        help="Results file to write. Defaults to the v1 name; pass an "
                             "explicit path for a later pilot so earlier frozen results "
                             "are not overwritten.")
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--mic-audio", default=None)
    parser.add_argument("--pickup-audio", default=None)
    args = parser.parse_args()

    from ..evaluation.temporal_v2_ablation import _base_track_id, _capture, _read_json
    from ..training.checkpoint import load_checkpoint

    root = Path(args.root)
    run_dir = Path(args.run_dir)
    gates = _read_json(args.gates)
    pilot_report = json.loads((run_dir / "pilot-report.json").read_text(encoding="utf-8"))

    v1_model, v1_meta = load_checkpoint(Path(args.v1_checkpoint))
    pipeline = v1_meta.get("featureVersion", "numpy-chroma-v1")

    dev_ids = json.loads((root / "manifests" / "slakh-tracks.json").read_text(encoding="utf-8"))
    size = json.loads((root / "manifests" / "dev-extraction.json").read_text(
        encoding="utf-8"))["requestedTrackIds"]
    seen: set[str] = set()
    chosen: list[str] = []
    for entry in sorted(dev_ids, key=lambda t: (t.get("midiSha256") or "", t["trackId"])):
        if entry.get("officialSplit") != "validation" or entry.get("error"):
            continue
        digest = entry.get("midiSha256")
        if not digest or digest in seen:
            continue
        seen.add(digest)
        chosen.append(entry["trackId"])
        if len(chosen) >= size:
            break

    annotations = _resolve_data_path(args.annotations, "TABSMITH_GUITARSET_ANNOTATIONS")
    mic = _resolve_data_path(args.mic_audio, "TABSMITH_GUITARSET_MIC_AUDIO")
    pickup = _resolve_data_path(args.pickup_audio, "TABSMITH_GUITARSET_PICKUP_AUDIO")
    performers = set(gates["domain"]["guitarSetDevelopmentPerformers"])
    tracks = [t for t in _load_tracks(annotations, {"audio_mono-mic": mic,
                                                    "audio_mono-pickup_mix": pickup})
              if t.artist in performers]
    if any(t.artist == "guitarset-p00" for t in tracks):
        raise SystemExit("p00 must not enter pilot evaluation")
    feature_map = _extract_feature_map(tracks, pipeline)

    views = ("full-mix", "oracle-harmonic", "guitar-absent-harmonic")
    cache_dir = None if args.no_eval_cache else run_dir / "eval-cache"
    shared = {"views": list(views), "pipeline": pipeline, "devTracks": chosen,
              "guitarSetTracks": sorted(t.track_id for t in tracks)}

    def _evaluate(model, digest: str) -> dict[str, Any]:
        return {
            "fullBand": evaluate_full_band(model, root / "extracted", chosen, views, pipeline),
            "guitarset": evaluate_guitarset(model, tracks, feature_map, _capture),
        }

    print("evaluating v1 baseline...", flush=True)
    v1_digest = file_digest(Path(args.v1_checkpoint))
    results = {"v1": cached_evaluation(
        cache_dir, "v1",
        json.dumps({"digest": v1_digest, **shared}, sort_keys=True),
        lambda: _evaluate(v1_model, v1_digest))}

    outcomes = {}
    for entry in pilot_report["results"]:
        cid = entry["candidateId"]
        checkpoint = run_dir / cid / "model.pt"
        if not checkpoint.exists():
            continue
        print(f"evaluating {cid}...", flush=True)
        digest = file_digest(checkpoint)
        results[cid] = cached_evaluation(
            cache_dir, cid,
            json.dumps({"digest": digest, **shared}, sort_keys=True),
            lambda c=checkpoint, d=digest: _evaluate(load_checkpoint(c)[0], d))
        outcomes[cid] = apply_gates(gates, results["v1"], results[cid])

    payload = {
        "schemaVersion": 1,
        "pilotId": pilot_report["pilotId"],
        "gateSetId": gates["gateSetId"],
        "developmentTracks": len(chosen),
        "guitarSetTracks": len(tracks),
        "featurePipeline": pipeline,
        "p00Accessed": False,
        "slakhTestSplitUsed": False,
        "results": results,
        "gateOutcomes": outcomes,
        "eligibleCandidates": [c for c, o in outcomes.items() if o["passed"]],
    }
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    output = resolve_results_path(report_dir, args.output, payload["pilotId"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {output.name}")
    print(json.dumps({"eligible": payload["eligibleCandidates"],
                      **{c: o["firstFailure"] for c, o in outcomes.items()}}, indent=2))


if __name__ == "__main__":
    main()
