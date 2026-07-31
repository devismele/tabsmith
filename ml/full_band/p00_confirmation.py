"""The single confirmatory measurement on the sealed performer p00.

Every preservation number in this workstream is measured on p01-p05, which v1
trained on and which the rehearsal-heavy model rehearsed on further. Those are
forgetting checks. p00 has never been trained on, evaluated on, or selected
against by anything here, so it is the one place a generalisation claim can be
tested honestly - and only once.

The protocol is frozen in ``ml/configs/p00-confirmation-v1.json`` before p00 is
read: one subject, one baseline, criteria declared in advance, and no
alternatives scored. Running this a second time with a different candidate would
turn the held-out performer into a selection set, so the subject and baseline are
read from the frozen file rather than passed on the command line.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..evaluation.metrics import evaluate_regions
from ..evaluation.segmental.priors import derive_priors, params_for_candidate
from ..evaluation.temporal_v2_ablation import (
    _capture,
    _extract_feature_map,
    _load_tracks,
    _resolve_data_path,
)
from .decoder_study import _reference_regions, evaluate, guitarset_responses
from .evaluate_pilot import CAPTURES, _summarise, file_digest

ML_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_ROOT.parent
REPORT_DIR = REPO_ROOT / "evaluation" / "reports"
PROTOCOL = ML_ROOT / "configs" / "p00-confirmation-v1.json"
GATES = ML_ROOT / "configs" / "full-band-pilot-gates.json"
STUDY = ML_ROOT / "configs" / "full-band-decoder-v1.json"
SEALED = "guitarset-p00"


def load_sealed_tracks(annotation_dir: Path, audio_dirs: dict[str, Path]) -> list[Any]:
    """Load p00, which the shared loader deliberately refuses to return.

    ``_load_tracks`` strips p00 before feature extraction so no study can reach
    it by accident - that seal protects every other pipeline and is left intact.
    Unsealing is therefore explicit and lives here, in the one script whose
    protocol was frozen and committed before p00 was read.
    """
    from ..preprocessing.import_guitarset import import_guitarset_track

    suffixes = {"audio_mono-mic": "_mic.wav", "audio_mono-pickup_mix": "_mix.wav"}
    tracks = []
    for jams_path in sorted(Path(annotation_dir).rglob("*.jams")):
        for capture, suffix in suffixes.items():
            audio_path = Path(audio_dirs[capture]) / f"{jams_path.stem}{suffix}"
            if not audio_path.exists():
                continue
            track = import_guitarset_track(jams_path, audio_dir=None)
            if track.artist != SEALED:
                continue
            track.audio_availability = "audio"
            track.audio_path = str(audio_path)
            track.track_id = f"{track.track_id}@{capture}"
            track.notes = f"{track.notes} capture={capture}"
            track.validate()
            tracks.append(track)
    if any(t.artist != SEALED for t in tracks):
        raise AssertionError("only p00 may be loaded here")
    return tracks


def compare(baseline: dict[str, Any], subject: dict[str, Any],
            tolerances: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen preservation tolerances, unchanged, on p00."""
    checks: list[dict[str, Any]] = []
    for capture in CAPTURES:
        base, cand = baseline.get(capture), subject.get(capture)
        if not base or not cand:
            continue
        root_drop = (base["rootAccuracy"] - cand["rootAccuracy"]) * 100.0
        detailed_drop = (base["detailedAccuracy"] - cand["detailedAccuracy"]) * 100.0
        frag_rise = cand["fragmentationRate"] - base["fragmentationRate"]
        rpm_rise = cand["regionsPerMinute"] - base["regionsPerMinute"]
        checks += [
            {"check": "rootRegression", "capture": capture, "measured": round(root_drop, 3),
             "required": f"<= {tolerances['rootAccuracyRegressionMaximumPercentagePoints']} pp",
             "passed": root_drop <= tolerances["rootAccuracyRegressionMaximumPercentagePoints"]},
            {"check": "detailedRegression", "capture": capture,
             "measured": round(detailed_drop, 3),
             "required": f"<= {tolerances['detailedAccuracyRegressionMaximumPercentagePoints']} pp",
             "passed": detailed_drop
             <= tolerances["detailedAccuracyRegressionMaximumPercentagePoints"]},
            {"check": "fragmentationIncrease", "capture": capture,
             "measured": round(frag_rise, 4),
             "required": f"<= {tolerances['fragmentationRateIncreaseMaximum']}",
             "passed": frag_rise <= tolerances["fragmentationRateIncreaseMaximum"]},
            {"check": "regionsPerMinuteIncrease", "capture": capture,
             "measured": round(rpm_rise, 3),
             "required": f"<= {tolerances['regionsPerMinuteIncreaseMaximum']}",
             "passed": rpm_rise <= tolerances["regionsPerMinuteIncreaseMaximum"]},
        ]
    failed = [c for c in checks if not c["passed"]]
    improved_everywhere = all(
        subject[c]["rootAccuracy"] > baseline[c]["rootAccuracy"]
        and subject[c]["detailedAccuracy"] > baseline[c]["detailedAccuracy"]
        for c in CAPTURES if c in baseline and c in subject)
    return {
        "checks": checks,
        "confirmed": not failed,
        "stronglyConfirmed": (not failed) and improved_everywhere,
        "failedChecks": [f"{c['check']}@{c['capture']}" for c in failed],
        "outcome": ("strongly-confirmed" if (not failed and improved_everywhere)
                    else "confirmed" if not failed else "refuted"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen single confirmatory measurement on sealed performer p00.")
    parser.add_argument("--protocol", default=str(PROTOCOL))
    parser.add_argument("--v1-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--mic-audio", default=None)
    parser.add_argument("--pickup-audio", default=None)
    parser.add_argument("--output", default=str(REPORT_DIR / "p00-confirmation-v1-results.json"))
    parser.add_argument("--i-understand-p00-is-spent-after-this", action="store_true",
                        required=True,
                        help="Acknowledges that this measurement consumes the sealed holdout.")
    args = parser.parse_args()

    from ..evaluation.adapters import predict_ml
    from ..training.checkpoint import load_checkpoint

    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    gates = json.loads(GATES.read_text(encoding="utf-8"))
    study = json.loads(STUDY.read_text(encoding="utf-8"))
    tolerances = gates["preservationGatesVsV1GuitarSetDevelopment"]

    existing = Path(args.output)
    if existing.exists():
        raise SystemExit(
            f"{existing.name} already exists. p00 is measured once; re-running it after "
            "seeing the result would turn the sealed performer into a selection set.")

    decoder = next(c for c in study["candidates"] if c["id"] == "duration-viterbi")
    print(f"protocol: {protocol['confirmationId']}", flush=True)
    print(f"subject : {protocol['whatIsMeasured']['subject']}", flush=True)
    print(f"baseline: {protocol['whatIsMeasured']['baseline']}", flush=True)

    annotations = _resolve_data_path(args.annotations, "TABSMITH_GUITARSET_ANNOTATIONS")
    mic = _resolve_data_path(args.mic_audio, "TABSMITH_GUITARSET_MIC_AUDIO")
    pickup = _resolve_data_path(args.pickup_audio, "TABSMITH_GUITARSET_PICKUP_AUDIO")
    audio_dirs = {"audio_mono-mic": mic, "audio_mono-pickup_mix": pickup}

    development = [t for t in _load_tracks(annotations, audio_dirs)
                   if t.artist in set(gates["domain"]["guitarSetDevelopmentPerformers"])]
    sealed = load_sealed_tracks(annotations, audio_dirs)
    if not sealed:
        raise SystemExit("no p00 tracks found; nothing to confirm")
    print(f"unsealing p00: {len(sealed)} tracks (development set {len(development)})", flush=True)

    v1_model, meta = load_checkpoint(Path(args.v1_checkpoint))
    candidate_model, _ = load_checkpoint(Path(args.candidate_checkpoint))
    pipeline = meta.get("featureVersion", "numpy-chroma-v1")

    # Priors come from p01-p05 only: nothing p00 implies may shape its own decoder.
    print("deriving decoder priors from p01-p05 only...", flush=True)
    dev_features = _extract_feature_map(development, pipeline)
    dev_responses = guitarset_responses(candidate_model, development, dev_features)
    params = params_for_candidate(decoder["kind"], decoder["knobs"], derive_priors(dev_responses))

    print("scoring p00...", flush=True)
    sealed_features = _extract_feature_map(sealed, pipeline)

    baseline_rows: dict[str, list[dict]] = {}
    for track in sealed:
        frames = sealed_features.get(track.track_id)
        if frames is None:
            continue
        predicted = predict_ml(v1_model, frames, transition_penalty=4.0)
        if predicted:
            baseline_rows.setdefault(_capture(track), []).append(
                evaluate_regions(track.chords, predicted, tolerances=(0.1, 0.25, 0.5, 1.0)))
    baseline = {c: _summarise(rows) for c, rows in baseline_rows.items() if rows}

    sealed_responses = guitarset_responses(candidate_model, sealed, sealed_features)
    subject = evaluate(sealed_responses, decoder["kind"], lambda _r: params)

    verdict = compare(baseline, subject, tolerances)
    payload = {
        "schemaVersion": 1,
        "confirmationId": protocol["confirmationId"],
        "gateSetId": gates["gateSetId"],
        "subject": protocol["whatIsMeasured"]["subject"],
        "baseline": protocol["whatIsMeasured"]["baseline"],
        "decoder": {"id": decoder["id"], "kind": decoder["kind"], "knobs": decoder["knobs"]},
        "candidateChecksum": file_digest(Path(args.candidate_checkpoint)),
        "v1Checksum": file_digest(Path(args.v1_checkpoint)),
        "sealedTracks": len(sealed),
        "priorsFrom": sorted({t.artist for t in development}),
        "results": {"v1": baseline, "candidate": subject},
        "verdict": verdict,
        "p00NowSpent": True,
        "productionWeightsReplaced": False,
        "slakhTestSplitUsed": False,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"outcome": verdict["outcome"],
                      "failed": verdict["failedChecks"]}, indent=2))


if __name__ == "__main__":
    main()
