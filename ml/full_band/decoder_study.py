"""Over-segmentation control on the rehearsal-heavy mixed-domain model.

Pilot v2 produced a model that beats v1 on full-band audio *and* on solo guitar
in every accuracy metric, and fails only the two segmentation gates. This study
varies the decoder and nothing else.

The weights are never touched. Each item is run through the network once and the
response is cached, then every candidate decodes that identical response - so
the decoder axis is isolated by construction rather than by assumption.

Parameter policy, which is the part that decides whether the result means
anything: the scalar dials are copied verbatim from the segmental-v3 frozen
configuration and are not re-tuned here, and the data-derived parameters come
from data that is never the set being scored - leave-one-performer-out for
GuitarSet, and the Slakh training compositions for full-band.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..evaluation.metrics import evaluate_regions
from ..evaluation.segmental.decoders import DecoderParams
from ..evaluation.segmental.pipeline import decode_regions
from ..evaluation.segmental.priors import derive_priors, params_for_candidate
from ..evaluation.temporal_v2_ablation import _extract_feature_map, _load_tracks, _resolve_data_path
from .evaluate_pilot import CAPTURES, NOT_EVALUABLE, _summarise, apply_gates, file_digest
from .pilot import cache_key, cached_samples

ML_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_ROOT.parent
REPORT_DIR = REPO_ROOT / "evaluation" / "reports"
GATES = ML_ROOT / "configs" / "full-band-pilot-gates.json"
STUDY = ML_ROOT / "configs" / "full-band-decoder-v1.json"
VIEWS = ("full-mix", "oracle-harmonic", "guitar-absent-harmonic")


class Response:
    """One cached network response plus the reference needed to score it.

    Mirrors the segmental study's CacheEntry closely enough for its priors and
    decoders, without importing its fold/dataset identity fields, which do not
    apply to a full-band pilot model.
    """

    __slots__ = ("times", "root", "quality", "nochord", "boundary", "hop_seconds",
                 "reference", "performer_id", "capture", "group")

    def __init__(self, *, times, root, quality, nochord, boundary, hop_seconds,
                 reference, performer_id, capture, group):
        self.times = times
        self.root = root
        self.quality = quality
        self.nochord = nochord
        self.boundary = boundary
        self.hop_seconds = hop_seconds
        self.reference = reference
        self.performer_id = performer_id
        self.capture = capture
        self.group = group


def _regions_as_dicts(regions) -> list[dict[str, Any]]:
    return [{"start": float(r.start), "end": float(r.end), "label": r.label} for r in regions]


def _respond(model, frames, *, reference, performer_id, capture, group) -> Response | None:
    from ..evaluation.adapters import forward_probabilities

    if frames is None or len(frames.times) == 0:
        return None
    root, quality, nochord, boundary = forward_probabilities(model, frames)
    return Response(times=np.asarray(frames.times), root=root, quality=quality,
                    nochord=nochord, boundary=boundary, hop_seconds=frames.hop_seconds,
                    reference=reference, performer_id=performer_id, capture=capture,
                    group=group)


def guitarset_responses(model, tracks, feature_map) -> list[Response]:
    from ..evaluation.temporal_v2_ablation import _capture

    out = []
    for track in tracks:
        response = _respond(model, feature_map.get(track.track_id),
                            reference=_regions_as_dicts(track.chords),
                            performer_id=track.artist, capture=_capture(track),
                            group="guitarset")
        if response is not None:
            out.append(response)
    return out


def slakh_responses(model, extracted: Path, track_ids: list[str], views: tuple[str, ...],
                    pipeline: str, group: str) -> list[Response]:
    import yaml

    from ..preprocessing.app_features import extract_app_features
    from ..preprocessing.features import extract_features
    from .midi import read_note_events
    from .symbolic import derive_chord_regions
    from .views import build_view

    out = []
    for track_id in track_ids:
        track_dir = Path(extracted) / track_id
        midi_path, metadata_path = track_dir / "all_src.mid", track_dir / "metadata.yaml"
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
            frames = (extract_app_features(audio, rate) if pipeline == "harmony-features-v1"
                      else extract_features(audio, rate))
            response = _respond(model, frames, reference=_regions_as_dicts(reference),
                                performer_id="slakh", capture=view, group=group)
            if response is not None:
                out.append(response)
    return out


def guitarset_params(kind: str, knobs: dict, responses: list[Response]) -> dict[str, DecoderParams]:
    """Per-performer parameters derived from the OTHER performers only.

    This is the segmental-v3 policy applied to a study that has no fold
    structure of its own: nothing a performer's own references imply is allowed
    to shape the decoder that scores them.
    """
    performers = sorted({r.performer_id for r in responses})
    params: dict[str, DecoderParams] = {}
    for performer in performers:
        others = [r for r in responses if r.performer_id != performer]
        if not others:
            raise ValueError(f"cannot derive priors for {performer} without other performers")
        params[performer] = params_for_candidate(kind, knobs, derive_priors(others))
    return params


def _reference_regions(response: Response):
    from ..schema import ChordRegion

    return [ChordRegion(float(r["start"]), float(r["end"]), r["label"])
            for r in response.reference]


def evaluate(responses: list[Response], kind: str, params_of) -> dict[str, Any]:
    grouped: dict[str, list[dict]] = {}
    for response in responses:
        predicted = decode_regions(kind, response, params_of(response))
        if not predicted:
            continue
        grouped.setdefault(response.capture, []).append(
            evaluate_regions(_reference_regions(response), predicted,
                             tolerances=(0.1, 0.25, 0.5, 1.0)))
    return {key: _summarise(rows) for key, rows in grouped.items() if rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Decoder study on the rehearsal-heavy model.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--study-config", default=str(STUDY))
    parser.add_argument("--gates", default=str(GATES))
    parser.add_argument("--pilot-run-dir", default=str(ML_ROOT / "runs" / "full-band-pilot-v2"))
    parser.add_argument("--run-dir", default=str(ML_ROOT / "runs" / "full-band-decoder-v1"))
    parser.add_argument("--pilot-results", default=str(
        REPORT_DIR / "full-band-pilot-v2-results.json"))
    parser.add_argument("--output", default=str(REPORT_DIR / "full-band-decoder-v1-results.json"))
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--mic-audio", default=None)
    parser.add_argument("--pickup-audio", default=None)
    args = parser.parse_args()

    from ..training.checkpoint import load_checkpoint

    study = json.loads(Path(args.study_config).read_text(encoding="utf-8"))
    gates = json.loads(Path(args.gates).read_text(encoding="utf-8"))
    pilot = json.loads(Path(args.pilot_results).read_text(encoding="utf-8"))
    if study["gateSet"] != gates["gateSetId"]:
        raise SystemExit("study gate set does not match the gate file")

    root = Path(args.root)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = run_dir / "response-cache"

    checkpoint = REPO_ROOT / study["baseModel"]["checkpoint"]
    if not checkpoint.exists():
        raise SystemExit(f"base model missing: {checkpoint}")
    digest = file_digest(checkpoint)
    model, meta = load_checkpoint(checkpoint)
    pipeline = meta.get("featureVersion", "numpy-chroma-v1")
    print(f"base model: {study['baseModel']['id']} ({pipeline}, {digest[:12]})", flush=True)

    annotations = _resolve_data_path(args.annotations, "TABSMITH_GUITARSET_ANNOTATIONS")
    mic = _resolve_data_path(args.mic_audio, "TABSMITH_GUITARSET_MIC_AUDIO")
    pickup = _resolve_data_path(args.pickup_audio, "TABSMITH_GUITARSET_PICKUP_AUDIO")
    performers = set(gates["domain"]["guitarSetDevelopmentPerformers"])
    tracks = [t for t in _load_tracks(annotations, {"audio_mono-mic": mic,
                                                    "audio_mono-pickup_mix": pickup})
              if t.artist in performers]
    if any(t.artist == "guitarset-p00" for t in tracks):
        raise SystemExit("p00 must not enter the decoder study")

    # Reuse the pilot's GuitarSet feature map: identical key, so this is a hit.
    feature_map = cached_samples(
        Path(args.pilot_run_dir) / "eval-cache", "guitarset-features",
        cache_key(pipeline=pipeline, trackIds=sorted(t.track_id for t in tracks)),
        lambda: _extract_feature_map(tracks, pipeline))

    print("building GuitarSet responses...", flush=True)
    gs = cached_samples(
        cache_dir, "guitarset-responses",
        cache_key(digest=digest, pipeline=pipeline,
                  trackIds=sorted(t.track_id for t in tracks)),
        lambda: guitarset_responses(model, tracks, feature_map))

    dev_ids = list(pilot.get("developmentTrackIds") or [])
    if not dev_ids:
        dev_ids = _dev_track_ids(root, pilot["developmentTracks"])
    print(f"building full-band development responses ({len(dev_ids)} tracks)...", flush=True)
    fb = cached_samples(
        cache_dir, "fullband-responses",
        cache_key(digest=digest, pipeline=pipeline, views=list(VIEWS), trackIds=dev_ids),
        lambda: slakh_responses(model, root / "extracted", dev_ids, VIEWS, pipeline, "fullband"))

    prep = json.loads((REPORT_DIR / "full-band-slakh-preparation.json").read_text(encoding="utf-8"))
    train_ids = list(prep["pilotSubset"]["trackIds"])
    print(f"building full-band prior responses ({len(train_ids)} training tracks)...", flush=True)
    fb_prior = cached_samples(
        cache_dir, "fullband-prior-responses",
        cache_key(digest=digest, pipeline=pipeline, views=["full-mix"], trackIds=train_ids),
        lambda: slakh_responses(model, root / "extracted", train_ids, ("full-mix",),
                                pipeline, "fullband-train"))
    if set(t for t in train_ids) & set(dev_ids):
        raise SystemExit("prior tracks overlap the development set")

    baseline = pilot["results"]["v1"]
    results: dict[str, Any] = {"v1": baseline}
    outcomes: dict[str, Any] = {}
    fb_priors = derive_priors(fb_prior)

    for candidate in study["candidates"]:
        cid, kind, knobs = candidate["id"], candidate["kind"], candidate["knobs"]
        print(f"decoding {cid}...", flush=True)
        gs_params = guitarset_params(kind, knobs, gs)
        fb_params = params_for_candidate(kind, knobs, fb_priors)
        results[cid] = {
            "guitarset": evaluate(gs, kind, lambda r: gs_params[r.performer_id]),
            "fullBand": evaluate(fb, kind, lambda r: fb_params),
            "decoder": {"kind": kind, "knobs": knobs, "role": candidate["role"]},
        }
        outcomes[cid] = apply_gates(gates, baseline, results[cid])
        first = outcomes[cid]["firstFailure"]
        print(f"  {'PASSED' if outcomes[cid]['passed'] else 'FAILED'}"
              f"{'' if outcomes[cid]['passed'] else ': ' + str(first)}", flush=True)

    payload = {
        "schemaVersion": 1,
        "pilotId": study["pilotId"],
        "gateSetId": gates["gateSetId"],
        "developmentTracks": len(dev_ids),
        "guitarSetTracks": len(tracks),
        "featurePipeline": pipeline,
        "baseModel": study["baseModel"]["id"],
        "baseModelChecksum": digest,
        "weightsModified": False,
        "p00Accessed": False,
        "slakhTestSplitUsed": False,
        "results": results,
        "gateOutcomes": outcomes,
        "eligibleCandidates": [c for c, o in outcomes.items() if o["passed"]],
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"eligible": payload["eligibleCandidates"]}, indent=2))


def _dev_track_ids(root: Path, size: int) -> list[str]:
    tracks = json.loads((root / "manifests" / "slakh-tracks.json").read_text(encoding="utf-8"))
    seen: set[str] = set()
    chosen: list[str] = []
    for entry in sorted(tracks, key=lambda t: (t.get("midiSha256") or "", t["trackId"])):
        if entry.get("officialSplit") != "validation" or entry.get("error"):
            continue
        digest = entry.get("midiSha256")
        if not digest or digest in seen:
            continue
        seen.add(digest)
        chosen.append(entry["trackId"])
        if len(chosen) >= size:
            break
    return chosen


if __name__ == "__main__":
    main()
