"""Run the frozen full-band CPU pilot: primary candidate and control.

Follows ``ml/configs/full-band-pilot-v1.json`` exactly. Nothing here selects a
strategy, a sampling ratio or a schedule -- those were frozen from the Phase 12
baselines before any pilot run started.

Both candidates share one dev set (the Slakh validation subset) and one
schedule, so the control isolates "kept training" from "saw full-band audio".

A note on what the GuitarSet preservation check does and does not claim: v1 was
already trained on p01-p05, and the frozen config rehearses on those same
performers. Measuring there is therefore a *catastrophic-forgetting* check on
data v1 already knew, not a held-out generalisation claim. That is what the
preservation gates are written against, and it is stated in the report rather
than left for the reader to infer.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..evaluation.temporal_v2_ablation import _read_json, _resolve_data_path
from .pilot import (
    DEFAULT_PILOT,
    guitarset_samples,
    load_pilot_config,
    slakh_samples,
    train_pilot,
)

ML_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_ROOT.parent
DEFAULT_BASE_CONFIG = ML_ROOT / "configs" / "temporal-harmony-v2.json"
DEFAULT_RUN_DIR = ML_ROOT / "runs" / "full-band-pilot-v1"


def _base_config_from_checkpoint(checkpoint: Path) -> dict:
    """Reconstruct the initialising model's own training configuration.

    The frozen strategy says to fine-tune the v1 checkpoint *keeping the v1
    objective*. v1 records its own feature pipeline (``numpy-chroma-v1``),
    boundary tolerance (0.12 s) and loss weights, all of which differ from
    temporal-v2's. Taking them from the checkpoint rather than from a v2 config
    file is what makes "keep the v1 objective" true rather than merely stated:
    feeding v1 a different feature pipeline silently changes its input
    distribution, and adding v2's duration/switch/agreement terms changes the
    objective it was trained under.
    """
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("metadata", {}).get("trainingConfig")
    if not config or "training" not in config or "features" not in config:
        raise SystemExit(
            f"{checkpoint.name} has no usable trainingConfig; refusing to guess the objective")
    return json.loads(json.dumps(config))


def _dev_track_ids(root: Path) -> list[str]:
    manifest = root / "manifests" / "dev-extraction.json"
    if not manifest.exists():
        raise SystemExit("dev-extraction.json missing; extract the validation subset first")
    # The extraction manifest records the count; the ids come from the same
    # deterministic rule, so recompute them from the tracks manifest.
    tracks = json.loads((root / "manifests" / "slakh-tracks.json").read_text(encoding="utf-8"))
    size = json.loads(manifest.read_text(encoding="utf-8"))["requestedTrackIds"]
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen full-band pilot.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--pilot-config", default=str(DEFAULT_PILOT))
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--base-config-override", action="store_true",
                        help="Use --base-config instead of the init checkpoint's own "
                             "recorded objective. Deviates from the frozen strategy.")
    parser.add_argument("--prep-report", default=str(
        REPO_ROOT / "evaluation" / "reports" / "full-band-slakh-preparation.json"))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--v1-checkpoint", required=True)
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--mic-audio", default=None)
    parser.add_argument("--pickup-audio", default=None)
    parser.add_argument("--candidate", action="append", default=None)
    args = parser.parse_args()

    pilot_config = load_pilot_config(Path(args.pilot_config))
    base_config = _base_config_from_checkpoint(Path(args.v1_checkpoint)) \
        if not args.base_config_override else _read_json(args.base_config)
    print(f"objective: {json.dumps(base_config['training'], sort_keys=True)}", flush=True)
    print(f"features : {base_config['features']['pipelineVersion']}", flush=True)
    print(f"labels   : boundaryTolerance={base_config['labels']['boundaryToleranceSeconds']}s",
          flush=True)
    root = Path(args.root)
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    sampling = pilot_config["domainSampling"]
    performers = set(sampling["guitarSetPerformers"])
    if "guitarset-p00" in performers:
        raise SystemExit("p00 must not appear in the pilot performers")

    annotations = _resolve_data_path(args.annotations, "TABSMITH_GUITARSET_ANNOTATIONS")
    mic = _resolve_data_path(args.mic_audio, "TABSMITH_GUITARSET_MIC_AUDIO")
    pickup = _resolve_data_path(args.pickup_audio, "TABSMITH_GUITARSET_PICKUP_AUDIO")
    if not annotations or not mic or not pickup:
        parser.error("requires GuitarSet paths via args or TABSMITH_GUITARSET_* env vars")

    started = time.time()
    print("extracting GuitarSet rehearsal features...", flush=True)
    guitarset = guitarset_samples(
        annotations, {"audio_mono-mic": mic, "audio_mono-pickup_mix": pickup},
        base_config, performers)
    print(f"  guitarset samples: {len(guitarset)}", flush=True)

    prep = json.loads(Path(args.prep_report).read_text(encoding="utf-8"))
    train_ids = list(prep["pilotSubset"]["trackIds"])
    views = tuple(sampling["slakhViews"])
    print(f"extracting Slakh training features ({len(train_ids)} tracks x {len(views)} views)...",
          flush=True)
    slakh = slakh_samples(root / "extracted", train_ids, base_config, views)
    print(f"  slakh samples: {len(slakh)}", flush=True)

    dev_ids = _dev_track_ids(root)
    print(f"extracting Slakh development features ({len(dev_ids)} validation tracks)...",
          flush=True)
    dev = slakh_samples(root / "extracted", dev_ids, base_config, ("full-mix",))
    print(f"  development samples: {len(dev)}", flush=True)
    if not dev.samples:
        raise SystemExit("no development samples; cannot early-stop honestly")

    fractions = {
        "mixed-domain-finetune": {"guitarset": sampling["guitarSetFraction"],
                                  "slakh": sampling["slakhFraction"]},
        "guitarset-only-control": {"guitarset": 1.0, "slakh": 0.0},
    }

    results = []
    for candidate in pilot_config["candidates"]:
        if args.candidate and candidate["id"] not in set(args.candidate):
            continue
        print(f"\n=== {candidate['id']} ({candidate['role']}) ===", flush=True)
        result = train_pilot(
            candidate=candidate,
            base_config=base_config,
            domains=[guitarset, slakh],
            dev_samples=dev.samples,
            fractions=fractions[candidate["id"]],
            checkpoint_dir=run_dir / candidate["id"],
            init_checkpoint=Path(args.v1_checkpoint),
        )
        result["role"] = candidate["role"]
        result["domainFractions"] = fractions[candidate["id"]]
        results.append(result)
        print(f"  done: best epoch {result['bestEpoch']} dev {result['bestDevLoss']:.4f} "
              f"in {result['trainSeconds'] / 60:.1f} min", flush=True)

    report = {
        "schemaVersion": 1,
        "pilotId": pilot_config["pilotId"],
        "gateSet": pilot_config["gateSet"],
        "strategy": pilot_config["strategy"]["id"],
        "developmentSet": "slakh-validation-subset",
        "developmentTracks": len(dev_ids),
        "p00UsedForSelection": False,
        "slakhTestSplitUsed": False,
        "results": results,
        "elapsedSeconds": round(time.time() - started, 1),
    }
    (run_dir / "pilot-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
