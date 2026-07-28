"""Bounded, resumable CPU pilot: mixed-domain fine-tune from v1.

Implements the strategy frozen in ``ml/configs/full-band-pilot-v1.json``:
GuitarSet rehearsal batches mixed 50/50 with Slakh full-band batches, fine-tuned
from the v1 checkpoint, keeping the v1 objective and the four exported heads.

The control trains on GuitarSet batches only under the identical schedule, which
separates "the model changed because it saw full-band audio" from "the model
changed because it kept training".

CPU-only realities are handled rather than ignored:

* features for both domains are extracted once and cached in memory per run
* a checkpoint is written every epoch, and a run resumes from the last epoch
  rather than restarting
* the frozen sampling ratio is applied per epoch with a seeded RNG, so a resumed
  run sees the same stream it would have seen uninterrupted
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ML_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PILOT = ML_ROOT / "configs" / "full-band-pilot-v1.json"


@dataclass
class DomainSamples:
    """Training samples for one domain, kept separate so mixing stays explicit."""

    name: str
    samples: list[Any]

    def __len__(self) -> int:
        return len(self.samples)


def load_pilot_config(path: Path = DEFAULT_PILOT) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def guitarset_samples(annotation_dir: Path, audio_dirs: dict[str, Path],
                      config: dict[str, Any], performers: set[str]) -> DomainSamples:
    """GuitarSet rehearsal samples for the frozen development performers."""
    from ..evaluation.temporal_v2_ablation import _extract_feature_map, _load_tracks
    from ..training.dataset import sample_from_features

    tracks = [t for t in _load_tracks(annotation_dir, audio_dirs) if t.artist in performers]
    if not tracks:
        raise ValueError("zero GuitarSet tracks for the requested performers")
    if any(t.artist == "guitarset-p00" for t in tracks):
        raise AssertionError("p00 entered the full-band pilot")
    features = _extract_feature_map(tracks, config["features"]["pipelineVersion"])
    tolerance = float(config["labels"]["boundaryToleranceSeconds"])
    samples = []
    for track in tracks:
        frames = features.get(track.track_id)
        if frames is None:
            continue
        sample = sample_from_features(track, frames, tolerance)
        if sample is not None:
            samples.append(sample)
    return DomainSamples("guitarset", samples)


def slakh_samples(extracted_root: Path, track_ids: list[str], config: dict[str, Any],
                  views: tuple[str, ...]) -> DomainSamples:
    """Slakh full-band samples, one per (track, view)."""
    import yaml

    from ..preprocessing.app_features import extract_app_features
    from ..preprocessing.features import extract_features
    from ..schema import Track
    from ..training.dataset import sample_from_features
    from .midi import read_note_events
    from .symbolic import derive_chord_regions
    from .views import build_view

    pipeline = config["features"]["pipelineVersion"]
    tolerance = float(config["labels"]["boundaryToleranceSeconds"])
    samples = []
    for track_id in track_ids:
        track_dir = Path(extracted_root) / track_id
        midi_path = track_dir / "all_src.mid"
        metadata_path = track_dir / "metadata.yaml"
        if not midi_path.exists() or not metadata_path.exists():
            continue
        events, duration = read_note_events(midi_path.read_bytes())
        if duration <= 0:
            continue
        regions = derive_chord_regions(events, duration)
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8", errors="replace"))
        for view in views:
            built = build_view(track_dir, metadata, view, sample_rate=22050)
            if built is None:
                continue
            audio, rate = built
            frames = (extract_app_features(audio, rate) if pipeline == "harmony-features-v1"
                      else extract_features(audio, rate))
            track = Track(track_id=f"slakh-{track_id}@{view}", artist="slakh",
                          title=track_id, duration=duration, source="slakh2100",
                          audio_availability="audio", chords=regions)
            sample = sample_from_features(track, frames, tolerance)
            if sample is not None:
                samples.append(sample)
    return DomainSamples("slakh", samples)


def mix_epoch(domains: list[DomainSamples], fractions: dict[str, float],
              rng: np.random.Generator) -> list[Any]:
    """One epoch's sample stream at the frozen domain ratio.

    The epoch length is set by the domain that can supply its share without
    repetition, so the ratio is honoured exactly rather than approximated by
    oversampling whichever domain happens to be larger.
    """
    available = {d.name: len(d) for d in domains if len(d)}
    if not available:
        raise ValueError("no samples in any domain")
    limits = []
    for name, count in available.items():
        share = fractions.get(name, 0.0)
        if share > 0:
            limits.append(count / share)
    total = int(min(limits)) if limits else 0
    stream: list[Any] = []
    for domain in domains:
        share = fractions.get(domain.name, 0.0)
        take = int(round(total * share))
        if take <= 0 or not len(domain):
            continue
        index = rng.permutation(len(domain))[:take]
        stream.extend(domain.samples[i] for i in index)
    rng.shuffle(stream)
    return stream


def resume_state(checkpoint_dir: Path) -> dict[str, Any]:
    """Last completed epoch and history, or an empty state."""
    path = Path(checkpoint_dir) / "pilot-state.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"completedEpochs": 0, "history": [], "bestDevLoss": None, "bestEpoch": -1}


def write_state(checkpoint_dir: Path, state: dict[str, Any]) -> None:
    path = Path(checkpoint_dir) / "pilot-state.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


def train_pilot(
    *,
    candidate: dict[str, Any],
    base_config: dict[str, Any],
    domains: list[DomainSamples],
    dev_samples: list[Any],
    fractions: dict[str, float],
    checkpoint_dir: Path,
    init_checkpoint: Path | None,
    seed: int = 20260728,
) -> dict[str, Any]:
    """Fine-tune one candidate, checkpointing every epoch and resuming safely."""
    import torch

    from ..models.temporal_baseline import ModelConfig, TemporalBaseline
    from ..training.checkpoint import build_metadata, load_checkpoint, save_checkpoint
    from ..training.dataset import collate, compute_class_weights
    from ..training.losses import combined_loss

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state = resume_state(checkpoint_dir)
    checkpoint_path = checkpoint_dir / "model.pt"

    config = json.loads(json.dumps(base_config))
    config.setdefault("training", {}).update(candidate.get("trainingOverrides", {}))
    config["modelName"] = f"full-band-pilot-{candidate['id']}"

    if checkpoint_path.exists() and state["completedEpochs"] > 0:
        model, _ = load_checkpoint(checkpoint_path)
        print(f"resume: {candidate['id']} from epoch {state['completedEpochs']}", flush=True)
    elif init_checkpoint and Path(init_checkpoint).exists():
        model, _ = load_checkpoint(Path(init_checkpoint))
        print(f"init: {candidate['id']} from {Path(init_checkpoint).name}", flush=True)
    else:
        model = TemporalBaseline(ModelConfig.from_dict(config))
        print(f"init: {candidate['id']} from scratch", flush=True)

    training = config["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learningRate"]),
                                 weight_decay=float(training.get("weightDecay", 0.0)))
    epochs = int(training["epochs"])
    patience = int(training.get("earlyStoppingPatience", 4))
    batch_size = int(training.get("batchSize", 16))
    all_samples = [s for d in domains for s in d.samples]
    weights = compute_class_weights(all_samples)
    started = time.time()

    for epoch in range(state["completedEpochs"], epochs):
        rng = np.random.default_rng(seed + epoch)
        stream = mix_epoch(domains, fractions, rng)
        model.train()
        train_total = 0.0
        batches = 0
        for start in range(0, len(stream), batch_size):
            batch = collate(stream[start:start + batch_size])
            optimizer.zero_grad()
            outputs = model(batch["features"])
            loss, _parts = combined_loss(outputs, batch, weights, config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           float(training.get("gradClip", 5.0)))
            optimizer.step()
            train_total += float(loss.detach())
            batches += 1

        model.eval()
        dev_total = 0.0
        dev_batches = 0
        with torch.no_grad():
            for start in range(0, len(dev_samples), batch_size):
                batch = collate(dev_samples[start:start + batch_size])
                outputs = model(batch["features"])
                loss, _parts = combined_loss(outputs, batch, weights, config)
                dev_total += float(loss.detach())
                dev_batches += 1

        train_loss = train_total / max(batches, 1)
        dev_loss = dev_total / max(dev_batches, 1)
        state["history"].append({"epoch": epoch, "train": train_loss, "dev": dev_loss,
                                 "streamSize": len(stream)})
        improved = state["bestDevLoss"] is None or dev_loss < state["bestDevLoss"]
        if improved:
            state["bestDevLoss"] = dev_loss
            state["bestEpoch"] = epoch
            metadata = build_metadata(config, "full-band-pilot", seed,
                                      model.parameter_count(), model)
            save_checkpoint(checkpoint_path, model, config, metadata)
        state["completedEpochs"] = epoch + 1
        write_state(checkpoint_dir, state)
        print(f"  epoch {epoch:02d} | train {train_loss:.4f} | dev {dev_loss:.4f}"
              f"{' *' if improved else ''} | stream {len(stream)}", flush=True)

        if epoch - state["bestEpoch"] >= patience:
            print(f"  early stop at epoch {epoch} (best {state['bestEpoch']})", flush=True)
            break

    return {
        "candidateId": candidate["id"],
        "epochsRun": state["completedEpochs"],
        "bestEpoch": state["bestEpoch"],
        "bestDevLoss": state["bestDevLoss"],
        "history": state["history"],
        "trainSeconds": round(time.time() - started, 1),
        "domainSizes": {d.name: len(d) for d in domains},
        "developmentSamples": len(dev_samples),
    }
