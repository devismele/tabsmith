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

Pilot v2 adds two bounded extensions, both inert unless a candidate asks for
them, so the v1 code path behaves exactly as it did:

* per-candidate domain fractions, because v2 varies the rehearsal ratio between
  candidates rather than across the whole pilot
* an optional preservation regulariser that anchors the root posterior to the
  frozen v1 teacher on rehearsal frames only, which is the quantity pilot v1
  measured as regressing
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
                  views: tuple[str, ...], per_track_cache: Path | None = None) -> DomainSamples:
    """Slakh full-band samples, one per (track, view).

    With ``per_track_cache`` each track's samples are written as they are built,
    so an interrupted extraction resumes instead of restarting. At 600 tracks
    this pass takes the better part of an hour, and caching only the finished
    whole meant a stop near the end discarded all of it.
    """
    import pickle

    def _cached_track(track_id: str, build):
        if per_track_cache is None:
            return build()
        per_track_cache.mkdir(parents=True, exist_ok=True)
        key = cache_key(pipeline=config["features"]["pipelineVersion"],
                        tolerance=config["labels"]["boundaryToleranceSeconds"],
                        views=list(views), track=track_id)
        path = per_track_cache / f"{track_id}.pkl"
        if path.exists():
            try:
                payload = pickle.loads(path.read_bytes())
                if payload.get("key") == key:
                    return payload["samples"]
            except Exception:
                pass  # a truncated or unreadable entry is simply rebuilt
        samples = build()
        tmp = path.with_suffix(".pkl.tmp")
        tmp.write_bytes(pickle.dumps({"key": key, "samples": samples},
                                     protocol=pickle.HIGHEST_PROTOCOL))
        tmp.replace(path)
        return samples

    return _slakh_samples_impl(extracted_root, track_ids, config, views, _cached_track)


def _slakh_samples_impl(extracted_root: Path, track_ids: list[str], config: dict[str, Any],
                        views: tuple[str, ...], cached_track) -> DomainSamples:
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

    def build_one(track_id: str) -> list[Any]:
        track_dir = Path(extracted_root) / track_id
        midi_path = track_dir / "all_src.mid"
        metadata_path = track_dir / "metadata.yaml"
        if not midi_path.exists() or not metadata_path.exists():
            return []
        events, duration = read_note_events(midi_path.read_bytes())
        if duration <= 0:
            return []
        regions = derive_chord_regions(events, duration)
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8", errors="replace"))
        built_samples: list[Any] = []
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
                built_samples.append(sample)
        return built_samples

    samples: list[Any] = []
    for position, track_id in enumerate(track_ids, start=1):
        samples.extend(cached_track(track_id, lambda t=track_id: build_one(t)))
        if position % 50 == 0:
            print(f"    features: {position}/{len(track_ids)} tracks", flush=True)
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


LEGACY_GUITARSET_ONLY_CONTROL = "guitarset-only-control"


def cache_key(**fields) -> str:
    """Identity of a cached artifact: everything that would change its contents."""
    import hashlib

    return hashlib.sha256(
        json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()


def cached_samples(cache_dir: Path | None, name: str, key: str, build):
    """Build once, reuse across restarts, and never reuse a stale artifact.

    Feature extraction costs about 13 minutes per run and is pure: the same
    inputs give the same output. An interrupted long CPU run should not have to
    pay it again. The recorded key covers the feature pipeline, the boundary
    tolerance and the exact track set, so a cache built for different inputs is
    recomputed rather than silently reused - the same rule the training
    checkpoints already follow.
    """
    import pickle

    if cache_dir is None:
        return build()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    blob = cache_dir / f"{name}.pkl"
    sidecar = cache_dir / f"{name}.json"
    if blob.exists() and sidecar.exists():
        recorded = json.loads(sidecar.read_text(encoding="utf-8"))
        if recorded.get("key") == key:
            with blob.open("rb") as handle:
                payload = pickle.load(handle)
            print(f"  cache hit: {name} ({len(payload)} entries)", flush=True)
            return payload
        print(f"  cache stale: {name} (key changed); recomputing", flush=True)
    payload = build()
    tmp = blob.with_suffix(".pkl.tmp")
    with tmp.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(blob)
    sidecar.write_text(json.dumps({"key": key, "count": len(payload)}, indent=2),
                       encoding="utf-8")
    return payload


def resolve_fractions(config: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    """Domain fractions for one candidate.

    Pilot v1 held a single pilot-wide ratio and expressed its control as
    "GuitarSet batches only" in prose, with the ratio hardcoded by the runner.
    Pilot v2 varies the ratio *between* candidates, so the ratio moves into the
    candidate. Resolution order keeps v1 producing exactly the numbers it did:

    1. the candidate's own ``domainSampling`` override, if present
    2. the v1 control, whose GuitarSet-only ratio is named rather than inferred
    3. the pilot-wide default
    """
    override = candidate.get("domainSampling")
    if override:
        fractions = {"guitarset": float(override["guitarSetFraction"]),
                     "slakh": float(override["slakhFraction"])}
    elif candidate.get("id") == LEGACY_GUITARSET_ONLY_CONTROL:
        fractions = {"guitarset": 1.0, "slakh": 0.0}
    else:
        sampling = config["domainSampling"]
        fractions = {"guitarset": float(sampling["guitarSetFraction"]),
                     "slakh": float(sampling["slakhFraction"])}
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"{candidate.get('id')} domain fractions sum to {total}, not 1.0")
    return fractions


def root_distillation(student_logits, teacher_logits, mask, temperature: float = 1.0):
    """KL(teacher || student) over the root posterior, averaged on ``mask``.

    The anchor is deliberately *not* a cross-entropy against the reference root:
    that term already exists in the v1 objective and is what full-band training
    is allowed to move. This constrains the student toward the frozen teacher's
    full distribution on rehearsal frames, which is the quantity the preservation
    gate measures.

    ``mask`` is padding AND domain, so frames from the new domain contribute
    nothing and the anchor cannot suppress full-band learning directly.
    """
    import torch.nn.functional as F

    t = float(temperature)
    if t <= 0:
        raise ValueError("distillation temperature must be positive")
    teacher_log_p = F.log_softmax(teacher_logits / t, dim=-1)
    student_log_p = F.log_softmax(student_logits / t, dim=-1)
    per_frame = (teacher_log_p.exp() * (teacher_log_p - student_log_p)).sum(dim=-1)
    # T^2 keeps the gradient scale comparable across temperatures (Hinton et al.).
    return (per_frame * mask).sum() / mask.sum().clamp(min=1.0) * (t * t)


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
    preservation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fine-tune one candidate, checkpointing every epoch and resuming safely.

    ``preservation`` is the optional pilot-v2 anchor. When absent this is the
    pilot-v1 procedure unchanged.
    """
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

    teacher = None
    anchor_weight = 0.0
    anchor_temperature = 1.0
    rehearsal_ids: set[str] = set()
    if preservation:
        if preservation.get("type") != "root-head-distillation":
            raise ValueError(f"unsupported preservation type {preservation.get('type')!r}")
        if not init_checkpoint or not Path(init_checkpoint).exists():
            raise ValueError("root-head distillation needs the teacher checkpoint")
        # The teacher is the *initialising* checkpoint, loaded separately and
        # frozen. Resuming must not re-anchor to a partially trained student.
        teacher, _ = load_checkpoint(Path(init_checkpoint))
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        anchor_weight = float(preservation.get("weight", 1.0))
        anchor_temperature = float(preservation.get("temperature", 1.0))
        applies_to = set(preservation.get("appliesToDomains", ["guitarset"]))
        rehearsal_ids = {s.track_id for d in domains if d.name in applies_to for s in d.samples}
        if not rehearsal_ids:
            raise ValueError(f"no samples in preservation domains {sorted(applies_to)}")
        print(f"  anchor: root distillation to {Path(init_checkpoint).name} "
              f"(weight {anchor_weight}, {len(rehearsal_ids)} rehearsal samples)", flush=True)

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
        anchor_total = 0.0
        batches = 0
        for start in range(0, len(stream), batch_size):
            chunk = stream[start:start + batch_size]
            batch = collate(chunk)
            optimizer.zero_grad()
            outputs = model(batch["features"])
            loss, _parts = combined_loss(outputs, batch, weights, config)
            if teacher is not None:
                rows = torch.tensor(
                    [1.0 if s.track_id in rehearsal_ids else 0.0 for s in chunk],
                    dtype=torch.float32)
                mask = batch["pad_mask"] * rows.unsqueeze(1)
                if float(mask.sum()) > 0:
                    with torch.no_grad():
                        teacher_root = teacher(batch["features"])["root"]
                    anchor = root_distillation(outputs["root"], teacher_root, mask,
                                               anchor_temperature)
                    loss = loss + anchor_weight * anchor
                    anchor_total += float(anchor.detach())
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
        entry = {"epoch": epoch, "train": train_loss, "dev": dev_loss,
                 "streamSize": len(stream)}
        if teacher is not None:
            # Reported separately because the development set is Slakh-only: the
            # anchor never applies there, so dev loss stays the same quantity for
            # every candidate and early stopping remains comparable.
            entry["anchor"] = anchor_total / max(batches, 1)
        state["history"].append(entry)
        improved = state["bestDevLoss"] is None or dev_loss < state["bestDevLoss"]
        if improved:
            state["bestDevLoss"] = dev_loss
            state["bestEpoch"] = epoch
            metadata = build_metadata(config, "full-band-pilot", seed,
                                      model.parameter_count(), model)
            save_checkpoint(checkpoint_path, model, config, metadata)
        state["completedEpochs"] = epoch + 1
        write_state(checkpoint_dir, state)
        anchor_note = f" | anchor {entry['anchor']:.4f}" if "anchor" in entry else ""
        print(f"  epoch {epoch:02d} | train {train_loss:.4f} | dev {dev_loss:.4f}"
              f"{' *' if improved else ''}{anchor_note} | stream {len(stream)}", flush=True)

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
        "preservation": preservation,
    }
