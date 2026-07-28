"""Phase 12 full-band baselines: engines x source views on a pilot subset.

Scores each engine on each deterministic source view, so three different
questions stay separable:

* *Can the model recognise full-band harmony at all?* -- oracle-harmonic
* *How much does the mixture itself cost?* -- full-mix vs oracle-harmonic
* *Does it depend on a guitar being present?* -- guitar-absent-harmonic
* *Does it hallucinate harmony from rhythm?* -- percussion-only

Engines share one interface so the existing metrics score them identically.
The rule engine needs no checkpoint; the learned engines are loaded from the
frozen research checkpoints and are never modified here.

Results are reported per view AND per guitar presence, because a headline
average over a guitar-heavy subset would hide exactly the failure this study
is looking for.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..evaluation.metrics import evaluate_regions
from .views import VIEW_NAMES, build_view

DEFAULT_VIEWS = VIEW_NAMES
# percussion-only carries no harmony; scoring accuracy there is meaningless, so
# it is judged on no-chord behaviour alone.
NO_HARMONY_VIEWS = {"percussion-only"}


@dataclass
class EngineResult:
    engine_id: str
    view: str
    track_id: str
    has_guitar: bool
    metrics: dict[str, Any]
    runtime_seconds: float
    fell_back: bool = False


def _features(audio: np.ndarray, sample_rate: int, pipeline: str):
    """Extract frames with the same pipelines the rest of the project uses."""
    from ..preprocessing.app_features import extract_app_features
    from ..preprocessing.features import extract_features

    if pipeline == "harmony-features-v1":
        return extract_app_features(audio, sample_rate)
    return extract_features(audio, sample_rate)


DEFAULT_PIPELINE = "numpy-chroma-v1"


def build_engines(checkpoints: dict[str, Path], *,
                  transition_penalty: float = 4.0,
                  rule_pipeline: str = "harmony-features-v1") -> dict[str, tuple[Callable, str]]:
    """Engine id -> ``(predict(features) -> [ChordRegion], feature pipeline)``.

    Each learned engine carries **its own checkpoint's** feature pipeline. v1 was
    trained on ``numpy-chroma-v1`` while temporal-v2 models use
    ``harmony-features-v1``; feeding a model the other pipeline silently changes
    its input distribution and understates it, so the pipeline is read from the
    checkpoint metadata rather than assumed globally.

    A checkpoint that is absent is simply not offered as an engine, so a partial
    local checkout still produces a usable rule-vs-learned comparison.
    """
    from ..evaluation.adapters import predict_hybrid, predict_ml, predict_rule
    from ..training.checkpoint import load_checkpoint

    engines: dict[str, tuple[Callable, str]] = {"rule-v3": (predict_rule, rule_pipeline)}
    for engine_id, path in checkpoints.items():
        path = Path(path)
        if not path.exists():
            continue
        model, metadata = load_checkpoint(path)
        pipeline = (metadata.get("featureVersion")
                    or metadata.get("trainingConfig", {}).get("features", {}).get(
                        "pipelineVersion")
                    or DEFAULT_PIPELINE)
        if engine_id.endswith("-hybrid"):
            predict = (lambda features, m=model: predict_hybrid(
                m, features, transition_penalty=transition_penalty))
        else:
            predict = (lambda features, m=model: predict_ml(
                m, features, transition_penalty=transition_penalty))
        engines[engine_id] = (predict, pipeline)
    return engines


def evaluate_track_views(
    track_dir: Path,
    metadata: dict[str, Any],
    reference_regions: list[Any],
    engines: dict[str, tuple[Callable, str]],
    *,
    track_id: str,
    has_guitar: bool,
    views: tuple[str, ...] = DEFAULT_VIEWS,
    sample_rate: int | None = 22050,
) -> list[EngineResult]:
    """Score every engine on every available view of one track.

    Features are extracted once per (view, pipeline) and shared by every engine
    that uses that pipeline, so honouring per-engine pipelines does not multiply
    the extraction cost by the engine count.
    """
    results: list[EngineResult] = []
    for view in views:
        built = build_view(track_dir, metadata, view, sample_rate=sample_rate)
        if built is None:
            continue  # view genuinely absent for this track
        audio, rate = built
        feature_cache: dict[str, Any] = {}
        for engine_id, (predict, pipeline) in engines.items():
            if pipeline not in feature_cache:
                feature_cache[pipeline] = _features(audio, rate, pipeline)
            features = feature_cache[pipeline]
            started = time.perf_counter()
            fell_back = False
            try:
                predicted = predict(features)
            except Exception:  # an engine failing on one view is data, not fatal
                predicted = []
                fell_back = True
            elapsed = time.perf_counter() - started
            metrics = evaluate_regions(
                reference_regions, predicted, tolerances=(0.1, 0.25, 0.5, 1.0)
            ) if predicted else {}
            results.append(EngineResult(
                engine_id=engine_id, view=view, track_id=track_id,
                has_guitar=has_guitar, metrics=metrics,
                runtime_seconds=elapsed, fell_back=fell_back))
    return results


def _weighted(rows: list[EngineResult], name: str) -> float:
    """Duration-weighted mean, skipping rows where the metric is undefined.

    Boundary metrics are ``None`` when a track produced no matched boundary, so
    those rows carry no information for this metric and must be excluded from
    both the numerator and the weight -- treating them as zero would silently
    drag the average toward zero.
    """
    usable = [(r.metrics.get(name), r.metrics.get("evaluatedDurationSeconds", 0.0))
              for r in rows]
    usable = [(value, weight) for value, weight in usable
              if value is not None and weight]
    total = sum(weight for _value, weight in usable)
    if not total:
        return 0.0
    return sum(value * weight for value, weight in usable) / total


def aggregate(results: list[EngineResult]) -> dict[str, Any]:
    """Aggregate by (engine, view), and additionally by guitar presence."""
    scored = [r for r in results if r.metrics]
    by_key: dict[tuple[str, str], list[EngineResult]] = defaultdict(list)
    for row in scored:
        by_key[(row.engine_id, row.view)].append(row)

    def summarise(rows: list[EngineResult]) -> dict[str, Any]:
        duration = sum(r.metrics.get("evaluatedDurationSeconds", 0.0) for r in rows)
        predicted = sum(r.metrics.get("predictedRegions", 0) for r in rows)
        return {
            "tracks": len(rows),
            "evaluatedDurationSeconds": round(duration, 2),
            "rootAccuracy": round(_weighted(rows, "rootAccuracy"), 4),
            "majorMinorAccuracy": round(_weighted(rows, "majorMinorAccuracy"), 4),
            "detailedAccuracy": round(_weighted(rows, "detailedAccuracy"), 4),
            "noChordPrecision": round(_weighted(rows, "noChordPrecision"), 4),
            "noChordRecall": round(_weighted(rows, "noChordRecall"), 4),
            "noChordF1": round(_no_chord_f1(rows), 4),
            "fragmentationRate": round(_weighted(rows, "fragmentationRate"), 4),
            "regionsPerMinute": round(predicted * 60.0 / duration, 3) if duration else 0.0,
            "meanAbsoluteBoundaryErrorMs": round(_weighted(rows, "meanBoundaryErrorMs"), 1),
            "runtimeSecondsPerAudioMinute": round(
                sum(r.runtime_seconds for r in rows) / (duration / 60.0), 4) if duration else 0.0,
            "fallbackCount": sum(1 for r in rows if r.fell_back),
        }

    out: dict[str, Any] = {"byEngineView": {}, "byGuitarPresence": {}, "noHarmonyControl": {}}
    for (engine_id, view), rows in sorted(by_key.items()):
        summary = summarise(rows)
        out["byEngineView"].setdefault(engine_id, {})[view] = summary
        if view in NO_HARMONY_VIEWS:
            # Only no-chord behaviour is meaningful here.
            out["noHarmonyControl"].setdefault(engine_id, {})[view] = {
                "tracks": summary["tracks"],
                "noChordPrecision": summary["noChordPrecision"],
                "noChordRecall": summary["noChordRecall"],
                "noChordF1": summary["noChordF1"],
                "regionsPerMinute": summary["regionsPerMinute"],
            }
        else:
            for label, subset in (("guitarPresent", [r for r in rows if r.has_guitar]),
                                  ("guitarAbsent", [r for r in rows if not r.has_guitar])):
                if subset:
                    out["byGuitarPresence"].setdefault(engine_id, {}).setdefault(
                        view, {})[label] = summarise(subset)
    out["totalFallbacks"] = sum(1 for r in results if r.fell_back)
    out["scoredRows"] = len(scored)
    return out


def _no_chord_f1(rows: list[EngineResult]) -> float:
    precision = _weighted(rows, "noChordPrecision")
    recall = _weighted(rows, "noChordRecall")
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def separation_gap(aggregated: dict[str, Any]) -> dict[str, Any]:
    """Oracle-vs-mixture gap per engine: what better separation could buy."""
    gaps: dict[str, Any] = {}
    for engine_id, views in aggregated["byEngineView"].items():
        mix = views.get("full-mix")
        oracle = views.get("oracle-harmonic")
        guitar = views.get("oracle-guitar")
        if not mix or not oracle:
            continue
        gaps[engine_id] = {
            "fullMixDetailed": mix["detailedAccuracy"],
            "oracleHarmonicDetailed": oracle["detailedAccuracy"],
            "harmonicOracleGain": round(
                oracle["detailedAccuracy"] - mix["detailedAccuracy"], 4),
            "oracleGuitarDetailed": guitar["detailedAccuracy"] if guitar else None,
            "guitarOracleGain": round(
                guitar["detailedAccuracy"] - mix["detailedAccuracy"], 4) if guitar else None,
        }
    return gaps


def write_report(aggregated: dict[str, Any], path: Path, *, extra: dict | None = None) -> Path:
    payload = {
        "schemaVersion": 1,
        "views": list(DEFAULT_VIEWS),
        "aggregate": aggregated,
        "separationGap": separation_gap(aggregated),
    }
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
