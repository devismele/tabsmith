"""Probability-domain temporal filters for ML-only/v2 ablations.

Filters operate on complete distributions before any decoding. They never
smooth hard chord labels, so uncertainty and alternate candidates remain
available to Viterbi and hybrid fusion.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.maximum(values, 1e-9)
    return values / values.sum(axis=-1, keepdims=True)


def _ema(values: np.ndarray, alpha: float, reset: np.ndarray | None = None) -> np.ndarray:
    if len(values) == 0:
        return values.copy()
    output = values.astype(np.float64, copy=True)
    for index in range(1, len(output)):
        local_alpha = alpha
        if reset is not None:
            local_alpha = alpha + float(np.clip(reset[index], 0.0, 1.0)) * (1.0 - alpha)
        output[index] = local_alpha * values[index] + (1.0 - local_alpha) * output[index - 1]
    return output


def _window_filter(values: np.ndarray, radius: int, *, median: bool) -> np.ndarray:
    if radius <= 0 or len(values) == 0:
        return values.copy()
    output = np.empty_like(values, dtype=np.float64)
    for index in range(len(values)):
        window = values[max(0, index - radius):min(len(values), index + radius + 1)]
        output[index] = np.median(window, axis=0) if median else np.mean(window, axis=0)
    return output


def smooth_learned_probabilities(
    root: np.ndarray,
    quality: np.ndarray,
    no_chord: np.ndarray,
    boundary: np.ndarray,
    settings: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Filter aligned probability arrays and preserve their shapes."""
    if not settings or settings.get("method", "none") == "none":
        return root.copy(), quality.copy(), no_chord.copy(), boundary.copy()
    if not (len(root) == len(quality) == len(no_chord) == len(boundary)):
        raise ValueError("learned probability arrays must have identical frame counts")

    method = str(settings["method"])
    alpha = float(settings.get("alpha", 0.65))
    boundary_alpha = float(settings.get("boundaryAlpha", alpha))
    if not 0.0 < alpha <= 1.0 or not 0.0 < boundary_alpha <= 1.0:
        raise ValueError("EMA alpha values must satisfy 0 < alpha <= 1")

    if method in {"ema", "persistence-ema"}:
        reset = boundary if method == "persistence-ema" else None
        smooth_root = _ema(root, alpha, reset)
        smooth_quality = _ema(quality, alpha, reset)
        smooth_no_chord = _ema(no_chord[:, None], alpha, reset)[:, 0]
        smooth_boundary = _ema(boundary[:, None], boundary_alpha)[:, 0]
    elif method in {"mean", "median"}:
        radius = int(settings.get("radiusFrames", 2))
        if radius < 0:
            raise ValueError("radiusFrames must be non-negative")
        median = method == "median"
        smooth_root = _window_filter(root, radius, median=median)
        smooth_quality = _window_filter(quality, radius, median=median)
        smooth_no_chord = _window_filter(no_chord[:, None], radius, median=median)[:, 0]
        smooth_boundary = _window_filter(boundary[:, None], radius, median=median)[:, 0]
    else:
        raise ValueError(f"unsupported probability smoothing method {method!r}")

    return (
        _normalize_rows(smooth_root).astype(np.float32),
        _normalize_rows(smooth_quality).astype(np.float32),
        np.clip(smooth_no_chord, 0.0, 1.0).astype(np.float32),
        np.clip(smooth_boundary, 0.0, 1.0).astype(np.float32),
    )
