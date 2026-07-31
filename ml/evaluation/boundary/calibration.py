"""Boundary-probability calibration: reliability, ECE, and post-hoc calibrators.

All calibrators map a boundary probability to a corrected probability and are
fitted on *training performers only* (never the held-out fold, never p00). Each
is invertible into a plain scalar transform so the exported four-head contract
is unchanged: calibration is applied to the boundary channel downstream of the
network, not by adding a head.

Three families are supported, in increasing flexibility:

* :class:`TemperatureCalibrator` -- one scalar on the logit. Cannot change the
  ranking of frames, so it isolates *calibration* from *discrimination*.
* :class:`PlattCalibrator` -- scale and shift on the logit (logistic
  regression). Also ranking-preserving when the scale is positive.
* :class:`IsotonicCalibrator` -- monotonic step function via pool-adjacent
  violators. Most flexible, still ranking-preserving, but can overfit small
  fold counts, so it is fitted on the pooled training performers.

``fit_calibrator`` selects by name and always returns something callable, so a
candidate that declines calibration can use ``identity`` uniformly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


@dataclass
class IdentityCalibrator:
    """No-op calibrator, so the uncalibrated control shares one code path."""

    name: str = "identity"

    def __call__(self, p):
        return np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)

    def as_dict(self) -> dict:
        return {"name": self.name}


@dataclass
class TemperatureCalibrator:
    """``sigmoid(logit(p) / T)``. T > 1 softens, T < 1 sharpens."""

    temperature: float
    name: str = "temperature"

    def __call__(self, p):
        return _sigmoid(_logit(p) / max(self.temperature, _EPS))

    def as_dict(self) -> dict:
        return {"name": self.name, "temperature": round(float(self.temperature), 6)}


@dataclass
class PlattCalibrator:
    """``sigmoid(a * logit(p) + b)``."""

    scale: float
    bias: float
    name: str = "platt"

    def __call__(self, p):
        return _sigmoid(self.scale * _logit(p) + self.bias)

    def as_dict(self) -> dict:
        return {"name": self.name, "scale": round(float(self.scale), 6),
                "bias": round(float(self.bias), 6)}


@dataclass
class IsotonicCalibrator:
    """Monotonic piecewise-constant map fitted by pool-adjacent-violators."""

    thresholds: np.ndarray
    values: np.ndarray
    name: str = "isotonic"

    def __call__(self, p):
        p = np.asarray(p, dtype=np.float64)
        idx = np.searchsorted(self.thresholds, p, side="right") - 1
        idx = np.clip(idx, 0, len(self.values) - 1)
        return self.values[idx]

    def as_dict(self) -> dict:
        return {"name": self.name, "knots": int(len(self.values))}


def _fit_temperature(probs: np.ndarray, labels: np.ndarray, *, iterations: int = 200) -> float:
    """Minimise NLL over log-temperature by bisection on the gradient.

    A 1-D convex-in-practice problem; a bounded ternary search is more robust
    here than hand-rolled gradient descent and needs no optimiser dependency.
    """
    z = _logit(probs)
    y = labels.astype(np.float64)

    def nll(log_t: float) -> float:
        t = np.exp(log_t)
        s = np.clip(_sigmoid(z / t), _EPS, 1.0 - _EPS)
        return float(-np.mean(y * np.log(s) + (1.0 - y) * np.log(1.0 - s)))

    lo, hi = -3.0, 3.0
    for _ in range(iterations):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if nll(m1) < nll(m2):
            hi = m2
        else:
            lo = m1
        if hi - lo < 1e-6:
            break
    return float(np.exp((lo + hi) / 2.0))


def _fit_platt(probs: np.ndarray, labels: np.ndarray, *,
               iterations: int = 300, learning_rate: float = 0.5) -> tuple[float, float]:
    """Logistic regression on the logit with plain full-batch gradient descent."""
    z = _logit(probs)
    y = labels.astype(np.float64)
    a, b = 1.0, 0.0
    n = max(len(z), 1)
    for _ in range(iterations):
        s = _sigmoid(a * z + b)
        err = s - y
        ga = float(np.dot(err, z) / n)
        gb = float(np.sum(err) / n)
        a -= learning_rate * ga
        b -= learning_rate * gb
    return a, b


def _fit_isotonic(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators on (probability, label) pairs sorted by probability."""
    order = np.argsort(probs, kind="mergesort")
    x = np.asarray(probs, dtype=np.float64)[order]
    y = np.asarray(labels, dtype=np.float64)[order]

    # Each block holds (sum of y, count, left-most x).
    block_sum: list[float] = []
    block_len: list[float] = []
    block_x: list[float] = []
    for xi, yi in zip(x, y):
        block_sum.append(float(yi))
        block_len.append(1.0)
        block_x.append(float(xi))
        while len(block_sum) > 1 and (
            block_sum[-2] / block_len[-2] > block_sum[-1] / block_len[-1]
        ):
            s = block_sum.pop()
            c = block_len.pop()
            block_x.pop()
            block_sum[-1] += s
            block_len[-1] += c
    thresholds = np.asarray(block_x, dtype=np.float64)
    values = np.asarray(block_sum, dtype=np.float64) / np.asarray(block_len, dtype=np.float64)
    return thresholds, values


def fit_calibrator(name: str, probs, labels):
    """Fit ``name`` on (probs, labels); always returns a callable calibrator."""
    probs = np.asarray(probs, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    if name in ("identity", "none", None) or len(probs) == 0:
        return IdentityCalibrator()
    if name == "temperature":
        return TemperatureCalibrator(_fit_temperature(probs, labels))
    if name == "platt":
        scale, bias = _fit_platt(probs, labels)
        return PlattCalibrator(scale, bias)
    if name == "isotonic":
        thresholds, values = _fit_isotonic(probs, labels)
        return IsotonicCalibrator(thresholds, values)
    raise ValueError(f"unknown calibrator: {name}")


def reliability_bins(probs, labels, *, bins: int = 10) -> list[dict[str, float]]:
    """Equal-width reliability diagram rows: confidence vs observed frequency."""
    probs = np.asarray(probs, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < bins - 1 else (probs >= lo) & (probs <= hi)
        count = int(mask.sum())
        rows.append({
            "binLow": round(float(lo), 4),
            "binHigh": round(float(hi), 4),
            "count": count,
            "meanPredicted": round(float(probs[mask].mean()), 6) if count else 0.0,
            "observedFrequency": round(float(labels[mask].mean()), 6) if count else 0.0,
        })
    return rows


def expected_calibration_error(probs, labels, *, bins: int = 10) -> float:
    """Count-weighted mean |confidence - accuracy| over equal-width bins."""
    rows = reliability_bins(probs, labels, bins=bins)
    total = sum(r["count"] for r in rows)
    if not total:
        return 0.0
    return float(sum(
        r["count"] * abs(r["meanPredicted"] - r["observedFrequency"]) for r in rows
    ) / total)


def maximum_calibration_error(probs, labels, *, bins: int = 10) -> float:
    rows = reliability_bins(probs, labels, bins=bins)
    populated = [r for r in rows if r["count"]]
    if not populated:
        return 0.0
    return float(max(abs(r["meanPredicted"] - r["observedFrequency"]) for r in populated))
