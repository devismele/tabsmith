"""Boundary-calibration study (boundary-calibration-v3).

The segmental-v3 study established that decoder-side segmental decoding reduces
full-v2 over-segmentation but cannot reach the frozen v1-replacement gates on
its own, and traced the residual to *model-side* boundary behaviour: false
transitions occur where the boundary head is weak (mean 0.23) relative to true
transitions (0.42), and decoded state changes agree with boundary-head peaks
only ~27% of the time.

This package quantifies that boundary behaviour (calibration, thresholding,
discrimination, timing, agreement) and evaluates a bounded set of interventions
against it. It never loads p00, and it changes no production weights, defaults
or release gating.
"""

from .calibration import (
    IdentityCalibrator,
    PlattCalibrator,
    TemperatureCalibrator,
    expected_calibration_error,
    fit_calibrator,
    reliability_bins,
)

__all__ = [
    "IdentityCalibrator",
    "PlattCalibrator",
    "TemperatureCalibrator",
    "expected_calibration_error",
    "fit_calibrator",
    "reliability_bins",
]
