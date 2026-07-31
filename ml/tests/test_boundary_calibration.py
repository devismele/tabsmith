r"""Boundary-calibration study tests (offline, no torch, no cache, no p00).

Cover the calibrators' fitting and invariants, the reliability/ECE reporting,
the frame-label construction, the diagnosis' category accounting, and the
reconciliation guards that keep p00 sealed and captures paired.

Run: .\.venv\Scripts\python.exe -m unittest discover -s ml/tests -t .
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np

from ml.evaluation.boundary.calibration import (
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureCalibrator,
    expected_calibration_error,
    fit_calibrator,
    maximum_calibration_error,
    reliability_bins,
)
from ml.evaluation.boundary.diagnose import (
    _frame_boundary_labels,
    _prf,
    chord_posteriors,
    classify_dominant_issue,
)
from ml.evaluation.boundary.reconcile import check_capture_pairing, check_p00_sealed


@dataclass
class _Entry:
    performer_id: str
    performance_id: str
    capture: str
    fold_id: str = "lopo-p01"


class CalibratorTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20260727)
        # Well-separated but compressed scores: the true rate is far above the
        # predicted probability, i.e. the underconfidence this study measured.
        self.labels = rng.binomial(1, 0.3, size=4000).astype(np.float64)
        base = np.where(self.labels > 0, 0.35, 0.10)
        self.probs = np.clip(base + rng.normal(0, 0.05, size=4000), 1e-4, 1 - 1e-4)

    def test_identity_is_a_noop(self):
        cal = IdentityCalibrator()
        np.testing.assert_allclose(cal(self.probs), self.probs)

    def test_temperature_improves_calibration(self):
        cal = fit_calibrator("temperature", self.probs, self.labels)
        self.assertIsInstance(cal, TemperatureCalibrator)
        before = expected_calibration_error(self.probs, self.labels)
        after = expected_calibration_error(cal(self.probs), self.labels)
        self.assertLess(after, before)

    def test_platt_improves_calibration(self):
        cal = fit_calibrator("platt", self.probs, self.labels)
        self.assertIsInstance(cal, PlattCalibrator)
        after = expected_calibration_error(cal(self.probs), self.labels)
        self.assertLess(after, expected_calibration_error(self.probs, self.labels))

    def test_isotonic_improves_calibration(self):
        cal = fit_calibrator("isotonic", self.probs, self.labels)
        self.assertIsInstance(cal, IsotonicCalibrator)
        after = expected_calibration_error(cal(self.probs), self.labels)
        self.assertLess(after, expected_calibration_error(self.probs, self.labels))

    def test_scalar_calibrators_preserve_ranking(self):
        """Temperature/Platt must not reorder frames: they change calibration only."""
        for name in ("temperature", "platt"):
            cal = fit_calibrator(name, self.probs, self.labels)
            out = cal(self.probs)
            order_in = np.argsort(self.probs, kind="mergesort")
            order_out = np.argsort(out, kind="mergesort")
            np.testing.assert_array_equal(order_in, order_out, err_msg=name)

    def test_isotonic_is_monotone_non_decreasing(self):
        cal = fit_calibrator("isotonic", self.probs, self.labels)
        grid = np.linspace(0.0, 1.0, 200)
        out = cal(grid)
        self.assertTrue(np.all(np.diff(out) >= -1e-12))

    def test_calibrators_stay_in_unit_interval(self):
        for name in ("identity", "temperature", "platt", "isotonic"):
            cal = fit_calibrator(name, self.probs, self.labels)
            out = np.asarray(cal(np.linspace(0.0, 1.0, 101)))
            self.assertTrue(np.all(out >= 0.0) and np.all(out <= 1.0), name)

    def test_unknown_calibrator_rejected(self):
        with self.assertRaises(ValueError):
            fit_calibrator("magic", self.probs, self.labels)

    def test_empty_input_falls_back_to_identity(self):
        self.assertIsInstance(fit_calibrator("platt", [], []), IdentityCalibrator)

    def test_calibrators_serialise(self):
        for name in ("identity", "temperature", "platt", "isotonic"):
            payload = fit_calibrator(name, self.probs, self.labels).as_dict()
            self.assertEqual(payload["name"], name)


class ReliabilityTests(unittest.TestCase):
    def test_perfect_calibration_has_zero_error(self):
        probs = np.repeat([0.05, 0.25, 0.45, 0.65, 0.85], 200)
        rng = np.random.default_rng(7)
        labels = rng.binomial(1, probs).astype(np.float64)
        # Sampling noise only; well under one bin width.
        self.assertLess(expected_calibration_error(probs, labels), 0.05)

    def test_bins_cover_all_frames(self):
        probs = np.linspace(0.0, 1.0, 1000)
        labels = np.zeros(1000)
        rows = reliability_bins(probs, labels, bins=10)
        self.assertEqual(sum(r["count"] for r in rows), 1000)

    def test_maximum_error_is_at_least_expected_error(self):
        rng = np.random.default_rng(3)
        probs = rng.random(500)
        labels = rng.binomial(1, 0.5, size=500).astype(np.float64)
        self.assertGreaterEqual(
            maximum_calibration_error(probs, labels) + 1e-12,
            expected_calibration_error(probs, labels),
        )


class FrameLabelTests(unittest.TestCase):
    def test_labels_mark_only_frames_within_tolerance(self):
        times = np.arange(0.0, 4.0, 0.25)
        labels = _frame_boundary_labels(times, [2.0], tolerance=0.25)
        marked = times[labels > 0]
        self.assertTrue(np.all(np.abs(marked - 2.0) <= 0.25 + 1e-9))
        self.assertEqual(len(marked), 3)  # 1.75, 2.00, 2.25

    def test_no_changes_gives_all_zero(self):
        labels = _frame_boundary_labels(np.arange(0.0, 2.0, 0.1), [])
        self.assertEqual(labels.sum(), 0.0)

    def test_multiple_changes_are_unioned(self):
        times = np.arange(0.0, 10.0, 0.5)
        labels = _frame_boundary_labels(times, [2.0, 7.0], tolerance=0.25)
        self.assertEqual(set(times[labels > 0].tolist()), {2.0, 7.0})


class PrfTests(unittest.TestCase):
    def test_zero_predictions_is_zero_not_error(self):
        self.assertEqual(_prf(0, 0, 10), {"precision": 0.0, "recall": 0.0, "f1": 0.0})

    def test_perfect_scores(self):
        self.assertEqual(_prf(10, 10, 10), {"precision": 1.0, "recall": 1.0, "f1": 1.0})


class PosteriorTests(unittest.TestCase):
    def test_states_sum_to_one(self):
        rng = np.random.default_rng(11)
        root = rng.dirichlet(np.ones(12), size=50)
        quality = rng.dirichlet(np.ones(3), size=50)
        nochord = rng.random(50)
        states = chord_posteriors(root, quality, nochord)
        self.assertEqual(states.shape, (50, 37))
        np.testing.assert_allclose(states.sum(axis=1), 1.0, atol=1e-9)


class DominantIssueTests(unittest.TestCase):
    def _diagnosis(self, *, mean_predicted, positive_rate, true_prob, false_prob,
                   agreement, f1_half, f1_best, best_threshold="0.25",
                   precision_half=0.75, recall_half=0.30):
        return {
            "calibration": {"meanPredicted": mean_predicted, "positiveRate": positive_rate},
            "categoryEvidence": {
                "trueTransition": {"boundaryProb": {"mean": true_prob}},
                "falseTransition": {"boundaryProb": {"mean": false_prob}},
            },
            "boundaryThresholdSweep": {
                "0.50": {"f1": f1_half, "precision": precision_half, "recall": recall_half},
                best_threshold: {"f1": f1_best, "precision": 0.75, "recall": 0.79},
            },
            "bestF1Threshold": best_threshold,
            "stateChangeBoundaryAgreement": agreement,
        }

    def test_detects_thresholding_and_recall_starvation(self):
        out = classify_dominant_issue(self._diagnosis(
            mean_predicted=0.165, positive_rate=0.180, true_prob=0.42, false_prob=0.23,
            agreement=0.27, f1_half=0.436, f1_best=0.770))
        self.assertIn("poor thresholding", out["findings"])
        self.assertIn("recall starvation at the operating threshold", out["findings"])
        self.assertAlmostEqual(out["trueFalseSeparation"], 0.19, places=4)

    def test_detects_overconfidence(self):
        out = classify_dominant_issue(self._diagnosis(
            mean_predicted=0.40, positive_rate=0.18, true_prob=0.6, false_prob=0.2,
            agreement=0.8, f1_half=0.70, f1_best=0.70, best_threshold="0.50"))
        self.assertIn("boundary overconfidence", out["findings"])

    def test_detects_weak_discrimination(self):
        out = classify_dominant_issue(self._diagnosis(
            mean_predicted=0.18, positive_rate=0.18, true_prob=0.30, false_prob=0.28,
            agreement=0.8, f1_half=0.50, f1_best=0.50, best_threshold="0.50"))
        self.assertIn("weak discrimination", out["findings"])

    def test_clean_model_triggers_nothing(self):
        out = classify_dominant_issue(self._diagnosis(
            mean_predicted=0.18, positive_rate=0.18, true_prob=0.70, false_prob=0.20,
            agreement=0.85, f1_half=0.80, f1_best=0.80, best_threshold="0.50",
            precision_half=0.80, recall_half=0.80))
        self.assertEqual(out["findings"], [])


class ReconciliationGuardTests(unittest.TestCase):
    def _entries(self):
        out = []
        for performer in ("guitarset-p01", "guitarset-p02", "guitarset-p03",
                          "guitarset-p04", "guitarset-p05"):
            index = performer[-2:]
            for capture in ("audio_mono-mic", "audio_mono-pickup_mix"):
                out.append(_Entry(performer, f"{index}_BN1", capture, f"lopo-p{index}"))
        return out

    def test_clean_set_passes(self):
        entries = self._entries()
        sealing = check_p00_sealed(entries)
        self.assertTrue(sealing["p00Sealed"])
        self.assertTrue(sealing["performersMatch"])
        pairing = check_capture_pairing(entries)
        self.assertTrue(pairing["allPaired"])
        self.assertTrue(pairing["pairingHeldWithinFold"])

    def test_p00_detected_by_performer_id(self):
        entries = self._entries() + [_Entry("guitarset-p00", "99_X", "audio_mono-mic")]
        sealing = check_p00_sealed(entries)
        self.assertFalse(sealing["p00Sealed"])
        self.assertEqual(sealing["p00EntriesByPerformerId"], 1)

    def test_p00_detected_by_performance_prefix_even_when_mislabelled(self):
        """A p00 performance relabelled under another performer must still trip."""
        entries = self._entries() + [_Entry("guitarset-p03", "00_BN1", "audio_mono-mic")]
        sealing = check_p00_sealed(entries)
        self.assertFalse(sealing["p00Sealed"])
        self.assertEqual(sealing["p00EntriesByPerformerId"], 0)
        self.assertEqual(sealing["p00EntriesByPerformanceIdPrefix"], 1)

    def test_unpaired_capture_detected(self):
        entries = [e for e in self._entries() if not (
            e.performance_id == "01_BN1" and e.capture == "audio_mono-pickup_mix")]
        pairing = check_capture_pairing(entries)
        self.assertFalse(pairing["allPaired"])
        self.assertIn("01_BN1", pairing["unpairedPerformances"])

    def test_performance_split_across_folds_detected(self):
        entries = self._entries()
        entries.append(_Entry("guitarset-p01", "01_BN1", "audio_mono-mic", "lopo-p02"))
        pairing = check_capture_pairing(entries)
        self.assertFalse(pairing["pairingHeldWithinFold"])
        self.assertIn("01_BN1", pairing["performancesSplitAcrossFolds"])


if __name__ == "__main__":
    unittest.main()
