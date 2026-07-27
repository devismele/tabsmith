r"""Boundary-calibration study evaluation and gate tests (offline, no torch).

Cover the frozen-gate application, the decoder-specific smoothing policy that
keeps the control faithful to the published baselines, and the cache-routing
rule that decides which candidates reuse the frozen full-v2 responses.

Run: .\.venv\Scripts\python.exe -m unittest discover -s ml/tests -t .
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from ml.evaluation.boundary.evaluate import DECODER_SMOOTHING, candidate_cache_dir
from ml.evaluation.boundary.gates import evaluate_candidate_gates, load_gates
from ml.evaluation.boundary.train_candidates import (
    trainable_candidates,
    validate_candidate_manifest,
)

CAPTURES = ("audio_mono-mic", "audio_mono-pickup_mix")
ML_ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ML_ROOT / "configs" / "boundary-calibration-v3-candidates.json"


def _passing_metrics():
    return {
        "rootAccuracy": 0.76, "detailedAccuracy": 0.71, "fragmentationRate": 0.60,
        "regionsPerMinute": 19.5, "meanAbsoluteBoundaryErrorMs": 480.0, "flickerCount": 2,
    }


def _decoder_results(**overrides):
    metrics = {**_passing_metrics(), **overrides.pop("metrics", {})}
    boundary = {
        "precision": 0.80, "recall": 0.80, "f1": 0.80,
        "stateChangeBoundaryAgreement": 0.80, "expectedCalibrationError": 0.01,
        **overrides.pop("boundary", {}),
    }
    taxonomy = {"false_long": 300, "missed": 900, **overrides.pop("taxonomy", {})}
    per_performer = overrides.pop("perPerformer", {f"guitarset-p0{i}": {"rootAccuracy": 0.75}
                                                  for i in range(1, 6)})
    return {
        "byCapture": {c: dict(metrics) for c in CAPTURES},
        "boundaryQuality": {c: dict(boundary) for c in CAPTURES},
        "regionTaxonomy": {c: dict(taxonomy) for c in CAPTURES},
        "perPerformer": {c: per_performer for c in CAPTURES},
    }


BASELINE_FULL_V2 = {c: {"flickerCount": 5} for c in CAPTURES}
BASELINE_BOUNDARY = {
    "precision": {c: 0.74 for c in CAPTURES},
    "recall": {c: 0.79 for c in CAPTURES},
    "agreement": {c: 0.67 for c in CAPTURES},
    "falseLong": {c: 450 for c in CAPTURES},
}


class GateTests(unittest.TestCase):
    def setUp(self):
        self.gates = load_gates()

    def _run(self, results):
        return evaluate_candidate_gates(
            "x", results, gates=self.gates,
            baseline_full_v2=BASELINE_FULL_V2, baseline_boundary=BASELINE_BOUNDARY)

    def test_all_gates_passing_is_eligible(self):
        self.assertTrue(self._run(_decoder_results())["eligible"])

    def test_fragmentation_gate_blocks(self):
        out = self._run(_decoder_results(metrics={"fragmentationRate": 0.6461}))
        self.assertFalse(out["eligible"])
        self.assertIn("fragmentationRateMaximum@audio_mono-mic", out["failedGates"])

    def test_regions_per_minute_gate_blocks(self):
        out = self._run(_decoder_results(metrics={"regionsPerMinute": 20.93}))
        self.assertFalse(out["eligible"])
        self.assertIn("regionsPerMinuteMaximum@audio_mono-mic", out["failedGates"])

    def test_accuracy_cannot_buy_off_stability(self):
        """A candidate cannot pass by being very accurate but fragmented."""
        out = self._run(_decoder_results(
            metrics={"rootAccuracy": 0.95, "detailedAccuracy": 0.94, "fragmentationRate": 0.70}))
        self.assertFalse(out["eligible"])

    def test_boundary_recall_collapse_blocks(self):
        out = self._run(_decoder_results(boundary={"recall": 0.50}))
        self.assertFalse(out["eligible"])
        self.assertIn("boundaryRecallDoesNotCollapse@audio_mono-mic", out["failedGates"])

    def test_boundary_precision_must_materially_improve(self):
        # Equal to the baseline is not "materially improved".
        out = self._run(_decoder_results(boundary={"precision": 0.74}))
        self.assertFalse(out["eligible"])
        self.assertIn("boundaryPrecisionMateriallyImproves@audio_mono-mic", out["failedGates"])

    def test_agreement_must_materially_improve(self):
        out = self._run(_decoder_results(boundary={"stateChangeBoundaryAgreement": 0.70}))
        self.assertFalse(out["eligible"])
        self.assertIn("stateChangeBoundaryAgreementImproves@audio_mono-mic", out["failedGates"])

    def test_false_long_must_drop_materially(self):
        out = self._run(_decoder_results(taxonomy={"false_long": 440}))
        self.assertFalse(out["eligible"])
        self.assertIn("falseLongMateriallyLower@audio_mono-mic", out["failedGates"])

    def test_flicker_must_be_below_full_v2(self):
        out = self._run(_decoder_results(metrics={"flickerCount": 5}))
        self.assertFalse(out["eligible"])

    def test_per_performer_collapse_blocks(self):
        out = self._run(_decoder_results(
            perPerformer={"guitarset-p01": {"rootAccuracy": 0.42},
                          "guitarset-p02": {"rootAccuracy": 0.80}}))
        self.assertFalse(out["eligible"])
        self.assertIn("perPerformerRootAccuracyMinimum@audio_mono-mic", out["failedGates"])

    def test_failure_reports_the_deciding_number(self):
        out = self._run(_decoder_results(metrics={"fragmentationRate": 0.6461}))
        self.assertIn("0.6461", out["firstFailure"])
        self.assertIn("0.635", out["firstFailure"])

    def test_gates_must_pass_on_both_captures(self):
        results = _decoder_results()
        results["byCapture"]["audio_mono-pickup_mix"]["fragmentationRate"] = 0.70
        out = self._run(results)
        self.assertFalse(out["eligible"])
        self.assertEqual(out["failedGates"], ["fragmentationRateMaximum@audio_mono-pickup_mix"])


class DecoderSmoothingTests(unittest.TestCase):
    def test_full_v2_path_smooths_and_segmental_does_not(self):
        """segmental-full is frozen as an unsmoothed decoder; keep it that way.

        Smoothing it would silently redefine the published baseline and stop the
        control from reproducing segmental-v3's numbers.
        """
        self.assertEqual(DECODER_SMOOTHING["full-v2-existing"],
                         {"method": "ema", "alpha": 0.65, "boundaryAlpha": 0.35})
        self.assertIsNone(DECODER_SMOOTHING["segmental-full"])


class CacheRoutingTests(unittest.TestCase):
    def test_checkpoint_reusers_share_the_frozen_cache(self):
        reuser = {"id": "calibrated-boundary", "reusesCheckpoint": "full-v2"}
        retrained = {"id": "precision-boundary-loss"}
        run_dir = Path("run")
        self.assertEqual(candidate_cache_dir(reuser, run_dir).name, "cache")
        self.assertNotEqual(candidate_cache_dir(retrained, run_dir),
                            candidate_cache_dir(reuser, run_dir))
        self.assertEqual(candidate_cache_dir(retrained, run_dir).name, "precision-boundary-loss")


class FrozenManifestTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(CANDIDATES.read_text(encoding="utf-8"))

    def test_frozen_manifest_is_valid_and_bounded(self):
        validate_candidate_manifest(self.manifest)
        self.assertLessEqual(len(self.manifest["candidates"]), 5)

    def test_manifest_forbids_p00_and_production_mutation(self):
        constraints = self.manifest["constraints"]
        self.assertFalse(constraints["p00UsedForSelection"])
        self.assertFalse(constraints["productionMutation"])
        self.assertFalse(constraints["hybridDefaultMutation"])
        self.assertNotIn("guitarset-p00", constraints["modelSelectionPerformers"])

    def test_exported_head_contract_is_unchanged(self):
        self.assertEqual(self.manifest["constraints"]["exportedHeads"],
                         ["root", "quality", "nochord", "boundary"])
        self.assertEqual(self.manifest["constraints"]["auxiliaryTrainingOnlyHeads"], [])

    def test_only_non_reusing_candidates_are_trained(self):
        trainable = [c["id"] for c in trainable_candidates(self.manifest)]
        self.assertNotIn("full-v2-control", trainable)
        self.assertNotIn("calibrated-boundary", trainable)
        self.assertIn("precision-boundary-loss", trainable)
        self.assertIn("agreement-coupled", trainable)

    def test_manifest_rejects_p00_in_selection_performers(self):
        broken = json.loads(CANDIDATES.read_text(encoding="utf-8"))
        broken["constraints"]["modelSelectionPerformers"].append("guitarset-p00")
        with self.assertRaises(ValueError):
            validate_candidate_manifest(broken)

    def test_manifest_rejects_enabling_p00_selection(self):
        broken = json.loads(CANDIDATES.read_text(encoding="utf-8"))
        broken["constraints"]["p00UsedForSelection"] = True
        with self.assertRaises(ValueError):
            validate_candidate_manifest(broken)

    def test_manifest_rejects_an_unbounded_candidate_set(self):
        broken = json.loads(CANDIDATES.read_text(encoding="utf-8"))
        broken["candidates"] = broken["candidates"] * 3
        with self.assertRaises(ValueError):
            validate_candidate_manifest(broken)


if __name__ == "__main__":
    unittest.main()
