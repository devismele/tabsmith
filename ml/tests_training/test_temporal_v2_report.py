from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from ml.evaluation.temporal_v2_ablation import build_ablation_plan
from ml.evaluation.temporal_v2_report import (
    _paired_rows,
    frozen_configuration_checksum,
    pareto_frontier,
    validate_completed_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "ml" / "configs"


class TemporalV2ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = json.loads((CONFIG_ROOT / "temporal-harmony-v2.json").read_text())
        cls.splits = json.loads((CONFIG_ROOT / "temporal-harmony-v2-splits.json").read_text())
        cls.ablations = json.loads(
            (CONFIG_ROOT / "temporal-harmony-v2-ablations.json").read_text()
        )
        cls.augmentations = json.loads(
            (CONFIG_ROOT / "temporal-harmony-v2-augmentations.json").read_text()
        )

    def test_frozen_configuration_checksum_matches_pre_run_identity(self):
        self.assertEqual(
            frozen_configuration_checksum(),
            "b2450d5110e14b83d9c161bb8746cfbe2ceed167abce8f2906f0444cedc302d0",
        )

    def test_completed_matrix_rejects_duplicate_or_p00_rows(self):
        plan = build_ablation_plan(
            self.base, self.splits, self.ablations, self.augmentations
        )
        report = {
            **plan,
            "status": "completed",
            "validForSelection": True,
            "expectedRunCount": 30,
            "completedRunCount": 30,
            "failedRunCount": 0,
            "p00UsedForSelection": False,
            "foldResults": [],
        }
        with self.assertRaises(ValueError):
            validate_completed_matrix(
                report, self.base, self.splits, self.ablations, self.augmentations
            )

    def test_pareto_frontier_keeps_accuracy_stability_tradeoff(self):
        def row(root, detail, frag, density, boundary):
            return {
                "pairedPerformances": {
                    "rootAccuracy": root,
                    "detailedAccuracy": detail,
                    "fragmentationRate": frag,
                    "regionsPerMinute": density,
                    "meanAbsoluteBoundaryErrorMs": boundary,
                }
            }

        frontier = pareto_frontier({
            "stable": row(0.6, 0.5, 0.4, 10, 600),
            "accurate": row(0.8, 0.7, 0.5, 12, 400),
            "dominated": row(0.7, 0.6, 0.6, 14, 500),
        })
        self.assertEqual(set(frontier), {"stable", "accurate"})

    def test_paired_capture_aggregation_preserves_missing_boundary_error(self):
        base = {
            "foldId": "lopo-p01",
            "trackId": "01_test",
            "evaluatedDurationSeconds": 10.0,
            "rootAccuracy": 0.5,
            "majorMinorAccuracy": 0.5,
            "detailedAccuracy": 0.5,
            "chordSymbolRecall": 0.5,
            "noChordPrecision": 0.0,
            "noChordRecall": 0.0,
            "meanBoundaryErrorMs": None,
            "medianBoundaryErrorMs": None,
            "meanSignedBoundaryErrorMs": None,
            "fragmentationRate": 0.0,
            "referenceRegions": 1,
            "predictedRegions": 1,
            "missingChords": 0,
            "extraChords": 0,
            "oneWindowRegionCount": 0,
            "shortRegionCount": 0,
            "flickerCount": 0,
            "meanTopChordConfidence": 0.5,
            "meanLearnedEntropy": 0.5,
            "meanFrameSwitchProbability": 0.0,
            "meanBoundaryProbability": 0.0,
            "boundaryF1": {
                tolerance: {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                for tolerance in ("100ms", "250ms", "500ms", "1000ms")
            },
        }
        paired = _paired_rows([
            {**base, "capture": "audio_mono-mic"},
            {**base, "capture": "audio_mono-pickup_mix"},
        ])
        self.assertEqual(len(paired), 1)
        self.assertIsNone(paired[0]["meanBoundaryErrorMs"])


if __name__ == "__main__":
    unittest.main()
