r"""Freeze-decision tests: retention must be the default and gates must bind.

The freeze record is the artifact that authorises (or forbids) opening p00, so
these tests pin the rules that matter: it reads the frozen gate outcomes rather
than re-deriving them, it cannot select a candidate that failed a gate, and it
keeps p00 sealed whenever nothing passed.

Run: .\.venv\Scripts\python.exe -m unittest discover -s ml/tests -t .
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from ml.evaluation.boundary.freeze_decision import build_markdown, decide

CAPTURES = ("audio_mono-mic", "audio_mono-pickup_mix")
ML_ROOT = Path(__file__).resolve().parents[1]


def _manifest():
    return json.loads(
        (ML_ROOT / "configs" / "boundary-calibration-v3-candidates.json").read_text(
            encoding="utf-8"))


def _gates():
    return json.loads(
        (ML_ROOT / "configs" / "boundary-calibration-v3-gates.json").read_text(
            encoding="utf-8"))


def _evaluation(frag=0.60, detailed=0.71):
    return {
        "foldCalibration": {
            "segmental-full|lopo-p01": {
                "calibrator": {"name": "isotonic", "knots": 12},
                "operatingThreshold": 0.5,
            }
        },
        "decoders": {
            "segmental-full": {
                "byCapture": {c: {
                    "rootAccuracy": 0.76, "detailedAccuracy": detailed,
                    "fragmentationRate": frag, "regionsPerMinute": 19.5,
                    "meanAbsoluteBoundaryErrorMs": 480.0,
                } for c in CAPTURES}
            }
        },
    }


def _results(gate_results, evaluations=None):
    eligible = [cid for cid, r in gate_results.items() if r["eligible"]]
    return {
        "evaluations": evaluations or {
            "full-v2-control": _evaluation(frag=0.6486, detailed=0.7133),
            **{cid: _evaluation() for cid in gate_results},
        },
        "gateResults": gate_results,
        "eligibleCandidates": eligible,
    }


def _gate(eligible, failure=None):
    return {"eligible": eligible,
            "failedGates": [] if eligible else ["fragmentationRateMaximum@audio_mono-mic"],
            "firstFailure": None if eligible else (failure or "fragmentation 0.6461 > 0.635")}


class RetentionTests(unittest.TestCase):
    def test_no_eligible_candidate_retains_v1_and_seals_p00(self):
        results = _results({"precision-boundary-loss": _gate(False),
                            "agreement-coupled": _gate(False)})
        record = decide(results, _manifest(), _gates(), run_dir=Path("run"))
        self.assertEqual(record["decision"], "retain-v1")
        self.assertIsNone(record["selectedCandidate"])
        self.assertTrue(record["p00Sealed"])
        self.assertFalse(record["p00Accessed"])
        self.assertEqual(record["retained"], "v1")
        self.assertNotIn("frozenConfiguration", record)

    def test_retention_never_touches_production(self):
        results = _results({"agreement-coupled": _gate(False)})
        record = decide(results, _manifest(), _gates(), run_dir=Path("run"))
        self.assertFalse(record["productionWeightsReplaced"])
        self.assertFalse(record["applicationDefaultsChanged"])
        self.assertFalse(record["releaseGatingChanged"])

    def test_empty_gate_results_retains(self):
        record = decide(_results({}), _manifest(), _gates(), run_dir=Path("run"))
        self.assertEqual(record["decision"], "retain-v1")
        self.assertTrue(record["p00Sealed"])


class SelectionTests(unittest.TestCase):
    def test_single_eligible_candidate_is_selected(self):
        results = _results({"precision-boundary-loss": _gate(True),
                            "agreement-coupled": _gate(False)})
        record = decide(results, _manifest(), _gates(), run_dir=Path("run"))
        self.assertEqual(record["decision"], "select-candidate")
        self.assertEqual(record["selectedCandidate"], "precision-boundary-loss")
        self.assertFalse(record["p00Sealed"])
        self.assertIn("frozenConfiguration", record)

    def test_selection_freezes_calibration_and_thresholds(self):
        results = _results({"precision-boundary-loss": _gate(True)})
        record = decide(results, _manifest(), _gates(), run_dir=Path("run"))
        frozen = record["frozenConfiguration"]
        self.assertEqual(frozen["decoder"], "segmental-full")
        self.assertIn("segmental-full|lopo-p01", frozen["boundaryCalibration"])
        self.assertIn("segmental-full|lopo-p01", frozen["operatingThresholds"])
        self.assertIn("metrics", frozen)

    def test_tie_break_prefers_lower_fragmentation_among_passers(self):
        evaluations = {
            "full-v2-control": _evaluation(frag=0.6486),
            "precision-boundary-loss": _evaluation(frag=0.62),
            "agreement-coupled": _evaluation(frag=0.58),
        }
        results = _results({"precision-boundary-loss": _gate(True),
                            "agreement-coupled": _gate(True)}, evaluations)
        record = decide(results, _manifest(), _gates(), run_dir=Path("run"))
        self.assertEqual(record["selectedCandidate"], "agreement-coupled")

    def test_selection_authorises_p00_only_as_a_next_step(self):
        results = _results({"precision-boundary-loss": _gate(True)})
        record = decide(results, _manifest(), _gates(), run_dir=Path("run"))
        self.assertFalse(record["p00Accessed"])
        self.assertIn("sealed p00 run", record["nextStep"])


class IntegrityTests(unittest.TestCase):
    def test_tampered_eligibility_list_is_rejected(self):
        """A results file claiming eligibility its checks do not support must fail."""
        results = _results({"agreement-coupled": _gate(False)})
        results["eligibleCandidates"] = ["agreement-coupled"]
        with self.assertRaises(ValueError):
            decide(results, _manifest(), _gates(), run_dir=Path("run"))

    def test_dropped_eligibility_entry_is_rejected(self):
        results = _results({"agreement-coupled": _gate(True)})
        results["eligibleCandidates"] = []
        with self.assertRaises(ValueError):
            decide(results, _manifest(), _gates(), run_dir=Path("run"))

    def test_record_carries_governing_checksums(self):
        results = _results({"agreement-coupled": _gate(False)})
        record = decide(results, _manifest(), _gates(), run_dir=Path("run"))
        for key in ("candidateManifestChecksum", "gateSetChecksum", "resultsChecksum"):
            self.assertEqual(len(record[key]), 64, key)


class MarkdownTests(unittest.TestCase):
    def test_retention_document_states_the_decision_plainly(self):
        results = _results({"agreement-coupled": _gate(False)})
        record = decide(results, _manifest(), _gates(), run_dir=Path("run"))
        text = build_markdown(record, results)
        self.assertIn("retain v1", text.lower())
        self.assertIn("p00 remains sealed", text.lower())
        self.assertIn("agreement-coupled", text)

    def test_selection_document_lists_checkpoints_section(self):
        results = _results({"precision-boundary-loss": _gate(True)})
        record = decide(results, _manifest(), _gates(), run_dir=Path("run"))
        text = build_markdown(record, results)
        self.assertIn("Frozen configuration", text)
        self.assertIn("segmental-full", text)


if __name__ == "__main__":
    unittest.main()
