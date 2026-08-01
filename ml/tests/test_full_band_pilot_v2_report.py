r"""Pilot-v2 reporting: retention by default, and no silently mismatched gates.

A reporting bug in a study like this is not cosmetic - the report is the record
the decision is read from. The properties asserted here are the ones that would
let a wrong decision look right: retention must be the default, an eligible
candidate must be named rather than implied, every evaluated candidate must
appear, and results scored under one gate set must never be reported under
another.

Run: .\.venv\Scripts\python.exe -m unittest discover -s ml/tests -t .
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ml.full_band.pilot_v2_report import _models, build_markdown, main


def _metrics(**overrides):
    base = {"rootAccuracy": 0.5, "detailedAccuracy": 0.4, "noChordF1": 0.3,
            "regionsPerMinute": 20.0, "fragmentationRate": 0.65}
    base.update(overrides)
    return base


def _outcome(passed: bool, cid: str):
    return {
        "checks": [{"gate": "guitarSetRootRegression", "scope": "audio_mono-mic",
                    "measured": 1.0 if passed else 9.9, "required": "<= 2.0 pp",
                    "passed": passed}],
        "notEvaluable": {"fullBandFragmentation": "reference labels are note-level"},
        "passed": passed,
        "failedGates": [] if passed else [f"guitarSetRootRegression@audio_mono-mic"],
        "firstFailure": None if passed else f"{cid} lost root accuracy",
    }


def _payload(eligible: list[str] | None = None, candidates=("rehearsal-heavy", "low-lr")):
    eligible = eligible or []
    results = {"v1": {"fullBand": {"full-mix": _metrics()},
                      "guitarset": {"audio_mono-mic": _metrics()}}}
    outcomes = {}
    for cid in candidates:
        results[cid] = {"fullBand": {"full-mix": _metrics(detailedAccuracy=0.55)},
                        "guitarset": {"audio_mono-mic": _metrics(rootAccuracy=0.49)}}
        outcomes[cid] = _outcome(cid in eligible, cid)
    return {"pilotId": "full-band-pilot-v2-20260731",
            "gateSetId": "full-band-pilot-gates-20260727",
            "developmentTracks": 40, "results": results,
            "gateOutcomes": outcomes, "eligibleCandidates": eligible}


PILOT = {"strategy": {"id": "preserve-root-identification-under-mixed-domain-finetune"},
         "gateSet": "full-band-pilot-gates-20260727"}


class ModelOrderTests(unittest.TestCase):
    def test_baseline_comes_first(self):
        self.assertEqual(_models(_payload())[0], "v1")

    def test_every_evaluated_candidate_is_listed(self):
        models = _models(_payload(candidates=("a", "b", "c")))
        self.assertEqual(models, ["v1", "a", "b", "c"])


class MarkdownTests(unittest.TestCase):
    def test_retention_is_the_default_headline(self):
        text = build_markdown(_payload(), PILOT)
        self.assertIn("**Decision: retain v1.", text)

    def test_an_eligible_candidate_is_named(self):
        text = build_markdown(_payload(eligible=["rehearsal-heavy"]), PILOT)
        self.assertIn("**Decision: rehearsal-heavy passed every evaluable gate.**", text)
        self.assertNotIn("retain v1.", text)

    def test_several_eligible_candidates_are_not_silently_ranked(self):
        """The gates admit rather than order; naming one would invent a choice."""
        text = build_markdown(
            _payload(eligible=["rehearsal-heavy", "low-lr"],
                     candidates=("rehearsal-heavy", "low-lr")), PILOT)
        self.assertIn("2 candidates passed", text)
        self.assertIn("does not select between them", text)
        self.assertNotIn("**Decision: rehearsal-heavy passed", text)

    def test_all_candidates_appear_in_the_tables(self):
        text = build_markdown(_payload(candidates=("rehearsal-heavy", "low-lr")), PILOT)
        self.assertIn("| rehearsal-heavy | full-mix |", text)
        self.assertIn("| low-lr | full-mix |", text)
        self.assertIn("| rehearsal-heavy | audio_mono-mic |", text)

    def test_failures_are_marked_not_buried(self):
        text = build_markdown(_payload(), PILOT)
        self.assertIn("**FAIL**", text)
        self.assertIn("rehearsal-heavy — FAILED", text)

    def test_it_states_the_gates_were_reused_unchanged(self):
        text = build_markdown(_payload(), PILOT)
        self.assertIn("reused", text)
        self.assertIn("not allowed to move", text)

    def test_not_evaluable_gates_are_reported_as_such(self):
        text = build_markdown(_payload(), PILOT)
        self.assertIn("Gates that could not be evaluated", text)
        self.assertIn("fullBandFragmentation", text)

    def test_narrative_is_appended_when_supplied(self):
        text = build_markdown(_payload(), PILOT, narrative="## Interpretation\n\nauthored.")
        self.assertTrue(text.rstrip().endswith("authored."))


class DecisionTests(unittest.TestCase):
    def _run(self, payload, tmp: Path, narrative: str | None = None):
        inp = tmp / "results.json"
        inp.write_text(json.dumps(payload), encoding="utf-8")
        cfg = tmp / "pilot.json"
        cfg.write_text(json.dumps(PILOT), encoding="utf-8")
        out, dec = tmp / "report.md", tmp / "decision.json"
        argv = ["--input", str(inp), "--pilot-config", str(cfg),
                "--output", str(out), "--decision", str(dec)]
        if narrative:
            path = tmp / "narrative.md"
            path.write_text(narrative, encoding="utf-8")
            argv += ["--narrative", str(path)]
        import sys
        original = sys.argv
        sys.argv = ["pilot_v2_report"] + argv
        try:
            main()
        finally:
            sys.argv = original
        return json.loads(dec.read_text(encoding="utf-8")), out.read_text(encoding="utf-8")

    def test_no_eligible_candidate_retains_v1(self):
        with tempfile.TemporaryDirectory() as tmp:
            decision, _ = self._run(_payload(), Path(tmp))
            self.assertEqual(decision["decision"], "retain-v1")
            self.assertIsNone(decision["selectedCandidate"])
            self.assertFalse(decision["productionWeightsReplaced"])
            self.assertFalse(decision["p00Accessed"])
            self.assertFalse(decision["slakhTestSplitUsed"])

    def test_several_eligible_candidates_leave_the_choice_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            decision, _ = self._run(
                _payload(eligible=["rehearsal-heavy", "low-lr"],
                         candidates=("rehearsal-heavy", "low-lr")), Path(tmp))
            self.assertEqual(decision["decision"], "multiple-eligible-no-ranking-rule")
            self.assertIsNone(decision["selectedCandidate"])
            self.assertEqual(sorted(decision["eligibleCandidates"]),
                             ["low-lr", "rehearsal-heavy"])

    def test_eligible_candidate_is_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            decision, _ = self._run(_payload(eligible=["rehearsal-heavy"]), Path(tmp))
            self.assertEqual(decision["decision"], "select-candidate")
            self.assertEqual(decision["selectedCandidate"], "rehearsal-heavy")

    def test_mismatched_gate_set_is_refused(self):
        """Scoring under one gate set and reporting under another must not happen."""
        payload = _payload()
        payload["gateSetId"] = "some-other-gate-set"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                self._run(payload, Path(tmp))

    def test_decision_records_a_results_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, _ = self._run(_payload(), Path(tmp))
            changed = _payload()
            changed["results"]["v1"]["fullBand"]["full-mix"]["rootAccuracy"] = 0.99
            second, _ = self._run(changed, Path(tmp))
            self.assertNotEqual(first["resultsChecksum"], second["resultsChecksum"])


if __name__ == "__main__":
    unittest.main()
