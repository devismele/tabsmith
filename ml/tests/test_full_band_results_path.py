r"""One pilot's frozen results must not be overwritten by a later pilot's run.

The default results filename predates pilot v2, so pointing v2's evaluation at
the reports directory would have silently replaced the file that pilot v1's
committed decision was read from. Silent because nothing downstream compares
pilot ids - the report would simply describe different numbers under the same
name.

Run: .\.venv\Scripts\python.exe -m unittest discover -s ml/tests -t .
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ml.full_band.evaluate_pilot import resolve_results_path

V1 = "full-band-pilot-v1-20260728"
V2 = "full-band-pilot-v2-20260731"


class ResolveResultsPathTests(unittest.TestCase):
    def test_default_name_is_used_when_no_output_given(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = resolve_results_path(Path(tmp), None, V1)
            self.assertEqual(path.name, "full-band-pilot-results.json")

    def test_explicit_output_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "full-band-pilot-v2-results.json"
            self.assertEqual(resolve_results_path(Path(tmp), str(target), V2), target)

    def test_rewriting_the_same_pilot_is_allowed(self):
        """Re-running an evaluation of the same pilot must stay possible."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "full-band-pilot-results.json"
            path.write_text(json.dumps({"pilotId": V1}), encoding="utf-8")
            self.assertEqual(resolve_results_path(Path(tmp), None, V1), path)

    def test_overwriting_another_pilots_results_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "full-band-pilot-results.json"
            path.write_text(json.dumps({"pilotId": V1}), encoding="utf-8")
            with self.assertRaises(SystemExit):
                resolve_results_path(Path(tmp), None, V2)

    def test_the_refusal_names_both_pilots(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "full-band-pilot-results.json"
            path.write_text(json.dumps({"pilotId": V1}), encoding="utf-8")
            with self.assertRaises(SystemExit) as caught:
                resolve_results_path(Path(tmp), None, V2)
            message = str(caught.exception)
            self.assertIn(V1, message)
            self.assertIn(V2, message)

    def test_an_explicit_output_is_guarded_too(self):
        """Passing --output does not license clobbering someone else's results."""
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "chosen.json"
            target.write_text(json.dumps({"pilotId": V1}), encoding="utf-8")
            with self.assertRaises(SystemExit):
                resolve_results_path(Path(tmp), str(target), V2)

    def test_a_file_without_a_pilot_id_is_not_treated_as_owned(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "full-band-pilot-results.json"
            path.write_text(json.dumps({"schemaVersion": 1}), encoding="utf-8")
            self.assertEqual(resolve_results_path(Path(tmp), None, V2), path)

    def test_unreadable_json_does_not_crash_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "full-band-pilot-results.json"
            path.write_text("{ truncated", encoding="utf-8")
            self.assertEqual(resolve_results_path(Path(tmp), None, V2), path)


if __name__ == "__main__":
    unittest.main()
