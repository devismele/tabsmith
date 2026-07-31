r"""Training one candidate at a time must not drop the others from the report.

The pilot report is what the evaluator reads to discover which candidates exist.
A long CPU pilot is naturally run one candidate per invocation, and each
invocation rewrote the report with only what it had just trained - so the gates
would have been applied to a subset while the report still looked complete.
That is the shape of error this study cannot detect by reading its own output.

Run: .\.venv\Scripts\python.exe -m unittest discover -s ml/tests -t .
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ml.full_band.run_pilot import merge_results


def _entry(cid: str, best: float = 4.0) -> dict:
    return {"candidateId": cid, "bestDevLoss": best, "epochsRun": 12}


def _write_report(path: Path, entries: list[dict]) -> None:
    path.write_text(json.dumps({"results": entries}), encoding="utf-8")


def _with_checkpoints(run_dir: Path, *cids: str) -> None:
    for cid in cids:
        (run_dir / cid).mkdir(parents=True, exist_ok=True)
        (run_dir / cid / "model.pt").write_bytes(b"weights")


class MergeResultsTests(unittest.TestCase):
    def test_no_previous_report_returns_the_fresh_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            merged = merge_results(run_dir / "pilot-report.json", [_entry("a")], run_dir)
            self.assertEqual([e["candidateId"] for e in merged], ["a"])

    def test_earlier_candidates_are_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            report = run_dir / "pilot-report.json"
            _write_report(report, [_entry("rehearsal-heavy"), _entry("root-anchored")])
            _with_checkpoints(run_dir, "rehearsal-heavy", "root-anchored", "low-lr")
            merged = merge_results(report, [_entry("low-lr")], run_dir)
            self.assertEqual([e["candidateId"] for e in merged],
                             ["low-lr", "rehearsal-heavy", "root-anchored"])

    def test_a_retrained_candidate_uses_the_fresh_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            report = run_dir / "pilot-report.json"
            _write_report(report, [_entry("a", best=9.9)])
            _with_checkpoints(run_dir, "a")
            merged = merge_results(report, [_entry("a", best=1.1)], run_dir)
            self.assertEqual(len(merged), 1)
            self.assertEqual(merged[0]["bestDevLoss"], 1.1)

    def test_a_candidate_without_a_checkpoint_is_dropped(self):
        """A deleted run must not linger as a result that no longer exists."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            report = run_dir / "pilot-report.json"
            _write_report(report, [_entry("gone")])
            merged = merge_results(report, [_entry("fresh")], run_dir)
            self.assertEqual([e["candidateId"] for e in merged], ["fresh"])

    def test_results_are_ordered_deterministically(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            report = run_dir / "pilot-report.json"
            _write_report(report, [_entry("c"), _entry("a")])
            _with_checkpoints(run_dir, "a", "b", "c")
            merged = merge_results(report, [_entry("b")], run_dir)
            self.assertEqual([e["candidateId"] for e in merged], ["a", "b", "c"])

    def test_all_three_v2_candidates_survive_three_separate_invocations(self):
        """The exact sequence this pilot was run in."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            report = run_dir / "pilot-report.json"
            _with_checkpoints(run_dir, "rehearsal-heavy",
                              "root-anchored-distillation", "low-learning-rate")
            accumulated: list[dict] = []
            for cid in ("rehearsal-heavy", "root-anchored-distillation", "low-learning-rate"):
                accumulated = merge_results(report, [_entry(cid)], run_dir)
                _write_report(report, accumulated)
            self.assertEqual([e["candidateId"] for e in accumulated],
                             ["low-learning-rate", "rehearsal-heavy",
                              "root-anchored-distillation"])


if __name__ == "__main__":
    unittest.main()
