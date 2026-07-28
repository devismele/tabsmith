r"""Full-band pilot tests: domain mixing, resume state, frozen config integrity.

The pilot is a long CPU run, so the parts that must not drift are the domain
sampling ratio (which the preservation gate depends on) and resume behaviour
(which decides whether an interruption costs an epoch or the whole run).

Run: .\.venv\Scripts\python.exe -m unittest discover -s ml/tests -t .
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml.full_band.pilot import (
    DEFAULT_PILOT,
    DomainSamples,
    load_pilot_config,
    mix_epoch,
    resume_state,
    write_state,
)


def _domain(name: str, count: int) -> DomainSamples:
    return DomainSamples(name, [f"{name}-{i}" for i in range(count)])


class DomainMixingTests(unittest.TestCase):
    def test_ratio_is_honoured(self):
        domains = [_domain("guitarset", 600), _domain("slakh", 240)]
        stream = mix_epoch(domains, {"guitarset": 0.5, "slakh": 0.5},
                           np.random.default_rng(0))
        slakh = sum(1 for s in stream if s.startswith("slakh"))
        self.assertAlmostEqual(slakh / len(stream), 0.5, places=2)

    def test_epoch_length_is_set_by_the_limiting_domain(self):
        """The ratio must be exact, not approximated by oversampling."""
        domains = [_domain("guitarset", 600), _domain("slakh", 240)]
        stream = mix_epoch(domains, {"guitarset": 0.5, "slakh": 0.5},
                           np.random.default_rng(0))
        # slakh has 240 at 50% -> epoch caps at 480, not 1200.
        self.assertEqual(len(stream), 480)

    def test_no_sample_is_repeated_within_an_epoch(self):
        domains = [_domain("guitarset", 100), _domain("slakh", 100)]
        stream = mix_epoch(domains, {"guitarset": 0.5, "slakh": 0.5},
                           np.random.default_rng(3))
        self.assertEqual(len(stream), len(set(stream)))

    def test_seeded_stream_is_reproducible(self):
        domains = [_domain("guitarset", 50), _domain("slakh", 50)]
        first = mix_epoch(domains, {"guitarset": 0.5, "slakh": 0.5}, np.random.default_rng(7))
        second = mix_epoch(domains, {"guitarset": 0.5, "slakh": 0.5}, np.random.default_rng(7))
        self.assertEqual(first, second)

    def test_different_epochs_draw_different_streams(self):
        domains = [_domain("guitarset", 50), _domain("slakh", 50)]
        first = mix_epoch(domains, {"guitarset": 0.5, "slakh": 0.5}, np.random.default_rng(1))
        second = mix_epoch(domains, {"guitarset": 0.5, "slakh": 0.5}, np.random.default_rng(2))
        self.assertNotEqual(first, second)

    def test_control_uses_one_domain_only(self):
        domains = [_domain("guitarset", 100), _domain("slakh", 100)]
        stream = mix_epoch(domains, {"guitarset": 1.0, "slakh": 0.0},
                           np.random.default_rng(0))
        self.assertTrue(all(s.startswith("guitarset") for s in stream))
        self.assertEqual(len(stream), 100)

    def test_empty_domain_is_skipped_not_fatal(self):
        domains = [_domain("guitarset", 50), DomainSamples("slakh", [])]
        stream = mix_epoch(domains, {"guitarset": 1.0, "slakh": 0.0},
                           np.random.default_rng(0))
        self.assertEqual(len(stream), 50)

    def test_all_domains_empty_raises(self):
        with self.assertRaises(ValueError):
            mix_epoch([DomainSamples("a", [])], {"a": 1.0}, np.random.default_rng(0))


class ResumeStateTests(unittest.TestCase):
    def test_missing_state_starts_at_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = resume_state(Path(tmp))
            self.assertEqual(state["completedEpochs"], 0)
            self.assertIsNone(state["bestDevLoss"])

    def test_state_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = {"completedEpochs": 3, "history": [{"epoch": 0}],
                       "bestDevLoss": 1.25, "bestEpoch": 1}
            write_state(Path(tmp), payload)
            self.assertEqual(resume_state(Path(tmp)), payload)

    def test_state_write_is_atomic(self):
        """A crash mid-write must not leave a truncated state file."""
        with tempfile.TemporaryDirectory() as tmp:
            write_state(Path(tmp), {"completedEpochs": 1, "history": [],
                                    "bestDevLoss": 2.0, "bestEpoch": 0})
            self.assertFalse((Path(tmp) / "pilot-state.json.tmp").exists())
            self.assertTrue((Path(tmp) / "pilot-state.json").exists())


class FrozenPilotConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = load_pilot_config()

    def test_config_is_the_committed_one(self):
        self.assertTrue(DEFAULT_PILOT.exists())
        self.assertEqual(self.config["gateSet"], "full-band-pilot-gates-20260727")

    def test_bounded_to_one_primary_and_one_control(self):
        roles = [c["role"] for c in self.config["candidates"]]
        self.assertEqual(roles.count("primary"), 1)
        self.assertEqual(roles.count("control"), 1)
        self.assertEqual(len(roles), 2)

    def test_domain_fractions_sum_to_one(self):
        sampling = self.config["domainSampling"]
        self.assertAlmostEqual(
            sampling["guitarSetFraction"] + sampling["slakhFraction"], 1.0, places=6)

    def test_p00_is_not_a_pilot_performer(self):
        self.assertNotIn("guitarset-p00", self.config["domainSampling"]["guitarSetPerformers"])

    def test_guitar_only_views_are_excluded_from_training(self):
        """Measured worse than full mix, so training on them is rejected."""
        views = self.config["domainSampling"]["slakhViews"]
        self.assertNotIn("oracle-guitar", views)
        self.assertNotIn("guitar-plus-bass", views)
        self.assertIn("full-mix", views)

    def test_both_candidates_start_from_v1(self):
        for candidate in self.config["candidates"]:
            self.assertEqual(candidate["initFrom"], "v1")

    def test_known_limitations_are_recorded(self):
        text = json.dumps(self.config["knownLimitations"])
        self.assertIn("note-level", text)
        self.assertIn("synthetic", text)


if __name__ == "__main__":
    unittest.main()
