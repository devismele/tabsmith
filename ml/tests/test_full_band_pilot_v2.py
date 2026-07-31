r"""Pilot-v2 frozen-config integrity and per-candidate domain sampling.

Pilot v1 failed one gate: GuitarSet root preservation. The two things that must
not drift in v2 are therefore the gate set (reusing it unchanged is what makes
the v2 result comparable rather than self-graded) and the per-candidate sampling
ratio, since that ratio is itself one of the interventions under test.

Torch-free by design; the distillation term is exercised in ml/tests_training.

Run: .\.venv\Scripts\python.exe -m unittest discover -s ml/tests -t .
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from ml.full_band.pilot import ML_ROOT, load_pilot_config, resolve_fractions

PILOT_V2 = ML_ROOT / "configs" / "full-band-pilot-v2.json"
PILOT_V1 = ML_ROOT / "configs" / "full-band-pilot-v1.json"
GATES = ML_ROOT / "configs" / "full-band-pilot-gates.json"


class FrozenPilotV2ConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = load_pilot_config(PILOT_V2)

    def test_config_is_the_committed_one(self):
        self.assertTrue(PILOT_V2.exists())
        self.assertEqual(self.config["pilotId"], "full-band-pilot-v2-20260731")

    def test_gate_set_is_reused_unchanged_from_v1(self):
        """The gate v1 failed on is not allowed to move for v2."""
        gates = json.loads(GATES.read_text(encoding="utf-8"))
        self.assertEqual(self.config["gateSet"], gates["gateSetId"])
        self.assertEqual(self.config["gateSet"],
                         load_pilot_config(PILOT_V1)["gateSet"])

    def test_root_regression_gate_still_two_percentage_points(self):
        gates = json.loads(GATES.read_text(encoding="utf-8"))
        preservation = gates["preservationGatesVsV1GuitarSetDevelopment"]
        self.assertEqual(
            preservation["rootAccuracyRegressionMaximumPercentagePoints"], 2.0)

    def test_p00_is_not_a_pilot_performer(self):
        self.assertNotIn("guitarset-p00",
                         self.config["domainSampling"]["guitarSetPerformers"])

    def test_every_candidate_starts_from_v1(self):
        for candidate in self.config["candidates"]:
            self.assertEqual(candidate["initFrom"], "v1")

    def test_candidate_ids_are_unique(self):
        ids = [c["id"] for c in self.config["candidates"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_schedule_is_shared_so_candidates_stay_comparable(self):
        for candidate in self.config["candidates"]:
            overrides = candidate["trainingOverrides"]
            self.assertEqual(overrides["epochs"], self.config["sharedSchedule"]["epochs"])
            self.assertEqual(overrides["earlyStoppingPatience"],
                             self.config["sharedSchedule"]["earlyStoppingPatience"])

    def test_exactly_one_lever_moves_per_candidate(self):
        """Each candidate changes sampling, or the anchor, or the step size."""
        base_lr = 0.0005
        for candidate in self.config["candidates"]:
            levers = 0
            if candidate.get("domainSampling"):
                levers += 1
            if candidate.get("preservation"):
                levers += 1
            if candidate["trainingOverrides"]["learningRate"] != base_lr:
                levers += 1
            self.assertEqual(levers, 1, f"{candidate['id']} moves {levers} levers")

    def test_guitar_only_views_stay_excluded_from_training(self):
        views = self.config["domainSampling"]["slakhViews"]
        self.assertNotIn("oracle-guitar", views)
        self.assertIn("full-mix", views)

    def test_cpu_budget_deviation_is_declared_not_silent(self):
        deviation = self.config["cpuBudget"]["deviationFromGateSetCpuPolicy"]
        self.assertIn("35.7 minutes", deviation["justification"])
        self.assertIn("no eligibility gate", deviation["unchanged"].lower())

    def test_untuned_distillation_weight_is_recorded_as_a_limit(self):
        text = json.dumps(self.config["knownLimitations"])
        self.assertIn("untuned", text)
        self.assertIn("synthetic", text)


class FractionResolutionTests(unittest.TestCase):
    def test_candidate_override_wins(self):
        config = {"domainSampling": {"guitarSetFraction": 0.5, "slakhFraction": 0.5}}
        candidate = {"id": "rehearsal-heavy",
                     "domainSampling": {"guitarSetFraction": 0.7, "slakhFraction": 0.3}}
        self.assertEqual(resolve_fractions(config, candidate),
                         {"guitarset": 0.7, "slakh": 0.3})

    def test_candidate_without_override_takes_the_pilot_default(self):
        config = {"domainSampling": {"guitarSetFraction": 0.5, "slakhFraction": 0.5}}
        candidate = {"id": "root-anchored-distillation"}
        self.assertEqual(resolve_fractions(config, candidate),
                         {"guitarset": 0.5, "slakh": 0.5})

    def test_v1_control_keeps_its_guitarset_only_ratio(self):
        """The v1 runner hardcoded this; resolution must reproduce it exactly."""
        config = {"domainSampling": {"guitarSetFraction": 0.5, "slakhFraction": 0.5}}
        candidate = {"id": "guitarset-only-control"}
        self.assertEqual(resolve_fractions(config, candidate),
                         {"guitarset": 1.0, "slakh": 0.0})

    def test_v1_candidates_resolve_to_the_ratios_v1_actually_ran(self):
        config = load_pilot_config(PILOT_V1)
        resolved = {c["id"]: resolve_fractions(config, c) for c in config["candidates"]}
        self.assertEqual(resolved["mixed-domain-finetune"],
                         {"guitarset": 0.5, "slakh": 0.5})
        self.assertEqual(resolved["guitarset-only-control"],
                         {"guitarset": 1.0, "slakh": 0.0})

    def test_fractions_that_do_not_sum_to_one_are_rejected(self):
        config = {"domainSampling": {"guitarSetFraction": 0.5, "slakhFraction": 0.5}}
        candidate = {"id": "bad",
                     "domainSampling": {"guitarSetFraction": 0.7, "slakhFraction": 0.4}}
        with self.assertRaises(ValueError):
            resolve_fractions(config, candidate)

    def test_every_v2_candidate_resolves(self):
        config = load_pilot_config(PILOT_V2)
        for candidate in config["candidates"]:
            fractions = resolve_fractions(config, candidate)
            self.assertAlmostEqual(sum(fractions.values()), 1.0, places=6)
            self.assertGreater(fractions["slakh"], 0.0,
                               f"{candidate['id']} must keep full-band exposure")


class RehearsalHeavyExposureTests(unittest.TestCase):
    """The 0.7/0.3 candidate must not buy preservation by seeing less Slakh.

    The epoch mixer sizes an epoch by the domain that can supply its share
    without repetition, so with the measured pool sizes (600 GuitarSet, 240
    Slakh) a 0.7/0.3 split still draws all 240 Slakh samples. That is what makes
    it an isolated rehearsal change rather than a dilution of the new domain.
    """

    def test_full_band_exposure_matches_the_v1_primary(self):
        import numpy as np

        from ml.full_band.pilot import DomainSamples, mix_epoch

        def pools():
            return [DomainSamples("guitarset", [f"g{i}" for i in range(600)]),
                    DomainSamples("slakh", [f"s{i}" for i in range(240)])]

        v1 = mix_epoch(pools(), {"guitarset": 0.5, "slakh": 0.5}, np.random.default_rng(0))
        v2 = mix_epoch(pools(), {"guitarset": 0.7, "slakh": 0.3}, np.random.default_rng(0))
        v1_slakh = sum(1 for s in v1 if s.startswith("s"))
        v2_slakh = sum(1 for s in v2 if s.startswith("s"))
        self.assertEqual(v1_slakh, 240)
        self.assertEqual(v2_slakh, 240)
        self.assertEqual(sum(1 for s in v1 if s.startswith("g")), 240)
        self.assertEqual(sum(1 for s in v2 if s.startswith("g")), 560)


if __name__ == "__main__":
    unittest.main()
