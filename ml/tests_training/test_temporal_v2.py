"""Temporal-harmony-v2 objectives, augmentation, smoothing, and gate tests."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ml.evaluation.probability_smoothing import smooth_learned_probabilities
from ml.evaluation.temporal_v2_ablation import (
    _load_completed_run,
    _write_json_atomic,
    build_ablation_plan,
    candidate_config,
    evaluate_selection_gates,
    validate_split_manifest,
)
from ml.evaluation.temporal_v2_baseline import verify_baseline
from ml.models.temporal_baseline import ModelConfig, TemporalBaseline
from ml.preprocessing.features import FeatureFrames
from ml.schema import ChordRegion, Track
from ml.training.augmentation import (
    augment_sample,
    build_augmented_samples,
    load_augmentation_manifest,
)
from ml.training.dataset import FrameSample, build_frame_targets, collate
from ml.training.losses import combined_loss, predicted_switch_probability
from ml.training.train import fit_samples

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "ml" / "configs"


def _sample(length: int = 16) -> FrameSample:
    rng = np.random.default_rng(11)
    return FrameSample(
        features=rng.random((length, 25), dtype=np.float32),
        root=np.zeros(length, dtype=np.int64),
        quality=np.zeros(length, dtype=np.int64),
        nochord=np.zeros(length, dtype=np.float32),
        boundary=np.zeros(length, dtype=np.float32),
        change=np.zeros(length, dtype=np.float32),
        stable=np.concatenate([
            np.zeros(1, dtype=np.float32),
            np.ones(length - 1, dtype=np.float32),
        ]),
        track_id="guitarset-01_example",
        capture_type="audio_mono-mic",
    )


def _loss_config(**training_overrides) -> dict:
    training = {
        "boundaryLossWeight": 2.0,
        "boundaryPosWeight": 8.0,
        "noChordLossWeight": 1.0,
        **training_overrides,
    }
    return {"training": training}


class TemporalTargetTests(unittest.TestCase):
    def test_hard_change_and_stable_targets_are_aligned(self):
        features = FeatureFrames(
            times=np.asarray([0.25, 0.75, 1.25, 1.75]),
            chroma=np.zeros((4, 12)),
            bass_chroma=np.zeros((4, 12)),
            energy=np.ones(4),
            sample_rate=22050,
            hop_seconds=0.5,
        )
        track = Track(
            track_id="x",
            artist="a",
            title="x",
            duration=2.0,
            source="synthetic",
            audio_availability="annotations",
            chords=[ChordRegion(0, 1, "C:maj"), ChordRegion(1, 2, "G:maj")],
        )
        targets = build_frame_targets(track, features, 0.2)
        self.assertEqual(targets["change"].tolist(), [0.0, 0.0, 1.0, 0.0])
        self.assertEqual(targets["stable"].tolist(), [0.0, 1.0, 0.0, 1.0])

    def test_collate_backfills_transition_targets_for_v1_samples(self):
        sample = _sample()
        sample.change = None
        sample.stable = None
        sample.root[8:] = 7
        batch = collate([sample])
        self.assertEqual(float(batch["change"][0, 8]), 1.0)
        self.assertEqual(float(batch["stable"][0, 7]), 1.0)
        self.assertEqual(float(batch["stable"][0, 8]), 0.0)


class TemporalLossTests(unittest.TestCase):
    def test_v1_config_retains_original_total(self):
        model = TemporalBaseline(ModelConfig(channels=8, dilations=(1,), dropout=0.0))
        batch = collate([_sample()])
        outputs = model(batch["features"])
        weights = {"root": torch.ones(12), "quality": torch.ones(3)}
        total, parts = combined_loss(outputs, batch, weights, _loss_config())
        expected = parts["root"] + parts["quality"] + parts["nochord"] + 2.0 * parts["boundary"]
        self.assertAlmostEqual(float(total.detach()), expected, places=5)
        self.assertGreaterEqual(parts["durationConsistency"], 0.0)

    def test_switch_probability_detects_distribution_changes(self):
        stable = {
            "root": torch.full((1, 3, 12), -8.0),
            "quality": torch.full((1, 3, 3), -8.0),
            "nochord": torch.full((1, 3), -8.0),
            "boundary": torch.zeros(1, 3),
        }
        stable["root"][:, :, 0] = 8.0
        stable["quality"][:, :, 0] = 8.0
        switched = {key: value.clone() for key, value in stable.items()}
        switched["root"][:, 1:, 0] = -8.0
        switched["root"][:, 1:, 7] = 8.0
        stable_switch = predicted_switch_probability(stable)
        changed_switch = predicted_switch_probability(switched)
        self.assertLess(float(stable_switch[0, 0]), 0.01)
        self.assertGreater(float(changed_switch[0, 0]), 0.99)

    def test_duration_objective_penalizes_flicker_inside_stable_chord(self):
        batch = collate([_sample(length=4)])
        weights = {"root": torch.ones(12), "quality": torch.ones(3)}

        def outputs(roots):
            root_logits = torch.full((1, 4, 12), -5.0)
            for index, root in enumerate(roots):
                root_logits[0, index, root] = 5.0
            return {
                "root": root_logits,
                "quality": torch.tensor([[[5.0, -5.0, -5.0]]] * 4).transpose(0, 1),
                "nochord": torch.full((1, 4), -5.0),
                "boundary": torch.full((1, 4), -5.0),
            }

        config = _loss_config(durationConsistencyLossWeight=1.0)
        _, stable_parts = combined_loss(outputs([0, 0, 0, 0]), batch, weights, config)
        _, flicker_parts = combined_loss(outputs([0, 7, 0, 7]), batch, weights, config)
        self.assertLess(stable_parts["durationConsistency"], flicker_parts["durationConsistency"])

    def test_v2_objective_trains_and_exports_in_one_epoch(self):
        config = {
            "modelName": "temporal-harmony-v2-test",
            "seed": 9,
            "features": {"inputDim": 25, "pipelineVersion": "harmony-features-v1"},
            "labels": {"boundaryToleranceSeconds": 0.18},
            "model": {
                "channels": 8,
                "kernelSize": 3,
                "dilations": [1, 2],
                "dropout": 0.0,
            },
            "training": {
                "epochs": 1,
                "batchSize": 1,
                "learningRate": 0.001,
                "weightDecay": 0.0,
                "earlyStoppingPatience": 1,
                "gradClip": 5.0,
                "boundaryLossWeight": 2.0,
                "boundaryPosWeight": 8.0,
                "hardBoundaryLossWeight": 0.5,
                "hardBoundaryPosWeight": 12.0,
                "noChordLossWeight": 1.0,
                "durationConsistencyLossWeight": 0.2,
                "switchLossWeight": 0.35,
                "boundarySwitchAgreementLossWeight": 0.1,
            },
        }
        training = [_sample(20), _sample(18)]
        development = [_sample(16)]
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "v2.pt"
            summary = fit_samples(config, training, development, checkpoint, quiet=True)
            history = json.loads(checkpoint.with_suffix(".history.json").read_text())
            self.assertTrue(checkpoint.with_suffix(".onnx").exists())
        self.assertEqual(summary["epochsRun"], 1)
        self.assertIn("durationConsistency", history["history"][0]["dev"])
        self.assertIn("switch", history["history"][0]["dev"])


class AugmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_augmentation_manifest(
            CONFIG_ROOT / "temporal-harmony-v2-augmentations.json"
        )

    def test_recipe_selection_is_deterministic_and_clean_input_is_unchanged(self):
        sample = _sample()
        original = sample.features.copy()
        first = build_augmented_samples(sample, self.manifest)
        second = build_augmented_samples(sample, self.manifest)
        self.assertEqual([item.augmentation_id for item in first],
                         [item.augmentation_id for item in second])
        for left, right in zip(first, second):
            np.testing.assert_allclose(left.features, right.features)
        np.testing.assert_allclose(sample.features, original)

    def test_no_harmonic_negative_has_no_chord_targets(self):
        recipe = next(
            item for item in self.manifest["recipes"]
            if item["kind"] == "no-harmonic-negative"
        )
        negative = augment_sample(_sample(), recipe, self.manifest["seed"])
        self.assertTrue(np.all(negative.root == -1))
        self.assertTrue(np.all(negative.quality == -1))
        self.assertTrue(np.all(negative.nochord == 1.0))
        self.assertTrue(np.all(negative.boundary == 0.0))


class ProbabilitySmoothingTests(unittest.TestCase):
    def test_ema_smooths_distributions_without_argmax_labels(self):
        root = np.zeros((6, 12), dtype=np.float32)
        root[::2, 0] = 1.0
        root[1::2, 7] = 1.0
        quality = np.zeros((6, 3), dtype=np.float32)
        quality[:, 0] = 1.0
        no_chord = np.zeros(6, dtype=np.float32)
        boundary = np.zeros(6, dtype=np.float32)
        smoothed = smooth_learned_probabilities(
            root,
            quality,
            no_chord,
            boundary,
            {"method": "ema", "alpha": 0.5, "boundaryAlpha": 0.5},
        )
        self.assertLess(np.abs(np.diff(smoothed[0], axis=0)).sum(),
                        np.abs(np.diff(root, axis=0)).sum())
        np.testing.assert_allclose(smoothed[0].sum(axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(smoothed[1].sum(axis=1), 1.0, atol=1e-6)


class ProtocolTests(unittest.TestCase):
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

    def test_plan_is_deterministic_bounded_and_excludes_p00(self):
        first = build_ablation_plan(self.base, self.splits, self.ablations, self.augmentations)
        second = build_ablation_plan(self.base, self.splits, self.ablations, self.augmentations)
        self.assertEqual(first, second)
        self.assertFalse(first["p00UsedForSelection"])
        self.assertEqual(len(first["candidates"]), 6)
        self.assertEqual(first["candidateFoldRuns"], 30)
        self.assertNotIn("guitarset-p00", json.dumps(first["folds"]))

    def test_p00_in_fold_is_rejected(self):
        broken = copy.deepcopy(self.splits)
        broken["folds"][0]["trainPerformers"].append("guitarset-p00")
        with self.assertRaises(ValueError):
            validate_split_manifest(broken)

    def test_candidate_does_not_mutate_base_config(self):
        before = copy.deepcopy(self.base)
        candidate_config(self.base, self.ablations["candidates"][0])
        self.assertEqual(self.base, before)

    def test_completed_run_resume_requires_exact_identity(self):
        identity = {
            "foldId": "lopo-p01",
            "candidateId": "v1-objective",
            "trainingConfigChecksum": "abc",
            "seed": 20260726,
            "epochsOverride": None,
        }
        with tempfile.TemporaryDirectory() as temporary:
            result_path = Path(temporary) / "run-result.json"
            expected_result = {"foldId": "lopo-p01", "candidateId": "v1-objective"}
            _write_json_atomic(result_path, {
                "schemaVersion": 1,
                "status": "completed",
                "runIdentity": identity,
                "result": expected_result,
            })
            self.assertEqual(_load_completed_run(result_path, identity), expected_result)
            changed = {**identity, "trainingConfigChecksum": "different"}
            with self.assertRaises(ValueError):
                _load_completed_run(result_path, changed)

    def test_fragmentation_gate_cannot_be_bypassed_by_accuracy(self):
        baseline = {
            capture: {
                "rootAccuracy": 0.5,
                "detailedAccuracy": 0.4,
                "fragmentationRate": 0.6,
                "regionsPerMinute": 20.0,
                "meanAbsoluteBoundaryErrorMs": 1000.0,
            }
            for capture in ("audio_mono-mic", "audio_mono-pickup_mix")
        }
        unstable = {
            capture: {
                "rootAccuracy": 0.7,
                "detailedAccuracy": 0.6,
                "fragmentationRate": 0.61,
                "regionsPerMinute": 21.0,
                "meanAbsoluteBoundaryErrorMs": 900.0,
            }
            for capture in ("audio_mono-mic", "audio_mono-pickup_mix")
        }
        result = evaluate_selection_gates(baseline, unstable, self.base["selectionGates"])
        self.assertFalse(result["passed"])
        self.assertTrue(any(
            check["gate"] == "fragmentationRelativeReduction" and not check["passed"]
            for check in result["checks"]
        ))

    def test_frozen_v1_baseline_reconciles(self):
        result = verify_baseline(require_local_artifacts=False)
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["selectionEligibility"]["p00Eligible"])

    def test_v2_receptive_field_is_longer_than_v1(self):
        v2 = ModelConfig.from_dict(self.base)
        v1 = candidate_config(self.base, self.ablations["candidates"][0])
        self.assertGreater(v2.receptive_field_frames(),
                           ModelConfig.from_dict(v1).receptive_field_frames())


if __name__ == "__main__":
    unittest.main()
