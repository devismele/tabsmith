"""Heavy ML tests (require the training extra: torch/onnx/onnxruntime).

Run: .\.venv\Scripts\python.exe -m unittest discover -s ml/tests_training -t .

Kept OUT of the normal test run so Tabsmith's suite works without PyTorch.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ml.evaluation.adapters import predict_hybrid, predict_ml
from ml.evaluation.metrics import evaluate_regions
from ml.models.temporal_baseline import ModelConfig, TemporalBaseline, export_onnx
from ml.preprocessing.features import FeatureFrames, extract_features
from ml.preprocessing.synth_generator import generate_varied_track
from ml.schema import ChordRegion, Track
from ml.training.checkpoint import load_checkpoint, save_checkpoint, build_metadata
from ml.training.dataset import FrameSample, build_frame_targets, collate, compute_class_weights
from ml.training.losses import combined_loss
from ml.training.train import run_training


def tiny_config() -> dict:
    return {
        "modelName": "temporal-baseline-test", "seed": 7,
        "data": {"trainingTracks": 4, "developmentTracks": 2, "syntheticTestTracks": 2,
                 "durationSeconds": 6.0, "tempoRange": [110, 120]},
        "features": {"inputDim": 25, "pipelineVersion": "numpy-chroma-v1"},
        "labels": {"roots": 12, "qualities": ["maj", "min", "7"], "boundaryToleranceSeconds": 0.12},
        "model": {"channels": 16, "kernelSize": 3, "dilations": [1, 2], "dropout": 0.0},
        "training": {"epochs": 3, "batchSize": 2, "learningRate": 0.005, "weightDecay": 0.0001,
                     "earlyStoppingPatience": 5, "gradClip": 5.0, "boundaryLossWeight": 2.0,
                     "boundaryPosWeight": 8.0, "noChordLossWeight": 1.0},
    }


def sample_batch(n_frames=(20, 14)):
    samples = []
    for t in n_frames:
        samples.append(FrameSample(
            features=np.random.default_rng(t).standard_normal((t, 25)).astype(np.float32),
            root=np.random.default_rng(t).integers(0, 12, t),
            quality=np.random.default_rng(t).integers(0, 3, t),
            nochord=np.zeros(t, dtype=np.float32),
            boundary=(np.arange(t) % 5 == 0).astype(np.float32),
        ))
    return samples


class DatasetTests(unittest.TestCase):
    def test_batching_and_padding(self):
        batch = collate(sample_batch((20, 14)))
        self.assertEqual(batch["features"].shape, (2, 20, 25))
        self.assertEqual(batch["root"].shape, (2, 20))

    def test_frame_mask_handling(self):
        batch = collate(sample_batch((20, 14)))
        self.assertTrue(torch.all(batch["pad_mask"][1, 14:] == 0))
        self.assertTrue(torch.all(batch["pad_mask"][1, :14] == 1))

    def test_root_and_quality_target_encoding(self):
        ff = FeatureFrames(times=np.array([0.5, 1.5]), chroma=np.zeros((2, 12)),
                           bass_chroma=np.zeros((2, 12)), energy=np.ones(2), sample_rate=22050, hop_seconds=0.09)
        track = Track(track_id="x", artist="a", title="t", duration=2.0, source="synthetic",
                      audio_availability="annotations",
                      chords=[ChordRegion(0, 1, "A:maj"), ChordRegion(1, 2, "E:min")])
        targets = build_frame_targets(track, ff, 0.12)
        self.assertEqual(targets["root"][0], 9)      # A
        self.assertEqual(targets["quality"][0], 0)   # maj
        self.assertEqual(targets["root"][1], 4)      # E
        self.assertEqual(targets["quality"][1], 1)   # min

    def test_boundary_target_generation(self):
        ff = FeatureFrames(times=np.array([0.9, 0.98, 1.02, 1.1]), chroma=np.zeros((4, 12)),
                           bass_chroma=np.zeros((4, 12)), energy=np.ones(4), sample_rate=22050, hop_seconds=0.02)
        track = Track(track_id="x", artist="a", title="t", duration=2.0, source="synthetic",
                      audio_availability="annotations",
                      chords=[ChordRegion(0, 1, "A:maj"), ChordRegion(1, 2, "E:min")])
        targets = build_frame_targets(track, ff, 0.12)
        self.assertGreater(targets["boundary"][1], targets["boundary"][0])  # closer to the 1.0s boundary
        self.assertGreater(targets["boundary"][2], targets["boundary"][3])


class LossTests(unittest.TestCase):
    def _model_and_batch(self):
        model = TemporalBaseline(ModelConfig(channels=16, dilations=(1, 2), dropout=0.0))
        batch = collate(sample_batch((20, 14)))
        return model, batch

    def test_combined_loss_is_finite_scalar(self):
        model, batch = self._model_and_batch()
        weights = {"root": torch.ones(12), "quality": torch.ones(3)}
        total, parts = combined_loss(model(batch["features"]), batch, weights, tiny_config())
        self.assertEqual(total.dim(), 0)
        self.assertTrue(torch.isfinite(total))
        self.assertIn("boundary", parts)

    def test_boundary_pos_weight_increases_boundary_loss(self):
        model, batch = self._model_and_batch()
        weights = {"root": torch.ones(12), "quality": torch.ones(3)}
        outputs = model(batch["features"])
        low = tiny_config(); low["training"]["boundaryPosWeight"] = 1.0
        high = tiny_config(); high["training"]["boundaryPosWeight"] = 20.0
        _, low_parts = combined_loss(outputs, batch, weights, low)
        _, high_parts = combined_loss(outputs, batch, weights, high)
        self.assertGreater(high_parts["boundary"], low_parts["boundary"])

    def test_class_weights_shape(self):
        weights = compute_class_weights(sample_batch((20, 14)))
        self.assertEqual(weights["root"].shape[0], 12)
        self.assertEqual(weights["quality"].shape[0], 3)


class TrainingTests(unittest.TestCase):
    def test_deterministic_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = run_training(tiny_config(), Path(tmp) / "a.pt", quiet=True)
            b = run_training(tiny_config(), Path(tmp) / "b.pt", quiet=True)
        self.assertEqual(a["checksum"], b["checksum"])

    def test_checkpoint_save_and_reload_matches(self):
        model = TemporalBaseline(ModelConfig(channels=16, dilations=(1, 2), dropout=0.0))
        model.eval()
        x = torch.zeros(1, 12, 25)
        before = model(x)
        with tempfile.TemporaryDirectory() as tmp:
            meta = build_metadata({"features": {"inputDim": 25}, "model": {"channels": 16, "dilations": [1, 2]}},
                                  "hash", 1, model.parameter_count(), model)
            path = save_checkpoint(Path(tmp) / "m.pt",
                                   model, {"features": {"inputDim": 25}, "model": {"channels": 16, "dilations": [1, 2]}}, meta)
            reloaded, _ = load_checkpoint(path)
        after = reloaded(x)
        self.assertTrue(torch.allclose(before["root"], after["root"], atol=1e-6))


class InferenceTests(unittest.TestCase):
    def _trained(self, tmp):
        run_training(tiny_config(), Path(tmp) / "m.pt", quiet=True)
        return load_checkpoint(Path(tmp) / "m.pt")

    def test_prediction_output_shapes(self):
        model = TemporalBaseline(ModelConfig(channels=16, dilations=(1, 2)))
        out = model(torch.zeros(2, 30, 25))
        self.assertEqual(out["root"].shape, (2, 30, 12))
        self.assertEqual(out["quality"].shape, (2, 30, 3))
        self.assertEqual(out["nochord"].shape, (2, 30))
        self.assertEqual(out["boundary"].shape, (2, 30))

    def test_onnx_export_and_agreement(self):
        import onnxruntime as ort
        model = TemporalBaseline(ModelConfig(channels=16, dilations=(1, 2), dropout=0.0))
        model.eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "m.onnx")
            export_onnx(model, path, input_dim=25)
            self.assertTrue(Path(path).exists())
            x = np.random.default_rng(0).standard_normal((1, 18, 25)).astype(np.float32)
            with torch.no_grad():
                torch_out = model(torch.from_numpy(x))
            session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            onnx_out = session.run(None, {"features": x})
            self.assertLess(float(np.max(np.abs(torch_out["root"].numpy() - onnx_out[0]))), 1e-4)

    def test_malformed_feature_shape_rejected(self):
        model = TemporalBaseline(ModelConfig(channels=16, dilations=(1, 2), input_dim=25))
        with self.assertRaises((RuntimeError, ValueError)):
            model(torch.zeros(1, 10, 9))  # wrong feature dim

    def test_adapters_produce_scorable_regions(self):
        _track, audio, sr = generate_varied_track(0, seed=3000, duration=6.0, tempo_range=(110, 120))
        features = extract_features(audio, sr)
        with tempfile.TemporaryDirectory() as tmp:
            model, _ = self._trained(tmp)
        for predict in (predict_ml, predict_hybrid):
            regions = predict(model, features)
            self.assertTrue(regions)
            result = evaluate_regions([ChordRegion(0, 6, "A:maj")], regions)
            self.assertIn("rootAccuracy", result)


if __name__ == "__main__":
    unittest.main()
