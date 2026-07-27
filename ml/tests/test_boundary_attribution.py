r"""Attribution tests: boundary vs decoder vs chord-posterior effects.

The study's 2x2 design (control/candidate x full-v2 decode/segmental-full) is
what lets a change be attributed to a channel. These tests pin the decomposition
arithmetic, the sign convention, and the key structural fact the attribution
rests on: the full-v2 decode never reads the boundary head, so a model effect
visible there is chord-posterior movement.

Run: .\.venv\Scripts\python.exe -m unittest discover -s ml/tests -t .
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass, field

import numpy as np

from ml.evaluation.boundary.attribution import (
    EXISTING,
    SEGMENTAL,
    boundary_channel_summary,
    chord_channel_summary,
    decompose,
    interpret,
)

CAPTURES = ("audio_mono-mic", "audio_mono-pickup_mix")


def _metrics(root=0.75, detailed=0.70, frag=0.65, rpm=21.0, mae=450.0, flicker=4):
    return {"rootAccuracy": root, "detailedAccuracy": detailed,
            "fragmentationRate": frag, "regionsPerMinute": rpm,
            "meanAbsoluteBoundaryErrorMs": mae, "flickerCount": flicker}


def _boundary_quality(precision=0.79, recall=0.84, f1=0.815, ece=0.037,
                      agreement=0.78, threshold=0.52):
    return {"precision": precision, "recall": recall, "f1": f1,
            "expectedCalibrationError": ece,
            "stateChangeBoundaryAgreement": agreement,
            "meanOperatingThreshold": threshold}


def _decoders(existing_metrics, segmental_metrics, *, boundary=None):
    boundary = boundary or _boundary_quality()
    return {
        EXISTING: {"byCapture": {c: dict(existing_metrics) for c in CAPTURES},
                   "boundaryQuality": {c: dict(boundary) for c in CAPTURES}},
        SEGMENTAL: {"byCapture": {c: dict(segmental_metrics) for c in CAPTURES},
                    "boundaryQuality": {c: dict(boundary) for c in CAPTURES}},
    }


@dataclass
class _Entry:
    performance_id: str
    capture: str
    times: np.ndarray
    root: np.ndarray
    quality: np.ndarray
    nochord: np.ndarray
    reference: list = field(default_factory=list)


def _entry(performance_id, capture, *, root_bias=0.0, frames=40, seed=0):
    rng = np.random.default_rng(seed)
    root = rng.dirichlet(np.ones(12), size=frames)
    if root_bias:
        root = root + root_bias
        root = root / root.sum(axis=1, keepdims=True)
    quality = rng.dirichlet(np.ones(3), size=frames)
    nochord = rng.random(frames) * 0.1
    return _Entry(performance_id, capture, np.arange(frames) * 0.25, root, quality, nochord)


class SignConventionTests(unittest.TestCase):
    def test_lower_is_better_metrics_invert(self):
        control = _decoders(_metrics(frag=0.67), _metrics(frag=0.65))
        candidate = _decoders(_metrics(frag=0.67), _metrics(frag=0.60))
        out = decompose(control, candidate)["audio_mono-mic"]["fragmentationRate"]
        # Candidate is 0.05 lower under segmental -> positive (better).
        self.assertAlmostEqual(out["modelEffectUnderSegmental"], 0.05, places=4)

    def test_higher_is_better_metrics_keep_sign(self):
        control = _decoders(_metrics(root=0.75), _metrics(root=0.75))
        candidate = _decoders(_metrics(root=0.75), _metrics(root=0.78))
        out = decompose(control, candidate)["audio_mono-mic"]["rootAccuracy"]
        self.assertAlmostEqual(out["modelEffectUnderSegmental"], 0.03, places=4)


class DecompositionTests(unittest.TestCase):
    def test_decoder_effect_uses_the_control_only(self):
        control = _decoders(_metrics(frag=0.68), _metrics(frag=0.64))
        candidate = _decoders(_metrics(frag=0.60), _metrics(frag=0.50))
        out = decompose(control, candidate)["audio_mono-mic"]["fragmentationRate"]
        self.assertAlmostEqual(out["decoderEffect"], 0.04, places=4)

    def test_pure_chord_posterior_change_shows_in_both_decoders(self):
        """A model change that helps equally under both decoders is chord-side."""
        control = _decoders(_metrics(frag=0.68), _metrics(frag=0.64))
        candidate = _decoders(_metrics(frag=0.64), _metrics(frag=0.60))
        out = decompose(control, candidate)["audio_mono-mic"]["fragmentationRate"]
        self.assertAlmostEqual(out["chordPosteriorChannel"], 0.04, places=4)
        self.assertAlmostEqual(out["boundaryChannel"], 0.0, places=4)
        self.assertAlmostEqual(out["interaction"], 0.0, places=4)

    def test_pure_boundary_change_shows_only_under_segmental(self):
        """The full-v2 decode ignores the boundary head, so a boundary-only
        improvement must be invisible there and present under segmental."""
        control = _decoders(_metrics(frag=0.68), _metrics(frag=0.64))
        candidate = _decoders(_metrics(frag=0.68), _metrics(frag=0.60))
        out = decompose(control, candidate)["audio_mono-mic"]["fragmentationRate"]
        self.assertAlmostEqual(out["chordPosteriorChannel"], 0.0, places=4)
        self.assertAlmostEqual(out["boundaryChannel"], 0.04, places=4)

    def test_channels_sum_to_the_segmental_model_effect(self):
        control = _decoders(_metrics(frag=0.68), _metrics(frag=0.64))
        candidate = _decoders(_metrics(frag=0.65), _metrics(frag=0.58))
        out = decompose(control, candidate)["audio_mono-mic"]["fragmentationRate"]
        self.assertAlmostEqual(
            out["chordPosteriorChannel"] + out["boundaryChannel"],
            out["modelEffectUnderSegmental"], places=6)

    def test_both_captures_are_reported(self):
        control = _decoders(_metrics(), _metrics())
        candidate = _decoders(_metrics(), _metrics())
        out = decompose(control, candidate)
        self.assertEqual(sorted(out), sorted(CAPTURES))


class BoundaryChannelTests(unittest.TestCase):
    def test_boundary_deltas_are_reported(self):
        control = _decoders(_metrics(), _metrics(), boundary=_boundary_quality(f1=0.80, ece=0.04))
        candidate = _decoders(_metrics(), _metrics(), boundary=_boundary_quality(f1=0.86, ece=0.01))
        out = boundary_channel_summary(control, candidate)["audio_mono-mic"]
        self.assertAlmostEqual(out["f1"]["delta"], 0.06, places=4)
        self.assertAlmostEqual(out["expectedCalibrationError"]["delta"], -0.03, places=6)

    def test_unchanged_head_reports_zero_delta(self):
        same = _boundary_quality()
        control = _decoders(_metrics(), _metrics(), boundary=same)
        candidate = _decoders(_metrics(), _metrics(), boundary=same)
        out = boundary_channel_summary(control, candidate)["audio_mono-mic"]
        self.assertEqual(out["f1"]["delta"], 0.0)
        self.assertEqual(out["stateChangeBoundaryAgreement"]["delta"], 0.0)


class ChordChannelTests(unittest.TestCase):
    def test_identical_posteriors_agree_completely(self):
        entries = [_entry("01_A", "audio_mono-mic", seed=1)]
        out = chord_channel_summary(entries, entries)
        self.assertEqual(out["comparedCaptures"], 1)
        self.assertAlmostEqual(out["top1AgreementWithControl"], 1.0, places=6)
        self.assertAlmostEqual(out["marginDelta"], 0.0, places=6)

    def test_shifted_posteriors_reduce_agreement(self):
        control = [_entry("01_A", "audio_mono-mic", seed=1)]
        candidate = [_entry("01_A", "audio_mono-mic", seed=2)]
        out = chord_channel_summary(control, candidate)
        self.assertLess(out["top1AgreementWithControl"], 1.0)

    def test_unmatched_captures_are_skipped(self):
        control = [_entry("01_A", "audio_mono-mic", seed=1)]
        candidate = [_entry("99_Z", "audio_mono-mic", seed=1)]
        self.assertEqual(chord_channel_summary(control, candidate)["comparedCaptures"], 0)

    def test_empty_input_is_safe(self):
        out = chord_channel_summary([], [])
        self.assertEqual(out["comparedCaptures"], 0)
        self.assertEqual(out["top1AgreementWithControl"], 0.0)


class InterpretationTests(unittest.TestCase):
    def _interpret(self, control, candidate, chord=None):
        return interpret(
            decompose(control, candidate),
            boundary_channel_summary(control, candidate),
            chord or {"top1AgreementWithControl": 0.99})

    def test_names_chord_posterior_when_it_dominates(self):
        control = _decoders(_metrics(frag=0.68), _metrics(frag=0.64))
        candidate = _decoders(_metrics(frag=0.63), _metrics(frag=0.59))
        out = self._interpret(control, candidate)["audio_mono-mic"]
        self.assertEqual(out["dominantChannel"], "chord posterior")

    def test_names_boundary_channel_when_it_dominates(self):
        control = _decoders(_metrics(frag=0.68), _metrics(frag=0.64))
        candidate = _decoders(_metrics(frag=0.68), _metrics(frag=0.58))
        out = self._interpret(control, candidate)["audio_mono-mic"]
        self.assertEqual(out["dominantChannel"], "boundary channel")

    def test_reports_no_material_movement_when_nothing_changes(self):
        control = _decoders(_metrics(frag=0.68), _metrics(frag=0.64))
        candidate = _decoders(_metrics(frag=0.68), _metrics(frag=0.6405))
        out = self._interpret(control, candidate)["audio_mono-mic"]
        self.assertEqual(out["dominantChannel"], "neither (no material movement)")


if __name__ == "__main__":
    unittest.main()
